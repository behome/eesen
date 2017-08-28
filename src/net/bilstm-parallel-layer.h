// net/bilstm-parallel-layer.h

// Copyright 2015  Yajie Miao
//           2017  Jayadev Billa

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef EESEN_BILSTM_PARALLEL_LAYER_H_
#define EESEN_BILSTM_PARALLEL_LAYER_H_

#include "net/layer.h"
#include "net/trainable-layer.h"
#include "net/bilstm-layer.h"
#include "net/utils-functions.h"
#include "gpucompute/cuda-math.h"

namespace eesen {

class BiLstmParallel : public BiLstm {
public:
    BiLstmParallel(int32 input_dim, int32 output_dim) : BiLstm(input_dim, output_dim)
    { }
    ~BiLstmParallel()
    { }

    Layer* Copy() const { return new BiLstmParallel(*this); }
    LayerType GetType() const { return l_BiLstm_Parallel; }
    LayerType GetTypeNonParal() const { return l_BiLstm; }

    void SetSeqLengths(std::vector<int> &sequence_lengths) {
        sequence_lengths_ = sequence_lengths;
    }

    void InitializeForwardMask(int32 T, int32 S) {

      if (!in_train) return;

      // create the mask on the CPU and copy it over (GPU version was significantly slower by factor of 4-5)
      forward_drop_mask_cpu_.Resize(T*S, 2 * cell_dim_, kUndefined);
      forward_drop_mask_.Resize(T*S, 2 * cell_dim_, kUndefined);

      if (forward_step_dropout)
        forward_drop_mask_cpu_.SetRandUniform();
      else if (forward_sequence_dropout)
        forward_drop_mask_cpu_.SetRandUniformCol();

      forward_drop_mask_cpu_.Add(-forward_dropout);
      forward_drop_mask_cpu_.ApplyHeaviside();
      forward_drop_mask_cpu_.Scale(1.0/(1.0-forward_dropout)); // scale mask
      forward_drop_mask_.CopyFromMat(forward_drop_mask_cpu_);

    }

    void InitializeRecurrentMasks(int32 T, int32 S) {

      if (!in_train) return;

      // Set the row size based on whether sequence or step dropout selected
      // assumes that if sequence_dropout is true step is false.
      int mask_row_size = recurrent_sequence_dropout? S: (T+2)*S;

      if (recurrent_dropout != 0.0 && (rnndrop || no_mem_loss_dropout)) {
          recurrent_drop_mask_fw_.Resize(mask_row_size, cell_dim_, kUndefined);
          recurrent_drop_mask_bw_.Resize(mask_row_size, cell_dim_, kUndefined);
          recurrent_drop_mask_cpu_.Resize(mask_row_size, 2*cell_dim_, kUndefined);

        if (recurrent_sequence_dropout)
          recurrent_drop_mask_cpu_.SetRandUniformCol();
        else
          recurrent_drop_mask_cpu_.SetRandUniform();

        recurrent_drop_mask_cpu_.Add(-recurrent_dropout);
        recurrent_drop_mask_cpu_.ApplyHeaviside();
        // scale the masks so as to not need it during test.
        recurrent_drop_mask_cpu_.Scale(1.0 /(1.0-recurrent_dropout));
        //forward cells mask
        recurrent_drop_mask_fw_.CopyFromMat(recurrent_drop_mask_cpu_.ColRange(0, cell_dim_));
        //backward cells mask
        recurrent_drop_mask_bw_.CopyFromMat(recurrent_drop_mask_cpu_.ColRange(cell_dim_, cell_dim_));
      }

    }

   void PropagateFncVanillaFast(const CuMatrixBase<BaseFloat> &in, int32 T, int32 S) {

      cudaStream_t stream[2];
      for (int i=0; i< 2; i++) {
         cudaStreamCreate( &stream[i] ) ;
      }

      CuSubMatrix<BaseFloat> fYG(propagate_buf_fw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> fYI(propagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYF(propagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYO(propagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYC(propagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYH(propagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYM(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYGIFO(propagate_buf_fw_.ColRange(0, 4 * cell_dim_));

      CuSubMatrix<BaseFloat> bYG(propagate_buf_bw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> bYI(propagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYF(propagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYO(propagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYC(propagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYH(propagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYM(propagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYGIFO(propagate_buf_bw_.ColRange(0, 4 * cell_dim_));


      // no temporal recurrence involved in the inputs
      cublasSetKernelStream(stream[0]);
      fYGIFO.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, wei_gifo_x_fw_, kTrans, 0.0);
      cublasSetKernelStream(stream[1]);
      bYGIFO.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, wei_gifo_x_bw_, kTrans, 0.0);

      cublasSetKernelStream(stream[0]);
      fYGIFO.RowRange(1*S,T*S).AddVecToRows(1.0, bias_fw_);
      cublasSetKernelStream(stream[1]);
      bYGIFO.RowRange(1*S,T*S).AddVecToRows(1.0, bias_bw_);

      if (!in_train) KALDI_ERR << "Recurrent dropout attempted in test mode";

      for (int ft = 1, bt = T; ft <= T && bt >=1; ft++, bt--) {

        KALDI_ASSERT ((ft+bt) == T+1);

        // variables representing invidivual units/gates
        CuSubMatrix<BaseFloat> fy_all(propagate_buf_fw_.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_g(fYG.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_i(fYI.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_f(fYF.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_o(fYO.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_c(fYC.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_h(fYH.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_m(fYM.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_GIFO(fYGIFO.RowRange(ft*S,S));

        CuSubMatrix<BaseFloat> by_all(propagate_buf_bw_.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_g(bYG.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_i(bYI.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_f(bYF.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_o(bYO.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_c(bYC.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_h(bYH.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_m(bYM.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_GIFO(bYGIFO.RowRange(bt*S,S));

        // add the recurrence of the previous memory cell to various gates/units
        cublasSetKernelStream(stream[0]);
        fy_GIFO.AddMatMat(1.0, fYM.RowRange((ft-1)*S,S), kNoTrans, wei_gifo_m_fw_, kTrans,  1.0);

        cublasSetKernelStream(stream[1]);
        by_GIFO.AddMatMat(1.0, bYM.RowRange((bt+1)*S,S), kNoTrans, wei_gifo_m_bw_, kTrans,  1.0);

        PropagatePointwiseOpsLSTM_nodrop(fy_i, fy_f, fy_g, fy_o, fy_c, fy_h, fy_m, fYC.RowRange((ft-1)*S,S),
                              phole_i_c_fw_, phole_f_c_fw_, phole_o_c_fw_, stream[0]);

        PropagatePointwiseOpsLSTM_nodrop(by_i, by_f, by_g, by_o, by_c, by_h, by_m, bYC.RowRange((bt+1)*S,S),
                      phole_i_c_bw_, phole_f_c_bw_, phole_o_c_bw_, stream[1]);

        for (int s = 0; s < S; s++) {
          if (bt > sequence_lengths_[s])
            by_all.Row(s).SetZero();
        }
      } // end of t

      for(int i =0; i< 2;i++){
       //destroy streams
        cudaStreamDestroy( stream[i] ) ;
      }
      cublasSetKernelStream(NULL);
    }

   void PropagateFncRecurrentDropoutFast(const CuMatrixBase<BaseFloat> &in, int32 T, int32 S) {

      cudaStream_t stream[2];
      for (int i=0; i< 2; i++) {
         cudaStreamCreate( &stream[i] ) ;
      }

      CuSubMatrix<BaseFloat> fYG(propagate_buf_fw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> fYI(propagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYF(propagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYO(propagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYC(propagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYH(propagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYM(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYGIFO(propagate_buf_fw_.ColRange(0, 4 * cell_dim_));

      CuSubMatrix<BaseFloat> bYG(propagate_buf_bw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> bYI(propagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYF(propagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYO(propagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYC(propagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYH(propagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYM(propagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYGIFO(propagate_buf_bw_.ColRange(0, 4 * cell_dim_));


      // no temporal recurrence involved in the inputs
      cublasSetKernelStream(stream[0]);
      fYGIFO.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, wei_gifo_x_fw_, kTrans, 0.0);
      cublasSetKernelStream(stream[1]);
      bYGIFO.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, wei_gifo_x_bw_, kTrans, 0.0);

      cublasSetKernelStream(stream[0]);
      fYGIFO.RowRange(1*S,T*S).AddVecToRows(1.0, bias_fw_);
      cublasSetKernelStream(stream[1]);
      bYGIFO.RowRange(1*S,T*S).AddVecToRows(1.0, bias_bw_);

      if (!in_train) KALDI_ERR << "Recurrent dropout attempted in test mode";

      CuSubMatrix<BaseFloat> fr_mask;
      CuSubMatrix<BaseFloat> br_mask;

      // point the mask to the correct position
      if (recurrent_sequence_dropout) {
          fr_mask =  recurrent_drop_mask_fw_;
          br_mask =  recurrent_drop_mask_bw_;
      }

      for (int ft = 1, bt = T; ft <= T && bt >=1; ft++, bt--) {

        KALDI_ASSERT ((ft+bt) == T+1);

        // variables representing invidivual units/gates
        CuSubMatrix<BaseFloat> fy_all(propagate_buf_fw_.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_g(fYG.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_i(fYI.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_f(fYF.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_o(fYO.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_c(fYC.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_h(fYH.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_m(fYM.RowRange(ft*S,S));
        CuSubMatrix<BaseFloat> fy_GIFO(fYGIFO.RowRange(ft*S,S));

        CuSubMatrix<BaseFloat> by_all(propagate_buf_bw_.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_g(bYG.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_i(bYI.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_f(bYF.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_o(bYO.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_c(bYC.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_h(bYH.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_m(bYM.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_GIFO(bYGIFO.RowRange(bt*S,S));

        // point the mask to the correct position
        if (recurrent_step_dropout) {
          fr_mask =  recurrent_drop_mask_fw_.RowRange(ft*S,S);
          br_mask =  recurrent_drop_mask_bw_.RowRange(bt*S,S);
        }

        // add the recurrence of the previous memory cell to various gates/units
        cublasSetKernelStream(stream[0]);
        fy_GIFO.AddMatMat(1.0, fYM.RowRange((ft-1)*S,S), kNoTrans, wei_gifo_m_fw_, kTrans,  1.0);

        cublasSetKernelStream(stream[1]);
        by_GIFO.AddMatMat(1.0, bYM.RowRange((bt+1)*S,S), kNoTrans, wei_gifo_m_bw_, kTrans,  1.0);


        PropagatePointwiseOpsLSTM(fy_i, fy_f, fy_g, fy_o, fy_c, fy_h, fy_m, fYC.RowRange((ft-1)*S,S),
                              phole_i_c_fw_, phole_f_c_fw_, phole_o_c_fw_, fr_mask, no_mem_loss_dropout, stream[0]);

        PropagatePointwiseOpsLSTM(by_i, by_f, by_g, by_o, by_c, by_h, by_m, bYC.RowRange((bt+1)*S,S),
                      phole_i_c_bw_, phole_f_c_bw_, phole_o_c_bw_, br_mask, no_mem_loss_dropout, stream[1]);

        for (int s = 0; s < S; s++) {
          if (bt > sequence_lengths_[s])
            by_all.Row(s).SetZero();
        }
      } // end of t

      for(int i =0; i< 2;i++){
       //destroy streams
        cudaStreamDestroy( stream[i] ) ;
      }
      cublasSetKernelStream(NULL);
    }

    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
      int32 nstream_ = sequence_lengths_.size();  // the number of sequences to be processed in parallel
      KALDI_ASSERT(in.NumRows() % nstream_ == 0);
      int32 T = in.NumRows() / nstream_;
      int32 S = nstream_;

      if (twiddle_forward) {
        twiddle_apply_forward = BernoulliDist(0.5);
      }

      bool apply_recurrent_dropout = in_train && (rnndrop || no_mem_loss_dropout) && (!twiddle_forward || (twiddle_forward && !twiddle_apply_forward));
      bool apply_forward_dropout   = in_train && forward_dropout > 0.0            && (!twiddle_forward || (twiddle_forward &&  twiddle_apply_forward));

      // initialize the propagation buffers
      propagate_buf_fw_.Resize((T+2)*S, 7 * cell_dim_, kSetZero);
      propagate_buf_bw_.Resize((T+2)*S, 7 * cell_dim_, kSetZero);

      // initialize recurrent masks as needed
      InitializeRecurrentMasks(T,S);

      // propagate forward and then backward cells (same procedures, but iterates from t=T to t=1)
      if (apply_recurrent_dropout) {
        PropagateFncRecurrentDropoutFast(in, T, S);
      } else {
        PropagateFncVanillaFast(in, T, S);
      }


      // final outputs now become the concatenation of the forward and backward activations
      CuMatrix<BaseFloat> YR_RB;
      YR_RB.Resize((T+2)*S, 2 * cell_dim_, kSetZero);
      YR_RB.ColRange(0, cell_dim_).CopyFromMat(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));
      YR_RB.ColRange(cell_dim_, cell_dim_).CopyFromMat(propagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));

      if (apply_forward_dropout) {
        InitializeForwardMask(T,S);
        YR_RB.RowRange(S,T*S).MulElements(forward_drop_mask_);
      }

      out->CopyFromMat(YR_RB.RowRange(S,T*S));
    }

    void BackpropagateFncVanillaFast(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out_diff_drop,
                                              CuMatrixBase<BaseFloat> *in_diff, int32 T, int32 S) {

      if (!in_train) KALDI_ERR << "Can't backpropagate in test mode";


      cudaStream_t stream[12];
      for (int i=0; i< 12; i++) {
         cudaStreamCreate( &stream[i] ) ;
      }

      // get the activations of the gates/units from the feedforward buffer; these variabiles will be used
      // in gradients computation
      CuSubMatrix<BaseFloat> fYG(propagate_buf_fw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> fYI(propagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYF(propagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYO(propagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYC(propagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYH(propagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYM(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));

      CuSubMatrix<BaseFloat> bYG(propagate_buf_bw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> bYI(propagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYF(propagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYO(propagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYC(propagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYH(propagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYM(propagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));


      // errors back-propagated to individual gates/units
      CuSubMatrix<BaseFloat> fDG(backpropagate_buf_fw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> fDI(backpropagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fDF(backpropagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fDO(backpropagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fDC(backpropagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fDH(backpropagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fDM(backpropagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fDGIFO(backpropagate_buf_fw_.ColRange(0, 4 * cell_dim_));
      CuSubMatrix<BaseFloat> fDCM(d_c_mask_fw.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> fDHM(d_h_mask_fw.ColRange(0, cell_dim_));

      CuSubMatrix<BaseFloat> bDG(backpropagate_buf_bw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> bDI(backpropagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bDF(backpropagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bDO(backpropagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bDC(backpropagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bDH(backpropagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bDM(backpropagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bDGIFO(backpropagate_buf_bw_.ColRange(0, 4 * cell_dim_));
      CuSubMatrix<BaseFloat> bDCM(d_c_mask_bw.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> bDHM(d_h_mask_bw.ColRange(0, cell_dim_));

      //  assume that the first half of out_diff is about the forward layer
      fDM.RowRange(1*S,T*S).CopyFromMat(out_diff_drop.ColRange(0, cell_dim_));
      // the second half of the error vector corresponds to the backward layer
      bDM.RowRange(1*S, T*S).CopyFromMat(out_diff_drop.ColRange(cell_dim_, cell_dim_));


      for (int bt = 1, ft = T; bt <= T && ft >=1; bt++, ft--) {
        // variables representing activations of invidivual units/gates

        KALDI_ASSERT ((ft+bt) == T+1);

        CuSubMatrix<BaseFloat> fy_g(fYG.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fy_i(fYI.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fy_f(fYF.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fy_o(fYO.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fy_c(fYC.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fy_h(fYH.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fy_m(fYM.RowRange(ft*S, S));

        CuSubMatrix<BaseFloat> by_g(bYG.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_i(bYI.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_f(bYF.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_o(bYO.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_c(bYC.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_h(bYH.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_m(bYM.RowRange(bt*S, S));

        // variables representing errors of invidivual units/gates
        CuSubMatrix<BaseFloat> fd_g(fDG.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_i(fDI.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_f(fDF.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_o(fDO.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_c(fDC.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_h(fDH.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_m(fDM.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_all(backpropagate_buf_fw_.RowRange(ft*S, S));

        CuSubMatrix<BaseFloat> fd_c_m(fDCM.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_h_m(fDHM.RowRange(ft*S, S));

        CuSubMatrix<BaseFloat> bd_g(bDG.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_i(bDI.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_f(bDF.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_o(bDO.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_c(bDC.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_h(bDH.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_m(bDM.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_all(backpropagate_buf_fw_.RowRange(bt*S, S));

        CuSubMatrix<BaseFloat> bd_c_m(bDCM.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_h_m(bDHM.RowRange(bt*S, S));

        // d_m comes from two parts: errors from the upper layer and errors from the following frame (t+1)
        cublasSetKernelStream(stream[0]);
        fd_m.AddMatMat(1.0, fDGIFO.RowRange((ft+1)*S,S), kNoTrans, wei_gifo_m_fw_, kNoTrans, 1.0);
        cublasSetKernelStream(stream[1]);
        bd_m.AddMatMat(1.0, bDGIFO.RowRange((bt-1)*S,S), kNoTrans, wei_gifo_m_bw_, kNoTrans, 1.0);

        BackpropagatePointwiseOpsLSTM_nodrop(fy_i, fy_f, fy_g, fy_o, fy_c, fy_h, fy_m,
                                      fd_i, fd_f, fd_g, fd_o, fd_c, fd_h, fd_m, fd_c_m,
                                      fDI.RowRange((ft+1)*S,S), fDF.RowRange((ft+1)*S,S),
                                      fDC.RowRange((ft+1)*S,S), fDCM.RowRange((ft+1)*S,S),
                                      fYF.RowRange((ft+1)*S,S), fYC.RowRange((ft-1)*S,S),
                                      phole_i_c_fw_, phole_f_c_fw_, phole_o_c_fw_,
                                      stream[0]);

        BackpropagatePointwiseOpsLSTM_nodrop(by_i, by_f, by_g, by_o, by_c, by_h, by_m,
                                      bd_i, bd_f, bd_g, bd_o, bd_c, bd_h, bd_m, bd_c_m,
                                      bDI.RowRange((bt-1)*S,S), bDF.RowRange((bt-1)*S,S),
                                      bDC.RowRange((bt-1)*S,S), bDCM.RowRange((bt-1)*S,S),
                                      bYF.RowRange((bt-1)*S,S), bYC.RowRange((bt+1)*S,S),
                                      phole_i_c_bw_, phole_f_c_bw_, phole_o_c_bw_,
                                      stream[1]);

      }  // end of t

      cublasSetKernelStream(NULL);
      //  updates to the model parameters
      const BaseFloat mmt = opts_.momentum;
      cublasSetKernelStream(stream[0]);
      wei_gifo_x_fw_corr_.AddMatMat(1.0, fDGIFO.RowRange(1*S, T*S), kTrans, in, kNoTrans, mmt);
      cublasSetKernelStream(stream[1]);
      wei_gifo_m_fw_corr_.AddMatMat(1.0, fDGIFO.RowRange(1*S, T*S), kTrans, fYM.RowRange(0*S,T*S), kNoTrans, mmt);
      cublasSetKernelStream(stream[2]);
      bias_fw_corr_.AddRowSumMat(1.0, fDGIFO.RowRange(1*S, T*S), mmt);
      cublasSetKernelStream(stream[3]);
      phole_i_c_fw_corr_.AddDiagMatMat(1.0, fDI.RowRange(1*S, T*S), kTrans, fYC.RowRange(0*S, T*S), kNoTrans, mmt);
      cublasSetKernelStream(stream[4]);
      phole_f_c_fw_corr_.AddDiagMatMat(1.0, fDF.RowRange(1*S, T*S), kTrans, fYC.RowRange(0*S, T*S), kNoTrans, mmt);
      cublasSetKernelStream(stream[5]);
      phole_o_c_fw_corr_.AddDiagMatMat(1.0, fDO.RowRange(1*S, T*S), kTrans, fYC.RowRange(1*S, T*S), kNoTrans, mmt);
      // updates to the parameters
      cublasSetKernelStream(stream[6]);
      wei_gifo_x_bw_corr_.AddMatMat(1.0, bDGIFO.RowRange(1*S,T*S), kTrans, in, kNoTrans, mmt);
      cublasSetKernelStream(stream[7]);
      wei_gifo_m_bw_corr_.AddMatMat(1.0, bDGIFO.RowRange(1*S,T*S), kTrans, bYM.RowRange(2*S,T*S), kNoTrans, mmt);
      cublasSetKernelStream(stream[8]);
      bias_bw_corr_.AddRowSumMat(1.0, bDGIFO.RowRange(1*S,T*S), mmt);
      cublasSetKernelStream(stream[9]);
      phole_i_c_bw_corr_.AddDiagMatMat(1.0, bDI.RowRange(1*S,T*S), kTrans, bYC.RowRange(2*S,T*S), kNoTrans, mmt);
      cublasSetKernelStream(stream[10]);
      phole_f_c_bw_corr_.AddDiagMatMat(1.0, bDF.RowRange(1*S,T*S), kTrans, bYC.RowRange(2*S,T*S), kNoTrans, mmt);
      cublasSetKernelStream(stream[11]);
      phole_o_c_bw_corr_.AddDiagMatMat(1.0, bDO.RowRange(1*S,T*S), kTrans, bYC.RowRange(1*S,T*S), kNoTrans, mmt);
      //cudaDeviceSynchronize();
      for(int i =0; i< 12;i++){
       //destroy streams
        cudaStreamDestroy( stream[i] ) ;
      }
      cublasSetKernelStream(NULL);
    }

    void BackpropagateFncRecurrentDropoutFast(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out_diff_drop,
                                              //CuMatrixBase<BaseFloat> *in_diff, int32 T, int32 S) {
                                              int32 T, int32 S) {

      if (!in_train) KALDI_ERR << "Can't backpropagate in test mode";

      cudaStream_t stream[12];
      for (int i=0; i< 12; i++) {
         cudaStreamCreate( &stream[i] ) ;
      }

      // get the activations of the gates/units from the feedforward buffer; these variabiles will be used
      // in gradients computation
      CuSubMatrix<BaseFloat> fYG(propagate_buf_fw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> fYI(propagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYF(propagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYO(propagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYC(propagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYH(propagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fYM(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));

      CuSubMatrix<BaseFloat> bYG(propagate_buf_bw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> bYI(propagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYF(propagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYO(propagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYC(propagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYH(propagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bYM(propagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));


      // errors back-propagated to individual gates/units
      CuSubMatrix<BaseFloat> fDG(backpropagate_buf_fw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> fDI(backpropagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fDF(backpropagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fDO(backpropagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fDC(backpropagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fDH(backpropagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fDM(backpropagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> fDGIFO(backpropagate_buf_fw_.ColRange(0, 4 * cell_dim_));
      CuSubMatrix<BaseFloat> fDCM(d_c_mask_fw.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> fDHM(d_h_mask_fw.ColRange(0, cell_dim_));

      CuSubMatrix<BaseFloat> bDG(backpropagate_buf_bw_.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> bDI(backpropagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bDF(backpropagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bDO(backpropagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bDC(backpropagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bDH(backpropagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bDM(backpropagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));
      CuSubMatrix<BaseFloat> bDGIFO(backpropagate_buf_bw_.ColRange(0, 4 * cell_dim_));
      CuSubMatrix<BaseFloat> bDCM(d_c_mask_bw.ColRange(0, cell_dim_));
      CuSubMatrix<BaseFloat> bDHM(d_h_mask_bw.ColRange(0, cell_dim_));

      //  assume that the first half of out_diff is about the forward layer
      fDM.RowRange(1*S,T*S).CopyFromMat(out_diff_drop.ColRange(0, cell_dim_));
      // the second half of the error vector corresponds to the backward layer
      bDM.RowRange(1*S, T*S).CopyFromMat(out_diff_drop.ColRange(cell_dim_, cell_dim_));

      CuSubMatrix<BaseFloat> fr_mask, br_mask;

      // point the mask to the correct position
      if (recurrent_sequence_dropout) {
        fr_mask =  recurrent_drop_mask_fw_;
        br_mask =  recurrent_drop_mask_bw_;
      }

      for (int bt = 1, ft = T; bt <= T && ft >=1; bt++, ft--) {
        // variables representing activations of individual units/gates

        KALDI_ASSERT ((ft+bt) == T+1);

        CuSubMatrix<BaseFloat> fy_g(fYG.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fy_i(fYI.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fy_f(fYF.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fy_o(fYO.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fy_c(fYC.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fy_h(fYH.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fy_m(fYM.RowRange(ft*S, S));

        CuSubMatrix<BaseFloat> by_g(bYG.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_i(bYI.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_f(bYF.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_o(bYO.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_c(bYC.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_h(bYH.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> by_m(bYM.RowRange(bt*S, S));

        // variables representing errors of invidivual units/gates
        CuSubMatrix<BaseFloat> fd_g(fDG.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_i(fDI.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_f(fDF.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_o(fDO.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_c(fDC.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_h(fDH.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_m(fDM.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_all(backpropagate_buf_fw_.RowRange(ft*S, S));

        CuSubMatrix<BaseFloat> fd_c_m(fDCM.RowRange(ft*S, S));
        CuSubMatrix<BaseFloat> fd_h_m(fDHM.RowRange(ft*S, S));

        CuSubMatrix<BaseFloat> bd_g(bDG.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_i(bDI.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_f(bDF.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_o(bDO.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_c(bDC.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_h(bDH.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_m(bDM.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_all(backpropagate_buf_fw_.RowRange(bt*S, S));

        CuSubMatrix<BaseFloat> bd_c_m(bDCM.RowRange(bt*S, S));
        CuSubMatrix<BaseFloat> bd_h_m(bDHM.RowRange(bt*S, S));

        // point the mask to the correct position
        if (recurrent_step_dropout) {
          fr_mask =  recurrent_drop_mask_fw_.RowRange(ft*S,S);
          br_mask =  recurrent_drop_mask_bw_.RowRange(bt*S,S);
        }

        // d_m comes from two parts: errors from the upper layer and errors from the following frame (t+1)
        cublasSetKernelStream(stream[0]);
        fd_m.AddMatMat(1.0, fDGIFO.RowRange((ft+1)*S,S), kNoTrans, wei_gifo_m_fw_, kNoTrans, 1.0);
        cublasSetKernelStream(stream[1]);
        bd_m.AddMatMat(1.0, bDGIFO.RowRange((bt-1)*S,S), kNoTrans, wei_gifo_m_bw_, kNoTrans, 1.0);

        BackpropagatePointwiseOpsLSTM(fy_i, fy_f, fy_g, fy_o, fy_c, fy_h, fy_m,
                                      fd_i, fd_f, fd_g, fd_o, fd_c, fd_h, fd_m, fd_c_m,
                                      fDI.RowRange((ft+1)*S,S), fDF.RowRange((ft+1)*S,S),
                                      fDC.RowRange((ft+1)*S,S), fDCM.RowRange((ft+1)*S,S),
                                      fYF.RowRange((ft+1)*S,S), fYC.RowRange((ft-1)*S,S),
                                      phole_i_c_fw_, phole_f_c_fw_, phole_o_c_fw_,
                                      fr_mask, no_mem_loss_dropout, stream[0]);

        BackpropagatePointwiseOpsLSTM(by_i, by_f, by_g, by_o, by_c, by_h, by_m,
                                      bd_i, bd_f, bd_g, bd_o, bd_c, bd_h, bd_m, bd_c_m,
                                      bDI.RowRange((bt-1)*S,S), bDF.RowRange((bt-1)*S,S),
                                      bDC.RowRange((bt-1)*S,S), bDCM.RowRange((bt-1)*S,S),
                                      bYF.RowRange((bt-1)*S,S), bYC.RowRange((bt+1)*S,S),
                                      phole_i_c_bw_, phole_f_c_bw_, phole_o_c_bw_,
                                      br_mask, no_mem_loss_dropout, stream[1]);

      }  // end of t

      cublasSetKernelStream(NULL);
      //  updates to the model parameters
      const BaseFloat mmt = opts_.momentum;
      cublasSetKernelStream(stream[0]);
      wei_gifo_x_fw_corr_.AddMatMat(1.0, fDGIFO.RowRange(1*S, T*S), kTrans, in, kNoTrans, mmt);
      cublasSetKernelStream(stream[1]);
      wei_gifo_m_fw_corr_.AddMatMat(1.0, fDGIFO.RowRange(1*S, T*S), kTrans, fYM.RowRange(0*S,T*S), kNoTrans, mmt);
      cublasSetKernelStream(stream[2]);
      bias_fw_corr_.AddRowSumMat(1.0, fDGIFO.RowRange(1*S, T*S), mmt);
      cublasSetKernelStream(stream[3]);
      phole_i_c_fw_corr_.AddDiagMatMat(1.0, fDI.RowRange(1*S, T*S), kTrans, fYC.RowRange(0*S, T*S), kNoTrans, mmt);
      cublasSetKernelStream(stream[4]);
      phole_f_c_fw_corr_.AddDiagMatMat(1.0, fDF.RowRange(1*S, T*S), kTrans, fYC.RowRange(0*S, T*S), kNoTrans, mmt);
      cublasSetKernelStream(stream[5]);
      phole_o_c_fw_corr_.AddDiagMatMat(1.0, fDO.RowRange(1*S, T*S), kTrans, fYC.RowRange(1*S, T*S), kNoTrans, mmt);
      // updates to the parameters
      cublasSetKernelStream(stream[6]);
      wei_gifo_x_bw_corr_.AddMatMat(1.0, bDGIFO.RowRange(1*S,T*S), kTrans, in, kNoTrans, mmt);
      cublasSetKernelStream(stream[7]);
      wei_gifo_m_bw_corr_.AddMatMat(1.0, bDGIFO.RowRange(1*S,T*S), kTrans, bYM.RowRange(2*S,T*S), kNoTrans, mmt);
      cublasSetKernelStream(stream[8]);
      bias_bw_corr_.AddRowSumMat(1.0, bDGIFO.RowRange(1*S,T*S), mmt);
      cublasSetKernelStream(stream[9]);
      phole_i_c_bw_corr_.AddDiagMatMat(1.0, bDI.RowRange(1*S,T*S), kTrans, bYC.RowRange(2*S,T*S), kNoTrans, mmt);
      cublasSetKernelStream(stream[10]);
      phole_f_c_bw_corr_.AddDiagMatMat(1.0, bDF.RowRange(1*S,T*S), kTrans, bYC.RowRange(2*S,T*S), kNoTrans, mmt);
      cublasSetKernelStream(stream[11]);
      phole_o_c_bw_corr_.AddDiagMatMat(1.0, bDO.RowRange(1*S,T*S), kTrans, bYC.RowRange(1*S,T*S), kNoTrans, mmt);
      //cudaDeviceSynchronize();
      for(int i =0; i< 12;i++){
       //destroy streams
        cudaStreamDestroy( stream[i] ) ;
      }
      cublasSetKernelStream(NULL);
    }


    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                            const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
      int32 nstream_ = sequence_lengths_.size();  // the number of sequences to be processed in parallel
      KALDI_ASSERT(in.NumRows() % nstream_ == 0);
      int32 T = in.NumRows() / nstream_;
      int32 S = nstream_;

      bool apply_recurrent_dropout = in_train && (rnndrop || no_mem_loss_dropout) && (!twiddle_forward || (twiddle_forward && !twiddle_apply_forward));
      bool apply_forward_dropout   = in_train && forward_dropout > 0.0                       && (!twiddle_forward || (twiddle_forward &&  twiddle_apply_forward));

      CuMatrix<BaseFloat> out_diff_drop;
      out_diff_drop.Resize(out_diff.NumRows(), out_diff.NumCols());
      out_diff_drop.CopyFromMat(out_diff);
      if (apply_forward_dropout) {
        out_diff_drop.MulElements(forward_drop_mask_);
      }

      // initialize the back-propagation buffer
      backpropagate_buf_fw_.Resize((T+2)*S, 7 * cell_dim_, kSetZero);
      backpropagate_buf_bw_.Resize((T+2)*S, 7 * cell_dim_, kSetZero);
      // buffers for intermediate values
      d_h_mask_bw.Resize((T+2)*S, cell_dim_, kSetZero);
      d_c_mask_bw.Resize((T+2)*S, cell_dim_, kSetZero);

      // buffers for intermediate values
      d_h_mask_fw.Resize((T+2)*S, cell_dim_, kSetZero);
      d_c_mask_fw.Resize((T+2)*S, cell_dim_, kSetZero);

     // back-propagation in the forward then backard cell layer
      if (apply_recurrent_dropout) {
        BackpropagateFncRecurrentDropoutFast(in, out_diff_drop, T, S);//in_diff, T, S);
      } else {
        BackpropagateFncVanillaFast(in, out_diff_drop, in_diff, T, S);
      }

      cublasSetKernelStream(NULL);

      // errors back-propagated to the inputs
      CuSubMatrix<BaseFloat> DGIFO_FW(backpropagate_buf_fw_.ColRange(0, 4 * cell_dim_));
      CuSubMatrix<BaseFloat> DGIFO_BW(backpropagate_buf_bw_.ColRange(0, 4 * cell_dim_));

      in_diff->AddMatMat(1.0, DGIFO_FW.RowRange(1*S,T*S), kNoTrans, wei_gifo_x_fw_, kNoTrans, 0.0);
      in_diff->AddMatMat(1.0, DGIFO_BW.RowRange(1*S,T*S), kNoTrans, wei_gifo_x_bw_, kNoTrans, 1.0);

    }

private:

    int32 nstream_;
    std::vector<int> sequence_lengths_;

};
} // namespace eesen

#endif
