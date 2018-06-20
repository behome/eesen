// gpucompute/cuda-rand.cc

// Copyright 2012  Karel Vesely
//           2013  Johns Hopkins University (author: Daniel Povey)

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


#include "base/kaldi-math.h"
#include "gpucompute/cuda-vector.h"
#include "gpucompute/cuda-matrix.h"
#include "gpucompute/cuda-rand.h"

namespace eesen {

#if HAVE_CUDA == 1
/// Wrappers of curand functions to interface both float and double as 1 function,

/// Wrapper of curandGenerateUniform(), curandGenerateUniformDouble(),
template<typename Real>
curandStatus_t curandGenerateUniformWrap(curandGenerator_t gen, Real *ptr, size_t num);
//
template<>
curandStatus_t curandGenerateUniformWrap(curandGenerator_t gen, float *ptr, size_t num) {
  return curandGenerateUniform(gen, ptr, num);
}
template<>
curandStatus_t curandGenerateUniformWrap(curandGenerator_t gen, double *ptr, size_t num) {
  return curandGenerateUniformDouble(gen, ptr, num);
}

/// Wrapper of curandGenerateNormal(), curandGenerateNormalDouble(),
template<typename Real>
curandStatus_t curandGenerateNormalWrap(
    curandGenerator_t gen, Real *ptr, size_t num);
//
template<>
curandStatus_t curandGenerateNormalWrap<float>(
    curandGenerator_t gen, float *ptr, size_t num) {
  return curandGenerateNormal(gen, ptr, num, 0.0 /*mean*/, 1.0 /*stddev*/);
}
template<>
curandStatus_t curandGenerateNormalWrap<double>(
    curandGenerator_t gen, double *ptr, size_t num) {
  return curandGenerateNormalDouble(gen, ptr, num, 0.0 /*mean*/, 1.0 /*stddev*/);
}
/// End of wrappers.
#endif

template<typename Real>
void CuRand<Real>::RandUniform(CuMatrixBase<Real> *tgt) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    // Better use 'tmp' matrix, 'tgt' can be a window into a larger matrix,
    // so we should not use it to generate random numbers over whole stride.
    // We're not using the kaldi option kStrideEqualNumCols - will revisit if needed
    CuMatrix<Real> tmp(tgt->NumRows(), tgt->NumCols(), kUndefined);
    size_t s = static_cast<size_t>(tmp.NumRows()) * static_cast<size_t>(tmp.Stride());
    CURAND_SAFE_CALL(curandGenerateUniformWrap(gen_, tmp.Data(), s));
    tgt->CopyFromMat(tmp);
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    tgt->SetRandUniform();
  }
}

template<typename Real>
void CuRand<Real>::RandUniform(CuVectorBase<Real> *tgt) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CURAND_SAFE_CALL(curandGenerateUniformWrap(gen_, tgt->Data(), tgt->Dim()));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    // not implemented tgt->Vec().SetRandUniform();
    KALDI_ERR << "CuRand<Real>::RandUniform not implemented for non CUDA env";
  }
}

template<typename Real>
void CuRand<Real>::RandGaussian(CuMatrixBase<Real> *tgt) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    // Better use 'tmp' matrix, 'tgt' can be a window into a larger matrix,
    // so we should not use it to generate random numbers over whole stride.
    // Also, we ensure to have 'even' number of elements for calling 'curand'
    // by possibly adding one column. Even number of elements is required by
    // curandGenerateUniform(), curandGenerateUniformDouble().
    // We're not using the kaldi option kStrideEqualNumCols - will revisit if needed

    MatrixIndexT num_cols_even = tgt->NumCols() + (tgt->NumCols() % 2); // + 0 or 1,
    CuMatrix<Real> tmp(tgt->NumRows(), num_cols_even, kUndefined);
    CURAND_SAFE_CALL(curandGenerateNormalWrap(gen_, tmp.Data(), tmp.NumRows()*tmp.Stride()));
    tgt->CopyFromMat(tmp.ColRange(0,tgt->NumCols()));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    tgt->SetRandn();
  }
}

template<typename Real>
void CuRand<Real>::RandGaussian(CuVectorBase<Real> *tgt) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    // To ensure 'even' number of elements, we use 'tmp' vector of even length.
    // Even number of elements is required by 'curand' functions:
    // curandGenerateUniform(), curandGenerateUniformDouble().
    MatrixIndexT num_elements = tgt->Dim();
    if (0 == (num_elements % 2)) {
      CURAND_SAFE_CALL(curandGenerateNormalWrap(gen_, tgt->Data(), tgt->Dim()));
    } else {
      MatrixIndexT dim_even = tgt->Dim() + (tgt->Dim() % 2); // + 0 or 1,
      CuVector<Real> tmp(dim_even, kUndefined);
      CURAND_SAFE_CALL(curandGenerateNormalWrap(gen_, tmp.Data(), tmp.Dim()));
      tgt->CopyFromVec(tmp.Range(0,tgt->Dim()));
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    tgt->Vec().SetRandn();
  }
}

/// convert probabilities binary values,
template<typename Real>
void CuRand<Real>::BinarizeProbs(const CuMatrix<Real> &probs, CuMatrix<Real> *states) {
  CuMatrix<Real> tmp(probs.NumRows(), probs.NumCols());
  this->RandUniform(&tmp);  // [0..1]
  tmp.Scale(-1.0);  // [-1..0]
  tmp.AddMat(1.0, probs);  // [-1..+1]
  tmp.ApplyHeaviside();
  states->CopyFromMat(tmp);  // negative
}

/// add gaussian noise to each element
template<typename Real>
void CuRand<Real>::AddGaussNoise(CuMatrix<Real> *tgt, Real gscale) {
  // We're not using the kaldi option kStrideEqualNumCols - will revisit if needed
  CuMatrix<Real> tmp(tgt->NumRows(), tgt->NumCols(), kUndefined);
  this->RandGaussian(&tmp);
  tgt->AddMat(gscale, tmp);
}

// Instantiate the class for float and double.
template class CuRand<float>;
template class CuRand<double>;

} // namespace




