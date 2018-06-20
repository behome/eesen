//#ifndef KALDI_CUDAMATRIX_COMMON_H_
//#define KALDI_CUDAMATRIX_COMMON_H_

// This file contains some #includes, forward declarations
// and typedefs that are needed by all the main header
// files in this directory.

#include "base/kaldi-common.h"
#include "cpucompute/blas.h"
#include "gpucompute/cuda-device.h"
#include "gpucompute/cuda-common.h"

namespace eesen {

#if HAVE_CUDA == 1
cublasOperation_t KaldiTransToCuTrans(MatrixTransposeType kaldi_trans) {
  cublasOperation_t cublas_trans;

  if (kaldi_trans == kNoTrans)
    cublas_trans = CUBLAS_OP_N;
  else if (kaldi_trans == kTrans)
    cublas_trans = CUBLAS_OP_T;
  else
    cublas_trans = CUBLAS_OP_C;
  return cublas_trans;
}

const char* cublasGetStatusString(cublasStatus_t status) {
  switch(status) {
    case CUBLAS_STATUS_SUCCESS:           return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:   return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:     return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:     return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:     return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:  return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:    return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:     return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:     return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "CUBLAS_STATUS_UNKNOWN_ERROR";
}

const char* curandGetStatusString(curandStatus_t status) {
  // detail info come from http://docs.nvidia.com/cuda/curand/group__HOST.html
  switch(status) {
    case CURAND_STATUS_SUCCESS:                     return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:            return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:             return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:           return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:                  return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:                return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:         return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:   return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:              return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:         return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:       return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:               return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:              return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "CURAND_STATUS_UNKNOWN_ERROR";
}

#endif

} // namespace


//#endif  // KALDI_CUDAMATRIX_COMMON_H_
