/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_blas_utils.h"

#include "absl/strings/str_cat.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"

namespace stream_executor {
namespace cuda {

const char* ToString(cublasStatus_t status) {
#if CUDA_VERSION >= 11050  // `GetStatusString` was added in 11.4 update 2.
  return cublasGetStatusString(status);
#else
  return "cublas error";
#endif  // CUDA_VERSION >= 11050
}

tsl::Status ToStatus(cublasStatus_t status, const char* prefix) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    return tsl::Status(absl::StatusCode::kInternal,
                       absl::StrCat(prefix, ": ", ToString(status)));
  }
  return tsl::OkStatus();
}

cudaDataType_t AsCudaDataType(blas::DataType type) {
  switch (type) {
#if CUDA_VERSION >= 11080
    case blas::DataType::kF8E5M2:
      return CUDA_R_8F_E5M2;
    case blas::DataType::kF8E4M3FN:
      return CUDA_R_8F_E4M3;
#endif
    case blas::DataType::kHalf:
      return CUDA_R_16F;
    case blas::DataType::kBF16:
      return CUDA_R_16BF;
    case blas::DataType::kFloat:
      return CUDA_R_32F;
    case blas::DataType::kDouble:
      return CUDA_R_64F;
    case blas::DataType::kInt8:
      return CUDA_R_8I;
    case blas::DataType::kInt32:
      return CUDA_R_32I;
    case blas::DataType::kComplexFloat:
      return CUDA_C_32F;
    case blas::DataType::kComplexDouble:
      return CUDA_C_64F;
    default:
      LOG(FATAL) << "unknown data type";
  }
}

cublasComputeType_t AsCublasComputeType(blas::ComputationType type) {
  switch (type) {
    case blas::ComputationType::kF16:
      return CUBLAS_COMPUTE_16F;
    case blas::ComputationType::kF32:
      return CUBLAS_COMPUTE_32F;
    case blas::ComputationType::kF64:
      return CUBLAS_COMPUTE_64F;
    case blas::ComputationType::kI32:
      return CUBLAS_COMPUTE_32I;
    case blas::ComputationType::kF16AsF32:
      return CUBLAS_COMPUTE_32F_FAST_16F;
    case blas::ComputationType::kBF16AsF32:
      return CUBLAS_COMPUTE_32F_FAST_16BF;
    case blas::ComputationType::kTF32AsF32:
      return CUBLAS_COMPUTE_32F_FAST_TF32;
  }
}

cublasOperation_t AsCublasOperation(blas::Transpose trans) {
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return CUBLAS_OP_N;
    case blas::Transpose::kTranspose:
      return CUBLAS_OP_T;
    case blas::Transpose::kConjugateTranspose:
      return CUBLAS_OP_C;
  }
}

}  // namespace cuda
}  // namespace stream_executor
