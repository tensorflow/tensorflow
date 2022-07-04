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

#include "tensorflow/stream_executor/cuda/cuda_blas_utils.h"

#include "absl/strings/str_cat.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/stream_executor/blas.h"

namespace stream_executor {
namespace cuda {

const char* ToString(cublasStatus_t status) {
#if CUDA_VERSION >= 11050  // `GetStatusString` was added in 11.4 update 2.
  return cublasGetStatusString(status);
#else
  return "cublas error";
#endif  // CUDA_VERSION >= 11050
}

port::Status ToStatus(cublasStatus_t status, const char* prefix) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(port::error::INTERNAL,
                        absl::StrCat(prefix, ": ", ToString(status)));
  }
  return port::Status::OK();
}

cudaDataType_t AsCudaDataType(blas::DataType type) {
  switch (type) {
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
