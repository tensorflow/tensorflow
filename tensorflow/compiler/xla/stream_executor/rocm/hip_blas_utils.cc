/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/stream_executor/rocm/hip_blas_utils.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"

namespace stream_executor {
namespace rocm {

tsl::Status ToStatus(hipblasStatus_t status, const char* prefix) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return tsl::errors::Internal(absl::StrCat(
        prefix, ": ",
        "HipblasLt error " + std::to_string(static_cast<int>(status))));
  }
  return tsl::OkStatus();
}

hipblasDatatype_t AsHipblasDataType(blas::DataType type) {
  switch (type) {
    case blas::DataType::kF8E5M2:
    case blas::DataType::kF8E4M3FN:
      LOG(FATAL) << "hipblaslt does not support F8 yet";
    case blas::DataType::kHalf:
      return HIPBLAS_R_16F;
    case blas::DataType::kBF16:
      return HIPBLAS_R_16B;
    case blas::DataType::kFloat:
      return HIPBLAS_R_32F;
    case blas::DataType::kDouble:
      return HIPBLAS_R_64F;
    case blas::DataType::kInt8:
      return HIPBLAS_R_8I;
    case blas::DataType::kInt32:
      return HIPBLAS_R_32I;
    case blas::DataType::kComplexFloat:
      return HIPBLAS_C_32F;
    case blas::DataType::kComplexDouble:
      return HIPBLAS_C_64F;
    default:
      LOG(FATAL) << "unknown data type";
  }
}

hipblasLtComputeType_t AsHipblasComputeType(blas::ComputationType type) {
  if (type == blas::ComputationType::kF32 ||
      type == blas::ComputationType::kTF32AsF32)
    return HIPBLASLT_COMPUTE_F32;
  else
    LOG(FATAL) << "unsupported hipblaslt computation type";
}

hipblasOperation_t AsHipblasOperation(blas::Transpose trans) {
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return HIPBLAS_OP_N;
    case blas::Transpose::kTranspose:
      return HIPBLAS_OP_T;
    case blas::Transpose::kConjugateTranspose:
      return HIPBLAS_OP_C;
  }
}

}  // namespace rocm
}  // namespace stream_executor
