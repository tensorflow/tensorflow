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

#ifndef XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_UTILS_H_
#define XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_UTILS_H_

#include <string>

#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/rocm/hipblaslt_wrapper.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

#if TF_HIPBLASLT

#if TF_ROCM_VERSION < 50700
#define hipblasltDatatype_t hipblasDatatype_t
#define HIPBLASLT_R_16F HIPBLAS_R_16F
#define HIPBLASLT_R_16B HIPBLAS_R_16B
#define HIPBLASLT_R_32F HIPBLAS_R_32F
#define HIPBLASLT_R_64F HIPBLAS_R_64F
#define HIPBLASLT_R_8I HIPBLAS_R_8I
#define HIPBLASLT_R_32I HIPBLAS_R_32I
#define HIPBLASLT_C_32F HIPBLAS_C_32F
#define HIPBLASLT_C_64F HIPBLAS_C_64F
#endif

namespace stream_executor {
namespace rocm {

#define SE_HIPBLAS_RETURN_IF_ERROR(expr) \
  TF_RETURN_IF_ERROR(::stream_executor::rocm::ToStatus(expr, #expr))

tsl::Status ToStatus(hipblasStatus_t status, const char* prefix);
hipblasltDatatype_t AsHipblasDataType(blas::DataType type);
hipblasLtComputeType_t AsHipblasComputeType(blas::ComputationType type);
hipblasOperation_t AsHipblasOperation(blas::Transpose trans);

}  // namespace rocm
}  // namespace stream_executor

#endif  // TF_HIPBLASLT

#endif  // XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_UTILS_H_
