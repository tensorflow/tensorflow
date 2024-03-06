/* Copyright 2023 The OpenXLA Authors.

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

#include "absl/status/status.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/rocm/hipblaslt_wrapper.h"
#include "tsl/platform/errors.h"

#if TF_HIPBLASLT

#if TF_ROCM_VERSION < 60000
#define hipDataType hipblasDatatype_t
#define HIP_R_16F HIPBLAS_R_16F
#define HIP_R_16BF HIPBLAS_R_16B
#define HIP_R_32F HIPBLAS_R_32F
#define HIP_R_64F HIPBLAS_R_64F
#define HIP_R_8I HIPBLAS_R_8I
#define HIP_R_32I HIPBLAS_R_32I
#define HIP_C_32F HIPBLAS_C_32F
#define HIP_C_64F HIPBLAS_C_64F

#define hipblasComputeType_t hipblasLtComputeType_t
#define HIPBLAS_COMPUTE_32F HIPBLASLT_COMPUTE_F32
#define HIPBLAS_COMPUTE_64F HIPBLASLT_COMPUTE_F64
#define HIPBLAS_COMPUTE_32I HIPBLASLT_COMPUTE_I32
#endif

namespace stream_executor {
namespace rocm {

#define SE_HIPBLAS_RETURN_IF_ERROR(expr) \
  TF_RETURN_IF_ERROR(::stream_executor::rocm::ToStatus(expr, #expr))

absl::Status ToStatus(hipblasStatus_t status, const char* prefix);
hipDataType AsHipblasDataType(blas::DataType type);
hipblasComputeType_t AsHipblasComputeType(blas::ComputationType type);
hipblasOperation_t AsHipblasOperation(blas::Transpose trans);

}  // namespace rocm
}  // namespace stream_executor

#endif  // TF_HIPBLASLT

#endif  // XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_UTILS_H_
