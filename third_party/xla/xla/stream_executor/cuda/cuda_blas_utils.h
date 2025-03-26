/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_BLAS_UTILS_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_BLAS_UTILS_H_


#include "absl/status/status.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/library_types.h"
#include "xla/stream_executor/blas.h"
#include "tsl/platform/errors.h"

#define SE_CUBLAS_RETURN_IF_ERROR(expr) \
  TF_RETURN_IF_ERROR(::stream_executor::cuda::ToStatus(expr, #expr))

namespace stream_executor {
namespace cuda {

const char* ToString(cublasStatus_t status);
absl::Status ToStatus(cublasStatus_t status, const char* prefix = "cublasLt");
cudaDataType_t AsCudaDataType(blas::DataType type);
cublasComputeType_t AsCublasComputeType(blas::ComputationType type);
cublasOperation_t AsCublasOperation(blas::Transpose trans);

}  // namespace cuda
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_BLAS_UTILS_H_
