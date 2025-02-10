/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_STATUS_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_STATUS_H_

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

namespace stream_executor::cuda {

namespace internal {
// Helper method to handle the slow path of ToStatus.  Assumes a non-successful
// result code.
absl::Status ToStatusSlow(CUresult result, absl::string_view detail);
absl::Status ToStatusSlow(cudaError_t result, absl::string_view detail);
}  // namespace internal

// Returns an absl::Status corresponding to the CUresult.
inline absl::Status ToStatus(CUresult result, absl::string_view detail = "") {
  if (ABSL_PREDICT_TRUE(result == CUDA_SUCCESS)) {
    return absl::OkStatus();
  }
  return internal::ToStatusSlow(result, detail);
}

// Returns an absl::Status corresponding to the cudaError_t (CUDA runtime API
// error type). The string `detail` will be included in the error message.
inline absl::Status ToStatus(cudaError_t result,
                             absl::string_view detail = "") {
  if (ABSL_PREDICT_TRUE(result == cudaSuccess)) {
    return absl::OkStatus();
  }
  return internal::ToStatusSlow(result, detail);
}

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_STATUS_H_
