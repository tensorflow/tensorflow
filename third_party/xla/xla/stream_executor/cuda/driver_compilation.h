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

#ifndef XLA_STREAM_EXECUTOR_CUDA_DRIVER_COMPILATION_H_
#define XLA_STREAM_EXECUTOR_CUDA_DRIVER_COMPILATION_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

// Links multiple relocatable GPU images (e.g. results of ptxas -c) into a
// single image using the CUDA driver linking API.
absl::StatusOr<std::vector<uint8_t>> LinkGpuAsmUsingDriver(
    StreamExecutor* executor, const CudaComputeCapability& cc,
    absl::Span<const std::vector<uint8_t>> images);

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_DRIVER_COMPILATION_H_
