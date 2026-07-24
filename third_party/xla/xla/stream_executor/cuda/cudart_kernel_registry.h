/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDART_KERNEL_REGISTRY_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDART_KERNEL_REGISTRY_H_

#include <cstdint>
#include <optional>

#include "absl/types/span.h"

namespace stream_executor::cuda {

// Returns a reference to the CUBIN of the kernel that has been registered
// in the CUDA runtime with the given host function pointer.
// Returns std::nullopt if the kernel is not found.
// Usage:
// Assuming you would call a CUDA kernel like this:
//   MyKernel<<<1, 2, 3>>>(a, b, c);
// where `MyKernel` is a CUDA C++ kernel that has been linked into the binary,
// then you can get the CUBIN of the kernel by calling:
//   auto cubin_span = FindCudaRuntimeKernel(MyKernel);
std::optional<absl::Span<const uint8_t>> FindCudaRuntimeKernel(
    const void* host_fun);

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDART_KERNEL_REGISTRY_H_
