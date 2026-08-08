#include "absl/base/casts.h"
/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

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
#include <string>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/kernel_spec.h"

namespace stream_executor::cuda {

struct CudaRuntimeKernel {
  absl::Span<const uint8_t> cubin;
  absl::string_view name;
};

// Returns the CUBIN and kernel symbol name of the kernel that has been
// registered in the CUDA runtime with the given host function pointer.
// Returns an error status if the kernel is not found.
// Usage:
// Assuming you would call a CUDA kernel like this:
//   MyKernel<<<1, 2, 3>>>(a, b, c);
// where `MyKernel` is a CUDA C++ kernel that has been linked into the binary,
// then you can get a KernelLoaderSpec for the kernel by calling:
//   auto cubin_spec = FindCudaRuntimeKernel(MyKernel);
absl::StatusOr<CudaRuntimeKernel> FindCudaRuntimeKernel(const void* host_fun);

template <typename ReturnT, typename... Args>
absl::StatusOr<KernelLoaderSpec> FindCudaRuntimeKernel(
    ReturnT (*host_fun)(Args...)) {
  ABSL_ASSIGN_OR_RETURN(
      CudaRuntimeKernel kernel,
      FindCudaRuntimeKernel(absl::bit_cast<const void*>(host_fun)));
  return KernelLoaderSpec::CreateCudaCubinInMemorySpec(
      kernel.cubin, std::string(kernel.name), sizeof...(Args));
}

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDART_KERNEL_REGISTRY_H_
