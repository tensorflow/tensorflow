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

#include "xla/service/gpu/kernels/ptx_custom_kernel.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::kernel {

namespace se = ::stream_executor;

template <int n>
absl::StatusOr<std::unique_ptr<se::KernelArgsPackedArrayBase>>
KernelArgsPacking(const se::Kernel &kernel, const se::KernelArgs &args) {
  auto *mem_args = se::Cast<se::KernelArgsDeviceMemoryArray>(&args);

  auto packed_args =
      std::make_unique<stream_executor::KernelArgsPackedArray<n>>();
  int number_of_arguments = mem_args->number_of_arguments();
  if (mem_args->number_of_shared_bytes() > 0) {
    number_of_arguments--;
  }
  for (int i = 0; i < number_of_arguments; ++i) {
    packed_args->add_argument(mem_args->device_memory_ptr(i));
  }

  return packed_args;
}

// Note: Make sure that the kernel_name matches the kernel name in the ptx,
// otherwise you will get a "CUDA_ERROR_NOT_FOUND: named symbol not found.".
// E.g. `.visible .entry AddI32(...)` would have a kernel name of "AddI32".
template <int n>
absl::StatusOr<CustomKernel> GetPtxCustomKernel(std::string kernel_name,
                                                std::string_view ptx,
                                                se::BlockDim block_dim,
                                                se::ThreadDim thread_dim,
                                                size_t shared_memory_bytes) {
  // LINT.IfChange(check_n)
  if (n != 3) {
    // LINT.ThenChange(:explicit_instantiations)
    return absl::UnimplementedError(
        "Only 3 arguments are supported for PTX custom kernels.");
  }

  se::MultiKernelLoaderSpec kernel_spec(/*arity=*/n, KernelArgsPacking<n>);
  kernel_spec.AddCudaPtxInMemory(ptx, kernel_name);
  return CustomKernel(kernel_name, kernel_spec, block_dim, thread_dim,
                      /*shared_memory_bytes=*/shared_memory_bytes);
};

// LINT.IfChange(explicit_instantiations)
template absl::StatusOr<std::unique_ptr<se::KernelArgsPackedArrayBase>>
KernelArgsPacking<3>(const se::Kernel &kernel, const se::KernelArgs &args);

template absl::StatusOr<CustomKernel> GetPtxCustomKernel<3>(
    std::string kernel_name, std::string_view ptx, se::BlockDim block_dim,
    se::ThreadDim thread_dim, size_t shared_memory_bytes);
// LINT.ThenChange(:check_n)

}  // namespace xla::gpu::kernel
