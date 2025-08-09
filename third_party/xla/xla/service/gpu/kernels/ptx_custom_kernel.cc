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
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::kernel {

namespace se = ::stream_executor;

absl::StatusOr<std::unique_ptr<se::KernelArgsPackedArrayBase>>
KernelArgsPacking(const se::Kernel &kernel, const se::KernelArgs &args) {
  auto *mem_args = se::Cast<se::KernelArgsDeviceMemoryArray>(&args);

  return se::PackKernelArgs<se::DeviceMemoryBase>(
      mem_args->device_memory_args(), mem_args->number_of_shared_bytes());
}

// Note: Make sure that the kernel_name matches the kernel name in the ptx,
// otherwise you will get a "CUDA_ERROR_NOT_FOUND: named symbol not found.".
// E.g. `.visible .entry AddI32(...)` would have a kernel name of "AddI32".
absl::StatusOr<CustomKernel> GetPtxCustomKernel(std::string kernel_name,
                                                absl::string_view ptx,
                                                int num_args,
                                                se::BlockDim block_dim,
                                                se::ThreadDim thread_dim,
                                                size_t shared_memory_bytes) {
  se::KernelLoaderSpec kernel_spec =
      se::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
          ptx, kernel_name, /*arity=*/num_args, KernelArgsPacking);
  return CustomKernel(std::move(kernel_name), kernel_spec, block_dim,
                      thread_dim,
                      /*shared_memory_bytes=*/shared_memory_bytes);
};

}  // namespace xla::gpu::kernel
