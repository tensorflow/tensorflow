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

#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/stream_executor/kernel_args_packing_spec.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::kernel {
namespace se = ::stream_executor;

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
          ptx, kernel_name, /*arity=*/num_args,
          se::KernelArgsPackingSpec::Identity(num_args));
  return CustomKernel(std::move(kernel_name), kernel_spec, block_dim,
                      thread_dim, shared_memory_bytes);
}

absl::StatusOr<CustomKernel> GetPtxCustomKernel(
    std::string kernel_name, absl::string_view ptx, int num_args,
    se::BlockDim block_dim, se::ThreadDim thread_dim,
    se::ClusterDim cluster_dim, size_t shared_memory_bytes) {
  se::KernelLoaderSpec kernel_spec =
      se::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
          ptx, kernel_name, /*arity=*/num_args,
          se::KernelArgsPackingSpec::Identity(num_args));
  return CustomKernel(std::move(kernel_name), kernel_spec, block_dim,
                      thread_dim, cluster_dim, shared_memory_bytes);
}

absl::StatusOr<CustomKernel> GetOwnedPtxCustomKernel(
    std::string kernel_name, std::string ptx, int num_args,
    se::BlockDim block_dim, se::ThreadDim thread_dim,
    size_t shared_memory_bytes) {
  se::KernelLoaderSpec kernel_spec =
      se::KernelLoaderSpec::CreateOwningCudaPtxInMemorySpec(
          std::move(ptx), kernel_name, /*arity=*/num_args,
          se::KernelArgsPackingSpec::Identity(num_args));
  return CustomKernel(std::move(kernel_name), kernel_spec, block_dim,
                      thread_dim, shared_memory_bytes);
}

absl::StatusOr<CustomKernel> CreateOwnedCubinCustomKernel(
    std::string kernel_name, std::vector<uint8_t> cubin, int num_args,
    se::BlockDim block_dim, se::ThreadDim thread_dim,
    size_t shared_memory_bytes) {
  se::KernelLoaderSpec kernel_spec =
      se::KernelLoaderSpec::CreateOwningCudaCubinInMemorySpec(
          std::move(cubin), kernel_name, /*arity=*/num_args,
          se::KernelArgsPackingSpec::Identity(num_args));
  return CustomKernel(std::move(kernel_name), std::move(kernel_spec), block_dim,
                      thread_dim, shared_memory_bytes);
}

absl::StatusOr<CustomKernel> CreateSharedCubinCustomKernel(
    const GpuKernelSpec& spec,
    std::shared_ptr<const std::vector<uint8_t>> cubin) {
  int num_args = spec.kernel_arguments().args().size();
  se::KernelLoaderSpec kernel_spec =
      se::KernelLoaderSpec::CreateSharedCudaCubinInMemorySpec(
          std::move(cubin), std::string{spec.name()}, /*arity=*/num_args,
          se::KernelArgsPackingSpec::Identity(num_args));
  if (spec.cluster_dim().has_value()) {
    return CustomKernel(std::string{spec.name()}, std::move(kernel_spec),
                        spec.launch_dimensions().block_counts(),
                        spec.launch_dimensions().thread_counts_per_block(),
                        *spec.cluster_dim(), spec.shmem_bytes());
  }
  return CustomKernel(spec, std::move(kernel_spec));
}

}  // namespace xla::gpu::kernel
