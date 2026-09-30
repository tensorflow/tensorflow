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

#ifndef XLA_BACKENDS_GPU_CODEGEN_KERNELS_PTX_CUSTOM_KERNEL_H_
#define XLA_BACKENDS_GPU_CODEGEN_KERNELS_PTX_CUSTOM_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::kernel {

absl::StatusOr<CustomKernel> GetPtxCustomKernel(std::string kernel_name,
                                                absl::string_view ptx,
                                                int num_args,
                                                se::BlockDim block_dim,
                                                se::ThreadDim thread_dim,
                                                size_t shared_memory_bytes = 0);

absl::StatusOr<CustomKernel> GetPtxCustomKernel(
    std::string kernel_name, absl::string_view ptx, int num_args,
    se::BlockDim block_dim, se::ThreadDim thread_dim,
    se::ClusterDim cluster_dim, size_t shared_memory_bytes = 0);

absl::StatusOr<CustomKernel> GetOwnedPtxCustomKernel(
    std::string kernel_name, std::string ptx, int num_args,
    se::BlockDim block_dim, se::ThreadDim thread_dim,
    size_t shared_memory_bytes = 0);

absl::StatusOr<CustomKernel> CreateOwnedCubinCustomKernel(
    std::string kernel_name, std::vector<uint8_t> cubin, int num_args,
    se::BlockDim block_dim, se::ThreadDim thread_dim,
    size_t shared_memory_bytes);

// Like CreateOwnedCubinCustomKernel, but the CUBIN buffer is shared instead of
// copied. Prefer this whenever the CUBIN is already held in a reference counted
// buffer (e.g. it comes from the KernelReuseCache), so that a kernel that is
// invoked multiple times in a module is only stored once.
//
// `cubin` must not be null.
absl::StatusOr<CustomKernel> CreateSharedCubinCustomKernel(
    std::string kernel_name, std::shared_ptr<const std::vector<uint8_t>> cubin,
    int num_args, se::BlockDim block_dim, se::ThreadDim thread_dim,
    size_t shared_memory_bytes);

}  // namespace xla::gpu::kernel
#endif  // XLA_BACKENDS_GPU_CODEGEN_KERNELS_PTX_CUSTOM_KERNEL_H_
