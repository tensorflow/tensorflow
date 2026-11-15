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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"

namespace xla::gpu::kernel {

absl::StatusOr<CustomKernel> GetPtxCustomKernel(const GpuKernelSpec& spec,
                                                absl::string_view ptx);

absl::StatusOr<CustomKernel> GetOwnedPtxCustomKernel(const GpuKernelSpec& spec,
                                                     std::string ptx);

absl::StatusOr<CustomKernel> CreateOwnedCubinCustomKernel(
    const GpuKernelSpec& spec, std::vector<uint8_t> cubin);

// Like CreateOwnedCubinCustomKernel, but the CUBIN buffer is shared instead of
// copied. Prefer this whenever the CUBIN is already held in a reference counted
// buffer (e.g. it comes from the KernelReuseCache), so that a kernel that is
// invoked multiple times in a module is only stored once.
//
// `cubin` must not be null.
absl::StatusOr<CustomKernel> CreateSharedCubinCustomKernel(
    const GpuKernelSpec& spec,
    std::shared_ptr<const std::vector<uint8_t>> cubin);

}  // namespace xla::gpu::kernel
#endif  // XLA_BACKENDS_GPU_CODEGEN_KERNELS_PTX_CUSTOM_KERNEL_H_
