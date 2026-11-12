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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/stream_executor/kernel_args_packing_spec.h"
#include "xla/stream_executor/kernel_spec.h"

namespace xla::gpu::kernel {
namespace se = ::stream_executor;

absl::StatusOr<CustomKernel> GetPtxCustomKernel(const GpuKernelSpec& spec,
                                                absl::string_view ptx) {
  int num_args = spec.kernel_arguments().args().size();
  se::KernelLoaderSpec kernel_spec =
      se::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
          ptx, std::string{spec.name()}, /*arity=*/num_args,
          se::KernelArgsPackingSpec::Identity(num_args));
  return CustomKernel(spec, std::move(kernel_spec));
}

absl::StatusOr<CustomKernel> GetOwnedPtxCustomKernel(const GpuKernelSpec& spec,
                                                     std::string ptx) {
  int num_args = spec.kernel_arguments().args().size();
  se::KernelLoaderSpec kernel_spec =
      se::KernelLoaderSpec::CreateOwningCudaPtxInMemorySpec(
          std::move(ptx), std::string{spec.name()}, /*arity=*/num_args,
          se::KernelArgsPackingSpec::Identity(num_args));
  return CustomKernel(spec, std::move(kernel_spec));
}

absl::StatusOr<CustomKernel> CreateOwnedCubinCustomKernel(
    const GpuKernelSpec& spec, std::vector<uint8_t> cubin) {
  int num_args = spec.kernel_arguments().args().size();
  se::KernelLoaderSpec kernel_spec =
      se::KernelLoaderSpec::CreateOwningCudaCubinInMemorySpec(
          std::move(cubin), std::string{spec.name()}, /*arity=*/num_args,
          se::KernelArgsPackingSpec::Identity(num_args));
  return CustomKernel(spec, std::move(kernel_spec));
}

absl::StatusOr<CustomKernel> CreateSharedCubinCustomKernel(
    const GpuKernelSpec& spec,
    std::shared_ptr<const std::vector<uint8_t>> cubin) {
  int num_args = spec.kernel_arguments().args().size();
  se::KernelLoaderSpec kernel_spec =
      se::KernelLoaderSpec::CreateSharedCudaCubinInMemorySpec(
          std::move(cubin), std::string{spec.name()}, /*arity=*/num_args,
          se::KernelArgsPackingSpec::Identity(num_args));
  return CustomKernel(spec, std::move(kernel_spec));
}

}  // namespace xla::gpu::kernel
