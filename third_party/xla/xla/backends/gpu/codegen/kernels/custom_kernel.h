/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_KERNELS_CUSTOM_KERNEL_H_
#define XLA_BACKENDS_GPU_CODEGEN_KERNELS_CUSTOM_KERNEL_H_

#include <cstddef>
#include <optional>
#include <string>

#include "absl/base/macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu {
namespace se = ::stream_executor;  // NOLINT

// Custom kernel is a mechanism for plugging pre-compiled device kernels into
// XLA GPU runtime. Custom kernel defines how to load the kernel on an executor
// and what are the grid size requirements for running it.
//
// We use this API to hide kernel implementation details from XLA (e.g. we can
// export CUTLASS gemm kernels as custom kernels to XLA), so that XLA can only
// implement generic interfaces, e.g. command buffer implementation for CUDA
// can automatically add a kernel node to a graph for arbitrary custom kernel.
class CustomKernel {
 public:
  CustomKernel(const GpuKernelSpec& spec, se::KernelLoaderSpec kernel_spec);

  absl::string_view name() const;

  ABSL_DEPRECATE_AND_INLINE()
  const se::KernelLoaderSpec& kernel_spec() const {
    return kernel_loader_spec();
  }

  const se::KernelLoaderSpec& kernel_loader_spec() const {
    return loader_spec_;
  }

  const se::BlockDim& block_dims() const;

  const se::ThreadDim& thread_dims() const;

  const std::optional<se::ClusterDim>& cluster_dims() const;

  size_t shared_memory_bytes() const;

  std::string ToString() const;

  absl::StatusOr<CustomKernelProto> ToProto() const;

  static absl::StatusOr<CustomKernel> FromProto(
      const CustomKernelProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations,
      const std::optional<se::KernelLoaderSpec::SymbolResolver>&
          symbol_resolver = std::nullopt);

 private:
  se::KernelLoaderSpec loader_spec_;
  GpuKernelSpec spec_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_KERNELS_CUSTOM_KERNEL_H_
