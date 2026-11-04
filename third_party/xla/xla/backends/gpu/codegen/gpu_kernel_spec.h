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

#ifndef XLA_BACKENDS_GPU_CODEGEN_GPU_KERNEL_SPEC_H_
#define XLA_BACKENDS_GPU_CODEGEN_GPU_KERNEL_SPEC_H_

#include <cstddef>
#include <optional>

#include "absl/strings/string_view.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/gpu/tma_metadata.h"

namespace xla::gpu {

// GPU-specific specification of an kernel produced during GPU codegen.
//
// Extends `KernelSpec` to encapsulate launch configurations and GPU metadata
// required by the GPU runtime and kernel compilers.
class GpuKernelSpec final : public KernelSpec {
 public:
  GpuKernelSpec(absl::string_view name,
                const LaunchDimensions& launch_dimensions,
                const emitters::KernelArguments& kernel_arguments,
                std::optional<size_t> scratch_bytes = std::nullopt,
                bool use_pdl = false,
                stream_executor::gpu::TmaMetadata tma_metadata = {});

  const LaunchDimensions& launch_dimensions() const {
    return launch_dimensions_;
  }

  const emitters::KernelArguments& kernel_arguments() const {
    return kernel_arguments_;
  }

  bool use_pdl() const { return use_pdl_; }

  const stream_executor::gpu::TmaMetadata& tma_metadata() const {
    return tma_metadata_;
  }

 private:
  LaunchDimensions launch_dimensions_;
  emitters::KernelArguments kernel_arguments_;
  bool use_pdl_;
  stream_executor::gpu::TmaMetadata tma_metadata_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_GPU_KERNEL_SPEC_H_
