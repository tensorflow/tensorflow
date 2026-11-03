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

#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/gpu/tma_metadata.h"

namespace xla::gpu {
namespace {
absl::flat_hash_set<int64_t> GetInvariantArguments(
    const emitters::KernelArguments& kernel_arguments) {
  absl::flat_hash_set<int64_t> res;
  const std::vector<emitters::KernelArgument>& args = kernel_arguments.args();
  for (int i = 0; i < args.size(); i++) {
    if (args[i].invariant()) {
      res.insert(i);
    }
  }
  return res;
}
}  // namespace

GpuKernelSpec::GpuKernelSpec(absl::string_view name,
                             const LaunchDimensions& launch_dimensions,
                             const emitters::KernelArguments& kernel_arguments,
                             std::optional<size_t> scratch_bytes, bool use_pdl,
                             stream_executor::gpu::TmaMetadata tma_metadata)
    : KernelSpec(name, launch_dimensions.AsWorkDimensions(),
                 kernel_arguments.GetArgumentShapedSlices(), {},
                 GetInvariantArguments(kernel_arguments), scratch_bytes),
      launch_dimensions_(launch_dimensions),
      kernel_arguments_(kernel_arguments),
      use_pdl_(use_pdl),
      tma_metadata_(tma_metadata) {}

}  // namespace xla::gpu
