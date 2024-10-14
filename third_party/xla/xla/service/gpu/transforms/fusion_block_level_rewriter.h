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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_FUSION_BLOCK_LEVEL_REWRITER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_FUSION_BLOCK_LEVEL_REWRITER_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class FusionBlockLevelRewriter : public HloModulePass {
 public:
  explicit FusionBlockLevelRewriter(
      const se::DeviceDescription& device_info,
      HloCostAnalysis::ShapeSizeFunction shape_size,
      absl::AnyInvocable<absl::StatusOr<bool>(const HloFusionInstruction*)>
          should_try_rewrite_if)
      : device_info_(device_info),
        shape_size_(shape_size),
        should_try_rewrite_if_(std::move(should_try_rewrite_if)) {}

  absl::string_view name() const override {
    return "fusion-block-level-rewriter";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const se::DeviceDescription& device_info_;
  HloCostAnalysis::ShapeSizeFunction shape_size_;
  absl::AnyInvocable<absl::StatusOr<bool>(const HloFusionInstruction*)>
      should_try_rewrite_if_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_FUSION_BLOCK_LEVEL_REWRITER_H_
