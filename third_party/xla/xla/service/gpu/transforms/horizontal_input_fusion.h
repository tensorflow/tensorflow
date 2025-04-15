/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_HORIZONTAL_INPUT_FUSION_H_
#define XLA_SERVICE_GPU_TRANSFORMS_HORIZONTAL_INPUT_FUSION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// This optimization pass horizontally fuses kInput fusions to both reduce the
// kernel launch overhead and increase parallelism degree. See
// HorizontalLoopFusion for general description and motivation about horizontal
// fusion. HorizontalLoopFusion deals with kLoop fusions while this pass deals
// with kInput fusions.
//
// Following HorizontalLoopFusion, a simple yet effective heuristic is used
// to search the fusion candidates while avoiding creating cycles. That is,
// we simply search for fusion candidates by looking for instructions whose
// outputs are all consumed by the same instruction. This catches the typical
// target cases; often, the candidate instructions are just consumed by the
// ROOT tuple of the entry computation.
class HorizontalInputFusion : public HloModulePass {
 public:
  explicit HorizontalInputFusion(const se::DeviceDescription& d)
      : device_info_(d) {}

  absl::string_view name() const override { return "horizontal_input_fusion"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<bool> RunOnComputation(HloComputation*);

  const se::DeviceDescription& device_info_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_HORIZONTAL_INPUT_FUSION_H_
