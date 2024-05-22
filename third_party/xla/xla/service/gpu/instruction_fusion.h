/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_INSTRUCTION_FUSION_H_
#define XLA_SERVICE_GPU_INSTRUCTION_FUSION_H_

#include <stdint.h>

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/fusion_node_indexing_evaluation.h"
#include "xla/service/fusion_queue.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class GpuInstructionFusion : public InstructionFusion {
 public:
  GpuInstructionFusion(bool may_duplicate, const se::DeviceDescription& d)
      : InstructionFusion(GpuInstructionFusion::IsExpensive, may_duplicate),
        device_info_(d) {}

  static bool IsExpensive(const HloInstruction& instruction);

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 protected:
  std::unique_ptr<FusionQueue> GetFusionQueue(
      HloComputation* computation) override;
  FusionDecision ShouldFuse(HloInstruction* consumer,
                            int64_t operand_index) override;

  HloInstruction::FusionKind ChooseKind(
      const HloInstruction* producer, const HloInstruction* consumer) override;

 private:
  // This method is called by ShouldFuse() to do all the computationally
  // inexpensive checks whether we should fuse the operand into 'consumer'.
  FusionDecision ShouldFuseInexpensiveChecks(HloInstruction* consumer,
                                             int64_t operand_index);

  HloInstruction* FuseInstruction(HloInstruction* fusion_instruction,
                                  HloInstruction* producer) override;

  // Keep track of the number of times each instruction inside a fusion node is
  // indexed with different index vectors.
  absl::flat_hash_set<const HloComputation*> fusible_computations_;
  absl::flat_hash_map<const HloInstruction*, FusionNodeIndexingEvaluation>
      fusion_node_evaluations_;

  se::DeviceDescription device_info_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_INSTRUCTION_FUSION_H_
