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

#ifndef XLA_SERVICE_CPU_CPU_INSTRUCTION_FUSION_H_
#define XLA_SERVICE_CPU_CPU_INSTRUCTION_FUSION_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/fusion_node_indexing_evaluation.h"
#include "xla/service/instruction_fusion.h"

namespace xla {
namespace cpu {

class CpuInstructionFusion : public InstructionFusion {
 public:
  CpuInstructionFusion()
      : InstructionFusion(CpuInstructionFusion::IsExpensive) {}
  ~CpuInstructionFusion() override = default;

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(HloModule* module,
                           const absl::flat_hash_set<absl::string_view>&
                               execution_threads) override {
    fusion_node_evaluations_.clear();
    return InstructionFusion::Run(module, execution_threads);
  }

  // Returns the threshold for a constant to be considered a large constant.
  static constexpr int64_t GetLargeConstantThresholdBytes() {
    constexpr int64_t kLargeConstantThresholdBytes = 10000;
    return kLargeConstantThresholdBytes;
  }

 protected:
  FusionDecision ShouldFuse(HloInstruction* consumer,
                            int64_t operand_index) override;
  HloInstruction::FusionKind ChooseKind(
      const HloInstruction* producer, const HloInstruction* consumer) override;

 private:
  HloInstruction* FuseInstruction(HloInstruction* fusion_instruction,
                                  HloInstruction* producer) override;

  // Returns if a constant is large enough to be considered a large constant.
  bool IsLargeConstant(const HloInstruction* constant) const;

  // Keep track of the number of times each instruction inside a fusion node is
  // indexed with different index vectors.
  absl::flat_hash_map<const HloInstruction*, FusionNodeIndexingEvaluation>
      fusion_node_evaluations_;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_CPU_INSTRUCTION_FUSION_H_
