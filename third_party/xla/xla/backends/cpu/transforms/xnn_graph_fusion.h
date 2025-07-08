/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_TRANSFORMS_XNN_GRAPH_FUSION_H_
#define XLA_BACKENDS_CPU_TRANSFORMS_XNN_GRAPH_FUSION_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/instruction_fusion.h"

namespace xla {
namespace cpu {

class XnnGraphFusion : public InstructionFusion {
 public:
  XnnGraphFusion() : InstructionFusion(XnnGraphFusion::IsExpensive) {}
  ~XnnGraphFusion() override = default;

 private:
  FusionDecision ShouldFuse(HloInstruction* consumer,
                            int64_t operand_index) override;
  HloInstruction::FusionKind ChooseKind(
      const HloInstruction* producer, const HloInstruction* consumer) override;

  HloInstruction* Fuse(HloInstruction* producer, HloInstruction* consumer,
                       HloComputation* computation) override;

  std::vector<HloComputation*> GetNonFusionComputations(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  static bool IsOpSupported(const HloInstruction* instr);

  static bool IsXnnGraphFusion(const HloInstruction* instr);
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_BACKENDS_CPU_TRANSFORMS_XNN_GRAPH_FUSION_H_
