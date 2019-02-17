/* Copyright 2018 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/not_supported_scatter_expander.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/scatter_expander.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include <map>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {
// We currently don't support any scatters - TODO T5743.
bool IsSupportedScatter(const HloInstruction* scatter_inst) { return false; }
}  // namespace

StatusOr<bool> NotSupportedScatterExpander::Run(HloModule* module) {
  VLOG(2) << "HLO module before NotSupportedScatterExpander:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  std::vector<HloInstruction*> not_supported_scatter_insts;
  for (auto* comp : module->MakeNonfusionComputations()) {
    absl::c_copy_if(comp->instructions(),
                    std::back_inserter(not_supported_scatter_insts),
                    [](const HloInstruction* instr) {
                      return instr->opcode() == HloOpcode::kScatter &&
                             !IsSupportedScatter(instr);
                    });
  }

  for (HloInstruction* scatter_inst : not_supported_scatter_insts) {
    VLOG(1) << "Expanding scatter " << scatter_inst->name();
    HloComputation* computation = scatter_inst->parent();
    const HloInstruction* operand_inst = scatter_inst->operand(0);
    const HloInstruction* updates_inst = scatter_inst->operand(2);

    TF_ASSIGN_OR_RETURN(HloInstruction * expanded_root,
                        ExpandScatter(scatter_inst));
    // The expand scatter returns either:
    // * operand tensor if the updates is a zero element array
    // * while loop to the expanded computation

    if (ShapeUtil::IsZeroElementArray(updates_inst->shape())) {
      CHECK_EQ(expanded_root, operand_inst);
    } else {
      // We need to copy the sharding for the while loop expansion.
      CHECK_EQ(expanded_root->opcode(), HloOpcode::kGetTupleElement);
      if (scatter_inst->has_sharding()) {
        expanded_root->set_sharding(scatter_inst->sharding());
        HloInstruction* while_inst = expanded_root->mutable_operand(0);
        CHECK_EQ(while_inst->opcode(), HloOpcode::kWhile);
        while_inst->set_sharding(scatter_inst->sharding());
        for (auto* inst :
             while_inst->while_condition()->MakeInstructionPostOrder()) {
          inst->set_sharding(scatter_inst->sharding());
        }
        for (auto* inst :
             while_inst->while_body()->MakeInstructionPostOrder()) {
          inst->set_sharding(scatter_inst->sharding());
        }
      }
    }
    TF_RETURN_IF_ERROR(
        computation->ReplaceInstruction(scatter_inst, expanded_root));
  }

  if (changed) {
    VLOG(2) << "HLO module after NotSupportedScatterExpander:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "HLO module unchanged after NotSupportedScatterExpander";
  }

  return !not_supported_scatter_insts.empty();
}

}  // namespace poplarplugin
}  // namespace xla
