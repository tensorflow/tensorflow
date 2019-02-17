/* Copyright 2019 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/not_supported_gather_expander.h"

#include "tensorflow/compiler/xla/service/gather_expander.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include <map>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {
// We currently don't support any gathers - TODO T5742.
bool IsSupportedGather(const HloInstruction* gather_inst) { return false; }
}  // namespace

StatusOr<bool> NotSupportedGatherExpander::Run(HloModule* module) {
  VLOG(2) << "HLO module before NotSupportedGatherExpander:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  std::vector<HloInstruction*> not_supported_gather_insts;
  for (auto* comp : module->MakeNonfusionComputations()) {
    absl::c_copy_if(
        comp->instructions(), std::back_inserter(not_supported_gather_insts),
        [](const HloInstruction* inst) {
          return inst->opcode() == HloOpcode::kGather &&
                 // Avoid expanding gather ops that produce zero sized tensors,
                 // instead punt these to ZeroSizedHloElimination.
                 !ShapeUtil::IsZeroElementArray(inst->shape()) &&
                 !IsSupportedGather(inst);
        });
  }

  for (HloInstruction* gather_inst : not_supported_gather_insts) {
    VLOG(1) << "Expanding gather " << gather_inst->name();
    HloComputation* computation = gather_inst->parent();
    TF_ASSIGN_OR_RETURN(HloInstruction * expanded_root,
                        ExpandGather(gather_inst));

    // The ExpandGather returns an on in the following format:
    // Transpose(Reshape(GTE_index1(While())))
    CHECK_EQ(expanded_root->opcode(), HloOpcode::kTranspose);
    CHECK_EQ(expanded_root->operand(0)->opcode(), HloOpcode::kReshape);
    HloInstruction* reshape = expanded_root->mutable_operand(0);
    CHECK_EQ(reshape->operand(0)->opcode(), HloOpcode::kGetTupleElement);
    CHECK_EQ(reshape->operand(0)->tuple_index(), 3);
    HloInstruction* gte = reshape->mutable_operand(0);
    CHECK_EQ(gte->operand(0)->opcode(), HloOpcode::kWhile);
    HloInstruction* while_inst = gte->mutable_operand(0);

    if (gather_inst->has_sharding()) {
      // We need to copy the sharding info.
      // Copy onto the Transpose.
      expanded_root->set_sharding(gather_inst->sharding());
      // Copy onto the Reshape.
      reshape->set_sharding(gather_inst->sharding());
      // Copy onto the GTE.
      gte->set_sharding(gather_inst->sharding());
      // Copy onto the while loop.
      while_inst->set_sharding(gather_inst->sharding());
      for (auto* inst :
           while_inst->while_condition()->MakeInstructionPostOrder()) {
        inst->set_sharding(gather_inst->sharding());
      }
      for (auto* inst : while_inst->while_body()->MakeInstructionPostOrder()) {
        inst->set_sharding(gather_inst->sharding());
      }
    }
    TF_RETURN_IF_ERROR(
        computation->ReplaceInstruction(gather_inst, expanded_root));
  }

  if (changed) {
    VLOG(2) << "HLO module after NotSupportedGatherExpander:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "HLO module unchanged after NotSupportedGatherExpander";
  }

  return !not_supported_gather_insts.empty();
}

}  // namespace poplarplugin
}  // namespace xla
