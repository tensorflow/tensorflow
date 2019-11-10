/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/defuser.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

namespace {

// Copy all the instructions in the given fusion instruction into the fusion
// instruction's parent computation and replace the use of the fusion
// instruction with the copy of the fusion expression root.
Status Defuse(HloInstruction* fusion_instruction) {
  VLOG(2) << "Defusing instruction: " << fusion_instruction->ToString();

  HloComputation* fused_computation =
      fusion_instruction->fused_instructions_computation();

  // A map from fused instruction to its defused clone.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      defused_instructions;
  // Initialize map to contain the fusion instruction parameters mapping
  // to the operands of the fusion instruction.
  for (int64 i = 0; i < fusion_instruction->operand_count(); ++i) {
    defused_instructions[fused_computation->parameter_instruction(i)] =
        fusion_instruction->mutable_operand(i);
  }

  // Create a clone of each instruction of the fused computation in the same
  // computation as the fusion instruction itself.
  // TODO(b/68227302): Moving instruction to new computation rather than
  // cloning and deleting.
  for (HloInstruction* fused_instruction :
       fused_computation->MakeInstructionPostOrder()) {
    if (fused_instruction->opcode() == HloOpcode::kParameter) {
      continue;
    }
    std::vector<HloInstruction*> new_operands;
    for (HloInstruction* operand : fused_instruction->operands()) {
      new_operands.push_back(defused_instructions.at(operand));
    }
    HloInstruction* defused_instruction =
        fusion_instruction->parent()->AddInstruction(
            fused_instruction->CloneWithNewOperands(fused_instruction->shape(),
                                                    new_operands));
    defused_instructions[fused_instruction] = defused_instruction;
  }

  TF_RETURN_IF_ERROR(fusion_instruction->ReplaceAllUsesWith(
      defused_instructions.at(fusion_instruction->fused_expression_root())));

  HloModule* module = fusion_instruction->parent()->parent();
  TF_RETURN_IF_ERROR(
      fusion_instruction->parent()->RemoveInstruction(fusion_instruction));
  return module->RemoveEmbeddedComputation(fused_computation);
}

}  // namespace

StatusOr<bool> Defuser::Run(HloModule* module) {
  VLOG(1) << "Defusing module " << module->name();
  XLA_VLOG_LINES(2, "Before defusion:\n" + module->ToString());

  bool changed = false;
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  TF_RETURN_IF_ERROR(call_graph->VisitNodes(
      [&](const CallGraphNode& call_graph_node) -> Status {
        if (call_graph_node.computation()->IsFusionComputation()) {
          TF_RET_CHECK(call_graph_node.caller_callsites().size() == 1);
          HloInstruction* fusion_instruction =
              call_graph_node.caller_callsites()[0].instruction();
          TF_RETURN_IF_ERROR(Defuse(fusion_instruction));
          changed = true;
        }
        return Status::OK();
      },
      /*visit_unreachable_nodes=*/true));

  XLA_VLOG_LINES(2, "After defusion:\n" + module->ToString());

  return changed;
}

}  // namespace xla
