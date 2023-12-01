/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/while_loop_fusible_sinking.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/while_util.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {

HloInstruction* WhileLoopFusibleSinking::GetSinkableFusion(
    HloInstruction* while_operand) {
  std::vector<HloInstruction*> worklist;
  worklist.push_back(while_operand);
  HloInstruction* fusion = nullptr;
  auto fuse = [&](HloInstruction* instr) -> bool {
    if (!instr->IsFusible()) {
      return false;
    }
    if (!fusion) {
      fusion = instr->AddInstruction(instr->CreateFusion(
          instr->shape(), HloInstruction::FusionKind::kLoop, instr));
      return true;
    }
    // The instruction has already been visited, just skip it.
    if (!fusion->IsUserOf(instr)) {
      return false;
    }
    fusion->FuseInstruction(instr);
    return true;
  };
  std::vector<HloInstruction*> new_operands;
  while (!worklist.empty()) {
    HloInstruction* to_process = worklist.back();
    worklist.pop_back();
    if (to_process->IsElementwise() && fuse(to_process)) {
      for (auto* op : to_process->operands()) {
        worklist.push_back(op);
      }
      continue;
    }
    switch (to_process->opcode()) {
      case HloOpcode::kBroadcast: {
        HloInstruction* op = to_process->mutable_operand(0);
        if (fuse(to_process) && (op->opcode() == HloOpcode::kConstant ||
                                 op->opcode() == HloOpcode::kIota)) {
          fuse(op);
        }
        break;
      }
      case HloOpcode::kConstant:
      case HloOpcode::kIota: {
        fuse(to_process);
        break;
      }
      case HloOpcode::kReshape:
      case HloOpcode::kTranspose: {
        HloInstruction* op = to_process->mutable_operand(0);
        if (fuse(to_process)) {
          worklist.push_back(op);
        }
        break;
      }
      default:
        if (fusion) {
          fusion->parent()->RemoveInstruction(fusion).IgnoreError();
        }
        return nullptr;
    }
  }
  return fusion;
}

StatusOr<bool> WhileLoopFusibleSinking::TrySinkingFusiblesIntoWhileLoop(
    HloInstruction* while_instr) {
  HloComputation* while_cond = while_instr->while_condition();
  HloComputation* while_body = while_instr->while_body();

  // Don't try to mutate unflattened while loop computations.
  if (call_graph_->GetNode(while_cond).callers().size() > 1 ||
      call_graph_->GetNode(while_body).callers().size() > 1) {
    return false;
  }
  HloInstruction* init_value = while_instr->mutable_operand(0);
  if (init_value->opcode() != HloOpcode::kTuple) {
    return false;
  }

  bool changed = false;

  absl::flat_hash_map<int64_t, absl::InlinedVector<HloInstruction*, 1>>
      conditional_gte_index_to_insts =
          WhileUtil::GetGTEsMapForWhileConditional(*while_cond);
  std::vector<HloInstruction*> invariant_body_gtes =
      WhileUtil::GetInvariantGTEsForWhileBody(*while_body);

  for (HloInstruction* invariant_body_gte : invariant_body_gtes) {
    int64_t index = invariant_body_gte->tuple_index();
    HloInstruction* invariant_value = init_value->mutable_operand(index);

    if (init_value->IsRoot() || init_value->user_count() > 1) {
      init_value = init_value->AddInstruction(init_value->Clone());
      TF_RETURN_IF_ERROR(while_instr->ReplaceOperandWith(0, init_value));
    }
    // Original value should be a fusible subgraph.
    HloInstruction* fusion = GetSinkableFusion(invariant_value);
    if (fusion == nullptr) {
      continue;
    }
    changed = true;
    auto uses = while_instr->users();
    if (fusion->operand_count() > 0 &&
        (while_instr->IsRoot() ||
         absl::c_any_of(uses, [&](HloInstruction* use) {
           return use->opcode() != HloOpcode::kGetTupleElement;
         }))) {
      std::vector<HloInstruction*> gtes(init_value->operand_count());
      for (int64_t i = 0; i < gtes.size(); ++i) {
        gtes[i] = while_instr->AddInstruction(
            HloInstruction::CreateGetTupleElement(while_instr, i));
      }
      HloInstruction* tuple =
          while_instr->AddInstruction(HloInstruction::CreateTuple(gtes));
      if (while_instr->IsRoot()) {
        while_instr->parent()->set_root_instruction(tuple);
      }
      if (!uses.empty()) {
        TF_RETURN_IF_ERROR(while_instr->ReplaceUsesWith(uses, tuple));
      }
    }
    for (auto use : while_instr->users()) {
      if (use->opcode() == HloOpcode::kGetTupleElement &&
          use->tuple_index() == index) {
        TF_RETURN_IF_ERROR(
            while_instr->parent()->ReplaceInstruction(use, invariant_value));
      }
    }

    HloInstruction* root = while_body->root_instruction();
    HloInstruction* parameter = while_body->parameter_instruction(0);
    std::vector<int64_t> tuple_indices(fusion->operand_count());
    int64_t next_index = init_value->operand_count();
    std::vector<HloInstruction*> new_operands(fusion->operand_count());
    for (int64_t i = 0; i < fusion->operand_count(); ++i) {
      init_value->AppendOperand(fusion->mutable_operand(i));
      parameter->mutable_shape()->mutable_tuple_shapes()->push_back(
          fusion->mutable_operand(i)->shape());
      new_operands[i] = root->AddInstruction(
          HloInstruction::CreateGetTupleElement(parameter, next_index++));
      root->AppendOperand(new_operands[i]);
    }
    *(init_value->mutable_shape()) = parameter->shape();
    *(while_instr->mutable_shape()) = parameter->shape();
    *(while_cond->parameter_instruction(0)->mutable_shape()) =
        parameter->shape();
    *(root->mutable_shape()) = parameter->shape();
    auto cloned_fusion = while_body->AddInstruction(
        fusion->CloneWithNewOperands(fusion->shape(), new_operands));
    TF_RETURN_IF_ERROR(fusion->parent()->RemoveInstruction(fusion));
    TF_RETURN_IF_ERROR(
        while_body->ReplaceInstruction(invariant_body_gte, cloned_fusion));
    TF_RETURN_IF_ERROR(cloned_fusion->Defuse());
  }

  return changed;
}

StatusOr<bool> WhileLoopFusibleSinking::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto call_graph = CallGraph::Build(module, execution_threads);
  call_graph_ = call_graph.get();
  bool changed = false;
  std::vector<HloInstruction*> while_instrs;
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    // Right now we don't particularly care about optimizing while-of-while
    // patterns.  If/When we do, we'll want to visit the outer while (while_0)
    // before we visit the inner while (while_1):
    //
    // while_1_body(state) {
    //   val = gte(state, 0) // Loop invariant
    //   use(val)
    // }
    //
    // while_0_body(state) {
    //   val = gte(state, 0) // Loop invariant
    //   while_1 = while(init=tuple(val, ...), body=while_1_body, ...)
    //   ...
    // }
    //
    // main {
    //   while_0 = while(init=(fusible, ...), body=while_0_body, ...)
    // }
    //
    // This will let us sink the fusible into the outer while first and then
    // into the inner while in a single run of this pass.
    absl::c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }

  for (HloInstruction* while_instr : while_instrs) {
    TF_ASSIGN_OR_RETURN(bool result,
                        TrySinkingFusiblesIntoWhileLoop(while_instr));
    changed |= result;
  }
  return changed;
}
}  // namespace xla
