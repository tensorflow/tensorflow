/* Copyright 2018 The OpenXLA Authors.

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

#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/while_util.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {
// Constant and Iota have no operands and an output and broadcasts add
// dimensions to the output so we are looking fusions that have much smaller
// operand sizes compared to output sizes to avoid materialization
bool IsPurelyExpanding(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kBroadcast ||
         (instr->opcode() == HloOpcode::kConstant &&
          instr->shape().dimensions_size() == 0) ||
         instr->opcode() == HloOpcode::kIota;
}

bool IsFusionCandidate(const HloInstruction* instr) {
  return instr->opcode() != HloOpcode::kRng &&
         (instr->IsElementwise() || instr->opcode() == HloOpcode::kReshape ||
          instr->opcode() == HloOpcode::kTranspose);
}

// For element-wise op 'instr' we have:
// forall index i in output shape: instr[i] = f(operand1[i], ...),  where
// f is the elementwise operation. We can see that all the indices of the output
// shape is written to.
bool IsShapeCoveringWriteOnlyInstruction(HloInstruction* instr) {
  // Clamp is tricky to handle, we bail.
  if (instr->opcode() == HloOpcode::kClamp) {
    return false;
  }
  return instr->IsElementwise();
}

// Updates the uses of the while loop with the equivalent tuple that retrieves
// the first original_operand_count elements of the while output.
absl::Status UpdateWhileUsesWithTuple(HloInstruction* while_instr,
                                      int64_t original_operand_count) {
  const std::vector<HloInstruction*> users = while_instr->users();
  std::vector<HloInstruction*> gtes(original_operand_count);
  for (int64_t i = 0; i < gtes.size(); ++i) {
    gtes[i] = while_instr->AddInstruction(
        HloInstruction::CreateGetTupleElement(while_instr, i));
  }
  HloInstruction* tuple =
      while_instr->AddInstruction(HloInstruction::CreateTuple(gtes));
  if (while_instr->IsRoot()) {
    while_instr->parent()->set_root_instruction(tuple);
  }
  if (!users.empty()) {
    TF_RETURN_IF_ERROR(while_instr->ReplaceUsesWith(users, tuple));
  }
  return absl::OkStatus();
}

// Appends the given new operand to while input and update loops computations
// and shape accordingly and returns the gte instruction within the body that
// represents the new operand.
absl::StatusOr<HloInstruction*> AppendToWhileState(
    HloInstruction* while_instr, HloInstruction* new_operand) {
  // Update the while initial value
  HloInstruction* while_input = while_instr->while_init();
  ShapeUtil::AppendShapeToTuple(new_operand->shape(),
                                while_input->mutable_shape());
  while_input->AppendOperand(new_operand);
  // Update the body computation.
  HloComputation* body = while_instr->while_body();
  *body->parameter_instruction(0)->mutable_shape() = while_input->shape();
  HloInstruction* new_gte =
      body->AddInstruction(HloInstruction::CreateGetTupleElement(
          body->parameter_instruction(0), while_input->operand_count() - 1));
  ShapeUtil::AppendShapeToTuple(new_gte->shape(),
                                body->root_instruction()->mutable_shape());
  body->root_instruction()->AppendOperand(new_gte);
  // Update the condition computation.
  HloComputation* condition = while_instr->while_condition();
  *condition->parameter_instruction(0)->mutable_shape() = while_input->shape();
  // Finalize the update by changing the uses of the while loop and updating its
  // shape.
  TF_RETURN_IF_ERROR(
      UpdateWhileUsesWithTuple(while_instr, while_input->operand_count() - 1));
  *while_instr->mutable_shape() = while_input->shape();
  return new_gte;
}

// Return the list of indices of the given while loop that are written to
// entirely in the loop body.
std::vector<int64_t> GetLoopShapeCoveringWriteIndices(
    HloInstruction* while_instr) {
  HloInstruction* tuple;
  if (!Match(while_instr->while_init(),
             match::Op(&tuple).WithOpcode(HloOpcode::kTuple).WithOneUse())) {
    return {};
  }

  std::vector<int64_t> loop_indices;
  for (int64_t tuple_index = 0; tuple_index < tuple->operand_count();
       ++tuple_index) {
    HloInstruction* arg_operand = tuple->mutable_operand(tuple_index);
    // We're looking for an argument that is a broadcast(constant) feeds a while
    // loop.
    if (!Match(arg_operand, match::Broadcast(match::ConstantScalar()))) {
      continue;
    }
    HloInstruction* broadcast_gte = hlo_query::GetUniqueGteInstruction(
        while_instr->while_body()->parameter_instruction(0), tuple_index);
    if (broadcast_gte == nullptr) {
      continue;
    }

    // If the buffer is not written to entirely, we won't sink it. We might be
    // able to support this case in the future, but for now we'll just skip it.
    HloInstruction* root_buffer_value =
        while_instr->while_body()->root_instruction()->mutable_operand(
            tuple_index);
    if (!IsShapeCoveringWriteOnlyInstruction(root_buffer_value)) {
      continue;
    }
    loop_indices.push_back(tuple_index);
  }

  return loop_indices;
}

// Returns true if the given instruction is monotonic, i.e. it is either
// monotonically increasing or decreasing. This is not an exhaustive list of
// monotonic operations.
bool IsMonotonic(HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kAdd ||
         instr->opcode() == HloOpcode::kSubtract;
}

// The idea is that certain constant-initialized buffers can be left as
// uninitialized if all the elements of the buffer are written to in the loop
// body. This way, we eliminate the need to initialize the buffer (with
// broadcast) in the critical path of the program. To summarize, the conditions
// to apply this optimization are:
// 1. The buffer is a constant-initialized buffer.
// 2. All the elements of the buffer are written to in the loop body.
// 3. The iteration variable of the loop is monotonically increasing or
// decreasing.
// The optimization is applied by creating a select between the initial value
// and the value in the body. The select is guarded by a predicate that checks
// if the loop iteration variable is equal to the first iteration value.
absl::StatusOr<bool> TryRewritingBroadcastAsAllocateBuffer(
    HloInstruction* while_instr) {
  std::optional<int64_t> induction_var_tuple_index =
      GetLoopInductionVarTupleIdx(while_instr);
  if (!induction_var_tuple_index.has_value() ||
      ComputeWhileLoopTripCount(while_instr).value_or(0) == 0) {
    return false;
  }
  HloComputation* while_body = while_instr->while_body();
  bool changed = false;
  std::vector<HloInstruction*> old_buffers;
  std::vector<int64_t> loop_indices =
      GetLoopShapeCoveringWriteIndices(while_instr);
  if (loop_indices.empty()) {
    return false;
  }
  HloInstruction* loop_iteration_variable_initial_value =
      while_instr->while_init()->mutable_operand(
          induction_var_tuple_index.value());
  // We only support integer loop iteration variables since these are the only
  // ones that can be compared to get the first iteration value.
  if (!ShapeUtil::ElementIsIntegral(
          loop_iteration_variable_initial_value->shape())) {
    return false;
  }

  // Also we have to make sure that the induction variable is either
  // monotonically increasing or decreasing since we rely on this fact to get
  // the first iteration value.
  HloInstruction* induction_var_update_fun =
      while_instr->while_body()->root_instruction()->mutable_operand(
          induction_var_tuple_index.value());
  if (!IsMonotonic(induction_var_update_fun)) {
    return false;
  }

  VLOG(3) << "Sinking fusible broadcast into " << while_instr->ToString();

  // If we find any sinkable indices, we prepare the loop state by adding the
  // initial value of the loop iteration variable to the loop state and use it
  // inside the body to create a predicate that checks if the loop iteration
  // variable is equal to the first iteration value. This is done only once
  // regardless of the number of sinkable indices.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * loop_iteration_variable_initial_value_gte,
      AppendToWhileState(while_instr, loop_iteration_variable_initial_value));
  HloInstruction* iteration_var_gte = hlo_query::GetUniqueGteInstruction(
      while_body->parameter_instruction(0), induction_var_tuple_index.value());
  if (iteration_var_gte == nullptr) {
    return false;
  }
  HloInstruction* is_first_iteration_pred =
      while_body->AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PRED, {}), iteration_var_gte,
          loop_iteration_variable_initial_value_gte,
          Comparison::Direction::kEq));
  for (int64_t loop_index : loop_indices) {
    HloInstruction* buffer =
        while_instr->while_init()->mutable_operand(loop_index);
    VLOG(3) << "Sinking " << buffer->ToString() << " at index " << loop_index;
    if (absl::c_find(old_buffers, buffer) == old_buffers.end()) {
      old_buffers.push_back(buffer);
    }
    // It is possible that the same broadcast has multiple users, first clone
    // the buffer and then replace this specific use with the clone.
    HloInstruction* buffer_clone = buffer->AddInstruction(buffer->Clone());
    TF_RETURN_IF_ERROR(while_instr->while_init()->ReplaceOperandWith(
        loop_index, buffer_clone));

    // Replace the clone with a free AllocateBuffer.
    HloInstruction* new_buffer =
        while_instr->parent()->AddInstruction(HloInstruction::CreateCustomCall(
            buffer_clone->shape(), {}, "AllocateBuffer"));
    TF_RETURN_IF_ERROR(buffer_clone->ReplaceAllUsesWith(new_buffer));
    TF_RETURN_IF_ERROR(buffer_clone->parent()->RemoveInstruction(buffer_clone));
    // Broadcast the predicate to the shape of the buffer.
    HloInstruction* is_first_iteration_pred_broadcast =
        while_body->AddInstruction(HloInstruction::CreateBroadcast(
            ShapeUtil::MakeShapeWithDescendingLayout(
                PRED, new_buffer->shape().dimensions()),
            is_first_iteration_pred, {}));
    HloInstruction* sunk_constant_broadcast =
        while_body->AddInstruction(HloInstruction::CreateBroadcast(
            new_buffer->shape(),
            while_body->AddInstruction(buffer->mutable_operand(0)->Clone()),
            {}));
    // Create a select between the initial broadcasted value (in the first
    // iteration of the loop) and the value in the body in the subsequent
    // iterations and replace the use of the buffer in the body with the select.
    HloInstruction* buffer_body_gte = hlo_query::GetUniqueGteInstruction(
        while_body->parameter_instruction(0), loop_index);
    HloInstruction* new_buffer_value =
        while_body->AddInstruction(HloInstruction::CreateTernary(
            new_buffer->shape(), HloOpcode::kSelect,
            is_first_iteration_pred_broadcast, sunk_constant_broadcast,
            buffer_body_gte));
    TF_RETURN_IF_ERROR(buffer_body_gte->ReplaceAllUsesWith(new_buffer_value));
    if (buffer->user_count() == 0) {
      TF_RETURN_IF_ERROR(buffer->parent()->RemoveInstruction(buffer));
    }
    changed = true;
  }
  return changed;
}
}  // namespace

bool WhileLoopFusibleSinking::IsSinkableFusion(HloInstruction* while_operand) {
  absl::InlinedVector<HloInstruction*, 8> worklist;
  absl::flat_hash_set<int> visited;
  worklist.push_back(while_operand);
  while (!worklist.empty()) {
    HloInstruction* to_process = worklist.back();
    worklist.pop_back();
    if (!to_process->IsFusible()) {
      return false;
    }
    if (!visited.insert(to_process->unique_id()).second) {
      // Do not sink extremely large subgraphs as they will be expensive to
      // recompute in the loop.
      if (visited.size() > 100) {
        return false;
      }
      continue;
    }
    if (IsPurelyExpanding(to_process)) {
      continue;
    }
    if (IsFusionCandidate(to_process)) {
      for (auto* op : to_process->operands()) {
        worklist.push_back(op);
      }
      continue;
    }
    return false;
  }
  return true;
}

HloInstruction* WhileLoopFusibleSinking::CreateSinkableFusion(
    HloInstruction* while_operand) {
  HloInstruction* fusion =
      while_operand->AddInstruction(while_operand->CreateFusion(
          while_operand->shape(), HloInstruction::FusionKind::kLoop,
          while_operand));
  bool did_fuse = IsFusionCandidate(while_operand);
  // Fuse up to broadcasts, this function expects that IsSinkableFusion is true
  // and does not verify that
  while (did_fuse) {
    did_fuse = false;
    for (int64_t i = fusion->operand_count() - 1; i >= 0; --i) {
      HloInstruction* op = fusion->mutable_operand(i);
      if (IsPurelyExpanding(op)) {
        continue;
      }
      fusion->FuseInstruction(op);
      did_fuse = true;
      break;
    }
  }
  // Fuse the broadcasts, constants and iota at the terminals.
  did_fuse = true;
  while (did_fuse) {
    did_fuse = false;
    for (int64_t i = fusion->operand_count() - 1; i >= 0; --i) {
      HloInstruction* op = fusion->mutable_operand(i);
      if (IsPurelyExpanding(op)) {
        fusion->FuseInstruction(op);
        did_fuse = true;
        break;
      }
    }
  }
  return fusion;
}

absl::StatusOr<bool> WhileLoopFusibleSinking::TrySinkingFusiblesIntoWhileLoop(
    HloInstruction* while_instr) {
  HloComputation* while_cond = while_instr->while_condition();
  HloComputation* while_body = while_instr->while_body();

  // Don't try to mutate unflattened while loop computations.
  if (call_counts_[while_body] > 1 || call_counts_[while_cond] > 1) {
    return false;
  }
  HloInstruction* init_value = while_instr->mutable_operand(0);
  if (init_value->opcode() != HloOpcode::kTuple) {
    return false;
  }

  bool changed = false;
  std::vector<HloInstruction*> invariant_body_gtes =
      WhileUtil::GetInvariantGTEsForWhileBody(*while_body);
  std::vector<HloInstruction*> new_operands;

  for (HloInstruction* invariant_body_gte : invariant_body_gtes) {
    int64_t index = invariant_body_gte->tuple_index();
    if (while_instr->operand_count() == 0 || init_value->operand_count() == 0) {
      // This is the case when each of tuple elements in the operand tuple of
      // the while loop was an invariant value and each of the usages has been
      // replaced.
      CHECK_EQ(while_instr->user_count(), 0);
      VLOG(3) << "Each element in the operand tuple of the while instruction '"
              << while_instr->name()
              << "' was an invariant value, whose usage has been replaced "
                 " directly by the value.";
      break;
    }

    HloInstruction* invariant_value = init_value->mutable_operand(index);

    // If a while operand is used by a slicing instruction, avoid fusing
    // invariant value into the loop.
    if (absl::c_any_of(invariant_body_gte->users(),
                       [](const HloInstruction* use) {
                         switch (use->opcode()) {
                           case HloOpcode::kDynamicSlice:
                           case HloOpcode::kGather:
                           case HloOpcode::kSlice:
                             return true;
                           default:
                             return false;
                         }
                       })) {
      continue;
    }

    if (init_value->IsRoot() || init_value->user_count() > 1) {
      init_value = init_value->AddInstruction(init_value->Clone());
      TF_RETURN_IF_ERROR(while_instr->ReplaceOperandWith(0, init_value));
    }
    // Original value should be a fusible subgraph.
    if (!IsSinkableFusion(invariant_value)) {
      continue;
    }
    HloInstruction* fusion = CreateSinkableFusion(invariant_value);
    changed = true;
    if (fusion->operand_count() > 0 &&
        (while_instr->IsRoot() ||
         absl::c_any_of(while_instr->users(), [&](HloInstruction* use) {
           return use->opcode() != HloOpcode::kGetTupleElement;
         }))) {
      // This really only occurs in unit tests or toy programs. Copy the current
      // users for later replacement.
      auto uses = while_instr->users();
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

    absl::InlinedVector<HloInstruction*, 2> invariant_output_uses;
    for (auto use : while_instr->users()) {
      if (use->opcode() == HloOpcode::kGetTupleElement &&
          use->tuple_index() == index) {
        invariant_output_uses.push_back(use);
      }
    }
    for (auto use : invariant_output_uses) {
      TF_RETURN_IF_ERROR(
          while_instr->parent()->ReplaceInstruction(use, invariant_value));
    }

    HloInstruction* root = while_body->root_instruction();
    HloInstruction* parameter = while_body->parameter_instruction(0);
    int64_t next_index = init_value->operand_count();
    new_operands.resize(fusion->operand_count());
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

absl::StatusOr<bool> WhileLoopFusibleSinking::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(5) << "Before WhileLoopFusibleSinking " << module->unique_id();
  XLA_VLOG_LINES(5, module->ToString());
  call_counts_.clear();
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
    call_counts_[while_instr->while_body()]++;
    call_counts_[while_instr->while_condition()]++;
  }

  for (HloInstruction* while_instr : while_instrs) {
    TF_ASSIGN_OR_RETURN(bool result,
                        TrySinkingFusiblesIntoWhileLoop(while_instr));
    changed |= result;
  }

  if (sink_broadcast_of_constant_) {
    for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
      for (HloInstruction* instr : comp->instructions()) {
        // TODO: b/358837872 - Handle loops with sharding.
        if (Match(instr, match::While()) && !instr->has_sharding()) {
          TF_ASSIGN_OR_RETURN(bool result,
                              TryRewritingBroadcastAsAllocateBuffer(instr));
          changed |= result;
        }
      }
    }
  }
  return changed;
}
}  // namespace xla
