/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/tools/hlo_control_flow_flattening.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/tuple_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// Create a constant (recursively for tuples) of the given shape and add it to
// the computation.
HloInstruction* CreateConstant(const Shape& shape,
                               HloComputation* computation) {
  if (shape.IsTuple()) {
    std::vector<HloInstruction*> tuple_arguments(shape.tuple_shapes_size());
    for (int index = 0; index < shape.tuple_shapes_size(); ++index) {
      tuple_arguments[index] =
          CreateConstant(shape.tuple_shapes(index), computation);
    }
    return computation->AddInstruction(
        HloInstruction::CreateTuple(tuple_arguments));
  } else {
    return computation->AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateFromShape(shape)));
  }
}

// Prints sub-expression rooted at inst for a given depth.
void PrintSubexpression(HloInstruction* inst, int depth) {
  if (depth == 0) {
    return;
  }
  for (auto* operand : inst->operands()) {
    PrintSubexpression(operand, depth - 1);
  }
  VLOG(2) << inst->ToString();
}

bool IsConstantScalarInt(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kConstant &&
         ShapeUtil::IsEffectiveScalar(inst->shape()) &&
         inst->shape().IsInteger();
}

bool IsNotContainedInLoop(const HloInstruction& while_hlo,
                          const CallGraph& call_graph) {
  const HloComputation* computation = while_hlo.parent();
  while (!computation->IsEntryComputation()) {
    auto& node = call_graph.GetNode(computation);
    CHECK_EQ(node.caller_callsites().size(), 1)
        << "The module is not flattened!";
    auto& callsite = node.caller_callsites()[0];
    if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
      // Another while loop has been found traversing up the call tree.
      return false;
    }
    computation = callsite.instruction()->parent();
  }
  // No calling while loops were found.
  return true;
}

}  // namespace

int GetLoopBound(const HloInstruction& while_hlo, const int default_loop_count,
                 const int max_loop_count) {
  HloInstruction* condition = while_hlo.while_condition()->root_instruction();
  if (condition->opcode() == HloOpcode::kCompare) {
    int64_t value = 0;
    Comparison::Direction cmp = condition->comparison_direction();
    if ((cmp == Comparison::Direction::kLt ||
         cmp == Comparison::Direction::kLe ||
         cmp == Comparison::Direction::kNe) &&
        IsConstantScalarInt(condition->operand(1))) {
      value = *condition->operand(1)->literal().GetFirstInteger();
    } else if ((cmp == Comparison::Direction::kGt ||
                cmp == Comparison::Direction::kGe ||
                cmp == Comparison::Direction::kNe) &&
               IsConstantScalarInt(condition->operand(0))) {
      value = *condition->operand(0)->literal().GetFirstInteger();
    }
    if (value > 0) {
      // Caps to a max loop count to avoid long execution times.
      return std::min(value, static_cast<int64_t>(max_loop_count));
    }
  }
  return default_loop_count;
}

int GetLoopBoundWithOuterLoopMax(const HloInstruction& while_hlo,
                                 const CallGraph& call_graph,
                                 const int default_loop_count,
                                 const int max_outer_loop_count,
                                 const int max_loop_count) {
  int loop_bound = GetLoopBound(while_hlo, default_loop_count, max_loop_count);
  if (loop_bound > max_outer_loop_count) {
    // First does the inexpensive loop bound check to avoid as many
    // expensive graph traversals in IsNotContainedInLoop as possible.
    if (IsNotContainedInLoop(while_hlo, call_graph)) {
      return max_outer_loop_count;
    }
  }
  return loop_bound;
}

absl::Status HloControlFlowFlattening::FlattenWhileLoop(
    HloInstruction* while_hlo, const CallGraph& call_graph) const {
  CHECK_EQ(while_hlo->opcode(), HloOpcode::kWhile);
  HloComputation* computation = while_hlo->parent();
  // Add a new induction variable.
  HloInstruction* initialization = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(0)));
  // Create a new while operand with the induction variable added.
  HloInstruction* old_tuple = while_hlo->mutable_operand(0);
  HloInstruction* new_tuple =
      TupleUtil::AppendSuffix(old_tuple, {initialization});
  int new_tuple_size = new_tuple->shape().tuple_shapes().size();
  TF_RETURN_IF_ERROR(while_hlo->ReplaceOperandWithDifferentShape(0, new_tuple));

  auto change_op_shape = [&](HloInstruction* instruction) {
    Shape* shape = instruction->mutable_shape();
    CHECK(shape->IsTuple());
    CHECK_EQ(shape->tuple_shapes().size(), new_tuple_size - 1);
    Shape* subshape = shape->add_tuple_shapes();
    return ShapeUtil::PopulateShape(S32, {}, subshape);
  };

  // Replace the given tuple-shaped instruction of size N in each of its
  // non-get-tuple-element users with a new tuple instruction which has the
  // first N - 1 elements.
  auto replace_non_gte_users =
      [](HloInstruction* new_tuple) -> absl::StatusOr<HloInstruction*> {
    CHECK(new_tuple->shape().IsTuple());
    HloInstruction* prefix = nullptr;
    std::vector<HloInstruction*> users(new_tuple->users());
    for (HloInstruction* user : users) {
      if (user->opcode() == HloOpcode::kGetTupleElement) {
        continue;
      }
      // Lazily extract the prefix on demand, reuse it as needed.
      if (prefix == nullptr) {
        prefix = TupleUtil::ExtractPrefix(
            new_tuple, new_tuple->shape().tuple_shapes_size() - 1);
      }
      TF_RETURN_IF_ERROR(new_tuple->ReplaceUseWithDifferentShape(user, prefix));
    }
    return prefix;
  };

  {
    // Add the new variable to the while loop condition.
    HloComputation* condition = while_hlo->while_condition();
    TF_RETURN_IF_ERROR(change_op_shape(condition->parameter_instruction(0)));
    TF_RETURN_IF_ERROR(
        replace_non_gte_users(condition->parameter_instruction(0)).status());
    if (VLOG_IS_ON(2)) {
      VLOG(2) << "Loop condition in " << while_hlo->parent()->name();
      PrintSubexpression(condition->root_instruction(), /*depth=*/3);
    }
    const int loop_bound = GetLoopBoundWithOuterLoopMax(
        *while_hlo, call_graph, while_execution_count_, max_outer_loop_count_,
        max_loop_count_);

    VLOG(1) << "loop_bound = " << loop_bound;

    HloInstruction* limit = condition->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(loop_bound)));
    Shape shape = initialization->shape();
    HloInstruction* induction_variable =
        condition->AddInstruction(HloInstruction::CreateGetTupleElement(
            shape, condition->parameter_instruction(0), new_tuple_size - 1));
    HloInstruction* compare =
        condition->AddInstruction(HloInstruction::CreateCompare(
            ShapeUtil::MakeShape(PRED, {}), induction_variable, limit,
            ComparisonDirection::kLt));
    TF_RETURN_IF_ERROR(
        condition->ReplaceInstruction(condition->root_instruction(), compare));
  }

  {
    // Add the new variable to the while loop body.
    HloComputation* body = while_hlo->while_body();
    TF_RETURN_IF_ERROR(change_op_shape(body->parameter_instruction(0)));
    TF_RETURN_IF_ERROR(
        replace_non_gte_users(body->parameter_instruction(0)).status());
    HloInstruction* old_root = body->root_instruction();
    Shape shape = initialization->shape();
    HloInstruction* induction_variable =
        body->AddInstruction(HloInstruction::CreateGetTupleElement(
            shape, body->parameter_instruction(0), new_tuple_size - 1));
    HloInstruction* increment = body->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(1)));
    induction_variable = body->AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, induction_variable, increment));
    HloInstruction* new_root =
        TupleUtil::AppendSuffix(old_root, {induction_variable});
    body->set_root_instruction(new_root, /*accept_different_shape=*/true);
  }

  // Snapshot the users of while hlo before we add new users.
  std::vector<HloInstruction*> while_users(while_hlo->users().begin(),
                                           while_hlo->users().end());

  // Take care of the users of this while loop.
  TF_RETURN_IF_ERROR(change_op_shape(while_hlo));
  TF_ASSIGN_OR_RETURN(HloInstruction * prefix,
                      replace_non_gte_users(while_hlo));

  // If the while loop had been the root of its computation, make the prefix new
  // root.
  if (while_hlo->parent()->root_instruction() == while_hlo) {
    // We need to set accept_different_shape=true to reset the root shape to the
    // original, because we have already changed the shape of the old root
    // (while).
    if (prefix == nullptr) {
      prefix = TupleUtil::ExtractPrefix(while_hlo, new_tuple_size - 1);
    }
    while_hlo->parent()->set_root_instruction(prefix,
                                              /*accept_different_shape=*/true);
  }

  return absl::OkStatus();
}

absl::Status HloControlFlowFlattening::RemoveInfeed(
    HloInstruction* infeed_hlo) const {
  CHECK_EQ(infeed_hlo->opcode(), HloOpcode::kInfeed);
  HloComputation* computation = infeed_hlo->parent();
  CHECK_EQ(infeed_hlo->shape().tuple_shapes_size(), 2);
  const Shape& infeed_shape = ShapeUtil::GetSubshape(infeed_hlo->shape(), {0});

  HloInstruction* custom_call = computation->AddInstruction(
      HloInstruction::CreateCustomCall(infeed_shape, {}, kNopCustomCallTarget));

  // Create a new tuple consisting of the constant and the token that was
  // originally the operand of infeed, and replace the infeed operation.
  auto new_tuple = HloInstruction::CreateTuple(
      {custom_call, infeed_hlo->mutable_operand(0)});
  TF_RETURN_IF_ERROR(
      computation->ReplaceWithNewInstruction(infeed_hlo, std::move(new_tuple)));
  custom_call->SetAndSanitizeName(infeed_hlo->name());

  return absl::OkStatus();
}

absl::StatusOr<std::pair<HloInstruction*, HloInstruction*>>
HloControlFlowFlattening::RemoveRecvAndRecvDone(
    HloInstruction* recv_done,
    absl::flat_hash_set<HloInstruction*>* additional_removed) const {
  CHECK_EQ(recv_done->opcode(), HloOpcode::kRecvDone);
  CHECK_EQ(recv_done->operand_count(), 1);
  HloInstruction* recv = recv_done->mutable_operand(0);
  CHECK_EQ(recv->opcode(), HloOpcode::kRecv);

  HloComputation* computation = recv_done->parent();
  CHECK_EQ(recv_done->shape().tuple_shapes_size(), 2);
  HloModule* module = computation->parent();

  HloInstruction* custom_call_recv =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          recv->shape(), recv->operands(), kNopCustomCallTarget));
  std::string original_recv_name(recv->name());
  if (module->has_schedule() &&
      module->schedule().is_computation_scheduled(computation)) {
    module->schedule().replace_instruction(computation, recv, custom_call_recv);
  }
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(recv, custom_call_recv));
  custom_call_recv->SetAndSanitizeName(original_recv_name);

  std::string original_recv_done_name(recv_done->name());
  HloInstruction* custom_call_recv_done = computation->AddInstruction(
      HloInstruction::CreateCustomCall(
          recv_done->shape(), recv_done->operands(), kNopCustomCallTarget),
      recv_done->name());
  if (module->has_schedule() &&
      module->schedule().is_computation_scheduled(computation)) {
    module->schedule().replace_instruction(computation, recv_done,
                                           custom_call_recv_done);
  }
  TF_RETURN_IF_ERROR(
      computation->ReplaceInstruction(recv_done, custom_call_recv_done));
  custom_call_recv_done->SetAndSanitizeName(original_recv_done_name);

  return std::make_pair(custom_call_recv, custom_call_recv_done);
}

absl::Status HloControlFlowFlattening::RemoveOutfeed(
    HloInstruction* outfeed_hlo) const {
  CHECK_EQ(outfeed_hlo->opcode(), HloOpcode::kOutfeed);
  HloComputation* computation = outfeed_hlo->parent();
  // Replace the outfeed with a no-op custom call with side effect to ensure the
  // operands aren't DCE'd.
  HloInstruction* custom_call =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          outfeed_hlo->shape(), outfeed_hlo->operands(),
          kNopReturnTokenCustomCallTarget));
  Cast<HloCustomCallInstruction>(custom_call)
      ->set_custom_call_has_side_effect(true);
  // For SPMD graphs, partitioner requires that side-effecting custom calls have
  // a sharding that is non-replicated.
  custom_call->set_sharding(HloSharding::Manual());
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(outfeed_hlo, custom_call));
  custom_call->SetAndSanitizeName(outfeed_hlo->name());

  return absl::OkStatus();
}

absl::StatusOr<std::pair<HloInstruction*, HloInstruction*>>
HloControlFlowFlattening::RemoveSendAndSendDone(
    HloInstruction* send_done,
    absl::flat_hash_set<HloInstruction*>* additional_removed) const {
  CHECK_EQ(send_done->opcode(), HloOpcode::kSendDone);
  CHECK_EQ(send_done->operand_count(), 1);
  HloInstruction* send = send_done->mutable_operand(0);
  CHECK_EQ(send->opcode(), HloOpcode::kSend);

  HloComputation* computation = send_done->parent();
  HloModule* module = computation->parent();

  HloInstruction* custom_call_send =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          send->shape(), send->operands(), kNopCustomCallTarget));
  std::string original_send_name(send->name());
  if (module->has_schedule() &&
      module->schedule().is_computation_scheduled(computation)) {
    module->schedule().replace_instruction(computation, send, custom_call_send);
  }
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(send, custom_call_send));
  custom_call_send->SetAndSanitizeName(original_send_name);

  HloInstruction* custom_call_send_done =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          send_done->shape(), send_done->operands(),
          kNopReturnTokenCustomCallTarget));
  std::string original_send_done_name(send_done->name());
  Cast<HloCustomCallInstruction>(custom_call_send_done)
      ->set_custom_call_has_side_effect(true);
  if (module->has_schedule() &&
      module->schedule().is_computation_scheduled(computation)) {
    module->schedule().replace_instruction(computation, send_done,
                                           custom_call_send_done);
  }
  TF_RETURN_IF_ERROR(
      computation->ReplaceInstruction(send_done, custom_call_send_done));
  custom_call_send_done->SetAndSanitizeName(original_send_done_name);

  return std::make_pair(custom_call_send, custom_call_send_done);
}

absl::StatusOr<HloInstruction*> HloControlFlowFlattening::RemoveCollective(
    HloInstruction* hlo) const {
  HloComputation* computation = hlo->parent();
  HloInstruction* custom_call =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          hlo->shape(), hlo->operands(), kNopCustomCallTarget));
  // Copy backend config. This is necessary for a collective op in megacore
  // fusion.
  custom_call->CopyBackendConfigFrom(hlo);
  HloModule* module = computation->parent();
  if (module->has_schedule() &&
      module->schedule().is_computation_scheduled(computation)) {
    module->schedule().replace_instruction(computation, hlo, custom_call);
  }
  std::string original_op_name(hlo->name());
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(hlo, custom_call));
  custom_call->SetAndSanitizeName(original_op_name);
  return custom_call;
}

absl::Status HloControlFlowFlattening::RemoveId(HloInstruction* hlo) const {
  HloComputation* computation = hlo->parent();
  HloInstruction* zero = CreateConstant(hlo->shape(), computation);
  std::string original_op_name(hlo->name());
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(hlo, zero));
  zero->SetAndSanitizeName(original_op_name);
  return absl::OkStatus();
}

absl::Status HloControlFlowFlattening::SetConditionalValue(
    HloInstruction* conditional) const {
  HloComputation* computation = conditional->parent();
  // This branch op is either a PRED or an index.
  HloInstruction* original_branch_op = conditional->mutable_operand(0);
  std::string original_op_name(original_branch_op->name());

  // If the original branch op has no other users, wrap in a custom call with
  // side effect to ensure the operands aren't DCE'd.
  if (original_branch_op->user_count() == 1) {
    HloInstruction* custom_call =
        computation->AddInstruction(HloInstruction::CreateCustomCall(
            original_branch_op->shape(), original_branch_op->operands(),
            kNopCustomCallTarget));
    Cast<HloCustomCallInstruction>(custom_call)
        ->set_custom_call_has_side_effect(true);
    // For SPMD graphs, partitioner requires that side-effecting custom calls
    // have a sharding that is non-replicated.
    custom_call->set_sharding(HloSharding::Manual());
  }

  HloInstruction* new_branch_op;
  if (original_branch_op->shape().element_type() == PRED) {
    // Predicated (if/else) conditional.
    new_branch_op = computation->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<bool>(conditional_value_)));
  } else {
    // Indexed (switch/case/default) conditional. Uses the N-1'th
    // branch_computation as default index.
    new_branch_op = computation->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32_t>(conditional->branch_count() - 1)));
  }
  new_branch_op->SetAndSanitizeName(original_op_name + "_flattened");
  TF_RETURN_IF_ERROR(conditional->ReplaceOperandWith(0, new_branch_op));

  return absl::OkStatus();
}

absl::StatusOr<bool> HloControlFlowFlattening::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto call_graph = CallGraph::Build(module);
  bool changed = false;
  absl::flat_hash_set<HloInstruction*> removed;
  for (HloComputation* computation : module->computations(execution_threads)) {
    // Do not change computations that are wrapped by async calls. Instead we
    // remove the async callers if needed.
    if (computation->IsAsyncComputation()) {
      continue;
    }
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (removed.contains(instruction)) {
        // Skip the instruction if it is already removed.
        continue;
      }
      if (flatten_while_loop_ && instruction->opcode() == HloOpcode::kWhile) {
        VLOG(1) << "Remove " << instruction->name();
        TF_RETURN_IF_ERROR(FlattenWhileLoop(instruction, *call_graph));
        changed = true;
      } else if (remove_infeed_outfeed_ &&
                 instruction->opcode() == HloOpcode::kInfeed) {
        VLOG(1) << "Remove " << instruction->name();
        TF_RETURN_IF_ERROR(RemoveInfeed(instruction));
        changed = true;
      } else if (remove_infeed_outfeed_ &&
                 instruction->opcode() == HloOpcode::kOutfeed) {
        VLOG(1) << "Remove " << instruction->name();
        TF_RETURN_IF_ERROR(RemoveOutfeed(instruction));
        changed = true;
      } else if (instruction->opcode() == HloOpcode::kSendDone) {
        auto send_done_instruction =
            DynCast<HloSendDoneInstruction>(instruction);
        CHECK(send_done_instruction);
        if (remove_comm_ || (remove_host_transfer_ &&
                             send_done_instruction->is_host_transfer())) {
          VLOG(1) << "Remove " << instruction->name();
          TF_RETURN_IF_ERROR(
              RemoveSendAndSendDone(instruction, &removed).status());
          changed = true;
        }
      } else if (instruction->opcode() == HloOpcode::kRecvDone) {
        auto recv_done_instruction =
            DynCast<HloRecvDoneInstruction>(instruction);
        CHECK(recv_done_instruction);
        if (remove_comm_ || (remove_host_transfer_ &&
                             recv_done_instruction->is_host_transfer())) {
          VLOG(1) << "Remove " << instruction->name();
          TF_RETURN_IF_ERROR(
              RemoveRecvAndRecvDone(instruction, &removed).status());
          changed = true;
        }
      } else if (remove_comm_ && IsCollective(instruction) &&
                 (instruction->opcode() != HloOpcode::kSend &&
                  instruction->opcode() != HloOpcode::kRecv) &&
                 !instruction->parent()->IsFusionComputation() &&
                 (instruction->opcode() != HloOpcode::kAsyncStart &&
                  instruction->opcode() != HloOpcode::kAsyncUpdate)) {
        // We do not remove kAsyncStart or kAsyncUpdate here since we expect
        // them to be removed as a part of the async chain above.
        // We should remove the async chain all together because the async
        // wrapped computation is only associated with the AsyncStart. So we
        // need to refer to the AsyncStart in order to determine whether
        // the Done or the Update wraps a collective.
        if (instruction->opcode() == HloOpcode::kAsyncDone) {
          while (instruction->opcode() == HloOpcode::kAsyncDone ||
                 instruction->opcode() == HloOpcode::kAsyncUpdate ||
                 instruction->opcode() == HloOpcode::kAsyncStart) {
            HloInstruction* operand = instruction->mutable_operand(0);
            VLOG(1) << "Remove " << instruction->name();
            TF_RETURN_IF_ERROR(RemoveCollective(instruction).status());
            instruction = operand;
          }
        } else {
          VLOG(1) << "Remove " << instruction->name();
          TF_RETURN_IF_ERROR(RemoveCollective(instruction).status());
        }
        changed = true;
      } else if ((remove_comm_ || remove_id_) &&
                 (instruction->opcode() == HloOpcode::kPartitionId ||
                  instruction->opcode() == HloOpcode::kReplicaId ||
                  (instruction->opcode() == HloOpcode::kCustomCall &&
                   instruction->custom_call_target() == "SliceId"))) {
        VLOG(1) << "Remove " << instruction->name();
        TF_RETURN_IF_ERROR(RemoveId(instruction));
        changed = true;
      } else if (flatten_conditional_ &&
                 instruction->opcode() == HloOpcode::kConditional) {
        TF_RETURN_IF_ERROR(SetConditionalValue(instruction));
        changed = true;
      }
    }
  }

  HloDCE hlo_dce;
  TF_ASSIGN_OR_RETURN(bool dce_changed, hlo_dce.Run(module, execution_threads));
  changed |= dce_changed;

  // Fix the schedule if the module was scheduled.
  if (changed && module->has_schedule()) {
    TF_RETURN_IF_ERROR(module->schedule().Update());
  }
  XLA_VLOG_LINES(3, module->ToString());
  return changed;
}

}  // namespace xla
