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

#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/core/lib/gtl/optional.h"

namespace xla {

using tensorflow::gtl::nullopt;
using tensorflow::gtl::optional;

// Finds and returns the non-constant operand in instr.
//
// CHECK-fails if instr doesn't have exactly one unique non-constant operand.
static const HloInstruction* NonConstantOperand(const HloInstruction* instr) {
  const HloInstruction* result = nullptr;
  for (const HloInstruction* operand : instr->operands()) {
    if (!operand->IsConstant()) {
      if (result != nullptr) {
        CHECK_EQ(result, operand);
      }
      result = operand;
    }
  }
  CHECK_NE(result, nullptr);
  return result;
}

// Determines whether the given instruction is a send/recv node, or has a
// subcomputation which contains a send/recv node.
static bool IsOrContainsSendOrRecv(const HloInstruction* instr);

// Determines whether the given computation contains a send or recv node.
static bool ContainsSendOrRecv(const HloComputation* comp) {
  for (const auto* instr : comp->instructions()) {
    if (IsOrContainsSendOrRecv(instr)) {
      return true;
    }
  }
  return false;
}

static bool IsOrContainsSendOrRecv(const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kSend ||
      instr->opcode() == HloOpcode::kRecv) {
    return true;
  }
  for (const auto& subcomp : instr->called_computations()) {
    if (ContainsSendOrRecv(subcomp)) {
      return true;
    }
  }
  return false;
}

// If all of instr's operands are either constants or have the form
//   get-tuple-element(gte_operand, N)
// for the same value N, returns N.  Otherwise, returns nullopt.
static optional<int64> GetGTEOperandIndex(const HloInstruction* instr,
                                          const HloInstruction* gte_operand) {
  VLOG(2) << "GetGTEOperandIndex(" << instr->ToString() << ", "
          << gte_operand->ToString() << ")";
  optional<int64> tuple_idx;
  for (const HloInstruction* operand : instr->operands()) {
    if (operand->IsConstant()) {
      continue;
    }
    if (operand->opcode() != HloOpcode::kGetTupleElement) {
      VLOG(2) << "instr uses something other than gte(gte_operand): "
              << operand->ToString();
      return nullopt;
    }
    if (operand->operand(0) != gte_operand) {
      VLOG(2) << "instr has gte whose operand is not gte_operand: "
              << operand->ToString();
      return nullopt;
    }
    if (tuple_idx && tuple_idx != operand->tuple_index()) {
      VLOG(2) << "instr has operands with conflicting gte indices, "
              << *tuple_idx << " vs " << operand->tuple_index();
      return nullopt;
    }

    tuple_idx = operand->tuple_index();
  }
  return tuple_idx;
}

// Tries to get the tuple index of the induction variable of a while loop.
//
// Checks that the loop condition and root both plumb the induction variable
// through the same tuple index, and that they both apply exactly one op to the
// induction variable before  deciding whether to do another loop iteration (in
// the loop condition's case) or packing the induction variable into the result
// tuple (in the loop body's case).
//
// Specifically, checks that the loop condition has structure
//
//   root = op(constants, get-tuple-elem(param0, N), constants)
//
// and the loop body has the structure
//
//   inc = op(constants, get-tuple-elem(param0, N), constants)
//   root = tuple(..., inc, ...)  // inc is N'th operand of tuple().
//
// If so, returns N.  Otherwise, returns nullopt.
static optional<int64> GetLoopInductionVarTupleIdx(
    const HloInstruction* while_op) {
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);
  VLOG(2) << "Finding induction variable for loop "
          << while_op->ToShortString();

  // The while_cond computation should have the form
  //
  //   while_cond_root =
  //       op(constants, get-tuple-elem(while_cond_param, N), constants).
  //
  // If it does, set indvar_tuple_idx to N.
  auto* while_cond = while_op->while_condition();
  auto* while_cond_root = while_cond->root_instruction();
  auto* while_cond_param = while_cond->parameter_instruction(0);
  optional<int64> indvar_tuple_idx =
      GetGTEOperandIndex(while_cond_root, while_cond_param);
  if (!indvar_tuple_idx) {
    VLOG(2) << "Induction variable not found in loop condition: "
            << while_cond->root_instruction()->ToString();
    return nullopt;
  }

  // The while_body computation should have the form
  //
  //   while_body_inc =
  //       op(constants, get-tuple-elem(while_body_param, N), constants)
  //   while_body_root = tuple(..., while_body_inc, ...)
  //
  // where while_body_inc is operand N of while_body_root.
  auto* while_body = while_op->while_body();
  auto* while_body_root = while_body->root_instruction();
  if (while_body_root->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "While body's root is not a tuple instruction: "
            << while_body_root->ToString();
    return nullopt;
  }

  auto* while_body_inc = while_body_root->operand(*indvar_tuple_idx);
  auto* while_body_param = while_body->parameter_instruction(0);
  optional<int64> while_body_indvar_tuple_idx =
      GetGTEOperandIndex(while_body_inc, while_body_param);
  if (!while_body_indvar_tuple_idx) {
    VLOG(2)
        << "Induction variable not found in while body increment instruction: "
        << while_body_inc->ToString();
    return nullopt;
  }
  if (while_body_indvar_tuple_idx != indvar_tuple_idx) {
    VLOG(2) << "Tuple index of induction variable does not match between loop "
               "condition ("
            << *indvar_tuple_idx << ") and while body ("
            << *while_body_indvar_tuple_idx << ")";
    return nullopt;
  }

  // Finally, check that the while loop's initial value is a tuple with enough
  // elements.
  auto* while_init = while_op->operand(0);
  if (while_init->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "While init expected to be a tuple: " << while_init->ToString();
    return nullopt;
  }

  VLOG(2) << "Induction variable's tuple index: " << *indvar_tuple_idx;
  return indvar_tuple_idx;
}

// Tries to determine the number of times the given loop executes.  Currently
// simply returns 0, 1, or "can't tell" (nullopt).
static optional<int64> GetLoopTripCount(HloInstruction* while_op) {
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);
  VLOG(2) << "Getting trip count for loop " << while_op->ToString();

  // The loop's induction variable is found at
  //
  //   get-tuple-elem(comp->parameter_instruction(0), *indvar_tuple_idx),
  //
  // where comp is while_op->while_body() or while_op->while_condition().
  optional<int64> indvar_tuple_idx = GetLoopInductionVarTupleIdx(while_op);
  if (!indvar_tuple_idx) {
    return nullopt;
  }

  VLOG(2) << "Induction variable is at index " << *indvar_tuple_idx
          << " in input tuple.";

  // Now that we know the index of the induction variable, we can we can try to
  // compute how many times the loop executes.  Start by computing the induction
  // variable's initial value.
  HloEvaluator evaluator;
  auto* while_init = while_op->mutable_operand(0);
  auto* indvar_init = while_init->mutable_operand(*indvar_tuple_idx);
  StatusOr<std::unique_ptr<Literal>> indvar_init_result =
      evaluator.Evaluate(indvar_init);
  if (!indvar_init_result.ok()) {
    VLOG(2) << "Couldn't evaluate induction variable init: "
            << indvar_init_result.status();
    return nullopt;
  }

  // Evaluates the while loop's condition, returning either "true" (continue
  // looping), "false" (stop looping), or nullopt (can't evaluate).
  auto evaluate_while_cond = [&](const Literal& indvar) -> optional<bool> {
    auto* while_cond = while_op->while_condition();
    auto* while_cond_root = while_cond->root_instruction();
    auto* while_cond_indvar = NonConstantOperand(while_cond_root);
    StatusOr<std::unique_ptr<Literal>> result =
        evaluator.EvaluateWithSubstitutions(while_cond_root,
                                            {{while_cond_indvar, &indvar}});
    if (!result.ok()) {
      VLOG(2) << "Couldn't evaluate while cond: " << result.status();
      return nullopt;
    }
    return result.ValueOrDie()->GetArraySlice<bool>() ==
           tensorflow::gtl::ArraySlice<bool>{true};
  };

  // The initial value of the induction variable.
  const Literal& indvar_iter0_val = *indvar_init_result.ValueOrDie();

  // Evaluate whether the while condition is true when seeded with
  // indvar_iter0_val.
  optional<bool> while_cond_iter0_val = evaluate_while_cond(indvar_iter0_val);
  if (while_cond_iter0_val == false) {
    VLOG(2) << "Loop has static trip count of 0.";
    return 0;
  }

  // Calculate the value of the induction variable after one iteration of the
  // loop, and check whether the while condition is true with this new value.
  auto* while_body = while_op->while_body();
  auto* while_body_indvar_update =
      while_body->root_instruction()->operand(*indvar_tuple_idx);
  auto* while_body_indvar = NonConstantOperand(while_body_indvar_update);
  StatusOr<std::unique_ptr<Literal>> indvar_iter1_result =
      evaluator.EvaluateWithSubstitutions(
          while_body_indvar_update, {{while_body_indvar, &indvar_iter0_val}});
  if (!indvar_iter1_result.ok()) {
    VLOG(2) << "Couldn't evaluate induction variable update: "
            << indvar_iter1_result.status();
    return nullopt;
  }
  const Literal& indvar_iter1_val = *indvar_iter1_result.ValueOrDie();
  optional<bool> while_cond_iter1_val = evaluate_while_cond(indvar_iter1_val);
  if (while_cond_iter1_val == false) {
    VLOG(2) << "Determined that loop has static trip count of 1.";
    return 1;
  }

  VLOG(2) << "Loop has unknown trip count >= 1.";
  return nullopt;
}

// Tries to remove a while loop from the graph.
//
//  - Loops with trip count of 0 can be replaced by the loop's "init" value.
//  - Loops with trip count of 1 can be replaced by the loop's body, with the
//    loop itself removed.
//
// Returns true if it made a change to the graph.
static StatusOr<bool> TryRemoveWhileLoop(HloInstruction* while_op) {
  // We can't remove while loops that contain send/recv nodes, because we rely
  // on the particular loop structure around the node matching on the send and
  // recv sides.
  if (ContainsSendOrRecv(while_op->while_body()) ||
      ContainsSendOrRecv(while_op->while_condition())) {
    VLOG(2) << "Not attempting to remove while loop because it contains a "
               "send/recv node: "
            << while_op->ToShortString();
    return false;
  }

  // Cowardly refuse to remove loops that are not removable.  In practice,
  // this means that we can't remove loops that contain side-effecting
  // instructions or have control predecessors/successors.
  //
  // This is not a fundamental limitation.  The control operands can be moved
  // onto the new HLOs after simplification, and any side-effecting ops inside
  // the loop aren't removed, just cloned and added back to the loop.
  // Nevertheless our infrastructure sees loop simplification as removal of
  // these nodes and currently doesn't allow it.
  if (!while_op->parent()->IsRemovable(while_op)) {
    VLOG(2) << "Not attempting to remove while loop it is not removable: "
            << while_op->ToShortString();
    return false;
  }

  // Remove while loops with static trip count of 0.
  optional<int64> trip_count = GetLoopTripCount(while_op);
  if (trip_count && *trip_count == 0) {
    // The loop never executes, so the value of the loop is the value of its
    // "init" operand.
    auto computation = while_op->parent();

    // Remove while_op (i.e., call ReplaceInstruction rather than
    // ReplaceUsesWithInstruction) so that if the algebraic simplifier is run in
    // a loop without an intervening DCE, we don't try to re-remove the loop.
    TF_RETURN_IF_ERROR(computation->ReplaceInstruction(
        while_op, while_op->mutable_operand(0)));
    return true;
  }

  // Transform while loops with static trip count of 1 into a call op, then
  // inline the call.
  if (trip_count && *trip_count == 1) {
    auto computation = while_op->parent();
    auto call_op = computation->AddInstruction(HloInstruction::CreateCall(
        while_op->shape(), while_op->operands(), while_op->while_body()));
    TF_RETURN_IF_ERROR(computation->ReplaceInstruction(while_op, call_op));
    TF_RETURN_IF_ERROR(CallInliner::Inline(call_op));
    return true;
  }
  return false;
}

StatusOr<bool> WhileLoopSimplifier::Run(HloModule* module) {
  XLA_VLOG_LINES(2,
                 "WhileLoopSimplifier::Run(), before:\n" + module->ToString());
  bool changed = false;

  // Gather all the while ops in our module.  We do this ahead of time so we
  // don't have to worry about mutating the lists of computations or
  // instructions while we iterate.
  std::vector<HloInstruction*> while_ops;
  for (auto* comp : module->computations()) {
    for (auto* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kWhile) {
        while_ops.push_back(instr);
      }
    }
  }

  for (HloInstruction* while_op : while_ops) {
    StatusOr<bool> result = TryRemoveWhileLoop(while_op);
    TF_RETURN_IF_ERROR(result.status());
    changed |= result.ValueOrDie();
  }

  XLA_VLOG_LINES(2,
                 "WhileLoopSimplifier::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
