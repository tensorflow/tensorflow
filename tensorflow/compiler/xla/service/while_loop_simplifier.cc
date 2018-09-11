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
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/while_loop_analysis.h"
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace xla {

using absl::optional;

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
      instr->opcode() == HloOpcode::kSendDone ||
      instr->opcode() == HloOpcode::kRecv ||
      instr->opcode() == HloOpcode::kRecvDone) {
    return true;
  }
  for (const auto& subcomp : instr->called_computations()) {
    if (ContainsSendOrRecv(subcomp)) {
      return true;
    }
  }
  return false;
}

// Tries to remove elements in a while loop's tuple that aren't used within the
// loop.
//
// Specifically, if a loop is tuple-shaped, and there exists some element of
// that tuple that is not used by the loop condition and is not used by the loop
// body except to pass it to the next iteration of the loop, then we can remove
// that element from the loop's tuples.
static StatusOr<bool> TryRemoveDeadWhileParams(HloInstruction* while_op) {
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);

  // Don't try this transformation if the while loop isn't removable, since if
  // it succeeds ultimately we're going to have to replace the old while loop
  // with a new one.
  if (!while_op->parent()->IsRemovable(while_op) || while_op->HasSideEffect()) {
    VLOG(2) << "Can't remove dead parameters from non-removable while op.";
    return false;
  }

  HloModule* module = while_op->GetModule();
  HloComputation* computation = while_op->parent();
  HloInstruction* while_init = while_op->mutable_operand(0);
  HloComputation* while_cond = while_op->while_condition();
  HloComputation* while_body = while_op->while_body();
  HloInstruction* while_body_root = while_body->root_instruction();

  if (!ShapeUtil::IsTuple(while_init->shape())) {
    VLOG(2) << "While op's carried value isn't tuple shaped.";
    return false;
  }

  if (while_body_root->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "While body's root is not a tuple(...) instruction.";
    return false;
  }

  auto print_no_metadata = HloPrintOptions().set_print_metadata(false);

  // Bail if param0 of while_cond or while_body has users which aren't of type
  // get-tuple-element.
  for (const HloInstruction* instr : {while_body->parameter_instruction(0),
                                      while_cond->parameter_instruction(0)}) {
    for (const HloInstruction* user : instr->users()) {
      if (user->opcode() != HloOpcode::kGetTupleElement) {
        VLOG(2) << "Cowardly refusing to analyze while loop with "
                << instr->ToString(print_no_metadata)
                << " used by non-GTE instruction "
                << user->ToString(print_no_metadata) << " in computation "
                << instr->parent()->name();
        return false;
      }
    }
  }

  const int64 tuple_size = ShapeUtil::TupleElementCount(while_init->shape());
  if (tuple_size == 0) {
    VLOG(2) << "Can't remove elements from while loop's tuple -- it's already "
               "empty.";
    return false;
  }

  tensorflow::gtl::FlatSet<int64> used_tuple_indices;
  for (HloComputation* comp : {while_body, while_cond}) {
    // The HLO verifier ensures that while_input's shape matches while_init's
    // shape, which we verified above is a tuple.
    HloInstruction* while_input = comp->parameter_instruction(0);

    for (const HloInstruction* user : while_input->users()) {
      // This user doesn't count if it's only used by the while body's root, and
      // the root places the tuple element into the same index of the tuple as
      // it came from.  That just amounts to us carrying the variable through
      // the loop.
      //
      // Careful: HloInstruction::operand_index returns the first index the
      // operand appears in, but it may appear more than once!
      if (user->user_count() == 1 && user->users().front() == while_body_root &&
          while_body_root->operand_index(user) == user->tuple_index() &&
          std::count(while_body_root->operands().begin(),
                     while_body_root->operands().end(), user) == 1) {
        continue;
      }

      used_tuple_indices.insert(user->tuple_index());
      if (used_tuple_indices.size() == tuple_size) {
        VLOG(2) << "Loop " << while_op->ToString(print_no_metadata)
                << " uses all of its inputs; no simplification possible.";
        return false;
      }
    }
  }

  // If a tuple element is not passed unmodified from the while body's param0
  // through to the while body's root, count that element as "used", since
  // removing that element would be observable.
  for (int64 i = 0; i < while_body_root->operand_count(); ++i) {
    if (used_tuple_indices.count(i)) {
      continue;
    }

    auto* operand = while_body_root->operand(i);
    if (operand->opcode() != HloOpcode::kGetTupleElement ||
        operand->operand(0) != while_body->parameter_instruction(0) ||
        operand->tuple_index() != i) {
      VLOG(2) << "Tuple index " << i
              << " is not passed through loop body unmodified.";
      used_tuple_indices.insert(i);

      if (used_tuple_indices.size() == tuple_size) {
        VLOG(2) << "Loop " << while_op->ToString(print_no_metadata)
                << " uses all of its inputs; no simplification possible.";
        return false;
      }
    }
  }

  // If we got here, used_tuple_indices.size() < tuple_size, meaning some
  // elements of the loop's tuple aren't used by while_body or while_cond.
  CHECK_LT(used_tuple_indices.size(), tuple_size);

  VLOG(1) << "Eliminating " << tuple_size - used_tuple_indices.size()
          << " elements from tuple of "
          << while_op->ToString(print_no_metadata);

  // Build up maps from the old/new to the new/old tuple indices.
  std::vector<int64> new_to_old_tuple_idx(used_tuple_indices.begin(),
                                          used_tuple_indices.end());
  std::sort(new_to_old_tuple_idx.begin(), new_to_old_tuple_idx.end());

  tensorflow::gtl::FlatMap<int64, int64> old_to_new_tuple_idx;
  for (int64 new_idx = 0; new_idx < new_to_old_tuple_idx.size(); ++new_idx) {
    int64 old_idx = new_to_old_tuple_idx[new_idx];
    old_to_new_tuple_idx[old_idx] = new_idx;
    VLOG(2) << "Remapping tuple index " << old_idx << " to " << new_idx;
  }

  // Compute the shape of the while op after we remove the dead indices.
  std::vector<Shape> new_while_tuple_elem_shapes;
  new_while_tuple_elem_shapes.reserve(new_to_old_tuple_idx.size());
  for (int64 old_idx : new_to_old_tuple_idx) {
    new_while_tuple_elem_shapes.push_back(
        while_init->shape().tuple_shapes(old_idx));
  }
  Shape new_while_shape =
      ShapeUtil::MakeTupleShape(new_while_tuple_elem_shapes);

  // Returns a map from elements in the computation to new instructions which
  // replace the old instructions after we remove unused elements from the while
  // tuple.
  auto make_while_computation_replacements = [&](const HloComputation* comp) {
    std::unordered_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
        replacements;

    auto* param = comp->parameter_instruction(0);
    replacements.emplace(param, HloInstruction::CreateParameter(
                                    0, new_while_shape, param->name()));

    // Materialize param's users, since we're about to add new ones below.
    std::vector<HloInstruction*> materialized_users(param->users().begin(),
                                                    param->users().end());
    for (const auto* user : materialized_users) {
      // The while body root is handled separately.
      if (user == while_body_root) {
        continue;
      }
      CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement)
          << user->ToString(print_no_metadata);

      int64 old_idx = user->tuple_index();
      auto new_idx_iter = old_to_new_tuple_idx.find(old_idx);
      if (new_idx_iter != old_to_new_tuple_idx.end()) {
        // This is a GTE of an index that survives.  Replace it.
        replacements.emplace(
            user, HloInstruction::CreateGetTupleElement(user->shape(), param,
                                                        new_idx_iter->second));
      } else {
        // This is a GTE of an index that we've removed.  Remove it from the
        // cloned computation.
        CHECK(user->user_count() == 0 ||
              user->user_count() == 1 &&
                  user->users().front() == while_body_root)
            << "Instruction " << user->ToString(print_no_metadata)
            << " should be unused (except by root of while body), but has "
               "users: {"
            << absl::StrJoin(user->users(), ", ",
                             [&](string* out, const HloInstruction* instr) {
                               absl::StrAppend(
                                   out, instr->ToString(print_no_metadata));
                             })
            << "}";

        replacements.emplace(user, nullptr);
      }
    }
    return replacements;
  };

  // Create the new while condition, body, and init value.
  std::unique_ptr<HloComputation> new_while_cond =
      while_cond->CloneWithReplacements(
          make_while_computation_replacements(while_cond));

  std::unordered_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      while_body_replacements = make_while_computation_replacements(while_body);
  std::vector<HloInstruction*> new_while_body_root_elems;
  new_while_body_root_elems.reserve(new_to_old_tuple_idx.size());
  for (int64 old_idx : new_to_old_tuple_idx) {
    new_while_body_root_elems.push_back(
        while_body_root->mutable_operand(old_idx));
  }
  while_body_replacements.emplace(
      while_body_root, HloInstruction::CreateTuple(new_while_body_root_elems));
  std::unique_ptr<HloComputation> new_while_body =
      while_body->CloneWithReplacements(std::move(while_body_replacements));

  // Add a new while_init instruction that repackages the old while_init
  // instruction's elements.  We rely on the AlgebraicSimplifier and DCE to
  // clean this up in the common case where while_init is a tuple op.  (It's
  // definitely tuple-shaped, but it's not necessarily a tuple op.)
  std::vector<HloInstruction*> new_while_init_elems;
  new_while_init_elems.reserve(new_to_old_tuple_idx.size());
  for (int64 old_idx : new_to_old_tuple_idx) {
    new_while_init_elems.push_back(
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            while_init->shape().tuple_shapes(old_idx), while_init, old_idx)));
  }
  auto* new_while_init = computation->AddInstruction(
      HloInstruction::CreateTuple(new_while_init_elems));

  // Create the new while op.
  auto* new_while_op = computation->AddInstruction(HloInstruction::CreateWhile(
      new_while_shape,
      module->AddEmbeddedComputation(std::move(new_while_cond)),
      module->AddEmbeddedComputation(std::move(new_while_body)),
      new_while_init));

  // Create a tuple op that recreates the output of the old while op.  That is,
  // we transform to
  //
  //  new_while_init   while_init
  //       |              |
  //       V              |
  //   new_while          |
  //       |              |
  //       -------|   |----
  //              V   V
  //            new_tuple
  //                |
  //                V
  //    (orig. users of while op)
  //
  // The tuple simplifier will then simplify this if possible, removing
  // new_tuple and while_init.
  std::vector<HloInstruction*> new_tuple_elems;
  for (int64 old_idx = 0; old_idx < tuple_size; ++old_idx) {
    auto new_tuple_idx_it = old_to_new_tuple_idx.find(old_idx);
    if (new_tuple_idx_it != old_to_new_tuple_idx.end()) {
      int64 gte_idx = new_tuple_idx_it->second;
      new_tuple_elems.push_back(
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              new_while_op->shape().tuple_shapes(gte_idx), new_while_op,
              gte_idx)));
    } else {
      new_tuple_elems.push_back(
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              while_init->shape().tuple_shapes(old_idx), while_init, old_idx)));
    }
  }
  HloInstruction* new_tuple =
      computation->AddInstruction(HloInstruction::CreateTuple(new_tuple_elems));
  TF_RETURN_IF_ERROR(while_op->ReplaceAllUsesWith(new_tuple));

  return true;
}

// Tries to remove a while loop from the graph.
//
//  - Loops with trip count of 0 can be replaced by the loop's "init" value.
//  - Loops with trip count of 1 can be replaced by the loop's body, with the
//    loop itself removed.
//
// Returns true if it made a change to the graph.
static StatusOr<bool> TryRemoveWhileLoop(HloInstruction* while_op) {
  // Cowardly refuse to remove loops that are not removable.  In practice,
  // this means that we can't remove loops that contain side-effecting
  // instructions or have control predecessors/successors.
  //
  // This is not a fundamental limitation.  The control operands can be moved
  // onto the new HLOs after simplification, and any side-effecting ops inside
  // the loop aren't removed, just cloned and added back to the loop.  But
  // moving an op out of the loop also removes implicit control dependencies
  // between the op and the ops outside the loop, so we'd have to add those back
  // for things like infeed/outfeed.  It gets complicated.  So for now we just
  // avoid it.
  if (!while_op->parent()->IsRemovable(while_op) || while_op->HasSideEffect()) {
    VLOG(2) << "Not attempting to remove while loop it is not removable: "
            << while_op->ToShortString();
    return false;
  }

  // Remove while loops with static trip count of 0.
  optional<int64> trip_count =
      ComputeWhileLoopTripCount(while_op,
                                /*max_value_returned=*/1);
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
    TF_ASSIGN_OR_RETURN(auto inlined_instructions_map,
                        CallInliner::Inline(call_op));
    (void)inlined_instructions_map;
    return true;
  }
  return false;
}

static StatusOr<bool> TryPropagateConstant(HloInstruction* while_op) {
  auto while_init = while_op->operand(0);
  if (while_init->opcode() != HloOpcode::kTuple) {
    return false;
  }

  auto while_body = while_op->while_body();
  auto while_body_root = while_body->root_instruction();
  if (while_body_root->opcode() != HloOpcode::kTuple) {
    return false;
  }

  auto while_body_param = while_body->parameter_instruction(0);
  const HloInstruction::InstructionVector& root_operands =
      while_body_root->operands();

  // Find the loop invariant tuple elements with scalar constant init value and
  // build a map from the tuple element index to the constant value. Limit this
  // to scalar constant values because propagating array constants can regress
  // performance by forcing us to copy constants.
  tensorflow::gtl::FlatMap<int, const HloInstruction*> index_to_constant;
  for (int i = 0; i < root_operands.size(); i++) {
    HloInstruction* instr = root_operands[i];
    if (instr->opcode() == HloOpcode::kGetTupleElement &&
        instr->tuple_index() == i && instr->operand(0) == while_body_param &&
        ShapeUtil::IsScalar(instr->shape())) {
      auto tuple_element = while_init->operand(i);
      if (tuple_element->IsConstant()) {
        VLOG(3) << "Found loop invariant tuple element " << i << " "
                << tuple_element->ToString();
        index_to_constant[i] = tuple_element;
      }
    }
  }

  if (index_to_constant.empty()) {
    return false;
  }

  // Replace the use of each constant tuple element in the loop_condition and
  // loop_body with the corresponding constant value.
  auto propagate_constant = [&](HloComputation* computation) -> StatusOr<bool> {
    HloInstruction* param = computation->parameter_instruction(0);
    bool changed = false;
    for (auto instr : param->users()) {
      // Since only a while-loop with a tuple result reaches here, we can safely
      // assume that `param` is a tuple and the first operand of the
      // GetTupleElement instruction is a use of `param`.
      if (instr->opcode() == HloOpcode::kGetTupleElement) {
        VLOG(3) << "tuple index " << instr->tuple_index() << " "
                << instr->ToString();
        auto iter = index_to_constant.find(instr->tuple_index());
        if (iter != index_to_constant.end()) {
          const HloInstruction* hlo_constant = (*iter).second;
          VLOG(3) << "Replace use of " << instr->ToString() << " with "
                  << hlo_constant->ToString();
          TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(
              computation->AddInstruction(hlo_constant->Clone())));
          changed = true;
        }
      }
    }
    return changed;
  };

  TF_ASSIGN_OR_RETURN(bool changed_cond,
                      propagate_constant(while_op->while_condition()));
  TF_ASSIGN_OR_RETURN(bool changed_body, propagate_constant(while_body));

  return changed_cond || changed_body;
}

StatusOr<bool> WhileLoopSimplifier::Run(HloModule* module) {
  XLA_VLOG_LINES(3,
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
    // We can't remove while loops that contain send/recv nodes, because we rely
    // on the particular loop structure around the node matching on the send and
    // recv sides.  Removing dead while params requires us to remove the loop
    // and replace it with a new one, so we can't do that either.
    if (ContainsSendOrRecv(while_op->while_body()) ||
        ContainsSendOrRecv(while_op->while_condition())) {
      VLOG(2) << "Not attempting to simplify while loop because it contains a "
                 "send/recv node: "
              << while_op->ToShortString();
      continue;
    }

    StatusOr<bool> result = TryPropagateConstant(while_op);
    TF_RETURN_IF_ERROR(result.status());
    changed |= result.ValueOrDie();

    result = TryRemoveWhileLoop(while_op);
    TF_RETURN_IF_ERROR(result.status());
    if (result.ValueOrDie()) {
      changed = true;
      // Don't try to remove dead while params after successfully removing the
      // while loop -- that would result in use-after-free nastiness.
      continue;
    }

    result = TryRemoveDeadWhileParams(while_op);
    TF_RETURN_IF_ERROR(result.status());
    changed |= result.ValueOrDie();
  }

  XLA_VLOG_LINES(3,
                 "WhileLoopSimplifier::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
