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
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/while_loop_analysis.h"

namespace xla {

namespace m = match;
using absl::optional;
using hlo_query::ContainsInstrWithOpcode;

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

  absl::flat_hash_set<int64> used_tuple_indices;
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

  absl::flat_hash_map<int64, int64> old_to_new_tuple_idx;
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

// Removes each loop parameter (i.e. member of the while loop tuple) that is a
// constant and is the same in the while loop body and the while loop init.
static StatusOr<bool> TryRemoveConstantParams(HloInstruction* while_op) {
  HloModule* module = while_op->GetModule();
  HloComputation* computation = while_op->parent();
  auto* while_init = while_op->mutable_operand(0);
  auto* while_body = while_op->while_body();
  auto* while_cond = while_op->while_condition();
  auto* while_body_root = while_body->root_instruction();
  if (while_init->opcode() != HloOpcode::kTuple ||
      while_body_root->opcode() != HloOpcode::kTuple) {
    return false;
  }

  TF_RET_CHECK(while_cond->num_parameters() == 1);
  TF_RET_CHECK(while_body->num_parameters() == 1);
  TF_RET_CHECK(
      ShapeUtil::Compatible(while_init->shape(), while_body_root->shape()));

  absl::flat_hash_set<int64> constant_tuple_indices;
  const auto& while_shape = while_init->shape();
  for (int64 i = 0; i < while_shape.tuple_shapes_size(); ++i) {
    auto* init_elem = while_init->operand(i);
    auto* body_elem = while_body_root->operand(i);
    if (init_elem->opcode() == HloOpcode::kConstant &&
        body_elem->opcode() == HloOpcode::kConstant &&
        init_elem->literal() == body_elem->literal()) {
      constant_tuple_indices.insert(i);
    }
  }

  if (constant_tuple_indices.empty()) {
    return false;
  }

  // OK, we found some constant elements of the while parameter!  Eliminate
  // them.
  std::vector<Shape> new_while_shape_elems;
  for (int64 i = 0; i < while_shape.tuple_shapes_size(); ++i) {
    if (!constant_tuple_indices.count(i)) {
      new_while_shape_elems.push_back(while_shape.tuple_shapes(i));
    }
  }
  Shape new_while_shape = ShapeUtil::MakeTupleShape(new_while_shape_elems);

  // `new_instrs` holds instructions created outside of a computation for
  // cloning.  Elements added here just need to live until the end of the
  // relevant CloneWithReplacement call.
  std::vector<std::unique_ptr<HloInstruction>> new_instrs;
  auto add_new_instr = [&](std::unique_ptr<HloInstruction> instr) {
    new_instrs.push_back(std::move(instr));
    return new_instrs.back().get();
  };

  // Returns a new tuple without the elements of constant_tuple_indices.
  auto remove_constant_elems = [&](HloInstruction* instr) {
    CHECK(ShapeUtil::Compatible(instr->shape(), while_shape));

    std::vector<HloInstruction*> tuple_elems;
    for (int64 i = 0; i < while_shape.tuple_shapes_size(); ++i) {
      if (!constant_tuple_indices.count(i)) {
        tuple_elems.push_back(
            add_new_instr(HloInstruction::CreateGetTupleElement(
                while_shape.tuple_shapes(i), instr, i)));
      }
    }
    return HloInstruction::CreateTuple(tuple_elems);
  };

  auto add_constant_elems = [&](HloInstruction* instr) {
    CHECK(ShapeUtil::Compatible(instr->shape(), new_while_shape));

    std::vector<HloInstruction*> tuple_elems;
    int64 j = 0;
    for (int64 i = 0; i < while_shape.tuple_shapes_size(); ++i) {
      if (constant_tuple_indices.count(i)) {
        tuple_elems.push_back(while_init->mutable_operand(i));
      } else {
        tuple_elems.push_back(
            add_new_instr(HloInstruction::CreateGetTupleElement(
                while_shape.tuple_shapes(i), instr, j)));
        ++j;
      }
    }
    return HloInstruction::CreateTuple(tuple_elems);
  };

  // Special case: constant_tuple_indices covers the whole while parameter, so
  // the new while shape is the empty tuple.  In this case, the value of the
  // while loop is simply equal to the value of `init`.
  //
  // It's unfortunate to special-case this, but it's simpler than the
  // alternative.  The problem is that if our while parameter has no
  // non-constant elems, the tuple returned by `add_constant_elems` won't depend
  // on instr (the loop body/cond parameter), and therefore
  // CloneWithReplacementPairs will *leave the parameter out entirely*, creating
  // invalid HLO.
  if (ShapeUtil::IsEmptyTuple(new_while_shape)) {
    TF_RETURN_IF_ERROR(computation->ReplaceInstruction(while_op, while_init));
    return true;
  }

  std::unique_ptr<HloComputation> new_while_cond =
      while_cond->CloneWithReplacementPairs({
          while_cond->parameter_instruction(0),
          add_constant_elems(add_new_instr(HloInstruction::CreateParameter(
              0, new_while_shape,
              while_cond->parameter_instruction(0)->name()))),
      });

  std::unique_ptr<HloComputation> new_while_body =
      while_body->CloneWithReplacementPairs(
          {
              while_body->parameter_instruction(0),
              add_constant_elems(add_new_instr(HloInstruction::CreateParameter(
                  0, new_while_shape,
                  while_cond->parameter_instruction(0)->name()))),
          },
          {
              while_body->root_instruction(),
              remove_constant_elems(
                  add_new_instr(while_body->root_instruction()->Clone())),
          });

  // Create the final while loop, and add any new instructions created to
  // `computation`.
  new_instrs.clear();
  TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
      while_op,
      add_constant_elems(
          computation->AddInstruction(HloInstruction::CreateWhile(
              new_while_shape,
              module->AddEmbeddedComputation(std::move(new_while_cond)),
              module->AddEmbeddedComputation(std::move(new_while_body)),
              add_new_instr(remove_constant_elems(while_init)))))));
  for (auto& instr : new_instrs) {
    computation->AddInstruction(std::move(instr));
  }
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
  absl::flat_hash_map<int, const HloInstruction*> index_to_constant;
  for (int i = 0; i < root_operands.size(); i++) {
    const HloInstruction* init_tuple_elem = nullptr;
    if (Match(root_operands[i],
              m::GetTupleElement(m::Op().Is(while_body_param), i)
                  .WithShape(m::Shape().IsScalar())) &&
        Match(while_init->operand(i), m::Constant(&init_tuple_elem))) {
      VLOG(3) << "Found loop invariant tuple element " << i << " "
              << init_tuple_elem->ToString();
      index_to_constant[i] = init_tuple_elem;
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

// Converts a flat list of instructions into a tuple of the desired shape.  For
// example, given a tuple shape ((x, x), x) and instructions {A, B, C}, returns
// a tuple of value ((A, B), C).
//
// desired_shape must be a tuple.  (This precondition allows us to return a
// unique_ptr rather than a raw ptr.)
static std::unique_ptr<HloInstruction> UnflattenTupleInstr(
    absl::Span<HloInstruction*> instrs, const Shape& desired_shape,
    std::vector<std::unique_ptr<HloInstruction>>* new_instrs) {
  CHECK(ShapeUtil::IsTuple(desired_shape))
      << ShapeUtil::HumanString(desired_shape);

  // For each child shape in `desired_shape`, slice out the correct number of
  // `instrs` and call UnflattenTupleInstr recursively.  At each step we remove
  // elements from `instrs` so that it only contains instructions we have not
  // yet processed.
  std::vector<HloInstruction*> elems;
  for (int64 i = 0; i < desired_shape.tuple_shapes_size(); ++i) {
    const Shape& subshape = desired_shape.tuple_shapes(i);
    if (!ShapeUtil::IsTuple(subshape)) {
      elems.push_back(instrs[0]);
      instrs.remove_prefix(1);
      continue;
    }

    // Count the number of leaf nodes underneath desired_shape[i].
    int64 num_leaves = 0;
    ShapeUtil::ForEachSubshape(
        subshape, [&](const Shape& s, const ShapeIndex& /*index*/) {
          if (!ShapeUtil::IsTuple(s)) {
            ++num_leaves;
          }
        });

    std::unique_ptr<HloInstruction> subinstr =
        UnflattenTupleInstr(instrs.subspan(0, num_leaves),
                            desired_shape.tuple_shapes(i), new_instrs);
    elems.push_back(subinstr.get());
    new_instrs->push_back(std::move(subinstr));
    instrs.remove_prefix(num_leaves);
  }
  return HloInstruction::CreateTuple(elems);
}

// Builds a vector whose elements are the values in the flattened tuple for
// `instr`.  For example, if `instr` is a tuple of form ((A, B), C), returns the
// vector {A, B, C} (or kGetTupleElement ops which point to A, B, and C).
static std::vector<HloInstruction*> GetFlatTupleElems(
    HloInstruction* instr,
    std::vector<std::unique_ptr<HloInstruction>>* new_instrs) {
  const auto& shape = instr->shape();
  if (!ShapeUtil::IsTuple(shape)) {
    return {instr};
  }
  std::vector<HloInstruction*> elems;
  for (int64 i = 0; i < shape.tuple_shapes_size(); ++i) {
    const Shape& subshape = shape.tuple_shapes(i);
    new_instrs->push_back(
        HloInstruction::CreateGetTupleElement(subshape, instr, i));
    auto* gte = new_instrs->back().get();
    auto flattened_subshape = GetFlatTupleElems(gte, new_instrs);
    elems.insert(elems.end(), flattened_subshape.begin(),
                 flattened_subshape.end());
  }
  return elems;
}

static StatusOr<bool> TryFlattenNestedTuples(HloInstruction* while_op) {
  HloModule* module = while_op->GetModule();
  HloComputation* computation = while_op->parent();
  auto* while_init = while_op->mutable_operand(0);
  auto* while_body = while_op->while_body();
  auto* while_cond = while_op->while_condition();
  auto* while_body_root = while_body->root_instruction();
  if (while_init->opcode() != HloOpcode::kTuple ||
      while_body_root->opcode() != HloOpcode::kTuple) {
    return false;
  }

  TF_RET_CHECK(while_cond->num_parameters() == 1);
  TF_RET_CHECK(while_body->num_parameters() == 1);
  TF_RET_CHECK(
      ShapeUtil::Compatible(while_init->shape(), while_body_root->shape()));
  Shape while_shape = while_init->shape();
  if (!ShapeUtil::IsNestedTuple(while_shape)) {
    return false;
  }

  std::vector<Shape> flattened_shape_elems;
  ShapeUtil::ForEachSubshape(while_shape,
                             [&](const Shape& s, const ShapeIndex& /*index*/) {
                               if (!ShapeUtil::IsTuple(s)) {
                                 flattened_shape_elems.push_back(s);
                               }
                             });
  Shape flattened_shape = ShapeUtil::MakeTupleShape(flattened_shape_elems);

  // `new_instrs` holds instructions created outside of a computation for
  // cloning.  Elements added here just need to live until the end of the
  // relevant CloneWithReplacement call.
  std::vector<std::unique_ptr<HloInstruction>> new_instrs;
  auto add_new_instr = [&](std::unique_ptr<HloInstruction> instr) {
    new_instrs.push_back(std::move(instr));
    return new_instrs.back().get();
  };

  auto nested = [&](HloInstruction* instr) {
    std::vector<HloInstruction*> gtes;
    const Shape& flat_shape = instr->shape();
    for (int64 i = 0; i < flat_shape.tuple_shapes_size(); ++i) {
      gtes.push_back(add_new_instr(HloInstruction::CreateGetTupleElement(
          flat_shape.tuple_shapes(i), instr, i)));
    }
    auto nested_instr =
        UnflattenTupleInstr(absl::MakeSpan(gtes), while_shape, &new_instrs);
    CHECK(ShapeUtil::Compatible(nested_instr->shape(), while_shape))
        << ShapeUtil::HumanString(nested_instr->shape()) << " vs "
        << ShapeUtil::HumanString(while_shape);
    return nested_instr;
  };

  auto flattened = [&](HloInstruction* instr) {
    return HloInstruction::CreateTuple(GetFlatTupleElems(instr, &new_instrs));
  };

  // Create a new while-condition computation, where parameter 0 has flat shape
  // but all uses of it go through the nested shape.
  std::unique_ptr<HloComputation> new_while_cond =
      while_cond->CloneWithReplacementPairs({
          while_cond->parameter_instruction(0),
          nested(add_new_instr(HloInstruction::CreateParameter(
              0, flattened_shape,
              while_cond->parameter_instruction(0)->name()))),
      });

  // Create a new while-body computation, where parameter 0 has a flat shape and
  // all uses of it go through the nested shape, and where the root has a flat
  // shape constructed from the old nested root.
  std::unique_ptr<HloComputation> new_while_body =
      while_body->CloneWithReplacementPairs(
          {
              while_body->parameter_instruction(0),
              nested(add_new_instr(HloInstruction::CreateParameter(
                  0, flattened_shape,
                  while_body->parameter_instruction(0)->name()))),
          },
          {
              while_body->root_instruction(),
              flattened(add_new_instr(while_body->root_instruction()->Clone())),
          });

  // Create the final while loop, and add any new instructions created to
  // `computation`.
  new_instrs.clear();
  TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
      while_op, nested(computation->AddInstruction(HloInstruction::CreateWhile(
                    flattened_shape,
                    module->AddEmbeddedComputation(std::move(new_while_cond)),
                    module->AddEmbeddedComputation(std::move(new_while_body)),
                    computation->AddInstruction(flattened(while_init)))))));
  for (auto& instr : new_instrs) {
    computation->AddInstruction(std::move(instr));
  }
  return true;
}

// Tries to merge loop induction variables of a given type.
//
// In this pass we're only concerned with elements of the loop's tuple that
// are effective-scalars of type `elem_ty`.  Some terminology:
//
//  - The trip counter is the first element of the loop's tuple that starts at
//    0 and does x++ on each iteration.
//
//  - An induction variable is an element of the loop's tuple that is not the
//    trip counter and does `x += <constant>` on each iteration of the loop.
//    Negative constants are OK.
//
// This pass adds a trip counter if one isn't already present, then replaces
// each induction variable with
//
//   <initial_value> + <trip_count> * <constant>.
//
// This reduces the number of scalar operations in the loop, which is important
// e.g. on GPUs, where each scalar operation is nontrivially expensive because
// it's a separate kernel launch.
//
// Returns the new loop if a change was made, or null if no change was made.
// Note that the new loop is not a valid replacement for the old loop; it may
// need to be wrapped in a tuple that changes its shape.  We return the loop
// itself so that you can call TryMergeInductionVariables in a loop, once for
// each integral type elem_ty.
static StatusOr<HloInstruction*> TryMergeInductionVariables(
    HloInstruction* while_op, PrimitiveType elem_ty) {
  CHECK(primitive_util::IsIntegralType(elem_ty)) << PrimitiveType_Name(elem_ty);
  HloModule* module = while_op->GetModule();
  HloComputation* computation = while_op->parent();
  auto* while_init = while_op->mutable_operand(0);
  auto* while_body = while_op->while_body();
  auto* while_cond = while_op->while_condition();
  auto* while_body_root = while_body->root_instruction();
  if (while_init->opcode() != HloOpcode::kTuple ||
      while_body_root->opcode() != HloOpcode::kTuple) {
    return nullptr;
  }

  TF_RET_CHECK(while_cond->num_parameters() == 1);
  TF_RET_CHECK(while_body->num_parameters() == 1);
  TF_RET_CHECK(
      ShapeUtil::Compatible(while_init->shape(), while_body_root->shape()));
  Shape while_shape = while_init->shape();

  // The tuple index of the trip counter, if one is present.
  absl::optional<int64> trip_counter;
  // Maps the tuple index of each induction variable to its constant increment.
  absl::flat_hash_map<int64, const HloConstantInstruction*> induction_vars;
  for (int64 i = 0; i < while_body_root->operand_count(); ++i) {
    HloInstruction* constant;
    if (!Match(while_body_root->mutable_operand(i),
               m::AddAnyOrder(m::GetTupleElement(m::Parameter(), i),
                              m::ConstantScalar(&constant))
                   .WithShape(m::Shape().WithElementType(elem_ty)))) {
      continue;
    }
    if (!trip_counter && constant->literal().IsAll(1) &&
        while_init->operand(i)->IsConstant() &&
        while_init->operand(i)->literal().IsAll(0)) {
      VLOG(10) << "Found existing trip counter at index " << i;
      trip_counter = i;
    } else {
      VLOG(10) << "Found induction variable at index " << i;
      induction_vars.emplace(i, Cast<HloConstantInstruction>(constant));
    }
  }

  // There's only something to simplify if we can either:
  //
  //  - combine one or more induction vars with an existing trip counter, or
  //  - replace two or more induction variables with a new trip counter.
  //
  // Put another way, there's only something to simplify if the number of
  // induction vars plus the number of existing trip counters (0 or 1) is >= 2.
  if (induction_vars.size() + (trip_counter.has_value() ? 1 : 0) < 2) {
    return nullptr;
  }

  // OK, we're going to do the transformation!  Set up some helpers.

  // `new_instrs` holds instructions created outside of a computation for
  // cloning.  Elements added here just need to live until the end of the
  // relevant CloneWithReplacement call.
  std::vector<std::unique_ptr<HloInstruction>> new_instrs;
  auto add_new_instr = [&](std::unique_ptr<HloInstruction> instr) {
    new_instrs.push_back(std::move(instr));
    return new_instrs.back().get();
  };

  auto add_binary_op = [&](const Shape& shape, HloOpcode opcode,
                           HloInstruction* lhs, HloInstruction* rhs) {
    // Reshape lhs/rhs to the output shape if necessary.  This deals with the
    // fact that induction variables need only be effective scalars, not true
    // scalars.
    if (!ShapeUtil::Compatible(shape, lhs->shape())) {
      lhs = add_new_instr(HloInstruction::CreateReshape(shape, lhs));
    }
    if (!ShapeUtil::Compatible(shape, rhs->shape())) {
      rhs = add_new_instr(HloInstruction::CreateReshape(shape, rhs));
    }
    return add_new_instr(HloInstruction::CreateBinary(shape, opcode, lhs, rhs));
  };

  auto add_gte = [&](HloInstruction* src, int64 idx) {
    return add_new_instr(HloInstruction::CreateGetTupleElement(
        src->shape().tuple_shapes(idx), src, idx));
  };

  // Our new while loop will have the same shape as the old while loop, except
  // we'll add a trip counter to the end if it wasn't originally present.
  Shape new_while_shape = while_shape;
  bool added_trip_counter = false;
  if (!trip_counter) {
    VLOG(10) << "Adding new trip counter to end of loop's tuple.";
    trip_counter = new_while_shape.tuple_shapes_size();
    *new_while_shape.add_tuple_shapes() =
        ShapeUtil::MakeShape(elem_ty, /*dimensions=*/{});
    added_trip_counter = true;
  }

  // Converts `instr` into a tuple of the "old" form -- that is, to a tuple with
  // shape `while_body->shape()` and where the induction variables are "reified"
  // (i.e. they have value <init> + <counter> * <constant>).
  auto convert_to_old_form = [&](HloInstruction* instr) {
    CHECK(ShapeUtil::Compatible(instr->shape(), new_while_shape));
    std::vector<HloInstruction*> tuple_elems;
    for (int64 i = 0; i < while_shape.tuple_shapes_size(); ++i) {
      const auto& elem_shape = while_shape.tuple_shapes(i);
      if (!induction_vars.count(i)) {
        tuple_elems.push_back(add_gte(instr, i));
        continue;
      }
      tuple_elems.push_back(add_binary_op(
          elem_shape, HloOpcode::kAdd, add_gte(instr, i),
          add_binary_op(elem_shape, HloOpcode::kMultiply,
                        add_gte(instr, *trip_counter),
                        add_new_instr(induction_vars.at(i)->Clone()))));
    }
    return HloInstruction::CreateTuple(tuple_elems);
  };

  // Converts `root` into a tuple of the "new" form -- that is, to a tuple with
  // shape `new_while_shape` and where the induction variables (but not trip
  // counters) are replaced with their unchanging <loop_body_param> values.
  auto convert_to_new_form = [&](HloInstruction* old_root,
                                 HloParameterInstruction* loop_body_param) {
    CHECK(ShapeUtil::Compatible(old_root->shape(), while_shape));
    std::vector<HloInstruction*> tuple_elems;

    // In the new form, induction variables come from `init`, everything else
    // (including the trip counter if it's not one we created ourselves) comes
    // from the `root` tuple unmodified.
    for (int64 i = 0; i < while_shape.tuple_shapes_size(); ++i) {
      tuple_elems.push_back(
          add_gte((induction_vars.count(i) ? loop_body_param : old_root), i));
    }
    // If we created a trip counter ourselves, add 1 to it in the next
    // iteration.
    if (added_trip_counter) {
      tuple_elems.push_back(add_binary_op(
          new_while_shape.tuple_shapes(*trip_counter), HloOpcode::kAdd,
          add_gte(loop_body_param, *trip_counter),
          add_new_instr(
              HloInstruction::CreateConstant(LiteralUtil::One(elem_ty)))));
    }

    return HloInstruction::CreateTuple(tuple_elems);
  };

  // Creates a new init tuple, which is the same as the old init tuple except if
  // we added a trip counter, it's set to 0.
  auto get_new_while_init = [&](HloInstruction* init) {
    CHECK(ShapeUtil::Compatible(init->shape(), while_shape));
    if (!added_trip_counter) {
      return init;
    }
    std::vector<HloInstruction*> tuple_elems;
    for (int64 i = 0; i < while_shape.tuple_shapes_size(); ++i) {
      tuple_elems.push_back(add_gte(init, i));
    }
    tuple_elems.push_back(add_new_instr(
        HloInstruction::CreateConstant(LiteralUtil::Zero(elem_ty))));
    return add_new_instr(HloInstruction::CreateTuple(tuple_elems));
  };

  std::unique_ptr<HloComputation> new_while_cond =
      while_cond->CloneWithReplacementPairs({
          while_cond->parameter_instruction(0),
          convert_to_old_form(add_new_instr(HloInstruction::CreateParameter(
              0, new_while_shape,
              while_cond->parameter_instruction(0)->name()))),
      });

  // Creating the new while body proceeds in two steps.  First we convert the
  // users of the parameter to the old form.  Then as a second
  // CloneWithReplacement operation we convert the root to the new form.  We
  // have to do this in two steps because the new root needs to use the new
  // param0, and during the first clone operation, only the *old-form* param0 is
  // accessible.
  //
  // We have to add temp_new_while_body to the module because cloning a
  // computation touches the module (to get its NameUniquer).
  HloComputation* temp_new_while_body =
      module->AddEmbeddedComputation(while_body->CloneWithReplacementPairs({
          while_body->parameter_instruction(0),
          convert_to_old_form(add_new_instr(HloInstruction::CreateParameter(
              0, new_while_shape,
              while_body->parameter_instruction(0)->name()))),
      }));
  std::unique_ptr<HloComputation> new_while_body =
      temp_new_while_body->CloneWithReplacementPairs({
          temp_new_while_body->root_instruction(),
          convert_to_new_form(
              add_new_instr(temp_new_while_body->root_instruction()->Clone()),
              Cast<HloParameterInstruction>(
                  temp_new_while_body->parameter_instruction(0))),
      });
  TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(temp_new_while_body));

  // Create the final while loop, and add any new instructions created to
  // `computation`.
  new_instrs.clear();
  auto* new_while = computation->AddInstruction(HloInstruction::CreateWhile(
      new_while_shape,
      module->AddEmbeddedComputation(std::move(new_while_cond)),
      module->AddEmbeddedComputation(std::move(new_while_body)),
      get_new_while_init(while_init)));
  TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
      while_op, convert_to_old_form(new_while)));
  for (auto& instr : new_instrs) {
    computation->AddInstruction(std::move(instr));
  }
  return new_while;
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
    // recv sides.  Other while simplifications require us to remove the loop
    // and replace it with a new one, so we can't do that either.
    if (ContainsInstrWithOpcode(while_op->while_body(),
                                {HloOpcode::kSend, HloOpcode::kSendDone,
                                 HloOpcode::kRecv, HloOpcode::kRecvDone}) ||
        ContainsInstrWithOpcode(while_op->while_condition(),
                                {HloOpcode::kSend, HloOpcode::kSendDone,
                                 HloOpcode::kRecv, HloOpcode::kRecvDone})) {
      VLOG(2) << "Not attempting to simplify while loop because it contains a "
                 "send/recv node: "
              << while_op->ToShortString();
      continue;
    }

    TF_ASSIGN_OR_RETURN(bool result, TryPropagateConstant(while_op));
    changed |= result;

    TF_ASSIGN_OR_RETURN(result, TryRemoveWhileLoop(while_op));
    changed |= result;
    if (result) {
      // Don't continue simplifying after successfully removing the while loop
      // -- that would result in use-after-free nastiness.
      continue;
    }

    // TODO(b/119281462): Cowardly refuse to perform any of the following
    // optimizations in the presence of kDomain instructions.  It seems that
    // modifying a while loop's tuple doesn't work when kDomain is present.
    if (ContainsInstrWithOpcode(while_op->while_body(), {HloOpcode::kDomain}) ||
        ContainsInstrWithOpcode(while_op->while_condition(),
                                {HloOpcode::kDomain})) {
      continue;
    }

    // Each of the optimizations below modifies the while loop itself if it's
    // successful, meaning that `while_op` is no longer valid after one of these
    // transformations returns true.

    TF_ASSIGN_OR_RETURN(result, TryFlattenNestedTuples(while_op));
    changed |= result;
    if (result) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(result, TryRemoveDeadWhileParams(while_op));
    changed |= result;
    if (result) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(result, TryRemoveConstantParams(while_op));
    changed |= result;
    if (result) {
      continue;
    }

    bool merged_induction_vars = false;
    // Notably missing from this list are S16 and U16.  These don't currently
    // work because S/U16 literals are not implemented.
    for (auto elem_ty : {S8, U8, S32, U32, S64, U64}) {
      TF_ASSIGN_OR_RETURN(auto* new_while_op,
                          TryMergeInductionVariables(while_op, elem_ty));
      if (new_while_op) {
        while_op = new_while_op;
        changed = true;
        merged_induction_vars = true;
      }
    }
    if (merged_induction_vars) {
      continue;
    }
  }

  XLA_VLOG_LINES(3,
                 "WhileLoopSimplifier::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
