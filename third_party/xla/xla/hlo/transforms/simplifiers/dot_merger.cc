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

#include "xla/hlo/transforms/simplifiers/dot_merger.h"

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/protobuf_util.h"
#include "xla/service/graphcycles/graphcycles.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace m = ::xla::match;

// Tries to merge dot instructions a and b if they share an operand.  Example:
//
//   lhs = f32[200,100] parameter(0)
//   rhs0 = f32[100,10] parameter(1)
//   rhs1 = f32[100,50] parameter(2)
//   dot0 = f32[200,10] dot(lhs, rhs0),
//     lhs_contracting_dims={1}, rhs_contracting_dims={0}
//   dot1 = f32[200,50] dot(lhs, rhs1),
//     lhs_contracting_dims={1}, rhs_contracting_dims={0}
//
// can be merged to
//
//   dot = f32[200,60] dot(lhs, concat(rhs0, lhs1))
//   dot0 = slice(dot)
//   dot1 = slice(dot)
//
// Preconditions:
//  - `a` and `b` are dots.
//  - `a` does not transitively depend on the value of `b`, and `b` does not
//    transitively depend on the value of `a`.
//
absl::StatusOr<HloInstruction*> TryMergeSameOperand(HloInstruction* a,
                                                    HloInstruction* b) {
  if (a->operand(0) != b->operand(0) && a->operand(1) != b->operand(1)) {
    VLOG(4) << "Can't merge dots because they don't share an operand.\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString();
    return nullptr;
  }

  if (a->operand(0)->shape().element_type() !=
          b->operand(0)->shape().element_type() ||
      a->operand(1)->shape().element_type() !=
          b->operand(1)->shape().element_type() ||
      a->shape().element_type() != b->shape().element_type()) {
    VLOG(3)
        << "Can't merge dots because their lhs/rhs/return-types don't match.\n"
        << "\t" << a->ToString() << "\n"
        << "\t" << b->ToString();
    return nullptr;
  }

  const DotDimensionNumbers& dnums_a = a->dot_dimension_numbers();
  const DotDimensionNumbers& dnums_b = b->dot_dimension_numbers();
  if (!absl::c_equal(dnums_a.lhs_batch_dimensions(),
                     dnums_b.lhs_batch_dimensions()) ||
      !absl::c_equal(dnums_a.rhs_batch_dimensions(),
                     dnums_b.rhs_batch_dimensions()) ||
      !absl::c_equal(dnums_a.lhs_contracting_dimensions(),
                     dnums_b.lhs_contracting_dimensions()) ||
      !absl::c_equal(dnums_a.rhs_contracting_dimensions(),
                     dnums_b.rhs_contracting_dimensions())) {
    VLOG(3) << "Can't merge dots because they have mismatching dnums.\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString() << "\n"
            << absl::c_equal(dnums_a.lhs_batch_dimensions(),
                             dnums_b.lhs_batch_dimensions())
            << ", "
            << absl::c_equal(dnums_a.rhs_batch_dimensions(),
                             dnums_b.rhs_batch_dimensions())
            << ", "
            << absl::c_equal(dnums_a.lhs_contracting_dimensions(),
                             dnums_b.lhs_contracting_dimensions())
            << ", "
            << absl::c_equal(dnums_a.rhs_contracting_dimensions(),
                             dnums_b.rhs_contracting_dimensions());
    return nullptr;
  }


  VLOG(2) << "Merging dots sharing an operand:\n"
          << "\t" << a->ToString() << "\n"
          << "\t" << b->ToString();

  // At this point we have dnums_a == dnums_b.  Rename to just `dnums`.
  const DotDimensionNumbers& dnums = a->dot_dimension_numbers();

  // At this point, either the LHS'es are the same, or the RHS'es are the same.
  bool lhs_same = a->operand(0) == b->operand(0);
  HloInstruction* shared_op = a->mutable_operand(lhs_same ? 0 : 1);
  HloInstruction* diff_op_a = a->mutable_operand(lhs_same ? 1 : 0);
  HloInstruction* diff_op_b = b->mutable_operand(lhs_same ? 1 : 0);
  if (diff_op_a->shape().layout() != diff_op_b->shape().layout()) {
    VLOG(3) << "Can't merge dots because the different operands have a "
               "different layout:\n"
            << "\t" << diff_op_a->ToString() << "\n"
            << "\t" << diff_op_b->ToString();
    return nullptr;
  }

  // Dimension along which we're going to concatenate diff_op_a and diff_op_b.
  // We only support the case where there is exactly one non-contracting
  // dimension. We can find it by collecting all other dimensions in a set, and
  // then picking the first dimension which is not in the set.
  CHECK_EQ(dnums.lhs_batch_dimensions_size(),
           dnums.rhs_batch_dimensions_size());
  std::set<int64_t> used_dims;
  int64_t shared_op_num_non_contracting_dims =
      shared_op->shape().dimensions().size() -
      dnums.lhs_batch_dimensions_size();
  if (lhs_same) {
    shared_op_num_non_contracting_dims -=
        dnums.lhs_contracting_dimensions_size();
    used_dims.insert(dnums.rhs_contracting_dimensions().begin(),
                     dnums.rhs_contracting_dimensions().end());
    used_dims.insert(dnums.rhs_batch_dimensions().begin(),
                     dnums.rhs_batch_dimensions().end());
  } else {
    shared_op_num_non_contracting_dims -=
        dnums.rhs_contracting_dimensions_size();
    used_dims.insert(dnums.lhs_contracting_dimensions().begin(),
                     dnums.lhs_contracting_dimensions().end());
    used_dims.insert(dnums.lhs_batch_dimensions().begin(),
                     dnums.lhs_batch_dimensions().end());
  }
  if (used_dims.size() + 1 != diff_op_a->shape().dimensions().size()) {
    VLOG(3)
        << "Can't merge dots because the different operands don't have exactly "
           "one non-contracting dimension:\n"
        << "\t" << a->ToString() << "\n"
        << "\t" << b->ToString();
    return nullptr;
  }
  int64_t outer_dim = 0;
  for (auto used_dim : used_dims) {
    if (used_dim != outer_dim) {
      break;
    }
    ++outer_dim;
  }

  HloDotInstruction* dot_a = Cast<HloDotInstruction>(a);
  std::vector<SparsityDescriptor> sparsity(dot_a->sparsity().begin(),
                                           dot_a->sparsity().end());
  std::vector<HloInstruction*> sparse_meta(sparsity.size());
  for (int i = 0; i < sparsity.size(); ++i) {
    HloInstruction* meta = a->mutable_operand(HloDotInstruction::kOperands + i);
    HloInstruction* other_meta =
        b->mutable_operand(HloDotInstruction::kOperands + i);
    if (sparsity[i].index() == (lhs_same ? 1 : 0)) {
      TF_ASSIGN_OR_RETURN(
          Shape meta_concat_shape,
          ShapeInference::InferConcatOpShape(
              {&meta->shape(), &other_meta->shape()}, outer_dim));
      meta = meta->AddInstruction(HloInstruction::CreateConcatenate(
          meta_concat_shape, {meta, other_meta}, outer_dim));
    } else {
      if (other_meta != meta) {
        VLOG(3)
            << "Can't merge dots because the sparsity metadata is different:\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString();
        return nullptr;
      }
    }
    sparse_meta[i] = meta;
  }

  TF_ASSIGN_OR_RETURN(
      Shape concat_shape,
      ShapeInference::InferConcatOpShape(
          {&diff_op_a->shape(), &diff_op_b->shape()}, outer_dim));
  *concat_shape.mutable_layout() = diff_op_a->shape().layout();
  HloInstruction* concat_op =
      diff_op_a->AddInstruction(HloInstruction::CreateConcatenate(
          concat_shape, {diff_op_a, diff_op_b}, outer_dim));

  HloInstruction* dot_lhs = lhs_same ? shared_op : concat_op;
  HloInstruction* dot_rhs = lhs_same ? concat_op : shared_op;
  TF_ASSIGN_OR_RETURN(
      Shape new_dot_shape,
      ShapeInference::InferDotOpShape(
          dot_lhs->shape(), dot_rhs->shape(), dnums,
          /*preferred_element_type=*/a->shape().element_type(), sparsity));
  *new_dot_shape.mutable_layout() = a->shape().layout();
  HloInstruction* new_dot = a->AddInstruction(
      HloInstruction::CreateDot(new_dot_shape, dot_lhs, dot_rhs, dnums,
                                a->precision_config(), sparsity, sparse_meta));

  // We can't keep both. But one is better then none.
  if (!a->metadata().op_name().empty()) {
    new_dot->set_metadata(a->metadata());
  } else if (!b->metadata().op_name().empty()) {
    new_dot->set_metadata(b->metadata());
  }

  // Slice the outputs.
  DimensionVector start_indices(new_dot_shape.dimensions().size(), 0);
  DimensionVector limit_indices(new_dot_shape.dimensions().begin(),
                                new_dot_shape.dimensions().end());
  DimensionVector strides(new_dot_shape.dimensions().size(), 1);

  int64_t slice_dim = new_dot_shape.dimensions().size() -
                      (lhs_same ? 1 : 1 + shared_op_num_non_contracting_dims);
  limit_indices[slice_dim] = a->shape().dimensions(slice_dim);
  // Important: We do RAUW, not ReplaceInstruction, because the old instruction
  // must live until the end of the pass.
  HloInstruction* new_a = a->AddInstruction(HloInstruction::CreateSlice(
      a->shape(), new_dot, start_indices, limit_indices, strides));
  TF_RETURN_IF_ERROR(a->ReplaceAllUsesWith(new_a));

  start_indices[slice_dim] = limit_indices[slice_dim];
  limit_indices[slice_dim] = new_dot_shape.dimensions(slice_dim);
  HloInstruction* new_b = b->AddInstruction(HloInstruction::CreateSlice(
      b->shape(), new_dot, start_indices, limit_indices, strides));
  TF_RETURN_IF_ERROR(b->ReplaceAllUsesWith(new_b));

  return new_dot;
}

bool EqualTransposed(HloInstruction* lhs, HloInstruction* rhs) {
  HloInstruction* transpose;
  return (Match(lhs,
                m::Transpose(&transpose).WithOperand(0, m::Op().Is(rhs))) ||
          Match(rhs,
                m::Transpose(&transpose).WithOperand(0, m::Op().Is(lhs)))) &&
         (transpose->dimensions().size() == 2 &&
          transpose->dimensions().at(0) == 1 &&
          transpose->dimensions().at(1) == 0);
}

// Try to merge dots that have a common operand on different sides (LHS vs RHS).
// Example:
//
// a = [m, n_a] dot([m, k] common, [k, n_a] rhs_a),
//     lhs_contracting_dims={1}, rhs_contracting_dims={0}
// common_T = [k, m] transpose(common)
// b = [n_b, m] dot([n_b, k] lhs_b, [k, m] common_T),
//     lhs_contracting_dims={1}, rhs_contracting_dims={0}
//
// is merged to:
//
// lhs_b_T = [k, n_b] transpose([n_b, k] lhs_b), dimensions={1,0}
// concat = [k, n_a+n_b] concat([k, n_a] rhs_a, [k, n_b] lhs_b_T))
// merged_dot = [m, n_a+n_b] dot([m, k] common, [k, n_a+n_b] concat),
//     lhs_contracting_dims={1}, rhs_contracting_dims={0}
// slice_b = [m, n_b] slice(merged_dot), slice={[0:m], [n_a:n_a+n_b]}
// new_a = [m, n_a] slice(merged_dot), slice={[0:m], [0:n_a]}
// new_b = [n_b, m] transpose(slice_b), dimensions={1,0}
//
// Preconditions:
//  - `a` and `b` are dots in the `DotDecomposer` normal form.
//  - `a` and `b` do not have batch dimensions, thus all operands are of rank 2.
//  - `a` does not transitively depend on the value of `b`, and `b` does not
//    transitively depend on the value of `a`.

absl::StatusOr<HloInstruction*> TryMergeLHSWithRHSOperand(HloInstruction* a,
                                                          HloInstruction* b) {
  if (!Cast<HloDotInstruction>(a)->sparsity().empty() ||
      !Cast<HloDotInstruction>(b)->sparsity().empty()) {
    VLOG(3) << "Merging sparse dots is not supported:\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString();
    return nullptr;
  }

  const DotDimensionNumbers& dnums_a = a->dot_dimension_numbers();
  const DotDimensionNumbers& dnums_b = b->dot_dimension_numbers();
  // TODO(tjoerg): Add support for batch dimensions.
  if (!dnums_a.lhs_batch_dimensions().empty() ||
      !dnums_a.rhs_batch_dimensions().empty() ||
      !dnums_b.lhs_batch_dimensions().empty() ||
      !dnums_b.rhs_batch_dimensions().empty()) {
    VLOG(3) << "Can't merge dots with batch dimensions.\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString()
            << absl::c_equal(dnums_a.lhs_batch_dimensions(),
                             dnums_b.lhs_batch_dimensions())
            << ", "
            << absl::c_equal(dnums_a.rhs_batch_dimensions(),
                             dnums_b.rhs_batch_dimensions());
    return nullptr;
  }

  HloInstruction* a_rhs;
  HloInstruction* a_lhs;
  if (!Match(a, m::Dot(m::Op(&a_lhs).WithShape(m::Shape().WithRank(2)),
                       m::Op(&a_rhs).WithShape(m::Shape().WithRank(2)))
                    .WithContractingDims({1}, {0}))) {
    VLOG(3) << "Only dots in DotDecomposer normal can be merged."
            << "\t" << a->ToString();
    return nullptr;
  }
  HloInstruction* b_lhs;
  HloInstruction* b_rhs;
  if (!Match(b, m::Dot(m::Op(&b_lhs).WithShape(m::Shape().WithRank(2)),
                       m::Op(&b_rhs).WithShape(m::Shape().WithRank(2)))
                    .WithContractingDims({1}, {0}))) {
    VLOG(3) << "Only dots in DotDecomposer normal can be merged."
            << "\t" << b->ToString();
    return nullptr;
  }

  if (a->shape().element_type() != b->shape().element_type() ||
      a_rhs->shape().element_type() != b_lhs->shape().element_type()) {
    VLOG(3) << "Can't merge dots because operand/return types are different:\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString();
    return nullptr;
  }

  if (a->shape().dimensions(0) != b->shape().dimensions(1)) {
    VLOG(3) << "Can't merge dots because their `m` dimension is different:\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString();
    return nullptr;
  }

  // Check that the LHS of `a` is equivalent to the RHS of `b` just transposed.
  // DotDecomposer insert transpose ops to establish a normal form and ensure
  // contracting dims are most major on the LHS and non-contracting dimensions
  // are most major on the RHS.
  if (!EqualTransposed(a_lhs, b_rhs)) {
    VLOG(4) << "LHS of first dot is different from RHS of second dot.\n"
            << "\t" << a_lhs->ToString() << "\n"
            << "\t" << b_rhs->ToString();
    return nullptr;
  }

  VLOG(2) << "Merging dots sharing an operand:\n"
          << "\t" << a->ToString() << "\n"
          << "\t" << b->ToString() << "\t" << a->parent()->ToString();

  // The new RHS is the concatenation of a's RHS and b's LHS transposed.
  HloInstruction* b_lhs_transposed =
      b_lhs->AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::PermuteDimensions({1, 0}, b_lhs->shape()), b_lhs, {1, 0}));
  TF_ASSIGN_OR_RETURN(Shape concat_shape,
                      ShapeInference::InferConcatOpShape(
                          {&a_rhs->shape(), &b_lhs_transposed->shape()}, 1));
  HloInstruction* new_rhs =
      a_rhs->AddInstruction(HloInstruction::CreateConcatenate(
          concat_shape, {a_rhs, b_lhs_transposed}, 1));
  TF_ASSIGN_OR_RETURN(
      Shape new_dot_shape,
      ShapeInference::InferDotOpShape(
          a_lhs->shape(),  // The new LHS is the LHS of a.
          new_rhs->shape(), dnums_a,
          /*preferred_element_type=*/a->shape().element_type()));
  *new_dot_shape.mutable_layout() = a->shape().layout();
  HloInstruction* new_dot = a->AddInstruction(HloInstruction::CreateDot(
      new_dot_shape, a_lhs, new_rhs, dnums_a, a->precision_config()));

  // We can't keep both. But one is better then none.
  if (!a->metadata().op_name().empty()) {
    new_dot->set_metadata(a->metadata());
  } else if (!b->metadata().op_name().empty()) {
    new_dot->set_metadata(b->metadata());
  }

  // Slice the outputs.
  HloInstruction* new_a = a->AddInstruction(HloInstruction::CreateSlice(
      a->shape(), new_dot, /*start_indices=*/{0, 0},
      /*limit_indices=*/{a->shape().dimensions(0), a->shape().dimensions(1)},
      /*strides=*/{1, 1}));
  HloInstruction* new_b_slice = b->AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::PermuteDimensions({1, 0}, b->shape()), new_dot,
      /*start_indices=*/{0, a->shape().dimensions(1)},
      /*limit_indices=*/
      {b->shape().dimensions(1),
       a->shape().dimensions(1) + b->shape().dimensions(0)},
      /*strides=*/{1, 1}));
  HloInstruction* new_b = new_b_slice->AddInstruction(
      HloInstruction::CreateTranspose(b->shape(), new_b_slice, {1, 0}));
  // Important: We do RAUW, not ReplaceInstruction, because the old
  // instruction must live until the end of the pass.
  TF_RETURN_IF_ERROR(a->ReplaceAllUsesWith(new_a));
  TF_RETURN_IF_ERROR(b->ReplaceAllUsesWith(new_b));

  return new_dot;
}

absl::StatusOr<HloInstruction*> TryMergeOperand(HloInstruction* a,
                                                HloInstruction* b) {
  if (a->shape().layout() != b->shape().layout()) {
    VLOG(3) << "Can't merge dots because they have a different layout:\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString();
    return nullptr;
  }

  if (!absl::c_equal(a->precision_config().operand_precision(),
                     b->precision_config().operand_precision())) {
    VLOG(3) << "Can't merge dots because they have mismatching operand "
               "precisions:\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString();
    return nullptr;
  }

  HloDotInstruction* dot_a = Cast<HloDotInstruction>(a);
  HloDotInstruction* dot_b = Cast<HloDotInstruction>(b);
  if (!absl::c_equal(dot_a->sparsity(), dot_b->sparsity(),
                     protobuf_util::HaveSameSerialization)) {
    VLOG(3) << "Can't merge dots because they have mismatching sparsity "
               "descriptors:\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString();
    return nullptr;
  }

  auto merged = TryMergeSameOperand(a, b);
  if (!merged.ok() || merged.value() != nullptr) {
    return merged;
  }
  // Merging on different sides is symmetric, try `a` and `b` on both sides.
  merged = TryMergeLHSWithRHSOperand(a, b);
  if (!merged.ok() || merged.value() != nullptr) {
    return merged;
  }
  return TryMergeLHSWithRHSOperand(b, a);
}

absl::StatusOr<bool> MergeDots(HloComputation* comp, int64_t max_size_to_merge,
                               std::function<bool(const HloInstruction* dot_a,
                                                  const HloInstruction* dot_b)>
                                   can_merge) {
  auto is_merge_candidate = [&](HloInstruction* instr) {
    int64_t bytes = ShapeUtil::ByteSizeOfElements(instr->shape());
    for (const HloInstruction* operand : instr->operands()) {
      bytes += ShapeUtil::ByteSizeOfElements(operand->shape());
    }
    return bytes <= max_size_to_merge;
  };

  // Collect equivalence classes.  Specifically, create the map
  //
  //   instruction -> [canonical dots that use the instruction].
  //
  // We'll then try to merge dots within each equivalence class.  A dot will be
  // a member of two equivalence classes (because it has two operands), but if
  // it's merged with a dot from one equivalence class, it won't also be merged
  // in another class.
  absl::flat_hash_map<HloInstruction*, absl::flat_hash_set<HloInstruction*>>
      equivalence_classes;
  for (HloInstruction* instr : comp->instructions()) {
    // Cowardly skip instructions with control dependencies.
    if (instr->opcode() != HloOpcode::kDot ||
        !instr->control_predecessors().empty() ||
        !instr->control_successors().empty()) {
      continue;
    }
    for (HloInstruction* operand : instr->operands()) {
      equivalence_classes[operand].insert(instr);
      // DotDecomposer inserts transposes to establish a normal form. Transposed
      // operands still count as equivalent.
      if (operand->opcode() == HloOpcode::kTranspose) {
        equivalence_classes[operand->mutable_operand(0)].insert(instr);
      }
    }
  }

  // Remove "uninteresting" equivalence classes where either
  //
  //  - there's just one instruction (nothing to merge!), or
  //  - there are zero instructions marked as mergeable.  (Our contract is that
  //    at least one instruction of the pair needs to be mergeable in order for
  //    us to merge.)
  absl::erase_if(
      equivalence_classes,
      [&](const std::pair<const HloInstruction*,
                          absl::flat_hash_set<HloInstruction*>>& kv) {
        const auto& v = kv.second;
        return v.size() < 2 || absl::c_none_of(v, is_merge_candidate);
      });

  // Are there any possible optimization opportunities?
  if (equivalence_classes.empty()) {
    return false;
  }

  // Build a dependency graph representing the whole computation.
  GraphCycles graph;

  absl::flat_hash_map<HloInstruction*, int32_t> graph_ids_map;
  auto graph_id = [&](HloInstruction* instr) {
    auto it_and_inserted = graph_ids_map.emplace(instr, -1);
    auto it = it_and_inserted.first;
    auto inserted = it_and_inserted.second;
    if (inserted) {
      it->second = graph.NewNode();
    }
    return it->second;
  };

  // Iteration order doesn't matter for correctness, but graph.InsertEdge() is
  // *much* faster if we iterate in topological order.
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    int32_t id = graph_id(instr);
    for (HloInstruction* operand : instr->operands()) {
      CHECK(graph.InsertEdge(graph_id(operand), id));
    }
    for (HloInstruction* control_pred : instr->control_predecessors()) {
      CHECK(graph.InsertEdge(graph_id(control_pred), id));
    }
  }

  // Merge within equivalence classes.  We keep a set of all instructions that
  // have been merged so we don't try to merge an instruction twice.  We'll
  // remove these dead instructions at the end of the pass.  (We can't remove
  // them earlier because removing an instruction deletes it; we'd then have
  // dangling pointers in our hashtable!)
  absl::flat_hash_set<HloInstruction*> dead_instrs;
  std::vector<HloInstruction*> keys;
  keys.reserve(equivalence_classes.size());
  for (auto& kv : equivalence_classes) {
    keys.push_back(kv.first);
  }
  absl::c_sort(keys, [](const HloInstruction* a, const HloInstruction* b) {
    return a->unique_id() < b->unique_id();
  });
  for (auto key : keys) {
    const auto& values = equivalence_classes[key];
    // For determinism, iterate in order of the instructions' IDs.
    absl::InlinedVector<HloInstruction*, 16> dots(values.begin(), values.end());
    absl::c_sort(dots, [](const HloInstruction* a, const HloInstruction* b) {
      return a->unique_id() < b->unique_id();
    });

    // Try merging all pairs of dots in this equivalence class.
    for (int64_t i = 0; i < dots.size(); i++) {
      HloInstruction*& a = dots[i];
      if (a == nullptr) {
        continue;
      }
      for (int64_t j = i + 1; j < dots.size(); j++) {
        HloInstruction* b = dots[j];
        if (b == nullptr) {
          continue;
        }
        int32_t a_id = graph_id(a);
        int32_t b_id = graph_id(b);

        if (dead_instrs.contains(a) || dead_instrs.contains(b) ||
            (!is_merge_candidate(a) && !is_merge_candidate(b)) ||
            // Perform reachability checks last since they can be expensive.
            graph.IsReachableNonConst(a_id, b_id) ||
            graph.IsReachableNonConst(b_id, a_id) || !can_merge(a, b)) {
          continue;
        }

        TF_ASSIGN_OR_RETURN(HloInstruction * merged, TryMergeOperand(a, b));
        if (merged != nullptr) {
          int32_t merged_id = graph_id(merged);
          graph.InsertEdge(a_id, merged_id);
          graph.InsertEdge(b_id, merged_id);
          for (int32_t succ : graph.SuccessorsCopy(a_id)) {
            graph.InsertEdge(merged_id, succ);
          }
          for (int32_t succ : graph.SuccessorsCopy(b_id)) {
            graph.InsertEdge(merged_id, succ);
          }

          dead_instrs.insert(a);
          dead_instrs.insert(b);
          dots[i] = merged;
          dots[j] = nullptr;
        }
      }
    }
  }

  // Now it's finally safe to delete the old instructions from the graph.
  for (HloInstruction* instr : dead_instrs) {
    TF_RETURN_IF_ERROR(comp->RemoveInstruction(instr));
  }

  return !dead_instrs.empty();
}

}  // anonymous namespace

absl::StatusOr<bool> DotMerger::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool changed_computation,
                        MergeDots(comp, max_size_to_merge_, can_merge_));
    changed |= changed_computation;
  }
  return changed;
}

}  // namespace xla
