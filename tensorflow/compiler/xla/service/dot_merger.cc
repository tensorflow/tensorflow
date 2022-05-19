/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/dot_merger.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/graphcycles/graphcycles.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"

namespace xla {
namespace {

// Determines whether the given list is [0, 1, ..., n].
bool IsIota(const tensorflow::protobuf::RepeatedField<int64_t>& vals) {
  for (int i = 0; i < vals.size(); i++) {
    if (vals[i] != i) {
      return false;
    }
  }
  return true;
}

// Determines if this dot is canonical according to DotDecomposer's rules.  The
// LHS and RHS must have
//
//  - batch dimensions [0,1,...] at the beginning, followed by
//  - one non-contracting dimension and one contracting dimension, in either
//    order.
//
// (Note: DotDecomposer doesn't guarantee that the LHS contracting dim is the
// last dim or the RHS contracting dim is the second-to-last.)
bool IsCanonicalDot(const HloInstruction* dot) {
  if (dot->opcode() != HloOpcode::kDot) {
    return false;
  }

  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  CHECK_EQ(dnums.lhs_batch_dimensions_size(),
           dnums.rhs_batch_dimensions_size());
  int64_t batch_size = dnums.lhs_batch_dimensions_size();
  return IsIota(dnums.lhs_batch_dimensions()) &&
         dnums.lhs_contracting_dimensions_size() == 1 &&
         dot->operand(0)->shape().dimensions_size() == batch_size + 2 &&
         IsIota(dnums.rhs_batch_dimensions()) &&
         dnums.rhs_contracting_dimensions_size() == 1 &&
         dot->operand(1)->shape().dimensions_size() == batch_size + 2;
}

std::tuple<int64_t /*m*/, int64_t /*k*/, int64_t /*n*/> GetDotMKN(
    const HloInstruction* dot) {
  CHECK(IsCanonicalDot(dot));

  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();

  int64_t lhs_contracting_dim = dnums.lhs_contracting_dimensions()[0];
  int64_t lhs_outer_dim =
      lhs_contracting_dim == dnums.lhs_batch_dimensions_size()
          ? lhs_contracting_dim + 1
          : lhs_contracting_dim - 1;

  int64_t rhs_contracting_dim = dnums.rhs_contracting_dimensions()[0];
  int64_t rhs_outer_dim =
      rhs_contracting_dim == dnums.rhs_batch_dimensions_size()
          ? rhs_contracting_dim + 1
          : rhs_contracting_dim - 1;

  const Shape& lhs_shape = dot->operand(0)->shape();
  const Shape& rhs_shape = dot->operand(1)->shape();
  return {
      lhs_shape.dimensions(lhs_outer_dim),
      lhs_shape.dimensions(lhs_contracting_dim),
      rhs_shape.dimensions(rhs_outer_dim),
  };
}

bool CommonMergeabilityChecks(HloInstruction* a, HloInstruction* b) {
  if (a->operand(0)->shape().element_type() !=
          b->operand(0)->shape().element_type() ||
      a->operand(1)->shape().element_type() !=
          b->operand(1)->shape().element_type() ||
      a->shape().element_type() != b->shape().element_type()) {
    VLOG(3)
        << "Can't merge dots because their lhs/rhs/return-types don't match.\n"
        << "\t" << a->ToString() << "\n"
        << "\t" << b->ToString();
    return false;
  }

  if (!absl::c_equal(a->precision_config().operand_precision(),
                     b->precision_config().operand_precision())) {
    VLOG(3) << "Can't merge dots because they have mismatching operand "
               "precisions:\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString();
    return false;
  }

  return true;
}

StatusOr<HloInstruction*> AddDegenerateDimAt(HloInstruction* instr,
                                             absl::optional<int64_t> dim) {
  if (!dim.has_value()) return instr;
  absl::InlinedVector<int64_t, 8> dims(instr->shape().dimensions().begin(),
                                       instr->shape().dimensions().end());
  dims.insert(dims.begin() + *dim, 1);
  return MakeReshapeHlo(dims, instr);
}

StatusOr<HloInstruction*> RemoveDegenerateDimAt(HloInstruction* instr,
                                                absl::optional<int64_t> dim) {
  if (!dim.has_value()) return instr;
  absl::InlinedVector<int64_t, 8> dims(instr->shape().dimensions().begin(),
                                       instr->shape().dimensions().end());
  CHECK_EQ(dims[*dim], 1);
  dims.erase(dims.begin() + *dim);
  return MakeReshapeHlo(dims, instr);
}

// Tries to merge dot instructions a and b if they share an operand.  Example:
//
//   lhs = f32[200,100] parameter(0)
//   rhs0 = f32[100,10] parameter(1)
//   rhs1 = f32[100,50] parameter(2)
//   dot0 = f32[200,10] dot(lhs, rhs0), lhs_contracting_dims={1},
//   rhs_contracting_dims={0} dot1 = f32[200,50] dot(lhs, rhs1),
//   lhs_contracting_dims={1}, rhs_contracting_dims={0}
//
// can be merged to
//
//   dot = f32[200,60] dot(lhs, concat(rhs0, lhs1))
//   dot0 = slice(dot)
//   dot1 = slice(dot)
//
// Preconditions:
//  - `a` and `b` are canonical dots.
//  - `a` does not transitively depend on the value of `b`, and `b` does not
//    transitively depend on the value of `a`.
//
StatusOr<HloInstruction*> TryMergeSameOperand(HloInstruction* a,
                                              HloInstruction* b) {
  if (a->operand(0) != b->operand(0) && a->operand(1) != b->operand(1)) {
    VLOG(4) << "Can't merge dots because they don't share an operand.\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString();
    return nullptr;
  }

  if (!CommonMergeabilityChecks(a, b)) {
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
            << "\t" << b->ToString();
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

  // Dimension along which we're going to concatenate diff_op_a and diff_op_b.
  // This is the outer (i.e. non-contracing) dim.  Because the dot is canonical,
  // we know that the dimensions are
  //
  //  [batch_dims ..., outer/contracting dim, contracting/outer dim].
  //
  CHECK_EQ(dnums.lhs_batch_dimensions_size(),
           dnums.rhs_batch_dimensions_size());
  int64_t contracting_dim = (lhs_same ? dnums.rhs_contracting_dimensions()
                                      : dnums.lhs_contracting_dimensions())[0];
  int64_t outer_dim = contracting_dim == dnums.lhs_batch_dimensions_size()
                          ? contracting_dim + 1
                          : contracting_dim - 1;

  HloComputation* comp = a->parent();
  TF_ASSIGN_OR_RETURN(
      Shape concat_shape,
      ShapeInference::InferConcatOpShape(
          {&diff_op_a->shape(), &diff_op_b->shape()}, outer_dim));
  HloInstruction* concat_op =
      comp->AddInstruction(HloInstruction::CreateConcatenate(
          concat_shape, {diff_op_a, diff_op_b}, outer_dim));

  HloInstruction* dot_lhs = lhs_same ? shared_op : concat_op;
  HloInstruction* dot_rhs = lhs_same ? concat_op : shared_op;
  TF_ASSIGN_OR_RETURN(
      Shape new_dot_shape,
      ShapeInference::InferDotOpShape(
          dot_lhs->shape(), dot_rhs->shape(), dnums,
          /*preferred_element_type=*/a->shape().element_type()));
  HloInstruction* new_dot = comp->AddInstruction(HloInstruction::CreateDot(
      new_dot_shape, dot_lhs, dot_rhs, dnums, a->precision_config()));

  // Slice the outputs.
  absl::InlinedVector<int64_t, 6> start_indices(new_dot_shape.dimensions_size(),
                                                0);
  absl::InlinedVector<int64_t, 6> limit_indices(
      new_dot_shape.dimensions().begin(), new_dot_shape.dimensions().end());
  absl::InlinedVector<int64_t, 6> strides(new_dot_shape.dimensions_size(), 1);

  int64_t slice_dim = new_dot_shape.dimensions_size() - (lhs_same ? 1 : 2);
  limit_indices[slice_dim] = a->shape().dimensions(slice_dim);
  // Important: We do RAUW, not ReplaceInstruction, because the old instruction
  // must live until the end of the pass.
  HloInstruction* new_a = comp->AddInstruction(HloInstruction::CreateSlice(
      a->shape(), new_dot, start_indices, limit_indices, strides));
  TF_RETURN_IF_ERROR(a->ReplaceAllUsesWith(new_a));
  new_a->set_metadata(a->metadata());

  start_indices[slice_dim] = limit_indices[slice_dim];
  limit_indices[slice_dim] = new_dot_shape.dimensions(slice_dim);
  HloInstruction* new_b = comp->AddInstruction(HloInstruction::CreateSlice(
      b->shape(), new_dot, start_indices, limit_indices, strides));
  TF_RETURN_IF_ERROR(b->ReplaceAllUsesWith(new_b));
  new_b->set_metadata(b->metadata());

  return new_dot;
}

StatusOr<HloInstruction*> TryMergeToBatchDot(HloInstruction* a,
                                             HloInstruction* b) {
  if (!CommonMergeabilityChecks(a, b)) {
    return nullptr;
  }

  const DotDimensionNumbers& dnums_a = a->dot_dimension_numbers();
  const DotDimensionNumbers& dnums_b = b->dot_dimension_numbers();
  const int64_t a_batch_rank = dnums_a.lhs_batch_dimensions().size();
  const int64_t b_batch_rank = dnums_b.lhs_batch_dimensions().size();

  // Cowardly refuse to merge if the dots' contracting dims don't match.
  // (Because the dots are canonical and only have one outer dim each, this also
  // implies that the outer dims match.)
  if (dnums_a.lhs_contracting_dimensions()[0] - a_batch_rank !=
          dnums_b.lhs_contracting_dimensions()[0] - b_batch_rank ||
      dnums_a.rhs_contracting_dimensions()[0] - a_batch_rank !=
          dnums_b.rhs_contracting_dimensions()[0] - b_batch_rank) {
    VLOG(3) << "Can't merge dots because their contracting dims are "
               "incompatible\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString();
    return nullptr;
  }

  // Figure out which batch dim we're going to concatenate the dots' LHSs
  // along.  There are three cases:
  //
  //  - Neither gemm has any batch dimensions.  In this case, we add a size-1
  //    dim at the front of both of the gemms and concat along that dim.
  //
  //  - Both gemms have the same number of batch dims.  In this case, we require
  //    that the batch dim sizes all be equal with the possible exception of one
  //    dim; we concatenate along that one.
  //
  //  - Gemm X has one more batch dim than gemm Y, but all of the other batch
  //    dims are equal.  In this case, we add a degenerate dim to Y in place of
  //    the missing dim and concatenate along that one.
  //
  absl::optional<int64_t> concat_dim;
  absl::optional<int64_t> a_add_dim_at;
  absl::optional<int64_t> b_add_dim_at;
  // When we concatenate along concat_dim, `a`'s results end and `b`'s results
  // begin at index slice_limit_a.
  int64_t slice_limit_a = -1;
  if (a_batch_rank == 0 && b_batch_rank == 0) {
    concat_dim = 0;
    a_add_dim_at = 0;
    b_add_dim_at = 0;
    slice_limit_a = 1;
  } else if (a_batch_rank == b_batch_rank) {
    for (int64_t i = 0; i < a_batch_rank; i++) {
      if (a->shape().dimensions(i) == b->shape().dimensions(i)) {
        continue;
      }
      // Found a mismatch.  If it's our second one, that's an error, we're done
      // here.
      if (concat_dim.has_value()) {
        VLOG(3)
            << "Can't merge dots because their batch dims are incompatible\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString();
        return nullptr;
      }
      concat_dim = i;
    }
    // If all of the dims are equal, arbitrarily concatenate along dim 0.
    if (!concat_dim.has_value()) {
      concat_dim = 0;
    }
    slice_limit_a = a->shape().dimensions(*concat_dim);
  } else if (std::abs(a_batch_rank - b_batch_rank) == 1) {
    // In this case the dims must all be equal, but we allow one dim to be
    // missing from the smaller shape.  Maybe there's a clever algorithm for
    // this, but we just brute-force it and try all possible missing dims.
    bool a_is_larger = a_batch_rank > b_batch_rank;
    int64_t large_rank = a_is_larger ? a_batch_rank : b_batch_rank;
    int64_t small_rank = a_is_larger ? b_batch_rank : a_batch_rank;
    const Shape& large_shape = a_is_larger ? a->shape() : b->shape();
    const Shape& small_shape = a_is_larger ? b->shape() : a->shape();
    for (int64_t cand = 0; cand < large_rank; cand++) {
      bool ok = true;
      absl::optional<int64_t> missing_dim;
      int64_t small_idx = 0;
      for (int64_t large_idx = 0;
           large_idx < large_rank && small_idx < small_rank; large_idx++) {
        if (large_shape.dimensions(large_idx) ==
            small_shape.dimensions(small_idx)) {
          small_idx++;
          continue;
        }
        if (missing_dim.has_value()) {
          ok = false;
          break;
        }
        missing_dim = large_idx;
      }
      if (ok) {
        concat_dim = missing_dim.value_or(small_rank);
        // In this dimension we're concat'ing either [x,1] (if `a` is the one
        // with more dims) or [1,x].
        slice_limit_a = a_is_larger ? a->shape().dimensions(*concat_dim) : 1;
        (a_is_larger ? b_add_dim_at : a_add_dim_at) = *concat_dim;
      }
    }
  }
  if (!concat_dim.has_value()) {
    VLOG(3) << "Can't merge dots because their batch dims are incompatible\n"
            << "\t" << a->ToString() << "\n"
            << "\t" << b->ToString();
    return nullptr;
  }

  VLOG(2) << "Merging dots into a single batch-dot:\n"
          << "\t" << a->ToString() << "\n"
          << "\t" << b->ToString();

  TF_ASSIGN_OR_RETURN(HloInstruction * lhs_a,
                      AddDegenerateDimAt(a->mutable_operand(0), a_add_dim_at));
  TF_ASSIGN_OR_RETURN(HloInstruction * rhs_a,
                      AddDegenerateDimAt(a->mutable_operand(1), a_add_dim_at));
  TF_ASSIGN_OR_RETURN(HloInstruction * lhs_b,
                      AddDegenerateDimAt(b->mutable_operand(0), b_add_dim_at));
  TF_ASSIGN_OR_RETURN(HloInstruction * rhs_b,
                      AddDegenerateDimAt(b->mutable_operand(1), b_add_dim_at));

  TF_ASSIGN_OR_RETURN(HloInstruction * new_lhs,
                      MakeConcatHlo({lhs_a, lhs_b}, *concat_dim));
  TF_ASSIGN_OR_RETURN(HloInstruction * new_rhs,
                      MakeConcatHlo({rhs_a, rhs_b}, *concat_dim));

  const int64_t new_batch_rank =
      dnums_a.lhs_batch_dimensions_size() + (a_add_dim_at.has_value() ? 1 : 0);
  DotDimensionNumbers new_dnums;
  for (int64_t i = 0; i < new_batch_rank; i++) {
    new_dnums.add_lhs_batch_dimensions(i);
    new_dnums.add_rhs_batch_dimensions(i);
  }
  new_dnums.add_lhs_contracting_dimensions(
      dnums_a.lhs_contracting_dimensions()[0] - a_batch_rank + new_batch_rank);
  new_dnums.add_rhs_contracting_dimensions(
      dnums_a.rhs_contracting_dimensions()[0] - a_batch_rank + new_batch_rank);

  CHECK(tensorflow::AreSerializedProtosEqual(a->precision_config(),
                                             b->precision_config()));
  CHECK_EQ(a->shape().element_type(), b->shape().element_type());
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_dot,
      MakeDotHlo(new_lhs, new_rhs, new_dnums, a->precision_config(),
                 /*preferred_element_type=*/a->shape().element_type()));

  // Slice the results.
  int64_t new_rank = new_batch_rank + 2;
  absl::InlinedVector<int64_t, 8> slice_starts(/*n=*/new_rank, /*v=*/0);
  absl::InlinedVector<int64_t, 8> slice_limits(
      new_dot->shape().dimensions().begin(),
      new_dot->shape().dimensions().end());
  absl::InlinedVector<int64_t, 8> slice_strides(/*n=*/new_rank, 1);

  slice_limits[*concat_dim] = slice_limit_a;
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_a,
      MakeSliceHlo(new_dot, slice_starts, slice_limits, slice_strides));

  slice_starts[*concat_dim] = slice_limits[*concat_dim];
  slice_limits[*concat_dim] = new_dot->shape().dimensions(*concat_dim);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_b,
      MakeSliceHlo(new_dot, slice_starts, slice_limits, slice_strides));

  TF_ASSIGN_OR_RETURN(new_a, RemoveDegenerateDimAt(new_a, a_add_dim_at));
  TF_ASSIGN_OR_RETURN(new_b, RemoveDegenerateDimAt(new_b, b_add_dim_at));

  // Important: We do RAUW, not ReplaceInstruction, because the old instruction
  // must live until the end of the pass.
  TF_RETURN_IF_ERROR(a->ReplaceAllUsesWith(new_a));
  TF_RETURN_IF_ERROR(b->ReplaceAllUsesWith(new_b));
  return new_dot;
}

StatusOr<bool> MergeDots(HloComputation* comp, int64_t max_size_to_merge) {
  auto is_merge_candidate = [&](HloInstruction* instr) {
    int64_t bytes = ShapeUtil::ByteSizeOfElements(instr->shape());
    for (const HloInstruction* operand : instr->operands()) {
      bytes += ShapeUtil::ByteSizeOfElements(operand->shape());
    }
    return bytes <= max_size_to_merge;
  };

  // Collect equivalence classes.  We build two sets of equivalence classes,
  // because we do two kinds of merging, same-operand and batch-dot merging.
  // Specifically, create the mappings
  //
  //   - operand_classes: instruction -> [dots that use the instruction], and
  //   - mkn_classes: (m, k, n) -> [dots with those shapes],
  //
  // where m,k,n are for an [m x k] x [k x n] -> [m x n] dot.
  //
  // We'll then try to merge dots within each equivalence class.  A dot will be
  // a member of three equivalence classes (two operands plus one mkn), but if
  // it's merged with a dot from one equivalence class, it won't also be merged
  // in another class.
  absl::flat_hash_map<HloInstruction* /*operand*/,
                      absl::flat_hash_set<HloInstruction*>>
      operand_classes;
  absl::flat_hash_map<std::tuple<int64_t, int64_t, int64_t> /*mkn*/,
                      absl::flat_hash_set<HloInstruction*>>
      mkn_classes;

  for (HloInstruction* instr : comp->instructions()) {
    // Cowardly skip instructions with control dependencies.
    if (!IsCanonicalDot(instr) || !instr->control_predecessors().empty() ||
        !instr->control_successors().empty()) {
      continue;
    }
    for (HloInstruction* operand : instr->operands()) {
      operand_classes[operand].insert(instr);
    }
    mkn_classes[GetDotMKN(instr)].insert(instr);
  }

  // Gather all the equivalence classes, ignoring "uninteresting" equivalence
  // classes where either
  //
  //  - there's just one instruction (nothing to merge!), or
  //  - there are zero instructions marked as mergeable.  (Our contract is that
  //    at least one instruction of the pair needs to be mergeable in order for
  //    us to merge.)
  //
  // Sort the equivalence classes and the elements in each class for
  // determinism.
  enum EquivalenceClassKind {
    kOperand,
    kMKN,
  };
  std::vector<std::pair<EquivalenceClassKind, std::vector<HloInstruction*>>>
      sorted_ecs;
  auto collect_ec = [&](const absl::flat_hash_set<HloInstruction*>& ec)
      -> std::vector<HloInstruction*> {
    if (ec.size() < 2 || absl::c_none_of(ec, is_merge_candidate)) {
      return {};
    }
    std::vector<HloInstruction*> sorted_ec(ec.begin(), ec.end());
    absl::c_sort(sorted_ec, HloPtrComparator());
    return sorted_ec;
  };
  for (const auto& kv : operand_classes) {
    std::vector<HloInstruction*> ec = collect_ec(kv.second);
    if (!ec.empty()) {
      sorted_ecs.push_back({kOperand, std::move(ec)});
    }
  }
  for (const auto& kv : mkn_classes) {
    std::vector<HloInstruction*> ec = collect_ec(kv.second);
    if (!ec.empty()) {
      sorted_ecs.push_back({kMKN, std::move(ec)});
    }
  }
  absl::c_sort(
      sorted_ecs,
      [](const std::pair<EquivalenceClassKind, std::vector<HloInstruction*>>& a,
         const std::pair<EquivalenceClassKind, std::vector<HloInstruction*>>&
             b) {
        if (a.first != b.first) {
          return a.first < b.first;
        }
        const auto& a_instrs = a.second;
        const auto& b_instrs = b.second;
        if (a_instrs.size() != b_instrs.size()) {
          return a_instrs.size() < b_instrs.size();
        }
        HloPtrComparator cmp;
        for (int i = 0; i < a_instrs.size(); i++) {
          if (a_instrs[i] != b_instrs[i]) {
            return cmp(a_instrs[i], b_instrs[i]);
          }
        }
        return false;
      });

  // Are there any possible optimization opportunities?
  if (sorted_ecs.empty()) {
    VLOG(2) << "No nontrivial dot equivalence classes to try merging.";
    return false;
  }

  // Build a dependency graph representing the whole computation.
  //
  // TODO(jlebar): If this is slow to create or use, could we make it faster by
  // collapsing elements of the graph that don't correspond to dots, or
  // otherwise not adding them to the graph in the first place?
  tensorflow::GraphCycles graph;

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

  for (HloInstruction* instr : comp->instructions()) {
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
  for (auto& ec_and_kind : sorted_ecs) {
    EquivalenceClassKind kind = ec_and_kind.first;
    auto& ec = ec_and_kind.second;

    // Try merging all pairs of dots in this equivalence class.
    for (int64_t i = 0; i < ec.size(); i++) {
      HloInstruction*& a = ec[i];
      if (a == nullptr) {
        continue;
      }
      for (int64_t j = i + 1; j < ec.size(); j++) {
        HloInstruction* b = ec[j];
        if (b == nullptr) {
          continue;
        }

        // At least one of `a` and `b` must be a merge candidate.
        if (!is_merge_candidate(a) && !is_merge_candidate(b)) {
          continue;
        }

        int32_t a_id = graph_id(a);
        int32_t b_id = graph_id(b);

        if (dead_instrs.contains(a) || dead_instrs.contains(b) ||
            graph.IsReachableNonConst(a_id, b_id) ||
            graph.IsReachableNonConst(b_id, a_id) ||
            (!is_merge_candidate(a) && !is_merge_candidate(b))) {
          continue;
        }

        HloInstruction* merged = nullptr;
        switch (kind) {
          case kOperand: {
            TF_ASSIGN_OR_RETURN(merged, TryMergeSameOperand(a, b));
            break;
          }
          case kMKN: {
            TF_ASSIGN_OR_RETURN(merged, TryMergeToBatchDot(a, b));
            break;
          }
        }
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
          ec[i] = merged;
          ec[j] = nullptr;
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

StatusOr<bool> DotMerger::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(bool changed_computation,
                        MergeDots(comp, max_size_to_merge_));
    changed |= changed_computation;
  }
  return changed;
}

}  // namespace xla
