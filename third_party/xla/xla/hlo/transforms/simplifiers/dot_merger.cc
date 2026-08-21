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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/hlo/analysis/shape_tracker.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/graphcycles/graphcycles.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

bool HasSubByteUpcast(const HloInstruction* instr) {
  while (instr != nullptr && (instr->opcode() == HloOpcode::kTranspose ||
                              instr->opcode() == HloOpcode::kReshape ||
                              instr->opcode() == HloOpcode::kBitcast ||
                              instr->opcode() == HloOpcode::kConvert)) {
    if (instr->opcode() == HloOpcode::kConvert &&
        primitive_util::IsSubByteNonPredType(
            instr->operand(0)->shape().element_type())) {
      return true;
    }
    instr = instr->operand(0);
  }
  return false;
}

using ReplacementMap = HloInstructionMap<HloInstruction*>;

// In order to be mergeable, the shared operand should have the same dimension
// categories for non-degenerate dimensions, have the same operand and output
// types, and same algorithm/precision config. And same queue_id, whatever it
// is.
struct EquivalenceKey {
  const HloInstruction* source;
  std::vector<DotOperandDims::Category> dim_categories;
  int64_t queue_id;
  PrimitiveType shared_op_type;
  PrimitiveType concat_op_type;
  PrimitiveType dot_type;
  std::string precision_config_bytes;

  template <typename H>
  friend H AbslHashValue(H h, const EquivalenceKey& k) {
    return H::combine(std::move(h), k.source, k.dim_categories, k.queue_id,
                      k.shared_op_type, k.concat_op_type, k.dot_type,
                      k.precision_config_bytes);
  }

  bool operator==(const EquivalenceKey& other) const {
    return source == other.source && dim_categories == other.dim_categories &&
           queue_id == other.queue_id &&
           shared_op_type == other.shared_op_type &&
           concat_op_type == other.concat_op_type &&
           dot_type == other.dot_type &&
           precision_config_bytes == other.precision_config_bytes;
  }

  bool operator<(const EquivalenceKey& other) const {
    int64_t id1 = source->unique_id();
    int64_t id2 = other.source->unique_id();
    auto t1 = std::tie(id1, queue_id, dim_categories, shared_op_type,
                       concat_op_type, dot_type, precision_config_bytes);
    auto t2 = std::tie(id2, other.queue_id, other.dim_categories,
                       other.shared_op_type, other.concat_op_type,
                       other.dot_type, other.precision_config_bytes);
    return t1 < t2;
  }
};

struct DotOperandUsage {
  HloInstruction* dot;
  int shared_operand_idx;  // 0 for LHS, 1 for RHS
  HloInstruction* shared_operand_source;
  HloInstruction* concat_operand_source;
  int64_t concat_nc_size;
  int tz;
};

absl::StatusOr<EquivalenceKey> GetEquivalenceKey(
    const HloInstruction* dot, const HloInstruction* operand,
    int shared_operand_idx, const HloInstruction* source,
    const DotOperandDims& dims,
    const std::function<int64_t(const HloInstruction* dot)>& queue_id) {
  DotOperandDims canonical_dims = dims;
  // Ignore degenerate dimensions because they can be freely reassigned to any
  // category.
  ABSL_RETURN_IF_ERROR(canonical_dims.RemoveDegenerateDimensions());
  std::vector<DotOperandDims::Category> dim_categories =
      canonical_dims.Categories();

  PrimitiveType shared_op_type = operand->shape().element_type();
  PrimitiveType concat_op_type =
      dot->operand(1 - shared_operand_idx)->shape().element_type();
  PrimitiveType dot_type = dot->shape().element_type();
  std::string precision_config_bytes =
      dot->precision_config().SerializeAsString();

  return EquivalenceKey{source,
                        dim_categories,
                        queue_id(dot),
                        shared_op_type,
                        concat_op_type,
                        dot_type,
                        std::move(precision_config_bytes)};
}

// Traverses backward from the operand of a dot instruction to find its source,
// tracking and mapping dimension categories step-by-step.
// Stops early (breaks) if category mixing occurs along the path, returning the
// last successfully mapped instruction and dimensions.
absl::StatusOr<std::pair<HloInstruction*, DotOperandDims>>
GetOperandSourceWithCategories(HloInstruction* dot, int operand_number) {
  HloInstruction* operand = dot->mutable_operand(operand_number);
  ABSL_ASSIGN_OR_RETURN(DotOperandDims dims,
                   DotOperandDims::FromDotOperand(dot, operand_number));
  HloInstruction* current = operand;
  while (current->opcode() == HloOpcode::kTranspose ||
         current->opcode() == HloOpcode::kReshape ||
         current->opcode() == HloOpcode::kBitcast) {
    ABSL_ASSIGN_OR_RETURN(std::optional<DotOperandDims> mapped,
                     dims.MapBackward(current));
    if (!mapped.has_value()) {
      break;  // Stop here, as moving further would mix categories.
    }
    dims = *mapped;
    current = current->mutable_operand(0);
  }
  return std::make_pair(current, std::move(dims));
}

// Builds equivalence classes for dots in the computation.
// The key is (source_instruction, dimension_categories_at_source, queue_id).
absl::StatusOr<absl::btree_map<EquivalenceKey, std::vector<DotOperandUsage>>>
BuildEquivalenceClasses(
    HloComputation* comp,
    std::function<int64_t(const HloInstruction* dot)> queue_id) {
  // Using sorted map to ensure deterministic behavior.
  absl::btree_map<EquivalenceKey, std::vector<DotOperandUsage>>
      equivalence_classes;

  for (HloInstruction* instr : comp->instructions()) {
    if (instr->opcode() != HloOpcode::kDot ||
        !instr->control_predecessors().empty() ||
        !instr->control_successors().empty() ||
        absl::c_any_of(instr->operands(), HasSubByteUpcast)) {
      continue;
    }

    // Go from operands up through the chain of transposes/reshapes/bitcasts
    // while possible. The result is the "source" of that operand.
    ABSL_ASSIGN_OR_RETURN(auto lhs, GetOperandSourceWithCategories(instr, 0));
    ABSL_ASSIGN_OR_RETURN(auto rhs, GetOperandSourceWithCategories(instr, 1));

    auto populate_class =
        [&](const std::pair<HloInstruction*, DotOperandDims>& shared,
            const std::pair<HloInstruction*, DotOperandDims>& concat,
            int shared_operand_idx) -> absl::Status {
      ABSL_ASSIGN_OR_RETURN(
          EquivalenceKey key,
          GetEquivalenceKey(instr, instr->operand(shared_operand_idx),
                            shared_operand_idx, shared.first, shared.second,
                            queue_id));

      int64_t concat_nc_size =
          concat.second.TotalSize(DotOperandDims::kNonContracting);
      int tz = absl::countr_zero(static_cast<uint64_t>(concat_nc_size));

      equivalence_classes[key].push_back({instr, shared_operand_idx,
                                          shared.first, concat.first,
                                          concat_nc_size, tz});
      return absl::OkStatus();
    };

    ABSL_RETURN_IF_ERROR(populate_class(lhs, rhs, 0));
    ABSL_RETURN_IF_ERROR(populate_class(rhs, lhs, 1));
  }

  // Sort dots ascending by non-contracting concat dimension size to be able to
  // iterate using a sliding window while keeping max/min ratio bounded.
  for (auto& [key, usages] : equivalence_classes) {
    absl::c_sort(
        usages, [](const DotOperandUsage& a, const DotOperandUsage& b) {
          return std::make_tuple(a.concat_nc_size, a.dot->unique_id()) <
                 std::make_tuple(b.concat_nc_size, b.dot->unique_id());
        });
  }

  return equivalence_classes;
}

std::vector<DotOperandUsage> GetUnmergedDots(
    absl::Span<const DotOperandUsage> dots,
    const ReplacementMap& replacements) {
  std::vector<DotOperandUsage> unmerged_dots;
  absl::c_copy_if(dots, std::back_inserter(unmerged_dots),
                  [&](const DotOperandUsage& dot_usage) {
                    return dot_usage.dot != nullptr &&
                           replacements.find(dot_usage.dot) ==
                               replacements.end();
                  });
  return unmerged_dots;
}

absl::StatusOr<HloInstruction*> CreateSharedOperand(
    HloInstruction* source, DotOperandDims& target_shared_dims) {
  ABSL_RETURN_IF_ERROR(target_shared_dims.RemoveDegenerateDimensions());

  // If there are multiple contracting dimensions, and they are consecutive,
  // combine them into one.
  if (target_shared_dims.Rank(DotOperandDims::kContracting) > 1 &&
      target_shared_dims.IsConsecutive(DotOperandDims::kContracting)) {
    ABSL_RETURN_IF_ERROR(target_shared_dims.CollapseCategory(
        DotOperandDims::kContracting, /*remove_if_empty=*/false));
  }

  if (source->shape() == target_shared_dims.shape()) {
    return source;
  }
  return source->parent()->AddInstruction(
      HloInstruction::CreateReshape(target_shared_dims.shape(), source));
}

// Given the operand shapes before/after transformation, and the dot dimensions
// before the transformation, returns the dot dimensions after the
// transformation. We only care about the dimension set, not the element order
// (we can pick any order).
absl::StatusOr<DotOperandDims> DetermineTargetSharedDims(
    const DotOperandUsage& dot_to_merge) {
  HloInstruction* source = dot_to_merge.shared_operand_source;
  const HloInstruction* dot_op =
      dot_to_merge.dot->operand(dot_to_merge.shared_operand_idx);

  ABSL_ASSIGN_OR_RETURN(ShapeTracker tracker,
                   ShapeTracker::FromSiblings(dot_op, source));
  ABSL_ASSIGN_OR_RETURN(DotOperandDims dot_dims,
                   DotOperandDims::FromDotOperand(
                       dot_to_merge.dot, dot_to_merge.shared_operand_idx));

  // Map batch and contracting dimensions.
  auto map_dims = [&](absl::Span<const int64_t> dims)
      -> absl::StatusOr<std::vector<int64_t>> {
    std::optional<std::vector<int64_t>> mapped =
        tracker.MapInputDimensionsToOutputUnordered(dims);
    if (!mapped.has_value()) {
      return absl::InternalError("Failed to map dimensions");
    }
    return *mapped;
  };
  ABSL_ASSIGN_OR_RETURN(std::vector<int64_t> final_batch_dims,
                   map_dims(dot_dims.Indices(DotOperandDims::kBatch)));
  ABSL_ASSIGN_OR_RETURN(std::vector<int64_t> final_contracting_dims,
                   map_dims(dot_dims.Indices(DotOperandDims::kContracting)));

  // Remaining dimensions are non-contracting.
  int64_t source_rank = source->shape().dimensions().size();
  std::vector<int64_t> final_non_contracting_dims;
  for (int64_t i = 0; i < source_rank; ++i) {
    if (!absl::c_linear_search(final_batch_dims, i) &&
        !absl::c_linear_search(final_contracting_dims, i)) {
      final_non_contracting_dims.push_back(i);
    }
  }

  return DotOperandDims(source->shape(), final_batch_dims,
                        final_non_contracting_dims, final_contracting_dims);
}

// Returns permutation that would sort e.g. {3,1,6} into {1,3,6}.
std::vector<int64_t> GetSortingPermutation(absl::Span<const int64_t> v) {
  std::vector<int64_t> p(v.size());
  absl::c_iota(p, 0);
  absl::c_stable_sort(p, [&](int64_t a, int64_t b) { return v[a] < v[b]; });
  return p;
}

// Suppose we have a "before" shape [a, 6, b, 35, c], with before_indices {3, 1}
// and "after" tracker with shape [x, 14, y, 2, z, 5, w], with after_indices {3,
// 1, 5}, and a tracker that somehow transforms one to another (not necessarily
// element order preserving). This function returns a tracker that converts a
// [35, 6] (sorting by before_indices) to [2, 14, 5] (sorting by after_indices).
absl::StatusOr<ShapeTracker> BuildIndicesTracker(
    absl::Span<const int64_t> before_indices, const Shape& after_shape,
    absl::Span<const int64_t> after_indices, const ShapeTracker& tracker,
    PrimitiveType output_type) {
  // 1. Narrow. Narrow() internally handles unsorted indices by prepending
  // a sorting transpose, so we can pass before_indices directly.
  ABSL_ASSIGN_OR_RETURN(ShapeTracker indices_tracker,
                   tracker.Narrow(before_indices));

  // 2. Reshape the output. The narrowed output may not automatically have the
  // correct dimensions because it cannot guess whether degenerate dimensions
  // are needed.
  std::vector<int64_t> sorted_after_indices(after_indices.begin(),
                                            after_indices.end());
  absl::c_sort(sorted_after_indices);
  std::vector<int64_t> sorted_after_sizes;
  sorted_after_sizes.reserve(sorted_after_indices.size());
  for (int64_t dim : sorted_after_indices) {
    sorted_after_sizes.push_back(after_shape.dimensions(dim));
  }
  ABSL_RETURN_IF_ERROR(indices_tracker.AppendReshape(sorted_after_sizes));

  // 3. Append transpose to sort by after_indices.
  std::vector<int64_t> p_sort_after = GetSortingPermutation(after_indices);
  std::vector<int64_t> p_append = InversePermutation(p_sort_after);
  ABSL_RETURN_IF_ERROR(indices_tracker.AppendTranspose(p_append));

  // 4. Set element type
  indices_tracker.SetElementType(output_type);
  return indices_tracker;
}

struct ConcatTargetInfo {
  DotOperandDims dims;
  ShapeTracker tracker;
};

// We have original (old) shared operand, new shared operand, and a tracker
// that maps from old to new.
// To compute new other operand (that we'll concatenate), we need to:
// - Make the same changes in batch and contracting dimensions as in the shared
//   operand.
// - Reshape non-contracting dimensions to rank 1 (as we'll going to concatenate
// over it).
// `contracting_is_minor` picks between [B, NonContracting, Contracting] and
// [B, Contracting, NonContracting]; see the caller for how it is decided.
absl::StatusOr<ConcatTargetInfo> TransformConcatDims(
    const DotOperandDims& concat_before, const DotOperandDims& shared_before,
    const DotOperandDims& shared_after, const ShapeTracker& shared_tracker,
    PrimitiveType output_type, bool contracting_is_minor) {
  // Make the transformations in the batch and contracting dimensions.
  ABSL_ASSIGN_OR_RETURN(
      ShapeTracker batch_tracker,
      BuildIndicesTracker(shared_before.Indices(DotOperandDims::kBatch),
                          shared_after.shape(),
                          shared_after.Indices(DotOperandDims::kBatch),
                          shared_tracker, output_type));
  ABSL_ASSIGN_OR_RETURN(
      ShapeTracker contracting_tracker,
      BuildIndicesTracker(shared_before.Indices(DotOperandDims::kContracting),
                          shared_after.shape(),
                          shared_after.Indices(DotOperandDims::kContracting),
                          shared_tracker, output_type));

  // Non-contracting tracker just reshapes to rank 1.
  Shape nc_shape = ShapeUtil::MakeShape(
      output_type, concat_before.Sizes(DotOperandDims::kNonContracting));
  ShapeTracker nc_tracker(nc_shape);
  int64_t total_nc_size = ShapeUtil::ElementsIn(nc_shape);
  ABSL_RETURN_IF_ERROR(nc_tracker.AppendReshape({total_nc_size}));

  ABSL_ASSIGN_OR_RETURN(
      ShapeTracker zipped_tracker,
      ShapeTracker::Zip({batch_tracker, nc_tracker, contracting_tracker}));

  // Now the tracker expects [B,N,C] as input. Prepend a transpose to use the
  // original order.
  std::vector<int64_t> p_in;
  p_in.reserve(concat_before.shape().dimensions().size());
  for (DotOperandDims::Category cat :
       {DotOperandDims::kBatch, DotOperandDims::kNonContracting,
        DotOperandDims::kContracting}) {
    absl::c_copy(concat_before.Indices(cat), std::back_inserter(p_in));
  }
  ABSL_RETURN_IF_ERROR(zipped_tracker.PrependTranspose(p_in));

  // The output is [B,N,C]
  const int64_t batch_rank_out = shared_after.Rank(DotOperandDims::kBatch);
  std::vector<int64_t> target_batch_dims(batch_rank_out);
  std::iota(target_batch_dims.begin(), target_batch_dims.end(), 0);
  std::vector<int64_t> target_nc_dims = {batch_rank_out};
  std::vector<int64_t> target_contracting_dims(
      shared_after.Rank(DotOperandDims::kContracting));
  std::iota(target_contracting_dims.begin(), target_contracting_dims.end(),
            batch_rank_out + 1);
  DotOperandDims target_dims(zipped_tracker.output_shape(), target_batch_dims,
                             target_nc_dims, target_contracting_dims);

  // Built [B,N,C] (contracting last) above. When a non-contracting dim is
  // already last, swap to [B,C,N] to keep it last. This transpose folds into
  // the alignment transpose above, staying a single fusible (loop) transpose.
  if (!contracting_is_minor) {
    std::vector<int64_t> to_ncminor(target_dims.shape().dimensions().size());
    std::iota(to_ncminor.begin(), to_ncminor.end(), 0);
    // [B, N, C] -> [B, C, N] (N is a single dim).
    std::rotate(to_ncminor.begin() + batch_rank_out,
                to_ncminor.begin() + batch_rank_out + 1, to_ncminor.end());
    ABSL_RETURN_IF_ERROR(zipped_tracker.AppendTranspose(to_ncminor));
    target_dims = target_dims.GetPermuted(to_ncminor);
  }

  return ConcatTargetInfo{target_dims, zipped_tracker};
}

absl::StatusOr<std::vector<ConcatTargetInfo>> ComputeTargetConcatDims(
    const std::vector<DotOperandUsage>& dots_to_merge,
    HloInstruction* new_shared_op, const DotOperandDims& target_shared_dims) {
  std::vector<ConcatTargetInfo> target_concat_infos;
  target_concat_infos.reserve(dots_to_merge.size());

  // Decide the order once for the class (operands must share an order to be
  // concatenable; the first dot's operand is representative). Keep the last
  // logical dimension in place to avoid a transpose. DotMerger runs before
  // layout assignment, so the last logical dimension, not a physical minor dim,
  // is what matters.
  ABSL_ASSIGN_OR_RETURN(
      DotOperandDims first_concat_dims,
      DotOperandDims::FromDotOperand(dots_to_merge[0].dot,
                                     1 - dots_to_merge[0].shared_operand_idx));
  const bool contracting_is_minor =
      first_concat_dims.Categories().back() == DotOperandDims::kContracting;

  for (const auto& usage : dots_to_merge) {
    ABSL_ASSIGN_OR_RETURN(DotOperandDims concat_dims_at_dot,
                     DotOperandDims::FromDotOperand(
                         usage.dot, 1 - usage.shared_operand_idx));
    ABSL_ASSIGN_OR_RETURN(
        DotOperandDims shared_dims_at_dot,
        DotOperandDims::FromDotOperand(usage.dot, usage.shared_operand_idx));

    HloInstruction* old_shared_operand =
        usage.dot->mutable_operand(usage.shared_operand_idx);
    // Build tracker from old to the new shared operand.
    ABSL_ASSIGN_OR_RETURN(
        ShapeTracker shared_dot_to_target,
        ShapeTracker::FromSiblings(old_shared_operand, new_shared_op));

    // Bring the concat (non-shared) operand to the canonical layout, where B
    // and C are identical for all dots, and NC is a single dimension (that
    // we'll concatenate over). The contracting dimension is placed last or
    // second-to-last according to `contracting_is_minor`.
    ABSL_ASSIGN_OR_RETURN(
        ConcatTargetInfo info,
        TransformConcatDims(concat_dims_at_dot, shared_dims_at_dot,
                            target_shared_dims, shared_dot_to_target,
                            usage.dot->shape().element_type(),
                            contracting_is_minor));

    // TransformConcatDims returns a old->new tracker for the concat operand.
    // What we actually want is a source->new tracker. To build it, we compose
    // the source->old tracker with the old->new tracker.
    HloInstruction* old_concat =
        usage.dot->mutable_operand(1 - usage.shared_operand_idx);
    ABSL_ASSIGN_OR_RETURN(ShapeTracker concat_source_to_dot,
                     ShapeTracker::FromProducerConsumer(
                         usage.concat_operand_source, old_concat));
    ABSL_RETURN_IF_ERROR(concat_source_to_dot.ConcatenateFrom(info.tracker));
    info.tracker = std::move(concat_source_to_dot);
    target_concat_infos.push_back(info);
  }
  return target_concat_infos;
}

absl::StatusOr<std::vector<HloInstruction*>> CreateMergedConcatOperands(
    const std::vector<DotOperandUsage>& dots_to_merge,
    const std::vector<ConcatTargetInfo>& target_concat_infos) {
  std::vector<HloInstruction*> transformed_concat_ops;
  transformed_concat_ops.reserve(dots_to_merge.size());
  for (auto [usage, info] : llvm::zip(dots_to_merge, target_concat_infos)) {
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * new_concat_op,
        info.tracker.ToInstructionChain(usage.concat_operand_source));
    transformed_concat_ops.push_back(new_concat_op);
  }
  return transformed_concat_ops;
}

absl::StatusOr<HloInstruction*> CreateConcatenatedOperand(
    HloInstruction* new_shared_op,
    const std::vector<HloInstruction*>& transformed_concat_ops,
    int64_t concat_dim) {
  std::vector<HloInstruction*> concat_operands(transformed_concat_ops.begin(),
                                               transformed_concat_ops.end());

  std::vector<const Shape*> operand_shapes;
  operand_shapes.reserve(concat_operands.size());
  for (auto* op : concat_operands) {
    operand_shapes.push_back(&op->shape());
  }

  ABSL_ASSIGN_OR_RETURN(Shape concat_shape, ShapeInference::InferConcatOpShape(
                                           operand_shapes, concat_dim));

  return new_shared_op->AddInstruction(HloInstruction::CreateConcatenate(
      concat_shape, concat_operands, concat_dim));
}

absl::StatusOr<HloInstruction*> CreateNewDot(
    HloInstruction* dot_lhs, HloInstruction* dot_rhs, bool target_is_lhs,
    const DotOperandDims& target_shared_dims,
    const DotOperandDims& target_concat_dims,
    const std::vector<DotOperandUsage>& dots_to_merge) {
  ABSL_ASSIGN_OR_RETURN(
      DotDimensionNumbers dnums,
      DotOperandDims::CreateDotDimensionNumbers(
          target_is_lhs ? target_shared_dims : target_concat_dims,
          target_is_lhs ? target_concat_dims : target_shared_dims));

  ABSL_ASSIGN_OR_RETURN(Shape new_dot_shape,
                   ShapeInference::InferDotOpShape(
                       dot_lhs->shape(), dot_rhs->shape(), dnums,
                       /*preferred_element_type=*/
                       dots_to_merge[0].dot->shape().element_type()));

  HloInstruction* new_dot = dot_lhs->AddInstruction(
      HloInstruction::CreateDot(new_dot_shape, dot_lhs, dot_rhs, dnums,
                                dots_to_merge[0].dot->precision_config()));

  auto it = absl::c_find_if(dots_to_merge, [](const DotOperandUsage& usage) {
    return !usage.dot->metadata().op_name().empty();
  });
  if (it != dots_to_merge.end()) {
    new_dot->set_metadata(it->dot->metadata());
  }

  return new_dot;
}

absl::StatusOr<ShapeTracker> BuildOutputShapeTracker(
    const DotOperandDims& shared_before, const DotOperandDims& shared_after,
    const ShapeTracker& shared_tracker_inv, const DotOperandDims& concat_before,
    const DotOperandDims& concat_after, const ShapeTracker& concat_tracker_inv,
    bool shared_is_lhs, bool orig_shared_is_lhs, PrimitiveType output_type) {
  // Undo the batch dimension changes (they are the same for shared and concat).
  ABSL_ASSIGN_OR_RETURN(
      ShapeTracker batch_shared,
      BuildIndicesTracker(shared_after.Indices(DotOperandDims::kBatch),
                          shared_before.shape(),
                          shared_before.Indices(DotOperandDims::kBatch),
                          shared_tracker_inv, output_type));
  ABSL_ASSIGN_OR_RETURN(
      ShapeTracker batch_concat,
      BuildIndicesTracker(concat_after.Indices(DotOperandDims::kBatch),
                          concat_before.shape(),
                          concat_before.Indices(DotOperandDims::kBatch),
                          concat_tracker_inv, output_type));
  if (!(batch_shared == batch_concat)) {
    // This should never happen, but if it would, it would lead to a wrong
    // result rather than a crash, so let's keep this check.
    return absl::InvalidArgumentError(
        "Shared and Concat batch trackers are not equal");
  }

  // Undo the non-contracting dimension changes.
  ABSL_ASSIGN_OR_RETURN(ShapeTracker shared_nc,
                   BuildIndicesTracker(
                       shared_after.Indices(DotOperandDims::kNonContracting),
                       shared_before.shape(),
                       shared_before.Indices(DotOperandDims::kNonContracting),
                       shared_tracker_inv, output_type));
  ABSL_ASSIGN_OR_RETURN(ShapeTracker concat_nc,
                   BuildIndicesTracker(
                       concat_after.Indices(DotOperandDims::kNonContracting),
                       concat_before.shape(),
                       concat_before.Indices(DotOperandDims::kNonContracting),
                       concat_tracker_inv, output_type));

  ABSL_ASSIGN_OR_RETURN(
      ShapeTracker tracker,
      shared_is_lhs ? ShapeTracker::Zip({batch_shared, shared_nc, concat_nc})
                    : ShapeTracker::Zip({batch_shared, concat_nc, shared_nc}));

  // If lhs vs rhs were swapped, swap them back.
  if (orig_shared_is_lhs != shared_is_lhs) {
    const int64_t batch_rank = shared_before.Rank(DotOperandDims::kBatch);
    std::vector<int64_t> perm(tracker.output_shape().dimensions().size());
    std::iota(perm.begin(), perm.end(), 0);
    std::rotate(perm.begin() + batch_rank,
                perm.begin() + batch_rank +
                    (orig_shared_is_lhs
                         ? concat_before.Rank(DotOperandDims::kNonContracting)
                         : shared_before.Rank(DotOperandDims::kNonContracting)),
                perm.end());
    ABSL_RETURN_IF_ERROR(tracker.AppendTranspose(perm));
  }

  return tracker;
}

absl::Status SliceAndReplaceOutput(
    HloInstruction* new_dot, HloInstruction* new_shared_op,
    HloInstruction* pre_concat_op, const DotOperandUsage& usage,
    bool shared_is_lhs, const DotOperandDims& target_shared_dims,
    const DotOperandDims& target_concat_dims, int64_t& slice_start,
    ReplacementMap& replacements) {
  HloInstruction* original_dot = usage.dot;
  const Shape& new_dot_shape = new_dot->shape();

  // 1. Slice from the new dot result.
  // The dot output shape is [batch, lhs_nc, rhs_nc], we slice over the
  // non-shared non-contracting dimension. Concat dimension is always singular,
  // to take either the last or the one that follows the batch.
  const int64_t slice_dim =
      shared_is_lhs ? new_dot_shape.dimensions().size() - 1
                    : target_shared_dims.Rank(DotOperandDims::kBatch);

  const int64_t slice_size =
      target_concat_dims.TotalSize(DotOperandDims::kNonContracting);

  DimensionVector start_indices(new_dot_shape.dimensions().size(), 0);
  DimensionVector limit_indices(new_dot_shape.dimensions().begin(),
                                new_dot_shape.dimensions().end());
  DimensionVector strides(new_dot_shape.dimensions().size(), 1);

  start_indices[slice_dim] = slice_start;
  limit_indices[slice_dim] = slice_start + slice_size;

  Shape slice_shape = new_dot_shape;
  slice_shape.set_dimensions(slice_dim, slice_size);

  HloInstruction* slice = new_dot->AddInstruction(HloInstruction::CreateSlice(
      slice_shape, new_dot, start_indices, limit_indices, strides));

  // 2. Now we have required elements, but they may be in wrong order.
  // We need to "undo" the transformations we did to our operands.
  ABSL_ASSIGN_OR_RETURN(
      ShapeTracker shared_tracker_inv,
      ShapeTracker::FromSiblings(new_shared_op, original_dot->mutable_operand(
                                                    usage.shared_operand_idx)));
  ABSL_ASSIGN_OR_RETURN(ShapeTracker concat_tracker_inv,
                   ShapeTracker::FromSiblings(
                       pre_concat_op, original_dot->mutable_operand(
                                          1 - usage.shared_operand_idx)));
  ABSL_ASSIGN_OR_RETURN(
      DotOperandDims old_shared_dims,
      DotOperandDims::FromDotOperand(original_dot, usage.shared_operand_idx));
  ABSL_ASSIGN_OR_RETURN(DotOperandDims old_concat_dims,
                   DotOperandDims::FromDotOperand(
                       original_dot, 1 - usage.shared_operand_idx));
  ABSL_ASSIGN_OR_RETURN(ShapeTracker output_tracker,
                   BuildOutputShapeTracker(
                       old_shared_dims, target_shared_dims, shared_tracker_inv,
                       old_concat_dims, target_concat_dims, concat_tracker_inv,
                       shared_is_lhs,
                       /*orig_shared_is_lhs=*/(usage.shared_operand_idx == 0),
                       original_dot->shape().element_type()));

  // 3. Materialize the tracker chain and update the replacements map.
  ABSL_ASSIGN_OR_RETURN(HloInstruction * result,
                   output_tracker.ToInstructionChain(slice));
  replacements[original_dot] = result;

  slice_start += slice_size;
  return absl::OkStatus();
}

void UpdateGraphForMergedDot(
    HloInstruction* new_dot, const std::vector<DotOperandUsage>& dots_to_merge,
    GraphCycles& graph,
    const std::function<int32_t(const HloInstruction*)>& graph_id) {
  int32_t new_dot_id = graph_id(new_dot);
  for (const auto& usage : dots_to_merge) {
    int32_t dot_id = graph_id(usage.dot);
    graph.InsertEdge(dot_id, new_dot_id);
    for (int32_t succ : graph.SuccessorsCopy(dot_id)) {
      graph.InsertEdge(new_dot_id, succ);
    }
  }
}

absl::StatusOr<std::vector<DotOperandUsage>> FindDotsToMerge(
    absl::Span<const DotOperandUsage> compatible_dots, GraphCycles& graph,
    const std::function<int32_t(const HloInstruction*)>& graph_id,
    int64_t max_size_to_merge) {
  // Number of flops that take the same time as a write of a single byte to HBM.
  // This depends on the GPU a lot, but 290 is what we on a typical modern
  // GPU (as of 2026).
  constexpr double kFlopsPerByte = 290.0;
  // Merging dots which are too dissimilar tend to result in a net slowdown,
  // e.g. due to different optimal tiling. Therefore we only merge dots where
  // number of elements differ by at most this factor.
  constexpr int64_t kMaxNonContractingRatio = 20;
  // If the savings ratio (initial runtime / new runtime)-1 is less than this,
  // don't merge. The win is so small that it's likely to be outweighed by other
  // arbitrary factors.
  constexpr double kMinSavingsRatio = 1e-5;

  if (compatible_dots.size() < 2) {
    return std::vector<DotOperandUsage>{};
  }

  const DotOperandUsage& first = compatible_dots.front();
  ABSL_ASSIGN_OR_RETURN(
      DotOperandDims shared_dims,
      DotOperandDims::FromDotOperand(first.dot, first.shared_operand_idx));
  const int64_t batch = shared_dims.TotalSize(DotOperandDims::Category::kBatch);
  const int64_t shared_nc =
      shared_dims.TotalSize(DotOperandDims::Category::kNonContracting);
  const int64_t contracting =
      shared_dims.TotalSize(DotOperandDims::Category::kContracting);

  const HloInstruction* shared_op =
      first.dot->operand(first.shared_operand_idx);
  const HloInstruction* concat_op =
      first.dot->operand(1 - first.shared_operand_idx);

  const int64_t shared_op_bytes =
      ShapeUtil::ByteSizeOfElements(shared_op->shape());
  const int64_t concat_and_out_bytes_per_nc =
      batch * (contracting * ShapeUtil::ByteSizeOfPrimitiveType(
                                 concat_op->shape().element_type()) +
               shared_nc * ShapeUtil::ByteSizeOfPrimitiveType(
                               first.dot->shape().element_type()));

  // The cost of a dot (before or after the merge) per non-contracting
  // element, in "HBM transfer bytes" (but also includes flops by dividing by
  // kFlopsPerByte).
  const double cost_per_nc =
      (2.0 * batch * shared_nc * contracting / kFlopsPerByte) +
      concat_and_out_bytes_per_nc;

  auto is_connected = [&](const HloInstruction* a, const HloInstruction* b) {
    int32_t id1 = graph_id(a), id2 = graph_id(b);
    return graph.IsReachableNonConst(id1, id2) ||
           graph.IsReachableNonConst(id2, id1);
  };

  // Try all dots as seeds, and append dots of increasing size (as they come
  // sorted by non-contracting dimension size), skipping those that are
  // connected to already chosen dots. Stop when the next dot would exceed the
  // max size. Return as soon as we find a group of 2 or more.
  for (size_t i = 0; i < compatible_dots.size(); ++i) {
    const DotOperandUsage& first_dot = compatible_dots[i];
    std::vector<DotOperandUsage> group = {first_dot};
    int64_t sum_nc = first_dot.concat_nc_size;
    const int64_t max_allowed_nc = sum_nc * kMaxNonContractingRatio;

    for (size_t j = i + 1; j < compatible_dots.size(); ++j) {
      if (concat_and_out_bytes_per_nc * sum_nc > max_size_to_merge) {
        break;
      }

      const DotOperandUsage& cand_dot = compatible_dots[j];
      if (cand_dot.concat_nc_size > max_allowed_nc) {
        break;  // Subsequent elements violate the kMaxNonContractingRatio.
      }

      if (absl::c_any_of(group, [&](const DotOperandUsage& member) {
            return is_connected(member.dot, cand_dot.dot);
          })) {
        continue;
      }

      const int64_t next_sum = sum_nc + cand_dot.concat_nc_size;
      const double sum_runtime =
          cost_per_nc * next_sum + (group.size() + 1) * shared_op_bytes;
      if (shared_op_bytes * group.size() / sum_runtime <= kMinSavingsRatio) {
        continue;
      }

      group.push_back(cand_dot);
      sum_nc = next_sum;
    }

    if (group.size() >= 2) {
      // Both Triton emitter and gemm fusion builder are happier when there's
      // less misaligned access. To make them enjoy life as much as possible,
      // let's sort the operands so that the ones that are multiples of higher
      // powers of 2 are first.
      absl::c_sort(
          group, [](const DotOperandUsage& a, const DotOperandUsage& b) {
            // We want tz and size descending, but unique_id ascending, so swap
            // a and b for two first fields.
            return std::make_tuple(b.tz, b.concat_nc_size, a.dot->unique_id()) <
                   std::make_tuple(a.tz, a.concat_nc_size, b.dot->unique_id());
          });
      return group;
    }
  }

  return std::vector<DotOperandUsage>{};
}

absl::StatusOr<HloInstruction*> MergeCluster(
    const EquivalenceKey& key, absl::Span<const DotOperandUsage> dots,
    GraphCycles& graph,
    const std::function<int32_t(const HloInstruction*)>& graph_id,
    ReplacementMap& replacements, int64_t max_size_to_merge) {
  std::vector<DotOperandUsage> compatible_dots =
      GetUnmergedDots(dots, replacements);
  if (compatible_dots.size() < 2) {
    return nullptr;
  }

  ABSL_ASSIGN_OR_RETURN(
      std::vector<DotOperandUsage> dots_to_merge,
      FindDotsToMerge(compatible_dots, graph, graph_id, max_size_to_merge));

  if (dots_to_merge.size() < 2) {
    return nullptr;
  }

  // The shared operand of the new dot will be lhs or rhs, depending on which
  // side it appeared more often in the original dots (rhs if tied).
  const bool shared_is_lhs =
      dots_to_merge.size() < 2 * absl::c_count_if(
                                     dots_to_merge,
                                     [](const DotOperandUsage& usage) {
                                       return usage.shared_operand_idx == 0;
                                     });

  if (VLOG_IS_ON(3)) {
    std::vector<absl::string_view> dot_names;
    dot_names.reserve(dots_to_merge.size());
    for (const auto& usage : dots_to_merge) {
      dot_names.push_back(usage.dot->name());
    }
    VLOG(3) << "Merging " << dots_to_merge.size() << " dots: ["
            << absl::StrJoin(dot_names, ", ") << "] with shared operand "
            << (shared_is_lhs ? "LHS" : "RHS");
  }

  // Use the common ancestor (called "source" in this code) as the new shared
  // operand. We derive target dimensions from the first dot, sorting dimensions
  // within categories (they are the same for all dots anyway as we only care
  // about categories and not the actual element order).
  ABSL_ASSIGN_OR_RETURN(DotOperandDims target_shared_dims,
                   DetermineTargetSharedDims(dots_to_merge[0]));
  ABSL_ASSIGN_OR_RETURN(HloInstruction * new_shared_op,
                   CreateSharedOperand(dots_to_merge[0].shared_operand_source,
                                       target_shared_dims));

  // Get compatible pre-concat shapes for the non-shared operands. They must
  // have exactly same shapes/categories, except for one non-contracting
  // dimension which we'll concatenate over. This function brings them to a
  // common canonical layout with exactly one Non-Contracting dimension.
  ABSL_ASSIGN_OR_RETURN(std::vector<ConcatTargetInfo> target_concat_infos,
                   ComputeTargetConcatDims(dots_to_merge, new_shared_op,
                                           target_shared_dims));
  TF_RET_CHECK(!target_concat_infos.empty());

  // Materialize the source->new trackers into the chains of instructions, which
  // we'll then concatenate.
  ABSL_ASSIGN_OR_RETURN(
      std::vector<HloInstruction*> transformed_concat_ops,
      CreateMergedConcatOperands(dots_to_merge, target_concat_infos));

  // The dimension we'll concatenate over. ComputeTargetConcatDims() returns
  // exactly one non-contracting dimension, we pick that one.
  const int64_t concat_dim =
      target_concat_infos[0].dims.Indices(DotOperandDims::kNonContracting)[0];

  // Concatenated operand for the new dot.
  ABSL_ASSIGN_OR_RETURN(HloInstruction * concatenated_concat_op,
                   CreateConcatenatedOperand(
                       new_shared_op, transformed_concat_ops, concat_dim));

  // Make the new dot.
  HloInstruction* dot_lhs =
      shared_is_lhs ? new_shared_op : concatenated_concat_op;
  HloInstruction* dot_rhs =
      shared_is_lhs ? concatenated_concat_op : new_shared_op;
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * new_dot,
      CreateNewDot(dot_lhs, dot_rhs, shared_is_lhs, target_shared_dims,
                   target_concat_infos[0].dims, dots_to_merge));
  VLOG(3) << "Created new merged dot: " << new_dot->name() << " with shape "
          << new_dot->shape().ToString();
  UpdateGraphForMergedDot(new_dot, dots_to_merge, graph, graph_id);

  // Restore the original output shapes from the new dot, through slice,
  // transposes and reshapes.
  int64_t slice_start = 0;
  for (size_t i = 0; i < dots_to_merge.size(); ++i) {
    ABSL_RETURN_IF_ERROR(SliceAndReplaceOutput(
        new_dot, new_shared_op, transformed_concat_ops[i], dots_to_merge[i],
        shared_is_lhs, target_shared_dims, target_concat_infos[i].dims,
        slice_start, replacements));
  }

  return new_dot;
}

absl::Status BuildDependencyGraph(
    HloComputation* comp, GraphCycles& graph,
    const std::function<int32_t(const HloInstruction*)>& graph_id) {
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    int32_t id = graph_id(instr);
    for (const HloInstruction* operand : instr->operands()) {
      if (!graph.InsertEdge(graph_id(operand), id)) {
        return absl::InternalError("Failed to insert edge in dependency graph");
      }
    }
    for (const HloInstruction* control_pred : instr->control_predecessors()) {
      if (!graph.InsertEdge(graph_id(control_pred), id)) {
        return absl::InternalError("Failed to insert edge in dependency graph");
      }
    }
  }
  return absl::OkStatus();
}

absl::Status SimplifyConsumerChain(HloInstruction* chain_start) {
  if (chain_start->operand_count() != 1) {
    return absl::OkStatus();
  }

  // In our case, "producer" is always a slice (doesn't matter).
  HloInstruction* producer = chain_start->mutable_operand(0);
  ShapeTracker tracker(producer->shape());
  std::vector<HloInstruction*> chain;
  HloInstruction* curr = chain_start;
  // Build a chain of transposes/reshapes/bitcasts while they have one user.
  while (curr->opcode() == HloOpcode::kTranspose ||
         curr->opcode() == HloOpcode::kReshape ||
         curr->opcode() == HloOpcode::kBitcast) {
    if (!curr->control_predecessors().empty() ||
        !curr->control_successors().empty()) {
      break;
    }
    ABSL_RETURN_IF_ERROR(tracker.AppendInstruction(curr));
    chain.push_back(curr);
    if (curr->user_count() != 1) {
      break;
    }
    curr = curr->users()[0];
  }

  if (chain.empty()) {
    return absl::OkStatus();
  }

  // Spawn a new chain of instructions.
  ABSL_ASSIGN_OR_RETURN(HloInstruction * new_chain_root,
                   tracker.ToInstructionChain(producer));

  // Replace the last instruction in the chain with the new chain root.
  HloInstruction* last_inst = chain.back();
  ABSL_RETURN_IF_ERROR(last_inst->ReplaceAllUsesWith(new_chain_root));

  // Clean up the old chain.
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    ABSL_RETURN_IF_ERROR(producer->parent()->RemoveInstruction(*it));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> MergeDots(
    HloComputation* comp, int64_t max_size_to_merge,
    std::function<int64_t(const HloInstruction* dot)> queue_id) {
  // Collect equivalence classes.  Specifically, create the map
  //
  //   instruction, dimension_categories_at_source, queue_id ->
  //        -> [canonical dots that use the instruction].
  //
  // queue_id is backend-specific. Dots with different queue_ids may run
  // concurrently on different streams and will not be merged.
  //
  // We'll then try to merge dots within each equivalence class.  A dot will be
  // a member of two equivalence classes (because it has two operands), but if
  // it's merged with a dot from one equivalence class, it won't also be merged
  // in another class.
  ABSL_ASSIGN_OR_RETURN(auto equivalence_classes,
                   BuildEquivalenceClasses(comp, queue_id));

  // Remove "uninteresting" equivalence classes where there's just one
  // instruction (nothing to merge!).
  absl::erase_if(
      equivalence_classes,
      [&](const std::pair<const EquivalenceKey, std::vector<DotOperandUsage>>&
              kv) { return kv.second.size() < 2; });

  // Are there any possible optimization opportunities?
  if (equivalence_classes.empty()) {
    return false;
  }

  if (VLOG_IS_ON(3)) {
    VLOG(3) << "Initial equivalence classes:";
    for (const auto& [key, values] : equivalence_classes) {
      VLOG(3) << "  Equivalence class for source: " << key.source->name()
              << ", shared_op_type: " << PrimitiveType_Name(key.shared_op_type)
              << ", concat_op_type: " << PrimitiveType_Name(key.concat_op_type)
              << ", dot_type: " << PrimitiveType_Name(key.dot_type)
              << ", queue_id: " << key.queue_id;
      VLOG(3) << "    Members:";
      for (const auto& usage : values) {
        VLOG(3) << "      " << usage.dot->name() << " (operand "
                << usage.shared_operand_idx << ")";
      }
    }
  }

  VLOG(1) << "Merging Dots in computation: " << comp->name();
  VLOG(1) << "Found " << equivalence_classes.size()
          << " equivalence classes with "
          << std::accumulate(equivalence_classes.begin(),
                             equivalence_classes.end(), std::uint64_t{0},
                             [](std::uint64_t total, auto const& values) {
                               return values.second.size() + total;
                             })
          << " dots in total.";

  // Build a dependency graph representing the whole computation.
  GraphCycles graph;

  absl::flat_hash_map<const HloInstruction*, int32_t> graph_ids_map;
  auto graph_id = [&](const HloInstruction* instr) {
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
  ABSL_RETURN_IF_ERROR(BuildDependencyGraph(comp, graph, graph_id));

  // Merge within equivalence classes.  We keep a map of all dot->new root
  // replacements. We'll apply all replacements at the end of the pass.
  ReplacementMap replacements;
  std::vector<HloInstruction*> merged_dots;

  for (const auto& [key, values] : equivalence_classes) {
    // Try merging all dots in this equivalence class repeatedly until no more
    // merging is possible. We may have several iterations because dots of one
    // cluster may have dependencies on each other so cannot be merged together.
    while (true) {
      ABSL_ASSIGN_OR_RETURN(HloInstruction * new_dot,
                       MergeCluster(key, values, graph, graph_id, replacements,
                                    max_size_to_merge));
      if (!new_dot) {
        if (VLOG_IS_ON(3)) {
          std::vector<DotOperandUsage> unmerged =
              GetUnmergedDots(values, replacements);
          if (!unmerged.empty()) {
            std::vector<absl::string_view> names;
            names.reserve(unmerged.size());
            for (const auto& u : unmerged) {
              names.push_back(u.dot->name());
            }
            VLOG(3) << "Stop merging for class. Remaining unmerged dots: ["
                    << absl::StrJoin(names, ", ") << "]";
          }
        }
        break;
      }
      merged_dots.push_back(new_dot);
    }
  }

  // Apply replacements first to disconnect all original dots from the graph.
  for (auto& [original, replacement] : replacements) {
    ABSL_RETURN_IF_ERROR(original->ReplaceAllUsesWith(replacement));
  }

  // Delete the dead dot instructions and recursively clean up their unused
  // operands.
  for (auto& [original, replacement] : replacements) {
    ABSL_RETURN_IF_ERROR(comp->RemoveInstructionAndUnusedOperands(original));
  }

  // Simplify consumer chains of merged dots: if the dot had a chain of
  // transpose/reshapes after the output, we can often shorten it.
  for (HloInstruction* merged_dot : merged_dots) {
    for (HloInstruction* slice : merged_dot->users()) {
      if (slice->opcode() != HloOpcode::kSlice) {
        continue;
      }
      // Copy the users to avoid iterator invalidation when we already replace
      // some of them.
      std::vector<HloInstruction*> slice_users(slice->users().begin(),
                                               slice->users().end());
      for (HloInstruction* user : slice_users) {
        ABSL_RETURN_IF_ERROR(SimplifyConsumerChain(user));
      }
    }
  }

  return !replacements.empty();
}

}  // anonymous namespace

absl::StatusOr<bool> DotMerger::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    ABSL_ASSIGN_OR_RETURN(bool changed_computation,
                     MergeDots(comp, max_size_to_merge_, queue_id_));
    changed |= changed_computation;
  }
  return changed;
}

}  // namespace xla
