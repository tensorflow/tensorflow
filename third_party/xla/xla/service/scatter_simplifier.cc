/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/scatter_simplifier.h"

#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/permutation_util.h"
#include "xla/service/call_inliner.h"
#include "xla/service/gather_scatter_utils.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

absl::StatusOr<HloInstruction*> FlattenAndTransposeUpdates(
    HloInstruction* updates, absl::Span<const int64_t> update_window_dims,
    absl::Span<const int64_t> inserted_window_dims,
    int64_t scatter_indices_size) {
  int64_t updates_rank = updates->shape().dimensions().size();

  std::vector<int64_t> permutation;
  const int64_t num_scatter_dims = updates_rank - update_window_dims.size();
  permutation.reserve(updates_rank);
  // Move the scatter dimensions to the front.
  for (int i = 0; i < updates_rank; ++i) {
    // update_window_dims is small, so linear search is acceptable.
    if (!absl::c_linear_search(update_window_dims, i)) {
      permutation.push_back(i);
    }
  }
  // Followed by the update_window_dims.
  absl::c_copy(update_window_dims, std::back_inserter(permutation));
  ABSL_ASSIGN_OR_RETURN(updates, MaybeTranspose(updates, permutation));

  // Collapse scatter dimensions to one.
  if (num_scatter_dims > 1) {
    ABSL_ASSIGN_OR_RETURN(updates, CollapseFirstNDims(updates, num_scatter_dims));
  } else if (num_scatter_dims == 0) {
    ABSL_ASSIGN_OR_RETURN(updates, InsertDegenerateDims(updates, {0}));
  }

  // Insert size 1 dimensions.
  if (!inserted_window_dims.empty()) {
    std::vector<int64_t> new_dims;
    new_dims.reserve(inserted_window_dims.size());
    for (int64_t i : inserted_window_dims) {
      new_dims.push_back(i + 1);
    }
    ABSL_ASSIGN_OR_RETURN(updates, InsertDegenerateDims(updates, new_dims));
  }
  return updates;
}

std::vector<int64_t> MakeUpdatePermutation(
    const std::vector<int64_t>& operand_permutation) {
  // For the updates, we need to add the scatter dimension to the permutation.
  std::vector<int64_t> update_permutation;
  update_permutation.reserve(operand_permutation.size() + 1);
  // After FlattenAndTransposeUpdates, the single scatter dimension is leading,
  // keep it that way.
  update_permutation.push_back(0);
  for (int64_t dim : operand_permutation) {
    update_permutation.push_back(dim + 1);
  }
  return update_permutation;
}

// Transforms the scatter_updates field of scatter. scatter_indices_size is the
// size of the scatter dimension in scatter_indices.
absl::StatusOr<std::vector<HloInstruction*>> TransformScatterUpdates(
    HloScatterInstruction* scatter,
    const std::vector<int64_t>& update_permutation,
    int64_t scatter_indices_size) {
  std::vector<HloInstruction*> scatter_updates;
  const auto& attrs = scatter->scatter_dimension_numbers();
  scatter_updates.reserve(scatter->scatter_updates().size());
  for (auto* update : scatter->scatter_updates()) {
    ABSL_ASSIGN_OR_RETURN(scatter_updates.emplace_back(),
                     FlattenAndTransposeUpdates(
                         update, attrs.update_window_dims(),
                         attrs.inserted_window_dims(), scatter_indices_size));
  }
  return MaybeTranspose(scatter_updates, update_permutation);
}

ScatterDimensionNumbers MakeScatterDimensionNumbers(
    int64_t operand_rank, int64_t scatter_indices_vector_size,
    absl::Span<const int64_t> scatter_dims_to_operand_dims) {
  ScatterDimensionNumbers dim_numbers;
  dim_numbers.mutable_update_window_dims()->Reserve(
      static_cast<int>(operand_rank));
  for (int i = 0; i < operand_rank; ++i) {
    dim_numbers.add_update_window_dims(1 + i);
  }
  dim_numbers.mutable_scatter_dims_to_operand_dims()->Reserve(
      static_cast<int>(scatter_indices_vector_size));
  for (int i = 0; i < scatter_indices_vector_size; ++i) {
    dim_numbers.add_scatter_dims_to_operand_dims(
        scatter_dims_to_operand_dims[i]);
  }
  dim_numbers.set_index_vector_dim(1);
  return dim_numbers;
}

// Returns true if permuting the operand dimensions so that
// scatter_dims_to_operand_dims maps to the leading dimensions makes the
// scatter's window writes more contiguous.
bool ShouldReorderOperandDims(const HloScatterInstruction* scatter) {
  const auto& attrs = scatter->scatter_dimension_numbers();
  if (!attrs.input_batching_dims().empty() ||
      !attrs.scatter_indices_batching_dims().empty() ||
      scatter->scatter_operand_count() != 1) {
    return false;
  }
  const Shape& operand_shape = scatter->scatter_operands().front()->shape();
  const Shape& updates_shape = scatter->scatter_updates().front()->shape();
  // The reorder wraps the scatter in two operand-sized transposes. Only
  // reorder when the written volume is large enough for the coalescing win
  // (strided writes measured an order of magnitude slower) to pay for the
  // copies.
  constexpr int64_t kMaxOperandToUpdatesRatio = 4;
  if (ShapeUtil::ElementsIn(updates_shape) * kMaxOperandToUpdatesRatio <
      ShapeUtil::ElementsIn(operand_shape)) {
    return false;
  }
  const int64_t operand_rank = operand_shape.dimensions().size();
  std::vector<int64_t> permutation =
      MakeOperandStartIndexPermutations(attrs.scatter_dims_to_operand_dims(),
                                        operand_rank)
          .first;
  return ScatterSimplifier::WriteRunLength(scatter, permutation) >
         ScatterSimplifier::WriteRunLength(scatter);
}

}  // namespace

absl::StatusOr<HloInstruction*> ScatterSimplifier::ExpandInstruction(
    HloInstruction* inst) {
  auto* scatter = Cast<HloScatterInstruction>(inst);

  if (scatter->called_computations().size() != 1) {
    return InvalidArgumentStrCat(
        "Expected scatter->called_computations() to have exactly one element, "
        "got ",
        scatter->called_computations().size());
  }
  HloComputation* called_computation = scatter->called_computations().front();

  const auto& attrs = scatter->scatter_dimension_numbers();
  const int operand_rank =
      attrs.update_window_dims().size() + attrs.inserted_window_dims().size();
  // Rewrite the scatter into the scalar operand.
  if (operand_rank == 0) {
    absl::InlinedVector<HloInstruction*, 2> scatter_operands_and_updates;
    scatter_operands_and_updates.reserve(2 * scatter->operand_count());
    absl::c_copy(scatter->scatter_operands(),
                 std::back_inserter(scatter_operands_and_updates));
    absl::c_copy(scatter->scatter_updates(),
                 std::back_inserter(scatter_operands_and_updates));

    auto* call_op = scatter->AddInstruction(HloInstruction::CreateCall(
        scatter->shape(), scatter_operands_and_updates, called_computation));
    call_op->set_original_value(scatter->original_value());
    ABSL_RETURN_IF_ERROR(scatter->ReplaceAllUsesWith(call_op));
    ABSL_ASSIGN_OR_RETURN(auto map, CallInliner::Inline(call_op));
    return map[call_op];
  }

  // When requested, permute the operand dimensions so that
  // scatter_dims_to_operand_dims maps to the leading dimensions, but only
  // when that makes the written windows more contiguous. Strided window
  // writes do not coalesce on GPUs, which costs more than the transposes
  // this inserts.
  std::vector<int64_t> operand_permutation(operand_rank);
  std::vector<int64_t> operand_permutation_inverse(operand_rank);
  absl::c_iota(operand_permutation, 0);
  absl::c_iota(operand_permutation_inverse, 0);
  if (reorder_operand_dims_for_coalescing_ &&
      ShouldReorderOperandDims(scatter)) {
    auto permutations = MakeOperandStartIndexPermutations(
        attrs.scatter_dims_to_operand_dims(), operand_rank);
    operand_permutation = std::move(permutations.first);
    operand_permutation_inverse = std::move(permutations.second);
  }
  auto update_permutation = MakeUpdatePermutation(operand_permutation);

  ABSL_ASSIGN_OR_RETURN(auto* scatter_indices,
                   TransformStartIndices(scatter->scatter_indices(),
                                         attrs.index_vector_dim()));
  ABSL_ASSIGN_OR_RETURN(
      auto scatter_updates,
      TransformScatterUpdates(scatter, update_permutation,
                              scatter_indices->shape().dimensions(0)));
  ABSL_ASSIGN_OR_RETURN(
      auto scatter_operands,
      MaybeTranspose(scatter->scatter_operands(), operand_permutation));

  // Map scatter_dims_to_operand_dims into the permuted operand.
  std::vector<int64_t> scatter_dims_to_operand_dims;
  scatter_dims_to_operand_dims.reserve(
      attrs.scatter_dims_to_operand_dims().size());
  for (int64_t dim : attrs.scatter_dims_to_operand_dims()) {
    scatter_dims_to_operand_dims.push_back(operand_permutation_inverse[dim]);
  }
  auto dim_numbers = MakeScatterDimensionNumbers(
      operand_rank, attrs.scatter_dims_to_operand_dims().size(),
      scatter_dims_to_operand_dims);
  Shape output_shape;
  if (scatter_operands.size() == 1) {
    output_shape = scatter_operands.front()->shape();
  } else {
    std::vector<Shape> shapes;
    shapes.reserve(scatter_operands.size());
    for (auto* operand : scatter_operands) {
      shapes.push_back(operand->shape());
    }
    output_shape = ShapeUtil::MakeTupleShape(shapes);
  }
  auto* result = scatter->AddInstruction(HloInstruction::CreateScatter(
      output_shape, scatter_operands, scatter_indices, scatter_updates,
      called_computation, dim_numbers,
      // TODO(unknown): Is this still correct?
      scatter->indices_are_sorted(), scatter->unique_indices()));

  if (IsIdentityPermutation(operand_permutation)) {
    return result;
  }

  // ShouldReorderOperandDims only allows non-variadic scatters, so the
  // result is a single array.
  return MaybeTranspose(result, operand_permutation_inverse);
}

bool ScatterSimplifier::IsSimplifiedScatter(
    const HloScatterInstruction* scatter) {
  const auto& dims = scatter->scatter_dimension_numbers();
  auto operand_rank =
      scatter->scatter_operands().front()->shape().dimensions().size();
  if (operand_rank == 0) {
    return false;
  }

  bool standard_index_vector_dim =
      dims.index_vector_dim() + 1 ==
      scatter->scatter_indices()->shape().dimensions().size();
  int64_t num_scatter_dims =
      scatter->scatter_updates().front()->shape().dimensions().size() -
      dims.update_window_dims().size();
  bool first_dim_not_in_update_window_dims =
      !absl::c_linear_search(dims.update_window_dims(), 0);
  bool update_window_dims_sorted = absl::c_is_sorted(dims.update_window_dims());

  return standard_index_vector_dim && num_scatter_dims <= 1 &&
         first_dim_not_in_update_window_dims && update_window_dims_sorted &&
         dims.inserted_window_dims().empty();
}

int64_t ScatterSimplifier::WriteRunLength(
    const HloScatterInstruction* scatter,
    absl::Span<const int64_t> permutation) {
  const auto& dims = scatter->scatter_dimension_numbers();
  const Shape& operand_shape = scatter->scatter_operands().front()->shape();
  const Shape& updates_shape = scatter->scatter_updates().front()->shape();
  const int64_t rank = operand_shape.dimensions().size();

  // Window slice size along each operand dimension. Inserted window dims and
  // operand batching dims write a single element.
  std::vector<int64_t> window_sizes(rank, 1);
  for (int64_t dim = 0, window_dim = 0; dim < rank; ++dim) {
    if (absl::c_linear_search(dims.inserted_window_dims(), dim) ||
        absl::c_linear_search(dims.input_batching_dims(), dim)) {
      continue;
    }
    window_sizes[dim] =
        updates_shape.dimensions(dims.update_window_dims(window_dim++));
  }

  // Walk from the minor-most dimension outwards while the writes stay
  // contiguous, i.e. while the window covers each dimension fully.
  int64_t run_length = 1;
  for (int64_t i = rank - 1; i >= 0; --i) {
    int64_t dim = permutation.empty() ? i : permutation[i];
    run_length *= window_sizes[dim];
    if (window_sizes[dim] != operand_shape.dimensions(dim)) {
      break;
    }
  }
  return run_length;
}

bool ScatterSimplifier::InstructionMatchesPattern(HloInstruction* inst) {
  auto* scatter = DynCast<HloScatterInstruction>(inst);
  if (scatter == nullptr) {
    return false;
  }
  return !IsSimplifiedScatter(scatter) ||
         (reorder_operand_dims_for_coalescing_ &&
          ShouldReorderOperandDims(scatter));
}

}  // namespace xla
