/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/scatter_simplifier.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/service/gather_scatter_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace {

StatusOr<HloInstruction*> FlattenAndTransposeUpdates(
    HloInstruction* updates, absl::Span<const int64_t> update_window_dims,
    absl::Span<const int64_t> inserted_window_dims,
    int64_t scatter_indices_size) {
  int64_t updates_rank = updates->shape().rank();

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
  TF_ASSIGN_OR_RETURN(updates, MaybeTranspose(updates, permutation));

  // Collapse scatter dimensions to one.
  if (num_scatter_dims > 1) {
    TF_ASSIGN_OR_RETURN(updates, CollapseFirstNDims(updates, num_scatter_dims));
  } else if (num_scatter_dims == 0) {
    TF_ASSIGN_OR_RETURN(updates, InsertDegenerateDims(updates, {0}));
  }

  // Insert size 1 dimensions.
  if (!inserted_window_dims.empty()) {
    std::vector<int64_t> new_dims;
    new_dims.reserve(inserted_window_dims.size());
    for (int64_t i : inserted_window_dims) {
      new_dims.push_back(i + 1);
    }
    TF_ASSIGN_OR_RETURN(updates, InsertDegenerateDims(updates, new_dims));
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
  for (auto& dim : operand_permutation) {
    update_permutation.push_back(dim + 1);
  }
  return update_permutation;
}

// Transforms the scatter_updates field of scatter. scatter_indices_size is the
// size of the scatter dimension in scatter_indices.
StatusOr<std::vector<HloInstruction*>> TransformScatterUpdates(
    HloScatterInstruction* scatter,
    const std::vector<int64_t>& update_permutation,
    int64_t scatter_indices_size) {
  std::vector<HloInstruction*> scatter_updates;
  const auto& attrs = scatter->scatter_dimension_numbers();
  scatter_updates.reserve(scatter->scatter_updates().size());
  for (auto* update : scatter->scatter_updates()) {
    TF_ASSIGN_OR_RETURN(
        scatter_updates.emplace_back(),
        FlattenAndTransposeUpdates(update, attrs.update_window_dims(),
                                   attrs.inserted_window_dims(),
                                   scatter_indices_size));
  }
  return MaybeTranspose(scatter_updates, update_permutation);
}

ScatterDimensionNumbers MakeScatterDimensionNumbers(
    int64_t operand_rank, int64_t scatter_indices_vector_size) {
  ScatterDimensionNumbers dim_numbers;
  dim_numbers.mutable_update_window_dims()->Reserve(
      static_cast<int>(operand_rank));
  for (int i = 0; i < operand_rank; ++i) {
    dim_numbers.add_update_window_dims(1 + i);
  }
  dim_numbers.mutable_scatter_dims_to_operand_dims()->Reserve(
      static_cast<int>(scatter_indices_vector_size));
  for (int i = 0; i < scatter_indices_vector_size; ++i) {
    dim_numbers.add_scatter_dims_to_operand_dims(i);
  }
  dim_numbers.set_index_vector_dim(1);
  return dim_numbers;
}

}  // namespace

StatusOr<HloInstruction*> ScatterSimplifier::ExpandInstruction(
    HloInstruction* inst) {
  auto* scatter = Cast<HloScatterInstruction>(inst);

  if (scatter->called_computations().size() != 1) {
    return InvalidArgumentStrCat(
        "Expected scatter->called_computations() to have exactly one element, "
        "got ",
        scatter->called_computations().size());
  }

  const auto& attrs = scatter->scatter_dimension_numbers();
  const int operand_rank =
      attrs.update_window_dims().size() + attrs.inserted_window_dims().size();

  // We permute updates and operands according to scatter_dims_to_operand_dims.
  auto [operand_permutation, operand_permutation_inverse] =
      MakeOperandStartIndexPermutations(attrs.scatter_dims_to_operand_dims(),
                                        operand_rank);
  auto update_permutation = MakeUpdatePermutation(operand_permutation);

  TF_ASSIGN_OR_RETURN(auto* scatter_indices,
                      TransformStartIndices(scatter->scatter_indices(),
                                            attrs.index_vector_dim()));
  TF_ASSIGN_OR_RETURN(
      auto scatter_updates,
      TransformScatterUpdates(scatter, update_permutation,
                              scatter_indices->shape().dimensions(0)));
  TF_ASSIGN_OR_RETURN(
      auto scatter_operands,
      MaybeTranspose(scatter->scatter_operands(), operand_permutation));

  auto dim_numbers = MakeScatterDimensionNumbers(
      operand_rank, attrs.scatter_dims_to_operand_dims().size());
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
      scatter->called_computations().front(), dim_numbers,
      // TODO(unknown): Is this still correct?
      scatter->indices_are_sorted(), scatter->unique_indices()));

  // No need to unpack the Scatter results if the operand permutation is a
  // no-op.
  if (IsIdentityPermutation(operand_permutation)) {
    return result;
  }

  if (scatter->scatter_operands().size() == 1) {
    return MaybeTranspose(result, operand_permutation_inverse);
  }

  std::vector<HloInstruction*> result_items;
  result_items.reserve(scatter->scatter_operands().size());
  for (int i = 0; i < scatter->scatter_operands().size(); ++i) {
    TF_ASSIGN_OR_RETURN(result_items.emplace_back(),
                        MakeGetTupleElementHlo(result, i));
    TF_ASSIGN_OR_RETURN(
        result_items.back(),
        MaybeTranspose(result_items.back(), operand_permutation_inverse));
  }

  return MaybeMakeTuple(result_items);
}

bool ScatterSimplifier::InstructionMatchesPattern(HloInstruction* inst) {
  if (auto* scatter = DynCast<HloScatterInstruction>(inst)) {
    const auto& dims = scatter->scatter_dimension_numbers();

    bool nonstandard_index_vector_dim =
        dims.index_vector_dim() !=
        scatter->scatter_indices()->shape().rank() - 1;
    int64_t num_scatter_dims =
        scatter->scatter_updates().front()->shape().rank() -
        dims.update_window_dims().size();
    bool scatter_indices_reordered =
        !IsIdentityPermutation(dims.scatter_dims_to_operand_dims());
    bool scatter_dim_not_first =
        absl::c_linear_search(dims.update_window_dims(), 0);

    return nonstandard_index_vector_dim || num_scatter_dims > 1 ||
           scatter_indices_reordered || scatter_dim_not_first ||
           !dims.inserted_window_dims().empty();
  }
  return false;
}

}  // namespace xla
