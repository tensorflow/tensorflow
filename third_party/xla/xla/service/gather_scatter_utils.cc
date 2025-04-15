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

#include "xla/service/gather_scatter_utils.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/permutation_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

// Generates the HLO to calculate the implicit and explicit batch dimension
// indices and returns the explicit batch dimension to the HLO indices in the
// order of major to minor.
std::vector<HloInstruction*> GenerateExplicitBatchDimIndices(
    const Shape& start_indices_shape, int64_t index_vector_dim,
    absl::Span<const int64_t> start_indices_batching_dims,
    HloInstruction* induction_var) {
  if (start_indices_batching_dims.empty()) {
    return {};
  }

  int64_t rank = start_indices_shape.dimensions_size();
  int64_t num_batch_dims = (rank == index_vector_dim) ? rank : rank - 1;
  HloComputation* computation = induction_var->parent();
  HloInstruction* divident = induction_var;
  const Shape& shape = induction_var->shape();

  std::vector<HloInstruction*> explicit_batch_dim_indices(
      start_indices_batching_dims.size());

  for (int64_t i = start_indices_shape.dimensions_size() - 1; i >= 0; i--) {
    if (i == index_vector_dim) {
      continue;
    }
    auto it = absl::c_find(start_indices_batching_dims, i);
    num_batch_dims--;  // Reuse the variable to count remaining batch dims.
    if (num_batch_dims == 0) {
      if (it != start_indices_batching_dims.end()) {
        // Avoid generating a remainder that just returns the divident itself.
        explicit_batch_dim_indices[it - start_indices_batching_dims.begin()] =
            divident;
      }
      break;
    }

    HloInstruction* divisor =
        computation->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(start_indices_shape.dimensions(i))));
    if (it != start_indices_batching_dims.end()) {
      explicit_batch_dim_indices[it - start_indices_batching_dims.begin()] =
          computation->AddInstruction(HloInstruction::CreateBinary(
              shape, HloOpcode::kRemainder, divident, divisor));
    }

    divident = computation->AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kDivide, divident, divisor));
  }

  return explicit_batch_dim_indices;
}

}  // namespace

absl::StatusOr<HloInstruction*> TransformStartIndices(
    HloInstruction* indices, int64_t index_vector_dim) {
  int64_t rank = indices->shape().dimensions_size();
  if (index_vector_dim == rank) {
    // Add a size 1 dimension to the indices if the index_vector_dim is
    // implicit.
    TF_ASSIGN_OR_RETURN(indices,
                        InsertDegenerateDims(indices, {index_vector_dim}));
    ++rank;
  } else if (index_vector_dim < rank - 1) {
    // Ensure index_vector_dim is the last dimension in scatter_indices.
    TF_ASSIGN_OR_RETURN(indices,
                        MoveDimensionToEnd(indices, index_vector_dim, rank));
  }

  // Flatten indices, making it two-dimensional.
  if (rank > 2) {
    TF_ASSIGN_OR_RETURN(indices, CollapseFirstNDims(indices, rank - 1));
  } else if (rank == 1) {
    TF_ASSIGN_OR_RETURN(indices, InsertDegenerateDims(indices, {0}));
  }
  return indices;
}

std::pair<std::vector<int64_t>, std::vector<int64_t>>
MakeOperandStartIndexPermutations(absl::Span<const int64_t> dim_map,
                                  int operand_rank) {
  std::vector<int64_t> perm;
  perm.reserve(operand_rank);
  absl::c_copy(dim_map, std::back_inserter(perm));
  for (int i = 0; i < operand_rank; ++i) {
    if (!absl::c_linear_search(dim_map, i)) {
      perm.push_back(i);
    }
  }
  return {perm, InversePermutation(perm)};
}

absl::StatusOr<HloInstruction*> MaybeTranspose(
    HloInstruction* operand, absl::Span<const int64_t> permutation) {
  if (IsIdentityPermutation(permutation)) {
    return operand;
  }
  TF_ASSIGN_OR_RETURN(auto* result, MakeTransposeHlo(operand, permutation));
  // Assign the default layout to the transpose. This method is also used after
  // layout normalization, and before, we don't care about the layout.
  *result->mutable_shape()->mutable_layout() =
      LayoutUtil::GetDefaultLayoutForShape(result->shape());
  // Propagate the element size in bits of the operand to the newly created
  // transpose. For sub-byte types, there is no canonical cross-platform
  // normalization for packing, so `GetDefaultLayoutForShape` may not end up
  // setting the element size in bits correctly.
  int64_t element_size_in_bits =
      operand->shape().layout().element_size_in_bits();
  result->mutable_shape()->mutable_layout()->set_element_size_in_bits(
      element_size_in_bits);
  return result;
}

absl::StatusOr<std::vector<HloInstruction*>> MaybeTranspose(
    absl::Span<HloInstruction* const> operands,
    const std::vector<int64_t>& operand_permutation) {
  std::vector<HloInstruction*> result;
  result.reserve(operands.size());
  for (auto* operand : operands) {
    TF_ASSIGN_OR_RETURN(result.emplace_back(),
                        MaybeTranspose(operand, operand_permutation));
  }
  return result;
}

absl::StatusOr<HloInstruction*> MoveDimensionToEnd(HloInstruction* operand,
                                                   size_t dimension,
                                                   size_t rank) {
  std::vector<int64_t> permutation;
  for (size_t i = 0; i < rank; ++i) {
    if (i != dimension) permutation.push_back(i);
  }
  permutation.push_back(dimension);
  return MaybeTranspose(operand, permutation);
}

absl::StatusOr<HloInstruction*> ExpandIndexVectorIntoOperandSpace(
    const Shape& start_indices_shape, int64_t operand_rank,
    int64_t index_vector_dim, absl::Span<const int64_t> start_index_map,
    absl::Span<const int64_t> start_indices_batching_dims,
    absl::Span<const int64_t> operand_batching_dims,
    HloInstruction* index_vector, HloInstruction* induction_var) {
  HloComputation* computation = index_vector->parent();
  const Shape& index_shape = index_vector->shape();

  if (operand_rank == 0) {
    // This is a Gather/Scatter from/on a scalar. Return a zero-sized vector of
    // indices.
    return computation->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateFromDimensions(index_shape.element_type(), {0})));
  }

  HloInstruction* zero =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateFromDimensions(index_shape.element_type(), {1})));

  // We extract out individual components from the smaller index and concatenate
  // them (interspersing zeros as needed) into the larger index.
  std::vector<HloInstruction*> expanded_index_components;
  std::vector<HloInstruction*> explicit_batch_dim_indices =
      GenerateExplicitBatchDimIndices(start_indices_shape, index_vector_dim,
                                      start_indices_batching_dims,
                                      induction_var);
  int64_t seen_explicit_batch_dims = 0;
  for (int i = 0; i < operand_rank; i++) {
    int64_t index_vector_dim_index = FindIndex(start_index_map, i);
    if (index_vector_dim_index != start_index_map.size()) {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * component_to_concat,
          MakeSliceHlo(index_vector, /*start_indices=*/{index_vector_dim_index},
                       /*limit_indices=*/{index_vector_dim_index + 1},
                       /*strides=*/{1}));
      expanded_index_components.push_back(component_to_concat);
    } else {
      if (absl::c_linear_search(operand_batching_dims, i)) {
        expanded_index_components.push_back(MakeBroadcastHlo(
            explicit_batch_dim_indices[seen_explicit_batch_dims++],
            /*broadcast_dimensions=*/{},
            /*result_shape_bounds=*/{1}));
      } else {
        expanded_index_components.push_back(zero);
      }
    }
  }

  return MakeConcatHlo(expanded_index_components, /*dimension=*/0);
}

bool IsCollapsedOrBatchingDim(absl::Span<const int64_t> collapsed_dims,
                              absl::Span<const int64_t> batching_dims,
                              int64_t dim) {
  return absl::c_linear_search(collapsed_dims, dim) ||
         absl::c_linear_search(batching_dims, dim);
}

absl::flat_hash_map<int64_t, int64_t>
GetStartIndicesDimToOutputDimForExplicitBatchingDims(
    absl::Span<const int64_t> start_indices_batching_dims,
    int64_t index_vector_dim, absl::Span<const int64_t> offset_dims,
    int64_t start_indices_rank, int64_t output_rank) {
  absl::flat_hash_map<int64_t, int64_t>
      explicit_batching_dims_start_indices_dim_to_output_dim;
  explicit_batching_dims_start_indices_dim_to_output_dim.reserve(
      start_indices_batching_dims.size());

  for (int64_t output_dim = 0, start_indices_dim = 0; output_dim < output_rank;
       ++output_dim) {
    if (absl::c_linear_search(offset_dims, output_dim)) {
      continue;
    }
    if (start_indices_dim == index_vector_dim) {
      start_indices_dim++;
    }
    CHECK_LT(start_indices_dim, start_indices_rank);
    if (absl::c_linear_search(start_indices_batching_dims, start_indices_dim)) {
      // Explicit batching dim.
      explicit_batching_dims_start_indices_dim_to_output_dim
          [start_indices_dim] = output_dim;
    }
    ++start_indices_dim;
  }
  return explicit_batching_dims_start_indices_dim_to_output_dim;
}

}  // namespace xla
