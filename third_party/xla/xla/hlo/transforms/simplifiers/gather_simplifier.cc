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

#include "xla/hlo/transforms/simplifiers/gather_simplifier.h"

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/literal_util.h"
#include "xla/permutation_util.h"
#include "xla/service/gather_scatter_utils.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape_util.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<HloInstruction*> GatherSimplifier::ExpandInstruction(
    HloInstruction* inst) {
  auto* gather = DynCast<HloGatherInstruction>(inst);

  // If any slice size is 0, we can just return a constant zero.
  if (absl::c_linear_search(gather->gather_slice_sizes(), 0)) {
    auto* zero = gather->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(gather->shape().element_type())));
    return gather->AddInstruction(
        HloInstruction::CreateBroadcast(gather->shape(), zero, {}));
  }

  const auto& dims = gather->gather_dimension_numbers();
  int operand_rank =
      dims.collapsed_slice_dims().size() + dims.offset_dims().size();

  // Make the operand conform to start_index_map.
  auto [operand_permutation, operand_permutation_inverse] =
      MakeOperandStartIndexPermutations(dims.start_index_map(), operand_rank);
  auto* operand = gather->operands()[0];
  auto* start_indices = gather->operands()[1];
  TF_ASSIGN_OR_RETURN(operand, MaybeTranspose(operand, operand_permutation));
  TF_ASSIGN_OR_RETURN(
      start_indices,
      TransformStartIndices(start_indices, dims.index_vector_dim()));

  // Permute the slice sizes according to start_index_map and compute the new
  // output shape for the Gather op.
  auto slice_sizes = Permute(gather->gather_slice_sizes(), operand_permutation);
  std::vector<int64_t> output_dims = {start_indices->shape().dimensions(0)};
  absl::c_copy(slice_sizes, std::back_inserter(output_dims));
  Shape output_shape =
      ShapeUtil::MakeShape(operand->shape().element_type(), output_dims);

  std::vector<int64_t> offset_dims(operand_rank);
  absl::c_iota(offset_dims, 1);
  std::vector<int64_t> start_index_map(dims.start_index_map().size());
  absl::c_iota(start_index_map, 0);

  auto* result = gather->AddInstruction(HloInstruction::CreateGather(
      output_shape, operand, start_indices,
      HloGatherInstruction::MakeGatherDimNumbers(
          offset_dims,
          /*collapsed_slice_dims=*/{}, start_index_map, /*index_vector_dim=*/1),
      slice_sizes, gather->indices_are_sorted()));

  // Undo the start_index_map transpose.
  std::vector<int64_t> output_permutation(1 +  // start index dimension.
                                          operand_rank);
  absl::c_transform(operand_permutation_inverse, output_permutation.begin() + 1,
                    [](int64_t dim) { return dim + 1; });
  TF_ASSIGN_OR_RETURN(result, MaybeTranspose(result, output_permutation));

  // Collapse the requested slice dimensions.
  if (!dims.collapsed_slice_dims().empty()) {
    std::vector<int64_t> collapsed_slice_dims(
        dims.collapsed_slice_dims().size());
    absl::c_transform(dims.collapsed_slice_dims(), collapsed_slice_dims.begin(),
                      [](int64_t dim) { return dim + 1; });
    TF_ASSIGN_OR_RETURN(result,
                        ElideDegenerateDims(result, collapsed_slice_dims));
  }

  // Expand the start index dimensions.
  auto original_start_index_dims = gather->operands()[1]->shape().dimensions();
  std::vector<int64_t> start_indices_dims;
  for (int i = 0; i < original_start_index_dims.size(); ++i) {
    if (i != dims.index_vector_dim()) {
      start_indices_dims.push_back(original_start_index_dims[i]);
    }
  }
  if (start_indices_dims.size() > 1) {
    TF_ASSIGN_OR_RETURN(result,
                        ExpandFirstDimIntoNDims(result, start_indices_dims));
  } else if (start_indices_dims.empty()) {
    TF_ASSIGN_OR_RETURN(result, ElideDegenerateDims(result, {0}));
  }

  // Move the offset dims to the final locations.
  std::vector<int64_t> output_perm;
  auto output_rank = static_cast<int64_t>(start_indices_dims.size() +
                                          dims.offset_dims().size());
  output_perm.reserve(output_rank);
  auto offset_dim_index = static_cast<int64_t>(start_indices_dims.size());
  int64_t start_index_dim_index = 0;
  for (int64_t i = 0; i < output_rank; ++i) {
    if (absl::c_linear_search(dims.offset_dims(), i)) {
      output_perm.push_back(offset_dim_index++);
    } else {
      output_perm.push_back(start_index_dim_index++);
    }
  }
  return MaybeTranspose(result, output_perm);
}

bool GatherSimplifier::IsSimplifiedGather(const HloGatherInstruction* gather) {
  auto* start_indices = gather->operands()[1];
  const auto& dims = gather->gather_dimension_numbers();
  return start_indices->shape().dimensions().size() == 2 &&
         dims.index_vector_dim() == 1 &&
         IsIdentityPermutation(dims.start_index_map()) &&
         dims.collapsed_slice_dims().empty() &&
         *dims.offset_dims().begin() == 1 &&
         *dims.offset_dims().rbegin() == dims.offset_dims().size();
}

bool GatherSimplifier::InstructionMatchesPattern(HloInstruction* inst) {
  auto* gather = DynCast<HloGatherInstruction>(inst);
  return gather && !IsSimplifiedGather(gather);
}

}  // namespace xla
