/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/scatter_expander.h"

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/gather_scatter_utils.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/scatter_utils.h"
#include "xla/service/while_util.h"
#include "xla/shape.h"

namespace xla {

static absl::StatusOr<HloInstruction*> CheckIndexValidity(
    HloComputation* computation, HloInstruction* index,
    absl::Span<const int64_t> operand_dims,
    absl::Span<const int64_t> window_sizes, HloModule* module) {
  DCHECK_NE(nullptr, module);
  DCHECK_EQ(operand_dims.size(), window_sizes.size());

  // Valid range for the index: [0, operand_dims - window_sizes]

  // Check if the index has any negative values.
  HloInstruction* zero_index = BroadcastZeros(
      computation, index->shape().element_type(), index->shape().dimensions());
  TF_ASSIGN_OR_RETURN(
      HloInstruction * negative_index_check,
      MakeCompareHlo(ComparisonDirection::kLe, zero_index, index));

  // Check if the index is OOB w.r.t. the operand dimensions and window sizes.
  std::vector<int64_t> max_valid_index(operand_dims.size());
  for (int i = 0; i < operand_dims.size(); ++i) {
    max_valid_index[i] = operand_dims[i] - window_sizes[i];
  }
  TF_ASSIGN_OR_RETURN(
      HloInstruction * max_valid_index_constant,
      MakeR1ConstantHlo<int64_t>(computation, index->shape().element_type(),
                                 max_valid_index));
  TF_ASSIGN_OR_RETURN(HloInstruction * oob_index_check,
                      MakeCompareHlo(ComparisonDirection::kGe,
                                     max_valid_index_constant, index));

  // Combine the results of the two checks above.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * valid_index,
      MakeBinaryHlo(HloOpcode::kAnd, negative_index_check, oob_index_check));

  // Reduce the index validity check vector into a scalar predicate.
  auto reduction_init = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  TF_ASSIGN_OR_RETURN(
      HloInstruction * valid_index_reduced,
      MakeReduceHlo(valid_index, reduction_init, HloOpcode::kAnd, module));

  // Return a broadcasted value of the scalar predicate to the same size as the
  // window.
  return MakeBroadcastHlo(valid_index_reduced, {}, window_sizes);
}

// Returns the sorted dimensions in a slice that are either collapsed or
// corresponding to an explicit batching dimension.
std::vector<int64_t> GetDegeneratedSliceDims(
    const ScatterDimensionNumbers& dim_numbers) {
  absl::Span<const int64_t> input_batching_dims =
      dim_numbers.input_batching_dims();
  absl::Span<const int64_t> inserted_window_dims =
      dim_numbers.inserted_window_dims();
  std::vector<int64_t> degenerated_dims;
  degenerated_dims.reserve(inserted_window_dims.size() +
                           input_batching_dims.size());
  absl::c_copy(inserted_window_dims, std::back_inserter(degenerated_dims));
  absl::c_copy(input_batching_dims, std::back_inserter(degenerated_dims));
  absl::c_sort(degenerated_dims);
  return degenerated_dims;
}

// Body of the while loop that performs the scatter operation using other HLOs.
static absl::StatusOr<std::vector<HloInstruction*>> ScatterLoopBody(
    HloScatterInstruction* scatter, HloInstruction* induction_var,
    absl::Span<HloInstruction* const> loop_state) {
  const ScatterDimensionNumbers& dim_numbers =
      scatter->scatter_dimension_numbers();
  CHECK_EQ(loop_state.size(), scatter->operand_count());
  auto operands = loop_state.first(scatter->scatter_operand_count());
  HloInstruction* scatter_indices = loop_state[operands.size()];
  auto updates = loop_state.last(operands.size());

  bool has_scalar_indices = scatter_indices->shape().dimensions_size() == 1;

  // Build a vector form of the induction variable of the while loop.
  HloInstruction* induction_var_as_vector =
      MakeBroadcastHlo(induction_var, /*broadcast_dimensions=*/{},
                       /*result_shape_bounds=*/{1});

  // Pick the index to scatter from scatter_indices based on the induction_var
  // and transform that to an index into the `operand` space.
  HloInstruction* index_vector;
  if (has_scalar_indices) {
    TF_ASSIGN_OR_RETURN(
        index_vector,
        MakeDynamicSliceHlo(scatter_indices, induction_var_as_vector, {1}));
  } else {
    TF_ASSIGN_OR_RETURN(
        HloInstruction * index_into_scatter_indices,
        PadVectorWithZeros(induction_var_as_vector,
                           /*zeros_to_prepend=*/0, /*zeros_to_append=*/1));
    int index_vector_size = scatter_indices->shape().dimensions(1);
    TF_ASSIGN_OR_RETURN(
        HloInstruction * index_vector_2d,
        MakeDynamicSliceHlo(scatter_indices, index_into_scatter_indices,
                            {1, index_vector_size}));
    TF_ASSIGN_OR_RETURN(index_vector,
                        ElideDegenerateDims(index_vector_2d, {0}));
  }
  TF_ASSIGN_OR_RETURN(
      HloInstruction * scatter_slice_start,
      ExpandIndexVectorIntoOperandSpace(
          scatter->scatter_indices()->shape(),
          operands[0]->shape().dimensions_size(),
          dim_numbers.index_vector_dim(),
          dim_numbers.scatter_dims_to_operand_dims(),
          dim_numbers.scatter_indices_batching_dims(),
          dim_numbers.input_batching_dims(), index_vector, induction_var));

  // Extract the slice to be used to update from `updates` tensor for the
  // induction_var corresponding to this iteration of the while loop.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * index_into_updates,
      PadVectorWithZeros(
          induction_var_as_vector, /*zeros_to_prepend=*/0,
          /*zeros_to_append=*/updates[0]->shape().dimensions_size() - 1));
  std::vector<int64_t> update_slice_bounds(
      updates[0]->shape().dimensions().begin(),
      updates[0]->shape().dimensions().end());
  update_slice_bounds[0] = 1;

  absl::InlinedVector<HloInstruction*, 2> map_operands(
      operands.size() + updates.size(), nullptr);
  auto operand_slices_to_update =
      absl::MakeSpan(map_operands).first(operands.size());
  auto update_slices_with_dims_inserted =
      absl::MakeSpan(map_operands).last(updates.size());
  absl::Span<const int64_t> actual_update_slice_dims;

  std::vector<int64_t> degenerated_dims = GetDegeneratedSliceDims(dim_numbers);

  for (int i = 0, n = operands.size(); i < n; ++i) {
    HloInstruction* update = updates[i];
    TF_ASSIGN_OR_RETURN(
        HloInstruction * update_slice,
        MakeDynamicSliceHlo(update, index_into_updates, update_slice_bounds));
    TF_ASSIGN_OR_RETURN(HloInstruction * update_slice_for_scatter,
                        ElideDegenerateDims(update_slice, {0}));
    TF_ASSIGN_OR_RETURN(
        HloInstruction * update_slice_with_dims_inserted,
        InsertDegenerateDims(update_slice_for_scatter, degenerated_dims));
    update_slices_with_dims_inserted[i] = update_slice_with_dims_inserted;
    // Note that the following transformation assumes that both DynamicSlice and
    // DynamicUpdateSlice follow the same semantics for OOB indices. For
    // example, if there are negative indices and DynamicSlice uses "clamping"
    // semantics, then the extracted data will be "shifted". Since
    // DynamicUpdateSlice also follows the same "clamping" semantics, writing
    // the update will also be "shifted" by exactly the same amount. So, this
    // transformation is correct as long as the semantics of handling OOB
    // indices remain the same in DynamicSlice and DynamicUpdateSlice.

    // Extract the slice to update from `operand` tensor.
    HloInstruction* operand = operands[i];
    const Shape& update_slice_shape = update_slice_with_dims_inserted->shape();
    TF_ASSIGN_OR_RETURN(HloInstruction * operand_slice_to_update,
                        MakeDynamicSliceHlo(operand, scatter_slice_start,
                                            update_slice_shape.dimensions()));
    operand_slices_to_update[i] = operand_slice_to_update;
    if (i == 0) {
      actual_update_slice_dims = update_slice_shape.dimensions();
    } else {
      TF_RET_CHECK(actual_update_slice_dims == update_slice_shape.dimensions());
    }
  }

  TF_ASSIGN_OR_RETURN(
      HloInstruction * is_index_valid,
      CheckIndexValidity(operands[0]->parent(), scatter_slice_start,
                         operands[0]->shape().dimensions(),
                         actual_update_slice_dims, scatter->GetModule()));

  // Write the updated value of the slice into `operand` tensor.
  std::vector<HloInstruction*> updated_loop_state;
  updated_loop_state.reserve(loop_state.size());
  for (int i = 0, n = operands.size(); i < n; ++i) {
    // Compute the new value for the slice to be updated in `operand` tensor by
    // combining the existing value and the update value using the update
    // computation.
    // NOTE: For scatters with N outputs, we currently have duplicate the Map
    // computation N times because we don't support multioutput Map yet.
    TF_ASSIGN_OR_RETURN(HloComputation * to_apply,
                        CallAndGetOutput(scatter->to_apply(), i));
    TF_ASSIGN_OR_RETURN(HloInstruction * updated_operand_slice,
                        MakeMapHlo(map_operands, to_apply));
    // Select the updated operand only if the index is valid. If not, select the
    // original value.
    TF_ASSIGN_OR_RETURN(HloInstruction * updates_to_apply,
                        MakeSelectHlo(is_index_valid, updated_operand_slice,
                                      operand_slices_to_update[i]));
    TF_ASSIGN_OR_RETURN(HloInstruction * updated_operand,
                        MakeDynamicUpdateSliceHlo(operands[i], updates_to_apply,
                                                  scatter_slice_start));
    updated_loop_state.push_back(updated_operand);
  }
  updated_loop_state.push_back(scatter_indices);
  absl::c_copy(updates, std::back_inserter(updated_loop_state));

  return updated_loop_state;
}

// High Level Algorithm.
//
// 1. Canonicalize the scatter_indices tensor such that it has rank 2, where
//    each row is an index into the operand.
// 2. Canonicalize the updates tensor such that is has rank `num_window_dims+1`
//    and the scatter dim is the most-major dimension.
// 3. Iterate over the set of indices in the canonicalized scatter_indices
//    tensor using a while loop, updating the operand for each such index. Each
//    iteration of this while loop performs the following:
//      a. Pick the index from scatter_indices for this iteration.
//      b. Transfrom this index into an index into the operand space.
//      c. Extract the slice to be used to update from the updates tensor.
//      d. Extract the slice to update from the operand tensor.
//      e. Compute the new value for the slice to update by combining the slices
//         from c. and d. using the update_computation of scatter.
//      f. Write the updated value of the slice into the operand tensor.

absl::StatusOr<HloInstruction*> ScatterExpander::ExpandInstruction(
    HloInstruction* inst) {
  auto* scatter = Cast<HloScatterInstruction>(inst);
  auto scatter_operands = scatter->scatter_operands();
  HloInstruction* scatter_indices = scatter->scatter_indices();
  auto scatter_updates = scatter->scatter_updates();
  const ScatterDimensionNumbers& dim_numbers =
      scatter->scatter_dimension_numbers();

  // If the updates tensors are empty, there is no need to update the operands.
  // The operands can be forwarded.
  if (ShapeUtil::IsZeroElementArray(scatter_updates[0]->shape())) {
    if (scatter_operands.size() == 1) {
      return scatter_operands[0];
    }
    return scatter->parent()->AddInstruction(
        HloInstruction::CreateTuple(scatter_operands));
  }

  // Compute the trip count for the while loop to be used for scatter. This
  // should be the number of indices we should scatter into the operand.
  int64_t scatter_loop_trip_count = ScatterIndicesCount(scatter);
  if (!IsInt32(scatter_loop_trip_count)) {
    return Unimplemented(
        "Scatter operations with more than 2147483647 scatter indices are not "
        "supported. This error occurred for %s.",
        scatter->ToString());
  }

  // Canonicalize the scatter_indices, after which the size of its most-major
  // dimension must be same as the while loop trip count.
  TF_ASSIGN_OR_RETURN(HloInstruction * canonical_scatter_indices,
                      CanonicalizeScatterIndices(
                          scatter_indices, dim_numbers.index_vector_dim()));
  CHECK_EQ(scatter_loop_trip_count,
           canonical_scatter_indices->shape().dimensions(0));

  // Canonicalize the updates, after which the size of its most-major dimension
  // must be same as the while loop trip count.
  std::vector<HloInstruction*> adjusted_canonical_updates;
  adjusted_canonical_updates.reserve(scatter_updates.size());
  for (HloInstruction* update : scatter_updates) {
    TF_ASSIGN_OR_RETURN(
        HloInstruction * canonical_update,
        PermuteScatterAndWindowDims(update, dim_numbers.update_window_dims()));
    TF_ASSIGN_OR_RETURN(
        HloInstruction * adjusted_canonical_update,
        AdjustScatterDims(scatter_indices->shape(), canonical_update,
                          dim_numbers.index_vector_dim()));
    CHECK_EQ(scatter_loop_trip_count,
             adjusted_canonical_update->shape().dimensions(0));
    adjusted_canonical_updates.push_back(adjusted_canonical_update);
  }

  // The while loop that implements the scatter operation.
  std::vector<HloInstruction*> loop_state;
  loop_state.reserve(scatter->operand_count());
  absl::c_copy(scatter_operands, std::back_inserter(loop_state));
  loop_state.push_back(canonical_scatter_indices);
  absl::c_copy(adjusted_canonical_updates, std::back_inserter(loop_state));
  absl::StatusOr<std::vector<HloInstruction*>> scatter_loop_result_status =
      WhileUtil::MakeCountedLoop(
          scatter->parent(), scatter_loop_trip_count, loop_state,
          [scatter](HloInstruction* induction_var,
                    const std::vector<HloInstruction*>& loop_state) {
            return ScatterLoopBody(scatter, induction_var, loop_state);
          },
          scatter->metadata());
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> scatter_loop_result,
                      scatter_loop_result_status);
  auto results =
      absl::MakeSpan(scatter_loop_result).first(scatter_operands.size());
  return MaybeMakeTuple(results);
}

bool ScatterExpander::InstructionMatchesPattern(HloInstruction* inst) {
  auto* scatter = DynCast<HloScatterInstruction>(inst);
  return (scatter != nullptr) && (mode_ == kEliminateAllScatters ||
                                  (mode_ == kEliminateSimpleScatters &&
                                   ScatterIndicesCount(scatter) == 1) ||
                                  (mode_ == kEliminateIndeterministicScatters &&
                                   !IsScatterDeterministic(scatter)));
}

}  // namespace xla
