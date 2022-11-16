/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/scatter_expander.h"

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/while_util.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Transposes the given scatter_indices such that the index_vector_dim becomes
// the most-minor dimension.
static StatusOr<HloInstruction*> TransposeIndexVectorDimToLast(
    HloInstruction* scatter_indices, int64_t index_vector_dim) {
  const Shape& scatter_indices_shape = scatter_indices->shape();

  if (scatter_indices_shape.dimensions_size() == index_vector_dim) {
    return scatter_indices;
  }

  if (index_vector_dim == (scatter_indices_shape.dimensions_size() - 1)) {
    return scatter_indices;
  }

  std::vector<int64_t> permutation;
  permutation.reserve(scatter_indices_shape.dimensions_size());
  for (int64_t i = 0, e = scatter_indices_shape.dimensions_size(); i < e; i++) {
    if (i != index_vector_dim) {
      permutation.push_back(i);
    }
  }
  permutation.push_back(index_vector_dim);
  return MakeTransposeHlo(scatter_indices, permutation);
}

// Canonicalizes the scatter_indices tensor in order to keep them uniform while
// performing the scatter operation.
static StatusOr<HloInstruction*> CanonicalizeScatterIndices(
    HloInstruction* scatter_indices, int64_t index_vector_dim) {
  // Transpose the non-index-vector dimensions to the front.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * transposed_scatter_indices,
      TransposeIndexVectorDimToLast(scatter_indices, index_vector_dim));
  if (scatter_indices->shape().rank() == index_vector_dim + 1 &&
      scatter_indices->shape().dimensions(index_vector_dim) == 1) {
    auto new_shape =
        ShapeUtil::DeleteDimension(index_vector_dim, scatter_indices->shape());
    TF_ASSIGN_OR_RETURN(scatter_indices,
                        MakeReshapeHlo(new_shape, scatter_indices));
  }
  bool indices_are_scalar =
      index_vector_dim == scatter_indices->shape().dimensions_size();

  // The number of dimensions in scatter_indices that are index dimensions.
  const int64_t index_dims_in_scatter_indices = indices_are_scalar ? 0 : 1;

  // If there is only one index (i.e. scatter_indices has rank 1 and this
  // scatter is really just a dynamic update slice) add a leading degenerate
  // dimension for uniformity.  Otherwise create a "collapsed" leading dimension
  // that subsumes all of the non-index-vector dimensions.
  const Shape& shape = transposed_scatter_indices->shape();
  if (shape.dimensions_size() == index_dims_in_scatter_indices) {
    return PrependDegenerateDims(transposed_scatter_indices, 1);
  } else {
    // Collapse all but the dimensions (0 or 1) in scatter_indices containing
    // the index vectors.
    return CollapseFirstNDims(
        transposed_scatter_indices,
        shape.dimensions_size() - index_dims_in_scatter_indices);
  }
}

// Permutes the `updates` tensor such that all the scatter dims appear in the
// major dimensions and all the window dimensions appear in the minor
// dimensions.
static StatusOr<HloInstruction*> PermuteScatterAndWindowDims(
    HloInstruction* updates, absl::Span<const int64_t> update_window_dims) {
  std::vector<int64_t> permutation;
  const int64_t updates_rank = updates->shape().rank();
  permutation.reserve(updates_rank);

  for (int64_t i = 0; i < updates_rank; ++i) {
    bool is_scatter_dim = !absl::c_binary_search(update_window_dims, i);
    if (is_scatter_dim) {
      permutation.push_back(i);
    }
  }
  for (auto window_dim : update_window_dims) {
    permutation.push_back(window_dim);
  }

  return MakeTransposeHlo(updates, permutation);
}

// Expands or contracts the scatter indices in the updates tensor.
static StatusOr<HloInstruction*> AdjustScatterDims(
    const Shape& scatter_indices_shape, HloInstruction* updates,
    int64_t index_vector_dim) {
  int64_t num_scatter_dims = scatter_indices_shape.dimensions_size();
  if (index_vector_dim < scatter_indices_shape.dimensions_size()) {
    --num_scatter_dims;
  }
  if (num_scatter_dims == 0) {
    // If there are no scatter dims, this must be a dynamic-update-slice kind of
    // scatter. In this case, we prepend a degenerate dimension to work
    // uniformly in the while loop.
    return PrependDegenerateDims(updates, 1);
  }
  return CollapseFirstNDims(updates, num_scatter_dims);
}

// Expands an index vector from the scatter_indices tensor into a vector that
// can be used to dynamic-update-slice to perform the scatter update.
static StatusOr<HloInstruction*> ExpandIndexVectorIntoOperandSpace(
    HloInstruction* index_vector, const ScatterDimensionNumbers& dim_numbers,
    int64_t operand_rank) {
  HloComputation* computation = index_vector->parent();
  const Shape& index_shape = index_vector->shape();

  // Scatter of a scalar. Return a zero-sized vector of indices.
  if (operand_rank == 0) {
    return computation->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateFromDimensions(index_shape.element_type(), {0})));
  }

  HloInstruction* zero =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateFromDimensions(index_shape.element_type(), {1})));

  // We extract out individual components from the smaller index and concatenate
  // them (interspersing zeros as needed) into the larger index.
  std::vector<HloInstruction*> expanded_index_components;

  for (int i = 0; i < operand_rank; i++) {
    int64_t index_vector_dim_index =
        FindIndex(dim_numbers.scatter_dims_to_operand_dims(), i);
    if (index_vector_dim_index !=
        dim_numbers.scatter_dims_to_operand_dims_size()) {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * component_to_concat,
          MakeSliceHlo(index_vector, /*start_indices=*/{index_vector_dim_index},
                       /*limit_indices=*/{index_vector_dim_index + 1},
                       /*strides=*/{1}));
      expanded_index_components.push_back(component_to_concat);
    } else {
      expanded_index_components.push_back(zero);
    }
  }

  return MakeConcatHlo(expanded_index_components, /*dimension=*/0);
}

static StatusOr<HloInstruction*> CheckIndexValidity(
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

static StatusOr<HloComputation*> CallAndGetOutput(HloComputation* original,
                                                  int output_index) {
  HloInstruction* original_root = original->root_instruction();
  if (!original_root->shape().IsTuple()) {
    return original;
  }
  HloComputation* new_comp = [&] {
    HloComputation::Builder builder(
        absl::StrCat(original->name(), ".dup.", output_index));
    for (int i = 0, n = original->num_parameters(); i < n; ++i) {
      HloInstruction* original_param = original->parameter_instruction(i);
      builder.AddInstruction(HloInstruction::CreateParameter(
          i, original_param->shape(), original_param->name()));
    }
    return original->parent()->AddEmbeddedComputation(builder.Build());
  }();
  HloInstruction* call_original = new_comp->AddInstruction(
      HloInstruction::CreateCall(original_root->shape(),
                                 new_comp->parameter_instructions(), original));
  new_comp->set_root_instruction(
      new_comp->AddInstruction(
          HloInstruction::CreateGetTupleElement(call_original, output_index)),
      /*accept_different_shape=*/true);
  TF_RETURN_IF_ERROR(CallInliner::Inline(call_original).status());
  return new_comp;
}

// Body of the while loop that performs the scatter operation using other HLOs.
static StatusOr<std::vector<HloInstruction*>> ScatterLoopBody(
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
          index_vector, dim_numbers, operands[0]->shape().dimensions_size()));

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
  for (int i = 0, n = operands.size(); i < n; ++i) {
    HloInstruction* update = updates[i];
    TF_ASSIGN_OR_RETURN(
        HloInstruction * update_slice,
        MakeDynamicSliceHlo(update, index_into_updates, update_slice_bounds));
    TF_ASSIGN_OR_RETURN(HloInstruction * update_slice_for_scatter,
                        ElideDegenerateDims(update_slice, {0}));
    TF_ASSIGN_OR_RETURN(
        HloInstruction * update_slice_with_dims_inserted,
        InsertDegenerateDims(update_slice_for_scatter,
                             dim_numbers.inserted_window_dims()));
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

static int64_t ScatterTripCount(const HloScatterInstruction* scatter) {
  // Compute the trip count for the while loop to be used for scatter. This
  // should be the number of indices we should scatter into the operand.
  const HloInstruction* scatter_indices = scatter->scatter_indices();
  const Shape& scatter_indices_shape = scatter_indices->shape();
  const ScatterDimensionNumbers& dim_numbers =
      scatter->scatter_dimension_numbers();
  int64_t scatter_loop_trip_count = 1;
  for (int64_t i = 0, e = scatter_indices_shape.dimensions_size(); i < e; i++) {
    if (i != dim_numbers.index_vector_dim()) {
      scatter_loop_trip_count *= scatter_indices_shape.dimensions(i);
    }
  }
  return scatter_loop_trip_count;
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

StatusOr<HloInstruction*> ScatterExpander::ExpandInstruction(
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
  int64_t scatter_loop_trip_count = ScatterTripCount(scatter);
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
  StatusOr<std::vector<HloInstruction*>> scatter_loop_result_status =
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
  return scatter &&
         (mode_ == kEliminateAllScatters || ScatterTripCount(scatter) == 1);
}

}  // namespace xla
