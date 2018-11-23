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
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/while_util.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {


// Transposes the given scatter_indices such that the index_vector_dim becomes
// the most-minor dimension.
static StatusOr<HloInstruction*> TransposeIndexVectorDimToLast(
    HloInstruction* scatter_indices, int64 index_vector_dim) {
  const Shape& scatter_indices_shape = scatter_indices->shape();

  if (scatter_indices_shape.dimensions_size() == index_vector_dim) {
    return scatter_indices;
  }

  if (index_vector_dim == (scatter_indices_shape.dimensions_size() - 1)) {
    return scatter_indices;
  }

  std::vector<int64> permutation;
  permutation.reserve(scatter_indices_shape.dimensions_size());
  for (int64 i = 0, e = scatter_indices_shape.dimensions_size(); i < e; i++) {
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
    HloInstruction* scatter_indices, int64 index_vector_dim) {
  // Transpose the non-index-vector dimensions to the front.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * transposed_scatter_indices,
      TransposeIndexVectorDimToLast(scatter_indices, index_vector_dim));
  bool indices_are_scalar =
      index_vector_dim == scatter_indices->shape().dimensions_size();

  // The number of dimensions in scatter_indices that are index dimensions.
  const int64 index_dims_in_scatter_indices = indices_are_scalar ? 0 : 1;

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
    HloInstruction* updates, absl::Span<const int64> update_window_dims) {
  std::vector<int64> permutation;
  const int64 updates_rank = ShapeUtil::Rank(updates->shape());
  permutation.reserve(updates_rank);

  for (int64 i = 0; i < updates_rank; ++i) {
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
    int64 index_vector_dim) {
  int64 num_scatter_dims = scatter_indices_shape.dimensions_size();
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
    int64 operand_rank) {
  HloComputation* computation = index_vector->parent();
  const Shape& index_shape = index_vector->shape();
  HloInstruction* zero =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateFromDimensions(index_shape.element_type(), {1})));

  // We extract out individual components from the smaller index and concatenate
  // them (interspersing zeros as needed) into the larger index.
  std::vector<HloInstruction*> expanded_index_components;

  for (int i = 0; i < operand_rank; i++) {
    int64 index_vector_dim_index =
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
    absl::Span<const int64> operand_dims, absl::Span<const int64> window_sizes,
    HloModule* module) {
  DCHECK_NE(nullptr, module);
  DCHECK_EQ(operand_dims.size(), window_sizes.size());

  // Valid range for the index: [0, operand_dims - window_sizes]

  // Check if the index has any negative values.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * zero_index,
      BroadcastZeros(computation, index->shape().element_type(),
                     AsInt64Slice(index->shape().dimensions())));
  TF_ASSIGN_OR_RETURN(HloInstruction * negative_index_check,
                      MakeBinaryHlo(HloOpcode::kLe, zero_index, index));

  // Check if the index is OOB w.r.t. the operand dimensions and window sizes.
  std::vector<int64> max_valid_index(operand_dims.size());
  for (int i = 0; i < operand_dims.size(); ++i) {
    max_valid_index[i] = operand_dims[i] - window_sizes[i];
  }
  TF_ASSIGN_OR_RETURN(
      HloInstruction * max_valid_index_constant,
      MakeR1ConstantHlo<int64>(computation, index->shape().element_type(),
                               max_valid_index));
  TF_ASSIGN_OR_RETURN(
      HloInstruction * oob_index_check,
      MakeBinaryHlo(HloOpcode::kGe, max_valid_index_constant, index));

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

// Body of the while loop that performs the scatter operation using other HLOs.
static StatusOr<std::vector<HloInstruction*>> ScatterLoopBody(
    HloInstruction* scatter, HloInstruction* induction_var,
    const std::vector<HloInstruction*>& loop_state) {
  const ScatterDimensionNumbers& dim_numbers =
      scatter->scatter_dimension_numbers();
  CHECK_EQ(loop_state.size(), 3);
  HloInstruction* operand = loop_state[0];
  HloInstruction* scatter_indices = loop_state[1];
  HloInstruction* updates = loop_state[2];

  bool has_scalar_indices = scatter_indices->shape().dimensions_size() == 1;
  CHECK_EQ(has_scalar_indices,
           dim_numbers.index_vector_dim() ==
               scatter->operand(1)->shape().dimensions_size());

  // Build a vector form of the induction variable of the while loop.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * induction_var_as_vector,
      MakeBroadcastHlo(induction_var, /*broadcast_dimensions=*/{},
                       /*result_shape_bounds=*/{1}));

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
      ExpandIndexVectorIntoOperandSpace(index_vector, dim_numbers,
                                        operand->shape().dimensions_size()));

  // Extract the slice to be used to update from `updates` tensor for the
  // induction_var corresponding to this iteration of the while loop.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * index_into_updates,
      PadVectorWithZeros(
          induction_var_as_vector, /*zeros_to_prepend=*/0,
          /*zeros_to_append=*/updates->shape().dimensions_size() - 1));
  std::vector<int64> update_slice_bounds(updates->shape().dimensions().begin(),
                                         updates->shape().dimensions().end());
  update_slice_bounds[0] = 1;
  TF_ASSIGN_OR_RETURN(
      HloInstruction * update_slice,
      MakeDynamicSliceHlo(updates, index_into_updates, update_slice_bounds));
  TF_ASSIGN_OR_RETURN(HloInstruction * update_slice_for_scatter,
                      ElideDegenerateDims(update_slice, {0}));
  TF_ASSIGN_OR_RETURN(
      HloInstruction * update_slice_with_dims_inserted,
      InsertDegenerateDims(update_slice_for_scatter,
                           AsInt64Slice(dim_numbers.inserted_window_dims())));

  // Note that the following transformation assumes that both DynamicSlice and
  // DynamicUpdateSlice follow the same semantics for OOB indices. For example,
  // if there are negative indices and DynamicSlice uses "clamping" semantics,
  // then the extracted data will be "shifted". Since DynamicUpdateSlice also
  // follows the same "clamping" semantics, writing the update will also be
  // "shifted" by exactly the same amount. So, this transformation is correct as
  // long as the semantics of handling OOB indices remain the same in
  // DynamicSlice and DynamicUpdateSlice.

  // Extract the slice to update from `operand` tensor.
  const Shape& update_slice_shape = update_slice_with_dims_inserted->shape();
  TF_ASSIGN_OR_RETURN(
      HloInstruction * operand_slice_to_update,
      MakeDynamicSliceHlo(operand, scatter_slice_start,
                          AsInt64Slice(update_slice_shape.dimensions())));

  // Compute the new value for the slice to be updated in `operand` tensor by
  // combining the existing value and the update value using the update
  // computation.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * updated_operand_slice,
      MakeMapHlo({operand_slice_to_update, update_slice_with_dims_inserted},
                 scatter->to_apply()));

  TF_ASSIGN_OR_RETURN(
      HloInstruction * is_index_valid,
      CheckIndexValidity(
          operand->parent(), scatter_slice_start,
          AsInt64Slice(operand->shape().dimensions()),
          AsInt64Slice(update_slice_with_dims_inserted->shape().dimensions()),
          scatter->GetModule()));

  // Select the updated operand only if the index is valid. If not, select the
  // original value.
  TF_ASSIGN_OR_RETURN(HloInstruction * update_to_apply,
                      MakeSelectHlo(is_index_valid, updated_operand_slice,
                                    operand_slice_to_update));

  // Write the updated value of the slice into `operand` tensor.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * updated_operand,
      MakeDynamicUpdateSliceHlo(operand, update_to_apply, scatter_slice_start));

  return StatusOr<std::vector<HloInstruction*>>{
      {updated_operand, scatter_indices, updates}};
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

StatusOr<HloInstruction*> ScatterExpander::ExpandScatter(
    HloInstruction* scatter) {
  HloInstruction* operand = scatter->mutable_operand(0);
  HloInstruction* scatter_indices = scatter->mutable_operand(1);
  HloInstruction* updates = scatter->mutable_operand(2);
  const ScatterDimensionNumbers& dim_numbers =
      scatter->scatter_dimension_numbers();

  // If the updates tensor is empty, there is no need to update the operand. We
  // can return the operand as is.
  if (ShapeUtil::IsZeroElementArray(updates->shape())) {
    return operand;
  }

  // Compute the trip count for the while loop to be used for scatter. This
  // should be the number of indices we should scatter into the operand.
  const Shape& scatter_indices_shape = scatter_indices->shape();
  int64 scatter_loop_trip_count = 1;
  for (int64 i = 0, e = scatter_indices_shape.dimensions_size(); i < e; i++) {
    if (i != dim_numbers.index_vector_dim()) {
      scatter_loop_trip_count *= scatter_indices_shape.dimensions(i);
    }
  }
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
  TF_ASSIGN_OR_RETURN(
      HloInstruction * canonical_updates,
      PermuteScatterAndWindowDims(
          updates, AsInt64Slice(dim_numbers.update_window_dims())));
  TF_ASSIGN_OR_RETURN(
      HloInstruction * adjusted_canonical_updates,
      AdjustScatterDims(scatter_indices->shape(), canonical_updates,
                        dim_numbers.index_vector_dim()));
  CHECK_EQ(scatter_loop_trip_count,
           adjusted_canonical_updates->shape().dimensions(0));

  // The while loop that implements the scatter operation.
  StatusOr<std::vector<HloInstruction*>> scatter_loop_result_status =
      WhileUtil::MakeCountedLoop(
          scatter->parent(), scatter_loop_trip_count,
          {operand, canonical_scatter_indices, adjusted_canonical_updates},
          [&](HloInstruction* induction_var,
              const std::vector<HloInstruction*>& loop_state) {
            return ScatterLoopBody(scatter, induction_var, loop_state);
          },
          scatter->metadata());
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> scatter_loop_result,
                      scatter_loop_result_status);
  return scatter_loop_result.front();
}

StatusOr<bool> ScatterExpander::Run(HloModule* module) {
  std::vector<HloInstruction*> scatter_instrs;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kScatter) {
        scatter_instrs.push_back(instr);
      }
    }
  }

  for (auto instr : scatter_instrs) {
    TF_ASSIGN_OR_RETURN(HloInstruction * expanded_root, ExpandScatter(instr));
    TF_RETURN_IF_ERROR(
        instr->parent()->ReplaceInstruction(instr, expanded_root));
  }

  return !scatter_instrs.empty();
}

}  // namespace xla
