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

#include "xla/service/gather_expander.h"

#include <cstdint>
#include <iterator>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/while_util.h"
#include "xla/util.h"

namespace xla {

namespace {
absl::StatusOr<HloInstruction*> TransposeIndexVectorDimToLast(
    HloInstruction* start_indices, int64_t index_vector_dim) {
  const Shape& start_indices_shape = start_indices->shape();

  if (start_indices_shape.dimensions_size() == index_vector_dim) {
    return start_indices;
  }

  if (index_vector_dim == (start_indices_shape.dimensions_size() - 1)) {
    return start_indices;
  }

  std::vector<int64_t> permutation;
  permutation.reserve(start_indices_shape.dimensions_size());
  for (int64_t i = 0, e = start_indices_shape.dimensions_size(); i < e; i++) {
    if (i != index_vector_dim) {
      permutation.push_back(i);
    }
  }
  permutation.push_back(index_vector_dim);
  return MakeTransposeHlo(start_indices, permutation);
}

// Canonicalizes the start_indices tensors so that we only have deal with some
// specific cases in the while loop that does the heavy lifting.
//
// See the "High Level Algorithm" section for a broader picture.
absl::StatusOr<HloInstruction*> CanonicalizeGatherIndices(
    HloInstruction* start_indices, int64_t index_vector_dim) {
  // Transpose the non-index-vector dimensions to the front.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * transposed_start_indices,
      TransposeIndexVectorDimToLast(start_indices, index_vector_dim));
  bool indices_are_scalar =
      index_vector_dim == start_indices->shape().dimensions_size();

  // The number of dimensions in start_indices that are index dimensions.
  const int64_t index_dims_in_start_indices = indices_are_scalar ? 0 : 1;

  // If there is only one index (i.e. start_indices has rank 1 and this gather
  // is really just a dynamic slice) add a leading degenerate dimension for
  // uniformity.  Otherwise create a "collapsed" leading dimension that subsumes
  // all of the non-index-vector dimensions.
  const Shape& shape = transposed_start_indices->shape();
  if (shape.dimensions_size() == index_dims_in_start_indices) {
    return PrependDegenerateDims(transposed_start_indices, 1);
  } else {
    // Collapse all but the dimensions (0 or 1) in start_indices containing the
    // index vectors.
    return CollapseFirstNDims(
        transposed_start_indices,
        shape.dimensions_size() - index_dims_in_start_indices);
  }
}

// Expands out or contracts away the gather dimensions in the accumulator
// produced by the while loop.
absl::StatusOr<HloInstruction*> AdjustBatchDimsInAccumulator(
    const Shape& start_indices_shape, HloInstruction* accumulator,
    int64_t index_vector_dim) {
  std::vector<int64_t> batch_dim_bounds;
  batch_dim_bounds.reserve(start_indices_shape.dimensions_size());
  for (int64_t i = 0, e = start_indices_shape.dimensions_size(); i < e; i++) {
    if (i != index_vector_dim) {
      batch_dim_bounds.push_back(start_indices_shape.dimensions(i));
    }
  }

  if (batch_dim_bounds.empty()) {
    // If batch_dim_bounds is empty we must be lowering a (effectively)
    // dynamic-slice.  In that case, there is a leading degenerate gather
    // dimension that we added to make this special case play well with the
    // general while loop which we need to remove now.
    return ElideDegenerateDims(accumulator, {0});
  }

  return ExpandFirstDimIntoNDims(accumulator, batch_dim_bounds);
}

// Generates the HLO to calculate the implicit and explicit batch dimension
// indices and returns the explicit batch dimension to the HLO indices in the
// order of major to minor.
std::vector<HloInstruction*> GenerateExplicitBatchDimIndices(
    const Shape& start_indices_shape, const GatherDimensionNumbers& dim_numbers,
    HloInstruction* induction_var) {
  if (dim_numbers.operand_batching_dims().empty()) {
    return {};
  }

  int64_t index_vector_dim = dim_numbers.index_vector_dim();
  int64_t rank = start_indices_shape.dimensions_size();
  int64_t num_batch_dims = (rank == index_vector_dim) ? rank : rank - 1;
  HloComputation* computation = induction_var->parent();
  HloInstruction* divident = induction_var;
  const Shape& shape = induction_var->shape();

  absl::Span<const int64_t> start_indices_batching_dims =
      dim_numbers.start_indices_batching_dims();
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

// Expands an index vector from the start_indices tensor into a vector that can
// be used to dynamic-slice out of the gather operand.
absl::StatusOr<HloInstruction*> ExpandIndexVectorIntoOperandSpace(
    const Shape& start_indices_shape, HloInstruction* index_vector,
    const GatherDimensionNumbers& dim_numbers, int64_t operand_rank,
    HloInstruction* induction_var) {
  HloComputation* computation = index_vector->parent();
  const Shape& index_shape = index_vector->shape();

  if (operand_rank == 0) {
    // This is Gather from a scalar. So, the index vector in operand space must
    // be a zero-sized vector.
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
      GenerateExplicitBatchDimIndices(start_indices_shape, dim_numbers,
                                      induction_var);
  int64_t seen_explicit_batch_dims = 0;
  for (int i = 0; i < operand_rank; i++) {
    int64_t index_vector_dim_index =
        FindIndex(dim_numbers.start_index_map(), i);
    if (index_vector_dim_index != dim_numbers.start_index_map_size()) {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * component_to_concat,
          MakeSliceHlo(index_vector, /*start_indices=*/{index_vector_dim_index},
                       /*limit_indices=*/{index_vector_dim_index + 1},
                       /*strides=*/{1}));
      expanded_index_components.push_back(component_to_concat);
    } else {
      if (absl::c_linear_search(dim_numbers.operand_batching_dims(), i)) {
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

// Returns the dimensions in a slice that are either collapsed or corresponding
// to an operand batching dimension.
std::vector<int64_t> GetDegeneratedSliceDims(
    const GatherDimensionNumbers& dim_numbers) {
  absl::Span<const int64_t> collapsed_slice_dims =
      dim_numbers.collapsed_slice_dims();
  absl::Span<const int64_t> batching_dims = dim_numbers.operand_batching_dims();
  std::vector<int64_t> removed_dims;
  removed_dims.reserve(collapsed_slice_dims.size() + batching_dims.size());
  absl::c_copy(collapsed_slice_dims, std::back_inserter(removed_dims));
  absl::c_copy(batching_dims, std::back_inserter(removed_dims));
  return removed_dims;
}

// This generates the body of the while that implements the main data movement
// behavior of gather using dynamic-slice and dynamic-update-slice.
absl::StatusOr<std::vector<HloInstruction*>> GatherLoopBody(
    const HloInstruction& gather, HloInstruction* induction_var,
    const std::vector<HloInstruction*>& incoming_loop_state) {
  const GatherDimensionNumbers& dim_numbers = gather.gather_dimension_numbers();
  CHECK_EQ(incoming_loop_state.size(), 3);
  HloInstruction* const operand = incoming_loop_state[0];
  HloInstruction* const start_indices = incoming_loop_state[1];
  HloInstruction* const output_accumulator = incoming_loop_state[2];
  const Shape& orig_start_indices_shape = gather.operand(1)->shape();

  bool has_scalar_indices = start_indices->shape().dimensions_size() == 1;
  CHECK_EQ(has_scalar_indices, dim_numbers.index_vector_dim() ==
                                   orig_start_indices_shape.dimensions_size());

  HloInstruction* induction_var_as_vector =
      MakeBroadcastHlo(induction_var, /*broadcast_dimensions=*/{},
                       /*result_shape_bounds=*/{1});

  HloInstruction* index_vector;

  if (has_scalar_indices) {
    // In this case start_indices has rank 1 and induction_var_as_vector (of
    // shape {1}) is an index into this rank 1 tensor.
    TF_ASSIGN_OR_RETURN(
        index_vector,
        MakeDynamicSliceHlo(start_indices, induction_var_as_vector, {1}));
  } else {
    // In this case start_indices has rank 2 and induction_var_as_vector (of
    // shape {1}) is an index into just the first dimension of this rank 2
    // tensor.
    TF_ASSIGN_OR_RETURN(
        HloInstruction * index_into_start_indices,
        PadVectorWithZeros(induction_var_as_vector,
                           /*zeros_to_prepend=*/0, /*zeros_to_append=*/1));

    int64_t index_vector_size = start_indices->shape().dimensions(1);
    TF_ASSIGN_OR_RETURN(
        HloInstruction * index_vector_2d,
        MakeDynamicSliceHlo(start_indices, index_into_start_indices,
                            {1, index_vector_size}));

    TF_ASSIGN_OR_RETURN(index_vector,
                        ElideDegenerateDims(index_vector_2d, {0}));
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * gathered_slice_start,
                      ExpandIndexVectorIntoOperandSpace(
                          orig_start_indices_shape, index_vector, dim_numbers,
                          operand->shape().dimensions_size(), induction_var));

  TF_ASSIGN_OR_RETURN(HloInstruction * gathered_slice,
                      MakeDynamicSliceHlo(operand, gathered_slice_start,
                                          gather.gather_slice_sizes()));

  TF_ASSIGN_OR_RETURN(
      HloInstruction* const gathered_slice_with_dims_collapsed,
      ElideDegenerateDims(gathered_slice,
                          GetDegeneratedSliceDims(dim_numbers)));

  TF_ASSIGN_OR_RETURN(
      HloInstruction* const gathered_slice_for_update,
      PrependDegenerateDims(gathered_slice_with_dims_collapsed, 1));

  TF_ASSIGN_OR_RETURN(
      HloInstruction* const index_vector_into_accumulator,
      PadVectorWithZeros(
          induction_var_as_vector, /*zeros_to_prepend=*/0,
          /*zeros_to_append=*/
          gathered_slice_with_dims_collapsed->shape().dimensions_size()));

  TF_ASSIGN_OR_RETURN(
      HloInstruction* const updated_accumulator,
      MakeDynamicUpdateSliceHlo(output_accumulator, gathered_slice_for_update,
                                index_vector_into_accumulator));

  // New loop state -- only the accumulator has changed.  The
  // WhileUtil::MakeCountedLoop functions takes care of the induction variable
  // and the while loop exit condition.
  return absl::StatusOr<std::vector<HloInstruction*>>{
      {operand, start_indices, updated_accumulator}};
}

HloInstruction* CreateGatherLoopAccumulatorInitValue(
    HloComputation* computation, PrimitiveType element_type,
    absl::Span<const int64_t> slice_sizes, int64_t gather_loop_trip_count,
    const GatherDimensionNumbers& dim_numbers) {
  std::vector<int64_t> accumulator_state_shape_dims;
  accumulator_state_shape_dims.reserve(1 + slice_sizes.size());
  accumulator_state_shape_dims.push_back(gather_loop_trip_count);
  for (int64_t i = 0; i < slice_sizes.size(); i++) {
    if (!absl::c_linear_search(dim_numbers.collapsed_slice_dims(), i) &&
        !absl::c_linear_search(dim_numbers.operand_batching_dims(), i)) {
      accumulator_state_shape_dims.push_back(slice_sizes[i]);
    }
  }
  return BroadcastZeros(computation, element_type,
                        accumulator_state_shape_dims);
}

// `accumulator` is almost the tensor the gather operation would have produced,
// except that it has the dimensions in the wrong order -- the batch dimensions
// are the major dimensions and the offset dimensions are the minor dimensions.
// Fix this up with a transpose.
absl::StatusOr<HloInstruction*> PermuteBatchAndOffsetDims(
    HloInstruction* accumulator, absl::Span<const int64_t> offset_dims,
    int64_t output_rank) {
  std::vector<int64_t> permutation;
  permutation.reserve(output_rank);

  int64_t batch_idx_counter = 0;
  int64_t offset_idx_counter = output_rank - offset_dims.size();
  for (int64_t i = 0; i < output_rank; i++) {
    bool is_offset_dim = absl::c_binary_search(offset_dims, i);
    if (is_offset_dim) {
      permutation.push_back(offset_idx_counter++);
    } else {
      permutation.push_back(batch_idx_counter++);
    }
  }

  return MakeTransposeHlo(accumulator, permutation);
}

// Computes how many trips a loop implementing this gather op would take.
int64_t GatherLoopTripCount(HloInstruction* gather_instr) {
  HloInstruction* start_indices = gather_instr->mutable_operand(1);
  const Shape& start_indices_shape = start_indices->shape();
  const GatherDimensionNumbers& dim_numbers =
      gather_instr->gather_dimension_numbers();

  int64_t trip_count = 1;
  for (int64_t i = 0, e = start_indices_shape.dimensions_size(); i < e; i++) {
    if (i != dim_numbers.index_vector_dim()) {
      trip_count *= start_indices_shape.dimensions(i);
    }
  }
  return trip_count;
}

bool GatherIsBroadcast(HloInstruction* gather_instr) {
  return absl::c_equal(gather_instr->gather_slice_sizes(),
                       gather_instr->operand(0)->shape().dimensions());
}
}  // namespace
// High Level Algorithm
//
// We follow the following steps in sequence:
//
//  1. We canonicalize the start_indices tensor such that it has rank
//     2 (i.e. is a matrix) where each row is an index vector into the
//     operand.
//  2. We iterate over the set of indices in the canonicalized
//     start_indices tensor using a while loop, accumulating slices
//     of the operand tensor into an accumulator using
//     DynamicUpdateSlice.
//  3. The accumulator result from the while loop from (2) is then
//     reshaped to split out all the individual gather dimensions and
//     then transposed to give the final result.
//
// As an example, if we started with the following operation:
//
//   HloModule TensorFlowGatherMultipleBatchDims
//
//   ENTRY main {
//     operand = s32[3,3] parameter(0)
//     indices = s32[2,2] parameter(1)
//     ROOT gather = s32[2,3,2] gather(operand, indices),
//         offset_dims={1},
//         collapsed_slice_dims={1},
//         start_index_map={1},
//         index_vector_dim=2,
//         slice_sizes={3, 1}
//   }
//
// We'd first reshape indices to s32[4,1], where each row is an index
// into operand.  We'd then run a loop to slice out 4 tensors of shape
// [3,1] out of operand into an accumulator of shape [4,3,1].  We then
// reshape this result to [2,2,3] and finally transpose it to [2,3,2].

absl::StatusOr<HloInstruction*> GatherExpander::ExpandInstruction(
    HloInstruction* gather_instr) {
  CHECK(!ShapeUtil::IsZeroElementArray(gather_instr->shape()));

  if (GatherIsBroadcast(gather_instr)) {
    if (ShapeUtil::IsZeroElementArray(gather_instr->operand(0)->shape())) {
      return MakeScalarLike(gather_instr, 0);
    }
    Shape broadcast_operand_shape = ShapeUtil::DeleteDimensions(
        GetDegeneratedSliceDims(gather_instr->gather_dimension_numbers()),
        gather_instr->operand(0)->shape());
    TF_ASSIGN_OR_RETURN(HloInstruction * broadcast_operand,
                        MakeReshapeHlo(broadcast_operand_shape,
                                       gather_instr->mutable_operand(0)));
    gather_instr->SetupDerivedInstruction(broadcast_operand);
    HloInstruction* broadcast =
        MakeBroadcastHlo(broadcast_operand,
                         gather_instr->gather_dimension_numbers().offset_dims(),
                         gather_instr->shape());
    gather_instr->SetupDerivedInstruction(broadcast);
    return broadcast;
  }

  HloComputation* computation = gather_instr->parent();
  HloInstruction* operand = gather_instr->mutable_operand(0);
  HloInstruction* start_indices = gather_instr->mutable_operand(1);
  const Shape& output_shape = gather_instr->shape();
  int64_t output_rank = output_shape.dimensions_size();

  const GatherDimensionNumbers& dim_numbers =
      gather_instr->gather_dimension_numbers();

  int64_t gather_loop_trip_count = GatherLoopTripCount(gather_instr);
  if (!IsInt32(gather_loop_trip_count)) {
    return Unimplemented(
        "Gather operations with more than 2147483647 gather indices are not "
        "supported. This error occurred for %s.",
        gather_instr->ToString());
  }

  TF_ASSIGN_OR_RETURN(
      HloInstruction * canonical_start_indices,
      CanonicalizeGatherIndices(start_indices, dim_numbers.index_vector_dim()));

  CHECK_EQ(gather_loop_trip_count,
           canonical_start_indices->shape().dimensions(0));

  HloInstruction* accumulator_init = CreateGatherLoopAccumulatorInitValue(
      computation, output_shape.element_type(),
      gather_instr->gather_slice_sizes(), gather_loop_trip_count,
      gather_instr->gather_dimension_numbers());

  absl::StatusOr<std::vector<HloInstruction*>> gather_loop_result_or_error =
      WhileUtil::MakeCountedLoop(
          computation, gather_loop_trip_count,
          {operand, canonical_start_indices, accumulator_init},
          [&](HloInstruction* indvar,
              const std::vector<HloInstruction*>& loop_state) {
            return GatherLoopBody(*gather_instr, indvar, loop_state);
          },
          gather_instr->metadata());

  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> gather_loop_result,
                      gather_loop_result_or_error);

  HloInstruction* accumulator_result = gather_loop_result.back();

  TF_ASSIGN_OR_RETURN(
      HloInstruction* const accumulator_with_batch_dims_decanonicalized,
      AdjustBatchDimsInAccumulator(start_indices->shape(), accumulator_result,
                                   dim_numbers.index_vector_dim()));

  return PermuteBatchAndOffsetDims(accumulator_with_batch_dims_decanonicalized,
                                   dim_numbers.offset_dims(), output_rank);
}

bool GatherExpander::InstructionMatchesPattern(HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kGather &&
         // Avoid expanding gather ops that produce zero sized tensors,
         // instead punt these to ZeroSizedHloElimination.
         !ShapeUtil::IsZeroElementArray(inst->shape()) &&
         // In kEliminateSimpleGathers mode, we only simplify instructions
         // which can be represented without a loop -- i.e. we only simplify
         // gathers which have a trip count of 1.
         (mode_ == kEliminateAllGathers || GatherLoopTripCount(inst) == 1 ||
          GatherIsBroadcast(inst));
}

}  // namespace xla
