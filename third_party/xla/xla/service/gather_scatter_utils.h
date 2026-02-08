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

#ifndef XLA_SERVICE_GATHER_SCATTER_UTILS_H_
#define XLA_SERVICE_GATHER_SCATTER_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape.h"

namespace xla {

// Transforms the given index tensor to make it two-dimensional, with the index
// vector dimension being dimension 1.
// Example:
//   input: indices = tensor<4x2x3xi32>, index_vector_dim = 1
//   output: tensor<12x2xi32>
absl::StatusOr<HloInstruction*> TransformStartIndices(HloInstruction* indices,
                                                      int64_t index_vector_dim);

// Given a map from index vector positions to dimension numbers, returns a pair
// of permutations that when applied to the operand, let you replace the map
// with the identity permutation.
// In gather, the map is called `start_index_map`. In scatter, it's
// `scatter_dims_to_operand_dims`.
std::pair<std::vector<int64_t>, std::vector<int64_t>>
MakeOperandStartIndexPermutations(absl::Span<const int64_t>, int operand_rank);

absl::StatusOr<HloInstruction*> MaybeTranspose(
    HloInstruction* operand, absl::Span<const int64_t> permutation);

absl::StatusOr<std::vector<HloInstruction*>> MaybeTranspose(
    absl::Span<HloInstruction* const> operands,
    const std::vector<int64_t>& operand_permutation);

// Moves the given dimension to the last dimension.
// Example: MoveDimensionToEnd(tensor<1x2x3xi1>, 0): tensor<2x3x1xi1>.
absl::StatusOr<HloInstruction*> MoveDimensionToEnd(HloInstruction* operand,
                                                   size_t dimension,
                                                   size_t rank);

// Expands an index vector from the start_indices tensor into a vector that can
// be used to dynamic-slice out of the gather/scatter operand.
absl::StatusOr<HloInstruction*> ExpandIndexVectorIntoOperandSpace(
    const Shape& start_indices_shape, int64_t operand_rank,
    int64_t index_vector_dim, absl::Span<const int64_t> start_index_map,
    absl::Span<const int64_t> start_indices_batching_dims,
    absl::Span<const int64_t> operand_batching_dims,
    HloInstruction* index_vector, HloInstruction* induction_var);

// Returns true if the given dimension is a collapsed or batching dimension.
bool IsCollapsedOrBatchingDim(absl::Span<const int64_t> collapsed_dims,
                              absl::Span<const int64_t> batching_dims,
                              int64_t dim);

// Returns a map from start_indices explicit batching dims to their
// corresponding output dims.
absl::flat_hash_map<int64_t, int64_t>
GetStartIndicesDimToOutputDimForExplicitBatchingDims(
    absl::Span<const int64_t> start_indices_batching_dims,
    int64_t index_vector_dim, absl::Span<const int64_t> offset_dims,
    int64_t start_indices_rank, int64_t output_rank);

// Reshapes the gather indices input to have a trailing degenerate `1` dimension
// if necessary.  Hands over the ownership of the newly created literal (if
// there is one) to `reshaped_start_indices`.
absl::StatusOr<std::reference_wrapper<const Literal>> ReshapedGatherIndices(
    int64_t index_vector_dim, const Literal& start_indices,
    Literal* reshaped_start_indices);

// Returns an ShapeUtil::IndexIterationSpace that iterates over the output batch
// dimensions while keeping the rest of the output dimensions clamped to 0.
ShapeUtil::IndexIterationSpace IterationSpaceForOutputBatchIndices(
    const Shape& output_shape, const GatherDimensionNumbers& dim_numbers);

// Return an ShapeUtil::IndexIterationSpace that iterates over the output slice
// dimensions while keeping the rest of the output dimensions clamped to 0.
ShapeUtil::IndexIterationSpace IterationSpaceForOutputOffsetIndices(
    int64_t output_rank, absl::Span<const int64_t> slice_sizes,
    const GatherDimensionNumbers& dim_numbers);

// This functor computes the contribution of start_indices to an input index
// corresponding to an output index.  That is, given an output index I, it picks
// out the batch indices in I and uses them to look up a starting index, G, from
// the start indices tensor, and expands G into the input space according to
// start_index_map.
class OutputBatchIndexToInputIndex {
 public:
  // The constructor does some setup work that is amortized across all
  // iterations.
  explicit OutputBatchIndexToInputIndex(
      const GatherDimensionNumbers* dim_numbers, const Shape& input_shape,
      const Shape& output_shape, const Literal* start_indices)
      : dim_numbers_(*dim_numbers), start_indices_(*start_indices) {
    for (int64_t i = 0; i < output_shape.dimensions().size(); i++) {
      output_dim_is_batch_dims_.push_back(
          !absl::c_binary_search(dim_numbers_.offset_dims(), i));
    }

    for (int64_t i = 0; i < input_shape.dimensions().size(); i++) {
      int64_t index_of_input_dim_in_index_vector =
          std::distance(dim_numbers_.start_index_map().begin(),
                        absl::c_find(dim_numbers_.start_index_map(), i));
      if (index_of_input_dim_in_index_vector ==
          dim_numbers_.start_index_map_size()) {
        input_dim_value_to_index_vector_.push_back(-1);
      } else {
        input_dim_value_to_index_vector_.push_back(
            index_of_input_dim_in_index_vector);
      }
    }

    index_vector_index_.resize(start_indices_.shape().dimensions().size());
    input_index_.resize(input_shape.dimensions().size());
    int64_t index_vector_size =
        start_indices_.shape().dimensions(dim_numbers_.index_vector_dim());
    index_vector_.resize(index_vector_size);

    absl::flat_hash_map<int64_t, int64_t> start_indices_dims_to_output_dims =
        GetStartIndicesDimToOutputDimForExplicitBatchingDims(
            dim_numbers_.start_indices_batching_dims(),
            dim_numbers_.index_vector_dim(), dim_numbers_.offset_dims(),
            start_indices_.shape().dimensions().size(),
            output_shape.dimensions().size());
    for (int64_t i = 0; i < dim_numbers->operand_batching_dims().size(); ++i) {
      int64_t operand_dim = dim_numbers->operand_batching_dims(i);
      int64_t start_indices_dim = dim_numbers->start_indices_batching_dims(i);
      int64_t output_dim = start_indices_dims_to_output_dims[start_indices_dim];
      explicit_batch_dims_operand_dim_to_output_dim_[operand_dim] = output_dim;
    }
  }

  // Returns the contribution of start_indices to the input index corresponding
  // to output_index.  See gather_inner_loop_body.
  //
  // This is conceptually  a stateless transformation from output_index to the
  // gather input index, but:
  //
  //  - Instead of allocating memory to represent the gather input index on
  //    every invocation we reuse the same storage for the result
  //    (input_index_), mutating it in place.
  //  - Instead of allocating buffers for temporary values like
  //    index_vector_index_ and index_vector on every invocation, we reuse the
  //    same storage for all invocations.
  //
  // This returns a Span into memory owned by the class.
  absl::StatusOr<absl::Span<const int64_t>> operator()(
      absl::Span<const int64_t> output_index) {
    PropagateOutputIndexGatherDimsToIndexVectorIndex(output_index);
    TF_RETURN_IF_ERROR(FetchIndexVector());
    PropagateIndexVectorToInputIndex();
    PropagateExplicitBatchDimsToInputIndex(output_index);
    return absl::Span<const int64_t>(input_index_);
  }

 private:
  // Propagates the batch dimensions from the output index into
  // index_vector_index_ by mutating index_vector_index_ in place.  Does not
  // update the dim_numbers.index_vector_dim() dimension -- that's the dimension
  // we iterate over in FetchIndexVector.
  void PropagateOutputIndexGatherDimsToIndexVectorIndex(
      absl::Span<const int64_t> output_index) {
    int64_t index_vector_index_i = 0;
    for (int64_t i = 0, e = output_index.size(); i < e; i++) {
      if (!output_dim_is_batch_dims_[i]) {
        continue;
      }

      if (index_vector_index_i == dim_numbers_.index_vector_dim()) {
        index_vector_index_i++;
      }

      index_vector_index_[index_vector_index_i++] = output_index[i];
    }
  }

  // Populates index_vector_ by iterating over start_indices_ according to
  // index_vector_index_.
  absl::Status FetchIndexVector() {
    int64_t index_vector_dim = dim_numbers_.index_vector_dim();
    for (int64_t i = 0, e = index_vector_.size(); i < e; i++) {
      index_vector_index_[index_vector_dim] = i;
      auto start_index = start_indices_.GetIntegralAsS64(index_vector_index_);
      TF_RET_CHECK(start_index.has_value());
      index_vector_[i] = *start_index;
    }
    return absl::OkStatus();
  }

  // Populates input_index_.
  void PropagateIndexVectorToInputIndex() {
    for (int64_t i = 0, e = input_index_.size(); i < e; i++) {
      if (input_dim_value_to_index_vector_[i] != -1) {
        input_index_[i] = index_vector_[input_dim_value_to_index_vector_[i]];
      }

      // If input_dim_value_to_index_vector_[i] == -1 then input_index_[i]
      // remains 0, as set by the constructor.
    }
  }

  void PropagateExplicitBatchDimsToInputIndex(
      absl::Span<const int64_t> output_index) {
    for (const auto [operand_dim, output_dim] :
         explicit_batch_dims_operand_dim_to_output_dim_) {
      input_index_[operand_dim] = output_index[output_dim];
    }
  }

  // input_dim_value_to_index_vector_[i] tells us how to compute dimension i of
  // the input index from the index vector.  See
  // PropagateIndexVectorToInputIndex.
  std::vector<int64_t> input_dim_value_to_index_vector_;

  // output_dim_is_batch_dims_[i] is true iff the output index i is a gather
  // dimension.
  std::vector<bool> output_dim_is_batch_dims_;

  // The buffer into which we construct an index into start_indices_ to fetch
  // the index vector.
  std::vector<int64_t> index_vector_index_;

  // The index vector fetched from start_indices_.
  std::vector<int64_t> index_vector_;

  // The result computed by this functor.  operator() returns a Span into
  // this vector.
  std::vector<int64_t> input_index_;

  absl::flat_hash_map<int64_t, int64_t>
      explicit_batch_dims_operand_dim_to_output_dim_;

  const GatherDimensionNumbers& dim_numbers_;
  const Literal& start_indices_;
};

// This functor computes the contribution of the offset indices in an output
// index to an input index.  That is, given an output index I it picks out the
// output offset indices in I and expands it into an index into the input shape.
class OutputOffsetIndexToInputIndex {
 public:
  // The constructor does some setup work that is amortized across all
  // iterations.
  explicit OutputOffsetIndexToInputIndex(
      const GatherDimensionNumbers& dim_numbers, const Shape& input_shape) {
    CHECK(absl::c_is_sorted(dim_numbers.offset_dims()));
    int64_t window_dim_count = 0;
    for (int64_t i = 0; i < input_shape.dimensions().size(); i++) {
      if (IsCollapsedOrBatchingDim(dim_numbers.collapsed_slice_dims(),
                                   dim_numbers.operand_batching_dims(), i)) {
        input_dim_value_to_output_index_.push_back(-1);
      } else {
        input_dim_value_to_output_index_.push_back(
            dim_numbers.offset_dims()[window_dim_count++]);
      }
    }

    input_index_.resize(input_shape.dimensions().size());
  }

  // Returns the contribution of the window indices to the input index
  // corresponding to output_index.  See gather_inner_loop_body.
  //
  // This is conceptually a stateless transformation from output_index to the
  // window input index, but instead of allocating memory to represent the
  // gather input index on every invocation we reuse the same storage for the
  // result (input_index_), mutating it in place.
  //
  // This returns a Span into memory owned by the class.
  absl::StatusOr<absl::Span<const int64_t>> operator()(
      absl::Span<const int64_t> output_index) {
    PropagateOutputIndexWindowDimsToInputIndex(output_index);
    return absl::Span<const int64_t>(input_index_);
  }

  // Returns for a given 'input_dim' the corresponding output dimension index,
  // or -1 if 'input_dim' is an elided window dimension.
  int64_t input_dim_value_to_output_index(int64_t input_dim) {
    return input_dim_value_to_output_index_[input_dim];
  }

 private:
  // Propagates window dimensions from the output index to input_index_ by
  // mutating input_index_ in place.
  void PropagateOutputIndexWindowDimsToInputIndex(
      absl::Span<const int64_t> output_index) {
    for (int64_t i = 0, e = input_index_.size(); i < e; i++) {
      if (input_dim_value_to_output_index_[i] != -1) {
        input_index_[i] = output_index[input_dim_value_to_output_index_[i]];
      }

      // If input_dim_value_to_index_vector_[i] == -1 then input_index_[i]
      // remains 0, as set by the constructor.
    }
  }

  // input_dim_value_to_index_vector_[i] tells us how to compute dimension i of
  // the input index from the output index. See
  // PropagateOutputIndexWindowDimsToInputIndex.
  std::vector<int64_t> input_dim_value_to_output_index_;

  // The result computed by this functor.  operator() returns a Span into
  // this vector.
  std::vector<int64_t> input_index_;
};

}  // namespace xla

#endif  // XLA_SERVICE_GATHER_SCATTER_UTILS_H_
