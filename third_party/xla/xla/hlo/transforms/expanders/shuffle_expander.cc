/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/shuffle_expander.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/shuffle.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// Returns the start indices of a gather that expresses `shuffle`, which holds
// for every element of the result the coordinates of the element of the
// operand that it takes its value from.
absl::StatusOr<HloInstruction*> MakeShuffleStartIndices(
    HloShuffleInstruction* shuffle) {
  HloComputation* computation = shuffle->parent();
  const OpMetadata* metadata = &shuffle->metadata();
  const Shape& operand_shape = shuffle->operand(0)->shape();
  const int64_t rank = operand_shape.dimensions().size();
  absl::Span<const int64_t> dimensions = shuffle->dimensions();
  // The indices of the shuffle hold one coordinate per shuffled dimension, and
  // an index of the gather holds one coordinate per dimension of the operand,
  // so both are laid out with the coordinates in an innermost dimension.
  Literal indices;
  switch (shuffle->mode()) {
    case ShuffleMode::kPermute: {
      ABSL_ASSIGN_OR_RETURN(indices,
                       shuffle::GetPermuteIndices(shuffle->shuffle_mode()));
      break;
    }
    default:
      return InvalidArgument(
          "Expanding a shuffle in %s mode into a gather is not supported",
          ShuffleModeToString(shuffle->mode()));
  }
  HloInstruction* coordinates = computation->AddInstruction(
      HloInstruction::CreateConstant(std::move(indices)));
  // Coordinates of dimensions that the shuffle leaves in place are added below,
  // and those run up to the size of their dimension, so the coordinates need a
  // type that holds every dimension size, which the indices need not have.
  const absl::Span<const int64_t> operand_dimensions =
      operand_shape.dimensions();
  const int64_t largest_dimension =
      operand_dimensions.empty() ? 0 : *absl::c_max_element(operand_dimensions);
  const PrimitiveType coordinate_type =
      largest_dimension <= std::numeric_limits<int32_t>::max() ? S32 : S64;
  if (coordinates->shape().element_type() != coordinate_type) {
    coordinates = MakeConvertToHlo(coordinates, coordinate_type, metadata);
  }
  // The indices may share coordinates across dimensions of size 1, and hold a
  // single coordinate directly when one dimension is shuffled, so they are
  // broadcast to the shape that holds one coordinate per shuffled dimension for
  // every element of the result.
  DimensionVector coordinates_dimensions(operand_dimensions.begin(),
                                         operand_dimensions.end());
  coordinates_dimensions.push_back(dimensions.size());
  if (coordinates->shape().dimensions() !=
      absl::Span<const int64_t>(coordinates_dimensions)) {
    // A broadcast only adds dimensions, so the dimensions of size 1 that share
    // their coordinates are reshaped away and added back by the broadcast.
    const int64_t indices_rank = coordinates->shape().dimensions().size();
    std::vector<int64_t> broadcast_dimensions;
    DimensionVector kept_dimensions;
    for (int64_t dimension = 0; dimension < indices_rank; ++dimension) {
      const int64_t size = coordinates->shape().dimensions(dimension);
      if (size == coordinates_dimensions[dimension]) {
        broadcast_dimensions.push_back(dimension);
        kept_dimensions.push_back(size);
      }
    }
    if (static_cast<int64_t>(kept_dimensions.size()) != indices_rank) {
      ABSL_ASSIGN_OR_RETURN(
          coordinates,
          MakeReshapeHlo(ShapeUtil::MakeShape(coordinate_type, kept_dimensions),
                         coordinates));
    }
    coordinates = MakeBroadcastHlo(
        coordinates, broadcast_dimensions,
        ShapeUtil::MakeShape(coordinate_type, coordinates_dimensions),
        metadata);
  }
  // The gather indexes every dimension, so a shuffle of all of them in order
  // already holds all the coordinates that the gather needs.
  if (static_cast<int64_t>(dimensions.size()) == rank &&
      absl::c_is_sorted(dimensions)) {
    return coordinates;
  }
  std::vector<int64_t> coordinate_of_dimension(rank, -1);
  for (size_t coordinate = 0; coordinate < dimensions.size(); ++coordinate) {
    coordinate_of_dimension[dimensions[coordinate]] = coordinate;
  }
  // A dimension that the shuffle leaves in place keeps the coordinate that the
  // element has in the result, which is what an iota along that dimension
  // holds; a shuffled dimension takes its coordinate from the indices.
  DimensionVector component_dimensions(operand_dimensions.begin(),
                                       operand_dimensions.end());
  component_dimensions.push_back(1);
  const Shape component_shape =
      ShapeUtil::MakeShape(coordinate_type, component_dimensions);
  DimensionVector starts(rank + 1, 0);
  DimensionVector limits(component_dimensions);
  DimensionVector strides(rank + 1, 1);
  std::vector<HloInstruction*> components;
  components.reserve(rank);
  for (int64_t dimension = 0; dimension < rank; ++dimension) {
    const int64_t coordinate = coordinate_of_dimension[dimension];
    if (coordinate < 0) {
      components.push_back(
          MakeIotaHlo(computation, component_shape, dimension));
      continue;
    }
    // A single shuffled dimension leaves the coordinates holding just its
    // coordinate, so they need no slicing to be singled out.
    if (dimensions.size() == 1) {
      components.push_back(coordinates);
      continue;
    }
    starts.back() = coordinate;
    limits.back() = coordinate + 1;
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * component,
        MakeSliceHlo(coordinates, starts, limits, strides, metadata));
    components.push_back(component);
  }
  return MakeConcatHlo(components, rank, metadata);
}

// Expands `shuffle` into a gather of one element of the operand per element of
// the result, which expresses any shuffle that moves every element on its own.
absl::StatusOr<HloInstruction*> ExpandShuffleToGather(
    HloShuffleInstruction* shuffle) {
  ABSL_ASSIGN_OR_RETURN(HloInstruction * start_indices,
                   MakeShuffleStartIndices(shuffle));
  // Each index of `start_indices` addresses a single element, so every
  // dimension of the operand is indexed, sliced down to one element and
  // collapsed, which leaves the result with the shape of the operand.
  const int64_t rank = shuffle->operand(0)->shape().dimensions().size();
  std::vector<int64_t> all_dimensions(rank);
  absl::c_iota(all_dimensions, 0);
  const GatherDimensionNumbers dimension_numbers =
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/{}, /*collapsed_slice_dims=*/all_dimensions,
          /*start_index_map=*/all_dimensions, /*index_vector_dim=*/rank);
  const std::vector<int64_t> slice_sizes(rank, 1);
  HloInstruction* gather =
      shuffle->parent()->AddInstruction(HloInstruction::CreateGather(
          shuffle->shape(), shuffle->mutable_operand(0), start_indices,
          dimension_numbers, slice_sizes, /*indices_are_sorted=*/false));
  gather->set_metadata(shuffle->metadata());
  return gather;
}

}  // namespace

bool ShuffleExpander::InstructionMatchesPattern(HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kShuffle) {
    return false;
  }
  auto* shuffle = Cast<HloShuffleInstruction>(instruction);
  switch (shuffle->mode()) {
    case ShuffleMode::kRotate:
    case ShuffleMode::kPermute:
      return true;
    default:
      return false;
  }
}

absl::StatusOr<HloInstruction*> ShuffleExpander::ExpandInstruction(
    HloInstruction* instruction) {
  auto* shuffle = Cast<HloShuffleInstruction>(instruction);
  HloInstruction* result = shuffle->mutable_operand(0);
  absl::Span<const int64_t> dimensions = shuffle->dimensions();
  switch (shuffle->mode()) {
    case ShuffleMode::kRotate: {
      const auto& shifts = shuffle->rotate().shifts();
      for (size_t idx = 0; idx < dimensions.size(); ++idx) {
        int64_t dim = dimensions[idx];
        int64_t shift = shifts[idx];
        const Shape& curr_shape = result->shape();
        int64_t rank = curr_shape.dimensions().size();

        int64_t dim_size = curr_shape.dimensions(dim);
        int64_t norm_shift = shuffle::NormalizeShift(shift, dim_size);
        if (norm_shift == 0) {
          continue;
        }

        DimensionVector start_indices_0(rank, 0);
        DimensionVector limit_indices_0(curr_shape.dimensions().begin(),
                                        curr_shape.dimensions().end());
        DimensionVector strides(rank, 1);

        start_indices_0[dim] = norm_shift;

        DimensionVector start_indices_1(rank, 0);
        DimensionVector limit_indices_1(curr_shape.dimensions().begin(),
                                        curr_shape.dimensions().end());
        limit_indices_1[dim] = norm_shift;

        ABSL_ASSIGN_OR_RETURN(HloInstruction * slice0,
                         MakeSliceHlo(result, start_indices_0, limit_indices_0,
                                      strides, &shuffle->metadata()));
        ABSL_ASSIGN_OR_RETURN(HloInstruction * slice1,
                         MakeSliceHlo(result, start_indices_1, limit_indices_1,
                                      strides, &shuffle->metadata()));

        ABSL_ASSIGN_OR_RETURN(
            result, MakeConcatHlo({slice0, slice1}, dim, &shuffle->metadata()));
      }

      return result;
    }
    case ShuffleMode::kPermute:
      // A permute moves every element on its own, which a gather of one
      // element of the operand per element of the result expresses.
      return ExpandShuffleToGather(shuffle);
    default:
      return nullptr;
  }
}

}  // namespace xla
