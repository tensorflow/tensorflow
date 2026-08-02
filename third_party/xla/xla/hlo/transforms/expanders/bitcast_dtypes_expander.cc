/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/bitcast_dtypes_expander.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

absl::StatusOr<HloInstruction*> Downcast(HloInstruction* input,
                                         const Shape& output_shape,
                                         int64_t input_bit_width,
                                         int64_t output_bit_width) {
  HloComputation* computation = input->parent();
  PrimitiveType input_logical_type =
      primitive_util::UnsignedIntegralTypeForBitWidth(input_bit_width);
  PrimitiveType output_logical_type =
      primitive_util::UnsignedIntegralTypeForBitWidth(output_bit_width);

  const Shape& input_shape = input->shape();
  int64_t rank = input_shape.dimensions().size();
  std::vector<int64_t> broadcast_dims(rank);
  absl::c_iota(broadcast_dims, 0);

  std::vector<int64_t> broadcasted_dimensions(input_shape.dimensions().begin(),
                                              input_shape.dimensions().end());
  int64_t factor = input_bit_width / output_bit_width;
  broadcasted_dimensions.push_back(factor);

  Shape broadcasted_shape =
      ShapeUtil::MakeShape(input_shape.element_type(), broadcasted_dimensions);

  HloInstruction* broadcasted_input =
      MakeBroadcastHlo(input, broadcast_dims, broadcasted_shape);
  HloInstruction* logical_input =
      MakeBitcastConvertToHlo(broadcasted_input, input_logical_type);

  HloInstruction* iota = MakeIotaHlo(computation, logical_input->shape(), rank);

  HloInstruction* scalar_output_bit_width =
      MakeScalarLike(logical_input, output_bit_width);
  ASSIGN_OR_RETURN(
      HloInstruction * iota_m,
      MakeBinaryHlo(HloOpcode::kMultiply, scalar_output_bit_width, iota));

  ASSIGN_OR_RETURN(
      HloInstruction * shifted,
      MakeBinaryHlo(HloOpcode::kShiftRightLogical, logical_input, iota_m));

  int64_t output_bit_width_mask = (int64_t{1} << output_bit_width) - 1;
  HloInstruction* scalar_mask =
      MakeScalarLike(logical_input, output_bit_width_mask);

  ASSIGN_OR_RETURN(HloInstruction * masked,
                   MakeBinaryHlo(HloOpcode::kAnd, shifted, scalar_mask));

  return MakeConvertToHlo(masked, output_logical_type);
}

// Reshapes two last dimensions of the shape into one.
absl::StatusOr<HloInstruction*> CollapseLastDimension(HloInstruction* input) {
  const Shape& shape = input->shape();
  if (shape.dimensions().empty()) {
    return absl::InvalidArgumentError("Shape has less than 1 dimensions.");
  }
  if (shape.dimensions().size() == 1) {
    return input;
  }
  std::vector<int64_t> collapsed_dims(shape.dimensions().begin(),
                                      shape.dimensions().end() - 1);
  collapsed_dims.back() *= shape.dimensions().back();
  return MakeReshapeHlo(
      ShapeUtil::MakeShape(shape.element_type(), collapsed_dims), input);
}

// Rewrites bitcast-convert from smaller to larger dtype as a series of shifts
// and ors of the input slices.
absl::StatusOr<HloInstruction*> Upcast(HloInstruction* input,
                                       const Shape& output_shape,
                                       int64_t input_bit_width,
                                       int64_t output_bit_width) {
  PrimitiveType input_logical_type =
      primitive_util::UnsignedIntegralTypeForBitWidth(input_bit_width);
  PrimitiveType output_logical_type =
      primitive_util::UnsignedIntegralTypeForBitWidth(output_bit_width);
  int64_t factor = output_bit_width / input_bit_width;

  // Reshape input to collapse the last two dimensions.
  ASSIGN_OR_RETURN(HloInstruction * collapsed_input,
                   CollapseLastDimension(input));
  const Shape& collapsed_shape = collapsed_input->shape();
  int64_t collapsed_rank = collapsed_shape.dimensions().size();

  // Initialize start and limit indices for slicing.
  std::vector<int64_t> start_indices(collapsed_rank, 0);
  std::vector<int64_t> strides(collapsed_rank, 1);
  strides.back() = factor;
  std::vector<int64_t> limit_indices(collapsed_shape.dimensions().begin(),
                                     collapsed_shape.dimensions().end());
  limit_indices.back() -= factor;

  // Slice, convert and shift the input and accumulate the result using `or`.
  HloInstruction* acc = nullptr;
  for (int dim_index = 0; dim_index < factor; ++dim_index) {
    start_indices.back() = dim_index;
    limit_indices.back() += 1;
    ASSIGN_OR_RETURN(
        HloInstruction * slice,
        MakeSliceHlo(collapsed_input, start_indices, limit_indices, strides));
    HloInstruction* logical_slice =
        MakeBitcastConvertToHlo(slice, input_logical_type);
    HloInstruction* converted_slice =
        MakeConvertToHlo(logical_slice, output_logical_type);

    if (dim_index == 0) {
      acc = converted_slice;
      continue;
    }

    HloInstruction* scalar_shift =
        MakeScalarLike(converted_slice, dim_index * input_bit_width);
    ASSIGN_OR_RETURN(
        HloInstruction * shifted_slice,
        MakeBinaryHlo(HloOpcode::kShiftLeft, converted_slice, scalar_shift));
    ASSIGN_OR_RETURN(acc, MakeBinaryHlo(HloOpcode::kOr, acc, shifted_slice));
  }
  return input->shape().dimensions().size() > 1
             ? acc
             : MakeReshapeHlo(ShapeUtil::MakeShape(output_logical_type, {}),
                              acc);
}

}  // namespace

absl::StatusOr<HloInstruction*> BitcastDtypesExpander::ExpandInstruction(
    HloInstruction* instruction) {
  HloInstruction* input = instruction->mutable_operand(0);
  const Shape& input_shape = input->shape();
  const Shape& output_shape = instruction->shape();

  int input_bit_width =
      primitive_util::StorageBitWidth(input_shape.element_type());
  int output_bit_width =
      primitive_util::StorageBitWidth(output_shape.element_type());

  if (input_bit_width == output_bit_width) {
    return instruction;
  }

  ASSIGN_OR_RETURN(
      HloInstruction * dtype_cast,
      input_bit_width > output_bit_width
          ? Downcast(input, output_shape, input_bit_width, output_bit_width)
          : Upcast(input, output_shape, input_bit_width, output_bit_width));

  return MakeBitcastConvertToHlo(dtype_cast, output_shape.element_type());
}

bool BitcastDtypesExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kBitcastConvert &&
         primitive_util::StorageBitWidth(instruction->shape().element_type()) !=
             primitive_util::StorageBitWidth(
                 instruction->operand(0)->shape().element_type());
}

}  // namespace xla
