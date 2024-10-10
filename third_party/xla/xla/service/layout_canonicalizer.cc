/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/layout_canonicalizer.h"

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

bool CanonicalizeInstructionLayout(HloInstruction* instr, bool is_entry_root);

bool IsLayoutDescending(const Shape& shape) {
  return absl::c_is_sorted(shape.layout().minor_to_major(),
                           [](int64_t a, int64_t b) { return a > b; });
}

// Given an instruction (with non-tuple output shape), this function updates the
// output shape such that the layout will be descending. It returns the
// major-to-minor layout ordering which will be used when instr is used as an
// operand.
std::vector<int64_t> CanonicalizeOutputShape(HloInstruction* instr) {
  CHECK(!instr->shape().IsTuple());
  if (IsLayoutDescending(instr->shape())) {
    return {};
  }
  // Create the major-to-minor ordering to construct the new logical dimensions
  std::vector<int64_t> major_to_minor;
  absl::c_reverse_copy(instr->shape().layout().minor_to_major(),
                       std::back_inserter(major_to_minor));

  // Update the shape according to the new descending layout.
  *instr->mutable_shape() =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
          instr->shape());
  return major_to_minor;
}

bool HandleBroadcast(HloInstruction* broadcast, std::vector<int64_t> output_map,
                     std::vector<std::vector<int64_t>> operands_map,
                     bool is_entry_root) {
  VLOG(3) << "HandleBroadcast: " << broadcast->name();

  bool changed = false;
  // Compose dimension map with the inverse of the output map.
  if (!output_map.empty()) {
    std::vector<int64_t> inverse_output_map = InversePermutation(output_map);
    VLOG(3) << "inverse_output_map = "
            << absl::StrJoin(inverse_output_map, ",");
    std::vector<int64_t> new_broadcast_dimensions;
    new_broadcast_dimensions.reserve(broadcast->dimensions().size());
    for (int64_t dim : broadcast->dimensions()) {
      new_broadcast_dimensions.push_back(inverse_output_map[dim]);
    }
    VLOG(3) << "dimensions after applying output_map = "
            << absl::StrJoin(new_broadcast_dimensions, ",");
    *broadcast->mutable_dimensions() = new_broadcast_dimensions;
    changed = true;
  }

  // Compose dimension map with the operand map.
  if (!operands_map[0].empty()) {
    std::vector<int64_t> new_broadcast_dimensions =
        ComposePermutations(broadcast->dimensions(), operands_map[0]);
    VLOG(3) << "dimensions after applying operand_map = "
            << absl::StrJoin(new_broadcast_dimensions, ",");
    *broadcast->mutable_dimensions() = new_broadcast_dimensions;
    changed = true;
  }
  if (changed) {
    VLOG(3) << "Broadcast after: " << broadcast->ToString();
  }
  return changed;
}

bool HandleConvolution(HloInstruction* conv, std::vector<int64_t> output_map,
                       std::vector<std::vector<int64_t>> operands_map,
                       bool is_entry_root) {
  VLOG(3) << "HandleConvolution: " << conv->ToString();

  bool changed = false;

  const ConvolutionDimensionNumbers& conv_dim_numbers =
      conv->convolution_dimension_numbers();
  ConvolutionDimensionNumbers new_conv_dim_numbers = conv_dim_numbers;

  if (!output_map.empty()) {
    std::vector<int64_t> inverse_output_map = InversePermutation(output_map);
    VLOG(3) << "inverse_output_map = "
            << absl::StrJoin(inverse_output_map, ",");

    new_conv_dim_numbers.set_output_batch_dimension(
        inverse_output_map[conv_dim_numbers.output_batch_dimension()]);
    VLOG(3) << "new_conv_dim_numbers.output_batch = "
            << new_conv_dim_numbers.output_batch_dimension();

    new_conv_dim_numbers.set_output_feature_dimension(
        inverse_output_map[conv_dim_numbers.output_feature_dimension()]);
    VLOG(3) << "new_conv_dim_numbers.output_feature = "
            << new_conv_dim_numbers.output_feature_dimension();

    for (int64_t i = 0; i < conv_dim_numbers.output_spatial_dimensions_size();
         ++i) {
      new_conv_dim_numbers.set_output_spatial_dimensions(
          i, inverse_output_map[conv_dim_numbers.output_spatial_dimensions(i)]);
      VLOG(3) << "new_conv_dim_numbers.output_spatial_dimensions = "
              << new_conv_dim_numbers.output_spatial_dimensions(i);
    }
    changed = true;
  }

  // Compose lhs dimension numbers with the lhs operand maps.
  if (!operands_map[0].empty()) {
    std::vector<int64_t> inverse_lhs_operand_map =
        InversePermutation(operands_map[0]);
    VLOG(3) << "inverse_lhs_operand_map = "
            << absl::StrJoin(inverse_lhs_operand_map, ",");

    new_conv_dim_numbers.set_input_batch_dimension(
        inverse_lhs_operand_map[conv_dim_numbers.input_batch_dimension()]);
    VLOG(3) << "new_conv_dim_numbers.input_batch = "
            << new_conv_dim_numbers.input_batch_dimension();

    new_conv_dim_numbers.set_input_feature_dimension(
        inverse_lhs_operand_map[conv_dim_numbers.input_feature_dimension()]);
    VLOG(3) << "new_conv_dim_numbers.input_feature = "
            << new_conv_dim_numbers.input_feature_dimension();

    for (int64_t i = 0; i < conv_dim_numbers.input_spatial_dimensions_size();
         ++i) {
      new_conv_dim_numbers.set_input_spatial_dimensions(
          i, inverse_lhs_operand_map[conv_dim_numbers.input_spatial_dimensions(
                 i)]);
      VLOG(3) << "new_conv_dim_numbers.input_spatial_dimensions = "
              << new_conv_dim_numbers.input_spatial_dimensions(i);
    }
    changed = true;
  }

  // Compose rhs dimension numbers with the rhs operand maps.
  if (!operands_map[1].empty()) {
    std::vector<int64_t> inverse_rhs_operand_map =
        InversePermutation(operands_map[1]);
    VLOG(3) << "inverse_rhs_operand_map = "
            << absl::StrJoin(inverse_rhs_operand_map, ",");

    new_conv_dim_numbers.set_kernel_input_feature_dimension(
        inverse_rhs_operand_map[conv_dim_numbers
                                    .kernel_input_feature_dimension()]);
    VLOG(3) << "new_conv_dim_numbers.kernel_input_feature = "
            << new_conv_dim_numbers.kernel_input_feature_dimension();

    new_conv_dim_numbers.set_kernel_output_feature_dimension(
        inverse_rhs_operand_map[conv_dim_numbers
                                    .kernel_output_feature_dimension()]);
    VLOG(3) << "new_conv_dim_numbers.kernel_output_feature = "
            << new_conv_dim_numbers.kernel_output_feature_dimension();

    for (int64_t i = 0; i < conv_dim_numbers.kernel_spatial_dimensions_size();
         ++i) {
      new_conv_dim_numbers.set_kernel_spatial_dimensions(
          i, inverse_rhs_operand_map[conv_dim_numbers.kernel_spatial_dimensions(
                 i)]);
      VLOG(3) << "new_conv_dim_numbers.kernel_spatial_dimensions = "
              << new_conv_dim_numbers.kernel_spatial_dimensions(i);
    }
    changed = true;
  }

  conv->set_convolution_dimension_numbers(new_conv_dim_numbers);

  return changed;
}

// Given an instruction, this function canonicalizes each operand if it is not
// of type parameter and the layout is not already canonical. It returns the
// major-to-minor layout ordering for each operand.
std::vector<std::vector<int64_t>> CanonicalizeOperands(HloInstruction* instr) {
  std::vector<std::vector<int64_t>> operands_map;
  for (HloInstruction* operand : instr->operands()) {
    // Save a copy of operand's layout before possible canonicalization.
    std::vector<int64_t> operand_old_layout;
    absl::c_copy(operand->shape().layout().minor_to_major(),
                 std::back_inserter(operand_old_layout));
    // Try canonicalizing the operand layout. If succeeded, populate the operand
    // map.
    if (CanonicalizeInstructionLayout(operand, false)) {
      std::vector<int64_t> major_to_minor;
      absl::c_reverse_copy(operand_old_layout,
                           std::back_inserter(major_to_minor));
      operands_map.push_back(major_to_minor);
      VLOG(3) << operand->name()
              << " operand_map = " << absl::StrJoin(major_to_minor, ",");
    } else {
      operands_map.push_back({});
    }
  }
  return operands_map;
}

bool CanCanonicalize(HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kBroadcast ||
      instr->opcode() == HloOpcode::kBitcast ||
      instr->opcode() == HloOpcode::kConvolution) {
    return true;
  }
  return false;
}

// Check if the instruction has any users that cannot be canonicalized. The goal
// is to remove this check after all the operations are implemented.
bool CheckUsers(HloInstruction* instr) {
  for (HloInstruction* user : instr->users()) {
    if (!CanCanonicalize(user)) {
      return false;
    }
  }
  return true;
}

bool CanonicalizeInstructionLayout(HloInstruction* instr, bool is_entry_root) {
  // We ignore parameters
  if (instr->opcode() == HloOpcode::kParameter) {
    return false;
  }

  if (!CanCanonicalize(instr) || !CheckUsers(instr)) {
    return false;
  }

  VLOG(3) << "CanonicalizeInstructionLayout: " << instr->name();

  std::vector<std::vector<int64_t>> operands_map = CanonicalizeOperands(instr);

  // Handle output
  std::vector<int64_t> output_map =
      is_entry_root ? std::vector<int64_t>({}) : CanonicalizeOutputShape(instr);
  bool output_canonicalized = !output_map.empty();

  VLOG(3) << "output_map = " << absl::StrJoin(output_map, ",");
  // For now, we only handle broadcast, bitcast and convolution. I will add
  // other ops.
  // gradually.
  bool changed = false;
  switch (instr->opcode()) {
    case HloOpcode::kBroadcast:
      changed = HandleBroadcast(instr, output_map, operands_map, is_entry_root);
      break;
    case HloOpcode::kBitcast:
      // Bitcast does not require any special treatment. We only check if the
      // single operand has been canonicalized or not.
      changed = !operands_map[0].empty();
      break;
    case HloOpcode::kConvolution:
      changed =
          HandleConvolution(instr, output_map, operands_map, is_entry_root);
      break;
    default:
      break;
  }
  return output_canonicalized || changed;
}
};  // namespace

absl::StatusOr<bool> LayoutCanonicalizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(3) << "Before LayoutCanonicalizer::Run: \n" << module->ToString();
  bool changed = false;
  for (auto* comp : module->MakeComputationPostOrder(execution_threads)) {
    // We only canonicalize the entry computation for now.
    if (comp->IsEntryComputation()) {
      changed |= CanonicalizeInstructionLayout(comp->root_instruction(), true);
    }
  }
  if (changed) {
    std::cout << "CANONICALIZED_LAYOUT" << std::endl;
    VLOG(3) << "After LayoutCanonicalizer::Run: \n" << module->ToString();
  }
  return changed;
}

}  // namespace xla
