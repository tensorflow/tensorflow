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
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

bool IsLayoutDescending(const Shape& shape) {
  return absl::c_is_sorted(shape.layout().minor_to_major(),
                           [](int64_t a, int64_t b) { return a > b; });
}

bool CanCanonicalize(HloInstruction* instr) {
  // TODO(farzinh): Based on some tap failures, it seems like bitcast that is
  // fed by a custom-call need special handling and cannot be easily
  // canonicalized if the operand is not canonical. see
  // //third_party/py/jax/tests:linalg_test_tpu_puffylite
  // --test_filter=NumpyLinalgTest.testMatrixPower28 for example.
  if (instr->opcode() == HloOpcode::kBitcast &&
      instr->mutable_operand(0)->opcode() == HloOpcode::kCustomCall) {
    return false;
  }
  if (instr->opcode() == HloOpcode::kBroadcast ||
      instr->opcode() == HloOpcode::kBitcast ||
      instr->opcode() == HloOpcode::kConvolution ||
      instr->opcode() == HloOpcode::kGather) {
    return true;
  }
  return false;
}

// Check if the instruction has any users that cannot be canonicalized. The goal
// is to remove this check after all the operations are implemented.
bool CheckUsers(HloInstruction* instr) {
  if (!CanCanonicalize(instr)) {
    return false;
  }
  for (HloInstruction* user : instr->users()) {
    if (!CheckUsers(user)) {
      return false;
    }
  }
  return true;
}
bool HandleBroadcast(
    HloInstruction* broadcast,
    absl::flat_hash_map<HloInstruction*, std::vector<int64_t>>& old_layouts,
    bool is_entry_root) {
  bool changed = false;
  // Compose dimension map with the inverse of the output map.
  if (old_layouts.contains(broadcast)) {
    std::vector<int64_t> inverse_output_map =
        InversePermutation(old_layouts[broadcast]);
    VLOG(4) << "inverse_output_map = "
            << absl::StrJoin(inverse_output_map, ",");
    std::vector<int64_t> new_broadcast_dimensions;
    new_broadcast_dimensions.reserve(broadcast->dimensions().size());
    for (int64_t dim : broadcast->dimensions()) {
      new_broadcast_dimensions.push_back(inverse_output_map[dim]);
    }
    VLOG(4) << "dimensions after applying output_map = "
            << absl::StrJoin(new_broadcast_dimensions, ",");
    *broadcast->mutable_dimensions() = new_broadcast_dimensions;
    changed = true;
  }

  // Compose dimension map with the operand map.
  if (old_layouts.contains(broadcast->mutable_operand(0))) {
    std::vector<int64_t> new_broadcast_dimensions = ComposePermutations(
        broadcast->dimensions(), old_layouts[broadcast->mutable_operand(0)]);
    VLOG(4) << "dimensions after applying operand_map = "
            << absl::StrJoin(new_broadcast_dimensions, ",");
    *broadcast->mutable_dimensions() = new_broadcast_dimensions;
    changed = true;
  }
  return changed;
}

bool HandleConvolution(
    HloInstruction* conv,
    absl::flat_hash_map<HloInstruction*, std::vector<int64_t>>& old_layouts,
    bool is_entry_root) {
  bool changed = false;
  const ConvolutionDimensionNumbers& conv_dim_numbers =
      conv->convolution_dimension_numbers();
  ConvolutionDimensionNumbers new_conv_dim_numbers = conv_dim_numbers;

  if (old_layouts.contains(conv)) {
    std::vector<int64_t> inverse_output_map =
        InversePermutation(old_layouts[conv]);
    VLOG(4) << "inverse_output_map = "
            << absl::StrJoin(inverse_output_map, ",");

    new_conv_dim_numbers.set_output_batch_dimension(
        inverse_output_map[conv_dim_numbers.output_batch_dimension()]);
    VLOG(4) << "new_conv_dim_numbers.output_batch = "
            << new_conv_dim_numbers.output_batch_dimension();

    new_conv_dim_numbers.set_output_feature_dimension(
        inverse_output_map[conv_dim_numbers.output_feature_dimension()]);
    VLOG(4) << "new_conv_dim_numbers.output_feature = "
            << new_conv_dim_numbers.output_feature_dimension();

    for (int64_t i = 0; i < conv_dim_numbers.output_spatial_dimensions_size();
         ++i) {
      new_conv_dim_numbers.set_output_spatial_dimensions(
          i, inverse_output_map[conv_dim_numbers.output_spatial_dimensions(i)]);
      VLOG(4) << "new_conv_dim_numbers.output_spatial_dimensions = "
              << new_conv_dim_numbers.output_spatial_dimensions(i);
    }
    changed = true;
  }

  // Compose lhs dimension numbers with the lhs operand maps.
  if (old_layouts.contains(conv->mutable_operand(0))) {
    std::vector<int64_t> inverse_lhs_operand_map =
        InversePermutation(old_layouts[conv->mutable_operand(0)]);
    VLOG(4) << "inverse_lhs_operand_map = "
            << absl::StrJoin(inverse_lhs_operand_map, ",");

    new_conv_dim_numbers.set_input_batch_dimension(
        inverse_lhs_operand_map[conv_dim_numbers.input_batch_dimension()]);
    VLOG(4) << "new_conv_dim_numbers.input_batch = "
            << new_conv_dim_numbers.input_batch_dimension();

    new_conv_dim_numbers.set_input_feature_dimension(
        inverse_lhs_operand_map[conv_dim_numbers.input_feature_dimension()]);
    VLOG(4) << "new_conv_dim_numbers.input_feature = "
            << new_conv_dim_numbers.input_feature_dimension();

    for (int64_t i = 0; i < conv_dim_numbers.input_spatial_dimensions_size();
         ++i) {
      new_conv_dim_numbers.set_input_spatial_dimensions(
          i, inverse_lhs_operand_map[conv_dim_numbers.input_spatial_dimensions(
                 i)]);
      VLOG(4) << "new_conv_dim_numbers.input_spatial_dimensions = "
              << new_conv_dim_numbers.input_spatial_dimensions(i);
    }
    changed = true;
  }

  // Compose rhs dimension numbers with the rhs operand maps.
  if (old_layouts.contains(conv->mutable_operand(1))) {
    std::vector<int64_t> inverse_rhs_operand_map =
        InversePermutation(old_layouts[conv->mutable_operand(1)]);
    VLOG(4) << "inverse_rhs_operand_map = "
            << absl::StrJoin(inverse_rhs_operand_map, ",");

    new_conv_dim_numbers.set_kernel_input_feature_dimension(
        inverse_rhs_operand_map[conv_dim_numbers
                                    .kernel_input_feature_dimension()]);
    VLOG(4) << "new_conv_dim_numbers.kernel_input_feature = "
            << new_conv_dim_numbers.kernel_input_feature_dimension();

    new_conv_dim_numbers.set_kernel_output_feature_dimension(
        inverse_rhs_operand_map[conv_dim_numbers
                                    .kernel_output_feature_dimension()]);
    VLOG(4) << "new_conv_dim_numbers.kernel_output_feature = "
            << new_conv_dim_numbers.kernel_output_feature_dimension();

    for (int64_t i = 0; i < conv_dim_numbers.kernel_spatial_dimensions_size();
         ++i) {
      new_conv_dim_numbers.set_kernel_spatial_dimensions(
          i, inverse_rhs_operand_map[conv_dim_numbers.kernel_spatial_dimensions(
                 i)]);
      VLOG(4) << "new_conv_dim_numbers.kernel_spatial_dimensions = "
              << new_conv_dim_numbers.kernel_spatial_dimensions(i);
    }
    changed = true;
  }

  conv->set_convolution_dimension_numbers(new_conv_dim_numbers);

  return changed;
}

bool HandleGather(
    HloInstruction* gather,
    absl::flat_hash_map<HloInstruction*, std::vector<int64_t>>& old_layouts,
    bool is_entry_root) {
  bool changed = false;
  const GatherDimensionNumbers& gather_dim_numbers =
      gather->gather_dimension_numbers();
  GatherDimensionNumbers new_gather_dim_numbers = gather_dim_numbers;

  if (old_layouts.contains(gather)) {
    std::vector<int64_t> inverse_output_map =
        InversePermutation(old_layouts[gather]);
    VLOG(4) << "inverse_output_map = "
            << absl::StrJoin(inverse_output_map, ",");

    for (int64_t i = 0; i < gather_dim_numbers.offset_dims_size(); ++i) {
      new_gather_dim_numbers.set_offset_dims(
          i, inverse_output_map[gather_dim_numbers.offset_dims(i)]);
    }
    std::cout << "CANONICALIZED_GATHER_LAYOUT" << std::endl;
    changed = true;
  }
  HloGatherInstruction* gather_instruction =
      DynCast<HloGatherInstruction>(gather);
  gather_instruction->set_gather_dimension_numbers(new_gather_dim_numbers);
  return changed;
}

bool HandleInstruction(
    HloInstruction* instr,
    absl::flat_hash_map<HloInstruction*, std::vector<int64_t>>& old_layouts,
    bool is_entry_root) {
  VLOG(3) << "Handle Instruction: " << instr->name();
  // For now, we only handle broadcast and bitcast. I will add other ops
  // gradually.
  bool changed_instr = false;
  switch (instr->opcode()) {
    case HloOpcode::kBroadcast:
      changed_instr = HandleBroadcast(instr, old_layouts, is_entry_root);
      break;
    case HloOpcode::kConvolution:
      changed_instr = HandleConvolution(instr, old_layouts, is_entry_root);
      break;
    case HloOpcode::kGather:
      changed_instr = HandleGather(instr, old_layouts, is_entry_root);
      break;
    // Bitcast does not need special handling. It is only here for the sake of
    // completeness to indicate that it will be canonicalized.
    case HloOpcode::kBitcast:
      break;
    default:
      break;
  }
  return changed_instr;
}

};  // namespace

bool LayoutCanonicalizer::CanonicalizeOutputShape(HloInstruction* instr) {
  CHECK(!instr->shape().IsTuple());
  if (IsLayoutDescending(instr->shape())) {
    VLOG(3) << instr->name() << " did not change output.";
    return false;
  }
  // Create the major-to-minor ordering to construct the new logical dimensions
  std::vector<int64_t> major_to_minor;
  absl::c_reverse_copy(instr->shape().layout().minor_to_major(),
                       std::back_inserter(major_to_minor));
  original_layout_.insert({instr, major_to_minor});

  VLOG(4) << instr->name()
          << " output_map = " << absl::StrJoin(major_to_minor, ",");

  // Update the shape according to the new descending layout.
  *instr->mutable_shape() =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
          instr->shape());
  VLOG(3) << instr->name() << " changed output.";
  return true;
}

bool LayoutCanonicalizer::CanonicalizeOperands(HloInstruction* instr) {
  bool canonicalized = false;
  for (HloInstruction* operand : instr->operands()) {
    if (CanonicalizeInstructionLayout(operand, false)) {
      canonicalized |= true;
    }
  }
  if (canonicalized) {
    VLOG(3) << instr->name() << " changed operands.";
  } else {
    VLOG(3) << instr->name() << " did not change operands.";
  }
  return canonicalized;
}

bool LayoutCanonicalizer::CanonicalizeInstructionLayout(HloInstruction* instr,
                                                        bool is_entry_root) {
  // We ignore parameters
  if (instr->opcode() == HloOpcode::kParameter) {
    VLOG(3) << instr->name() << " not applied helper. is a parameter.";
    return false;
  }

  if (!CanCanonicalize(instr) || !CheckUsers(instr)) {
    VLOG(3) << instr->name() << " not applied helper. not doable.";
    return false;
  }

  if (canonicalized_instrs_.contains(instr)) {
    VLOG(3) << instr->name() << " alread canonicalized.";
    return false;
  }

  VLOG(3) << "inside CanonicalizeInstructionLayout: " << instr->ToString();

  bool changed_operands = CanonicalizeOperands(instr);

  // Canonicalize the output shape only if the instruction is not the root of
  // the entry computation.
  bool changed_output = false;
  if (!is_entry_root) {
    changed_output = CanonicalizeOutputShape(instr);
  }

  // Must rewrite the instruction according to the new state and add it to the
  // set of canonicalized instructions only if it is not changed any more.
  HandleInstruction(instr, original_layout_, is_entry_root);
  canonicalized_instrs_.insert(instr);

  // Canonicalize the users.
  bool changed = changed_operands || changed_output;
  // TODO: apparenlty, we don't need this
  // for (HloInstruction* user : instr->users()) {
  //   changed |= CanonicalizeInstructionLayout(user, user->IsRoot());
  // }

  if (changed) {
    VLOG(3) << instr->name() << " canonicalized: " << instr->ToString();
  } else {
    VLOG(3) << instr->name() << " did not canonicalize.";
  }
  return changed;
}

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
