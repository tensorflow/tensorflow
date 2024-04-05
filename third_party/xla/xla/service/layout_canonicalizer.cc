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
#include "xla/service/layout_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

bool CanonicalizeInstructionLayout(HloInstruction* instr, bool is_entry_root);

bool IsLayoutDescending(const Shape& shape) {
  return absl::c_is_sorted(shape.layout().minor_to_major(),
                           [](int64_t a, int64_t b) { return a > b; });
}

// Given an instruction (with non-tuple output shape), this function updates the
// output shape such that the layout is descending. It returns the
// major-to-minor layout ordering which will be used when instr is used as an
// operand.
std::vector<int64_t> HandleOutput(HloInstruction* instr) {
  CHECK(!instr->shape().IsTuple());
  if (IsLayoutDescending(instr->shape())) {
    return {};
  }
  // Create the major-to-minor ordering to construct the new logical dimensions
  std::vector<int64_t> major_to_minor;
  absl::c_reverse_copy(instr->shape().layout().minor_to_major(),
                       std::back_inserter(major_to_minor));

  // Compose shape's dimensions with the major-to-minor layout
  std::vector<int64_t> input_new_logical_dims =
      ComposePermutations(instr->shape().dimensions(), major_to_minor);

  // Update the shape
  *instr->mutable_shape() = ShapeUtil::MakeShapeWithDescendingLayout(
      instr->shape().element_type(), input_new_logical_dims);
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
    std::cout << "FARZIN_:dimensions after applying output_map = "
              << absl::StrJoin(new_broadcast_dimensions, ",") << std::endl;
    *broadcast->mutable_dimensions() = new_broadcast_dimensions;
    changed = true;
  }

  // Compose dimension map with the operand map.
  if (!operands_map[0].empty()) {
    std::vector<int64_t> new_broadcast_dimensions =
        ComposePermutations(broadcast->dimensions(), operands_map[0]);
    VLOG(3) << "dimensions after applying operand_map = "
            << absl::StrJoin(new_broadcast_dimensions, ",");
    std::cout << "FARZIN_:dimensions after applying operand_map = "
              << absl::StrJoin(new_broadcast_dimensions, ",") << std::endl;
    *broadcast->mutable_dimensions() = new_broadcast_dimensions;
    changed = true;
  }
  if (changed) {
    VLOG(3) << "Broadcast after: " << broadcast->ToString();
  }
  return changed;
}

bool HandleTranspose(HloInstruction* transpose, std::vector<int64_t> output_map,
                     std::vector<std::vector<int64_t>> operands_map,
                     bool is_entry_root) {
  VLOG(3) << "HandleTranspose: " << transpose->name();

  bool changed = false;

  // Compose dimension map with the output map.
  if (!output_map.empty()) {
    std::vector<int64_t> new_transpose_dimensions =
        ComposePermutations(output_map, transpose->dimensions());
    VLOG(3) << "dimensions after applying output_map = "
            << absl::StrJoin(new_transpose_dimensions, ",");
    *transpose->mutable_dimensions() = new_transpose_dimensions;
    changed = true;
  }

  // Compose dimension map with the inverse of the operand map.
  if (!operands_map[0].empty()) {
    std::vector<int64_t> inverse_operand_map =
        InversePermutation(operands_map[0]);
    VLOG(3) << "inverse_operand_map = "
            << absl::StrJoin(inverse_operand_map, ",");
    std::vector<int64_t> new_transpose_dimensions =
        ComposePermutations(inverse_operand_map, transpose->dimensions());
    VLOG(3) << "dimensions after applying operand_map = "
            << absl::StrJoin(new_transpose_dimensions, ",");
    *transpose->mutable_dimensions() = new_transpose_dimensions;
    changed = true;
  }
  VLOG(3) << "Transpose after: " << transpose->ToString();
  return changed;
}

// Given an instruction, this function canonicalizes each operand if it is not
// of type parameter and the layout is not already canonical. It returns the
// major-to-minor layout ordering for each operand.
std::vector<std::vector<int64_t>> CanonicalizeOperands(HloInstruction* instr) {
  std::vector<std::vector<int64_t>> operands_map;
  for (HloInstruction* operand : instr->operands()) {
    if (operand->opcode() == HloOpcode::kParameter ||
        IsLayoutDescending(operand->shape())) {
      operands_map.push_back({});
    } else {
      std::vector<int64_t> major_to_minor;
      absl::c_reverse_copy(operand->shape().layout().minor_to_major(),
                           std::back_inserter(major_to_minor));
      bool changed = CanonicalizeInstructionLayout(operand, false);
      operands_map.push_back(changed ? major_to_minor
                                     : std::vector<int64_t>({}));
      VLOG(3) << "operand_map = " << absl::StrJoin(major_to_minor, ",");
    }
  }
  return operands_map;
}

bool CanCanonicalize(HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kBroadcast ||
      instr->opcode() == HloOpcode::kTranspose) {
    return true;
  }
  return false;
}

bool CanonicalizeInstructionLayout(HloInstruction* instr, bool is_entry_root) {
  // We ignore parameters
  if (instr->opcode() == HloOpcode::kParameter) {
    return false;
  }

  if (!CanCanonicalize(instr)) {
    return false;
  }

  VLOG(3) << "CanonicalizeInstructionLayout: " << instr->name();

  std::vector<std::vector<int64_t>> operands_map = CanonicalizeOperands(instr);

  // Handle output
  std::vector<int64_t> output_map =
      is_entry_root ? std::vector<int64_t>({}) : HandleOutput(instr);
  VLOG(3) << "output_map = " << absl::StrJoin(output_map, ",");

  // For now, we only handle broadcast and transpose. I will add other ops
  // gradually.
  switch (instr->opcode()) {
    case HloOpcode::kBroadcast:
      return HandleBroadcast(instr, output_map, operands_map, is_entry_root);
    case HloOpcode::kTranspose:
      return HandleTranspose(instr, output_map, operands_map, is_entry_root);
    default:
      break;
  }
  return false;
}
};  // namespace

absl::StatusOr<bool> LayoutCanonicalizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(3) << "Before LayoutCanonicalizer::Run: \n" << module->ToString();
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    // We only canonicalize the entry computation for now.
    if (comp->IsEntryComputation()) {
      changed |= CanonicalizeInstructionLayout(comp->root_instruction(), true);
    }
  }
  if (changed) {
    std::cout << "CANONICALIZED_LAYOUT" << std::endl;
  }
  return changed;
}

}  // namespace xla
