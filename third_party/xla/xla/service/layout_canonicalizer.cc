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
#include "xla/hlo/ir/hlo_instruction.h"
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
  if (instr->opcode() == HloOpcode::kBroadcast ||
      instr->opcode() == HloOpcode::kBitcast) {
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
bool HandleBroadcast(HloInstruction* broadcast,
                     absl::flat_hash_map<HloInstruction*, std::vector<int64_t>>&
                         canonicalized_instrs,
                     bool is_entry_root) {
  VLOG(3) << "HandleBroadcast: " << broadcast->name();

  bool changed = false;
  // Compose dimension map with the inverse of the output map.
  if (canonicalized_instrs.contains(broadcast)) {
    std::vector<int64_t> inverse_output_map =
        InversePermutation(canonicalized_instrs[broadcast]);
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
  if (canonicalized_instrs.contains(broadcast->mutable_operand(0))) {
    std::vector<int64_t> new_broadcast_dimensions = ComposePermutations(
        broadcast->dimensions(),
        canonicalized_instrs[broadcast->mutable_operand(0)]);
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

};  // namespace

bool LayoutCanonicalizer::CanonicalizeOutputShape(HloInstruction* instr) {
  CHECK(!instr->shape().IsTuple());
  if (IsLayoutDescending(instr->shape())) {
    return false;
  }
  // Create the major-to-minor ordering to construct the new logical dimensions
  std::vector<int64_t> major_to_minor;
  absl::c_reverse_copy(instr->shape().layout().minor_to_major(),
                       std::back_inserter(major_to_minor));
  canonicalized_instrs_.insert({instr, major_to_minor});

  VLOG(3) << instr->name()
          << " output_map = " << absl::StrJoin(major_to_minor, ",");

  // Update the shape according to the new descending layout.
  *instr->mutable_shape() =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
          instr->shape());
  return true;
}

bool LayoutCanonicalizer::CanonicalizeOperands(HloInstruction* instr) {
  bool canonicalized = false;
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
      canonicalized_instrs_.insert({operand, major_to_minor});
      canonicalized |= true;
      VLOG(3) << operand->name()
              << " operand_map = " << absl::StrJoin(major_to_minor, ",");
    }
  }
  return canonicalized;
}

bool LayoutCanonicalizer::CanonicalizeInstructionLayout(HloInstruction* instr,
                                                        bool is_entry_root) {
  // We ignore parameters
  if (instr->opcode() == HloOpcode::kParameter) {
    return false;
  }

  if (!CanCanonicalize(instr) || !CheckUsers(instr)) {
    return false;
  }

  VLOG(3) << "CanonicalizeInstructionLayout: " << instr->name();

  bool changed_operands = CanonicalizeOperands(instr);

  // Canonicalize the output shape only if the instruction is not the root of
  // the entry computation.
  bool changed_output = false;
  if (!is_entry_root) {
    changed_output = CanonicalizeOutputShape(instr);
  }

  // For now, we only handle broadcast and bitcast. I will add other ops
  // gradually.
  bool changed_instr = false;
  switch (instr->opcode()) {
    case HloOpcode::kBroadcast:
      changed_instr =
          HandleBroadcast(instr, canonicalized_instrs_, is_entry_root);
      break;
    case HloOpcode::kBitcast:
      break;
    default:
      break;
  }
  return changed_output || changed_instr || changed_operands;
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
  }
  return changed;
}

}  // namespace xla
