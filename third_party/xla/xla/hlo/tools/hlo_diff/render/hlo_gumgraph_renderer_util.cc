// Copyright 2025 The OpenXLA Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_renderer_util.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"

namespace xla {
namespace hlo_diff {

std::string InstructionToString(const HloInstruction* instr, bool name_only) {
  if (name_only) {
    return std::string(instr->name());
  }
  return instr->ToString(HloPrintOptions::ShortParsable());
}

std::vector<ChangedInstructionDiffType> GetChangedInstructionDiffTypes(
    const HloInstruction& left, const HloInstruction& right) {
  // Compare shapes, layouts and memory spaces
  std::vector<ChangedInstructionDiffType> diff_types;
  if (left.shape() != right.shape()) {
    diff_types.push_back(ChangedInstructionDiffType::kShapeChange);

    if (left.shape().IsArray() && right.shape().IsArray() &&
        left.shape().has_layout() && right.shape().has_layout() &&
        (left.shape().layout() != right.shape().layout())) {
      diff_types.push_back(ChangedInstructionDiffType::kLayoutChange);
      if (left.shape().layout().memory_space() !=
          right.shape().layout().memory_space()) {
        diff_types.push_back(ChangedInstructionDiffType::kMemorySpaceChange);
      }
    }
  }

  // Compare operand numbers and shapes
  if (left.operand_count() != right.operand_count()) {
    diff_types.push_back(ChangedInstructionDiffType::kChangedOperandsNumber);
  } else {  // If operand numbers are the same, compare shapes
    for (int64_t i = 0; i < left.operand_count(); ++i) {
      if (left.operand(i)->shape() != right.operand(i)->shape()) {
        diff_types.push_back(ChangedInstructionDiffType::kChangedOperandsShape);
        break;
      }
    }
  }

  // Compare opcodes
  if (left.opcode() != right.opcode()) {
    diff_types.push_back(ChangedInstructionDiffType::kOpCodeChanged);
  }

  // Compare constants
  if (left.IsConstant() && right.IsConstant()) {
    if (left.literal() != right.literal()) {
      diff_types.push_back(ChangedInstructionDiffType::kConstantLiteralChanged);
    }
  }

  // If no diff type is found, return kOtherChange.
  if (diff_types.empty()) {
    diff_types.push_back(ChangedInstructionDiffType::kOtherChange);
  }

  return diff_types;
};

std::string GetChangedInstructionDiffTypeString(
    ChangedInstructionDiffType diff_type) {
  switch (diff_type) {
    case ChangedInstructionDiffType::kOtherChange:
      return "kOtherChange";
    case ChangedInstructionDiffType::kShapeChange:
      return "kShapeChange";
    case ChangedInstructionDiffType::kLayoutChange:
      return "kLayoutChange";
    case ChangedInstructionDiffType::kMemorySpaceChange:
      return "kMemorySpaceChange";
    case ChangedInstructionDiffType::kChangedOperandsNumber:
      return "kChangedOperandsNumber";
    case ChangedInstructionDiffType::kChangedOperandsShape:
      return "kChangedOperandsShape";
    case ChangedInstructionDiffType::kOpCodeChanged:
      return "kOpCodeChanged";
    case ChangedInstructionDiffType::kConstantLiteralChanged:
      return "kConstantLiteralChanged";
    default:
      return "";
  }
}

absl::flat_hash_map<HloOpcode, std::vector<const HloInstruction*>>
GroupInstructionsByOpcode(
    const absl::flat_hash_set<const HloInstruction*>& instructions) {
  absl::flat_hash_map<HloOpcode, std::vector<const HloInstruction*>>
      instructions_by_opcode;
  for (const HloInstruction* inst : instructions) {
    instructions_by_opcode[inst->opcode()].push_back(inst);
  }
  return instructions_by_opcode;
}

absl::flat_hash_map<
    HloOpcode,
    std::vector<std::pair<const HloInstruction*, const HloInstruction*>>>
GroupInstructionPairsByOpcode(
    const absl::flat_hash_map<const HloInstruction*, const HloInstruction*>&
        instructions) {
  absl::flat_hash_map<
      HloOpcode,
      std::vector<std::pair<const HloInstruction*, const HloInstruction*>>>
      instructions_by_opcode;
  for (const auto& pair : instructions) {
    instructions_by_opcode[pair.first->opcode()].push_back(pair);
  }
  return instructions_by_opcode;
}

}  // namespace hlo_diff
}  // namespace xla
