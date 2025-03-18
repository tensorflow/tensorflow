/*
 * Copyright 2025 The OpenXLA Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_GUMGRAPH_RENDERER_UTIL_H_
#define XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_GUMGRAPH_RENDERER_UTIL_H_

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {
namespace hlo_diff {

// Print the instruction to string.
std::string InstructionToString(const HloInstruction* instr, bool name_only);

// Enum representing the type of changes for a pair of changed instructions.
enum class ChangedInstructionDiffType : uint8_t {
  kOtherChange,
  kShapeChange,
  kLayoutChange,
  kMemorySpaceChange,
  kChangedOperandsNumber,
  kChangedOperandsShape,
  kOpCodeChanged,
  kConstantLiteralChanged,
};

// Returns details on what exactly has changed for a pair of changed
// instruction.
std::vector<ChangedInstructionDiffType> GetChangedInstructionDiffTypes(
    const HloInstruction& left, const HloInstruction& right);

// Converts the changed instruction diff type enum value to a string.
std::string GetChangedInstructionDiffTypeString(
    ChangedInstructionDiffType diff_type);

// Opcodes to be ignored when printing summaries.
inline constexpr auto kIgnoredOpcodes = std::array<HloOpcode, 6>(
    {HloOpcode::kReshape, HloOpcode::kBitcast, HloOpcode::kPad,
     HloOpcode::kCopyDone, HloOpcode::kCopyStart, HloOpcode::kGetTupleElement});

// Groups the instructions by opcode.
absl::flat_hash_map<HloOpcode, std::vector<const HloInstruction*>>
GroupInstructionsByOpcode(absl::Span<const HloInstruction* const> instructions);

// Groups the instruction pairs by opcode.
absl::flat_hash_map<
    HloOpcode,
    std::vector<std::pair<const HloInstruction*, const HloInstruction*>>>
GroupInstructionPairsByOpcode(
    const absl::flat_hash_map<const HloInstruction*, const HloInstruction*>&
        instructions);

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_GUMGRAPH_RENDERER_UTIL_H_
