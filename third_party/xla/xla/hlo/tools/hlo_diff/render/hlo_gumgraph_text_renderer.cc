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

#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_text_renderer.h"

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_renderer_util.h"

namespace xla {
namespace hlo_diff {
namespace {

// Prints unmatched instructions grouped by opcode and print in a descending
// order of the number of instructions for each opcode. If top_n_opcodes or
// max_instructions_per_opcode is a negative number, all the instructions will
// be printed.
void PrintUnmatchedInstructions(
    const absl::string_view header,
    const absl::flat_hash_set<const HloInstruction*>& instructions,
    std::ostringstream& out, const RenderTextOptions& options) {
  out << header;
  if (options.top_n_opcodes >= 0) {
    out << " (top " << options.top_n_opcodes << " frequent opcode)";
  }
  if (!options.opcodes_to_ignore.empty()) {
    out << " (ignoring "
        << absl::StrJoin(options.opcodes_to_ignore, ", ",
                         [](std::string* out, const HloOpcode& opcode) {
                           absl::StrAppend(out, HloOpcodeString(opcode));
                         })
        << ")";
  }
  out << ":\n";

  absl::flat_hash_map<HloOpcode, std::vector<const HloInstruction*>>
      instructions_by_opcode = GroupInstructionsByOpcode(instructions);
  std::vector<std::pair<HloOpcode, int64_t>> opcode_counts;
  // The counts are sorted below, so the iteration order doesn't matter.
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (const auto& [opcode, insts] : instructions_by_opcode) {
    opcode_counts.push_back({opcode, insts.size()});
  }
  // Most frequent first; ties are broken by opcode for a deterministic order.
  std::sort(opcode_counts.begin(), opcode_counts.end(),
            [](const auto& a, const auto& b) {
              return a.second != b.second ? a.second > b.second
                                          : a.first < b.first;
            });
  // Print the top N most frequent opcodes
  int i = 0;
  for (auto cit = opcode_counts.begin();
       (options.top_n_opcodes < 0 || i < options.top_n_opcodes) &&
       cit != opcode_counts.end();
       ++cit) {
    if (options.opcodes_to_ignore.contains(cit->first)) {
      continue;
    }
    absl::string_view op_name = HloOpcodeString(cit->first);
    out << "  " << op_name << " (" << cit->second << "):\n";
    std::vector<const HloInstruction*> insts =
        instructions_by_opcode[cit->first];
    // Print the M instructions for each opcode
    int j = 0;
    for (auto iit = insts.begin(); (options.max_instructions_per_opcode < 0 ||
                                    j < options.max_instructions_per_opcode) &&
                                   iit != insts.end();
         ++j, ++iit) {
      out << "    " << InstructionToString(*iit, options.name_only) << "\n";
    }
    if (j < insts.size()) {
      out << "    ... and " << insts.size() - j << " more " << op_name
          << " instructions\n";
    }
    out << "\n";
    ++i;
  }
  if (i < opcode_counts.size()) {
    out << "  ... and " << opcode_counts.size() - i << " more opcodes\n";
  }
  out << "\n";
}

// Prints changed or unchanged instructions grouped by opcode and print in a
// descending order of the number of instructions for each opcode. If
// top_n_opcodes or max_instructions_per_opcode is a negative number, all the
// instructions will be printed.
// The kind of matched instruction pairs to print.
enum class PairKind { kChanged, kMoved, kUnchanged };

// Returns the users of `instruction` with the operand indices of `instruction`
// in them, e.g. "subtract(1), add(0)".
std::string UsersToString(const HloInstruction* instruction) {
  std::vector<std::string> uses;
  for (const HloInstruction* user : instruction->users()) {
    for (int64_t i = 0; i < user->operand_count(); ++i) {
      if (user->operand(i) == instruction) {
        uses.push_back(absl::StrCat(user->name(), "(", i, ")"));
      }
    }
  }
  return uses.empty() ? "none" : absl::StrJoin(uses, ", ");
}

void PrintMatchedInstructionPairs(
    absl::string_view header,
    const absl::flat_hash_map<const HloInstruction*, const HloInstruction*>&
        instructions,
    std::ostringstream& out, PairKind kind, const RenderTextOptions& options) {
  out << header;
  if (options.top_n_opcodes >= 0) {
    out << " (top " << options.top_n_opcodes << " frequent opcode)";
  }
  if (!options.opcodes_to_ignore.empty()) {
    out << " (ignoring "
        << absl::StrJoin(options.opcodes_to_ignore, ", ",
                         [](std::string* out, const HloOpcode& opcode) {
                           absl::StrAppend(out, HloOpcodeString(opcode));
                         })
        << ")";
  }
  out << ":\n";
  absl::flat_hash_map<
      HloOpcode,
      std::vector<std::pair<const HloInstruction*, const HloInstruction*>>>
      instructions_by_opcode = GroupInstructionPairsByOpcode(instructions);
  std::vector<std::pair<HloOpcode, int64_t>> opcode_counts;
  // The counts are sorted below, so the iteration order doesn't matter.
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (const auto& [opcode, insts] : instructions_by_opcode) {
    opcode_counts.push_back({opcode, insts.size()});
  }
  // Most frequent first; ties are broken by opcode for a deterministic order.
  std::sort(opcode_counts.begin(), opcode_counts.end(),
            [](const auto& a, const auto& b) {
              return a.second != b.second ? a.second > b.second
                                          : a.first < b.first;
            });
  // Print the top N most frequent opcodes
  int i = 0;
  for (auto cit = opcode_counts.begin();
       (options.top_n_opcodes < 0 || i < options.top_n_opcodes) &&
       cit != opcode_counts.end();
       ++cit) {
    if (options.opcodes_to_ignore.contains(cit->first)) {
      continue;
    }
    absl::string_view op_name = HloOpcodeString(cit->first);
    out << "  " << op_name << " (" << cit->second << ")";
    if (kind == PairKind::kChanged) {
      // Count and sort the number of diff types for each opcode
      absl::flat_hash_map<ChangedInstructionDiffType, int64_t> diff_type_counts;
      for (const auto& inst_pair : instructions_by_opcode[cit->first]) {
        std::vector<ChangedInstructionDiffType> diff_types =
            GetChangedInstructionDiffTypes(*inst_pair.first, *inst_pair.second);
        for (const auto& diff_type : diff_types) {
          diff_type_counts[diff_type]++;
        }
      }
      std::vector<std::pair<ChangedInstructionDiffType, int64_t>>
          diff_type_counts_vec(diff_type_counts.begin(),
                               diff_type_counts.end());
      // Most frequent first; ties are broken by diff type for a deterministic
      // order.
      std::sort(diff_type_counts_vec.begin(), diff_type_counts_vec.end(),
                [](const auto& a, const auto& b) {
                  return a.second != b.second ? a.second > b.second
                                              : a.first < b.first;
                });

      out << ", top diff types: "
          << absl::StrJoin(
                 diff_type_counts_vec, ", ",
                 [](std::string* out, const auto& pair) {
                   absl::StrAppend(
                       out, GetChangedInstructionDiffTypeString(pair.first),
                       " (", pair.second, ")");
                 });
    }
    out << "\n";
    std::vector<std::pair<const HloInstruction*, const HloInstruction*>> insts =
        instructions_by_opcode[cit->first];

    // Print the M instructions for each opcode
    int j = 0;
    for (auto iit = insts.begin(); (options.max_instructions_per_opcode < 0 ||
                                    j < options.max_instructions_per_opcode) &&
                                   iit != insts.end();
         ++j, ++iit) {
      if (kind == PairKind::kChanged) {
        std::vector<ChangedInstructionDiffType> diff_types =
            GetChangedInstructionDiffTypes(*iit->first, *iit->second);
        out << "    " << InstructionToString(iit->first, /*name_only=*/true)
            << " and " << InstructionToString(iit->second, /*name_only=*/true)
            << " have changed: "
            << absl::StrJoin(
                   diff_types, ", ",
                   [](std::string* out, const auto& diff_type) {
                     return absl::StrAppend(
                         out, GetChangedInstructionDiffTypeString(diff_type));
                   })
            << "\n";
        if (!options.name_only) {
          out << "      Left: "
              << InstructionToString(iit->first, /*name_only=*/false) << "\n";
          out << "      Right: "
              << InstructionToString(iit->second, /*name_only=*/false) << "\n";
        }
      } else if (kind == PairKind::kMoved) {
        out << "    " << InstructionToString(iit->first, /*name_only=*/true)
            << " and " << InstructionToString(iit->second, /*name_only=*/true)
            << " are used differently\n";
        if (!options.name_only) {
          out << "      Left: "
              << InstructionToString(iit->first, /*name_only=*/false)
              << "\n        used by: " << UsersToString(iit->first) << "\n";
          out << "      Right: "
              << InstructionToString(iit->second, /*name_only=*/false)
              << "\n        used by: " << UsersToString(iit->second) << "\n";
        }
      } else {
        out << "    " << InstructionToString(iit->first, options.name_only)
            << "\n";
      }
    }
    if (j < insts.size()) {
      out << "    ... and " << insts.size() - j << " more " << op_name
          << " instructions\n";
    }
    out << "\n";
    ++i;
  }
  if (i < opcode_counts.size()) {
    out << "  ... and " << opcode_counts.size() - i << " more opcodes\n";
  }
  out << "\n";
}

}  // namespace

void RenderText(const DiffResult& diff_result, std::ostringstream& out,
                const RenderTextOptions& options) {
  // Print unmatched instructions
  PrintUnmatchedInstructions("Unmatched Instructions (left)",
                             diff_result.left_module_unmatched_instructions,
                             out, options);

  PrintUnmatchedInstructions("Unmatched Instructions (right)",
                             diff_result.right_module_unmatched_instructions,
                             out, options);

  // Print changed instructions (print both left and right)
  PrintMatchedInstructionPairs("Changed Instructions",
                               diff_result.changed_instructions, out,
                               PairKind::kChanged, options);

  // Print moved instructions (print both left and right)
  PrintMatchedInstructionPairs("Moved Instructions",
                               diff_result.moved_instructions, out,
                               PairKind::kMoved, options);

  if (options.print_unchanged_instructions) {
    // Print unchanged instructions (print only the first instruction)
    PrintMatchedInstructionPairs("Unchanged Instructions",
                                 diff_result.unchanged_instructions, out,
                                 PairKind::kUnchanged, options);
  }
}

void RenderTextSummary(const DiffResult& diff_result, std::ostringstream& out) {
  // Print a summary of the diff results
  out << "Diff Summary:\n";
  out << "  Unmatched instructions (left): "
      << diff_result.left_module_unmatched_instructions.size() << "\n";
  out << "  Unmatched instructions (right): "
      << diff_result.right_module_unmatched_instructions.size() << "\n";
  out << "  Changed instructions: " << diff_result.changed_instructions.size()
      << "\n";
  out << "  Moved instructions: " << diff_result.moved_instructions.size()
      << "\n";
  out << "  Unchanged instructions: "
      << diff_result.unchanged_instructions.size() << "\n";
}

}  // namespace hlo_diff
}  // namespace xla
