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

#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_html_renderer.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_renderer_util.h"

namespace xla {
namespace hlo_diff {
namespace {

/*** HTML printing functions ***/

// Prints the CSS styles for the HTML output.
std::string PrintCss() {
  return R"(
    <style>
    .section {
      margin: 10px;
      padding: 10px;
      border: 1px solid #ccc;
      border-radius: 5px;
    }
    .section > .header {
      font-size: 16px;
      font-weight: bold;
      margin-bottom: 5px;
    }
    .section > .content {
      font-size: 14px;
    }
  
    details {
      margin: 0;
      padding: 0;
    }
    details > summary {
      font-weight: bold;
      cursor: pointer;
    }
    details > summary:hover {
      background-color: #eee;
    }
    details > .content {
      padding-left: 10px;
    }
  
    .list {
      margin: 0;
      padding: 0;
    }
    .list > .item:hover {
      background-color: #eee;
    }
  
    .tooltip {
      position: relative;
      display: inline-block;
      border-bottom: 1px dotted black;
    }
    .tooltip .tooltiptext {
      visibility: hidden;
      background-color: #555;
      color: #fff;
      text-align: left;
      padding: 5px;
      border-radius: 6px;
      position: absolute;
      z-index: 1;
      top: 50%;
      transform: translateY(-50%);
      left: 110%;
      opacity: 0;
      transition: opacity 0.3s;
      white-space: pre;
      font-family: monospace;
    }
    .tooltip .tooltiptext::after {
      content: " ";
      position: absolute;
      top: 50%;
      right: 100%;
      margin-top: -5px;
      border-width: 5px;
      border-style: solid;
      border-color: transparent #555 transparent transparent;
      white-space: normal;
    }
    .tooltip:hover .tooltiptext {
      visibility: visible;
      opacity: 1;
    }
    </style>
    )";
}

// Prints the div html block.
std::string PrintDiv(absl::string_view content, absl::string_view class_name) {
  return absl::StrFormat("<div class=\"%s\">%s</div>", class_name, content);
}

// Prints the detail html block.
std::string PrintDetails(absl::string_view summary, absl::string_view content) {
  return absl::StrFormat(R"(<details><summary>%s</summary>%s</details>)",
                         summary, PrintDiv(content, "content"));
}

// Prints a html block with a header.
std::string PrintSectionWithHeader(absl::string_view header,
                                   absl::string_view content) {
  return PrintDiv(
      absl::StrCat(PrintDiv(header, "header"), PrintDiv(content, "content")),
      "section");
}

// Prints a list of items.
std::string PrintList(absl::Span<const std::string> items) {
  return PrintDiv(absl::StrJoin(items, "",
                                [](std::string* out, const auto& item) {
                                  absl::StrAppend(out, PrintDiv(item, "item"));
                                }),
                  "list");
}

// Prints a link to the instruction in model explorer if url_generator is not
// null, otherwise returns the text directly.
std::string PrintInstructionLink(const HloInstruction* left_inst,
                                 const HloInstruction* right_inst,
                                 absl::string_view text,
                                 UrlGenerator url_generator) {
  std::string url = url_generator(left_inst, right_inst);

  if (url.empty()) {
    return std::string(text);
  }
  return absl::StrFormat("<a href=\"%s\" target=\"_blank\">%s</a>", url, text);
}

std::string PrintTooltip(absl::string_view text,
                         absl::string_view tooltip_text) {
  return absl::StrFormat(
      R"(<span class="tooltip">%s<span class="tooltiptext">%s</span></span>)",
      text, tooltip_text);
}

/*** Summary logic ***/

// The location of the instruction in the diff result.
enum class InstructionLocation : std::uint8_t { kLeft, kRight };

// Prints a list of instructions.
std::string PrintInstructionsAsList(
    absl::Span<const HloInstruction* const> instructions,
    InstructionLocation location, bool name_only, UrlGenerator url_generator) {
  std::vector<std::string> instructions_list;
  for (const HloInstruction* inst : instructions) {
    std::string link;
    if (location == InstructionLocation::kLeft) {
      link = PrintInstructionLink(inst, /*right_inst=*/nullptr,
                                  InstructionToString(inst, name_only),
                                  url_generator);
    } else {
      link = PrintInstructionLink(/*left_inst=*/nullptr, inst,
                                  InstructionToString(inst, name_only),
                                  url_generator);
    }
    instructions_list.push_back(link);
  }
  return PrintList(instructions_list);
}

// Prints a list of instruction pairs.
std::string PrintInstructionPairsAsList(
    absl::Span<const std::pair<const HloInstruction*, const HloInstruction*>>
        instruction_pairs,
    const std::function<std::string(const HloInstruction*,
                                    const HloInstruction*)>&
        instruction_pair_printer) {
  std::vector<std::string> pair_list;
  for (const auto& pair : instruction_pairs) {
    pair_list.push_back(instruction_pair_printer(pair.first, pair.second));
  }
  return PrintList(pair_list);
}

// Prints unmatched instructions grouped by opcode and print in a descending
// order of the number of instructions for each opcode.
std::string PrintUnmatchedInstructions(
    absl::Span<const HloInstruction* const> instructions,
    InstructionLocation location,
    const absl::flat_hash_set<HloOpcode>& opcodes_to_ignore, bool name_only,
    UrlGenerator url_generator) {
  absl::flat_hash_map<HloOpcode, std::vector<const HloInstruction*>>
      instructions_by_opcode = GroupInstructionsByOpcode(instructions);
  std::vector<std::pair<HloOpcode, int64_t>> opcode_counts;
  for (const auto& [opcode, insts] : instructions_by_opcode) {
    opcode_counts.push_back({opcode, insts.size()});
  }
  std::sort(opcode_counts.begin(), opcode_counts.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  std::stringstream ss;
  for (auto cit = opcode_counts.begin(); cit != opcode_counts.end(); ++cit) {
    if (opcodes_to_ignore.contains(cit->first)) {
      continue;
    }
    ss << PrintDetails(
        absl::StrFormat("%s (%d)", HloOpcodeString(cit->first), cit->second),
        PrintInstructionsAsList(instructions_by_opcode[cit->first], location,
                                name_only, url_generator));
  }
  return ss.str();
}

// Prints instruction pairs grouped by opcode and print in a descending order
// of the number of instruction pairs for each opcode.
std::string PrintInstructionPairsByOpcode(
    const absl::flat_hash_map<const HloInstruction*, const HloInstruction*>&
        instructions,
    const absl::flat_hash_set<HloOpcode>& opcodes_to_ignore,
    const std::function<std::string(const HloInstruction*,
                                    const HloInstruction*)>&
        instruction_pair_printer) {
  absl::flat_hash_map<
      HloOpcode,
      std::vector<std::pair<const HloInstruction*, const HloInstruction*>>>
      instructions_by_opcode = GroupInstructionPairsByOpcode(instructions);
  std::vector<std::pair<HloOpcode, int64_t>> opcode_counts;
  for (const auto& [opcode, insts] : instructions_by_opcode) {
    opcode_counts.push_back({opcode, insts.size()});
  }
  std::sort(opcode_counts.begin(), opcode_counts.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  std::stringstream ss;
  for (auto cit = opcode_counts.begin(); cit != opcode_counts.end(); ++cit) {
    if (opcodes_to_ignore.contains(cit->first)) {
      continue;
    }
    absl::string_view op_name = HloOpcodeString(cit->first);
    ss << PrintDetails(
        absl::StrFormat("%s (%d)", op_name, cit->second),
        PrintInstructionPairsAsList(instructions_by_opcode.at(cit->first),
                                    instruction_pair_printer));
  }
  return ss.str();
}

// Prints the summary of the changed instruction diff type.
std::string PrintChangedInstructionDiffTypeSummary(
    const HloInstruction* left_inst, const HloInstruction* right_inst,
    ChangedInstructionDiffType diff_type) {
  switch (diff_type) {
    case ChangedInstructionDiffType::kShapeChange:
      return absl::StrFormat(
          "left:  %s\nright: %s",
          left_inst->shape().ToString(/*print_layout=*/true),
          right_inst->shape().ToString(/*print_layout=*/true));
    case ChangedInstructionDiffType::kLayoutChange:
      return absl::StrFormat("left:  %s\nright: %s",
                             left_inst->shape().layout().ToString(),
                             right_inst->shape().layout().ToString());
    case ChangedInstructionDiffType::kMemorySpaceChange:
      return absl::StrFormat("left:  %d\nright: %d",
                             left_inst->shape().layout().memory_space(),
                             right_inst->shape().layout().memory_space());
    case ChangedInstructionDiffType::kChangedOperandsNumber:
      return absl::StrFormat("left:  %d\nright: %d", left_inst->operand_count(),
                             right_inst->operand_count());
    case ChangedInstructionDiffType::kChangedOperandsShape: {
      std::vector<std::string> operand_shape_diffs;
      for (int64_t i = 0; i < left_inst->operand_count(); ++i) {
        if (left_inst->operand(i)->shape() != right_inst->operand(i)->shape()) {
          operand_shape_diffs.push_back(absl::StrFormat(
              "operand %d (%s):\n  left:  %s\n  right: %s", i,
              HloOpcodeString(left_inst->operand(i)->opcode()),
              left_inst->operand(i)->shape().ToString(/*print_layout=*/true),
              right_inst->operand(i)->shape().ToString(/*print_layout=*/true)));
        }
      }
      return absl::StrJoin(operand_shape_diffs, "\n");
    }
    case ChangedInstructionDiffType::kOpCodeChanged:
      return absl::StrFormat("left:  %s\nright: %s",
                             HloOpcodeString(left_inst->opcode()),
                             HloOpcodeString(right_inst->opcode()));
    case ChangedInstructionDiffType::kConstantLiteralChanged:
      return absl::StrFormat("left:  %s\nright: %s",
                             left_inst->literal().ToString(),
                             right_inst->literal().ToString());
    default:
      return "Other changes";
  }
}

// Prints changed instructions grouped by opcode and print in a
// descending order of the number of instructions for each opcode.
std::string PrintChangedInstructions(
    const absl::flat_hash_map<const HloInstruction*, const HloInstruction*>&
        instructions,
    const absl::flat_hash_set<HloOpcode>& opcodes_to_ignore,
    UrlGenerator url_generator) {
  auto decorated_printer = [&url_generator](const HloInstruction* left_inst,
                                            const HloInstruction* right_inst) {
    std::vector<ChangedInstructionDiffType> diff_types =
        GetChangedInstructionDiffTypes(*left_inst, *right_inst);
    return absl::StrFormat(
        "%s have changed: %s",
        PrintInstructionLink(
            left_inst, right_inst,
            absl::StrFormat(
                "%s and %s", InstructionToString(left_inst, /*name_only=*/true),
                InstructionToString(right_inst, /*name_only=*/true)),
            url_generator),
        absl::StrJoin(
            diff_types, ", ",
            [&left_inst, &right_inst](std::string* out, const auto& diff_type) {
              std::string diff_type_string =
                  GetChangedInstructionDiffTypeString(diff_type);
              return absl::StrAppend(
                  out,
                  diff_type == ChangedInstructionDiffType::kOtherChange
                      ? diff_type_string
                      : PrintTooltip(diff_type_string,
                                     PrintChangedInstructionDiffTypeSummary(
                                         left_inst, right_inst, diff_type)));
            }));
  };
  return PrintInstructionPairsByOpcode(instructions, opcodes_to_ignore,
                                       decorated_printer);
}

// Prints unchanged instructions grouped by opcode and print in a
// descending order of the number of instructions for each opcode.
std::string PrintUnchangedInstructions(
    const absl::flat_hash_map<const HloInstruction*, const HloInstruction*>&
        instructions,
    const absl::flat_hash_set<HloOpcode>& opcodes_to_ignore,
    UrlGenerator url_generator) {
  auto simple_printer = [&url_generator](const HloInstruction* left_inst,
                                         const HloInstruction* right_inst) {
    return PrintInstructionLink(
        left_inst, right_inst,
        absl::StrFormat("%s and %s",
                        InstructionToString(left_inst, /*name_only=*/true),
                        InstructionToString(right_inst, /*name_only=*/true)),
        url_generator);
  };
  return PrintInstructionPairsByOpcode(instructions, opcodes_to_ignore,
                                       simple_printer);
}

std::string PrintUnmatchedMetricsDiff(
    absl::Span<const HloInstruction* const> instructions,
    GetOpMetricFn get_op_metrics, UrlGenerator url_generator) {
  std::vector<std::pair<const HloInstruction*, double>> sorted_metrics_diff;
  for (const HloInstruction* inst : instructions) {
    if (auto metric = get_op_metrics(inst->name()); metric.has_value()) {
      sorted_metrics_diff.push_back({inst, static_cast<double>(*metric)});
    }
  }

  std::sort(sorted_metrics_diff.begin(), sorted_metrics_diff.end());
  std::vector<std::string> metrics_diff_list(sorted_metrics_diff.size());
  for (const auto& [inst, metrics_diff] : sorted_metrics_diff) {
    metrics_diff_list.push_back(
        absl::StrFormat("%s: %.2f (us)",
                        PrintInstructionLink(inst, /*right_inst=*/nullptr,
                                             inst->name(), url_generator),
                        metrics_diff / 1e6));
  }
  return PrintList(metrics_diff_list);
}

std::string PrintMatchedMetricsDiff(
    const absl::flat_hash_map<const HloInstruction*, const HloInstruction*>&
        instructions,
    GetOpMetricFn left_op_metrics, GetOpMetricFn right_op_metrics,
    UrlGenerator url_generator) {
  std::vector<std::pair<std::pair<const HloInstruction*, const HloInstruction*>,
                        double>>
      sorted_metrics_diff;
  for (const auto& [left_inst, right_inst] : instructions) {
    auto left_metric = left_op_metrics(left_inst->name());
    auto right_metric = right_op_metrics(right_inst->name());
    if (left_metric.has_value() && right_metric.has_value()) {
      sorted_metrics_diff.push_back(
          {{left_inst, right_inst},
           static_cast<double>(*left_metric - *right_metric)});
    }
  }
  std::sort(sorted_metrics_diff.begin(), sorted_metrics_diff.end());
  std::vector<std::string> metrics_diff_list(sorted_metrics_diff.size());
  for (const auto& [inst_pair, metrics_diff] : sorted_metrics_diff) {
    const auto& [left_inst, right_inst] = inst_pair;
    metrics_diff_list.push_back(absl::StrFormat(
        "%s: %.2f (us)",
        PrintInstructionLink(
            left_inst, right_inst,
            absl::StrFormat("%s and %s", left_inst->name(), right_inst->name()),
            url_generator),
        metrics_diff / 1e6));
  }
  return PrintList(metrics_diff_list);
}

}  // namespace

void RenderHtml(const DiffResult& diff_result, const DiffSummary& diff_summary,
                UrlGenerator url_generator, GetOpMetricFn left_op_metrics,
                GetOpMetricFn right_op_metrics, std::ostringstream& out) {
  const absl::flat_hash_set<HloOpcode> ignored_opcodes(kIgnoredOpcodes.begin(),
                                                       kIgnoredOpcodes.end());
  out << PrintCss();

  // Print full diff results
  out << PrintSectionWithHeader(
      "Full Diff Results",
      absl::StrCat(
          PrintDetails(
              absl::StrFormat(
                  "Unmatched Instructions (left) (%d)",
                  diff_result.left_module_unmatched_instructions.size()),
              PrintUnmatchedInstructions(
                  diff_result.left_module_unmatched_instructions,
                  InstructionLocation::kLeft, ignored_opcodes,
                  /*name_only=*/false, url_generator)),
          PrintDetails(
              absl::StrFormat(
                  "Unmatched Instructions (right) (%d)",
                  diff_result.right_module_unmatched_instructions.size()),
              PrintUnmatchedInstructions(
                  diff_result.right_module_unmatched_instructions,
                  InstructionLocation::kRight, ignored_opcodes,
                  /*name_only=*/false, url_generator)),
          PrintDetails(
              absl::StrFormat("Changed Instructions (%d)",
                              diff_result.changed_instructions.size()),
              PrintChangedInstructions(diff_result.changed_instructions,
                                       ignored_opcodes, url_generator))));

  // Print profile metrics diff
  out << PrintSectionWithHeader(
      "Profile Metrics Diff",
      absl::StrCat(
          PrintDetails("Left Module Unmatched Instructions",
                       PrintUnmatchedMetricsDiff(
                           diff_result.left_module_unmatched_instructions,
                           left_op_metrics, url_generator)),
          PrintDetails("Right Module Unmatched Instructions",
                       PrintUnmatchedMetricsDiff(
                           diff_result.right_module_unmatched_instructions,
                           right_op_metrics, url_generator)),
          PrintDetails("Changed Instructions",
                       PrintMatchedMetricsDiff(
                           diff_result.changed_instructions, left_op_metrics,
                           right_op_metrics, url_generator)),
          PrintDetails("Unchanged Instructions",
                       PrintMatchedMetricsDiff(
                           diff_result.unchanged_instructions, left_op_metrics,
                           right_op_metrics, url_generator))));
}

}  // namespace hlo_diff
}  // namespace xla
