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
#include "xla/hlo/tools/hlo_diff/render/graph_url_generator.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_renderer_util.h"

namespace xla {
namespace hlo_diff {
namespace {

/*** HTML printing functions ***/

// Prints the CSS styles for the HTML output.
std::string PrintCss() {
  return R"html(
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
      padding-left: 20px;
    }

    .list {
      margin: 0;
      padding: 0;
    }
    .list > .item:hover {
      background-color: #eee;
    }

    .attributes-list {
      margin: 0;
      padding: 0;
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
      left: calc(100% + 10px);
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

  .click-to-copy {
    position: relative;
    display: inline-block;
    cursor: pointer;
  }
    </style>
  )html";
}

// Prints javascript for the HTML output.
std::string PrintJavascript() {
  return R"html(
  <script>
  function CopyToClipboard(text) {
    navigator.clipboard.writeText(text);
    const tooltip = event.srcElement.querySelector('.tooltiptext');
    tooltip.textContent = 'Copied to clipboard';
    setTimeout(() => {
      tooltip.textContent = 'Click to copy';
    }, 2000);
  }
  </script>
  )html";
}

// Escapes the string for html attribute.
std::string EscapeStringForHtmlAttribute(absl::string_view str) {
  std::string escaped_str;
  for (char c : str) {
    switch (c) {
      case '&':
        absl::StrAppend(&escaped_str, "&amp;");
        break;
      case '<':
        absl::StrAppend(&escaped_str, "&lt;");
        break;
      case '>':
        absl::StrAppend(&escaped_str, "&gt;");
        break;
      case '"':
        absl::StrAppend(&escaped_str, "&quot;");
        break;
      case '\'':
        absl::StrAppend(&escaped_str, "&#39;");
        break;
      default:
        absl::StrAppend(&escaped_str, absl::string_view(&c, 1));
        break;
    }
  }
  return escaped_str;
}

// Prints the div html block.
std::string PrintDiv(absl::string_view content, absl::string_view class_name) {
  return absl::StrFormat(R"html(<div class="%s">%s</div>)html", class_name,
                         content);
}

// Prints the detail html block.
std::string PrintDetails(absl::string_view summary, absl::string_view content) {
  return absl::StrFormat(
      R"html(<details><summary>%s</summary>%s</details>)html", summary,
      PrintDiv(content, "content"));
}

// Prints a link to the given url.
std::string PrintLink(absl::string_view text, absl::string_view url) {
  return absl::StrFormat("<a href=\"%s\" target=\"_blank\">%s</a>", url, text);
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

// Prints a list of attribute items.
std::string PrintAttributesList(absl::Span<const std::string> items) {
  return PrintDiv(absl::StrJoin(items, "",
                                [](std::string* out, const auto& item) {
                                  absl::StrAppend(out, PrintDiv(item, "item"));
                                }),
                  "attributes-list");
}

// Prints a span with a tooltip.
std::string PrintTooltip(absl::string_view text,
                         absl::string_view tooltip_text) {
  return absl::StrFormat(
      R"html(<span class="tooltip">%s<span class="tooltiptext">%s</span></span>)html",
      text, tooltip_text);
}

// Print click to copy button.
std::string PrintClickToCopyButton(absl::string_view text,
                                   absl::string_view content) {
  return absl::StrFormat(
      R"html(<span class="click-to-copy" onclick="CopyToClipboard(`%s`)">%s</span>)html",
      EscapeStringForHtmlAttribute(content),
      PrintTooltip(text, "Click to copy"));
}

/*** Summary logic ***/

// Prints the instruction name and click to copy button that copy the text
// format.
std::string PrintInstruction(const HloInstruction& inst) {
  return absl::StrFormat("%s (%s)", inst.name(),
                         PrintClickToCopyButton("text", inst.ToString()));
}

// Prints a pair of instructions. If url_generator is not null, a link to the
// pair of instructions in model explorer will be printed.
std::string PrintInstructionPair(const HloInstruction* left_inst,
                                 const HloInstruction* right_inst,
                                 GraphUrlGenerator* url_generator) {
  std::vector<std::string> instructions;
  if (left_inst != nullptr) {
    instructions.push_back(PrintInstruction(*left_inst));
  }
  if (right_inst != nullptr) {
    instructions.push_back(PrintInstruction(*right_inst));
  }
  std::string text = absl::StrJoin(instructions, " ↔ ");
  if (url_generator == nullptr) {
    return text;
  }
  std::string url = url_generator->Generate(left_inst, right_inst);
  if (url.empty()) {
    return text;
  }
  return absl::StrCat(text, " (", PrintLink("Model Explorer", url), ")");
}

// Prints computation name and click to copy button that copy the text format.
std::string PrintComputation(const HloComputation& comp) {
  return absl::StrFormat("%s (%s)", comp.name(),
                         PrintClickToCopyButton("text", comp.ToString()));
}

// Prints a pair of computations. If url_generator is not null, a link to the
// pair of computations in model explorer will be printed.
std::string PrintComputationPair(const HloComputation* left_comp,
                                 const HloComputation* right_comp,
                                 GraphUrlGenerator* url_generator) {
  std::vector<std::string> computations;
  if (left_comp != nullptr) {
    computations.push_back(PrintComputation(*left_comp));
  }
  if (right_comp != nullptr) {
    computations.push_back(PrintComputation(*right_comp));
  }
  std::string text = absl::StrJoin(computations, " ↔ ");
  if (url_generator == nullptr) {
    return text;
  }
  std::string url = url_generator->Generate(left_comp, right_comp);
  if (url.empty()) {
    return text;
  }
  return absl::StrCat(text, " (", PrintLink("Model Explorer", url), ")");
}

// The location of the instruction in the diff result.
enum class InstructionLocation : std::uint8_t { kLeft, kRight };

// Prints a list of instructions.
std::string PrintInstructionsAsList(
    absl::Span<const HloInstruction* const> instructions,
    InstructionLocation location, bool name_only,
    GraphUrlGenerator* url_generator) {
  std::vector<std::string> instructions_list;
  for (const HloInstruction* inst : instructions) {
    std::string link;
    if (location == InstructionLocation::kLeft) {
      link = PrintInstructionPair(inst, /*right_inst=*/nullptr, url_generator);
    } else {
      link = PrintInstructionPair(/*left_inst=*/nullptr, inst, url_generator);
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
    const absl::flat_hash_set<const HloInstruction*>& instructions,
    InstructionLocation location,
    const absl::flat_hash_set<HloOpcode>& opcodes_to_ignore, bool name_only,
    GraphUrlGenerator* url_generator) {
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
    GraphUrlGenerator* url_generator) {
  auto decorated_printer = [&url_generator](const HloInstruction* left_inst,
                                            const HloInstruction* right_inst) {
    std::vector<ChangedInstructionDiffType> diff_types =
        GetChangedInstructionDiffTypes(*left_inst, *right_inst);
    return absl::StrFormat(
        "%s have changed: %s",
        PrintInstructionPair(left_inst, right_inst, url_generator),
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
    GraphUrlGenerator* url_generator) {
  auto simple_printer = [&url_generator](const HloInstruction* left_inst,
                                         const HloInstruction* right_inst) {
    return PrintInstructionPair(left_inst, right_inst, url_generator);
  };
  return PrintInstructionPairsByOpcode(instructions, opcodes_to_ignore,
                                       simple_printer);
}

std::string PrintUnmatchedMetricsDiff(
    const absl::flat_hash_set<const HloInstruction*>& instructions,
    GetOpMetricFn get_op_metrics, GraphUrlGenerator* url_generator) {
  std::vector<std::pair<const HloInstruction*, double>> sorted_metrics_diff;
  for (const HloInstruction* inst : instructions) {
    if (auto metric = get_op_metrics(inst->name()); metric.has_value()) {
      sorted_metrics_diff.push_back({inst, static_cast<double>(*metric)});
    }
  }

  std::sort(sorted_metrics_diff.begin(), sorted_metrics_diff.end());
  std::vector<std::string> metrics_diff_list(sorted_metrics_diff.size());
  for (const auto& [inst, metrics_diff] : sorted_metrics_diff) {
    metrics_diff_list.push_back(absl::StrFormat(
        "%s: %.2f (us)",
        PrintInstructionPair(inst, /*right_inst=*/nullptr, url_generator),
        metrics_diff / 1e6));
  }
  return PrintList(metrics_diff_list);
}

std::string PrintMatchedMetricsDiff(
    const absl::flat_hash_map<const HloInstruction*, const HloInstruction*>&
        instructions,
    GetOpMetricFn left_op_metrics, GetOpMetricFn right_op_metrics,
    GraphUrlGenerator* url_generator) {
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
        PrintInstructionPair(left_inst, right_inst, url_generator),
        metrics_diff / 1e6));
  }
  return PrintList(metrics_diff_list);
}

// Summarize a diff pattern.
std::string SummarizeDiffPattern(const ComputationDiffPattern& diff_pattern) {
  if (diff_pattern.computation_groups.size() > 1) {
    return absl::StrFormat("Summarized %d computations with the same diff",
                           diff_pattern.computation_groups.size());
  }
  return "A single computation has unique diff";
}

// Prints the summary of the repetitive diff patterns.
std::string PrintRepetitiveDiffPatterns(
    absl::Span<const ComputationDiffPattern> diff_patterns,
    GraphUrlGenerator* url_generator) {
  // Sort the diff patterns by the number of computations in each group in
  // descending order.
  std::vector<ComputationDiffPattern> sorted_diff_patterns;
  for (const ComputationDiffPattern& diff_pattern : diff_patterns) {
    sorted_diff_patterns.push_back(diff_pattern);
  }
  std::sort(
      sorted_diff_patterns.begin(), sorted_diff_patterns.end(),
      [](const ComputationDiffPattern& a, const ComputationDiffPattern& b) {
        return a.computation_groups.size() > b.computation_groups.size();
      });
  std::string computation_group_list;
  int i = 0;
  for (const auto& diff_pattern : sorted_diff_patterns) {
    if (diff_pattern.computation_groups.empty()) {
      continue;
    }
    const ComputationGroup& sample = diff_pattern.computation_groups[0];
    // We only print the one-to-one mapping for now.
    if (sample.left_computations.size() != 1 ||
        sample.right_computations.size() != 1) {
      continue;
    }
    std::vector<std::string> computation_pair_list;
    for (const ComputationGroup& computation_group :
         diff_pattern.computation_groups) {
      if (computation_group.left_computations.size() != 1 ||
          computation_group.right_computations.size() != 1) {
        continue;
      }
      const HloComputation* left_computation =
          computation_group.left_computations[0];
      const HloComputation* right_computation =
          computation_group.right_computations[0];
      computation_pair_list.push_back(PrintComputationPair(
          left_computation, right_computation, url_generator));
    }
    absl::StrAppend(
        &computation_group_list,
        PrintDetails(
            absl::StrFormat("Group %d: %s (Sample: %s → %s)", ++i,
                            SummarizeDiffPattern(diff_pattern),
                            sample.left_computations[0]->name(),
                            sample.right_computations[0]->name()),
            PrintAttributesList(
                {absl::StrFormat(
                     "Instruction count: %d → %d",
                     sample.left_computations[0]->instruction_count(),
                     sample.right_computations[0]->instruction_count()),
                 absl::StrFormat(
                     "Diff summary: %d changed, %d left unmatched, %d right "
                     "unmatched",
                     diff_pattern.diff_metrics.changed_instruction_count,
                     diff_pattern.diff_metrics.left_unmatched_instruction_count,
                     diff_pattern.diff_metrics
                         .right_unmatched_instruction_count),
                 PrintDetails("Instances",
                              PrintList(computation_pair_list))})));
  }
  return computation_group_list;
}

}  // namespace

void RenderHtml(const DiffResult& diff_result, const DiffSummary& diff_summary,
                GraphUrlGenerator* url_generator, GetOpMetricFn left_op_metrics,
                GetOpMetricFn right_op_metrics, std::ostringstream& out) {
  const absl::flat_hash_set<HloOpcode> ignored_opcodes(kIgnoredOpcodes.begin(),
                                                       kIgnoredOpcodes.end());
  out << PrintCss() << PrintJavascript();

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

  // Print repetitive computation groups
  out << PrintSectionWithHeader(
      "Group of computations with the same diff",
      PrintRepetitiveDiffPatterns(diff_summary.computation_diff_patterns,
                                  url_generator));
}

}  // namespace hlo_diff
}  // namespace xla
