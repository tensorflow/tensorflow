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
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
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
#include "xla/hlo/tools/hlo_diff/render/op_metric_getter.h"

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
    details > summary > .decoration {
      font-weight: normal;
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
    .tooltip > .tooltiptext {
      visibility: hidden;
      background-color: #555;
      color: #fff;
      padding: 5px;
      border-radius: 6px;
      position: absolute;
      z-index: 1;
      opacity: 0;
      transition: opacity 0.3s;
      white-space: pre;
      font-family: monospace;
    }
    .tooltip > .tooltiptext::after {
      content: " ";
      position: absolute;
      border-width: 5px;
      border-style: solid;
      white-space: normal;
    }
    .tooltip > .tooltiptext-left {
      top: 50%;
      transform: translateY(-50%);
      right: calc(100% + 10px);
      text-align: left;
    }
    .tooltip > .tooltiptext-left::after {
      top: 50%;
      left: 100%;
      margin-top: -5px;
      border-color: transparent transparent transparent #555;
    }
    .tooltip > .tooltiptext-right {
      top: 50%;
      transform: translateY(-50%);
      left: calc(100% + 10px);
      text-align: right;
    }
    .tooltip > .tooltiptext-right::after {
      top: 50%;
      right: 100%;
      margin-top: -5px;
      border-color: transparent #555 transparent transparent;
    }
    .tooltip:hover > .tooltiptext {
      visibility: visible;
      opacity: 1;
    }

    .hlo-textbox-pair {
      display: flex;
      flex-direction: row;
      width: 100%;
      gap: 10px;
    }
    .hlo-textbox {
      flex: 1;
      min-width: 0;
    }

    .textbox {
      position: relative;
      padding: 10px;
      border: 1px solid #ccc;
      border-radius: 5px;
    }
    .textbox > pre {
      width: 100%;
      margin: 0;
      padding: 0;
      overflow: auto;
      white-space: pre-wrap;
    }
    .textbox > .click-to-copy {
      position: absolute;
      display: inline-block;
      cursor: pointer;
      right: 0px;
      bottom: 0px;
      z-index: 1;
      padding: 5px;
      background-color: #ddd;
      border-radius: 5px;
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
std::string PrintDiv(absl::string_view content,
                     absl::Span<const absl::string_view> class_names) {
  return absl::StrFormat(R"html(<div class="%s">%s</div>)html",
                         absl::StrJoin(class_names, " "), content);
}

// Print the span html block.
std::string PrintSpan(absl::string_view content,
                      absl::Span<const absl::string_view> class_names) {
  return absl::StrFormat(R"html(<span class="%s">%s</span>)html",
                         absl::StrJoin(class_names, " "), content);
}

// Prints the detail html block.
std::string PrintDetails(absl::string_view summary, absl::string_view content) {
  return absl::StrFormat(
      R"html(<details><summary>%s</summary>%s</details>)html", summary,
      PrintDiv(content, {"content"}));
}

// Prints a link to the given url.
std::string PrintLink(absl::string_view text, absl::string_view url) {
  return absl::StrFormat(R"html(<a href="%s" target="_blank">%s</a>)html", url,
                         text);
}

// Prints a html block with a header.
std::string PrintSectionWithHeader(absl::string_view header,
                                   absl::string_view content) {
  return PrintDiv(absl::StrCat(PrintDiv(header, {"header"}),
                               PrintDiv(content, {"content"})),
                  {"section"});
}

// Prints a list of items.
std::string PrintList(absl::Span<const std::string> items) {
  return PrintDiv(absl::StrJoin(items, "",
                                [](std::string* out, const auto& item) {
                                  absl::StrAppend(out,
                                                  PrintDiv(item, {"item"}));
                                }),
                  {"list"});
}

// Prints a list of attribute items.
std::string PrintAttributesList(absl::Span<const std::string> items) {
  return PrintDiv(absl::StrJoin(items, "",
                                [](std::string* out, const auto& item) {
                                  absl::StrAppend(out,
                                                  PrintDiv(item, {"item"}));
                                }),
                  {"attributes-list"});
}

// The position of the tooltip.
enum class TooltipPosition : std::uint8_t { kLeft, kRight };

// Prints a span with a tooltip.
std::string PrintTooltip(absl::string_view text, absl::string_view tooltip_text,
                         TooltipPosition position) {
  return PrintSpan(
      absl::StrCat(text,
                   PrintSpan(tooltip_text,
                             {"tooltiptext", position == TooltipPosition::kLeft
                                                 ? "tooltiptext-left"
                                                 : "tooltiptext-right"})),
      {"tooltip"});
}

// Print click to copy button.
std::string PrintClickToCopyButton(absl::string_view text,
                                   absl::string_view content) {
  return absl::StrFormat(
      R"html(<span class="click-to-copy" onclick="CopyToClipboard(`%s`)">%s</span>)html",
      EscapeStringForHtmlAttribute(content),
      PrintTooltip(text, "Click to copy", TooltipPosition::kLeft));
}

// Print textbox with click to copy button.
std::string PrintTextbox(absl::string_view content) {
  return PrintDiv(
      absl::StrCat(absl::StrFormat(R"html(<pre>%s</pre>)html", content),
                   PrintClickToCopyButton("📋", content)),
      {"textbox"});
}

/*** Summary logic ***/

// Prints a pair of instructions or computations in a text box.
template <typename T>
std::string PrintHloTextboxPair(const T* left_node, const T* right_node) {
  std::vector<std::string> textareas;
  if (left_node != nullptr) {
    textareas.push_back(
        PrintDiv(PrintTextbox(left_node->ToString()), {"hlo-textbox"}));
  }
  if (right_node != nullptr) {
    textareas.push_back(
        PrintDiv(PrintTextbox(right_node->ToString()), {"hlo-textbox"}));
  }
  return PrintDiv(absl::StrJoin(textareas, ""), {"hlo-textbox-pair"});
}

template <typename T>
using DecorationPrinter = std::string(const T*, const T*);

// Prints a pair of instructions or computations. If url_generator is not
// null, a link to the pair of instructions or computations in model
// explorer will be printed.
template <typename T>
std::string PrintNodePair(const T* left_node, const T* right_node,
                          GraphUrlGenerator* url_generator,
                          std::optional<absl::FunctionRef<DecorationPrinter<T>>>
                              decoration_printer = std::nullopt) {
  std::vector<std::string> nodes;
  if (left_node != nullptr) {
    nodes.push_back(std::string(left_node->name()));
  }
  if (right_node != nullptr) {
    nodes.push_back(std::string(right_node->name()));
  }
  std::string text = absl::StrJoin(nodes, " ↔ ");
  std::string url;
  if (url_generator != nullptr) {
    url = url_generator->GenerateWithSelectedNodes(left_node, right_node);
  }
  std::string decoration;
  if (decoration_printer.has_value()) {
    decoration =
        PrintSpan((*decoration_printer)(left_node, right_node), {"decoration"});
  }
  return PrintDetails(
      absl::StrCat(text, " ", decoration),
      absl::StrCat(PrintHloTextboxPair(left_node, right_node),
                   url.empty() ? ""
                               : PrintDiv(PrintLink("Model Explorer", url),
                                          {"model-explorer-url"})));
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
      link = PrintNodePair<HloInstruction>(inst, /*right_node=*/nullptr,
                                           url_generator);
    } else {
      link = PrintNodePair<HloInstruction>(
          /*left_node=*/nullptr, inst, url_generator);
    }
    instructions_list.push_back(link);
  }
  return PrintList(instructions_list);
}

// Prints a list of instruction or computation pairs.
template <typename T>
std::string PrintNodePairsAsList(
    absl::Span<const std::pair<const T*, const T*>> node_pairs,
    GraphUrlGenerator* url_generator,
    std::optional<absl::FunctionRef<DecorationPrinter<T>>> decoration_printer) {
  std::vector<std::string> pair_list;
  for (const auto& pair : node_pairs) {
    pair_list.push_back(PrintNodePair(pair.first, pair.second, url_generator,
                                      decoration_printer));
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
    GraphUrlGenerator* url_generator,
    std::optional<absl::FunctionRef<DecorationPrinter<HloInstruction>>>
        decoration_printer = std::nullopt) {
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
    ss << PrintDetails(absl::StrFormat("%s (%d)", op_name, cit->second),
                       PrintNodePairsAsList<HloInstruction>(
                           instructions_by_opcode.at(cit->first), url_generator,
                           decoration_printer));
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
  std::function<DecorationPrinter<HloInstruction>> decorated_printer =
      [](const HloInstruction* left_inst,
         const HloInstruction* right_inst) -> std::string {
    std::vector<ChangedInstructionDiffType> diff_types =
        GetChangedInstructionDiffTypes(*left_inst, *right_inst);
    return absl::StrCat(
        "have changed: ",
        absl::StrJoin(
            diff_types, ", ",
            [&left_inst, &right_inst](std::string* out, const auto& diff_type) {
              std::string diff_type_string =
                  GetChangedInstructionDiffTypeString(diff_type);
              if (diff_type == ChangedInstructionDiffType::kOtherChange) {
                absl::StrAppend(out, diff_type_string);
              } else {
                absl::StrAppend(
                    out, PrintTooltip(diff_type_string,
                                      PrintChangedInstructionDiffTypeSummary(
                                          left_inst, right_inst, diff_type),
                                      TooltipPosition::kRight));
              }
            }));
  };
  return PrintInstructionPairsByOpcode(instructions, opcodes_to_ignore,
                                       url_generator, decorated_printer);
}

// Prints unchanged instructions grouped by opcode and print in a
// descending order of the number of instructions for each opcode.
std::string PrintUnchangedInstructions(
    const absl::flat_hash_map<const HloInstruction*, const HloInstruction*>&
        instructions,
    const absl::flat_hash_set<HloOpcode>& opcodes_to_ignore,
    GraphUrlGenerator* url_generator) {
  return PrintInstructionPairsByOpcode(instructions, opcodes_to_ignore,
                                       url_generator);
}

/* Metrics diff */

// Prints unmatched instructions sorted by the metrics diff.
std::string PrintUnmatchedMetricsDiff(
    const absl::flat_hash_set<const HloInstruction*>& instructions,
    const OpMetricGetter& op_metric_getter, GraphUrlGenerator* url_generator) {
  std::vector<std::pair<const HloInstruction*, double>> sorted_metrics_diff;
  for (const HloInstruction* inst : instructions) {
    if (auto time_ps = op_metric_getter.GetOpTimePs(inst->name());
        time_ps.ok()) {
      sorted_metrics_diff.push_back({inst, static_cast<double>(*time_ps)});
    }
  }

  std::sort(sorted_metrics_diff.begin(), sorted_metrics_diff.end(),
            [](const auto& a, const auto& b) {
              // Sort by the absolute value of the diff in descending order.
              return std::abs(a.second) > std::abs(b.second);
            });
  std::vector<std::string> metrics_diff_list(sorted_metrics_diff.size());
  for (const auto& entry : sorted_metrics_diff) {
    const HloInstruction* inst = entry.first;
    double metrics_diff = entry.second;
    metrics_diff_list.push_back(PrintNodePair<HloInstruction>(
        inst, /*right_node=*/nullptr, url_generator,
        [&metrics_diff](const HloInstruction* inst,
                        const HloInstruction*) -> std::string {
          return absl::StrFormat("%+.2f (us)", metrics_diff / 1e6);
        }));
  }
  return PrintList(metrics_diff_list);
}

// Prints matched instructions sorted by the metrics diff.
std::string PrintMatchedMetricsDiff(
    const absl::flat_hash_map<const HloInstruction*, const HloInstruction*>&
        instructions,
    const OpMetricGetter& left_op_metric_getter,
    const OpMetricGetter& right_op_metric_getter,
    GraphUrlGenerator* url_generator) {
  std::vector<std::pair<std::pair<const HloInstruction*, const HloInstruction*>,
                        double>>
      sorted_metrics_diff;
  for (const auto& [left_inst, right_inst] : instructions) {
    absl::StatusOr<uint64_t> left_time_ps =
        left_op_metric_getter.GetOpTimePs(left_inst->name());
    absl::StatusOr<uint64_t> right_time_ps =
        right_op_metric_getter.GetOpTimePs(right_inst->name());
    if (!left_time_ps.ok() || !right_time_ps.ok()) {
      continue;
    }
    sorted_metrics_diff.push_back({{left_inst, right_inst},
                                   static_cast<double>(*right_time_ps) -
                                       static_cast<double>(*left_time_ps)});
  }
  std::sort(sorted_metrics_diff.begin(), sorted_metrics_diff.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  std::vector<std::string> metrics_diff_list(sorted_metrics_diff.size());
  for (const auto& entry : sorted_metrics_diff) {
    const HloInstruction* left_inst = entry.first.first;
    const HloInstruction* right_inst = entry.first.second;
    double metrics_diff = entry.second;
    metrics_diff_list.push_back(PrintNodePair<HloInstruction>(
        left_inst, right_inst, url_generator,
        [&metrics_diff](const HloInstruction* left_inst,
                        const HloInstruction* right_inst) -> std::string {
          return absl::StrFormat("%+.2f (us)", metrics_diff / 1e6);
        }));
  }
  return PrintList(metrics_diff_list);
}

/* Diff pattern */

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
      computation_pair_list.push_back(PrintNodePair<HloComputation>(
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
                GraphUrlGenerator* url_generator,
                OpMetricGetter* left_op_metric_getter,
                OpMetricGetter* right_op_metric_getter,
                std::ostringstream& out) {
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
  if (left_op_metric_getter != nullptr && right_op_metric_getter != nullptr) {
    out << PrintSectionWithHeader(
        "Profile Metrics Diff",
        absl::StrCat(
            PrintDetails("Left Module Unmatched Instructions",
                         PrintUnmatchedMetricsDiff(
                             diff_result.left_module_unmatched_instructions,
                             *left_op_metric_getter, url_generator)),
            PrintDetails("Right Module Unmatched Instructions",
                         PrintUnmatchedMetricsDiff(
                             diff_result.right_module_unmatched_instructions,
                             *right_op_metric_getter, url_generator)),
            PrintDetails(
                "Changed Instructions",
                PrintMatchedMetricsDiff(
                    diff_result.changed_instructions, *left_op_metric_getter,
                    *right_op_metric_getter, url_generator)),
            PrintDetails(
                "Unchanged Instructions",
                PrintMatchedMetricsDiff(
                    diff_result.unchanged_instructions, *left_op_metric_getter,
                    *right_op_metric_getter, url_generator))));
  }

  // Print repetitive computation groups
  out << PrintSectionWithHeader(
      "Group of computations with the same diff",
      PrintRepetitiveDiffPatterns(diff_summary.computation_diff_patterns,
                                  url_generator));
}

}  // namespace hlo_diff
}  // namespace xla
