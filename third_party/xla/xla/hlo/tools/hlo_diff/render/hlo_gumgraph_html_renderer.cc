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
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"
#include "xla/hlo/tools/hlo_diff/render/graph_url_generator.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_renderer_util.h"
#include "xla/hlo/tools/hlo_diff/render/op_metric_getter.h"
#include "xla/printer.h"
#include "xla/shape_util.h"
#include "tsl/platform/fingerprint.h"

namespace xla {
namespace hlo_diff {
namespace {

/*** HTML printing functions ***/

// Prints the CSS styles for the HTML output.
std::string PrintCss() {
  return R"html(
    <style>
    html {
      font-family: 'Google Sans', sans-serif;
    }
    .section {
      margin: 10px;
      padding: 10px;
      border: 1px solid #cccccc;
      border-radius: 5px;
    }
    .section > .header {
      font-size: 16px;
      font-weight: bold;
      padding: 0.5rem;
      background-color: white; /* Add a background to cover content while sticking */
      position: sticky;
      top: 0;
      z-index: 2; /* Ensure it stays above other content */
      border-bottom: 1px solid #cccccc;
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
      padding: 3px 5px;
    }
    details > summary:hover {
      background-color: #eeeeee;
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
      background-color: #eeeeee;
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
      background-color: #555555;
      color: #ffffff;
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
      border-color: transparent transparent transparent #555555;
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
      border-color: transparent #555555 transparent transparent;
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
      display: flex;
      flex-direction: column;
    }
    .hlo-textbox > .textbox {
      position: relative;
      padding: 10px;
      border: 1px solid #cccccc;
      border-radius: 5px;
      height: 100%;
      box-sizing: border-box;
    }
    .hlo-textbox > .textbox > pre {
      width: 100%;
      margin: 0;
      padding: 0;
      overflow: auto;
      white-space: pre-wrap;
      max-height: 800px;
    }
    .hlo-textbox > .textbox > .click-to-copy {
      position: absolute;
      display: inline-block;
      cursor: pointer;
      right: 0px;
      top: 0px;
      z-index: 1;
      padding: 5px;
      background-color: #dddddd;
      border-radius: 5px;
    }

    span.yellow {
      color: #fbbc04;
    }
    span.green {
      color: #34a853;
    }
    span.red {
      color: #ea4335;
    }
    span.grey {
      color: #999999;
    }

    span.hlo-instruction {
      display: inline-block;
      cursor: pointer;
    }
    span.hlo-instruction:hover {
      border: 2px solid #4285F4;
    }
    span.bordered {
      border: 2px solid #4285F4;
    }

    span.red-highlight {
      background-color: #fad2cf;
    }
    span.green-highlight {
      background-color: #ceead6;
    }
    span.yellow-highlight {
      background-color: #feefc3;
    }
    span.temp-highlight {
      background-color: #a8c7fa;
      opacity: 0.7;
      animation: breathe-highlight 1s infinite alternate;
    }
    @keyframes breathe-highlight {
      0% { opacity: 0.7; }
      50% { opacity: 0.9; }
      100% { opacity: 0.7; }
    }

    .hlo-instruction.hidden {
      display: none;
    }
    button {
      background-color: #3f51b5;
      color: white;
      font-weight: bold;
      padding: 0.2rem 0.5rem;
      margin-left: 1rem;
      border-radius: 0.5rem;
      transition: all 0.2s ease-in-out;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    button:hover {
      transform: translateY(-1px) scale(1.02);
    }

    /* Styles for the system message pop-up */
    .system-message-container {
        /* Position fixed at the bottom of the viewport */
        position: fixed;
        bottom: 7px;
        left: 50%;
        transform: translateX(-50%) translateY(100%);

        /* Appearance and transitions */
        background-color: #F5F3FF;
        border: 2px solid #8B5CF6;
        opacity: 0;
        visibility: hidden;
        transition: transform 0.3s ease-in-out, opacity 0.3s ease-in-out, visibility 0s 0.3s;
        z-index: 1000;
    }

    /* Class to show the message */
    .system-message-container.show-message {
        transform: translateX(-50%) translateY(0);
        opacity: 1;
        visibility: visible;
        transition: transform 0.3s ease-in-out, opacity 0.3s ease-in-out;
    }

    .system-message-container.show-message span {
      color: #4C1D95;
      font-weight: 500;
      font-size: 16px;
      padding: 1.5rem;
    }

    </style>
  )html";
}

// Prints javascript for the HTML output.
std::string PrintJavascript() {
  return R"html(
  <script defer>
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

std::string PrintJavascriptForHoverEvent() {
  return R"html(
  <script>
  function ShowSystemMessage(message) {
      const messageContainer = document.getElementById('system-message');
      const messageText = messageContainer.querySelector('span');
      messageText.textContent = message;

      // Show the message pop-up
      messageContainer.classList.add('show-message');

      // Automatically hide the message after 3 seconds
      setTimeout(() => {
      messageContainer.classList.remove('show-message');
      }, 3000);
  }

  const allSpans = document.querySelectorAll('span[data-diffid]');
  allSpans.forEach(span => {
      span.addEventListener('mouseover', handleMouseOver);
      span.addEventListener('mouseout', handleMouseOut);
      span.addEventListener('click', handleSpanClick);
      span.addEventListener('dblclick', handleSpanDoubleClick);
  });
  function handleMouseOver(event) {
      const diffId = event.target.getAttribute('data-diffid');
      if (!diffId) {
          return;
      }
      const relatedSpans = document.querySelectorAll(`span[data-diffid="${diffId}"]`);
      relatedSpans.forEach(relatedSpan => {
          relatedSpan.classList.add('bordered');
      });
  }

  function handleMouseOut(event) {
      const diffId = event.target.getAttribute('data-diffid');
      if (!diffId) {
          return;
      }
      const relatedSpans = document.querySelectorAll(`span[data-diffid="${diffId}"]`);
      relatedSpans.forEach(relatedSpan => {
          relatedSpan.classList.remove('bordered');
      });
  }

  function handleSpanClick(event) {
      const diffId = event.target.getAttribute('data-diffid');
      if (!diffId) {
          return;
      }

      const clickedSpan = event.target;
      const clickedPre = clickedSpan.closest('.hlo-textbox').querySelector('pre');
      if (!clickedPre) return;

      const idParts = clickedPre.id.split('-');
      const fingerprint = idParts[0];
      const isLeft = idParts[1] === 'left';
      const siblingPreId = fingerprint + '-' + (isLeft ? 'right' : 'left');
      const siblingPre = document.getElementById(siblingPreId);

      if (siblingPre) {
          const targetSpan = siblingPre.querySelector(`span[data-diffid="${diffId}"]`);
          if (targetSpan) {
              // Calculate the vertical offset of the clicked span from the top of its visible area.
              const clickedSpanViewportOffset = clickedSpan.offsetTop - clickedPre.scrollTop;

              // Calculate the percentage of this offset within the clickedPre's visible height.
              const percentage = clickedPre.clientHeight > 0 ?
                  clickedSpanViewportOffset / clickedPre.clientHeight : 0;

              // Calculate the desired offset for the targetSpan within the siblingPre's visible area.
              const desiredSiblingViewportOffset = percentage * siblingPre.clientHeight;

              // Calculate the new scrollTop for siblingPre to achieve this alignment.
              const newScrollTop = targetSpan.offsetTop - desiredSiblingViewportOffset;

              // Scroll the siblingPre element smoothly.
              siblingPre.scrollTo({
                  top: Math.max(0, newScrollTop),
                  behavior: 'smooth'
              });

              // Temporarily highlight the target span
              targetSpan.classList.add('temp-highlight');
              setTimeout(() => {
                  targetSpan.classList.remove('temp-highlight');
              }, 2000); // Remove highlight after 2 seconds
          } else {
              ShowSystemMessage("Corresponding instruction is in another computation, double click to jump to it.");
          }
      }
  }

  function handleSpanDoubleClick(event) {
      const diffId = event.target.getAttribute('data-diffid');
      if (!diffId) {
          return;
      }

      const clickedSpan = event.target;
      const clickedPre = clickedSpan.closest('.hlo-textbox').querySelector('pre');
      if (!clickedPre) return;

      const idParts = clickedPre.id.split('-');
      const fingerprint = idParts[0];
      const isLeft = idParts[1] === 'left';
      const siblingPreId = fingerprint + '-' + (isLeft ? 'right' : 'left');
      const siblingPre = document.getElementById(siblingPreId);

      if (siblingPre) {
          const targetSpan = siblingPre.querySelector(`span[data-diffid="${diffId}"]`);
          if (!targetSpan) {
              // Case 2: Corresponding span NOT found in siblingPre.
              // Search for the span with the same diffId in other hlo-textbox-pairs.
              const allMatchingSpans = document.querySelectorAll(`span[data-diffid="${diffId}"]`);
              let foundSpanInOtherPre = null;
              allMatchingSpans.forEach(span => {
                  if (span !== clickedSpan) {
                      foundSpanInOtherPre = span;
                  }
              });

              if (foundSpanInOtherPre) {
                  const foundPre = foundSpanInOtherPre.closest('.hlo-textbox').querySelector('pre');
                  if (foundPre) {
                      // Find ancestor detail and open it.
                      let parentDetails = foundSpanInOtherPre.closest('details');
                      while (parentDetails) {
                          parentDetails.open = true;
                          parentDetails = parentDetails.parentElement ? parentDetails.parentElement.closest('details') : null;
                      }

                      // Scroll the foundPre to make the foundSpanInOtherPre visible.
                      foundSpanInOtherPre.scrollIntoView({ behavior: 'smooth', block: 'center' });

                      // Temporarily highlight the found span
                      foundSpanInOtherPre.classList.add('temp-highlight');
                      setTimeout(() => {
                          foundSpanInOtherPre.classList.remove('temp-highlight');
                      }, 3000);
                  }
              }
          }
      }
  }
  </script>
  )html";
}

std::string PrintJavascriptForToggleButton() {
  return R"html(
  <script>
  const toggleButton = document.getElementById('toggleButton');
    const hloInstructions = document.querySelectorAll('.hlo-instruction:not(.highlighted)');
    let isHidden = false;

    toggleButton.addEventListener('click', () => {
      isHidden = !isHidden;
      hloInstructions.forEach(instruction => {
        instruction.classList.toggle('hidden', isHidden);
      });
      toggleButton.textContent = isHidden ? 'Show Unchanged Instructions' : 'Hide Unchanged Instructions';
    });
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
                     absl::Span<const absl::string_view> class_names,
                     absl::string_view id = "") {
  std::string div_id = id.empty() ? "" : absl::StrCat(" id=\"", id, "\"");
  return absl::StrFormat(R"html(<div class="%s"%s>%s</div>)html",
                         absl::StrJoin(class_names, " "), div_id, content);
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

// Prints a system message placeholder.
std::string PrintSystemMessagePlaceholder() {
  return PrintDiv(PrintSpan("System message placeholder", {}),
                  {"system-message-container"}, "system-message");
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

// Prints a button
std::string PrintButton(absl::string_view id, absl::string_view text) {
  return absl::StrFormat(
      R"html(<button id="%s" class="button">%s</button>)html", id, text);
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
std::string PrintTextbox(absl::string_view title, absl::string_view content,
                         absl::string_view id = "") {
  return absl::StrCat(
      PrintDiv(title, {"title"}),
      PrintDiv(absl::StrCat(absl::StrFormat(R"html(<pre id="%s">%s</pre>)html",
                                            id, content),
                            PrintClickToCopyButton("📋", content)),
               {"textbox"}));
}

/*** Summary logic ***/

// The attributes of an instruction that will be applied to the corresponding
// span element in the HTML output.
struct Attributes {
  std::string highlight;
  std::string diffid;
};

// Generate span attributes for all instructions given diff result.
absl::flat_hash_map<const HloInstruction*, Attributes> GenerateSpanAttributes(
    const DiffResult& diff_result) {
  absl::flat_hash_map<const HloInstruction*, Attributes> span_attributes;
  for (auto& instruction : diff_result.left_module_unmatched_instructions) {
    span_attributes[instruction] = {"red-highlight"};
  }
  for (auto& instruction : diff_result.right_module_unmatched_instructions) {
    span_attributes[instruction] = {"green-highlight"};
  }
  for (const auto& [l_instruction, r_instruction] :
       diff_result.changed_instructions) {
    span_attributes[l_instruction] = {
        "yellow-highlight",
        absl::StrCat(l_instruction->name(), "::", r_instruction->name())};
    span_attributes[r_instruction] = {
        "yellow-highlight",
        absl::StrCat(l_instruction->name(), "::", r_instruction->name())};
  }
  for (const auto& [l_instruction, r_instruction] :
       diff_result.unchanged_instructions) {
    span_attributes[l_instruction] = {
        "", absl::StrCat(l_instruction->name(), "::", r_instruction->name())};
    span_attributes[r_instruction] = {
        "", absl::StrCat(l_instruction->name(), "::", r_instruction->name())};
  }
  return span_attributes;
};

// Generates HTML for a single HloComputation with diff highlights.
std::string PrintHloComputationToHtml(
    const HloComputation* comp,
    const absl::flat_hash_map<const HloInstruction*, Attributes>&
        span_attributes) {
  if (comp == nullptr) {
    return "";
  }
  StringPrinter printer;

  // Mimic HloComputation::Print structure with default options.
  printer.Append("%");
  printer.Append(comp->name());
  printer.Append(" ");
  ShapeUtil::PrintHumanString(&printer,
                              comp->ComputeProgramShape(/*include_ids=*/true));
  printer.Append(" ");
  printer.Append("{\n");

  // Print instructions in this computation.
  {
    // Options for printing individual instructions.
    // Default indent_amount + 1 = 1. is_in_nested_computation = true.
    HloPrintOptions instruction_print_options = HloPrintOptions::Default();
    instruction_print_options.set_indent_amount(1);
    instruction_print_options.set_is_in_nested_computation(true);

    CanonicalNameMap name_map;
    name_map.Reserve(comp->instruction_count());

    // Iterate through instructions in the default order: post-order.
    std::vector<HloInstruction*> instruction_order =
        comp->MakeInstructionPostOrder();
    for (const HloInstruction* instruction : instruction_order) {
      DCHECK_EQ(comp, instruction->parent());

      auto it = span_attributes.find(instruction);
      std::string highlight_class =
          it != span_attributes.end() && !it->second.highlight.empty()
              ? std::string(it->second.highlight) + " highlighted"
              : "";
      std::string diffid =
          it != span_attributes.end() && !it->second.diffid.empty()
              ? absl::StrCat("data-diffid=\"",
                             EscapeStringForHtmlAttribute(it->second.diffid),
                             "\"")
              : "";
      printer.Append(absl::StrCat("<span class=\"hlo-instruction ",
                                  highlight_class, "\"", diffid, " >"));
      printer.Append("  ");  // Instruction indentation (2 spaces)
      if (instruction == comp->root_instruction()) {
        printer.Append("ROOT ");
      }
      instruction->PrintWithCanonicalNameMap(
          &printer, instruction_print_options, &name_map);
      printer.Append("</span>");
    }
  }

  printer.Append("}");  // Closing brace for computation.

  // Default print_ids is true, so execution thread is printed if not main.
  if (!comp->IsMainThread()) {
    printer.Append(", execution_thread=\"");
    printer.Append(comp->execution_thread());
    printer.Append("\"");
  }

  return std::move(printer).ToString();
}

// Prints a pair of instructions or computations in a text box.
template <typename T>
std::string PrintHloTextboxPair(
    const T* left_node, const T* right_node,
    const absl::flat_hash_map<const HloInstruction*, Attributes>&
        span_attributes) {
  std::string left_title = "None", right_title = "None", left_text, right_text;
  if (left_node != nullptr) {
    left_title = left_node->name();
    if constexpr (std::is_same_v<T, HloComputation>) {
      left_text = PrintHloComputationToHtml(left_node, span_attributes);
    } else {
      left_text = left_node->ToString();
    }
  }
  if (right_node != nullptr) {
    right_title = right_node->name();
    if constexpr (std::is_same_v<T, HloComputation>) {
      right_text = PrintHloComputationToHtml(right_node, span_attributes);
    } else {
      right_text = right_node->ToString();
    }
  }

  uint64_t fingerprint =
      tsl::Fingerprint64(absl::StrCat(left_text, right_text));
  return PrintDiv(
      absl::StrCat(
          PrintDiv(PrintTextbox(left_title, left_text,
                                absl::StrFormat("%016x-left", fingerprint)),

                   {"hlo-textbox"}),
          PrintDiv(PrintTextbox(right_title, right_text,
                                absl::StrFormat("%016x-right", fingerprint)),

                   {"hlo-textbox"})),
      {"hlo-textbox-pair"});
}

template <typename T>
using DecorationPrinter = std::string(const T*, const T*);

template <typename T>
std::string PrintNodePairContent(
    const T* left_node, const T* right_node,
    const absl::flat_hash_map<const HloInstruction*, Attributes>&
        span_attributes,
    GraphUrlGenerator* url_generator) {
  std::string url;
  if (url_generator != nullptr) {
    url = url_generator->GenerateWithSelectedNodes(left_node, right_node);
  }
  return absl::StrCat(
      PrintHloTextboxPair(left_node, right_node, span_attributes),
      url.empty()
          ? ""
          : PrintDiv(PrintLink("Model Explorer", url), {"model-explorer-url"}));
}

// Prints a pair of instructions or computations. If url_generator is not
// null, a link to the pair of instructions or computations in model
// explorer will be printed.
template <typename T>
std::string PrintNodePair(
    const T* left_node, const T* right_node,
    const absl::flat_hash_map<const HloInstruction*, Attributes>&
        span_attributes,
    GraphUrlGenerator* url_generator,
    std::optional<absl::FunctionRef<DecorationPrinter<T>>> decoration_printer =
        std::nullopt) {
  std::vector<std::string> nodes;
  if (left_node != nullptr) {
    nodes.push_back(std::string(left_node->name()));
  }
  if (right_node != nullptr) {
    nodes.push_back(std::string(right_node->name()));
  }
  std::string text = absl::StrJoin(nodes, " ↔ ");
  std::string decoration;
  if (decoration_printer.has_value()) {
    decoration =
        PrintSpan((*decoration_printer)(left_node, right_node), {"decoration"});
  }
  return PrintDetails(absl::StrCat(text, " ", decoration),
                      PrintNodePairContent<T>(left_node, right_node,
                                              span_attributes, url_generator));
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
      link =
          PrintNodePair<HloInstruction>(inst, /*right_node=*/nullptr,
                                        /*span_attributes=*/{}, url_generator);
    } else {
      link = PrintNodePair<HloInstruction>(
          /*left_node=*/nullptr, inst, /*span_attributes=*/{}, url_generator);
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
    pair_list.push_back(PrintNodePair(pair.first, pair.second,
                                      /*span_attributes=*/{}, url_generator,
                                      decoration_printer));
  }
  return PrintList(pair_list);
}

// Prints unmatched instructions grouped by opcode and print in a descending
// order of the number of instructions for each opcode.
std::string PrintUnmatchedInstructions(
    const absl::flat_hash_set<const HloInstruction*>& instructions,
    InstructionLocation location, bool name_only,
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
  return PrintInstructionPairsByOpcode(instructions, url_generator,
                                       decorated_printer);
}

// Prints unchanged instructions grouped by opcode and print in a
// descending order of the number of instructions for each opcode.
std::string PrintUnchangedInstructions(
    const absl::flat_hash_map<const HloInstruction*, const HloInstruction*>&
        instructions,
    GraphUrlGenerator* url_generator) {
  return PrintInstructionPairsByOpcode(instructions, url_generator);
}

/* Metrics diff */

// Prints unmatched instructions sorted by the metrics diff.
std::string PrintUnmatchedMetricsDiff(
    const absl::flat_hash_set<const HloInstruction*>& instructions,
    const OpMetricGetter& op_metric_getter, GraphUrlGenerator* url_generator,
    InstructionLocation location) {
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
    const HloInstruction *left_inst = nullptr, *right_inst = nullptr;
    if (location == InstructionLocation::kLeft) {
      left_inst = inst;
    } else {
      right_inst = inst;
    }
    metrics_diff_list.push_back(PrintNodePair<HloInstruction>(
        left_inst, right_inst, /*span_attributes=*/{}, url_generator,
        [&metrics_diff](const HloInstruction* inst,
                        const HloInstruction*) -> std::string {
          return absl::StrFormat("%.2f (us)", metrics_diff / 1e6);
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
        left_inst, right_inst, /*span_attributes=*/{}, url_generator,
        [&metrics_diff](const HloInstruction* left_inst,
                        const HloInstruction* right_inst) -> std::string {
          return absl::StrFormat("%+.2f (us)", metrics_diff / 1e6);
        }));
  }
  return PrintList(metrics_diff_list);
}

/* Diff pattern */

// Prints a text summary of the computation group. At least one of the left or
// right computations should be non-empty.
std::string PrintComputationGroupSummary(const ComputationGroup& group) {
  CHECK(!group.left_computations.empty() || !group.right_computations.empty());
  std::vector<std::string> left_computation_names(
      group.left_computations.size()),
      right_computation_names(group.right_computations.size());
  for (int i = 0; i < group.left_computations.size(); ++i) {
    left_computation_names[i] = group.left_computations[i]->name();
  }
  for (int i = 0; i < group.right_computations.size(); ++i) {
    right_computation_names[i] = group.right_computations[i]->name();
  }
  std::string summary;
  if (left_computation_names.empty()) {
    absl::StrAppend(
        &summary,
        absl::StrFormat("%s (%s)", absl::StrJoin(right_computation_names, ","),
                        PrintSpan("Right Unmatched Computation", {"green"})));
  } else if (right_computation_names.empty()) {
    absl::StrAppend(
        &summary,
        absl::StrFormat("%s (%s)", absl::StrJoin(left_computation_names, ","),
                        PrintSpan("Left Unmatched Computation", {"red"})));
  } else {
    absl::StrAppend(
        &summary,
        absl::StrFormat("%s ↔ %s", absl::StrJoin(left_computation_names, ","),
                        absl::StrJoin(right_computation_names, ",")));
  }
  return summary;
}

std::string PrintDiffMetrics(const DiffMetrics& diff_metrics) {
  std::vector<std::string> diff_metrics_list;
  if (diff_metrics.changed_instruction_count > 0) {
    diff_metrics_list.push_back(PrintSpan(
        absl::StrFormat("%d changed", diff_metrics.changed_instruction_count),
        {"yellow"}));
  }
  if (diff_metrics.left_unmatched_instruction_count > 0) {
    diff_metrics_list.push_back(PrintSpan(
        absl::StrFormat("%d left unmatched",
                        diff_metrics.left_unmatched_instruction_count),
        {"red"}));
  }
  if (diff_metrics.right_unmatched_instruction_count > 0) {
    diff_metrics_list.push_back(PrintSpan(
        absl::StrFormat("%d right unmatched",
                        diff_metrics.right_unmatched_instruction_count),
        {"green"}));
  }
  return absl::StrCat(absl::StrJoin(diff_metrics_list, ", "),
                      " instruction(s)");
}

// Prints the computation summary
std::string PrintComputationSummary(
    const ComputationDiffPattern& diff_pattern,
    const absl::flat_hash_map<const HloInstruction*, Attributes>&
        span_attributes,
    GraphUrlGenerator* url_generator) {
  const ComputationGroup& sample = diff_pattern.computation_groups[0];
  // We only support unmatched computation and one-to-one computation pairs.
  if (sample.left_computations.size() > 1 ||
      sample.right_computations.size() > 1) {
    return "";
  }
  std::vector<std::string> computation_pair_list(
      diff_pattern.computation_groups.size() - 1);
  for (int i = 1; i < diff_pattern.computation_groups.size(); ++i) {
    const ComputationGroup& computation_group =
        diff_pattern.computation_groups[i];
    const HloComputation* left_computation =
        computation_group.left_computations.empty()
            ? nullptr
            : computation_group.left_computations[0];
    const HloComputation* right_computation =
        computation_group.right_computations.empty()
            ? nullptr
            : computation_group.right_computations[0];
    computation_pair_list[i - 1] = PrintNodePair<HloComputation>(
        left_computation, right_computation, span_attributes, url_generator);
  }
  const HloComputation* left_computation =
      sample.left_computations.empty() ? nullptr : sample.left_computations[0];
  const HloComputation* right_computation = sample.right_computations.empty()
                                                ? nullptr
                                                : sample.right_computations[0];
  std::vector<std::string> contents;
  contents.push_back(PrintNodePairContent(left_computation, right_computation,
                                          span_attributes, url_generator));
  if (!computation_pair_list.empty()) {
    contents.push_back(
        PrintDetails(absl::StrFormat("%d other similar computations",
                                     computation_pair_list.size()),
                     PrintList(computation_pair_list)));
  }
  return PrintDetails(
      absl::StrFormat(
          "%s (%s) %s", PrintComputationGroupSummary(sample),
          PrintDiffMetrics(diff_pattern.diff_metrics),
          computation_pair_list.empty()
              ? ""
              : PrintSpan(
                    absl::StrFormat("(%d more computations has the same diff)",
                                    computation_pair_list.size()),
                    {"grey"})),
      PrintAttributesList(contents));
}

// Prints the summary of the repetitive diff patterns.
std::string PrintRepetitiveDiffPatterns(
    absl::Span<const ComputationDiffPattern> diff_patterns,
    const absl::flat_hash_map<const HloInstruction*, Attributes>&
        span_attributes,
    GraphUrlGenerator* url_generator) {
  // Sort the diff patterns by the number of computations in each group in
  // descending order.
  std::vector<ComputationDiffPattern> sorted_diff_patterns;
  for (const ComputationDiffPattern& diff_pattern : diff_patterns) {
    if (diff_pattern.computation_groups.empty()) {
      continue;
    }
    sorted_diff_patterns.push_back(diff_pattern);
  }
  std::sort(
      sorted_diff_patterns.begin(), sorted_diff_patterns.end(),
      [](const ComputationDiffPattern& a, const ComputationDiffPattern& b) {
        const uint64_t a_diff_size =
            a.diff_metrics.changed_instruction_count +
            a.diff_metrics.left_unmatched_instruction_count +
            a.diff_metrics.right_unmatched_instruction_count;
        const uint64_t b_diff_size =
            b.diff_metrics.changed_instruction_count +
            b.diff_metrics.left_unmatched_instruction_count +
            b.diff_metrics.right_unmatched_instruction_count;
        return a_diff_size > b_diff_size;
      });
  std::string computation_group_list;
  for (const ComputationDiffPattern& diff_pattern : sorted_diff_patterns) {
    absl::StrAppend(
        &computation_group_list,
        PrintComputationSummary(diff_pattern, span_attributes, url_generator));
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

  DiffResult filtered_diff_result =
      FilterDiffResultByOpcode(diff_result, ignored_opcodes);

  absl::flat_hash_map<const HloInstruction*, Attributes> span_attributes =
      GenerateSpanAttributes(diff_result);

  out << PrintCss() << PrintJavascript();

  // Print profile metrics diff
  if (left_op_metric_getter != nullptr && right_op_metric_getter != nullptr) {
    out << PrintSectionWithHeader(
        "XProf Op Metrics Diff by Instructions (Ordered by absolute execution "
        "time difference in descending order)",
        absl::StrCat(
            PrintDetails(
                "Left Module Unmatched Instructions",
                PrintUnmatchedMetricsDiff(
                    filtered_diff_result.left_module_unmatched_instructions,
                    *left_op_metric_getter, url_generator,
                    InstructionLocation::kLeft)),
            PrintDetails(
                "Right Module Unmatched Instructions",
                PrintUnmatchedMetricsDiff(
                    filtered_diff_result.right_module_unmatched_instructions,
                    *right_op_metric_getter, url_generator,
                    InstructionLocation::kRight)),
            PrintDetails("Changed Instructions",
                         PrintMatchedMetricsDiff(
                             filtered_diff_result.changed_instructions,
                             *left_op_metric_getter, *right_op_metric_getter,
                             url_generator)),
            PrintDetails("Unchanged Instructions",
                         PrintMatchedMetricsDiff(
                             filtered_diff_result.unchanged_instructions,
                             *left_op_metric_getter, *right_op_metric_getter,
                             url_generator))));
  }

  // Print repetitive computation groups
  out << PrintSectionWithHeader(
      "Diffs grouped by computation (Ordered by # of different instructions) " +
          PrintButton("toggleButton", "Hide Unchanged Instructions"),
      PrintRepetitiveDiffPatterns(diff_summary.computation_diff_patterns,
                                  span_attributes, url_generator));

  // Print full diff results
  out << PrintSectionWithHeader(
      "Full Diff Results",
      absl::StrCat(
          PrintDetails(
              absl::StrFormat("Left Module Unmatched Instructions (%d)",
                              filtered_diff_result
                                  .left_module_unmatched_instructions.size()),
              PrintUnmatchedInstructions(
                  filtered_diff_result.left_module_unmatched_instructions,
                  InstructionLocation::kLeft,
                  /*name_only=*/false, url_generator)),
          PrintDetails(
              absl::StrFormat("Right Module Unmatched Instructions (%d)",
                              filtered_diff_result
                                  .right_module_unmatched_instructions.size()),
              PrintUnmatchedInstructions(
                  filtered_diff_result.right_module_unmatched_instructions,
                  InstructionLocation::kRight,
                  /*name_only=*/false, url_generator)),
          PrintDetails(
              absl::StrFormat("Changed Instructions (%d)",
                              filtered_diff_result.changed_instructions.size()),
              PrintChangedInstructions(
                  filtered_diff_result.changed_instructions, url_generator))));

  out << PrintSystemMessagePlaceholder();
  out << PrintJavascriptForHoverEvent();
  out << PrintJavascriptForToggleButton();
}

}  // namespace hlo_diff
}  // namespace xla
