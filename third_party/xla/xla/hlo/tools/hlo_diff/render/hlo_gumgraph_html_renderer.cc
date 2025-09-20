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
#include "xla/hlo/tools/hlo_diff/utils/text_diff.h"
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
    .section > .content ul {
      margin-top: 5px;
      margin-bottom: 5px;
      padding-left: 20px;
    }
    .section > .content li {
      margin-bottom: 3px;
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
    .hlo-textboxes {
      flex: 1;
      display: flex;
      min-width: 0;
      flex-direction: column;
      width: 100%;
      max-height: 1200px;
    }
    .hlo-textbox {
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 10px 0px;
      min-height: 0;
    }
    .hlo-textbox > .textbox {
      position: relative;
      padding: 10px;
      border: 1px solid #cccccc;
      border-radius: 5px;
      height: 100%;
      box-sizing: border-box;
      min-height: 0;
    }
    .hlo-textbox > .textbox > pre {
      width: 100%;
      margin: 0;
      padding: 2px;
      overflow: auto;
      white-space: pre-wrap;
      height: 100%;
      counter-reset: instruction;
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
    div.hlo-instruction {
      display: flex;
      width: 100%;
      align-items: flex-start;
      border: 2px solid transparent;
      box-sizing: border-box;
      counter-increment: instruction;
      position: relative;
      padding-left: 3.5em;
    }
    div.hlo-instruction::before {
      content: counter(instruction);
      position: absolute;
      left: 0;
      width: 3em;
      text-align: right;
      padding-right: 0.5em;
      color: #888;
      user-select: none;
    }
    div.hlo-instruction.expanded {
      max-width: unset;
    }
    span.hlo-instruction-text {
      flex: 1 1 auto;
      min-width: 0;
      white-space: pre;
      overflow: hidden;
      text-overflow: ellipsis;
      cursor: pointer;
    }
    div.hlo-instruction.expanded span.hlo-instruction-text {
      white-space: pre-wrap;
      overflow: visible;
    }
    button.hlo-expand-btn {
      flex-shrink: 0;
      background: #e8eaf6;
      color: #3f51b5;
      padding: 0 4px;
      border: 1px solid #c5cae9;
      box-shadow: none;
      font-weight: bold;
      margin-left: 4px;
      min-width: 25px;
      height: 1.3em;
      line-height: 1.1;
      visibility: hidden;
      cursor: pointer;
    }
    div.hlo-instruction.has-overflow.expanded button.hlo-expand-btn,
    div.hlo-instruction.has-overflow:not(.expanded):hover button.hlo-expand-btn {
      visibility: visible;
    }
    button.hlo-program-shape-btn {
      background: #e8eaf6;
      color: #3f51b5;
      padding: 0 4px;
      border: 1px solid #c5cae9;
      box-shadow: none;
      font-weight: bold;
      margin-left: 4px;
      height: 1.3em;
      line-height: 1.1;
      cursor: pointer;
    }
    div.hlo-instruction.bordered {
      border: 2px solid #4285F4;
    }

    .red-highlight {
      background-color: #fad2cf;
    }
    .green-highlight {
      background-color: #ceead6;
    }
    .yellow-highlight {
      background-color: #feefc3;
    }
    .darker-yellow-highlight {
      background-color: #FAD67F;
      /* Ensure empty or minimal content spans are visible as a thin line */
      display: inline-block;
      min-width: 2px; /* Thin darker yellow line */
      height: 1em; /* Approximately line height */
      vertical-align: middle;
      line-height: 1; /* Prevent extra space */
      overflow: hidden; /* Hide any overflow */
    }
    .temp-highlight {
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
  function CopyToClipboard(id) {
    const text = document.getElementById(id).textContent;
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

  const allInstructions = document.querySelectorAll('.hlo-instruction');
  allInstructions.forEach(instructionDiv => {
      instructionDiv.addEventListener('mouseover', handleInstructionMouseOver);
      instructionDiv.addEventListener('mouseout', handleInstructionMouseOut);
      instructionDiv.addEventListener('dblclick', handleInstructionDoubleClick);
      instructionDiv.addEventListener('click', handleInstructionClick);

      const button = instructionDiv.querySelector('.hlo-expand-btn');
      const textSpan = instructionDiv.querySelector('.hlo-instruction-text');
      if(textSpan.scrollWidth > textSpan.clientWidth) {
        instructionDiv.classList.add('has-overflow');
        button.addEventListener('click', (e) => {
          instructionDiv.classList.toggle('expanded');
          button.textContent = instructionDiv.classList.contains('expanded') ? '-' : '+';
          e.stopPropagation();
        });
      }
  });

  const allProgramShapeButtons = document.querySelectorAll('.hlo-program-shape-btn');
  allProgramShapeButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const shapeSpan = btn.nextElementSibling;
      if (shapeSpan && shapeSpan.classList.contains('hlo-program-shape')) {
        if (shapeSpan.style.display === 'none') {
          shapeSpan.style.display = 'inline';
          btn.textContent = 'hide program shape';
        } else {
          shapeSpan.style.display = 'none';
          btn.textContent = 'show program shape';
        }
      }
      e.stopPropagation();
    });
  });

  function getRelatedDivs(diffId, mappedId) {
    return document.querySelectorAll(`div[data-diffid="${diffId}"][data-diffid-mapped="${mappedId}"], div[data-diffid="${mappedId}"][data-diffid-mapped="${diffId}"]`);
  }

  function handleInstructionMouseOver(event) {
      const instructionDiv = event.currentTarget;
      const diffId = instructionDiv.getAttribute('data-diffid');
      const mappedId = instructionDiv.getAttribute('data-diffid-mapped');
      if (!diffId || !mappedId) {
          return;
      }
      const relatedDivs = getRelatedDivs(diffId, mappedId);
      relatedDivs.forEach(relatedDiv => {
          relatedDiv.classList.add('bordered');
      });
  }

  function handleInstructionMouseOut(event) {
      const instructionDiv = event.currentTarget;
      const diffId = instructionDiv.getAttribute('data-diffid');
      const mappedId = instructionDiv.getAttribute('data-diffid-mapped');
      if (!diffId || !mappedId) {
          return;
      }
      const relatedDivs = getRelatedDivs(diffId, mappedId);
      relatedDivs.forEach(relatedDiv => {
          relatedDiv.classList.remove('bordered');
      });
  }

  function handleInstructionClick(event) {
      const instructionDiv = event.currentTarget;
      const diffId = instructionDiv.getAttribute('data-diffid');
      const mappedId = instructionDiv.getAttribute('data-diffid-mapped');
      if (!diffId || !mappedId) {
          return;
      }

      const clickedPre = instructionDiv.closest('pre');
      if (!clickedPre) return;

      const selfTextboxes = instructionDiv.closest('.hlo-textboxes');
      if (!selfTextboxes) return;
      const siblingTextboxes = selfTextboxes.nextElementSibling || selfTextboxes.previousElementSibling;
      if (!siblingTextboxes || !siblingTextboxes.classList.contains('hlo-textboxes')) return;

      const targetDiv = siblingTextboxes.querySelector(`div[data-diffid="${mappedId}"][data-diffid-mapped="${diffId}"]`);

      if (targetDiv) {
          const siblingPre = targetDiv.closest('pre');
          if (!siblingPre) return;

          // Calculate the vertical offset of the clicked div from the top of its visible area.
          const clickedDivViewportOffset = (instructionDiv.offsetTop - clickedPre.offsetTop) - clickedPre.scrollTop;

          // Calculate the percentage of this offset within the clickedPre's visible height.
          const percentage = clickedPre.clientHeight > 0 ?
              clickedDivViewportOffset / clickedPre.clientHeight : 0;

          // Calculate the desired offset for the targetDiv within the siblingPre's visible area.
          const desiredSiblingViewportOffset = percentage * siblingPre.clientHeight;

          // Calculate the new scrollTop for siblingPre to achieve this alignment.
          const newScrollTop = (targetDiv.offsetTop - siblingPre.offsetTop) - desiredSiblingViewportOffset;

          // Scroll the siblingPre element smoothly.
          siblingPre.scrollTo({
              top: Math.max(0, newScrollTop),
              behavior: 'smooth'
          });

          // Temporarily highlight the target div
          targetDiv.classList.add('temp-highlight');
          setTimeout(() => {
              targetDiv.classList.remove('temp-highlight');
          }, 2000); // Remove highlight after 2 seconds
      } else {
          ShowSystemMessage("Corresponding instruction is in another computation, double click to jump to it.");
      }
  }

  function handleInstructionDoubleClick(event) {
      const instructionDiv = event.currentTarget;
      const diffId = instructionDiv.getAttribute('data-diffid');
      const mappedId = instructionDiv.getAttribute('data-diffid-mapped');
      if (!diffId || !mappedId) {
          return;
      }

      const clickedEl = event.target;
      const clickedPre = clickedEl.closest('pre');
      if (!clickedPre) return;

      const selfTextboxes = clickedEl.closest('.hlo-textboxes');
      if (!selfTextboxes) return;
      const siblingTextboxes = selfTextboxes.nextElementSibling || selfTextboxes.previousElementSibling;
      if (!siblingTextboxes || !siblingTextboxes.classList.contains('hlo-textboxes')) return;

      const targetDivInSibling = siblingTextboxes.querySelector(`div[data-diffid="${mappedId}"][data-diffid-mapped="${diffId}"]`);

      if (!targetDivInSibling) {
          // Case 2: Corresponding div NOT found in sibling textboxes in same pair.
          // Search for the div with the same diffId in other hlo-textbox-pairs.
          const allMatchingDivs = document.querySelectorAll(`div[data-diffid="${mappedId}"][data-diffid-mapped="${diffId}"]`);
          let foundDivInOtherPre = null;
          allMatchingDivs.forEach(div => {
              if (div !== instructionDiv) {
                  foundDivInOtherPre = div;
              }
          });

          if (foundDivInOtherPre) {
              const foundPre = foundDivInOtherPre.closest('pre');
              if (foundPre) {
                  // Find ancestor detail and open it.
                  let parentDetails = foundDivInOtherPre.closest('details');
                  while (parentDetails) {
                      parentDetails.open = true;
                      parentDetails = parentDetails.parentElement ? parentDetails.parentElement.closest('details') : null;
                  }

                  // Scroll the foundPre to make the foundDivInOtherPre visible.
                  foundDivInOtherPre.scrollIntoView({ behavior: 'smooth', block: 'center' });

                  // Temporarily highlight the found div
                  foundDivInOtherPre.classList.add('temp-highlight');
                  setTimeout(() => {
                      foundDivInOtherPre.classList.remove('temp-highlight');
                  }, 3000);
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

// Prints overview section.
std::string PrintOverviewSection(const DiffResult& diff_result) {
  std::string content = absl::StrFormat(
      R"html(
        <p>This report highlights the differences between two HLO modules.</p>
        <p>
          <b>Diff Statistics:</b>
          <span>%d unchanged</span>,
          <span class="yellow">%d changed</span>,
          <span class="red">%d left unmatched</span>,
          <span class="green">%d right unmatched</span> instruction(s).
        </p>
      )html",
      diff_result.unchanged_instructions.size(),
      diff_result.changed_instructions.size(),
      diff_result.left_module_unmatched_instructions.size(),
      diff_result.right_module_unmatched_instructions.size());
  return PrintSectionWithHeader("Overview", content);
}

// Prints the "How to use this report" section.
std::string PrintHowToUseSection() {
  std::string content = R"html(
        <p>
          <b>Highlights:</b>
          <ul>
            <li><span class="red-highlight">&nbsp;Red&nbsp;</span>: Instruction only present in the left module.</li>
            <li><span class="green-highlight">&nbsp;Green&nbsp;</span>: Instruction only present in the right module.</li>
            <li><span class="yellow-highlight">&nbsp;Yellow&nbsp;</span>: Instruction present in both modules but with differences. Character-level differences are highlighted in <span class="darker-yellow-highlight">&nbsp;darker yellow&nbsp;</span> (this is skipped for instructions with >10k characters due to performance concern).</li>
            <li>Instructions without highlights are identical in both modules.</li>
          </ul>
        </p>
        <p>
          <b>Interactions:</b>
          <ul>
            <li><b>Hover</b> over an instruction to highlight its counterpart in the other module.</li>
            <li><b>Click</b> on an instruction to scroll its counterpart into view within the same computation diff.</li>
            <li><b>Double-click</b> on an instruction to jump to its counterpart if it resides in a different computation diff.</li>
            <li>Click the <b>[+]</b> button on an instruction to expand/collapse overflowing text.</li>
          </ul>
        </p>
        <p>
          <b>Sections:</b>
          <ul>
            <li><b>XProf Op Metrics Diff</b>: Shows instructions with the largest execution time differences based on XProf data (if available).</li>
            <li><b>Diffs grouped by computation</b>: Groups computations with similar diff patterns to help identify repetitive changes.</li>
            <li><b>Full Diff Results</b>: A flat list of all instructions that are unmatched or have changed.</li>
          </ul>
        </p>
      )html";
  return PrintSectionWithHeader("How to use this report", content);
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
                                   absl::string_view pre_id) {
  return absl::StrFormat(
      R"html(<span class="click-to-copy" onclick="CopyToClipboard('%s')">%s</span>)html",
      pre_id, PrintTooltip(text, "Click to copy", TooltipPosition::kLeft));
}

// Print textbox with click to copy button.
std::string PrintTextbox(absl::string_view title, absl::string_view content,
                         absl::string_view id = "") {
  return absl::StrCat(
      PrintDiv(title, {"title"}),
      PrintDiv(absl::StrCat(absl::StrFormat(R"html(<pre id="%s">%s</pre>)html",
                                            id, content),
                            PrintClickToCopyButton("📋", id)),
               {"textbox"}));
}

/*** Summary logic ***/

// The attributes of an instruction that will be applied to the corresponding
// span element in the HTML output.
struct Attributes {
  // The class name of the highlight. Empty if no highlight.
  std::string highlight;
  // The diffid of instruction.
  std::string diffid;
  // The diffid of mapped instruction. Empty if no mapping to another span.
  std::string mapped_diffid;
  // The mapped instruction of the span. Null if no mapping to another span.
  const HloInstruction* mapped_instruction;
};

// Generate span attributes for all instructions given diff result.
absl::flat_hash_map<const HloInstruction*, Attributes> GenerateSpanAttributes(
    const DiffResult& diff_result) {
  absl::flat_hash_map<const HloInstruction*, Attributes> span_attributes;
  for (auto& instruction : diff_result.left_module_unmatched_instructions) {
    span_attributes[instruction] =
        Attributes{.highlight = "red-highlight",
                   .diffid = absl::StrCat("left:", instruction->name()),
                   .mapped_diffid = "",
                   .mapped_instruction = nullptr};
  }
  for (auto& instruction : diff_result.right_module_unmatched_instructions) {
    span_attributes[instruction] =
        Attributes{.highlight = "green-highlight",
                   .diffid = absl::StrCat("right:", instruction->name()),
                   .mapped_diffid = "",
                   .mapped_instruction = nullptr};
  }
  for (const auto& [l_instruction, r_instruction] :
       diff_result.changed_instructions) {
    span_attributes[l_instruction] = Attributes{
        .highlight = "yellow-highlight",
        .diffid = absl::StrCat("left:", l_instruction->name()),
        .mapped_diffid = absl::StrCat("right:", r_instruction->name()),
        .mapped_instruction = r_instruction};
    span_attributes[r_instruction] = Attributes{
        .highlight = "yellow-highlight",
        .diffid = absl::StrCat("right:", r_instruction->name()),
        .mapped_diffid = absl::StrCat("left:", l_instruction->name()),
        .mapped_instruction = l_instruction};
  }
  for (const auto& [l_instruction, r_instruction] :
       diff_result.unchanged_instructions) {
    span_attributes[l_instruction] = Attributes{
        .highlight = "",
        .diffid = absl::StrCat("left:", l_instruction->name()),
        .mapped_diffid = absl::StrCat("right:", r_instruction->name()),
        .mapped_instruction = r_instruction};
    span_attributes[r_instruction] = Attributes{
        .highlight = "",
        .diffid = absl::StrCat("right:", r_instruction->name()),
        .mapped_diffid = absl::StrCat("left:", l_instruction->name()),
        .mapped_instruction = l_instruction};
  }
  return span_attributes;
};

// Generates HTML for a single HloComputation with diff highlights.
std::string PrintHloComputationToHtml(
    const HloComputation* comp, DiffSide side,
    const absl::flat_hash_map<const HloInstruction*, Attributes>&
        span_attributes) {
  if (comp == nullptr) {
    return "";
  }
  StringPrinter printer;

  // Mimic HloComputation::Print structure with default options.
  printer.Append("<b>%");
  printer.Append(comp->name());
  printer.Append(" ");
  printer.Append(
      "<button class='hlo-program-shape-btn'>show program shape</button>");
  printer.Append("<span class='hlo-program-shape' style='display: none;'>");
  ShapeUtil::PrintHumanString(&printer,
                              comp->ComputeProgramShape(/*include_ids=*/true));
  printer.Append("</span>");
  printer.Append(" ");
  printer.Append("{</b>\n");

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
      std::string diffid_attrs;
      if (it != span_attributes.end()) {
        absl::StrAppend(&diffid_attrs, " data-diffid=\"",
                        EscapeStringForHtmlAttribute(it->second.diffid), "\"");
        if (!it->second.mapped_diffid.empty()) {
          absl::StrAppend(
              &diffid_attrs, " data-diffid-mapped=\"",
              EscapeStringForHtmlAttribute(it->second.mapped_diffid), "\"");
        }
      }
      printer.Append(absl::StrCat("<div class=\"hlo-instruction ",
                                  highlight_class, "\"", diffid_attrs, " >",
                                  "<span class='hlo-instruction-text'>"));
      printer.Append("  ");  // Instruction indentation (2 spaces)
      if (instruction == comp->root_instruction()) {
        printer.Append("ROOT ");
      }
      if (it == span_attributes.end() ||
          it->second.mapped_instruction == nullptr ||
          it->second.highlight.empty()) {
        instruction->PrintWithCanonicalNameMap(
            &printer, instruction_print_options, &name_map);
      } else {
        // Instruction is part of a changed pair. Show character-level diff.
        const HloInstruction* mapped_instruction =
            it->second.mapped_instruction;
        bool is_left_node = side == DiffSide::kLeft;

        StringPrinter current_printer, mapped_printer;
        instruction->PrintWithCanonicalNameMap(
            &current_printer, instruction_print_options, &name_map);
        mapped_instruction->PrintWithCanonicalNameMap(
            &mapped_printer, instruction_print_options, &name_map);

        std::string current_str = std::move(current_printer).ToString();
        std::string mapped_str = std::move(mapped_printer).ToString();

        // Skip text diff if the two strings are longer than 10000 characters.
        if (current_str.size() > 10000 || mapped_str.size() > 10000) {
          printer.Append(current_str);
        } else {
          std::vector<TextDiffChunk> diff_chunks;
          if (is_left_node) {
            // Left side: diff current (left) vs mapped (right).
            diff_chunks = ComputeTextDiff(current_str, mapped_str);
          } else {
            // Right side: diff mapped (left) vs current (right).
            diff_chunks = ComputeTextDiff(mapped_str, current_str);
          }

          for (const auto& chunk : diff_chunks) {
            if (chunk.type == TextDiffType::kUnchanged) {
              printer.Append(EscapeStringForHtmlAttribute(chunk.text));
            } else if (is_left_node && chunk.type == TextDiffType::kRemoved) {
              // On the left side, highlight text REMOVED from left compared to
              // right.
              printer.Append("<span class=\"darker-yellow-highlight\">");
              printer.Append(EscapeStringForHtmlAttribute(chunk.text));
              printer.Append("</span>");
            } else if (!is_left_node && chunk.type == TextDiffType::kAdded) {
              // On the right side, highlight text ADDED to right compared to
              // left.
              printer.Append("<span class=\"darker-yellow-highlight\">");
              printer.Append(EscapeStringForHtmlAttribute(chunk.text));
              printer.Append("</span>");
            } else {
              printer.Append("<span class=\"darker-yellow-highlight\">");
              printer.Append("</span>");
            }
          }
        }
      }
      printer.Append("</span><button class='hlo-expand-btn'>+</button></div>");
    }
  }

  printer.Append("<b>}</b>");  // Closing brace for computation.

  // Default print_ids is true, so execution thread is printed if not main.
  if (!comp->IsMainThread()) {
    printer.Append(", execution_thread=\"");
    printer.Append(comp->execution_thread());
    printer.Append("\"");
  }

  return std::move(printer).ToString();
}

// Prints a single HLO instruction in a text box.
template <typename T>
std::string PrintHloTextbox(
    const T* node, DiffSide side,
    const absl::flat_hash_map<const HloInstruction*, Attributes>&
        span_attributes) {
  std::string title = "None", text;
  if (node != nullptr) {
    title = node->name();
    if constexpr (std::is_same_v<T, HloComputation>) {
      text = PrintHloComputationToHtml(node, side, span_attributes);
    } else {
      text = node->ToString();
    }
  }
  uint64_t fingerprint = tsl::Fingerprint64(text);
  return PrintDiv(
      PrintTextbox(title, text, absl::StrFormat("%016x", fingerprint)),
      {"hlo-textbox"});
}

// Prints a pair of instructions or computations in a text box.
template <typename T>
std::string PrintHloTextboxPair(
    absl::Span<const T* const> left_nodes,
    absl::Span<const T* const> right_nodes,
    const absl::flat_hash_map<const HloInstruction*, Attributes>&
        span_attributes) {
  std::string left_textbox, right_textbox;
  for (const T* node : left_nodes) {
    absl::StrAppend(&left_textbox,
                    PrintHloTextbox(node, DiffSide::kLeft, span_attributes));
  }
  for (const T* node : right_nodes) {
    absl::StrAppend(&right_textbox,
                    PrintHloTextbox(node, DiffSide::kRight, span_attributes));
  }
  return PrintDiv(absl::StrCat(PrintDiv(left_textbox, {"hlo-textboxes"}),
                               PrintDiv(right_textbox, {"hlo-textboxes"})),
                  {"hlo-textbox-pair"});
}

template <typename T>
using DecorationPrinter = std::string(absl::Span<const T* const>,
                                      absl::Span<const T* const>);

template <typename T>
std::string PrintNodePairContent(
    absl::Span<const T* const> left_nodes,
    absl::Span<const T* const> right_nodes,
    const absl::flat_hash_map<const HloInstruction*, Attributes>&
        span_attributes,
    GraphUrlGenerator* url_generator) {
  std::string url;
  const T* left_node = left_nodes.empty() ? nullptr : left_nodes[0];
  const T* right_node = right_nodes.empty() ? nullptr : right_nodes[0];
  if (url_generator != nullptr) {
    url = url_generator->GenerateWithSelectedNodes(left_node, right_node);
  }
  return absl::StrCat(
      PrintHloTextboxPair(left_nodes, right_nodes, span_attributes),
      url.empty()
          ? ""
          : PrintDiv(PrintLink("Model Explorer", url), {"model-explorer-url"}));
}

// Prints a pair of instructions or computations. If url_generator is not
// null, a link to the pair of instructions or computations in model
// explorer will be printed.
template <typename T>
std::string PrintNodePair(
    absl::Span<const T* const> left_nodes,
    absl::Span<const T* const> right_nodes,
    const absl::flat_hash_map<const HloInstruction*, Attributes>&
        span_attributes,
    GraphUrlGenerator* url_generator,
    std::optional<absl::FunctionRef<DecorationPrinter<T>>> decoration_printer =
        std::nullopt) {
  std::vector<std::string> nodes;
  if (!left_nodes.empty()) {
    nodes.push_back(
        absl::StrJoin(left_nodes, ", ", [](std::string* out, const T* node) {
          absl::StrAppend(out, node->name());
        }));
  }
  if (!right_nodes.empty()) {
    nodes.push_back(
        absl::StrJoin(right_nodes, ", ", [](std::string* out, const T* node) {
          absl::StrAppend(out, node->name());
        }));
  }
  std::string text = absl::StrJoin(nodes, " ↔ ");
  std::string decoration;
  if (decoration_printer.has_value()) {
    decoration = PrintSpan((*decoration_printer)(left_nodes, right_nodes),
                           {"decoration"});
  }
  return PrintDetails(absl::StrCat(text, " ", decoration),
                      PrintNodePairContent<T>(left_nodes, right_nodes,
                                              span_attributes, url_generator));
}

template <typename T>
std::string PrintNodePair(
    const T* left_node, const T* right_node,
    const absl::flat_hash_map<const HloInstruction*, Attributes>&
        span_attributes,
    GraphUrlGenerator* url_generator,
    std::optional<absl::FunctionRef<DecorationPrinter<T>>> decoration_printer =
        std::nullopt) {
  std::vector<const T*> left_nodes;
  if (left_node != nullptr) {
    left_nodes.push_back(left_node);
  }
  std::vector<const T*> right_nodes;
  if (right_node != nullptr) {
    right_nodes.push_back(right_node);
  }
  return PrintNodePair<T>(left_nodes, right_nodes, span_attributes,
                          url_generator, decoration_printer);
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
      [](absl::Span<const HloInstruction* const> left_insts,
         absl::Span<const HloInstruction* const> right_insts) -> std::string {
    CHECK_EQ(left_insts.size(), 1);
    CHECK_EQ(right_insts.size(), 1);
    const HloInstruction* left_inst = left_insts[0];
    const HloInstruction* right_inst = right_insts[0];
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
        [&metrics_diff](absl::Span<const HloInstruction* const> left_insts,
                        absl::Span<const HloInstruction* const> right_insts)
            -> std::string {
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
        [&metrics_diff](absl::Span<const HloInstruction* const> left_insts,
                        absl::Span<const HloInstruction* const> right_insts)
            -> std::string {
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
  std::vector<std::string> computation_pair_list(
      diff_pattern.computation_groups.size() - 1);
  for (int i = 1; i < diff_pattern.computation_groups.size(); ++i) {
    const ComputationGroup& computation_group =
        diff_pattern.computation_groups[i];
    computation_pair_list[i - 1] = PrintNodePair<HloComputation>(
        computation_group.left_computations,
        computation_group.right_computations, span_attributes, url_generator);
  }
  std::vector<std::string> contents;
  contents.push_back(PrintNodePairContent<HloComputation>(
      sample.left_computations, sample.right_computations, span_attributes,
      url_generator));
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

  // Print overview section
  out << PrintOverviewSection(filtered_diff_result);

  // Print "How to use this report" section
  out << PrintHowToUseSection();

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
