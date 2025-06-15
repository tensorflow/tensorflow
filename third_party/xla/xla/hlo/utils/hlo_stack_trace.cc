/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/utils/hlo_stack_trace.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_value.h"

namespace xla {
namespace {

// A mapping from a computation to its caller instruction.
using CallerMap =
    absl::flat_hash_map<const HloComputation*, const HloInstruction*>;

// Builds a caller map for all computations in the module.
CallerMap BuildCallerMap(const HloModule* module) {
  CallerMap map;
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instr : computation->instructions()) {
      for (const HloComputation* called : instr->called_computations()) {
        map.emplace(called, instr);
      }
    }
  }
  return map;
}

// Returns a stack trace from module entry to `instr`.
// The trace includes computation names and optionally instruction metadata.
std::vector<absl::string_view> BuildStackTrace(const HloInstruction* instr,
                                               const CallerMap& caller_map) {
  std::vector<absl::string_view> trace;
  const HloInstruction* current_instr = instr;

  // Traverse upward through the call stack using caller_map.
  while (current_instr) {
    const HloComputation* comp = current_instr->parent();
    if (!comp->name().empty()) {
      trace.push_back(comp->name());
    }

    // Move up one level in the call stack.
    auto it = caller_map.find(comp);
    current_instr = (it != caller_map.end()) ? it->second : nullptr;
  }
  std::reverse(trace.begin(), trace.end());

  // Add the instruction’s name or metadata label as the leaf node.
  absl::string_view leaf = !instr->metadata().op_name().empty()
                               ? instr->metadata().op_name()
                               : instr->name();
  trace.push_back(leaf);
  return trace;
}

// Internal tree structure to accumulate memory sizes by stack path.
struct StackNode {
  std::string name;
  int64_t size = 0;
  absl::flat_hash_map<std::string, std::unique_ptr<StackNode>> children;
};

void InsertTrace(StackNode* root, absl::Span<absl::string_view> trace,
                 int64_t size) {
  StackNode* node = root;

  for (const auto& part : trace) {
    // Create a new child node if it doesn't exist.
    auto& child_node = node->children[part];
    if (!child_node) {
      child_node = std::make_unique<StackNode>();
      child_node->name = part;
    }

    // Move to the child and accumulate size.
    node = child_node.get();
    node->size += size;
  }

  // Track cumulative size at the root.
  root->size += size;
}

void PrintStackTree(const StackNode* node, int64_t total, int64_t* remaining,
                    std::string* output, const std::string& prefix = "  ",
                    bool is_last = true) {
  const bool is_leaf = node->children.empty();
  if (!node->name.empty()) {
    std::string connector = prefix == "  " ? "  " : (is_last ? "└── " : "├── ");
    double pct = 100.0 * static_cast<double>(node->size) / total;
    if (is_leaf) {
      *remaining -= node->size;
      absl::StrAppendFormat(
          output, "%s%s%s (%.1f%%, %lld bytes, remaining: %lld bytes)\n",
          prefix, connector, node->name, pct, node->size, *remaining);
    } else {
      absl::StrAppendFormat(output, "%s%s%s (%.1f%%, %lld bytes)\n", prefix,
                            connector, node->name, pct, node->size);
    }
  }

  // Collect and sort children by descending size.
  std::vector<const StackNode*> children;
  children.reserve(node->children.size());
  for (const auto& [_, child] : node->children) {
    children.push_back(child.get());
  }

  std::sort(children.begin(), children.end(),
            [](const StackNode* a, const StackNode* b) {
              if (a->size != b->size) {
                return a->size > b->size;  // Descending size
              }
              return a->name < b->name;  // Ascending name tie-break
            });

  // Recurse into children, adjusting prefix for visual alignment.
  for (size_t i = 0; i < children.size(); ++i) {
    const StackNode* child = children[i];
    bool last = (i == children.size() - 1);

    // Use vertical line for intermediate siblings.
    std::string child_prefix =
        prefix + (node->name.empty() ? "" : (is_last ? "    " : "│   "));

    PrintStackTree(child, total, remaining, output, child_prefix, last);
  }
}
}  // namespace

std::string FormatStackTraceBreakdown(
    const std::vector<std::pair<int64_t, const HloValue*>>& buffers,
    const HloModule* module) {
  // Build a map from computation to its calling instruction for stack tracing.
  auto caller_map = BuildCallerMap(module);

  // Root of the stack tree that accumulates hierarchical buffer usage.
  auto root = std::make_unique<StackNode>();

  int64_t total_bytes = 0;

  for (const auto& [bytes, value] : buffers) {
    // Add the stack path into the tree and accumulate size.
    std::vector<absl::string_view> trace =
        BuildStackTrace(value->instruction(), caller_map);
    InsertTrace(root.get(), absl::MakeSpan(trace), bytes);
    total_bytes += bytes;
  }

  std::string output;
  // Print the aggregated stack tree with visual structure.
  absl::StrAppendFormat(&output,
                        "  Stack trace breakdown for peak usage: %lld bytes\n",
                        total_bytes);
  int64_t remaining_bytes = total_bytes;
  PrintStackTree(root.get(), total_bytes, &remaining_bytes, &output);
  return output;
}

}  // namespace xla
