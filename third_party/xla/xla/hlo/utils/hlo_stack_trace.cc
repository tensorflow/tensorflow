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
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"
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

bool IsValidParamsString(const std::string& input) {
  static const LazyRE2 pattern = {
      R"(^[a-zA-Z_][a-zA-Z0-9_]*(\[.+\])+(?:\.[a-zA-Z0-9_]+)*$)"};
  return RE2::FullMatch(input, *pattern);
}

// SplitParamsString splits a string representing a parameter path into its
// components. The string can contain a variable name, followed by a series of
// bracketed keys and dot-separated attributes.
// Examples:
// - params['transformer/layer_1/mlp/linear/AqtEinsum_0'] -> {params,
// transformer, layer_1, mlp, linear, AqtEinsum_0}
// - foo['a/b']['c'].d -> {foo, a, b, c, d}
// If the string does not match the expected pattern, a vector containing the
// original string is returned.
std::vector<std::string> SplitParamsString(const std::string& input) {
  if (!IsValidParamsString(input)) {
    return {input};
  }

  std::vector<std::string> result;
  static const LazyRE2 pattern = {
      R"(([a-zA-Z_][a-zA-Z0-9_]*)|'([^']+)'|\.([a-zA-Z0-9_]+))"};
  absl::string_view input_view(input);
  std::string word1, word2, word3;
  while (RE2::FindAndConsume(&input_view, *pattern, &word1, &word2, &word3)) {
    if (!word1.empty()) {
      result.push_back(word1);
    } else if (!word2.empty()) {
      std::vector<std::string> parts = absl::StrSplit(word2, '/');
      result.insert(result.end(), parts.begin(), parts.end());
    } else if (!word3.empty()) {
      result.push_back(word3);
    }
  }
  return result;
}

// Returns a stack trace from module entry to `instr`.
// The trace includes computation names and optionally instruction metadata.
std::vector<std::string> BuildStackTrace(const HloInstruction* instr,
                                         const CallerMap& caller_map) {
  std::vector<std::string> trace;
  const HloInstruction* current_instr = instr;

  // Traverse upward through the call stack using caller_map.
  while (current_instr) {
    const HloComputation* comp = current_instr->parent();
    if (!comp->name().empty()) {
      trace.push_back(std::string(comp->name()));
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
  std::vector<std::string> leaf_parts = SplitParamsString(std::string(leaf));
  trace.insert(trace.end(), leaf_parts.begin(), leaf_parts.end());
  return trace;
}

// Internal tree structure to accumulate memory sizes by stack path.
struct StackNode {
  std::string name;
  int64_t total_size = 0;
  int64_t node_size = 0;
  absl::flat_hash_map<std::string, std::unique_ptr<StackNode>> children;
};

// Inserts a stack trace into the tree, accumulating buffer sizes at each node.
//
// Traverses the tree along the given trace, creating new nodes as needed. The
// buffer size is added to each node in the path, including the root.
void InsertTrace(StackNode* root, const std::vector<std::string>& trace,
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
    node->total_size += size;
  }

  // Track the size of the leaf node.
  node->node_size = size;
  // Track cumulative size at the root.
  root->total_size += size;
}

// Formats the string representation of a single stack node.
std::string FormatNodeString(const StackNode* node, int64_t total,
                             int64_t* remaining, const std::string& prefix,
                             bool is_last) {
  if (node->name.empty()) {
    return "";
  }

  std::string connector = prefix == "  " ? "  " : (is_last ? "└── " : "├── ");
  double pct = 100.0 * static_cast<double>(node->total_size) / total;

  *remaining -= node->node_size;
  return absl::StrFormat(
      "%s%s%s (%.1f%%, total: %lld bytes, current: %lld bytes, "
      "remaining: %lld bytes)\n",
      prefix, connector, node->name, pct, node->total_size, node->node_size,
      *remaining);
}

// Recursively prints the stack tree to the output string.
//
// Nodes are printed with indentation and connectors to show the tree structure.
// Children are sorted by size in descending order before printing.
void PrintStackTree(const StackNode* node, int64_t total, int64_t* remaining,
                    std::string* output, const std::string& prefix = "  ",
                    bool is_last = true) {
  *output += FormatNodeString(node, total, remaining, prefix, is_last);

  // Collect and sort children by descending size.
  std::vector<const StackNode*> children;
  children.reserve(node->children.size());
  for (const auto& [_, child] : node->children) {
    children.push_back(child.get());
  }

  std::sort(children.begin(), children.end(),
            [](const StackNode* a, const StackNode* b) {
              if (a->total_size != b->total_size) {
                return a->total_size > b->total_size;  // Descending size
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
    std::vector<std::string> trace =
        BuildStackTrace(value->instruction(), caller_map);
    if (!value->index().empty()) {
      trace[trace.size() - 1] += "{" + absl::StrJoin(value->index(), ",") + "}";
    }
    InsertTrace(root.get(), trace, bytes);
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
