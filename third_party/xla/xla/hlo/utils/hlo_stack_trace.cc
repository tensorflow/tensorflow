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
#include <deque>
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

// Mapping from a computation (a function) to a list of all the 'call'
// instructions that invoke it.
using CallerMap = absl::flat_hash_map<const HloComputation*,
                                      std::vector<const HloInstruction*>>;

// Builds a caller map for all computations in the module. This map is
// essential for traversing the call graph upwards from an instruction to find
// all possible execution paths (stack traces).
CallerMap BuildCallerMap(const HloModule* module) {
  CallerMap map;
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instr : computation->instructions()) {
      for (const HloComputation* called : instr->called_computations()) {
        map[called].push_back(instr);
      }
    }
  }
  return map;
}

// Splits a string representing a parameter path into its components and
// validates its format.
//
// A valid path must consist of:
// 1. A variable name (e.g., `params`).
// 2. One or more single-quoted, bracketed keys (e.g., `['layer/fc']`).
// 3. Optional dot-separated attributes (e.g., `.qvalue`).
//
// The components must be contiguous with no invalid characters in between. If
// the string is not a valid parameter path, a vector containing the original,
// unmodified string is returned.
//
// Examples:
// - "params['layer/fc']" -> {"params", "layer", "fc"}
// - "var['a']['b'].attr" -> {"var", "a", "b", "attr"}
// - "invalid-string"     -> {"invalid-string"}
// - "params ['gap']"     -> {"params ['gap']"} (invalid due to the space).
std::vector<std::string> SplitParamsString(const std::string& input) {
  absl::string_view input_view(input);
  std::vector<std::string> result;

  // Consume the initial variable name.
  static const LazyRE2 kVarPattern = {R"(([a-zA-Z_]\w+))"};
  std::string var_name;
  if (!RE2::Consume(&input_view, *kVarPattern, &var_name)) {
    return {input};
  }
  result.push_back(var_name);

  // Consume subsequent parts. Using separate regexes for keys and attributes
  // is more robust than using a single regex with an alternation.
  static const LazyRE2 kKeyPattern = {R"(\['([^']+)'\])"};
  static const LazyRE2 kAttrPattern = {R"(\.(\w+))"};
  bool has_keys = false;

  while (true) {
    std::string key;
    if (RE2::Consume(&input_view, *kKeyPattern, &key)) {
      has_keys = true;
      std::vector<std::string> parts = absl::StrSplit(key, '/');
      result.insert(result.end(), parts.begin(), parts.end());
      continue;
    }

    std::string attr;
    if (RE2::Consume(&input_view, *kAttrPattern, &attr)) {
      result.push_back(attr);
      continue;
    }

    break;
  }

  // The entire string must have been consumed, and at least one key must have
  // been present.
  if (!input_view.empty() || !has_keys) {
    return {input};
  }

  return result;
}

// Builds a hierarchical tree of memory usage from stack traces.
class StackTreeBuilder {
 public:
  struct StackNode {
    std::string name;
    int64_t total_size = 0;
    int64_t node_size = 0;
    absl::flat_hash_map<std::string, std::unique_ptr<StackNode>> children;
  };

  StackTreeBuilder() : root_(std::make_unique<StackNode>()) {}

  // Inserts a stack trace path into the tree, building the node structure and
  // accumulating the buffer's size at the final leaf node.
  void InsertTrace(const std::vector<std::string>& trace, int64_t size) {
    StackNode* node = root_.get();
    for (const std::string& part : trace) {
      std::unique_ptr<StackNode>& child_node = node->children[part];
      if (!child_node) {
        child_node = std::make_unique<StackNode>();
        child_node->name = part;
      }
      node = child_node.get();
    }
    node->node_size += size;
  }

  // Finalizes the tree by calculating the total aggregated size for each node.
  // Transfers ownership of the tree to the caller.
  std::unique_ptr<StackNode> Build() {
    if (!root_) {
      return nullptr;
    }
    CalculateTotalSizes(root_.get());
    return std::move(root_);
  }

 private:
  // Recursively calculates the total size for each node.
  void CalculateTotalSizes(StackNode* node) {
    int64_t cumulative_size = node->node_size;
    for (auto& [_, child] : node->children) {
      CalculateTotalSizes(child.get());
      cumulative_size += child->total_size;
    }
    node->total_size = cumulative_size;
  }

  std::unique_ptr<StackNode> root_;
};

// Recursively builds all possible stack traces for an instruction.
//
// Since a computation can have multiple callers, this function branches and
// creates a separate stack trace for each unique call path.
void BuildStackTracesRecursive(
    const HloInstruction* instr, const CallerMap& caller_map,
    std::deque<std::string> current_path,
    std::vector<std::vector<std::string>>& all_traces) {
  // Prepend the leaf node (the current instruction's name or metadata).
  // This builds the stack trace from the inside out.
  absl::string_view leaf = !instr->metadata().op_name().empty()
                               ? instr->metadata().op_name()
                               : instr->name();
  std::vector<std::string> leaf_parts = SplitParamsString(std::string(leaf));
  // Prepend parts in reverse order to maintain the correct sequence.
  for (auto it = leaf_parts.rbegin(); it != leaf_parts.rend(); ++it) {
    current_path.push_front(*it);
  }

  // Prepend the name of the computation that contains the instruction.
  const HloComputation* parent_comp = instr->parent();
  current_path.push_front(std::string(parent_comp->name()));

  // Find the callers of the current computation.
  auto it = caller_map.find(parent_comp);
  if (it == caller_map.end()) {
    all_traces.emplace_back(current_path.begin(), current_path.end());
    return;
  }

  for (const HloInstruction* caller : it->second) {
    BuildStackTracesRecursive(caller, caller_map, current_path, all_traces);
  }
}

std::string FormatBytesWithSpaces(int64_t num) {
  std::string s = absl::StrFormat("%lld", num);
  // Insert a space every three digits from the right.
  for (int i = s.length() - 3; i > 0; i -= 3) {
    s.insert(i, " ");
  }
  return s;
}

std::string FormatNodeString(const StackTreeBuilder::StackNode* node,
                             int64_t total, int64_t* remaining,
                             const std::string& prefix, bool is_last) {
  if (node->name.empty()) {
    return "";
  }

  std::string connector = prefix == "  " ? "  " : (is_last ? "└── " : "├── ");

  double pct =
      total > 0 ? 100.0 * static_cast<double>(node->total_size) / total : 0.0;

  *remaining -= node->node_size;

  std::string total_str = FormatBytesWithSpaces(node->total_size);
  std::string current_str = FormatBytesWithSpaces(node->node_size);
  std::string remaining_str = FormatBytesWithSpaces(*remaining);

  return absl::StrFormat(
      "%s%s%s (%.1f%%, total: %s bytes, current: %s bytes, "
      "remaining: %s bytes)\n",
      prefix, connector, node->name, pct, total_str, current_str,
      remaining_str);
}

// Recursively prints the stack tree to the output string.
void PrintStackTree(const StackTreeBuilder::StackNode* node, int64_t total,
                    int64_t* remaining, std::string* output,
                    const std::string& prefix = "  ", bool is_last = true) {
  *output += FormatNodeString(node, total, remaining, prefix, is_last);

  // Collect and sort children to ensure the most significant memory
  // contributors are listed first.
  std::vector<const StackTreeBuilder::StackNode*> children;
  children.reserve(node->children.size());
  for (const auto& [_, child] : node->children) {
    children.push_back(child.get());
  }
  std::sort(children.begin(), children.end(),
            [](const StackTreeBuilder::StackNode* a,
               const StackTreeBuilder::StackNode* b) {
              if (a->total_size != b->total_size) {
                return a->total_size > b->total_size;
              }
              return a->name < b->name;
            });

  // Recurse into the sorted children to print the next level of the tree.
  for (size_t i = 0; i < children.size(); ++i) {
    const StackTreeBuilder::StackNode* child = children[i];
    bool last = (i == children.size() - 1);

    std::string child_prefix =
        prefix + (node->name.empty() ? "" : (is_last ? "    " : "│   "));
    PrintStackTree(child, total, remaining, output, child_prefix, last);
  }
}
}  // namespace

std::string FormatStackTraceBreakdown(
    const std::vector<std::pair<int64_t, const HloValue*>>& buffers,
    const HloModule* module) {
  auto caller_map = BuildCallerMap(module);
  StackTreeBuilder tree_builder;

  int64_t total_bytes = 0;
  for (const auto& [bytes, value] : buffers) {
    total_bytes += bytes;
  }

  for (const auto& [bytes, value] : buffers) {
    std::vector<std::vector<std::string>> all_traces;
    BuildStackTracesRecursive(value->instruction(), caller_map, {}, all_traces);

    if (all_traces.empty()) {
      continue;
    }

    const int64_t kBaseSizePerTrace = bytes / all_traces.size();
    int64_t remainder = bytes % all_traces.size();

    for (size_t i = 0; i < all_traces.size(); ++i) {
      std::vector<std::string>& trace = all_traces[i];
      int64_t size_for_this_trace = kBaseSizePerTrace;
      if (remainder > 0) {
        size_for_this_trace++;
        remainder--;
      }

      if (!value->index().empty()) {
        trace.back() += "{" + absl::StrJoin(value->index(), ",") + "}";
      }
      tree_builder.InsertTrace(trace, size_for_this_trace);
    }
  }

  std::unique_ptr<StackTreeBuilder::StackNode> root = tree_builder.Build();

  std::string output;
  absl::StrAppendFormat(&output,
                        "  Stack trace breakdown for peak usage: %s bytes\n",
                        FormatBytesWithSpaces(total_bytes));
  int64_t remaining_bytes = total_bytes;
  PrintStackTree(root.get(), total_bytes, &remaining_bytes, &output);
  return output;
}

}  // namespace xla
