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

#include "xla/hlo/tools/hlo_diff/graph/utils/cycle_detector.h"

#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"

namespace xla {
namespace hlo_diff {

namespace {

// Helper function to log a detected cycle.
void LogCycle(const std::vector<const HloInstructionNode*>& cycle_path) {
  std::string cycle_str;
  if (cycle_path.empty()) {
    LOG(WARNING) << "Unexpected empty cycle path detected.";
    return;
  }

  for (auto* node : cycle_path) {
    absl::StrAppend(&cycle_str, node->GetName());
    absl::StrAppend(&cycle_str, " -> ");
  }

  absl::StrAppend(&cycle_str, cycle_path.front()->GetName());
  LOG(INFO) << "Detected Cycle: " << cycle_str;
}

// Helper function to detect cycles in a HloGumgraph using DFS starting from
// the given node.
void DetectCyclesDFS(
    const HloInstructionNode* node,
    absl::flat_hash_set<const HloInstructionNode*>& visited_globally,
    absl::flat_hash_set<const HloInstructionNode*>& current_recursion_stack,
    std::vector<const HloInstructionNode*>& current_path,
    std::vector<std::vector<const HloInstructionNode*>>& cycles) {
  visited_globally.insert(node);
  current_recursion_stack.insert(node);
  current_path.push_back(node);

  for (HloInstructionNode* child : node->children) {
    if (child->is_root) {
      continue;
    }

    if (current_recursion_stack.contains(child)) {
      std::vector<const HloInstructionNode*> cycle;
      auto cycle_start_it = absl::c_find(current_path, child);
      CHECK(cycle_start_it != current_path.end())
          << "Node " << child->GetName()
          << " found in recursion stack but not in current path "
             "during cycle detection.";

      cycle.assign(cycle_start_it, current_path.end());
      cycles.push_back(cycle);
      LogCycle(cycle);
    } else if (!visited_globally.contains(child)) {
      DetectCyclesDFS(child, visited_globally, current_recursion_stack,
                      current_path, cycles);
    }
  }

  current_recursion_stack.erase(node);
  current_path.pop_back();
}

}  // namespace

std::vector<std::vector<const HloInstructionNode*>> DetectAndLogAllCycles(
    const std::vector<HloInstructionNode*>& graph_nodes) {
  LOG(INFO) << "Cycle suspected. Starting cycle detection...";
  absl::flat_hash_set<const HloInstructionNode*> visited_globally;
  absl::flat_hash_set<const HloInstructionNode*> current_recursion_stack;
  std::vector<const HloInstructionNode*> current_path;
  std::vector<std::vector<const HloInstructionNode*>> all_cycles;

  for (HloInstructionNode* node : graph_nodes) {
    if (node->is_root) {
      continue;
    }
    if (!visited_globally.contains(node)) {
      DetectCyclesDFS(node, visited_globally, current_recursion_stack,
                      current_path, all_cycles);
    }
  }

  if (all_cycles.empty()) {
    LOG(INFO) << "No cycles detected in the HloGumgraph.";
  } else {
    LOG(INFO) << "Finished cycle detection. Found " << all_cycles.size()
              << " cycle instance(s). See logs above for details.";
  }

  return all_cycles;
}

}  // namespace hlo_diff
}  // namespace xla
