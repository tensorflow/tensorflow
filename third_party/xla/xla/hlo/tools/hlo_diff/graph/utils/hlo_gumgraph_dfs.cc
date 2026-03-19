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

#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_dfs.h"

#include <cstdint>
#include <vector>

#include "absl/functional/function_ref.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"

namespace xla::hlo_diff {
namespace {

enum class VisitState : uint8_t { kNew = 0, kVisiting = 1, kVisited = 2 };

}  // namespace

void HloGumgraphDfs(
    const HloInstructionNode& start_node,
    absl::FunctionRef<void(const HloInstructionNode&)> per_node_fn,
    DfsTraversalOrder order, int node_limit,
    absl::FunctionRef<bool(const HloInstructionNode&)> expand_node_fn) {
  std::vector<VisitState> visited(node_limit);

  std::vector<const HloInstructionNode*> stack = {&start_node};

  while (!stack.empty()) {
    const HloInstructionNode* node = stack.back();
    VisitState& visit_state = visited[node->unique_node_index];

    if (visit_state == VisitState::kNew) {
      visit_state = VisitState::kVisiting;
      if (order == DfsTraversalOrder::kPreOrder) {
        per_node_fn(*node);
      }
    } else {
      stack.pop_back();
      if (visit_state == VisitState::kVisiting) {
        visit_state = VisitState::kVisited;
        if (order == DfsTraversalOrder::kPostOrder) {
          per_node_fn(*node);
        }
      }
      continue;
    }

    if (!expand_node_fn(*node)) {
      continue;
    }
    // Push children in reverse order (right-to-left) onto the stack.
    // This is to ensure that nodes are processed in a left-to-right order,
    // aligning with the intended traversal logic and consistency
    // with the BFS.
    for (auto it = node->children.rbegin(); it != node->children.rend(); ++it) {
      auto* child = *it;
      if (visited[child->unique_node_index] == VisitState::kNew) {
        stack.push_back(child);
      } else {
        // Already fully visited, no need to visit.
      }
    }
  }
}

std::vector<const HloInstructionNode*> GetAllNodesInDfsOrder(
    const HloInstructionNode& root, DfsTraversalOrder order, int node_limit) {
  std::vector<const HloInstructionNode*> subgraph;
  HloGumgraphDfs(
      root, [&](const HloInstructionNode& node) { subgraph.push_back(&node); },
      order, node_limit);
  return subgraph;
}

}  // namespace xla::hlo_diff
