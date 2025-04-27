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

#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_bfs.h"

#include <cstdint>
#include <queue>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"

namespace xla {
namespace hlo_diff {
namespace {

bool GetVisited(std::vector<uint64_t>& visited, int node_index) {
  int index = node_index / 64;
  CHECK_LT(index, visited.size());
  return visited[index] & (1ull << (node_index % 64));
}

void SetVisited(std::vector<uint64_t>& visited, int node_index) {
  int index = node_index / 64;
  CHECK_LT(index, visited.size());
  visited[index] |= (1ull << (node_index % 64));
}
}  // namespace

void HloGumgraphBfs(
    absl::Span<const HloInstructionNode* const> start_nodes,
    absl::FunctionRef<bool(const HloInstructionNode&)> per_node_fn,
    BfsTraversalDirection direction, int node_limit,
    absl::FunctionRef<bool(const HloInstructionNode&)> expand_node_fn) {
  std::queue<const HloInstructionNode*> nodes_to_expand;
  std::vector<uint64_t> visited((node_limit + 63) / 64, 0);

  for (const HloInstructionNode* start_node : start_nodes) {
    CHECK(start_node != nullptr) << "Expected a non-null root node";
    if (!per_node_fn(*start_node)) {
      return;
    }
    if (expand_node_fn(*start_node)) {
      nodes_to_expand.push(start_node);
    }
    SetVisited(visited, start_node->unique_node_index);
  }

  while (!nodes_to_expand.empty()) {
    const HloInstructionNode* current_node = nodes_to_expand.front();
    nodes_to_expand.pop();

    std::vector<HloInstructionNode*> adjacent_nodes =
        direction == BfsTraversalDirection::kForward ? current_node->children
                                                     : current_node->parents;

    for (auto* adjacent_node : adjacent_nodes) {
      if (!GetVisited(visited, adjacent_node->unique_node_index)) {
        if (!per_node_fn(*adjacent_node)) {
          return;
        }
        if (expand_node_fn(*adjacent_node)) {
          nodes_to_expand.push(adjacent_node);
        }
        SetVisited(visited, adjacent_node->unique_node_index);
      }
    }
  }
}

void HloGumgraphBfs(
    const HloInstructionNode& start_node,
    absl::FunctionRef<bool(const HloInstructionNode&)> per_node_fn,
    BfsTraversalDirection direction, int node_limit,
    absl::FunctionRef<bool(const HloInstructionNode&)> expand_node_fn) {
  HloGumgraphBfs(std::vector<const HloInstructionNode*>({&start_node}),
                 per_node_fn, direction, node_limit, expand_node_fn);
}

std::vector<const HloInstructionNode*> GetAllNodesInBfsOrder(
    const HloInstructionNode& root, BfsTraversalDirection direction,
    int node_limit) {
  std::vector<const HloInstructionNode*> subgraph;
  HloGumgraphBfs(
      root,
      [&](const HloInstructionNode& node) {
        subgraph.push_back(&node);
        return true;
      },
      BfsTraversalDirection::kForward, node_limit);
  return subgraph;
}

}  // namespace hlo_diff
}  // namespace xla
