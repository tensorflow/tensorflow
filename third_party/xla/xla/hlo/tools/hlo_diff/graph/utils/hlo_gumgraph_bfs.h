/*
 * Copyright 2025 The OpenXLA Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_HLO_TOOLS_HLO_DIFF_GRAPH_UTILS_HLO_GUMGRAPH_BFS_H_
#define XLA_HLO_TOOLS_HLO_DIFF_GRAPH_UTILS_HLO_GUMGRAPH_BFS_H_

#include <cstdint>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/types/span.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"

namespace xla::hlo_diff {

// Direction of the BFS traversal.
enum class BfsTraversalDirection : std::int8_t { kForward, kReverse };

// Performs a breadth first search of the HLO Module starting with specified
// instruction node as start and calls supplied per node execution function.
//
// If the per_node_fn returns false for a node, the BFS traversal will be
// terminate immediately.
//
// The BFS traversal is performed in the specified direction.
// kForward: Start from the start node and traverse forward to the nodes
// children.
// kReverse: Start from the start node and traverse backwards to the
// nodes parents.
//
// The node_limit parameter should be set to the number of nodes in the
// HLOGumgraph, as its used to track the visit state of each node during
// traversal.
//
// If the expand_node_fn returns false for a node, the children of the node
// will not be visited.
//
// Both per_node_fn and expand_node_fn receive an 'int distance' parameter,
// representing the distance (number of edges) from the start node(s) to the
// currently visited node.
void HloGumgraphBfs(
    absl::Span<const HloInstructionNode* const> start_nodes,
    absl::FunctionRef<bool(const HloInstructionNode&, int distance)>
        per_node_fn,
    BfsTraversalDirection direction, int node_limit,
    absl::FunctionRef<bool(const HloInstructionNode&, int distance)>
        expand_node_fn =
            [](const HloInstructionNode&, int distance) { return true; });

// Same function as above but with a single start node.
inline void HloGumgraphBfs(
    const HloInstructionNode& start_node,
    absl::FunctionRef<bool(const HloInstructionNode&, int distance)>
        per_node_fn,
    BfsTraversalDirection direction, int node_limit,
    absl::FunctionRef<bool(const HloInstructionNode&, int distance)>
        expand_node_fn =
            [](const HloInstructionNode&, int distance) { return true; }) {
  return HloGumgraphBfs(std::vector<const HloInstructionNode*>({&start_node}),
                        per_node_fn, direction, node_limit, expand_node_fn);
};

// Breadth first search from multiple start nodes. Takes no extra distance
// parameter.
inline void HloGumgraphBfs(
    absl::Span<const HloInstructionNode* const> start_nodes,
    absl::FunctionRef<bool(const HloInstructionNode&)> per_node_fn,
    BfsTraversalDirection direction, int node_limit,
    absl::FunctionRef<bool(const HloInstructionNode&)> expand_node_fn =
        [](const HloInstructionNode&) { return true; }) {
  HloGumgraphBfs(
      start_nodes,
      [&](const HloInstructionNode& node, int distance) {
        return per_node_fn(node);
      },
      direction, node_limit,
      [&](const HloInstructionNode& node, int distance) {
        return expand_node_fn(node);
      });
};

// Performs a breadth first search of the HLO Module starting with specified
// instruction node. Takes no extra distance parameter.
inline void HloGumgraphBfs(
    const HloInstructionNode& start_node,
    absl::FunctionRef<bool(const HloInstructionNode&)> per_node_fn,
    BfsTraversalDirection direction, int node_limit,
    absl::FunctionRef<bool(const HloInstructionNode&)> expand_node_fn =
        [](const HloInstructionNode&) { return true; }) {
  HloGumgraphBfs(
      std::vector<const HloInstructionNode*>({&start_node}),
      [&](const HloInstructionNode& node, int distance) {
        return per_node_fn(node);
      },
      direction, node_limit,
      [&](const HloInstructionNode& node, int distance) {
        return expand_node_fn(node);
      });
}

// Returns all nodes start from the given node in BFS order. Check comment of
// HloGumgraphBfs for more details.
std::vector<const HloInstructionNode*> GetAllNodesInBfsOrder(
    const HloInstructionNode& root, BfsTraversalDirection direction,
    int node_limit = 100000);

}  // namespace xla::hlo_diff

#endif  // XLA_HLO_TOOLS_HLO_DIFF_GRAPH_UTILS_HLO_GUMGRAPH_BFS_H_
