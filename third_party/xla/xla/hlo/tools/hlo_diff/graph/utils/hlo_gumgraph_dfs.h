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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_GRAPH_UTILS_HLO_GUMGRAPH_DFS_H_
#define XLA_HLO_TOOLS_HLO_DIFF_GRAPH_UTILS_HLO_GUMGRAPH_DFS_H_

#include <cstdint>
#include <vector>

#include "absl/functional/function_ref.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"

namespace xla::hlo_diff {

// DFS traversal order: pre-order or post-order.
enum class DfsTraversalOrder : std::int8_t { kPreOrder, kPostOrder };

// Performs a depth first search of the HLO Module starting with specified
// instruction node as start and calls supplied per node execution function for
// each visited node.
//
// The traversal order determines whether the per node function is invoked
// before or after the children of the node are visited, i.e. pre-order or
// post-order traversal.
//
// The node_limit parameter should be set to the number of nodes in the
// HLOGumgraph, as its used to track the visit state of each node during
// traversal.
//
// If the expand_node_fn returns false for a node, the children of the node
// will not be visited.
void HloGumgraphDfs(
    const HloInstructionNode& start_node,
    absl::FunctionRef<void(const HloInstructionNode&)> per_node_fn,
    DfsTraversalOrder order, int node_limit,
    absl::FunctionRef<bool(const HloInstructionNode&)> expand_node_fn =
        [](const HloInstructionNode&) { return true; });

// Returns all nodes in the HLO Module in DFS order starting from the provided
// root node. Check  comment of HloGumgraphDfs for more details.
std::vector<const HloInstructionNode*> GetAllNodesInDfsOrder(
    const HloInstructionNode& root, DfsTraversalOrder order, int node_limit);

}  // namespace xla::hlo_diff

#endif  // XLA_HLO_TOOLS_HLO_DIFF_GRAPH_UTILS_HLO_GUMGRAPH_DFS_H_
