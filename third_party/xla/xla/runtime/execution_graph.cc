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

#include "xla/runtime/execution_graph.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"
#include "xla/util.h"

namespace xla {

ExecutionGraph::ExecutionGraph(NodesEdges nodes_in_edges,
                               NodesEdges nodes_out_edges,
                               std::vector<NodeDef> nodes_defs)
    : nodes_in_edges_(std::move(nodes_in_edges)),
      nodes_out_edges_(std::move(nodes_out_edges)),
      nodes_defs_(std::move(nodes_defs)),
      is_sequential_(true) {
  // Identify source and sink nodes in the execution graph.
  for (NodeId i = 0; i < nodes_defs_.size(); ++i) {
    // Mark nodes with empty in-edges as source nodes.
    if (nodes_defs_[i].in_edges.empty()) {
      source_.push_back(i);
    }

    // Mark nodes with empty out-edges as sink nodes.
    if (nodes_defs_[i].out_edges.empty()) {
      sink_.push_back(i);
    }
  }

  // Check if constructed execution DAG is sequential: every node depends on the
  // completion of the previous node.
  for (NodeId i = 1; i < nodes_defs_.size() && is_sequential_; ++i) {
    is_sequential_ &= (absl::c_count(nodes_defs_[i].in_edges, i - 1) != 0);
  }

  VLOG(2) << absl::StreamFormat(
      "Constructed execution graph with %d nodes: #source_nodes=%d "
      "#sink_nodes=%d, is_sequential=%v",
      nodes_defs_.size(), source_.size(), sink_.size(), is_sequential_);

  // Sanity check that all vectors are empty or all vectors are non-empty.
  DCHECK((!source_.empty() && !sink_.empty()) ||
         (source_.empty() && sink_.empty()));
}

absl::StatusOr<ExecutionGraph> ExecutionGraph::Create(
    absl::Span<const Operation* const> operations) {
  // Make sure that operations sequence size fits into NodeId.
  if (operations.size() > std::numeric_limits<NodeId>::max()) {
    return Internal("Can't create ExecutionGraph for more than %d operations",
                    std::numeric_limits<NodeId>::max());
  }

  std::vector<NodeDefBuilder> builders(operations.size());

  std::vector<BufferUse::ReadWriteSet> buffer_rwsets(operations.size());
  std::vector<ResourceUse::ReadWriteSet> resource_rwsets(operations.size());

  // TODO(ezhulenev): This is very inefficient O(N^2) complexity algorithm
  // that will create a lot of redundant edges. We can do much better by
  // stopping traversal once we prove that we already have dependencies on the
  // most recent updates that touch the whole buffer slice.

  for (NodeId i = 0; i < operations.size(); ++i) {
    builders[i].id = i;

    const Operation* op = operations[i];
    buffer_rwsets[i].AddAll(op->BufferUses());
    resource_rwsets[i].AddAll(op->ResourceUses());

    for (NodeId j = 0; j < i; ++j) {
      // Check if node `i` must be executed after node `j`.
      if (buffer_rwsets[j].HasConflicts(buffer_rwsets[i]) ||
          resource_rwsets[j].HasConflicts(resource_rwsets[i])) {
        builders[j].out_edges.push_back(i);
        builders[i].in_edges.push_back(j);
      }
    }
  }

  // Verify that both in-edges and out-edges are sorted in ascending order as we
  // use this property later.
  for (NodeId i = 0; i < builders.size(); ++i) {
    DCHECK(absl::c_is_sorted(builders[i].out_edges));
    DCHECK(absl::c_is_sorted(builders[i].in_edges));
  }

  // Erase redundant edges between nodes.
  int64_t num_erased_edges =
      RunTransitiveReductionAndUpdatePriorities(absl::MakeSpan(builders));
  VLOG(5) << absl::StreamFormat(
      "Transitive reduction erased %d edges from the execution graph",
      num_erased_edges);

  auto [in_edges, out_edges, nodes_defs] = CreateNodeDefs(std::move(builders));
  return ExecutionGraph(std::move(in_edges), std::move(out_edges),
                        std::move(nodes_defs));
}

std::tuple<ExecutionGraph::NodesEdges, ExecutionGraph::NodesEdges,
           std::vector<ExecutionGraph::NodeDef>>
ExecutionGraph::CreateNodeDefs(std::vector<NodeDefBuilder> builders) {
  // Find how many in-edges and out-edges we have in total.
  size_t num_in_edges = 0, num_out_edges = 0;
  for (const NodeDefBuilder& b : builders) {
    num_in_edges += b.in_edges.size();
    num_out_edges += b.out_edges.size();
  }

  NodesEdges nodes_in_edges;
  NodesEdges nodes_out_edges;
  std::vector<NodeDef> nodes_defs;

  // Reserve memory to avoid re-allocation and dangling spans into freed memory.
  nodes_in_edges.reserve(num_in_edges);
  nodes_out_edges.reserve(num_out_edges);
  nodes_defs.reserve(builders.size());

  for (const NodeDefBuilder& b : builders) {
    size_t num_in_edges = b.in_edges.size();
    size_t num_out_edges = b.out_edges.size();

    auto inserted_in_edges = nodes_in_edges.insert(
        nodes_in_edges.end(), b.in_edges.begin(), b.in_edges.end());
    auto inserted_out_edges = nodes_out_edges.insert(
        nodes_out_edges.end(), b.out_edges.begin(), b.out_edges.end());

    nodes_defs.push_back(NodeDef{
        b.id,
        num_in_edges ? absl::MakeConstSpan(&*inserted_in_edges, num_in_edges)
                     : absl::Span<const NodeId>(),
        num_out_edges ? absl::MakeConstSpan(&*inserted_out_edges, num_out_edges)
                      : absl::Span<const NodeId>(),
        b.priority,
    });
  }

  return std::make_tuple(std::move(nodes_in_edges), std::move(nodes_out_edges),
                         std::move(nodes_defs));
}

int64_t ExecutionGraph::EraseEdge(NodeDefBuilder& from, NodeDefBuilder& to) {
  DCHECK_NE(from.id, to.id) << "Nodes must be different";
  DCHECK_LT(from.id, to.id) << "Nodes must be ordered";

  // Short-circuit if out or in-edges are empty.
  if (from.out_edges.empty() || to.in_edges.empty()) {
    DCHECK_EQ(absl::c_count(from.out_edges, to.id), 0) << "Unexpected out edge";
    DCHECK_EQ(absl::c_count(to.in_edges, from.id), 0) << "Unexpected in edge";
    return 0;
  }

  // Short-circuit if out-edges or in-edges don't intersect with `to` or `from`
  // node ids (remember that edges are sorted).
  if (from.out_edges.back() < to.id || to.in_edges.front() > from.id) {
    DCHECK_EQ(absl::c_count(from.out_edges, to.id), 0) << "Unexpected out edge";
    DCHECK_EQ(absl::c_count(to.in_edges, from.id), 0) << "Unexpected in edge";
    return 0;
  }

  // Check if `from` node has an out edge to `to` node.
  auto out_edges_it = absl::c_lower_bound(from.out_edges, to.id);
  bool has_out_edge =
      out_edges_it != from.out_edges.end() && *out_edges_it == to.id;

  // Short-circuit if there is no out edge from `from` node to `to` node.
  if (!has_out_edge) {
    DCHECK_EQ(absl::c_count(to.in_edges, from.id), 0) << "Unexpected in edge";
    return 0;
  }

  // Check if `to` node has an in edge from `from` node.
  auto in_edges_it = absl::c_lower_bound(to.in_edges, from.id);
  bool has_in_edge =
      in_edges_it != to.in_edges.end() && *in_edges_it == from.id;

  DCHECK(has_in_edge) << "In-edge must exist if out-edge exists";

  from.out_edges.erase(out_edges_it);
  to.in_edges.erase(in_edges_it);

  // We erased one edge between `from` and `to` nodes.
  return 1;
}

int64_t ExecutionGraph::RunTransitiveReductionAndUpdatePriorities(
    absl::Span<NodeDefBuilder> builders) {
  int64_t num_erased_edges = 0;

  // Keep workspace for DFS traversal between iterations.
  std::vector<int64_t> stack;
  std::vector<bool> visited;

  auto add_to_stack = [&](int64_t node_id) {
    if (!visited[node_id]) {
      stack.push_back(node_id);
      visited[node_id] = true;
    }
  };

  // For each node we do a DFS traversal and delete redundant edges that
  // connect source node with the node reachable via DFS. We do traversal in
  // reverse order as we end up traversing fewer edges this way.
  for (int64_t i = builders.size() - 1; i >= 0; --i) {
    NodeDefBuilder& source_node = builders[i];

    // Clear DFS workspace from previous iteration.
    stack.clear();
    visited.assign(builders.size(), false);

    // Initialize stack with nodes reachable via immediate out nodes. We mark
    // immediate out nodes as visited to correctly compute node priority below.
    for (int64_t out_id : source_node.out_edges) {
      NodeDefBuilder& out_node = builders[out_id];
      visited[out_id] = true;
      for (int64_t start_id : out_node.out_edges) add_to_stack(start_id);
    }

    // Traverse the graph and delete redundant edges.
    while (!stack.empty()) {
      int64_t node_id = stack.back();
      stack.pop_back();

      NodeDefBuilder& node = builders[node_id];
      num_erased_edges += EraseEdge(source_node, node);

      for (int64_t out_id : node.out_edges) add_to_stack(out_id);
    }

    // Set node priority to the number of visited nodes in the DFS traversal.
    source_node.priority = absl::c_count(visited, true);
  }

  return num_erased_edges;
}

}  // namespace xla
