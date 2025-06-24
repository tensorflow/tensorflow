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

#include <sys/stat.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"
#include "xla/util.h"

namespace xla {

// Give aliases to the edge kinds to make code more readable.
static constexpr auto kExecution = ExecutionGraph::NodeEdge::Kind::kExecution;
static constexpr auto kScheduling = ExecutionGraph::NodeEdge::Kind::kScheduling;

// A helper function to create a predicate that checks if a given node edge
// points to a given node id.
static auto EdgePredicate(ExecutionGraph::NodeId id) {
  return [id](const ExecutionGraph::NodeEdge& edge) { return edge.id == id; };
}

// If any of the resource uses requires execution edge, we return kExecution
// edge kind, otherwise we return kScheduling edge kind.
static auto EdgeKind(absl::Span<const ResourceUse> resource_uses) {
  auto requires_execution_edge = [](const ResourceUse& resource_use) {
    auto kind = resource_use.resource()->kind();
    return ExecutionGraph::NodeEdge::KindOf(kind) == kExecution;
  };
  return absl::c_any_of(resource_uses, requires_execution_edge) ? kExecution
                                                                : kScheduling;
}

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
    is_sequential_ &=
        (absl::c_count_if(nodes_defs_[i].in_edges, EdgePredicate(i - 1)) != 0);
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
      if (buffer_rwsets[j].HasConflicts(buffer_rwsets[i])) {
        // If we have buffer conflicts we must add an execution edge to
        // guarantee that we don't have data races at run time.
        builders[j].out_edges.push_back(NodeEdge{kExecution, i});
        builders[i].in_edges.push_back(NodeEdge{kExecution, j});

      } else if (resource_rwsets[j].HasConflicts(resource_rwsets[i])) {
        // If we have resource conflicts, we must check resources that are
        // accessed by both nodes to find out what kind of edge we need to add.
        auto kind = EdgeKind(resource_rwsets[j].Conflicts(resource_rwsets[i]));
        builders[j].out_edges.push_back(NodeEdge{kind, i});
        builders[i].in_edges.push_back(NodeEdge{kind, j});
      }
    }
  }

  // Verify that both in-edges and out-edges are sorted in ascending order
  // according to node id as we use this property later.
  for (NodeId i = 0; i < builders.size(); ++i) {
    auto by_id = [](const NodeEdge& a, const NodeEdge& b) {
      return a.id < b.id;
    };
    DCHECK(absl::c_is_sorted(builders[i].out_edges, by_id));
    DCHECK(absl::c_is_sorted(builders[i].in_edges, by_id));
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
                     : absl::Span<const NodeEdge>(),
        num_out_edges ? absl::MakeConstSpan(&*inserted_out_edges, num_out_edges)
                      : absl::Span<const NodeEdge>(),
        b.priority,
    });
  }

  return std::make_tuple(std::move(nodes_in_edges), std::move(nodes_out_edges),
                         std::move(nodes_defs));
}

int64_t ExecutionGraph::EraseEdge(NodeDefBuilder& from, NodeDefBuilder& to,
                                  NodeEdge::Kind kind) {
  DCHECK_NE(from.id, to.id) << "Nodes must be different";
  DCHECK_LT(from.id, to.id) << "Nodes must be ordered";

  // Short-circuit if out or in-edges are empty.
  if (from.out_edges.empty() || to.in_edges.empty()) {
    DCHECK_EQ(absl::c_count_if(from.out_edges, EdgePredicate(to.id)), 0)
        << "Unexpected out edge from " << from.id << " to " << to.id;
    DCHECK_EQ(absl::c_count_if(to.in_edges, EdgePredicate(from.id)), 0)
        << "Unexpected in edge from " << from.id << " to " << to.id;
    return 0;
  }

  // Short-circuit if out-edges or in-edges don't intersect with `to` or `from`
  // node ids (remember that edges are sorted).
  if (from.out_edges.back().id < to.id || to.in_edges.front().id > from.id) {
    DCHECK_EQ(absl::c_count_if(from.out_edges, EdgePredicate(to.id)), 0)
        << "Unexpected out edge from " << from.id << " to " << to.id;
    DCHECK_EQ(absl::c_count_if(to.in_edges, EdgePredicate(from.id)), 0)
        << "Unexpected in edge from " << from.id << " to " << to.id;
    return 0;
  }

  // Comparator to find a node edge with a given node id.
  auto less_than = [](const NodeEdge& edge, NodeId id) { return edge.id < id; };

  // Check if `from` node has an out edge to `to` node.
  auto out_edges_it = absl::c_lower_bound(from.out_edges, to.id, less_than);
  bool has_out_edge =
      out_edges_it != from.out_edges.end() && out_edges_it->id == to.id;

  // Short-circuit if there is no out edge from `from` node to `to` node.
  if (!has_out_edge) {
    DCHECK_EQ(absl::c_count_if(to.in_edges, EdgePredicate(from.id)), 0)
        << "Unexpected in edge from " << from.id << " to " << to.id;
    return 0;
  }

  // Check if `to` node has an in edge from `from` node.
  auto in_edges_it = absl::c_lower_bound(to.in_edges, from.id, less_than);
  bool has_in_edge =
      in_edges_it != to.in_edges.end() && in_edges_it->id == from.id;

  DCHECK(has_in_edge) << "In-edge must exist if out-edge exists";
  DCHECK_EQ(static_cast<int>(in_edges_it->kind),
            static_cast<int>(out_edges_it->kind))
      << "Edges kind must match";

  // At this point we must have exactly one edge between `from` and `to` nodes.
  DCHECK_EQ(absl::c_count_if(from.out_edges, EdgePredicate(to.id)), 1)
      << "Expected exactly one out edge from " << from.id << " to " << to.id;
  DCHECK_EQ(absl::c_count_if(to.in_edges, EdgePredicate(from.id)), 1)
      << "Expected exactly one in edge from " << from.id << " to " << to.id;

  // We can't erase an edge with a stronger ordering guarantee.
  if (in_edges_it->kind > kind) {
    return 0;
  }

  // We erased exactly one edge between `from` and `to` nodes.
  from.out_edges.erase(out_edges_it);
  to.in_edges.erase(in_edges_it);
  return 1;
}

namespace {

// A state of a DFS traversal for transitive reduction.
class TransitiveReductionDfsState {
 public:
  void PushToStack(ExecutionGraph::NodeEdge edge) {
    if (!visited_[edge.id]) {
      ++(edge.kind == kExecution ? num_execution_edges_
                                 : num_scheduling_edges_);
      stack_.push_back(edge);
      visited_[edge.id] = true;
    }
  }

  void PushToStack(absl::Span<const ExecutionGraph::NodeEdge> edges) {
    for (const ExecutionGraph::NodeEdge& edge : edges) {
      PushToStack(edge);
    }
  }

  ExecutionGraph::NodeEdge PopFromStack() {
    ExecutionGraph::NodeEdge edge = stack_.back();
    --(edge.kind == kExecution ? num_execution_edges_ : num_scheduling_edges_);
    stack_.pop_back();
    return edge;
  }

  bool Empty() const { return stack_.empty(); }

  void Visited(ExecutionGraph::NodeId id) { visited_[id] = true; }
  size_t NumVisited() const { return absl::c_count(visited_, true); }

  void Clear(size_t num_nodes) {
    stack_.clear();
    visited_.assign(num_nodes, false);
  }

  bool num_execution_edges() const { return num_execution_edges_; }
  bool num_scheduling_edges() const { return num_scheduling_edges_; }

 private:
  std::vector<ExecutionGraph::NodeEdge> stack_;
  std::vector<bool> visited_;

  // The number of execution and scheduling edges currently in the stack.
  size_t num_execution_edges_ = 0;
  size_t num_scheduling_edges_ = 0;
};

}  // namespace

int64_t ExecutionGraph::RunTransitiveReductionAndUpdatePriorities(
    absl::Span<NodeDefBuilder> builders) {
  int64_t num_erased_edges = 0;

  // Keep workspace for DFS traversal between iterations.
  TransitiveReductionDfsState state;

  // For each node we do a DFS traversal and delete redundant edges that
  // connect source node with the node reachable via DFS. We do traversal in
  // reverse order as we end up traversing fewer edges this way.
  for (int64_t i = builders.size() - 1; i >= 0; --i) {
    NodeDefBuilder& source_node = builders[i];

    // Clear DFS state from previous iteration.
    state.Clear(builders.size());

    // Make a copy of out edges to avoid invalidating iterators.
    for (NodeEdge out_edge : std::vector<NodeEdge>(source_node.out_edges)) {
      DCHECK(state.Empty()) << "Stack must be empty at the start of the DFS";

      // Initialize state with nodes reachable via `out_edge`. We mark immediate
      // out nodes as visited to correctly compute node priority below.
      NodeDefBuilder& out_node = builders[out_edge.id];
      state.Visited(out_edge.id);
      state.PushToStack(out_node.out_edges);

      // Do a round of DFS traversal and delete redundant edges from the
      // `source_node` to the nodes reachable via DFS.
      while (!state.Empty()) {
        NodeEdge node_edge = state.PopFromStack();
        NodeDefBuilder& node = builders[node_edge.id];

        // If we reached `node` via a scheduling edge, then we can't remove an
        // execution edge from the `source_node`, as we might weaker the
        // execution order and introduce a data race.
        bool has_scheduling_edge = out_edge.kind == kScheduling ||
                                   node_edge.kind == kScheduling ||
                                   state.num_scheduling_edges();
        NodeEdge::Kind kind = has_scheduling_edge ? kScheduling : kExecution;
        num_erased_edges += EraseEdge(source_node, node, kind);

        // Keep following nodes reachable via `node` out edges.
        state.PushToStack(node.out_edges);
      }
    }

    // Set node priority to the number of visited nodes in the DFS traversal.
    source_node.priority = state.NumVisited();
  }

  return num_erased_edges;
}

// Execution graph renderer registration logic

absl::Mutex renderer_mu(absl::kConstInit);
ExecutionGraph::Renderer* graph_renderer ABSL_GUARDED_BY(renderer_mu) = nullptr;

ExecutionGraph::Renderer* ExecutionGraph::GetRenderer() {
  absl::MutexLock lock(&renderer_mu);
  return graph_renderer;
}

void ExecutionGraph::RegisterRenderer(
    std::unique_ptr<ExecutionGraph::Renderer> renderer) {
  absl::MutexLock lock(&renderer_mu);
  if (graph_renderer != nullptr) {
    LOG(WARNING) << "Multiple calls to RegisterRenderer. Last "
                    "call wins, but because order of initialization in C++ is "
                    "nondeterministic, this may not be what you want.";
    delete graph_renderer;
  }
  graph_renderer = renderer.release();
}

}  // namespace xla
