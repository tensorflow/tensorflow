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

#ifndef XLA_RUNTIME_EXECUTION_GRAPH_H_
#define XLA_RUNTIME_EXECUTION_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"

namespace xla {

// Execution graph defines the execution order of operations based on their
// buffer use and resource use dependencies.
//
// In XLA:GPU and XLA:CPU we compile HLO programs to a sequence of operations
// executed on the underlying device. XLA compiler creates a sequential schedule
// that is used to assign buffers to operations. These operation can be
// implemented as thunks, or as commands (only on XLA:GPU backend with CUDA
// graphs). Each operations reads and writes from/to buffer slices and uses
// resources (i.e. collective communicator).
//
// At run time we can relax sequential schedule and execute operations
// concurrently, as long as we don't create data races (reading and writing
// from/to the same or overlapping buffer slices concurrently), or resource
// races (using the same mutable resource concurrently).
//
// Resources can behave as buffers and require an execution order (operation
// must wait for the completion of execution of all dependencies), or as a
// scheduling barrier (operation must wait for the completion of scheduling of
// all dependencies). See more details in the `NodeEdge::Kind` definition.
//
// We use buffer and resource use conflicts to define an execution order of
// operations as a directed acyclic graph (DAG) that satisfies all dependencies.
//
// Backend-specific runtime relies on the execution graph to execute operations
// concurrently usult the underlying device concurrency mechanism, e.g.
// thread pools on CPU device, or CUDA streams on NVIDIA GPU device.
class ExecutionGraph {
 public:
  // Nodes identified by their index in the operation sequence.
  using NodeId = int32_t;

  static constexpr NodeId kInvalidNodeId = std::numeric_limits<NodeId>::min();

  // A base class for an operation that can be executed by the runtime.
  class Operation {
   public:
    virtual ~Operation() = default;

    virtual absl::string_view name() const = 0;
    // Optional function that allows grouping operations of the same kind. E.x.
    // on XLA:CPU this is the id of the thunk kind, and is used for color coding
    // graph visualization.
    virtual int64_t op_type_id() const { return 0; };
    virtual absl::Span<const BufferUse> BufferUses() const = 0;
    virtual absl::Span<const ResourceUse> ResourceUses() const = 0;

   protected:
    Operation() = default;

    Operation(const Operation&) = default;
    Operation& operator=(const Operation&) = default;

    Operation(Operation&&) = default;
    Operation& operator=(Operation&&) = default;
  };

  // An edge between two nodes created for the execution graph operations.
  struct NodeEdge {
    // Edge kind defines execution ordering between two operations. Scheduling
    // edge is weaker than an execution edge, as it gives more flexibility
    // to the backend runtime to execute operations concurrently.
    enum class Kind {
      // If two operations have a scheduling edge between them, then the
      // dependent operation must be scheduled (start execution) after the
      // dependency operation scheduled (started execution), however it doesn't
      // have to wait for the completion of execution. We use this type of
      // edge to guarantee that operations that share the same resource (i.e.
      // collective communicator) start execution in a deterministic order
      // across different ranks, however the execution of operations can
      // overlap and finish in any order, and backend-implementation specific.
      kScheduling,

      // If two operations have an execution edge between them, then the
      // dependent operation must wait for the completion of dependency
      // operation execution. We use this type of edge to order execution of
      // operations that read and write from/to the same buffers, as otherwise
      // we may create data races.
      kExecution,
    };

    static constexpr NodeEdge::Kind KindOf(Resource::Kind resource) {
      switch (resource) {
        case Resource::kToken:
          return NodeEdge::Kind::kExecution;
        case Resource::kCollectiveCommunicator:
          return NodeEdge::Kind::kScheduling;
      }
    }

    bool operator==(const NodeEdge& other) const {
      return kind == other.kind && id == other.id;
    }

    template <typename Sink>
    friend void AbslStringify(Sink& sink, Kind kind) {
      sink.Append(kind == Kind::kScheduling ? "scheduling" : "execution");
    }

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const NodeEdge& edge) {
      absl::Format(&sink, "NodeEdge {kind: %v, id: %v}", edge.kind, edge.id);
    }

    Kind kind;
    NodeId id;
  };

  // NodeDef defines a dependency-based execution order for all operations.
  struct NodeDef {
    NodeId id = kInvalidNodeId;

    absl::Span<const NodeEdge> in_edges;
    absl::Span<const NodeEdge> out_edges;

    // When doing the transitive reduction, we assign a priority to each node
    // based on the number of nodes that are reachable from the given node. The
    // assumption is that by executing nodes with higher priority first we will
    // unlock more nodes for execution.
    int64_t priority = 0;
  };

  class Renderer {
   public:
    Renderer() = default;
    virtual ~Renderer() = default;

    // Generates a string representation for the given execution graph
    // operations which can be published to a URL using `PublishGraph`.
    virtual std::string GenerateGraphAsString(
        absl::Span<const ExecutionGraph::Operation* const> operations) = 0;

    // Publishes the generated graph.
    virtual absl::StatusOr<std::string> PublishGraph(
        absl::string_view graph_as_string) = 0;
  };

  // Returns the registered renderer for execution graphs.
  static Renderer* GetRenderer();

  // Registers a renderer for execution graphs.
  static void RegisterRenderer(std::unique_ptr<Renderer> renderer);

  // Constructs an execution graph from a sequence of operations.
  template <typename Op,
            std::enable_if_t<std::is_base_of_v<Operation, Op>>* = nullptr>
  static absl::StatusOr<ExecutionGraph> Create(absl::Span<const Op> ops) {
    absl::InlinedVector<const Operation*, 32> ptrs(ops.size());
    for (size_t i = 0; i < ops.size(); ++i) {
      ptrs[i] = &ops[i];
    }
    return Create(ptrs);
  }

  // Constructs an execution graph from a sequence of operations.
  static absl::StatusOr<ExecutionGraph> Create(
      absl::Span<const Operation* const> operations);

  // Returns execution graph nodes definitions.
  absl::Span<const NodeDef> nodes_defs() const { return nodes_defs_; }

  // Source nodes are the nodes that do not have any in-edges.
  absl::Span<const NodeId> source() const { return source_; }

  // Sink nodes are the nodes that do not have any out-edges.
  absl::Span<const NodeId> sink() const { return sink_; }

  // Returns true if a given node id is a source node.
  bool is_source(NodeId id) const {
    return absl::c_find(source_, id) != source_.end();
  }

  // Returns true if a given node id is a sink node.
  bool is_sink(NodeId id) const {
    return absl::c_find(sink_, id) != sink_.end();
  }

  // Returns in-edges for a given node id.
  absl::Span<const NodeEdge> in_edges(NodeId id) const {
    DCHECK_EQ(id, nodes_defs_[id].id);
    return nodes_defs_[id].in_edges;
  }

  // Returns out-edges for a given node id.
  absl::Span<const NodeEdge> out_edges(NodeId id) const {
    DCHECK_EQ(id, nodes_defs_[id].id);
    return nodes_defs_[id].out_edges;
  }

  // Returns priority for a given node id.
  int64_t priority(NodeId id) const {
    DCHECK_EQ(id, nodes_defs_[id].id);
    return nodes_defs_[id].priority;
  }

  bool is_sequential() const { return is_sequential_; }

 private:
  // We store all `in_edges` and `out_edges` referenced by the `NodeDef` inside
  // large vectors to optimize for data locality on a hot path.
  using NodesEdges = std::vector<NodeEdge>;

  // A NodeDef builder to collect all in-edges and out-edges before constructing
  // a NodeDef. We use it at dependency graph construction time when we don't
  // know how many in-edges and out-edges we have in total.
  struct NodeDefBuilder {
    NodeId id = kInvalidNodeId;
    int64_t priority = 0;
    std::vector<NodeEdge> in_edges;
    std::vector<NodeEdge> out_edges;
  };

  ExecutionGraph(NodesEdges nodes_in_edges, NodesEdges nodes_out_edges,
                 std::vector<NodeDef> nodes_defs);

  // Converts a vector of NodeDefBuilder to a tuple of NodesEdges and a vector
  // of NodeDef.
  static std::tuple<NodesEdges, NodesEdges, std::vector<NodeDef>>
  CreateNodeDefs(std::vector<NodeDefBuilder> builders);

  // Erases edge from `from` node to `to` node if it exists and it has a weaker
  // ordering than the given `kind`. We rely on the fact that out and in-edges
  // are sorted and use binary search on a critical path.
  static int64_t EraseEdge(NodeDefBuilder& from, NodeDefBuilder& to,
                           NodeEdge::Kind kind);

  // Runs a transitive reduction on the NodeDefBuilder graph to remove redundant
  // edges, and updates nodes priorities. Returns the number of removed edges.
  //
  // See: https://en.wikipedia.org/wiki/Transitive_reduction
  static int64_t RunTransitiveReductionAndUpdatePriorities(
      absl::Span<NodeDefBuilder> builders);

  NodesEdges nodes_in_edges_;   // `in_edges` referenced by `nodes_defs_`
  NodesEdges nodes_out_edges_;  // `out_edges` referenced by `nodes_defs_`
  std::vector<NodeDef> nodes_defs_;

  std::vector<NodeId> source_;
  std::vector<NodeId> sink_;

  // If NodeDef graph dependency structure is sequential and does not have any
  // opportunities for executing operations concurrently. XLA runtime can use
  // this property of the execution graph to skip expensive async execution and
  // simply run all operations one by one.
  bool is_sequential_;
};

}  // namespace xla

#endif  // XLA_RUNTIME_EXECUTION_GRAPH_H_
