/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPH_TOPOLOGY_VIEW_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPH_TOPOLOGY_VIEW_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/graph_view.h"

namespace tensorflow {
namespace grappler {

// GraphTopologyView is a helper class to simplify `node-to-node` connectivity
// traversals. Regular `GraphView` simplifies `tensor-to-tensor` traversals:
// connections between output tensors and inputs of a consumer nodes. For the
// topology view we are focused on nodes connected to nodes, and it's irrelevant
// if this connection is formed by one or multiple individual tensors.
//
// Example:
//   a = Placeholder(..)
//   b = Placeholder(..)
//   c = AddN([a, a, b])
//
// GraphView edges:         [a:0 -> c:0, a:0 -> c:1, b:0 -> c:2]
// GraphTopologyView edges: [a -> c, b -> c]
//
// GraphView is used for exploring single node fanins and fanouts, and
// GraphTopologyView is focused on efficient full graph traversals (computing
// graph node properties from transitive fanouts, etc...).
class GraphTopologyView {
 public:
  GraphTopologyView() = default;
  explicit GraphTopologyView(bool skip_invalid_edges)
      : skip_invalid_edges_(skip_invalid_edges) {}

  // Initialize graph topology view from the graph. It's possible to pass
  // additional edges that do not exist in a graph, but must be respected when
  // computing graph topology. Example: Tensorflow runtime allows concurrent
  // execution of dequeue/enqueue ops from the same queue resource, but we might
  // want to enforce ordering between them for the purpose of graph analysis.
  Status InitializeFromGraph(const GraphDef& graph,
                             absl::Span<const GraphView::Edge> ephemeral_edges,
                             bool ignore_control_edges);
  Status InitializeFromGraph(const GraphDef& graph,
                             absl::Span<const GraphView::Edge> ephemeral_edges);
  Status InitializeFromGraph(const GraphDef& graph, bool ignore_control_edges);
  Status InitializeFromGraph(const GraphDef& graph);

  bool is_initialized() const { return graph_ != nullptr; }
  int num_nodes() const { return num_nodes_; }
  const GraphDef* graph() const { return graph_; }

  // Returns true iff the node exists in the underlying graph.
  bool HasNode(absl::string_view node_name) const;

  // Finds a node by name or returns `nullptr` if it's not in the graph.
  const NodeDef* GetNode(absl::string_view node_name) const;
  // Returns a node corresponding to the given node index.
  const NodeDef* GetNode(int node_idx) const;

  // Returns a node index for the given node name, if the name exists in the
  // underlying graph. Otherwise returns empty optional.
  const absl::optional<int> GetNodeIndex(absl::string_view node_name) const;
  // Returns a node index for the given node, if the node belongs to the
  // underlying graph. Otherwise returns empty optional.
  const absl::optional<int> GetNodeIndex(const NodeDef& node) const;

  // Returns all the node indexes that are in the direct fanin of the given
  // node. If the `node_idx` is outside of [0, num_nodes_) returns empty vector.
  const absl::InlinedVector<int, 4>& GetFanin(int node_idx) const;
  // Returns all the node indexes that are in the direct fanout of the given
  // node. If the `node_idx` is outside of [0, num_nodes_) returns empty vector.
  const absl::InlinedVector<int, 2>& GetFanout(int node_idx) const;

 private:
  // If true, all invalid edges and inputs (srd, dst or input node not found in
  // a graph) will be skipped, otherwise initialization will fail with error.
  bool skip_invalid_edges_ = false;

  // WARN: `graph_` must outlive this object and graph nodes must not be
  // destructed, because node names captured with absl::string_view.
  const GraphDef* graph_ = nullptr;  // do not own
  int num_nodes_ = 0;
  std::vector<absl::string_view> index_to_node_name_;
  absl::flat_hash_map<absl::string_view, int> node_name_to_index_;
  std::vector<absl::InlinedVector<int, 4>> fanins_;   // node_idx->input nodes
  std::vector<absl::InlinedVector<int, 2>> fanouts_;  // node_idx->output nodes

  // We need a valid reference to return from GetFanin/GetFanout if the
  // `node_idx` argument is outside of the [0, num_nodes_) range.
  absl::InlinedVector<int, 4> empty_fanin_;
  absl::InlinedVector<int, 2> empty_fanout_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPH_TOPOLOGY_VIEW_H_
