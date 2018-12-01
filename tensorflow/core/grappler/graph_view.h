/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPH_VIEW_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPH_VIEW_H_

#include <unordered_map>
#include <unordered_set>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {

// Map a node/op's input/output port_id to arg_id.
//
// The port_id refers to the n-th tensor of the node, while the arg_id refers to
// the n-th arg of the op. These two can be different if an op's arg is a list
// of tensors.
//
// We return -1 for any invalid port_id (i.e., no corresponding arg_id).
int OpOutputPortIdToArgId(const NodeDef& node, const OpDef& op, int port_id);
int OpInputPortIdToArgId(const NodeDef& node, const OpDef& op, int port_id);

namespace internal {

// GraphViewInternal is a helper class to simplify graph traversal. It creates
// an immutable view of the nodes and edges represented by a GraphDef protocol
// buffer.
//
// There are two public classes implementing GraphViewInternal:
//
// - GraphView: constructed from the `const GraphDef` and doesn't allow
//   to mutate underlying graph via input/output ports lookup functions (ports
//   have const pointers to nodes).
//
// - MutableGraphView: constructed from the 'GraphDef` and allows to mutate
//   the graph via input/output ports lookup functions (ports have non-const
//   pointers to nodes), and also have couple additional functions to
//   add/remove/replace nodes in the graph.
//
// --------------------------- !!! WARNING !!! ---------------------------------
//     Removing nodes from the graph outside of MutableGraphView will
//     lead to segfaults! Guaranteed by absl::string_view!
// -----------------------------------------------------------------------------
//
template <typename GraphDefT, typename NodeDefT>
class GraphViewInternal {
 public:
  struct Port {
    Port() : node(nullptr), port_id(0) {}
    Port(NodeDefT* n, int port) : node(n), port_id(port) {}

    bool operator==(const Port& other) const {
      return node == other.node && port_id == other.port_id;
    }

    template <typename H>
    friend H AbslHashValue(H h, const Port& p) {
      return H::combine(std::move(h), p.node, p.port_id);
    }

    NodeDefT* node;
    int port_id;
  };

  struct InputPort : public Port {
    using Port::Port;
  };

  struct OutputPort : public Port {
    using Port::Port;
  };

  struct Edge {
    Edge(OutputPort s, InputPort d) : src(s), dst(d) {}

    bool operator==(const Edge& other) const {
      return src == other.src && dst == other.dst;
    }

    template <typename H>
    friend H AbslHashValue(H h, const Edge& e) {
      return H::combine(std::move(h), e.src, e.dst);
    }

    OutputPort src;
    InputPort dst;
  };

  GraphDefT* graph() const { return graph_; }

  // Find a node by name or return `nullptr` if it's not in a graph view.
  NodeDefT* GetNode(absl::string_view node_name) const {
    return gtl::FindWithDefault(nodes_, node_name, nullptr);
  }

  // Get the specified input port. Note that the special '-1' port_id can be
  // used to access the controlling nodes (i.e. the nodes connected to node_name
  // through an incoming control dependency).
  InputPort GetInputPort(absl::string_view node_name, int port_id) const {
    return InputPort(GetNode(node_name), port_id);
  }

  // Get the specified output port. Note that the special '-1' port_id can be
  // used to access the controlled nodes (i.e. the nodes connected to node_name
  // through an outgoing control dependency).
  OutputPort GetOutputPort(absl::string_view node_name, int port_id) const {
    return OutputPort(GetNode(node_name), port_id);
  }

  // Get the input (resp. output) port(s) in the immediate fanout (resp. fanin)
  // of an output (resp. input) port.
  const absl::flat_hash_set<InputPort>& GetFanout(
      const OutputPort& port) const {
    return gtl::FindWithDefault(fanouts_, port, fanout_not_found_value_);
  }

  absl::flat_hash_set<OutputPort> GetFanin(const InputPort& port) const {
    if (port.port_id >= 0) return {GetRegularFanin(port)};

    // Collect fanin for the control input.
    absl::flat_hash_set<OutputPort> result;
    for (int i = port.node->input_size() - 1; i >= 0; --i) {
      TensorId tensor_id = ParseTensorName(port.node->input(i));
      if (tensor_id.index() >= 0) break;  // we reached regular inputs

      auto it = nodes_.find(tensor_id.node());
      if (it != nodes_.end()) result.emplace(it->second, tensor_id.index());
    }
    return result;
  }

  // Special case: regular (i.e. non-control) input ports can only have one
  // fanin.
  const OutputPort GetRegularFanin(const InputPort& port) const {
    DCHECK_GE(port.port_id, 0);
    if (port.port_id < 0) return OutputPort();

    TensorId tensor_id = ParseTensorName(port.node->input(port.port_id));
    return GetOutputPort(tensor_id.node(), tensor_id.index());
  }

  // Get all the input (resp. output) ports in the immediate fanout (resp
  // fanin) of a node. Include the controlling nodes iff
  // include_controlling_nodes is true.
  absl::flat_hash_set<InputPort> GetFanouts(
      const NodeDef& node, bool include_controlled_nodes) const {
    absl::flat_hash_set<InputPort> result;

    OutputPort port;
    port.node = const_cast<NodeDefT*>(&node);
    const int first_port_id = include_controlled_nodes ? -1 : 0;
    const int last_port_id =
        gtl::FindWithDefault(max_regular_output_port_, port.node, -1);

    for (int i = first_port_id; i <= last_port_id; ++i) {
      port.port_id = i;
      auto it = fanouts_.find(port);
      if (it != fanouts_.end()) {
        result.insert(it->second.begin(), it->second.end());
      }
    }
    return result;
  }

  absl::flat_hash_set<OutputPort> GetFanins(
      const NodeDef& node, bool include_controlling_nodes) const {
    absl::flat_hash_set<OutputPort> result;
    for (int i = 0; i < node.input_size(); ++i) {
      TensorId tensor_id = ParseTensorName(node.input(i));
      if (tensor_id.index() < 0 && !include_controlling_nodes) break;

      auto it = nodes_.find(tensor_id.node());
      if (it != nodes_.end()) result.emplace(it->second, tensor_id.index());
    }
    return result;
  }

  // Get the number of ports in the immediate fanin of a node. Count the
  // controlling nodes iff include_controlling_nodes is true.
  int NumFanins(const NodeDef& node, bool include_controlling_nodes) const {
    int count = 0;
    for (const string& input : node.input()) {
      if (!include_controlling_nodes && IsControlInput(input)) {
        break;
      }
      count += 1;
    }
    return count;
  }

  // Get the number of ports in the immediate fanout of a node. Count the
  // controlling nodes iff include_controlling_nodes is true.
  int NumFanouts(const NodeDef& node, bool include_controlling_nodes) const {
    int count = 0;

    OutputPort port;
    port.node = const_cast<NodeDefT*>(&node);
    const int first_port_id = include_controlling_nodes ? -1 : 0;
    const int last_port_id =
        gtl::FindWithDefault(max_regular_output_port_, port.node, -1);

    for (int i = first_port_id; i <= last_port_id; ++i) {
      port.port_id = i;
      auto it = fanouts_.find(port);
      if (it != fanouts_.end()) count += it->second.size();
    }

    return count;
  }

  // Get all the edges in the immediate fanout (resp fanin) of a node.
  // Include the control edges iff include_controlling_edges is true.
  absl::flat_hash_set<Edge> GetFanoutEdges(
      const NodeDef& node, bool include_controlled_edges) const {
    absl::flat_hash_set<Edge> result;

    OutputPort port;
    port.node = const_cast<NodeDefT*>(&node);
    const int first_port_id = include_controlled_edges ? -1 : 0;
    const int last_port_id =
        gtl::FindWithDefault(max_regular_output_port_, &node, -1);

    for (int i = first_port_id; i <= last_port_id; ++i) {
      port.port_id = i;
      auto it = fanouts_.find(port);
      if (it != fanouts_.end()) {
        for (auto itr = it->second.begin(); itr != it->second.end(); ++itr) {
          result.emplace(/*src*/ OutputPort(const_cast<NodeDefT*>(&node), i),
                         /*dst*/ *itr);
        }
      }
    }
    return result;
  }

  absl::flat_hash_set<Edge> GetFaninEdges(
      const NodeDef& node, bool include_controlling_edges) const {
    absl::flat_hash_set<Edge> result;
    for (int i = 0; i < node.input_size(); ++i) {
      TensorId tensor_id = ParseTensorName(node.input(i));
      if (tensor_id.index() < 0 && !include_controlling_edges) break;

      auto it = nodes_.find(tensor_id.node());
      if (it != nodes_.end()) {
        result.emplace(/*src*/ OutputPort(it->second, tensor_id.index()),
                       /*dst*/ InputPort(const_cast<NodeDefT*>(&node), i));
      }
    }
    return result;
  }

 protected:
  explicit GraphViewInternal(GraphDefT* graph) : graph_(graph) {}

  void AddUniqueNodeOrDie(NodeDefT* node) {
    auto result = nodes_.emplace(node->name(), node);
    // TODO(ezhulenev): Replace CHECK with factory method returning
    // absl::StatusOr (when available).
    CHECK(result.second) << "Non unique node name detected: " << node->name();
  }

  void AddFanouts(NodeDefT* node) {
    for (int i = 0; i < node->input_size(); ++i) {
      TensorId tensor_id = ParseTensorName(node->input(i));
      OutputPort output(nodes_[tensor_id.node()], tensor_id.index());

      if (output.port_id < 0) {
        fanouts_[output].emplace(node, -1);
      } else {
        max_regular_output_port_[output.node] =
            std::max(max_regular_output_port_[output.node], output.port_id);
        fanouts_[output].emplace(node, i);
      }
    }
  }

  // Access to the mutable internal state for MutableGraphView.
  absl::flat_hash_map<absl::string_view, NodeDefT*>& nodes() { return nodes_; }

  absl::flat_hash_map<OutputPort, absl::flat_hash_set<InputPort>>& fanouts() {
    return fanouts_;
  }

  absl::flat_hash_map<const NodeDef*, int>& max_regular_output_port() {
    return max_regular_output_port_;
  }

 private:
  GraphDefT* graph_;  // must outlive the graph view

  // A mapping from the node name to the node itself.
  absl::flat_hash_map<absl::string_view, NodeDefT*> nodes_;

  // A mapping from the output port to all inputs that read from it.
  absl::flat_hash_map<OutputPort, absl::flat_hash_set<InputPort>> fanouts_;

  // Keep a maximum index of tensor fetched from the node. It doesn't guarantee
  // that all tensors in the [0, max_regular_output_port] range are actually
  // fetched by other nodes.
  absl::flat_hash_map<const NodeDef*, int> max_regular_output_port_;

  // If the node has no fanouts at given output port (output tensor consumers)
  // we return a reference to this set from `GetFanout` (we can't construct new
  // empty set every time, because we need a non-dangling reference).
  absl::flat_hash_set<InputPort> fanout_not_found_value_;
};

}  // namespace internal

// Immutable GraphView that keeps the constness of the GraphDef. If you need to
// mutate the graph or the nodes via the graph view lookup functions, see
// MutableGraphView.
class GraphView
    : public internal::GraphViewInternal<const GraphDef, const NodeDef> {
 public:
  explicit GraphView(const GraphDef* graph) : GraphViewInternal(graph) {
    for (const NodeDef& node : graph->node()) AddUniqueNodeOrDie(&node);
    for (const NodeDef& node : graph->node()) AddFanouts(&node);
  }
};

// Returns true if node has one (or zero) fanout nodes at given output port.
bool HasSingleFanoutNode(const GraphView& graph_view, const NodeDef* node,
                         int port = 0);

// Returns true if node has at least one fanout node at given output port.
bool HasFanouts(const GraphView& graph_view, const NodeDef* node, int port = 0);

bool NoControlFanin(const GraphView& graph_view, const NodeDef* node);
bool NoControlFanout(const GraphView& graph_view, const NodeDef* node);
bool NoControlFaninOrFanout(const GraphView& graph_view, const NodeDef* node);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPH_VIEW_H_
