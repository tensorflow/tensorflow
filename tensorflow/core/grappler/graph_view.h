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
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {

// A utility class to simplify the traversal of a GraphDef.
class GraphView {
 public:
  struct Port {
    Port() = default;
    Port(NodeDef* n, int port) : node(n), port_id(port) {}

    // TODO(prazek): ports should keep the constness of GraphView.  The only way
    // to modify graph through the view should be using MutableGraphView.
    NodeDef* node = nullptr;
    int port_id = -1;

    bool operator==(const Port& other) const {
      return node == other.node && port_id == other.port_id;
    }
  };
  struct InputPort : public Port {
    InputPort() = default;
    InputPort(NodeDef* n, int port_id) : Port(n, port_id) {}
    InputPort(const NodeDef* n, int port_id)
        : Port(const_cast<NodeDef*>(n), port_id) {}
  };
  struct OutputPort : public Port {
    OutputPort() = default;
    OutputPort(NodeDef* n, int port_id) : Port(n, port_id) {}
  };

  struct HashPort {
    std::size_t operator()(const Port& port) const {
      return reinterpret_cast<std::size_t>(port.node) + port.port_id;
    }
  };

  struct Edge {
    OutputPort src;
    InputPort tgt;

    bool operator==(const Edge& other) const {
      return src == other.src && tgt == other.tgt;
    }
  };
  struct HashEdge {
    std::size_t operator()(const Edge& edge) const {
      return HashPort()(edge.src) + HashPort()(edge.tgt);
    }
  };

  explicit GraphView(GraphDef* graph);
  GraphDef* GetGraph() const { return graph_; }
  NodeDef* GetNode(const string& node_name) const;
  // Get the specified input port. Note that the special '-1' port_id can be
  // used to access the controlling nodes (i.e. the nodes connected to node_name
  // through an incoming control dependency).
  InputPort GetInputPort(const string& node_name, int port_id) const;
  // Get the specified output port. Note that the special '-1' port_id can be
  // used to access the controlled nodes (i.e. the nodes connected to node_name
  // through an outgoing control dependency).
  OutputPort GetOutputPort(const string& node_name, int port_id) const;

  // Get the input (resp. output) port(s) in the immediate fanout (resp. fanin)
  // of an output (resp. input) port.
  const std::unordered_set<InputPort, HashPort>& GetFanout(
      const OutputPort& port) const;
  std::unordered_set<OutputPort, HashPort> GetFanin(
      const InputPort& port) const;

  // Special case: regular (i.e. non-control) input ports can only have one
  // fanin.
  const OutputPort GetRegularFanin(const InputPort& port) const;

  // Get all the input (resp. output) ports in the immediate fanout (resp fanin)
  // of a node. Include the controlling nodes iff include_controlling_nodes is
  // true.
  std::unordered_set<InputPort, HashPort> GetFanouts(
      const NodeDef& node, bool include_controlled_nodes) const;
  std::unordered_set<OutputPort, HashPort> GetFanins(
      const NodeDef& node, bool include_controlling_nodes) const;

  // Get the number of ports in the immediate fanin of a node. Count the
  // controlling nodes iff include_controlling_nodes is true.
  int NumFanins(const NodeDef& node, bool include_controlling_nodes) const;

  // Get all the edge in the immediate fanout (resp fanin) of a node. Include
  // the control edges iff include_controlling_edges is true.
  std::unordered_set<Edge, HashEdge> GetFanoutEdges(
      const NodeDef& node, bool include_controlled_edges) const;
  std::unordered_set<Edge, HashEdge> GetFaninEdges(
      const NodeDef& node, bool include_controlling_edges) const;

 protected:
  // Add fanout to every `node` input.
  void AddFanouts(NodeDef* node);
  std::unordered_map<string, NodeDef*>* MutableNodes() { return &nodes_; }
  GraphDef* MutableGraph() { return graph_; }

  using FanoutsMapType =
      std::unordered_map<OutputPort, std::unordered_set<InputPort, HashPort>,
                         HashPort>;
  FanoutsMapType* MutableFanouts() { return &fanouts_; }

 private:
  GraphDef* graph_;
  std::unordered_map<string, NodeDef*> nodes_;
  std::unordered_set<InputPort, HashPort> empty_set_;
  FanoutsMapType fanouts_;
  std::unordered_map<const NodeDef*, int> num_regular_outputs_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPH_VIEW_H_
