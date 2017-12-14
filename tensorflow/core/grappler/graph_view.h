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

#ifndef TENSORFLOW_GRAPPLER_GRAPH_VIEW_H_
#define TENSORFLOW_GRAPPLER_GRAPH_VIEW_H_

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
    NodeDef* node;
    int port_id;

    bool operator==(const Port& other) const {
      return node == other.node && port_id == other.port_id;
    }
  };
  struct InputPort : public Port {};
  struct OutputPort : public Port {};

  struct HashPort {
    std::size_t operator()(const Port& port) const {
      return reinterpret_cast<std::size_t>(port.node) + port.port_id;
    }
  };

  explicit GraphView(GraphDef* graph);
  NodeDef* GetNode(const string& node_name) const;
  // Get the specified input port. Note that the special '-1' port_id can be
  // used to access the controlling nodes (i.e. the nodes connected to node_name
  // through an incoming control dependency).
  InputPort GetInputPort(const string& node_name, int port_id) const;
  // Get the specified input port. Note that the special '-1' port_id can be
  // used to access the controlled nodes (i.e. the nodes connected to node_name
  // through an outgoing control dependency).

  // Special case: regular (i.e. non-control) ports can only have one fanin.
  OutputPort GetOutputPort(const string& node_name, int port_id) const;

  const std::unordered_set<InputPort, HashPort>& GetFanout(
      const OutputPort& port) const;
  const std::unordered_set<OutputPort, HashPort> GetFanin(
      const InputPort& port) const;
  const OutputPort GetRegularFanin(const InputPort& port) const;

 private:
  GraphDef* graph_;
  std::unordered_map<string, NodeDef*> nodes_;
  std::unordered_set<InputPort, HashPort> empty_set_;
  std::unordered_map<OutputPort, std::unordered_set<InputPort, HashPort>,
                     HashPort>
      fanouts_;
  std::unordered_map<NodeDef*, std::unordered_set<NodeDef*>> controlled_nodes_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_GRAPH_VIEW_H_
