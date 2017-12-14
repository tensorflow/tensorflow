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

#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

GraphView::GraphView(GraphDef* graph) : graph_(graph) {
  for (int i = 0; i < graph_->node_size(); i++) {
    auto node = graph_->mutable_node(i);
    auto rslt = nodes_.insert(std::make_pair(node->name(), node));
    // Check that the graph doesn't contain multiple nodes with the same name.
    CHECK(rslt.second);
  }
  for (NodeDef& node : *graph_->mutable_node()) {
    for (int i = 0; i < node.input_size(); ++i) {
      OutputPort fanin;
      string fanin_name = ParseNodeName(node.input(i), &fanin.port_id);
      fanin.node = nodes_[fanin_name];

      InputPort input;
      input.node = &node;
      if (fanin.port_id < 0) {
        input.port_id = -1;
      } else {
        input.port_id = i;
      }

      fanouts_[fanin].insert(input);
    }
  }
}

NodeDef* GraphView::GetNode(const string& node_name) const {
  auto it = nodes_.find(node_name);
  if (it == nodes_.end()) {
    return nullptr;
  }
  return it->second;
}

GraphView::InputPort GraphView::GetInputPort(const string& node_name,
                                             int port_id) const {
  InputPort result;
  result.node = GetNode(node_name);
  // TODO(bsteiner): verify that the node has at least port_id input ports
  result.port_id = port_id;
  return result;
}

GraphView::OutputPort GraphView::GetOutputPort(const string& node_name,
                                               int port_id) const {
  OutputPort result;
  result.node = GetNode(node_name);
  // TODO(bsteiner): verify that the node has at least port_id output ports
  result.port_id = port_id;
  return result;
}

const std::unordered_set<GraphView::InputPort, GraphView::HashPort>&
GraphView::GetFanout(const GraphView::OutputPort& port) const {
  auto it = fanouts_.find(port);
  if (it == fanouts_.end()) {
    return empty_set_;
  }
  return it->second;
}

const std::unordered_set<GraphView::OutputPort, GraphView::HashPort>
GraphView::GetFanin(const GraphView::InputPort& port) const {
  std::unordered_set<GraphView::OutputPort, GraphView::HashPort> result;
  if (port.port_id >= 0) {
    result.insert(GetRegularFanin(port));
  } else {
    for (int i = port.node->input_size() - 1; i >= 0; --i) {
      OutputPort fanin;
      string fanin_name = ParseNodeName(port.node->input(i), &fanin.port_id);
      if (fanin.port_id < 0) {
        auto it = nodes_.find(fanin_name);
        if (it != nodes_.end()) {
          fanin.node = it->second;
          result.insert(fanin);
        }
      } else {
        break;
      }
    }
  }
  return result;
}

const GraphView::OutputPort GraphView::GetRegularFanin(
    const GraphView::InputPort& port) const {
  CHECK_LE(0, port.port_id);
  OutputPort fanin;
  string fanin_name =
      ParseNodeName(port.node->input(port.port_id), &fanin.port_id);
  auto it = nodes_.find(fanin_name);
  if (it == nodes_.end()) {
    fanin.node = nullptr;
  } else {
    fanin.node = it->second;
  }
  return fanin;
}

}  // end namespace grappler
}  // end namespace tensorflow
