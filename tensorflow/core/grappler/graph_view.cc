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
    CHECK(rslt.second) << "Non unique node name detected: " << node->name();
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
        num_regular_outputs_[fanin.node] =
            std::max(num_regular_outputs_[fanin.node], fanin.port_id);
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

std::unordered_set<GraphView::OutputPort, GraphView::HashPort>
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

std::unordered_set<GraphView::InputPort, GraphView::HashPort>
GraphView::GetFanouts(const NodeDef& node,
                      bool include_controlled_nodes) const {
  std::unordered_set<InputPort, HashPort> result;
  OutputPort port;
  port.node = const_cast<NodeDef*>(&node);
  const int first_port_id = include_controlled_nodes ? -1 : 0;
  auto it = num_regular_outputs_.find(&node);
  const int last_port_id = (it != num_regular_outputs_.end()) ? it->second : -1;

  for (int i = first_port_id; i <= last_port_id; ++i) {
    port.port_id = i;
    auto it = fanouts_.find(port);
    if (it != fanouts_.end()) {
      result.insert(it->second.begin(), it->second.end());
    }
  }
  return result;
}

std::unordered_set<GraphView::OutputPort, GraphView::HashPort>
GraphView::GetFanins(const NodeDef& node,
                     bool include_controlling_nodes) const {
  std::unordered_set<OutputPort, HashPort> result;
  for (int i = 0; i < node.input_size(); ++i) {
    OutputPort fanin;
    string fanin_name = ParseNodeName(node.input(i), &fanin.port_id);
    if (fanin.port_id < 0) {
      if (!include_controlling_nodes) {
        break;
      }
    }
    auto it = nodes_.find(fanin_name);
    if (it != nodes_.end()) {
      fanin.node = it->second;
      result.insert(fanin);
    }
  }
  return result;
}

int GraphView::NumFanins(const NodeDef& node,
                         bool include_controlling_nodes) const {
  int count = 0;
  for (const string& input : node.input()) {
    if (!include_controlling_nodes && IsControlInput(input)) {
      break;
    }
    count += 1;
  }
  return count;
}

std::unordered_set<GraphView::Edge, GraphView::HashEdge>
GraphView::GetFanoutEdges(const NodeDef& node,
                          bool include_controlled_edges) const {
  std::unordered_set<Edge, HashEdge> result;
  OutputPort port;
  port.node = const_cast<NodeDef*>(&node);
  const int first_port_id = include_controlled_edges ? -1 : 0;
  auto it = num_regular_outputs_.find(&node);
  const int last_port_id = (it != num_regular_outputs_.end()) ? it->second : -1;

  for (int i = first_port_id; i <= last_port_id; ++i) {
    port.port_id = i;
    auto it = fanouts_.find(port);
    if (it != fanouts_.end()) {
      Edge fanout;
      fanout.src.node = const_cast<NodeDef*>(&node);
      fanout.src.port_id = i;
      for (auto itr = it->second.begin(); itr != it->second.end(); ++itr) {
        fanout.tgt = *itr;
        result.insert(fanout);
      }
    }
  }
  return result;
}

std::unordered_set<GraphView::Edge, GraphView::HashEdge>
GraphView::GetFaninEdges(const NodeDef& node,
                         bool include_controlling_edges) const {
  std::unordered_set<Edge, HashEdge> result;
  for (int i = 0; i < node.input_size(); ++i) {
    Edge fanin;
    fanin.tgt.node = const_cast<NodeDef*>(&node);
    fanin.tgt.port_id = i;
    string fanin_name = ParseNodeName(node.input(i), &fanin.src.port_id);
    if (fanin.src.port_id < 0) {
      if (!include_controlling_edges) {
        break;
      }
    }
    auto it = nodes_.find(fanin_name);
    if (it != nodes_.end()) {
      fanin.src.node = it->second;
      result.insert(fanin);
    }
  }
  return result;
}

}  // end namespace grappler
}  // end namespace tensorflow
