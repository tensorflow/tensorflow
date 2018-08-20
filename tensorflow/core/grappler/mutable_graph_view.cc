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

#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

NodeDef* MutableGraphView::AddNode(NodeDef&& node) {
  auto* node_in_graph = GetGraph()->add_node();
  *node_in_graph = std::move(node);

  AddUniqueNodeOrDie(node_in_graph);

  AddFanouts(node_in_graph);
  return node_in_graph;
}

NodeDef* MutableGraphView::InsertNode(const NodeDef& input_node, NodeDef&& node,
                                      const int output_port_id) {
  auto* node_in_graph = GetGraph()->add_node();
  *node_in_graph = std::move(node);

  AddUniqueNodeOrDie(node_in_graph);

  // replace input for the output nodes of `input_node` with `node`
  ReplaceInput(input_node, *node_in_graph, output_port_id);

  AddFanouts(node_in_graph);
  return node_in_graph;
}

void MutableGraphView::ReplaceInput(const NodeDef& old_input,
                                    const NodeDef& new_input,
                                    const int output_port_id) {
  GraphView::OutputPort output_port =
      GetOutputPort(old_input.name(), output_port_id);
  auto fanout = GetFanout(output_port);
  for (auto& input_port : fanout) {
    input_port.node->set_input(input_port.port_id, new_input.name());
    AddFanouts(input_port.node);
  }
}

void MutableGraphView::DeleteNodes(const std::set<string>& nodes_to_delete) {
  for (const string& node_name_to_delete : nodes_to_delete)
    RemoveFanouts(MutableNodes()->at(node_name_to_delete));
  for (const string& node_name_to_delete : nodes_to_delete)
    MutableNodes()->erase(node_name_to_delete);
  EraseNodesFromGraph(nodes_to_delete, GetGraph());
}

void MutableGraphView::RemoveFanouts(NodeDef* node) {
  for (int i = 0; i < node->input_size(); ++i) {
    OutputPort fanin;
    string fanin_name = ParseNodeName(node->input(i), &fanin.port_id);
    fanin.node = (*MutableNodes())[fanin_name];

    InputPort input;
    input.node = node;
    if (fanin.port_id < 0)
      input.port_id = -1;
    else
      input.port_id = i;

    (*MutableFanouts())[fanin].erase(input);
  }
}

}  // end namespace grappler
}  // end namespace tensorflow
