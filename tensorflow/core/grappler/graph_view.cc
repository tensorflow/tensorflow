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

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

int OpOutputPortIdToArgId(const NodeDef& node, const OpDef& op, int port_id) {
  return OpPortIdToArgId(node, op.output_arg(), port_id);
}

int OpInputPortIdToArgId(const NodeDef& node, const OpDef& op, int port_id) {
  return OpPortIdToArgId(node, op.input_arg(), port_id);
}

bool HasSingleFanoutNode(const GraphView& graph_view, const NodeDef* node,
                         int port) {
  const auto output = GraphView::OutputPort(node, port);
  return graph_view.GetFanout(output).size() <= 1;
}

bool HasFanouts(const GraphView& graph_view, const NodeDef* node, int port) {
  const auto output = GraphView::OutputPort(node, port);
  return !graph_view.GetFanout(output).empty();
}

bool HasControlFanin(const GraphView& graph_view, const NodeDef* node) {
  const auto control_port = GraphView::InputPort(node, Graph::kControlSlot);
  return !graph_view.GetFanin(control_port).empty();
}

bool HasControlFanout(const GraphView& graph_view, const NodeDef* node) {
  const auto control_port = GraphView::OutputPort(node, Graph::kControlSlot);
  return !graph_view.GetFanout(control_port).empty();
}

bool HasControlFaninOrFanout(const GraphView& graph_view, const NodeDef* node) {
  return HasControlFanin(graph_view, node) ||
         HasControlFanout(graph_view, node);
}

}  // end namespace grappler
}  // end namespace tensorflow
