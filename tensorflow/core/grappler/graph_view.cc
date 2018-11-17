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
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

namespace {
int OpPortIdToArgId(const NodeDef& node,
                    const protobuf::RepeatedPtrField<OpDef::ArgDef>& args,
                    int port_id) {
  for (int arg_id = 0; arg_id < args.size(); ++arg_id) {
    if (port_id < 0) {
      return -1;
    } else if (port_id == 0) {
      return arg_id;
    }

    // Default is 1 port per arg.
    int n = 1;

    const auto& arg = args.Get(arg_id);
    if (!arg.number_attr().empty()) {
      n = node.attr().at(arg.number_attr()).i();
    } else if (!arg.type_list_attr().empty()) {
      n = node.attr().at(arg.type_list_attr()).list().type_size();
    }

    if (n < 0) {
      // This should never happen.
      DCHECK_GE(n, 0);
      return -1;
    } else if (port_id < n) {
      return arg_id;
    }
    port_id -= n;
  }

  return -1;
}
}  // end namespace

int OpOutputPortIdToArgId(const NodeDef& node, const OpDef& op, int port_id) {
  return OpPortIdToArgId(node, op.output_arg(), port_id);
}

int OpInputPortIdToArgId(const NodeDef& node, const OpDef& op, int port_id) {
  return OpPortIdToArgId(node, op.input_arg(), port_id);
}

bool HasSingleFanoutNode(const GraphView& graph_view, const NodeDef* node,
                         int port) {
  const auto output = GraphView::OutputPort(node, port);
  const auto fanout = graph_view.GetFanout(output);
  return fanout.size() <= 1;
}

bool HasFanouts(const GraphView& graph_view, const NodeDef* node, int port) {
  const auto output = GraphView::OutputPort(node, port);
  const auto fanout = graph_view.GetFanout(output);
  return !fanout.empty();
}

bool NoControlFanin(const GraphView& graph_view, const NodeDef* node) {
  const auto control_port = GraphView::InputPort(node, -1);
  return graph_view.GetFanin(control_port).empty();
}

bool NoControlFanout(const GraphView& graph_view, const NodeDef* node) {
  const auto control_port = GraphView::OutputPort(node, -1);
  return graph_view.GetFanout(control_port).empty();
}

bool NoControlFaninOrFanout(const GraphView& graph_view, const NodeDef* node) {
  return NoControlFanin(graph_view, node) && NoControlFanout(graph_view, node);
}

}  // end namespace grappler
}  // end namespace tensorflow
