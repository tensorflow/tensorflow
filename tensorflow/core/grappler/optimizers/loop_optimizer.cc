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

#include "tensorflow/core/grappler/optimizers/loop_optimizer.h"

#include <unordered_map>
#include <unordered_set>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace grappler {
namespace {

std::vector<int> GetStackPushNodesToConvert(const SimpleGraphView& graph_view,
                                            int stack_node_idx) {
  VLOG(1) << "Stack node: " << graph_view.graph()->node(stack_node_idx).name();
  const std::unordered_set<string> op_types_to_traverse(
      {"Stack", "StackV2", "Enter", "RefEnter", "Switch", "RefSwitch",
       "Identity", "RefIdentity"});
  std::vector<int> nodes_to_convert;
  std::set<int> fanout;
  graph_view.DepthFirstSearch(op_types_to_traverse, stack_node_idx, &fanout);
  for (int fanout_idx : fanout) {
    const NodeDef& fanout_node = graph_view.graph()->node(fanout_idx);
    VLOG(1) << "Fanout " << fanout_idx << " : " << fanout_node.name();
    if (IsStackPushOp(fanout_node)) {
      nodes_to_convert.push_back(fanout_idx);
    } else if (IsStackOp(fanout_node) || IsStackCloseOp(fanout_node) ||
               op_types_to_traverse.find(fanout_node.op()) !=
                   op_types_to_traverse.end()) {
      continue;
    } else if (!IsStackPopOp(fanout_node) ||
               !graph_view.outputs(fanout_idx).empty()) {
      // The node is either a stack pop with consumers or something unexpected
      // so we leave the graph alone.
      nodes_to_convert.clear();
      break;
    }
  }
  return nodes_to_convert;
}

Status RemoveStackOps(const GraphDef& graph, GraphDef* optimized_graph) {
  *optimized_graph = graph;
  NodeMap node_map(optimized_graph);
  SimpleGraphView graph_view;
  TF_RETURN_IF_ERROR(graph_view.Initialize(graph));
  for (int node_idx = 0; node_idx < graph.node_size(); ++node_idx) {
    if (IsStackOp(graph.node(node_idx))) {
      for (int push_node_idx :
           GetStackPushNodesToConvert(graph_view, node_idx)) {
        // We found push nodes without corresponding pops. Convert them to
        // Identity passing the data through and add a control dependency from
        // the op supplying the stack handle.
        NodeDef* push_node = optimized_graph->mutable_node(push_node_idx);
        VLOG(1) << "Converting " << push_node_idx << " : "
                << push_node->DebugString();
        if (push_node->attr().count("swap_memory") != 0) {
          push_node->mutable_attr()->erase("swap_memory");
        }
        push_node->set_op("Identity");
        push_node->mutable_input()->SwapElements(0, 1);
        const string ctrl_dep = ConstantFolding::AddControlDependency(
            push_node->input(1), optimized_graph, &node_map);
        push_node->set_input(1, ctrl_dep);
        VLOG(1) << "After converting: " << push_node->DebugString();
      }
    }
  }
  return Status::OK();
}

}  // namespace

Status LoopOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  Status status = RemoveStackOps(item.graph, optimized_graph);
  return status;
}

void LoopOptimizer::Feedback(Cluster* /*cluster*/, const GrapplerItem& /*item*/,
                             const GraphDef& /*optimized_graph*/,
                             double /*result*/) {
  // Nothing to do for LoopOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow
