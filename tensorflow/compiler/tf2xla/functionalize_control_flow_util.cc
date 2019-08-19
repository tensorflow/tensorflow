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

#include "tensorflow/compiler/tf2xla/functionalize_control_flow_util.h"

#include "tensorflow/core/framework/node_def.pb.h"

namespace tensorflow {

bool NodeCmpByNameResourcesLast::operator()(const Node* lhs,
                                            const Node* rhs) const {
  bool lhs_is_resource =
      lhs->num_inputs() > 0 ? (lhs->input_type(0) == DT_RESOURCE) : false;
  bool rhs_is_resource =
      rhs->num_inputs() > 0 ? (rhs->input_type(0) == DT_RESOURCE) : false;
  return std::tie(lhs_is_resource, lhs->name()) <
         std::tie(rhs_is_resource, rhs->name());
}

xla::StatusOr<Node*> AddNodeDefToGraph(const NodeDef& node_def, Graph* graph) {
  Status status;
  Node* inserted_node = graph->AddNode(node_def, &status);
  if (!status.ok()) {
    return status;
  }
  return inserted_node;
}

xla::StatusOr<Node*> BuildRetvalNode(Graph* graph, DataType type, int index) {
  const char* const kRetValOp = "_Retval";
  NodeDef ret_def;
  ret_def.set_op(kRetValOp);
  ret_def.set_name(absl::StrCat(kRetValOp, index));
  AddNodeAttr("T", type, &ret_def);
  AddNodeAttr("index", index, &ret_def);
  return AddNodeDefToGraph(ret_def, graph);
}

Status ExtractWhileLoopFrames(
    const std::vector<ControlFlowInfo>& cf_info, const Graph* graph,
    std::unordered_map<string, WhileLoopFrame>* frames) {
  for (Node* node : graph->op_nodes()) {
    const ControlFlowInfo& cf = cf_info[node->id()];

    VLOG(2) << "node: " << node->name() << " (" << node->id()
            << ") frame_name: " << cf.frame_name
            << " frame: " << (cf.frame ? cf.frame->name() : "---")
            << " parent_frame: "
            << (cf.parent_frame ? cf.parent_frame->name() : "---");
    TF_RET_CHECK(cf.frame != nullptr && cf.parent_frame != nullptr);

    WhileLoopFrame& frame = (*frames)[cf.frame_name];
    WhileLoopFrame* parent =
        &(*frames)[cf_info[cf.parent_frame->id()].frame_name];
    if (frame.parent == nullptr) {
      frame.parent = parent;
      frame.name = cf.frame_name;
      ++parent->num_children;
    }

    if (IsEnter(node)) {
      WhileLoopArg arg;
      arg.enter = node;
      TF_RETURN_IF_ERROR(GetNodeAttr(arg.enter->attrs(), "is_constant",
                                     &arg.is_loop_invariant));
      frame.args.push_back(arg);
    } else if (IsLoopCond(node)) {
      frame.loop_cond = node;
    }
    frame.nodes.insert(node);
  }

  return Status::OK();
}

// Check that the graph has no cycle containing the given node.
Status CheckNodeNotInCycle(const Node* node, const int num_nodes) {
  std::vector<const Node*> ready;
  ready.push_back(node);
  std::vector<bool> visited(num_nodes);
  while (!ready.empty()) {
    const Node* current_node = ready.back();
    ready.pop_back();
    visited[current_node->id()] = true;
    for (const Edge* out : current_node->out_edges()) {
      if (out->dst() == node) {
        return errors::Internal("Detected a cycle: ", FormatNodeForError(*node),
                                " (", node->def().op(), ") feeds into itself.");
      } else if (!visited[out->dst()->id()]) {
        ready.push_back(out->dst());
      }
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
