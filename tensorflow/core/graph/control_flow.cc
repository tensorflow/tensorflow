/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/control_flow.h"

#include <deque>
#include <vector>

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {
// Information about a loop frame structure.
struct Frame {
  string name;

  // Pointer to the parent frame. The root frame has a pointer to itself.
  Frame* parent = nullptr;

  // The loop condition of the loop. There should be exactly one loop condition
  // in every loop.
  const Node* loop_cond = nullptr;
};

// Verify that the ControlFlowInfo of the graph has valid loop structure.
Status ValidateControlFlowInfo(const Graph* graph,
                               const std::vector<ControlFlowInfo>& cf_info) {
  std::unordered_map<string, Frame> frames;
  for (const Node* node : graph->op_nodes()) {
    const ControlFlowInfo& cf = cf_info[node->id()];
    if (!cf.frame || !cf.parent_frame) {
      // Skip nodes unreachable from the source node. They might be pruned
      // later.
      continue;
    }

    Frame& frame = frames[cf.frame_name];
    Frame* parent = &frames[cf_info[cf.parent_frame->id()].frame_name];
    if (frame.parent == nullptr) {
      frame.parent = parent;
      frame.name = cf.frame_name;
    } else if (frame.parent != parent) {
      return errors::Internal(
          "Invalid loop structure: Mismatched parent frames for \"",
          cf.frame_name, "\": \"", parent->name, "\" vs \"", frame.parent->name,
          "\". The node giving this error: ", FormatNodeForError(*node),
          ". This is an internal bug, please file a bug report with "
          "instructions on how to reproduce the error.");
    }
    if (IsLoopCond(node)) {
      // ForwardLoopCounter runs in the same frame as the forward loop and
      // BackPropLoopCounter runs in the same frame as the backprop loop. They
      // are the only cases that multiple loops share the same frame.
      if (frame.loop_cond &&
          !absl::StrContains(frame.loop_cond->name(), "LoopCounter") &&
          !absl::StrContains(node->name(), "LoopCounter")) {
        return errors::InvalidArgument(
            "Invalid loop structure: Loop \"", cf.frame_name,
            "\" has more than one LoopCond node: ", FormatNodeForError(*node),
            " and ", FormatNodeForError(*frame.loop_cond),
            ". This is an internal bug, please file a bug report with "
            "instructions on how to reproduce the error.");
      }
      frame.loop_cond = node;
    }
  }
  return OkStatus();
}
}  // namespace

Status BuildControlFlowInfo(const Graph* g, std::vector<ControlFlowInfo>* info,
                            std::vector<string>* unreachable_nodes) {
  info->clear();
  info->resize(g->num_node_ids());

  std::vector<const Node*> parent_nodes;
  parent_nodes.resize(g->num_node_ids());

  const Node* src_node = g->source_node();
  ControlFlowInfo& src_info = (*info)[src_node->id()];
  src_info.frame = src_node;
  src_info.parent_frame = src_node;

  string frame_name;
  std::deque<const Node*> ready;
  ready.push_back(src_node);
  while (!ready.empty()) {
    const Node* curr_node = ready.front();
    ready.pop_front();
    const ControlFlowInfo& curr_info = (*info)[curr_node->id()];
    const Node* frame = curr_info.frame;
    const Node* parent = curr_info.parent_frame;
    frame_name = curr_info.frame_name;

    if (IsExit(curr_node)) {
      // Exit to the parent frame.
      const ControlFlowInfo& parent_info = (*info)[parent->id()];
      frame = parent_info.frame;
      parent = parent_info.parent_frame;
      frame_name = parent_info.frame_name;
    }

    for (const Edge* out_edge : curr_node->out_edges()) {
      const Node* out = out_edge->dst();
      int out_id = out->id();
      ControlFlowInfo* out_info = &(*info)[out_id];
      const Node* out_parent = out_info->parent_frame;
      bool is_visited = (parent_nodes[out_id] != nullptr);

      // Skip Sink/Source nodes.
      if (!out->IsOp()) continue;

      // Add to ready queue if not seen.
      if (!is_visited) {
        parent_nodes[out->id()] = curr_node;
        ready.push_back(out);
      }

      // Process the node 'out'.
      if (IsEnter(out)) {
        if (is_visited) {
          const string& parent_frame = (*info)[out_parent->id()].frame_name;
          if (parent_frame != frame_name) {
            return errors::InvalidArgument(
                FormatNodeForError(*out),
                " has inputs from different frames. The input ",
                FormatNodeForError(*curr_node), " is in frame '", frame_name,
                "'. The input ", FormatNodeForError(*parent_nodes[out->id()]),
                " is in frame '", parent_frame, "'.");
          }
        } else {
          out_info->frame = out;
          out_info->parent_frame = frame;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(out->attrs(), "frame_name", &out_info->frame_name));
          if (out_info->frame_name.empty()) {
            return errors::InvalidArgument("The Enter ",
                                           FormatNodeForError(*out),
                                           " must have a frame name.");
          }
        }
      } else {
        if (is_visited) {
          if (out_info->frame_name != frame_name) {
            return errors::InvalidArgument(
                FormatNodeForError(*out),
                " has inputs from different frames. The input ",
                FormatNodeForError(*curr_node), " is in frame '", frame_name,
                "'. The input ", FormatNodeForError(*parent_nodes[out->id()]),
                " is in frame '", out_info->frame_name, "'.");
          }
        } else {
          out_info->frame = frame;
          out_info->parent_frame = parent;
          out_info->frame_name = frame_name;
        }
      }
    }
  }
  if (unreachable_nodes) {
    for (const Node* node : g->op_nodes()) {
      if (!parent_nodes[node->id()]) {
        unreachable_nodes->push_back(node->name());
      }
    }
  }
  TF_RETURN_IF_ERROR(ValidateControlFlowInfo(g, *info));
  return OkStatus();
}

}  // namespace tensorflow
