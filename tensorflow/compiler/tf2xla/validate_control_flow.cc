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

#include "tensorflow/compiler/tf2xla/validate_control_flow.h"

#include <vector>

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
      return errors::InvalidArgument(
          "Invalid loop structure: Mismatched parent frames for \"",
          cf.frame_name, "\": \"", parent->name, "\" vs \"", frame.parent->name,
          "\". This is an internal bug, please file a bug report with "
          "instructions on how to reproduce the error.");
    }
    if (IsLoopCond(node)) {
      if (frame.loop_cond) {
        return errors::InvalidArgument(
            "Invalid loop structure: Loop \"", cf.frame_name,
            "\" has more than one LoopCond node: \"", node->name(), "\" and \"",
            frame.loop_cond->name(),
            "\". This is an internal bug, please file a bug report with "
            "instructions on how to reproduce the error.");
      }
      frame.loop_cond = node;
    }
  }
  return Status::OK();
}
}  // namespace

Status BuildAndValidateControlFlowInfo(const Graph* graph,
                                       std::vector<ControlFlowInfo>* info,
                                       std::vector<string>* unreachable_nodes) {
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph, info, unreachable_nodes));
  return ValidateControlFlowInfo(graph, *info);
}

}  // namespace tensorflow
