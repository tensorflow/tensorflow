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

#ifndef TENSORFLOW_CORE_GRAPH_CONTROL_FLOW_H_
#define TENSORFLOW_CORE_GRAPH_CONTROL_FLOW_H_

#include <vector>

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Control flow info for a graph node.
struct ControlFlowInfo {
  // 'frame' and 'parent_frame' are pointers to:
  //
  // a) One of the Enter nodes corresponding to the loop body, if the node
  //    executes inside a loop. If multiple tensors enter the while loop, it's
  //    undefined which Enter node will be used.
  //
  // b) SOURCE node (node.id() == Graph::kSourceId), if the node is not inside
  //    any of the while loops.

  const Node* frame = nullptr;         // frame of a node
  const Node* parent_frame = nullptr;  // parent frame of a node
  string frame_name;                   // frame name of a node
};

// Clear and populate `info` with each node's frame and the level it belongs to.
// We check the well-formedness of the graph:
// 1) All inputs to a node must come from the same frame and have the same
//    "static" iteration level.
// 2) Each frame has at most one LoopCond node.
// 3) Each frame has a single parent frame.
// If `unreachable_nodes` is set, return names of nodes unreachable from the
// source node. We cannot build ControlFlowInfo for such nodes. They might be
// pruned later.
//
// NOTE(yuanbyu): For now, we require all sends/recvs have iteration level 0.
// This essentially means there can't be multiple serial Nexts in an iteration,
// which all sane front-ends should satisfy.
Status BuildControlFlowInfo(const Graph* g, std::vector<ControlFlowInfo>* info,
                            std::vector<string>* unreachable_nodes = nullptr);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_CONTROL_FLOW_H_
