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

#include "tensorflow/core/grappler/utils/frame.h"
#include <deque>
#include <stack>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

int IdentifyFrames(
    const GraphDef& graph,
    std::unordered_map<const NodeDef*, std::vector<int>>* frames) {
  NodeMap node_map(const_cast<GraphDef*>(&graph));
  std::deque<std::pair<const NodeDef*, std::vector<int>>> ready_nodes;
  for (const NodeDef& node : graph.node()) {
    if (node.input_size() == 0) {
      std::vector<int> empty;
      ready_nodes.emplace_back(&node, empty);
      (*frames)[&node] = empty;
    }
  }
  std::map<string, int> name_to_id;
  while (!ready_nodes.empty()) {
    auto ready_node = ready_nodes.front();
    for (const auto& fanout : node_map.GetOutputs(ready_node.first->name())) {
      if (frames->count(fanout) < 1) {
        std::vector<int> frame_ids = ready_node.second;
        if (IsExit(*ready_node.first)) {
          frame_ids.pop_back();
        }
        if (IsEnter(*fanout)) {
          CHECK(fanout->attr().count("frame_name"))
              << "Missing frame name for the Enter node " << fanout->name();
          string name = fanout->attr().at("frame_name").s();
          int id;
          if (name_to_id.count(name)) {
            id = name_to_id[name];
          } else {
            id = name_to_id.size();
            name_to_id[name] = id;
          }
          frame_ids.push_back(id);
        }
        ready_nodes.emplace_back(fanout, frame_ids);
        (*frames)[fanout] = frame_ids;
      } else {
        auto frame_ids_fanout = (*frames)[fanout];
        auto frame_ids_node = ready_node.second;
        if (IsEnter(*fanout)) {
          frame_ids_fanout.pop_back();
        }
        if (IsExit(*ready_node.first)) {
          frame_ids_node.pop_back();
        }
        CHECK(frame_ids_node == frame_ids_fanout);
      }
    }
    ready_nodes.pop_front();
  }
  return name_to_id.size();
}

}  // namespace grappler
}  // namespace tensorflow
