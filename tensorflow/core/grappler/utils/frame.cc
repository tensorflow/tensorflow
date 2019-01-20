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
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {

namespace {}  // namespace

Status FrameView::InferFromGraphView(const GraphView& graph_view) {
  if (is_inferred_) {
    return errors::Internal("FrameView was already inferred from the graph");
  }
  is_inferred_ = true;

  std::deque<const NodeDef*> ready_nodes;

  // All nodes without inputs are automatically added to the ready queue.
  for (const NodeDef& node : graph_view.graph()->node()) {
    if (node.input_size() == 0) {
      ready_nodes.push_back(&node);
      node_to_frames_[&node] = node_has_no_frames_;
    }
  }

  // We assign unique int id to each frame, and use this map to track what
  // frames we've already seen in the graph.
  absl::flat_hash_map<string, int> frame_name_to_id;

  while (!ready_nodes.empty()) {
    const NodeDef* ready_node = ready_nodes.front();

    absl::flat_hash_set<GraphView::InputPort> fanouts =
        graph_view.GetFanouts(*ready_node, /*include_controlled_nodes=*/true);

    for (const GraphView::InputPort& fanout : fanouts) {
      if (node_to_frames_.count(fanout.node) < 1) {
        // If we have never seen this node before, we add all frames from the
        // incoming node (and pop/push frames if coming from Exit/Enter nodes).
        std::vector<int> frame_ids = node_to_frames_[ready_node];

        if (IsExit(*ready_node)) {
          frame_ids.pop_back();
        }

        if (IsEnter(*fanout.node)) {
          const AttrValue* frame_name_attr =
              AttrSlice(*fanout.node).Find("frame_name");

          if (!frame_name_attr) {
            return errors::InvalidArgument(
                "Missing frame name for the Enter node: ",
                SummarizeNodeDef(*fanout.node));
          }

          absl::string_view frame_name = frame_name_attr->s();
          int frame_id;

          if (frame_name_to_id.count(frame_name)) {
            frame_id = frame_name_to_id[frame_name];
          } else {
            frame_id = static_cast<int>(frame_name_to_id.size());
            frame_name_to_id[frame_name] = frame_id;
          }

          frame_ids.push_back(frame_id);
        }

        ready_nodes.push_back(fanout.node);
        node_to_frames_[fanout.node] = std::move(frame_ids);

      } else {
        // If we've already seen this node before, we need to make sure that
        // graph is correct and same nodes doesn't have incoming edges with
        // conflicting frames (all inputs must be produces in the same frame).

        std::vector<int> frame_ids_fanout = node_to_frames_[fanout.node];
        std::vector<int> frame_ids_node = node_to_frames_[ready_node];

        if (IsEnter(*fanout.node)) {
          frame_ids_fanout.pop_back();
        }
        if (IsExit(*ready_node)) {
          frame_ids_node.pop_back();
        }

        if (frame_ids_node != frame_ids_fanout) {
          return errors::InvalidArgument(
              "Invalid graph: Frame ids for node ", ready_node->name(),
              " does not match frame ids for it's fanout ",
              fanout.node->name());
        }
      }
    }

    ready_nodes.pop_front();
  }

  num_frames_ = static_cast<int>(frame_name_to_id.size());
  return Status::OK();
}

Status FrameView::InferFromGraph(const GraphDef& graph) {
  return InferFromGraphView(GraphView(&graph));
}

const std::vector<int>& FrameView::Frames(const NodeDef& node) const {
  DCHECK(is_inferred_) << "FrameView is not initialized";
  auto frames = node_to_frames_.find(&node);
  if (frames == node_to_frames_.end()) {
    LOG(WARNING) << "Node doesn't belong to the graph used for initialization";
    return node_has_no_frames_;
  } else {
    return frames->second;
  }
}

bool FrameView::IsInFrame(const NodeDef& node) const {
  return !Frames(node).empty();
}

}  // namespace grappler
}  // namespace tensorflow
