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

#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_FRAME_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_FRAME_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

// FrameView is a helper class that allows to find in what execution frames (if
// any) the given node can be running in. It's constructed from an immutable
// GraphView, and any modification of the underlying graph might invalidate it.
//
// All execution frames assigned a unique integer id, but they do not have any
// meaning whatsoever, it's just a sequence number.
//
// See the paper "Dynamic Control Flow in Large-Scale Machine Learning" for
// detailed explanation of execution frames (https://arxiv.org/abs/1805.01772).
class FrameView {
 public:
  FrameView() : is_inferred_(false), num_frames_(0) {}

  // Infers nodes execution frames from the GraphView. Returns an error if
  // called multiple times.
  Status InferFromGraphView(const utils::GraphView& graph_view);
  // Infers nodes execution frames from the MutableGraphView. Returns an error
  // if called multiple times.
  Status InferFromGraphView(const utils::MutableGraphView& graph_view);
  // Infers nodes execution by constructing temporary GraphView and passing it
  // to InferFromGraphView.
  Status InferFromGraph(const GraphDef& graph);

  // Returns all frames of the given node (denoted by their frame ids) in
  // outermost-to-innermost order.
  const std::vector<int>& Frames(const NodeDef& node) const;

  // Returns true iff the node is at least in one execution frame.
  bool IsInFrame(const NodeDef& node) const;

  int num_frames() const { return num_frames_; }
  bool is_inferred() const { return is_inferred_; }

 private:
  template <typename GraphViewT>
  inline Status InferFromGraphViewT(const GraphViewT& graph_view);

  bool is_inferred_;  // true if it was inferred from the graph
  int num_frames_;    // number of frames present in a graph
  absl::flat_hash_map<const NodeDef*, std::vector<int>> node_to_frames_;

  // We return a reference to this vector if node has no frames.
  const std::vector<int> node_has_no_frames_;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_FRAME_H_
