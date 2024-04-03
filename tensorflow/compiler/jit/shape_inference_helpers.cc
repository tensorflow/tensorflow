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

// Contains helpers for use in shape inference.

#include "tensorflow/compiler/jit/shape_inference_helpers.h"

#include <vector>

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

Status BackEdgeHelper::Remove(Graph* graph) {
  if (graph_ != nullptr) {
    return errors::Internal("BackEdgeHelper duplicate call to Remove.");
  }
  graph_ = graph;
  for (Node* n : graph_->nodes()) {
    if (n->IsMerge()) {
      for (const Edge* e : n->in_edges()) {
        if (e->src()->IsNextIteration()) {
          back_edges_.push_back(
              BackEdge{e, e->src(), e->src_output(), e->dst(), e->dst_input()});
        }
      }
    }
  }
  for (const BackEdge& be : back_edges_) {
    graph_->RemoveEdge(be.edge);
  }
  return absl::OkStatus();
}

const std::vector<BackEdgeHelper::BackEdge>& BackEdgeHelper::RemovedEdges()
    const {
  return back_edges_;
}

Status BackEdgeHelper::Replace() {
  if (graph_ == nullptr) {
    return errors::Internal("BackEdgeHelper Replace called before Remove.");
  }
  if (replaced_) {
    return errors::Internal("BackEdgeHelper Replace called more than once.");
  }
  replaced_ = true;
  for (const BackEdge& be : back_edges_) {
    graph_->AddEdge(be.src, be.src_output, be.dst, be.dst_input);
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
