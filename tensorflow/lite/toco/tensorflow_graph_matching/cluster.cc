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
#include "tensorflow/lite/toco/tensorflow_graph_matching/cluster.h"

namespace toco {

void Cluster::SetGraphDefInfo(const tensorflow::GraphDef* graph_def) {
  graph_def_ = graph_def;
  for (const tensorflow::NodeDef& node : graph_def_->node()) {
    if (StrContains(node.name(), name_)) {
      nodes_.push_back(&node);
    }
  }
}

bool Cluster::FindClusterInputsAndOutputs() {
  // For every node N in the graph:
  // If N belongs to this cluster C, then each of N's inputs that are not part
  // of C are then inputs of C.
  // If N does not belong to cluster C, then each of N's inputs that belong to C
  // are then outputs of C.
  for (const tensorflow::NodeDef& node : graph_def_->node()) {
    if (StrContains(node.name(), name_)) {
      for (int i = 0; i < node.input_size(); i++) {
        if (!StrContains(node.input(i), name_)) {
          inputs_.push_back(node.input(i));
        }
      }
    } else {
      for (int i = 0; i < node.input_size(); i++) {
        if (StrContains(node.input(i), name_)) {
          outputs_.push_back(node.input(i));
        }
      }
    }
  }
  return (!inputs_.empty()) && (!outputs_.empty());
}

}  // end namespace toco
