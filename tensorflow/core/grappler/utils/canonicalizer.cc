/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/canonicalizer.h"

#include <algorithm>

#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

void CanonicalizeNode(NodeDef* node) {
  if (node->input_size() < 2) return;
  // Partition control and regular inputs.
  int index = 0;
  for (; index < node->input_size(); ++index) {
    if (IsControlInput(node->input(index))) {
      break;
    }
  }
  auto* input = node->mutable_input();
  // Maybe sort regular inputs.
  if (IsCommutative(*node) && index > 0) {
    std::sort(input->begin(), input->begin() + index);
  }
  // Sort and dedup control inputs.
  if (index < node->input_size()) {
    std::sort(input->begin() + index, input->end());
    input->erase(std::unique(input->begin() + index, input->end()),
                 input->end());
  }
}

void CanonicalizeGraph(GraphDef* graph) {
  for (int i = 0; i < graph->node_size(); ++i) {
    CanonicalizeNode(graph->mutable_node(i));
  }
}

void CompressConstants(GraphDef* graph) {
  for (int i = 0; i < graph->node_size(); ++i) {
    NodeDef* node = graph->mutable_node(i);
    if ((IsConstant(*node) || IsHostConstant(*node)) &&
        HasNodeAttr(*node, "value")) {
      AttrValue& attr_val = (*node->mutable_attr())["value"];
      tensor::CompressTensorProtoInPlace(attr_val.mutable_tensor());
    }
  }
}

}  // namespace grappler
}  // namespace tensorflow
