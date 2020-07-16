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

#include "tensorflow/compiler/jit/test_util.h"

#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace tensorflow {

Status ShapeAnnotationsMatch(
    const Graph& graph, const GraphShapeInfo& shape_info,
    std::map<string, std::vector<PartialTensorShape>> expected_shapes) {
  for (Node* node : graph.op_nodes()) {
    auto sit = shape_info.find(node->name());
    TF_RET_CHECK(sit != shape_info.end())
        << "Missing shape information for node " << node->name();
    std::vector<PartialTensorShape> shapes;
    for (const auto& output : sit->second) shapes.push_back(output.shape);

    auto it = expected_shapes.find(node->name());
    if (it != expected_shapes.end()) {
      if (!PartialTensorShapeUtils::AreIdentical(shapes, it->second)) {
        return errors::InvalidArgument(
            "Shape mismatch for ", node->name(), ". Expected: ",
            PartialTensorShapeUtils::PartialShapeListString(it->second),
            ", actual: ",
            PartialTensorShapeUtils::PartialShapeListString(shapes));
      }
      expected_shapes.erase(it);
    }
  }
  if (!expected_shapes.empty()) {
    std::vector<string> missing;
    missing.reserve(expected_shapes.size());
    for (const auto& entry : expected_shapes) {
      missing.push_back(entry.first);
    }
    return errors::InvalidArgument("Missing shapes for nodes: ",
                                   absl::StrJoin(missing, ","));
  }
  return Status::OK();
}

}  // namespace tensorflow
