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

#include "tensorflow/compiler/jit/encapsulate_util.h"
#include <algorithm>
#include <iterator>

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/shape_inference.h"

namespace tensorflow {

namespace {

// Returns string attribute value for the node if the attribute is present,
// otherwise returns empty optional value.
absl::optional<string> GetStringAttr(const Node& n, const string& attr_name) {
  auto attr = n.attrs().Find(attr_name);
  if (!attr) {
    return absl::nullopt;
  } else {
    return attr->s();
  }
}

}  // namespace

const char kXlaInferredShapesAttrName[] = "_xla_inferred_shapes";

Status PerformStaticShapeInferenceBeforeEncapsulation(
    Graph* g, const string& xla_computation_attr_name,
    const string& outside_compilation_attr_name) {
  // Find all outside compilation to XLA computation data edges.
  std::unordered_set<Node*> outside_compilation_send_nodes;
  for (auto e : g->edges()) {
    if (e->IsControlEdge()) {
      continue;
    }

    auto src_computation = GetStringAttr(*e->src(), xla_computation_attr_name);
    auto dst_computation = GetStringAttr(*e->dst(), xla_computation_attr_name);
    if (!src_computation || !dst_computation ||
        *src_computation != *dst_computation) {
      continue;
    }

    auto src_outside_compilation =
        GetStringAttr(*e->src(), outside_compilation_attr_name);
    auto dst_outside_compilation =
        GetStringAttr(*e->dst(), outside_compilation_attr_name);
    if (src_outside_compilation && !dst_outside_compilation) {
      outside_compilation_send_nodes.insert(e->src());
    }
  }

  // Perform shape inference.
  std::map<int, InferredShape> arg_shapes;
  GraphShapeInfo shape_info;
  TF_RETURN_IF_ERROR(
      InferShapes(g, arg_shapes, /*fnlib_def=*/nullptr, &shape_info));

  // Add attribute for output shapes.
  for (Node* n : outside_compilation_send_nodes) {
    auto iter = shape_info.find(n->name());
    if (iter == shape_info.end()) {
      continue;
    }

    std::vector<PartialTensorShape> output_shapes;
    std::transform(iter->second.begin(), iter->second.end(),
                   std::back_inserter(output_shapes),
                   [](const InferredShape& inferred_shape) {
                     return inferred_shape.shape;
                   });
    n->AddAttr(kXlaInferredShapesAttrName, output_shapes);
  }

  return Status::OK();
}

}  // namespace tensorflow
