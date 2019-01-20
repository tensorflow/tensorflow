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

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/optimizers/data/vectorization/vectorizer_registry.h"

namespace tensorflow {
namespace grappler {
namespace {

class UnpackVectorizer : public Vectorizer {
 public:
  Status Vectorize(const Node& node, Graph* outer_scope,
                   VectorizerInput&& inputs,
                   VectorizerOutput* outputs) override {
    NodeBuilder::NodeOut value;
    TF_RETURN_IF_ERROR(inputs.stacked(0, &value));

    int axis = 0;
    if (HasNodeAttr(node.def(), "axis")) {
      TF_RETURN_IF_ERROR(GetNodeAttr(node.attrs(), "axis", &axis));
    }

    if (axis >= 0) {
      // Since the vectorized input has an extra leading dimension, we need
      // to increment `axis` attr by 1 for non-negative axis values.
      // Note: negative axis values wrap around.
      axis += 1;
    }

    int num;
    TF_RETURN_IF_ERROR(GetNodeAttr(node.attrs(), "num", &num));

    Node* new_node;
    TF_RETURN_IF_ERROR(NodeBuilder(strings::StrCat("vectorized/", node.name()),
                                   node.type_string())
                           .Input(value)
                           .Attr("axis", axis)
                           .Attr("num", num)
                           .Finalize(outer_scope, &new_node));

    // Add the output mappings
    for (int i = 0; i < num; ++i) {
      outputs->push_back({new_node, i, true});
    }

    return Status::OK();
  }
};

REGISTER_VECTORIZER("Unpack", UnpackVectorizer);

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
