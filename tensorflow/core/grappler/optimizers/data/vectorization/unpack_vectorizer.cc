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
                   std::vector<WrappedTensor>&& inputs,
                   std::vector<WrappedTensor>* outputs) override {
    Status s;
    if (node.num_inputs() != 1 || inputs.size() != 1) {
      return errors::Internal("Unpack op should only have one input.");
    }

    // Add new Unpack node with the same op and attrs as the original node
    auto new_unpack_node = outer_scope->AddNode(node.def(), &s);
    TF_RETURN_IF_ERROR(s);

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
    new_unpack_node->AddAttr("axis", axis);

    outer_scope->AddEdge(inputs[0].node, inputs[0].output_index,
                         new_unpack_node, 0);

    int num;
    TF_RETURN_IF_ERROR(GetNodeAttr(node.attrs(), "num", &num));

    // Add the output mappings
    for (int i = 0; i < num; ++i) {
      outputs->push_back({new_unpack_node, i, true});
    }

    return Status::OK();
  }
};

REGISTER_VECTORIZER("Unpack", UnpackVectorizer);

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
