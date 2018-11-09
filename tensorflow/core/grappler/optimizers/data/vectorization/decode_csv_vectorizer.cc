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

#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/grappler/optimizers/data/vectorization/vectorizer_registry.h"

namespace tensorflow {
namespace grappler {

namespace {

// DecodeCSV is the vectorized version of itself.
class DecodeCSVVectorizer : public Vectorizer {
 public:
  Status Vectorize(const Node& node, Graph* outer_scope,
                   VectorizerInput&& inputs,
                   VectorizerOutput* outputs) override {
    NodeBuilder::NodeOut records;
    TF_RETURN_IF_ERROR(inputs.stacked(0, &records));

    std::vector<NodeBuilder::NodeOut> defaults;
    defaults.resize(inputs.size() - 1);
    for (size_t i = 1; i < inputs.size(); ++i) {
      TF_RETURN_IF_ERROR(inputs.unstacked(i, &defaults[i - 1]));
    }

    Node* new_node;
    auto node_builder = NodeBuilder(node.type_string(), node.type_string())
                            .Input(records)
                            .Input(defaults);

    for (const auto& attr : node.attrs()) {
      node_builder = node_builder.Attr(attr.first, attr.second);
    }
    TF_RETURN_IF_ERROR(node_builder.Finalize(outer_scope, &new_node));

    // Add output mappings
    for (int i = 0; i < node.num_outputs(); ++i) {
      outputs->emplace_back(new_node, i, true);
    }
    return Status::OK();
  }
};

REGISTER_VECTORIZER("DecodeCSV", DecodeCSVVectorizer);

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
