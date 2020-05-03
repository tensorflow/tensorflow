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

#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/grappler/optimizers/data/vectorization/vectorizer_registry.h"

namespace tensorflow {
namespace grappler {

namespace {

// ParseExample is the vectorized version of ParseSingleExample.
class ParseSingleExampleVectorizer : public Vectorizer {
 public:
  Status Vectorize(const Node& node, Graph* outer_scope,
                   VectorizerInput&& inputs,
                   VectorizerOutput* outputs) override {
    NodeBuilder::NodeOut serialized;
    TF_RETURN_IF_ERROR(inputs.stacked(0, &serialized));

    std::vector<NodeBuilder::NodeOut> dense_defaults;
    dense_defaults.resize(inputs.size() - 1);
    for (size_t i = 1; i < inputs.size(); ++i) {
      TF_RETURN_IF_ERROR(inputs.unstacked(i, &dense_defaults[i - 1]));
    }

    Status scope_status;
    Scope parent = NewInternalScope(outer_scope, &scope_status, nullptr);
    Scope s = parent.NewSubScope("vectorize/parse_single_example");

    // Empty string vector
    Node* names = ops::Const(s, std::initializer_list<string>({})).node();

    // sparse_keys and dense_keys are attrs on ParseSingleExample, but are
    // inputs on ParseExample. We have to add const input nodes for these.
    auto make_list_input_from_attr =
        [&s, &node](StringPiece attr_name,
                    std::vector<NodeBuilder::NodeOut>* result) {
          std::vector<string> attr_vals;
          TF_RETURN_IF_ERROR(GetNodeAttr(node.attrs(), attr_name, &attr_vals));
          result->reserve(attr_vals.size());

          for (const auto& val : attr_vals) {
            result->push_back(ops::Const(s, val).node());
          }
          return Status::OK();
        };

    std::vector<NodeBuilder::NodeOut> sparse_keys;
    TF_RETURN_IF_ERROR(make_list_input_from_attr("sparse_keys", &sparse_keys));

    std::vector<NodeBuilder::NodeOut> dense_keys;
    TF_RETURN_IF_ERROR(make_list_input_from_attr("dense_keys", &dense_keys));

    TF_RETURN_IF_ERROR(scope_status);

    Node* new_node;
    auto node_builder =
        NodeBuilder(strings::StrCat("vectorized/", node.name()), "ParseExample")
            .Input(serialized)
            .Input(names)
            .Input(sparse_keys)
            .Input(dense_keys)
            .Input(dense_defaults);

    for (const auto& attr : {"sparse_types", "dense_shapes"}) {
      // Copy attrs if they exist
      const AttrValue* val;
      TF_RETURN_IF_ERROR(node.attrs().Find(attr, &val));
      node_builder = node_builder.Attr(attr, *val);
    }

    TF_RETURN_IF_ERROR(node_builder.Finalize(outer_scope, &new_node));

    // Add output mappings
    for (size_t i = 0; i < node.num_outputs(); ++i) {
      outputs->emplace_back(new_node, i, true);
    }
    return Status::OK();
  }
};

REGISTER_VECTORIZER("ParseSingleExample", ParseSingleExampleVectorizer);

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
