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
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/vectorization/vectorizer_registry.h"

namespace tensorflow {
namespace grappler {
namespace vectorization_utils {

class UnpackVectorizer : public Vectorizer {
 public:
  Status Vectorize(const NodeDef& node, gtl::ArraySlice<string> inputs,
                   FunctionDef* outer_scope,
                   std::map<string, string>* conversion_map) override {
    if (inputs.size() != 1) {
      return errors::Internal("Unpack op should only have one input.");
    }

    // Add new Unpack node
    NodeDef* new_unpack_node = outer_scope->add_node_def();
    *new_unpack_node = node;
    new_unpack_node->clear_name();
    function_utils::SetUniqueFunctionNodeName(
        strings::StrCat("vectorized/", node.name()), outer_scope,
        new_unpack_node);

    // Increment "axis" attr by 1:
    (*new_unpack_node->mutable_attr())["axis"].set_i(
        node.attr().at("axis").i() + 1);
    new_unpack_node->set_input(0, inputs[0]);

    // Add the output mappings to conversion map
    int num = new_unpack_node->attr().at("num").i();
    for (int i = 0; i < num; ++i) {
      (*conversion_map)[strings::StrCat(node.name(), ":output:", i)] =
          strings::StrCat(new_unpack_node->name(), ":output:", i);
    }

    return Status::OK();
  }
};

REGISTER_VECTORIZER("Unpack", UnpackVectorizer);

}  // namespace vectorization_utils
}  // namespace grappler
}  // namespace tensorflow
