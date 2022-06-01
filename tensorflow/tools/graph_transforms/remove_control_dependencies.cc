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
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Remove control dependencies in preparation for inference.
// In the tensorflow graph, control dependencies are represented as extra
// inputs which are referenced with "^tensor_name".
// See node_def.proto for more details.
Status RemoveControlDependencies(const GraphDef& input_graph_def,
                const TransformFuncContext& context,
                GraphDef* output_graph_def) {
    output_graph_def->Clear();
    for (const NodeDef& node : input_graph_def.node()) {
        NodeDef* new_node = output_graph_def->mutable_node()->Add();
        *new_node = node;
        new_node->clear_input();
        for (const auto& input : node.input()) {
            if (input[0] != '^') {
                new_node->add_input(input);
            }
        }
    }
    return OkStatus();
}

REGISTER_GRAPH_TRANSFORM("remove_control_dependencies", RemoveControlDependencies);

}  // namespace graph_transforms
}  // namespace tensorflow
