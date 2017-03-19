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

#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Switch any ConcatV2 nodes to the v1 version, swapping the input order.
Status BackportConcatV2Transform(const GraphDef& input_graph_def,
                                 const TransformFuncContext& context,
                                 GraphDef* output_graph_def) {
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def, {"ConcatV2"},
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        const NodeDef& concat_v2_node = match.node;
        NodeDef concat_node = concat_v2_node;
        concat_node.set_op("Concat");
        // The last input is inserted at the head of the inputs, because Concat
        // expects the dimension as the first input (not the last as in
        // ConcatV2).
        concat_node.mutable_input()->Clear();
        const string& dim_input =
            concat_v2_node.input(concat_v2_node.input_size() - 1);
        concat_node.add_input(dim_input);
        for (int i = 0; i < (concat_v2_node.input_size() - 1); ++i) {
          concat_node.add_input(concat_v2_node.input(i));
        }
        // Tidx attribute must be deleted because it's not used in Concat.
        concat_node.mutable_attr()->erase("Tidx");
        new_nodes->push_back(concat_node);
        return Status::OK();
      },
      {true}, output_graph_def));

  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("backport_concatv2", BackportConcatV2Transform);

}  // namespace graph_transforms
}  // namespace tensorflow
