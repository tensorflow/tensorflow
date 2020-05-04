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

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
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

// Switch any TensorArrayV3 nodes to the v2 version, removing the second output.
Status BackportTensorArrayV3Transform(const GraphDef& input_graph_def,
                                      const TransformFuncContext& context,
                                      GraphDef* output_graph_def) {
  std::map<string, string> inputs_to_rename;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def, {"TensorArrayV3|TensorArrayGradV3"},
      [&inputs_to_rename](const NodeMatch& match,
                          const std::set<string>& input_nodes,
                          const std::set<string>& output_nodes,
                          std::vector<NodeDef>* new_nodes) {
        const NodeDef& tensor_array_v3_node = match.node;

        // All we need to do here is rename the op type, since the attributes
        // remain the same.
        NodeDef tensor_array_v2_node = tensor_array_v3_node;
        if (tensor_array_v3_node.op() == "TensorArrayV3") {
          tensor_array_v2_node.set_op("TensorArrayV2");
        } else {
          tensor_array_v2_node.set_op("TensorArrayGradV2");
        }

        // The v3 version has a second 'flow' output that's not present in v2,
        // so substitute a dummy constant instead in any places that use it.
        NodeDef replacement_flow_node;
        replacement_flow_node.set_op("Const");
        SetNodeAttr("dtype", DT_FLOAT, &replacement_flow_node);
        replacement_flow_node.set_name(tensor_array_v3_node.name() +
                                       "/replacement_flow_node");
        Tensor replacement_flow_tensor(DT_FLOAT, {});
        // I'm picking an arbitrary value for the gradient flow here, for lack
        // of a better alternative.
        replacement_flow_tensor.flat<float>()(0) = 1.0f;
        SetNodeTensorAttr<float>("value", replacement_flow_tensor,
                                 &replacement_flow_node);
        inputs_to_rename[tensor_array_v3_node.name() + ":1"] =
            replacement_flow_node.name();

        new_nodes->push_back(tensor_array_v2_node);
        new_nodes->push_back(replacement_flow_node);
        return Status::OK();
      },
      {true}, &replaced_graph_def));
  // Update the graph so that any nodes that referred to removed inputs now
  // pull from the substitute constants we've added.
  GraphDef renamed_graph_def;
  TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                      std::unordered_set<string>(),
                                      &renamed_graph_def));
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      renamed_graph_def,
      {"TensorArrayWriteV3|TensorArrayReadV3|TensorArrayGatherV3|"
       "TensorArrayScatterV3|TensorArrayConcatV3|TensorArraySplitV3|"
       "TensorArraySizeV3|TensorArrayCloseV3"},
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        const NodeDef& v3_node = match.node;
        NodeDef v2_node = v3_node;
        v2_node.set_op(v3_node.op().substr(0, v3_node.op().size() - 1) + "2");
        new_nodes->push_back(v2_node);
        return Status::OK();
      },
      {true}, output_graph_def));
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("backport_tensor_array_v3",
                         BackportTensorArrayV3Transform);

}  // namespace graph_transforms
}  // namespace tensorflow
