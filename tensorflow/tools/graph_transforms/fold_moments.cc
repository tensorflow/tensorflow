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
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

Status FoldMoments(const GraphDef& input_graph_def,
                   const TransformFuncContext& context,
                   GraphDef* output_graph_def) {
  std::map<string, string> inputs_to_rename;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
		{"Mean",
			{
				{"Mul",
					{
						{"Sub",
							{
								{"*"},
								{"Mean",
									{
										{"*"},
										{"Const"}
									}
								}
							}
						},
						{"*"}
					}
				},
				{"Const"}
			}
		},  // clang-format on
      [&inputs_to_rename](const NodeMatch& match,
                          const std::set<string>& input_nodes,
                          const std::set<string>& output_nodes,
                          std::vector<NodeDef>* new_nodes) {
        // Find all the nodes we expect in the subgraph.
        const NodeDef& variance_mean_node = match.node;
        const NodeDef& mul_node = match.inputs[0].node;
        const NodeDef& sub_node = match.inputs[0].inputs[0].node;
        const NodeDef& mean_node = match.inputs[0].inputs[0].inputs[1].node;
        const NodeDef& mean_node_input_node =
            match.inputs[0].inputs[0].inputs[1].inputs[0].node;
        const NodeDef& mean_reduction_indices_node =
            match.inputs[0].inputs[0].inputs[1].inputs[1].node;
        CHECK_EQ(sub_node.input(0), mean_node.input(0))
            << "sub and mean should have the same input!";

        NodeDef moments_node;
        moments_node.set_op("Moments");
        moments_node.set_name(mean_node.name() + "__moments");
        SetNodeAttr("T", DT_FLOAT, &moments_node);
        CopyNodeAttr(mean_node, "keep_dims", "keep_dims", &moments_node);
        CopyNodeAttr(mean_node, "Tidx", "Tidx", &moments_node);

        NodeDef moments_axes_node;
        moments_axes_node.set_op("Const");
        moments_axes_node.set_name(mean_node.name() + "_axes");
        CopyNodeAttr(mean_reduction_indices_node, "dtype", "dtype",
                     &moments_axes_node);
        CopyNodeAttr(mean_reduction_indices_node, "value", "value",
                     &moments_axes_node);

        AddNodeInput(mean_node.input(0), &moments_node);
        AddNodeInput(moments_axes_node.name(), &moments_node);

        inputs_to_rename[mean_node.name()] = moments_node.name() + ":0";
        inputs_to_rename[variance_mean_node.name()] =
            moments_node.name() + ":1";

        new_nodes->push_back(moments_node);
        new_nodes->push_back(moments_axes_node);
        new_nodes->push_back(mean_node_input_node);
        return Status::OK();
      },
      {true}, &replaced_graph_def));

  // Change the input_name of the nodes that use mean and variance.
  TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                      std::unordered_set<string>(),
                                      output_graph_def));
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("fold_moments", FoldMoments);

}  // namespace graph_transforms
}  // namespace tensorflow
