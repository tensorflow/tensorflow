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

// Fold batchnorm which is unfolded into series of algebraic expressions
// For example:(x - mean) * rsqrt(variance + epsilon) * gamma + beta
// gamma and beta are Const nodes, mean and variance may be Const nodes, or come
// from the outputs of nn.moments()
Status FoldBatchNormsAlgebraic(const GraphDef& input_graph_def,
                               const TransformFuncContext& context,
                               GraphDef* output_graph_def) {
  std::map<string, string> inputs_to_rename;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
    {"Add",
      {
        {"Mul",                   // mul_1-->x * (rsqrt(variance + epsilon) * gamma)
          {
            {"*"},
            {"Mul",               // mul-->rsqrt(variance + epsilon) * gamma
              {
                {"Rsqrt",
                  {
                    {"Add",       // add-->variance + epsilon
                      {
                        {"*"},    // variance node
                        {"Const"} // epsilon
                      }
                    }
                  }
                },
                {"Const"}         // gamma const value
              }
            }
          }
        },
        {"Sub",                   // sub-->beta - (rsqrt(variance + epsilon) * gamma) * mean
          {
            {"Const"},            // beta const value
            {"Mul",               // mul_2-->(rsqrt(variance + epsilon) * gamma) * mean
              {
                {"*"},            // mean node
                {"Mul"}           // mul
              }
            }
          }
        }
      }
    },  // clang-format on
      [&inputs_to_rename](const NodeMatch& match,
                          const std::set<string>& input_nodes,
                          const std::set<string>& output_nodes,
                          std::vector<NodeDef>* new_nodes) {
        // Find all the nodes we expect in the subgraph.
        const NodeDef& add_node = match.node;
        const NodeDef& mul1_node = match.inputs[0].node;
        const NodeDef& sub_node = match.inputs[1].node;
        const NodeDef& mul1_input0_node = match.inputs[0].inputs[0].node;
        const NodeDef& mul_node = match.inputs[0].inputs[1].node;
        const NodeDef& add_epsilon_node =
            match.inputs[0].inputs[1].inputs[0].inputs[0].node;
        const NodeDef& epsilon_node =
            match.inputs[0].inputs[1].inputs[0].inputs[0].inputs[1].node;
        const NodeDef& gamma_node = match.inputs[0].inputs[1].inputs[1].node;
        const NodeDef& beta_node = match.inputs[1].inputs[0].node;
        const NodeDef& mul2_node = match.inputs[1].inputs[1].node;
        const NodeDef& mul_node_alias =
            match.inputs[1].inputs[1].inputs[1].node;

        const NodeDef& mean_node = match.inputs[1].inputs[1].inputs[0].node;
        const NodeDef& variance_node =
            match.inputs[0].inputs[1].inputs[0].inputs[0].inputs[0].node;
        CHECK_EQ(mul_node.name(), mul_node_alias.name())
            << "Sub graph not matched!";

        CHECK_EQ("Const", epsilon_node.op()) << "Sub graph not matched!";
        CHECK_EQ("Const", gamma_node.op())
            << "You Should Apply remove_nodes(op=Identity) first!";
        CHECK_EQ("Const", beta_node.op())
            << "You Should Apply remove_nodes(op=Identity) first!";

        NodeDef instance_norms_node;
        instance_norms_node.set_op("InstanceNorm");
        instance_norms_node.set_name(add_node.name() + "__InstanceNorm");
        SetNodeAttr("T", DT_FLOAT, &instance_norms_node);
        CopyNodeAttr(epsilon_node, "value", "epsilon", &instance_norms_node);
        AddNodeInput(mul1_input0_node.name(), &instance_norms_node);
        AddNodeInput(gamma_node.name(), &instance_norms_node);
        AddNodeInput(beta_node.name(), &instance_norms_node);
        AddNodeInput(mul2_node.input(0), &instance_norms_node);
        AddNodeInput(add_epsilon_node.input(0), &instance_norms_node);

        new_nodes->push_back(instance_norms_node);
        new_nodes->push_back(gamma_node);
        new_nodes->push_back(beta_node);
        new_nodes->push_back(mean_node);
        new_nodes->push_back(variance_node);
        new_nodes->push_back(mul1_input0_node);

        inputs_to_rename[add_node.name()] = instance_norms_node.name();
        return Status::OK();
      },
      {true}, &replaced_graph_def));

  // Chang the input_name which use nodes in this sub graph
  TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                      std::unordered_set<string>(),
                                      output_graph_def));
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("fold_batch_norms_algebraic", FoldBatchNormsAlgebraic);

}  // namespace graph_transforms
}  // namespace tensorflow
