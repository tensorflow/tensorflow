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

#define EIGEN_USE_THREADS

#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// EXPERIMENTAL: This can change without warning.
// Given a graph that has gone through the FakeQuantizeTraining transform and
// has been frozen afterwards, RemoveEMA simplifies the FakeQuantize estimated
// moving average subgraphs to make it compatible with the QuantizeNodes
// transform.
Status RemoveEMA(const GraphDef& input_graph_def,
                 const TransformFuncContext& context,
                 GraphDef* output_graph_def) {
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"FakeQuantWithMinMaxVars",
       {
         {"*"},
         {"Assign",
          {
            {"Const"},
            {"Merge",
             {
               {"Switch",
                {
                  {"Min",
                   {
                     {"*"},
                     {"Range",
                      {
                        {"*"},
                        {"*"},
                        {"*"},
                      }
                     }
                   }
                  },
                  {"IsVariableInitialized"}
                }
               },
               {"Sub",
                {
                  {"Const"},
                  {"Mul",
                   {
                     {"Sub"},
                     {"Sub",
                      {
                        {"Const"},
                        {"Const"}
                      }
                     }
                   }
                  }
                }
               }
             }
            }
          }
         },
         {"Assign",
          {
            {"Const"},
            {"Merge",
             {
               {"Switch",
                {
                  {"Max"},
                  {"IsVariableInitialized"}
                }
               },
               {"Sub",
                {
                  {"Const"},
                  {"Mul",
                   {
                     {"Sub"},
                     {"Sub",
                      {
                        {"Const"},
                        {"Const"}
                      }
                     }
                   }
                  }
                }
               }
             }
            }
          }
         },
       }
      },  // clang-format on
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        const NodeDef& fake_quant_node = match.node;
        const NodeDef& input_node = match.inputs[0].node;
        const NodeDef& min_var_node = match.inputs[1].inputs[0].node;
        const NodeDef& max_var_node = match.inputs[2].inputs[0].node;

        // Make a new FakeQuantizeWithMinMaxVars operation that uses constants
        // for its min/max arguments rather than an entire EMA subgraph.
        NodeDef new_fake_quant_node;
        new_fake_quant_node.set_op(fake_quant_node.op());
        new_fake_quant_node.set_name(fake_quant_node.name());
        AddNodeInput(input_node.name(), &new_fake_quant_node);
        AddNodeInput(min_var_node.name(), &new_fake_quant_node);
        AddNodeInput(max_var_node.name(), &new_fake_quant_node);
        CopyNodeAttr(fake_quant_node, "narrow_range", "narrow_range",
                     &new_fake_quant_node);
        CopyNodeAttr(fake_quant_node, "num_bits", "num_bits",
                     &new_fake_quant_node);

        new_nodes->push_back(new_fake_quant_node);
        new_nodes->push_back(input_node);
        new_nodes->push_back(min_var_node);
        new_nodes->push_back(max_var_node);

        return Status::OK();
      },
      {}, output_graph_def));
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("remove_ema", RemoveEMA);

}  // namespace graph_transforms
}  // namespace tensorflow
