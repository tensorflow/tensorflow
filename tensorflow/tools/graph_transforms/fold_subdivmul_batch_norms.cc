/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Convolution is:
// ---------------
// A quick recap of what a convolution layer calculates: if x is the
// pixels in the input image and w is the weights for the layer, then
// the convolution basically computes the following for each output pixel:
// out[j] = x[i]*w[0] + x[i+1]*w[1] + x[i+2]*w[2] + ... + x[i+k]*w[k] + b
// So x[i] is the input to Conv2D.
//    w[i] is the Const to Conv2D.
//
// BiasNorm After Convolution where out[j] is output of Conv2D
// ------------------------------------------------------------
//          gamma * (out[j] - mean)
//  bn[j] = ---------------------- + beta
//              sqrt(variance)
// So mean is the Const to Sub.
//    sqrt(variance) is the Const to TrueDiv.
//    gamma is the Const to Mul.
//    beta is the Const to BiasAdd.
//
// Now We can Fold these Operations into Conv2D and BiasAdd
// ---------------------------------------------------------
//           gamma * w
// w_new = --------------
//         sqrt(variance)
//
//         gamma*(b - mean)
// b_new = ---------------- + beta
//          sqrt(variance)
//
// So w_new[i] is the new Const to Conv2D.
//    Sub, TrueDiv, Mul can be removed(Batchnorm removed).
//    b_new    is the new Const to BiasAdd.
//    b is 0 for YOLOv2

Status FoldSubDivMulBatchNorms(const GraphDef& input_graph_def,
                               const TransformFuncContext& context,
                               GraphDef* output_graph_def) {
  std::map<string, const NodeDef*> node_map;
  MapNamesToNodes(input_graph_def, &node_map);
  GraphDef replaced_graph_def;
  ReplaceMatchingOpTypesOptions options;
  options.allow_inconsistencies = false;

  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      // ConstNode--->Mul--->Maximum(leaky Relu)
      {"BiasAdd",                       // biasadd node
        {
          {"Mul",                       // mul node
            {
              {"RealDiv",               // Realdiv node
                {
                  {"Sub",               // sub node
                    {
                      {"Conv2D",        // conv2d node
                        {
                          {"*"},        // input node x[i]
                          {"Const"}     // Conv2d const weights w[i]
                        }
                      },
                      {"Const"}         // sub const mean
                    }
                  },
                  {"Const"}             // div const variance
                }
              },
              {"Const"}                 // mul const gamma
            }
          },
          {"Const"}                     // bias const beta
        }
      },  // clang-format on */
      [&node_map](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        // Find all the nodes we expect in the subgraph.
        const NodeDef& biasadd_node   = match.node;
        const NodeDef& beta_node      = match.inputs[1].node;
        const NodeDef& gamma_node     = match.inputs[0].inputs[1].node;
        const NodeDef& variance_node  =
            match.inputs[0].inputs[0].inputs[1].node;
        const NodeDef& conv_node      =
            match.inputs[0].inputs[0].inputs[0].inputs[0].node;
        const NodeDef& mean_node      =
            match.inputs[0].inputs[0].inputs[0].inputs[1].node;
        const NodeDef& input_node     =
            match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node;
        const NodeDef& weights_node   =
            match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[1].node;

        // Verify all the Const Nodes are proper.
        CHECK_EQ("Const", beta_node.op());
        CHECK_EQ("Const", gamma_node.op());
        CHECK_EQ("Const", variance_node.op());
        CHECK_EQ("Const", mean_node.op());
        CHECK_EQ("Const", weights_node.op());

        // Get the Tensor Values of all the constant nodes
        Tensor beta     = GetNodeTensorAttr(beta_node,    "value");
        Tensor gamma    = GetNodeTensorAttr(gamma_node,   "value");
        Tensor variance = GetNodeTensorAttr(variance_node, "value");
        Tensor mean     = GetNodeTensorAttr(mean_node,    "value");
        Tensor weights  = GetNodeTensorAttr(weights_node, "value");

        // Let us find the new weights(Modify const node to Conv2D)
        // --------------------------------------------------------
        // Get the Shape and the actual dimensions of each needed tensor
        const int64 weights_cols = weights.shape().dim_size(3);
        const int64 gamma_cols = gamma.shape().dim_size(0);
        const int64 variance_cols = variance.shape().dim_size(0);

        // verify the weights and the constants have same dimenions.
        CHECK_EQ(weights_cols, gamma_cols);
        CHECK_EQ(weights_cols, variance_cols);

        // Flat all dimensions except the last one so that we can iterate.
        auto weights_matrix = weights.flat_inner_dims<float>();
        // gamma is a scalar value
        auto gamma_matrix = gamma.flat_inner_dims<float>();
        // variance is 1D vector
        auto variance_matrix = variance.flat_inner_dims<float>();

        // Perform the actual calculation to get w_new(new_weights_matrix)
        Tensor new_weights(DT_FLOAT, weights.shape());
        auto new_weights_matrix = new_weights.flat_inner_dims<float>();
        for (int64 row = 0; row < weights_matrix.dimension(0); ++row) {
          for (int64 col = 0; col < weights_cols; ++col) {
            new_weights_matrix(row, col) = weights_matrix(row, col)
                * gamma_matrix(col) / variance_matrix(col);
          }
        }

        // Let us find the new Beta (Modify Const node to BiasAdd)
        // -------------------------------------------------------

        // Get the Shape and the actual dimensions of each needed tensor
        const int64 beta_cols = beta.shape().dim_size(0);
        const int64 mean_cols = mean.shape().dim_size(0);

        // verify the beta and the constants have same dimenions.
        CHECK_EQ(beta_cols, gamma_cols);
        CHECK_EQ(beta_cols, mean_cols);
        CHECK_EQ(beta_cols, variance_cols);

        // Flat all dimensions except the last one so that we can iterate.
        // beta is a scalar value
        auto beta_matrix = beta.flat_inner_dims<float>();
        // mean is a scalar value
        auto mean_matrix = mean.flat_inner_dims<float>();

        // Perform the actual calculation to get b_new(new_beta_matrix)
        Tensor new_beta(DT_FLOAT, beta.shape());
        auto new_beta_matrix = new_beta.flat_inner_dims<float>();
        for (int64 col = 0; col < beta_cols; ++col) {
          new_beta_matrix(col) = (((0- mean_matrix(col)) * gamma_matrix(col))
                                 / variance_matrix(col) ) + beta_matrix(col);
        }

        // Construct the new modified graph
        // --------------------------------
        // Construct the new weights const node.
        NodeDef new_weights_node;
        new_weights_node.set_op("Const");
        new_weights_node.set_name(weights_node.name());
        SetNodeAttr("dtype", DT_FLOAT, &new_weights_node);
        SetNodeTensorAttr<float>("value", new_weights, &new_weights_node);

        // Construct the new beta const node.
        NodeDef new_beta_node;
        new_beta_node.set_op("Const");
        new_beta_node.set_name(beta_node.name());
        SetNodeAttr("dtype", DT_FLOAT, &new_beta_node);
        SetNodeTensorAttr<float>("value", new_beta, &new_beta_node);

        // Construct the new BiasAdd node.
        // Note that the input of BiasAdd node is the new beta const node
        // and the conv 2d node. (The Sub, TrueDiv and Mul is removed.)
        NodeDef bias_add_node;
        bias_add_node.set_op("BiasAdd");
        bias_add_node.set_name(biasadd_node.name());
        if (conv_node.attr().count("data_format") > 0) {
          CopyNodeAttr(conv_node, "data_format", "data_format", &bias_add_node);
        }
        CopyNodeAttr(conv_node, "T", "T", &bias_add_node);
        AddNodeInput(conv_node.name(), &bias_add_node);
        AddNodeInput(new_beta_node.name(), &bias_add_node);

        // The input and convolution can be copied straight over.

        // Construct the new graph with the new nodes.
        new_nodes->push_back(input_node);        // insert input node first.
        new_nodes->push_back(new_weights_node);  // insert the new weights node.
        new_nodes->push_back(conv_node);         // insert the conv node.
        new_nodes->push_back(new_beta_node);     // insert the beta node.
        new_nodes->push_back(bias_add_node);     // insert the biasadd node.

        return Status::OK();
      },
      options, &replaced_graph_def));

  *output_graph_def = replaced_graph_def;
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("fold_subdivmul_batch_norms", FoldSubDivMulBatchNorms);

}  // namespace graph_transforms
}  // namespace tensorflow
