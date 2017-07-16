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
namespace {
// Ensures the tensor is the expected shape.
Status ErrorIfNotVector(const Tensor& input, const string& input_name,
                        int expected_width) {
  if ((input.shape().dims() != 1) ||
      (input.shape().dim_size(0) != expected_width)) {
    return errors::InvalidArgument(input_name,
                                   " input to batch norm has bad shape: ",
                                   input.shape().DebugString());
  }
  return Status::OK();
}
}  // namespace

// Finds monolithic batch norm ops (as used in early versions of TensorFlow) and
// converts them into premultiplied weight inputs to convolutions.
Status FoldOldBatchNorms(const GraphDef& input_graph_def,
                         const TransformFuncContext& context,
                         GraphDef* output_graph_def) {
  GraphDef current_graph_def = input_graph_def;
  // We have to do several passes to catch all the old BN nodes, since many of
  // them may share inputs and so be excluded from replacement in one pass.
  bool did_graph_change;
  do {
    did_graph_change = false;
    GraphDef replaced_graph_def;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def,  // clang-format off
      {"BatchNormWithGlobalNormalization|FusedBatchNorm",    // batch_norm_node
        {
          {"Conv2D",                          // conv_node
            {
              {"*"},                          // input_node
              {"Const"},                      // weights_node
            }
          },
          {"Const"},                          // mean_node
          {"Const"},                          // variance_node
          {"Const"},                          // beta_node
          {"Const"},                          // gamma_node
        }
      },  // clang-format on
        [&did_graph_change](const NodeMatch& match,
                            const std::set<string>& input_nodes,
                            const std::set<string>& output_nodes,
                            std::vector<NodeDef>* new_nodes) {
          // Find all the nodes we expect in the subgraph.
          const NodeDef& batch_norm_node = match.node;
          // BatchNormWithGlobalNormalization and FusedBatchNorm ops only differ
          // by input order and attribute names.
          CHECK(batch_norm_node.op() == "BatchNormWithGlobalNormalization" ||
                batch_norm_node.op() == "FusedBatchNorm");
          const bool is_fused = batch_norm_node.op() == "FusedBatchNorm";
          const int mean_idx = is_fused ? 3 : 1;
          const int var_idx = is_fused ? 4 : 2;
          const int beta_idx = is_fused ? 2 : 3;
          const int gamma_idx = is_fused ? 1 : 4;
          const string epsilon_attr = is_fused ? "epsilon" : "variance_epsilon";
          // FusedBatchNorm always scales after normalization.
          const bool scale_after_normalization =
              is_fused ||
              batch_norm_node.attr().at("scale_after_normalization").b();

          const NodeDef& conv_node = match.inputs[0].node;
          CHECK_EQ("Conv2D", conv_node.op());
          const NodeDef& input_node = match.inputs[0].inputs[0].node;
          const NodeDef& weights_node = match.inputs[0].inputs[1].node;
          CHECK_EQ("Const", weights_node.op());
          const NodeDef& mean_node = match.inputs[mean_idx].node;
          CHECK_EQ("Const", mean_node.op());
          const NodeDef& variance_node = match.inputs[var_idx].node;
          CHECK_EQ("Const", variance_node.op());
          const NodeDef& beta_node = match.inputs[beta_idx].node;
          CHECK_EQ("Const", beta_node.op());
          const NodeDef& gamma_node = match.inputs[gamma_idx].node;
          CHECK_EQ("Const", gamma_node.op());

          // We have a set of vectors that we want to combine into a vector of
          // scale values to apply column-wise to the weight input to the conv,
          // and an offset vector that we'll apply to the output of the conv.
          Tensor weights = GetNodeTensorAttr(weights_node, "value");
          Tensor mean = GetNodeTensorAttr(mean_node, "value");
          Tensor variance = GetNodeTensorAttr(variance_node, "value");
          Tensor beta = GetNodeTensorAttr(beta_node, "value");
          Tensor gamma = GetNodeTensorAttr(gamma_node, "value");
          const float variance_epsilon =
              batch_norm_node.attr().at(epsilon_attr).f();

          // Make sure all the inputs really are vectors, with as many entries
          // as there are columns in the weights.
          const int64 weights_cols = weights.shape().dim_size(3);
          TF_RETURN_IF_ERROR(ErrorIfNotVector(mean, "Mean", weights_cols));
          TF_RETURN_IF_ERROR(
              ErrorIfNotVector(variance, "Variance", weights_cols));
          TF_RETURN_IF_ERROR(ErrorIfNotVector(beta, "Beta", weights_cols));
          TF_RETURN_IF_ERROR(ErrorIfNotVector(gamma, "gamma", weights_cols));

          // Calculate the scale and offset values to apply.
          std::vector<float> scale_values(weights_cols);
          std::vector<float> offset_values(weights_cols);
          if (scale_after_normalization) {
            for (int i = 0; i < weights_cols; ++i) {
              scale_values[i] =
                  (1.0f / sqrtf(variance.flat<float>()(i) + variance_epsilon)) *
                  gamma.flat<float>()(i);
            }
          } else {
            for (int i = 0; i < weights_cols; ++i) {
              scale_values[i] =
                  (1.0f / sqrtf(variance.flat<float>()(i) + variance_epsilon));
            }
          }
          for (int i = 0; i < weights_cols; ++i) {
            offset_values[i] = (-mean.flat<float>()(i) * scale_values[i]) +
                               beta.flat<float>()(i);
          }

          // Multiply the original weights by the scale vector.
          auto weights_matrix = weights.flat_inner_dims<float>();
          Tensor scaled_weights(DT_FLOAT, weights.shape());
          auto scaled_weights_matrix = scaled_weights.flat_inner_dims<float>();
          for (int64 row = 0; row < weights_matrix.dimension(0); ++row) {
            for (int64 col = 0; col < weights_cols; ++col) {
              scaled_weights_matrix(row, col) =
                  weights_matrix(row, col) * scale_values[col];
            }
          }
          // Figure out the remaining bias to add on.
          Tensor bias_offset(DT_FLOAT, {weights_cols});
          auto bias_offset_vector = bias_offset.flat<float>();
          for (int64 col = 0; col < weights_cols; ++col) {
            bias_offset_vector(col) = offset_values[col];
          }

          // Construct the new nodes.
          NodeDef scaled_weights_node;
          scaled_weights_node.set_op("Const");
          scaled_weights_node.set_name(weights_node.name());
          SetNodeAttr("dtype", DT_FLOAT, &scaled_weights_node);
          SetNodeTensorAttr<float>("value", scaled_weights,
                                   &scaled_weights_node);
          new_nodes->push_back(scaled_weights_node);

          // The input and convolution can be copied straight over, since the
          // name of the scaled weights constant is the same as the original.
          new_nodes->push_back(input_node);
          new_nodes->push_back(conv_node);

          NodeDef bias_offset_node;
          bias_offset_node.set_op("Const");
          bias_offset_node.set_name(conv_node.name() + "_bn_offset");
          SetNodeAttr("dtype", DT_FLOAT, &bias_offset_node);
          SetNodeTensorAttr<float>("value", bias_offset, &bias_offset_node);
          new_nodes->push_back(bias_offset_node);

          NodeDef bias_add_node;
          bias_add_node.set_op("BiasAdd");
          bias_add_node.set_name(batch_norm_node.name());
          CopyNodeAttr(conv_node, "T", "T", &bias_add_node);
          AddNodeInput(conv_node.name(), &bias_add_node);
          AddNodeInput(bias_offset_node.name(), &bias_add_node);
          new_nodes->push_back(bias_add_node);

          did_graph_change = true;

          return Status::OK();
        },
        {}, &replaced_graph_def));
    current_graph_def = replaced_graph_def;
  } while (did_graph_change);
  *output_graph_def = current_graph_def;
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("fold_old_batch_norms", FoldOldBatchNorms);

}  // namespace graph_transforms
}  // namespace tensorflow
