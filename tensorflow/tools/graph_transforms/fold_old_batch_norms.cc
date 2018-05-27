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
    return errors::InvalidArgument(
        input_name,
        " input to batch norm has bad shape: ", input.shape().DebugString());
  }
  return Status::OK();
}

Status GetScaleAndOffsetValues(const NodeMatch& match,
                               std::vector<float>* scale_values,
                               std::vector<float>* offset_values) {
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
      is_fused || batch_norm_node.attr().at("scale_after_normalization").b();

  const NodeDef& mean_node = match.inputs[mean_idx].node;
  CHECK_EQ("Const", mean_node.op());
  const NodeDef& variance_node = match.inputs[var_idx].node;
  CHECK_EQ("Const", variance_node.op());
  const NodeDef& beta_node = match.inputs[beta_idx].node;
  CHECK_EQ("Const", beta_node.op());
  const NodeDef& gamma_node = match.inputs[gamma_idx].node;
  CHECK_EQ("Const", gamma_node.op());

  // We have a set of vectors that we want to combine into a vector of
  // scale values and offset values.
  Tensor mean = GetNodeTensorAttr(mean_node, "value");
  Tensor variance = GetNodeTensorAttr(variance_node, "value");
  Tensor beta = GetNodeTensorAttr(beta_node, "value");
  Tensor gamma = GetNodeTensorAttr(gamma_node, "value");
  const float variance_epsilon = batch_norm_node.attr().at(epsilon_attr).f();

  // Make sure all the inputs really are vectors with the same shape.
  const int64 num_cols = mean.shape().dim_size(0);
  TF_RETURN_IF_ERROR(ErrorIfNotVector(variance, "Variance", num_cols));
  TF_RETURN_IF_ERROR(ErrorIfNotVector(beta, "Beta", num_cols));
  TF_RETURN_IF_ERROR(ErrorIfNotVector(gamma, "gamma", num_cols));

  scale_values->resize(num_cols);
  offset_values->resize(num_cols);

  // Calculate the scale and offset values to apply.
  if (scale_after_normalization) {
    for (int i = 0; i < num_cols; ++i) {
      (*scale_values)[i] =
          (1.0f / sqrtf(variance.flat<float>()(i) + variance_epsilon)) *
          gamma.flat<float>()(i);
    }
  } else {
    for (int i = 0; i < num_cols; ++i) {
      (*scale_values)[i] =
          (1.0f / sqrtf(variance.flat<float>()(i) + variance_epsilon));
    }
  }
  for (int i = 0; i < num_cols; ++i) {
    (*offset_values)[i] =
        (-mean.flat<float>()(i) * (*scale_values)[i]) + beta.flat<float>()(i);
  }
  return Status::OK();
}

Status FuseScaleOffsetToConvWeights(const std::vector<float>& scale_values,
                                    const std::vector<float>& offset_values,
                                    const NodeMatch& conv_node_match,
                                    const string& conv_output_name,
                                    std::vector<NodeDef>* new_nodes) {
  const NodeDef& conv_node = conv_node_match.node;
  CHECK_EQ("Conv2D", conv_node.op());
  const NodeDef& input_node = conv_node_match.inputs[0].node;
  const NodeDef& weights_node = conv_node_match.inputs[1].node;
  CHECK_EQ("Const", weights_node.op());

  Tensor weights = GetNodeTensorAttr(weights_node, "value");
  const int64 weights_cols = weights.shape().dim_size(3);
  CHECK_EQ(weights_cols, scale_values.size());

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
  SetNodeTensorAttr<float>("value", scaled_weights, &scaled_weights_node);
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
  bias_add_node.set_name(conv_output_name);
  if (!conv_node.attr().count("data_format")) {
    CopyNodeAttr(conv_node, "data_format", "data_format", &bias_add_node);
  }
  CopyNodeAttr(conv_node, "T", "T", &bias_add_node);
  AddNodeInput(conv_node.name(), &bias_add_node);
  AddNodeInput(bias_offset_node.name(), &bias_add_node);
  new_nodes->push_back(bias_add_node);
  return Status::OK();
}

Status FuseBatchNormWithConv(const NodeMatch& match,
                             std::vector<NodeDef>* new_nodes) {
  // Calculate the scale and offset values to apply.
  std::vector<float> scale_values;
  std::vector<float> offset_values;
  TF_RETURN_IF_ERROR(
      GetScaleAndOffsetValues(match, &scale_values, &offset_values));

  // Fuse conv weights, and set the final output node name as batch_norm_node.
  const NodeDef& batch_norm_node = match.node;
  TF_RETURN_IF_ERROR(
      FuseScaleOffsetToConvWeights(scale_values, offset_values, match.inputs[0],
                                   batch_norm_node.name(), new_nodes));
  return Status::OK();
}

Status FuseBatchNormWithBatchToSpace(const NodeMatch& match,
                             std::vector<NodeDef>* new_nodes) {
  // Calculate the scale and offset values to apply.
  std::vector<float> scale_values;
  std::vector<float> offset_values;
  TF_RETURN_IF_ERROR(
      GetScaleAndOffsetValues(match, &scale_values, &offset_values));

  // Fuse conv weights, and set the final output node name as batch_norm_node.
  const NodeDef& batch_norm_node = match.node;
  const NodeMatch& batch_to_space_node_match = match.inputs[0];
  const NodeMatch& conv_node_match = batch_to_space_node_match.inputs[0];
  const NodeDef& batch_to_space_node = batch_to_space_node_match.node;
  const NodeDef& conv_node = conv_node_match.node;

  string biasadd_name = conv_node.name() + "/biasadd";
  TF_RETURN_IF_ERROR(
      FuseScaleOffsetToConvWeights(scale_values, offset_values, conv_node_match,
                                   biasadd_name , new_nodes));

  NodeDef new_batch_to_space_node = batch_to_space_node;
  // reuse batch_norm node name
  new_batch_to_space_node.set_name(batch_norm_node.name());
  new_batch_to_space_node.set_input(0, biasadd_name);
  new_nodes->push_back(batch_to_space_node_match.inputs[1].node);
  new_nodes->push_back(batch_to_space_node_match.inputs[2].node);
  new_nodes->push_back(new_batch_to_space_node);
  return Status::OK();
}

Status FuseBatchNormWithConvConcat(const NodeMatch& match,
                                   std::vector<NodeDef>* new_nodes) {
  // Calculate the scale and offset values to apply.
  std::vector<float> scale_values;
  std::vector<float> offset_values;
  TF_RETURN_IF_ERROR(
      GetScaleAndOffsetValues(match, &scale_values, &offset_values));

  // Find all the nodes we expect in the subgraph.
  const NodeDef& batch_norm_node = match.node;
  const NodeMatch& concat_node_match = match.inputs[0];
  NodeDef concat_node = concat_node_match.node;
  CHECK_EQ("ConcatV2", concat_node.op());

  // First process the axis.
  NodeDef axis_node = concat_node_match.inputs[2].node;
  CHECK_EQ("Const", axis_node.op());
  Tensor axis = GetNodeTensorAttr(axis_node, "value");
  int32 axis_scalar = (axis.scalar<int32>())();

  // Set both conv0 and conv1 have the same scale and offset in default.
  std::vector<float> scale0(scale_values);
  std::vector<float> offset0(offset_values);
  std::vector<float> scale1(scale_values);
  std::vector<float> offset1(offset_values);
  if (axis_scalar == 3) {
    // If axis is 3, then scale and offset will be split into two halfs.
    const NodeDef& weights0_node = concat_node_match.inputs[0].inputs[1].node;
    Tensor weights0 = GetNodeTensorAttr(weights0_node, "value");
    const int64 split_cols = weights0.shape().dim_size(3);
    // Only keep the first half for scale0/offset0.
    scale0.erase(scale0.begin() + split_cols, scale0.end());
    offset0.erase(offset0.begin() + split_cols, offset0.end());
    // Only keep the second half for scale1/offset1.
    scale1.erase(scale1.begin(), scale1.begin() + split_cols);
    offset1.erase(offset1.begin(), offset1.begin() + split_cols);
  }

  // Fuse the weights for input0 of conv2d.
  const string concat0_output_name = concat_node.name() + "_bn_in0";
  TF_RETURN_IF_ERROR(
      FuseScaleOffsetToConvWeights(scale0, offset0, concat_node_match.inputs[0],
                                   concat0_output_name, new_nodes));

  // Fuse the weights for input1 of conv2d.
  const string concat1_output_name = concat_node.name() + "_bn_in1";
  TF_RETURN_IF_ERROR(
      FuseScaleOffsetToConvWeights(scale1, offset1, concat_node_match.inputs[1],
                                   concat1_output_name, new_nodes));

  // Push the shape node.
  new_nodes->push_back(concat_node_match.inputs[2].node);

  // Set the final output op name to batch_normal_node.
  concat_node.set_name(batch_norm_node.name());
  concat_node.set_input(0, concat0_output_name);
  concat_node.set_input(1, concat1_output_name);
  new_nodes->push_back(concat_node);
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
          TF_RETURN_IF_ERROR(FuseBatchNormWithConv(match, new_nodes));
          did_graph_change = true;
          return Status::OK();
        },
        {}, &replaced_graph_def));
    current_graph_def = replaced_graph_def;
  } while (did_graph_change);

  do {
    did_graph_change = false;
    GraphDef replaced_graph_def;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def,  // clang-format off
        {"BatchNormWithGlobalNormalization|FusedBatchNorm",    // batch_norm_node
         {
             {"BatchToSpaceND",                  // batch_to_space_node
              {
                  {"Conv2D",                     // conv_node
                   {
                       {"*"},                    // input_node
                       {"Const"},                // weights_node
                   }
                  },
                  {"Const"},                     // block_shape
                  {"Const"},                     // crops
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
          TF_RETURN_IF_ERROR(FuseBatchNormWithBatchToSpace(match, new_nodes));
          did_graph_change = true;
          return Status::OK();
        },
        {}, &replaced_graph_def));
    current_graph_def = replaced_graph_def;
  } while (did_graph_change);

  do {
    did_graph_change = false;
    GraphDef replaced_graph_def;
    // Replace BatchNorm with concat as input.
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def,  // clang-format off
      {"BatchNormWithGlobalNormalization|FusedBatchNorm",    // batch_norm_node
        {
          {"ConcatV2|Concat",                     // concat two conv2d.
            {
              {"Conv2D",                          // conv_node
                {
                  {"*"},                          // input_node
                  {"Const"},                      // weights_node
                }
              },
              {"Conv2D",                          // conv_node
                {
                  {"*"},                          // input_node
                  {"Const"},                      // weights_node
                }
              },
              {"Const"},                          // axis
            },
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
          TF_RETURN_IF_ERROR(FuseBatchNormWithConvConcat(match, new_nodes));
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
