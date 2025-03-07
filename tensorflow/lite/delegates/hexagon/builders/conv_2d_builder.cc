/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/hexagon/builders/conv_2d_builder.h"

#include <stdint.h>

#include <cmath>
#include <vector>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/hexagon/builders/op_builder.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
namespace {

// Channel count to split depthwise convolution op.
// See Conv2dOpBuilder.should_split_dwconv_ for details.
constexpr int kDwConv5x5Filt2x2StrideChannelCount = 32;

// Dilated Depthwise Convolution performs SpaceToBatchND & BatchToSpaceND before
// and after the op respectively.
// This helper computes the paddings param for SpaceToBatchND and crops param
// for BatchToSpaceND.
//
// Inspired by tf.nn.with_space_to_batch & tf.required_space_to_batch_paddings.
void ComputeSpaceToBatchParams(int input_height, int input_width,
                               int weights_height, int weights_width,
                               const std::vector<int>& dilation_factors_h_w,
                               const TfLitePadding padding_type,
                               std::vector<int>* paddings,
                               std::vector<int>* crops) {
  // Base paddings depend on padding applied to the Depthwise Conv op.
  // 4-element array: {top, bottom, left, right}.
  std::vector<int> base_paddings(4, 0);
  if (padding_type == kTfLitePaddingSame) {
    const int dilated_weights_h =
        dilation_factors_h_w[0] * (weights_height - 1) + 1;
    const int dilated_weights_w =
        dilation_factors_h_w[1] * (weights_width - 1) + 1;
    base_paddings[0] = (dilated_weights_h - 1) / 2;
    base_paddings[1] = dilated_weights_h - 1 - (dilated_weights_h - 1) / 2;
    base_paddings[2] = (dilated_weights_w - 1) / 2;
    base_paddings[3] = dilated_weights_w - 1 - (dilated_weights_w - 1) / 2;
  }

  // paddings represents {pad_top, pad_bottom, pad_left, pad_right}.
  paddings->resize(4, 0);
  // crops represents {crop_top, crop_bottom, crop_left, crop_right}.
  crops->resize(4, 0);

  // Logic for computing paddings & crops follows.
  // Taken from tf.required_space_to_batch_paddings, but without array
  // operations since we only deal with 2 dimensions.
  int pad_start_h = base_paddings[0];
  int pad_start_w = base_paddings[2];
  int orig_pad_end_h = base_paddings[1];
  int orig_pad_end_w = base_paddings[3];
  int full_input_h = input_height + pad_start_h + orig_pad_end_h;
  int full_input_w = input_width + pad_start_w + orig_pad_end_w;
  int pad_end_extra_h =
      (dilation_factors_h_w[0] - full_input_h % dilation_factors_h_w[0]) %
      dilation_factors_h_w[0];
  int pad_end_extra_w =
      (dilation_factors_h_w[1] - full_input_w % dilation_factors_h_w[1]) %
      dilation_factors_h_w[1];
  int pad_end_h = orig_pad_end_h + pad_end_extra_h;
  int pad_end_w = orig_pad_end_w + pad_end_extra_w;

  // Assign values.
  (*paddings)[0] = pad_start_h;
  (*paddings)[1] = pad_end_h;
  (*paddings)[2] = pad_start_w;
  (*paddings)[3] = pad_end_w;
  (*crops)[0] = 0;
  (*crops)[1] = pad_end_extra_h;
  (*crops)[2] = 0;
  (*crops)[3] = pad_end_extra_w;
}

// Computes output dimensions for the SpaceToBatchND op used in the dilated
// Depthwise Conv case.
// space_to_batch_paddings should be in format {top, bottom, left, right}.
// These are computed from the documentation for SpaceToBatchND_8's output.
void PopulateSpaceToBatchOutputDims(
    int input_batch_size, int input_height_size, int input_width_size,
    int input_depth_size, const std::vector<int>& dilation_factors_h_w,
    const std::vector<int>& space_to_batch_paddings,
    std::vector<int>* space_to_batch_output_dims) {
  // Batches.
  space_to_batch_output_dims->push_back(
      input_batch_size * dilation_factors_h_w[0] * dilation_factors_h_w[1]);
  // Height.
  space_to_batch_output_dims->push_back((space_to_batch_paddings[0] +
                                         input_height_size +
                                         space_to_batch_paddings[1]) /
                                        dilation_factors_h_w[0]);
  // Width.
  space_to_batch_output_dims->push_back((space_to_batch_paddings[2] +
                                         input_width_size +
                                         space_to_batch_paddings[3]) /
                                        dilation_factors_h_w[1]);
  // Depth.
  space_to_batch_output_dims->push_back(input_depth_size);
}

}  // namespace

void Conv2dOpBuilder::BuildDilatedDwConv(
    const TfLiteIntArray* inputs, const TfLiteTensor& data_tensor,
    const TfLiteTensor& output_data_tensor, OpBuilder* data_min_const,
    OpBuilder* data_max_const, OpBuilder* conv_output_min_const,
    OpBuilder* conv_output_max_const, OpBuilder* stride_node, int stride_height,
    const TfLitePadding padding_type, TensorID* output_tensor,
    TensorID* output_min_tensor, TensorID* output_max_tensor) {
  static std::vector<int> dilation_factors_shape = {1, 1, 1, 2};
  static std::vector<int> paddings_shape = {1, 1, 2, 2};
  // Output dimensions.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, output_data_tensor.dims);
  // For dilated Depthwise Conv, we convert this node into SpaceToBatchND, and
  // then chain Supernode & BatchToSpaceND after it.
  int input_batch_size, input_height_size, input_width_size, input_depth_size;
  GetDims(&input_batch_size, &input_height_size, &input_width_size,
          &input_depth_size, data_tensor.dims);
  ComputeSpaceToBatchParams(input_height_size, input_width_size,
                            weight_shape_[0], weight_shape_[1],
                            dilation_factors_h_w_, padding_type,
                            &space_to_batch_paddings_, &batch_to_space_crops_);
  auto* dilation_factors_const = graph_builder_->AddConstNodeWithData(
      dilation_factors_shape.data(),
      reinterpret_cast<char*>(dilation_factors_h_w_.data()),
      dilation_factors_h_w_.size() * sizeof(stride_height));
  auto* paddings_const = graph_builder_->AddConstNodeWithData(
      paddings_shape.data(),
      reinterpret_cast<char*>(space_to_batch_paddings_.data()),
      space_to_batch_paddings_.size() * sizeof(stride_height));
  auto* crops_const = graph_builder_->AddConstNodeWithData(
      paddings_shape.data(),
      reinterpret_cast<char*>(batch_to_space_crops_.data()),
      batch_to_space_crops_.size() * sizeof(stride_height));

  // 1. SpaceToBatch.
  SetOpType(OP_SpaceToBatchND_8);
  AddInput(graph_builder_->GetHexagonTensorId(inputs->data[0]));
  AddInput(TensorID(dilation_factors_const->GetID(), 0));
  AddInput(TensorID(paddings_const->GetID(), 0));
  AddInput(TensorID(data_min_const->GetID(), 0));
  AddInput(TensorID(data_max_const->GetID(), 0));
  std::vector<int> space_to_batch_output_dims;
  PopulateSpaceToBatchOutputDims(
      input_batch_size, input_height_size, input_width_size, input_depth_size,
      dilation_factors_h_w_, space_to_batch_paddings_,
      &space_to_batch_output_dims);
  TensorID space_to_batch_op_out =
      AddOutput(sizeof(uint8_t), 4, space_to_batch_output_dims);
  AddOutput(sizeof(float), 4, kScalarShape);
  AddOutput(sizeof(float), 4, kScalarShape);

  // 2. Depthwise Conv.
  auto* conv_op = graph_builder_->AddNode(GetTFLiteNodeID());
  conv_op->SetOpType(OP_DepthwiseSupernode_8x8p32to8);
  conv_op->AddInput(space_to_batch_op_out);
  conv_op->AddInput(graph_builder_->GetHexagonTensorId(inputs->data[1]));
  conv_op->AddInput(TensorID(data_min_const->GetID(), 0));
  conv_op->AddInput(TensorID(data_max_const->GetID(), 0));
  conv_op->AddInput(TensorID(weights_min_node_->GetID(), 0));
  conv_op->AddInput(TensorID(weights_max_node_->GetID(), 0));
  conv_op->AddInput(TensorID(stride_node->GetID(), 0));
  conv_op->AddInput(graph_builder_->GetHexagonTensorId(inputs->data[2]));
  conv_op->AddInput(TensorID(bias_min_node_->GetID(), 0));
  conv_op->AddInput(TensorID(bias_max_node_->GetID(), 0));
  conv_op->AddInput(TensorID(conv_output_min_const->GetID(), 0));
  conv_op->AddInput(TensorID(conv_output_max_const->GetID(), 0));
  if (per_channel_quant_.channel_scales_node != nullptr) {
    conv_op->AddInput(
        TensorID(per_channel_quant_.channel_scales_node->GetID(), 0));
  }
  // The padding is handled by the SpaceToBatch/BatchToSpace ops surrounding
  // this node. Hence, this op's padding remains VALID only.
  // tf.nn.with_space_to_batch's docs state the following pattern:
  // """
  // batch_to_space_nd(
  //  op(space_to_batch_nd(input, adjusted_dilation_rate, adjusted_paddings),
  //     num_spatial_dims,
  //     "VALID")
  //  adjusted_dilation_rate,
  //  adjusted_crops)
  // """
  conv_op->SetPaddingType(NN_PAD_VALID);
  // These dimensions are probably a little excessive, but they upper-bound
  // the possible output from DepthwiseConv.
  // TODO(b/139955809): Find better bounds?
  TensorID conv_output = conv_op->AddOutput(
      sizeof(uint8_t), 4,
      {output_batch_size * dilation_factors_h_w_[0] * dilation_factors_h_w_[1],
       output_height_size, output_width_size, output_depth_size});
  conv_op->AddOutput(sizeof(float), 4, kScalarShape);
  conv_op->AddOutput(sizeof(float), 4, kScalarShape);

  // 3. BatchToSpace.
  auto* batch_to_space_op = graph_builder_->AddNode(GetTFLiteNodeID());
  batch_to_space_op->SetOpType(OP_BatchToSpaceND_8);
  batch_to_space_op->AddInput(conv_output);
  batch_to_space_op->AddInput(TensorID(dilation_factors_const->GetID(), 0));
  batch_to_space_op->AddInput(TensorID(crops_const->GetID(), 0));
  batch_to_space_op->AddInput(TensorID(conv_output_min_const->GetID(), 0));
  batch_to_space_op->AddInput(TensorID(conv_output_max_const->GetID(), 0));
  *output_tensor =
      batch_to_space_op->AddOutput(sizeof(uint8_t), 4,
                                   {output_batch_size, output_height_size,
                                    output_width_size, output_depth_size});
  *output_min_tensor =
      batch_to_space_op->AddOutput(sizeof(float), 4, kScalarShape);
  *output_max_tensor =
      batch_to_space_op->AddOutput(sizeof(float), 4, kScalarShape);
}

// Workaround for depthwise conv accuracy issues.
// See Conv2dOpBuilder.should_split_dwconv_ for details.
void Conv2dOpBuilder::BuildSplittedDwConv(
    const TfLiteIntArray* inputs, const TfLiteTensor& data_tensor,
    const TfLiteTensor& output_data_tensor, OpBuilder* data_min_const,
    OpBuilder* data_max_const, OpBuilder* conv_output_min_const,
    OpBuilder* conv_output_max_const, OpBuilder* stride_node,
    const TfLitePadding padding_type, TensorID* output_tensor,
    TensorID* output_min_tensor, TensorID* output_max_tensor) {
  // Input dimensions.
  int input_batch_size, input_height_size, input_width_size, input_depth_size;
  GetDims(&input_batch_size, &input_height_size, &input_width_size,
          &input_depth_size, data_tensor.dims);
  // Output dimensions.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, output_data_tensor.dims);

  auto* split = this;
  split->SetOpType(OP_QuantizedSplit_8);
  int32_t dim_channel = 3;
  auto* dimension_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&dim_channel), sizeof(dim_channel));
  split->AddInput(TensorID(dimension_const->GetID(), 0));
  split->AddInput(graph_builder_->GetHexagonTensorId(inputs->data[0]));
  split->AddInput(TensorID(data_min_const->GetID(), 0));
  split->AddInput(TensorID(data_max_const->GetID(), 0));
  std::vector<TensorID> data_nodes;
  data_nodes.reserve(per_channel_quant_.splits);
  for (auto i = 0; i < per_channel_quant_.splits; i++) {
    data_nodes.emplace_back(
        split->AddOutput(sizeof(uint8_t), 4,
                         {input_batch_size, input_height_size, input_width_size,
                          kDwConv5x5Filt2x2StrideChannelCount}));
  }
  auto data_min = split->AddOutput(sizeof(float), 4, kScalarShape);
  auto data_max = split->AddOutput(sizeof(float), 4, kScalarShape);

  std::vector<TensorID> dconv_outputs, dconv_min, dconv_max;
  for (auto i = 0; i < per_channel_quant_.splits; i++) {
    auto* dw_conv = graph_builder_->AddNode(GetTFLiteNodeID());
    dw_conv->SetOpType(OP_DepthwiseSupernode_8x8p32to8);
    if (padding_type == kTfLitePaddingSame) {
      dw_conv->SetPaddingType(NN_PAD_SAME);
    } else if (padding_type == kTfLitePaddingValid) {
      dw_conv->SetPaddingType(NN_PAD_VALID);
    }
    dw_conv->AddInput(data_nodes[i]);
    dw_conv->AddInput(TensorID(weights_nodes_[i]->GetID(), 0));
    dw_conv->AddInput(data_min);
    dw_conv->AddInput(data_max);
    dw_conv->AddInput(TensorID(weights_min_node_->GetID(), 0));
    dw_conv->AddInput(TensorID(weights_max_node_->GetID(), 0));
    dw_conv->AddInput(TensorID(stride_node->GetID(), 0));
    dw_conv->AddInput(TensorID(bias_nodes_[i]->GetID(), 0));
    dw_conv->AddInput(TensorID(bias_min_node_->GetID(), 0));
    dw_conv->AddInput(TensorID(bias_max_node_->GetID(), 0));
    dw_conv->AddInput(TensorID(conv_output_min_const->GetID(), 0));
    dw_conv->AddInput(TensorID(conv_output_max_const->GetID(), 0));
    dw_conv->AddInput(
        TensorID(per_channel_quant_.channel_scales_nodes[i]->GetID(), 0));
    dconv_outputs.push_back(dw_conv->AddOutput(
        sizeof(uint8_t), 4,
        {output_batch_size, output_height_size, output_width_size,
         kDwConv5x5Filt2x2StrideChannelCount}));
    dconv_min.push_back(dw_conv->AddOutput(sizeof(float), 4, kScalarShape));
    dconv_max.push_back(dw_conv->AddOutput(sizeof(float), 4, kScalarShape));
  }

  auto* concat = graph_builder_->AddNode(GetTFLiteNodeID());
  concat->SetOpType(OP_QuantizedConcat_8);
  concat->AddInput(TensorID(dimension_const->GetID(), 0));
  for (auto i = 0; i < per_channel_quant_.splits; i++) {
    concat->AddInput(dconv_outputs[i]);
  }
  for (auto i = 0; i < per_channel_quant_.splits; i++) {
    concat->AddInput(dconv_min[i]);
  }
  for (auto i = 0; i < per_channel_quant_.splits; i++) {
    concat->AddInput(dconv_max[i]);
  }
  concat->AddInput(TensorID(conv_output_min_const->GetID(), 0));
  concat->AddInput(TensorID(conv_output_max_const->GetID(), 0));
  *output_tensor = concat->AddOutput(sizeof(uint8_t), 4,
                                     {output_batch_size, output_height_size,
                                      output_width_size, output_depth_size});
  *output_min_tensor = concat->AddOutput(sizeof(float), 4, kScalarShape);
  *output_max_tensor = concat->AddOutput(sizeof(float), 4, kScalarShape);
}

void Conv2dOpBuilder::BuildStandardConv(
    const TfLiteIntArray* inputs, const TfLiteTensor& output_data_tensor,
    OpBuilder* data_min_const, OpBuilder* data_max_const,
    OpBuilder* conv_output_min_const, OpBuilder* conv_output_max_const,
    OpBuilder* stride_node, const TfLitePadding padding_type,
    TensorID* output_tensor, TensorID* output_min_tensor,
    TensorID* output_max_tensor) {
  // Standard case.
  // Output dimensions.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, output_data_tensor.dims);
  // Padding type.
  if (padding_type == kTfLitePaddingSame) {
    SetPaddingType(NN_PAD_SAME);
  } else if (padding_type == kTfLitePaddingValid) {
    SetPaddingType(NN_PAD_VALID);
  }
  // Inputs
  AddInput(graph_builder_->GetHexagonTensorId(inputs->data[0]));
  AddInput(graph_builder_->GetHexagonTensorId(inputs->data[1]));
  AddInput(TensorID(data_min_const->GetID(), 0));
  AddInput(TensorID(data_max_const->GetID(), 0));
  AddInput(TensorID(weights_min_node_->GetID(), 0));
  AddInput(TensorID(weights_max_node_->GetID(), 0));
  AddInput(TensorID(stride_node->GetID(), 0));
  AddInput(graph_builder_->GetHexagonTensorId(inputs->data[2]));
  AddInput(TensorID(bias_min_node_->GetID(), 0));
  AddInput(TensorID(bias_max_node_->GetID(), 0));
  AddInput(TensorID(conv_output_min_const->GetID(), 0));
  AddInput(TensorID(conv_output_max_const->GetID(), 0));
  if (per_channel_quant_.channel_scales_node != nullptr) {
    AddInput(TensorID(per_channel_quant_.channel_scales_node->GetID(), 0));
  }
  // Outputs
  *output_tensor = AddOutput(sizeof(uint8_t), 4,
                             {output_batch_size, output_height_size,
                              output_width_size, output_depth_size});
  *output_min_tensor = AddOutput(sizeof(float), 4, kScalarShape);
  *output_max_tensor = AddOutput(sizeof(float), 4, kScalarShape);
}

TfLiteStatus Conv2dOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                               const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
  // Input data tensor.
  const auto& data_tensor = context->tensors[inputs->data[0]];
  const auto& output_data_tensor = context->tensors[outputs->data[0]];
  int input_batch_size, input_height_size, input_width_size, input_depth_size;
  GetDims(&input_batch_size, &input_height_size, &input_width_size,
          &input_depth_size, data_tensor.dims);
  float data_min = 0;
  float data_max = 0;
  TF_LITE_ENSURE_STATUS(
      ComputeMinAndMaxQuantValues(data_tensor, &data_min, &data_max));
  auto* data_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&data_min), sizeof(data_min));
  auto* data_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&data_max), sizeof(data_max));

  // Gather information about the Convolution operations.
  TfLitePadding padding_type = kTfLitePaddingUnknown;
  TfLiteFusedActivation activation = kTfLiteActNone;
  int stride_height = 0;
  int stride_width = 0;
  bool is_dilated_depthwise_conv = false;
  int channel_multiplier = 1;
  if (op_node_.op_type == OP_Supernode_8x8p32to8) {
    const TfLiteConvParams* conv_params =
        reinterpret_cast<const TfLiteConvParams*>(builtin_data_);
    stride_height = conv_params->stride_height;
    stride_width = conv_params->stride_width;
    padding_type = conv_params->padding;
    activation = conv_params->activation;
  } else if (op_node_.op_type == OP_DepthwiseSupernode_8x8p32to8) {
    const TfLiteDepthwiseConvParams* conv_params =
        reinterpret_cast<const TfLiteDepthwiseConvParams*>(builtin_data_);
    stride_height = conv_params->stride_height;
    stride_width = conv_params->stride_width;
    padding_type = conv_params->padding;
    activation = conv_params->activation;
    channel_multiplier = conv_params->depth_multiplier;
    // We only support dilation for DepthwiseConv.
    if (conv_params->dilation_height_factor > 1 ||
        conv_params->dilation_width_factor > 1) {
      is_dilated_depthwise_conv = true;
      dilation_factors_h_w_.push_back(conv_params->dilation_height_factor);
      dilation_factors_h_w_.push_back(conv_params->dilation_width_factor);
    }
  }

  // Weights tensor
  TF_LITE_ENSURE_STATUS(
      InitializeWeightsNodes(inputs, outputs, context, input_depth_size));

  // Stride node.
  static int dummy = 0;
  stride_shape_ = {1, stride_height, stride_width, 1};
  auto* stride_node = graph_builder_->AddConstNodeWithData(
      stride_shape_.data(), reinterpret_cast<char*>(&dummy), sizeof(dummy));

  // Output dimensions.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);
  // Output bounds.
  // TODO(b/129276536): Add support for other activations here. Current
  // implementation assumes None/Relu.
  float output_min = 0;
  float output_max = 0;
  TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
      context->tensors[outputs->data[0]], &output_min, &output_max));
  // These denote the bounds fed to Hexagon's Conv mechanism, which will be
  // different from the TFLite tensor bounds if there is a RELU activation.
  float conv_output_min = output_min;
  float conv_output_max = output_max;
  if (activation == kTfLiteActRelu6) {
    conv_output_min = 0;
    conv_output_max = 6;
  } else if (activation == kTfLiteActReluN1To1) {
    conv_output_min = -1;
    conv_output_max = 1;
  } else if (activation == kTfLiteActRelu) {
    conv_output_min = 0;
  }
  auto* conv_output_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&conv_output_min),
      sizeof(conv_output_min));
  auto* conv_output_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&conv_output_max),
      sizeof(conv_output_max));

  // Bias node.
  TF_LITE_ENSURE_STATUS(InitializeBiasNodes(inputs, outputs, context));

  // TODO(b/143759564): Simplify this method when depth_multiplier support needs
  // generalizing.
  if (channel_multiplier > 1 && input_depth_size == 1) {
    // Depthwise Conv with input_depth == 1 & channel_multiplier > 1 is
    // equivalent to Conv.
    SetOpType(OP_Supernode_8x8p32to8);
  } else if (channel_multiplier > 1) {
    TF_LITE_KERNEL_LOG(
        context, "depth_multiplier > 1 not supported with input_depth > 1");
    return kTfLiteError;
  }

  TensorID output_tensor, output_min_tensor, output_max_tensor;
  if (is_dilated_depthwise_conv) {
    BuildDilatedDwConv(inputs, data_tensor, output_data_tensor, data_min_const,
                       data_max_const, conv_output_min_const,
                       conv_output_max_const, stride_node, stride_height,
                       padding_type, &output_tensor, &output_min_tensor,
                       &output_max_tensor);
  } else if (should_split_dwconv_) {
    BuildSplittedDwConv(inputs, data_tensor, output_data_tensor, data_min_const,
                        data_max_const, conv_output_min_const,
                        conv_output_max_const, stride_node, padding_type,
                        &output_tensor, &output_min_tensor, &output_max_tensor);
  } else {
    BuildStandardConv(inputs, output_data_tensor, data_min_const,
                      data_max_const, conv_output_min_const,
                      conv_output_max_const, stride_node, padding_type,
                      &output_tensor, &output_min_tensor, &output_max_tensor);
  }

  // Requantize if activation was not None & the TFLite tensor's min/max is
  // different (diff > 1e-2) from the RELU bounds.
  const float min_bound_diff = std::abs(conv_output_min - output_min);
  const float max_bound_diff = std::abs(conv_output_max - output_max);
  if (activation != kTfLiteActNone &&
      (min_bound_diff > 0.01 || max_bound_diff > 0.01)) {
    auto* requantized_min_const = graph_builder_->AddConstNodeWithData(
        kScalarShape, reinterpret_cast<char*>(&output_min), sizeof(output_min));
    auto* requantized_max_const = graph_builder_->AddConstNodeWithData(
        kScalarShape, reinterpret_cast<char*>(&output_max), sizeof(output_max));
    auto* requantize_op = graph_builder_->AddNode(GetTFLiteNodeID());
    requantize_op->SetOpType(OP_Requantize_8to8);
    requantize_op->AddInput(output_tensor);
    requantize_op->AddInput(output_min_tensor);
    requantize_op->AddInput(output_max_tensor);
    requantize_op->AddInput(TensorID(requantized_min_const->GetID(), 0));
    requantize_op->AddInput(TensorID(requantized_max_const->GetID(), 0));
    node_output_ =
        requantize_op->AddOutput(sizeof(uint8_t), 4,
                                 {output_batch_size, output_height_size,
                                  output_width_size, output_depth_size});
    requantize_op->AddOutput(sizeof(float), 4, kScalarShape);
    requantize_op->AddOutput(sizeof(float), 4, kScalarShape);
  } else {
    node_output_ = output_tensor;
  }

  return kTfLiteOk;
}

TfLiteStatus Conv2dOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                              TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

Conv2dOpBuilder::~Conv2dOpBuilder() {}

OpBuilder* CreateConv2DBuilder(GraphBuilder* graph_builder, int op_type) {
  return new Conv2dOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
