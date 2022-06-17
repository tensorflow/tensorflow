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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status ConvertPureConvToDepthwise::Run(Model* model,
                                                     std::size_t op_index,
                                                     bool* modified) {
  *modified = false;
  auto conv_it = model->operators.begin() + op_index;
  if (conv_it->get()->type != OperatorType::kConv) {
    return ::tensorflow::OkStatus();
  }
  const auto* conv_op = static_cast<ConvOperator*>(conv_it->get());
  if (conv_op->stride_width != conv_op->stride_height) {
    return ::tensorflow::OkStatus();
  }
  if ((conv_op->dilation_width_factor != 1) ||
      (conv_op->dilation_height_factor != 1)) {
    // Depthwise conv does not support dilation
    return ::tensorflow::OkStatus();
  }
  auto& input_array = model->GetArray(conv_op->inputs[0]);
  if (!input_array.has_shape()) {
    // Shapes not propagated yet
    return ::tensorflow::OkStatus();
  }
  if (input_array.shape().dims(3) != 1) {
    // Not a pure convolution: Conv does accumulation across the depth
    // dimension.
    return ::tensorflow::OkStatus();
  }

  const auto& weights_name = conv_op->inputs[1];
  if (CountOpsWithInput(*model, weights_name) > 1) {
    // TODO(yunluli): Come up with a way to do the weights shuffling only once.
    AddMessageF(
        "Not changing %s to DepthwiseConv because the weights is consumed by "
        "another op.",
        LogName(*conv_op));
    return ::tensorflow::OkStatus();
  }
  auto& weights_array = model->GetArray(weights_name);
  if (!weights_array.buffer) {
    // Yield until the weights are resolved as a constant array.
    return ::tensorflow::OkStatus();
  }
  if (weights_array.data_type != ArrayDataType::kFloat) {
    return ::tensorflow::OkStatus();
  }
  // At this point we know we have a pure conv. Rewrite it as DepthwiseConv.
  AddMessageF(
      "%s is purely convolutional (input/weights depth is 1), replacing it by "
      "a DepthwiseConv.",
      LogName(*conv_op));
  auto* depthwiseconv_op = new DepthwiseConvOperator;
  // Conv and DepthwiseConv take the same inputs
  depthwiseconv_op->inputs = conv_op->inputs;
  // Conv may have a 2nd output for im2col
  depthwiseconv_op->outputs = {conv_op->outputs[0]};
  if (conv_op->outputs.size() > 1) {
    // delete the im2col array.
    model->EraseArray(conv_op->outputs[1]);
  }
  depthwiseconv_op->fused_activation_function =
      conv_op->fused_activation_function;
  // Let PropagateFixedSizes recompute fixed padding, just in case some day it
  // may be different for Conv vs DepthwiseConv.
  depthwiseconv_op->padding.type = conv_op->padding.type;
  depthwiseconv_op->stride_height = conv_op->stride_height;
  depthwiseconv_op->stride_width = conv_op->stride_width;
  depthwiseconv_op->depth_multiplier = weights_array.shape().dims(0);
  // Replace the operator in the graph.
  model->operators.emplace(conv_it, depthwiseconv_op);
  DeleteOpAndArrays(model, conv_op);
  // Shuffle the weights.
  const auto& weights_shape = weights_array.shape();
  auto& weights_buffer =
      weights_array.GetMutableBuffer<ArrayDataType::kFloat>();
  const std::vector<float>& conv_weights_data = weights_buffer.data;
  std::vector<float> depthwise_conv_weights_data(conv_weights_data.size());
  const int depth = weights_shape.dims(0);
  const int width = weights_shape.dims(1);
  const int height = weights_shape.dims(2);
  const int width_height = width * height;
  for (int c = 0; c < depth; c++) {
    for (int xy = 0; xy < width_height; xy++) {
      depthwise_conv_weights_data[c + depth * xy] =
          conv_weights_data[xy + width_height * c];
    }
  }
  *weights_array.mutable_shape()->mutable_dims() = {1, width, height, depth};
  weights_buffer.data = depthwise_conv_weights_data;
  *modified = true;
  return ::tensorflow::OkStatus();
}

}  // namespace toco
