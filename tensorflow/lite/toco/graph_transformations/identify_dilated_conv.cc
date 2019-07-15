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
#include <string>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

// A dilated convolution can be emulated with a regular convolution by chaining
// SpaceToBatch and BatchToSpace ops before and after it:
//
//     SpaceToBatchND -> Conv2D -> BatchToSpaceND
//
// This method was common before Conv2D fully supported dilated convolution in
// TensorFlow. This transformation detects this "emulation", and replaces it
// with a true dilated convolution, eliminating the SpaceToBatch and
// BatchtoSpace ops.
//
// Detecting this alone would be relatively easy. However, in practice some
// extra ops are used, so we detect the following patterns:
//
//
//   SpaceToBatchND -> Expand -> Conv2D -> Squeeze -> BatchToSpaceND -> BiasAdd
//
//   SpaceToBatchND -> Expand -> Conv2D -> Squeeze -> Pad -> BatchToSpaceND ->
//   BiasAdd
//
//   SpaceToBatchND -> Expand -> Conv2D -> Squeeze -> BiasAdd -> BatchToSpaceND
//
//   SpaceToBatchND -> Conv2D -> Pad -> BatchToSpaceND -> BiasAdd
//
//   SpaceToBatchND -> Conv2D -> BatchToSpaceND -> BiasAdd
//
//
// The Expand/Squeeze combination is used to adapt a 3D array (such as in
// WaveNet) to the 4D arrays that Conv2D requires. Padding and BiasAdd are
// thrown in just for the extra headache. Padding adapts non-conforming input
// sizes, and can be discarded. The bias is necessary, so is kept.

template <typename T>
bool ResolveDilatedConv(Model* model, Operator* conv_base_op, Operator* stb_op,
                        Operator* post_stb_op, bool has_expand_op,
                        int dilation_factor) {
  auto* conv_op = static_cast<T*>(conv_base_op);
  if (conv_op->inputs.size() != 2) {
    // The conv op must only have weights, no bias.
    return false;
  }
  CHECK_EQ(conv_op->outputs.size(), 1);

  // Squeeze Op
  auto* post_conv_op = GetOpWithInput(*model, conv_op->outputs[0]);
  if (!post_conv_op) {
    return false;
  }
  if (has_expand_op) {
    if (post_conv_op->type != OperatorType::kSqueeze) {
      // If an expand op was used, the post-conv op must be a squeeze op
      return false;
    }
    CHECK_EQ(post_conv_op->inputs.size(), 1);
    CHECK_EQ(post_conv_op->outputs.size(), 1);
  }

  // Pad Op
  const auto* pad_op = has_expand_op
                           ? GetOpWithInput(*model, post_conv_op->outputs[0])
                           : GetOpWithInput(*model, conv_op->outputs[0]);
  bool has_pad_op = false;
  if (pad_op->type == OperatorType::kPad) {
    has_pad_op = true;
    CHECK_EQ(pad_op->inputs.size(), 2);
    CHECK_EQ(pad_op->outputs.size(), 1);
  }
  // TODO(mjmatthews): Perform validity checking on padding dimensions.

  // Pre-BatchToSpace Bias Op
  auto* next_op = has_pad_op
                      ? GetOpWithInput(*model, pad_op->outputs[0])
                      : has_expand_op
                            ? GetOpWithInput(*model, post_conv_op->outputs[0])
                            : GetOpWithInput(*model, conv_op->outputs[0]);
  bool has_bias_before_bts = false;
  if (next_op->type == OperatorType::kAdd) {
    has_bias_before_bts = true;
  }
  auto final_op = GetOpWithInput(*model, next_op->outputs[0]);

  // BatchToSpace Op
  const auto* bts_op = has_bias_before_bts ? final_op : next_op;
  if (bts_op->type != OperatorType::kBatchToSpaceND) {
    return false;
  }
  CHECK_EQ(bts_op->inputs.size(), 3);
  CHECK_EQ(bts_op->outputs.size(), 1);

  // Post-BatchToSpace Bias Op
  Operator* bias_add_op = !has_bias_before_bts ? final_op : next_op;
  if (bias_add_op->type != OperatorType::kAdd) {
    // Bias op is required before or after BatchToSpace
    return false;
  }
  CHECK_EQ(bias_add_op->inputs.size(), 2);
  CHECK_EQ(bias_add_op->outputs.size(), 1);

  // 2. RE-WIRE OPERATORS
  // ***************************************************************************
  // Re-use the existing Conv2D op.
  conv_op->dilation_width_factor = dilation_factor;
  conv_op->dilation_height_factor = dilation_factor;
  conv_op->padding.type = PaddingType::kSame;

  // Rewire the ops to bypass SpaceToBatch, BatchToSpace, and Pad.
  bias_add_op->outputs[0] = final_op->outputs[0];
  if (has_expand_op) {
    bias_add_op->inputs[0] = post_conv_op->outputs[0];
    post_conv_op->inputs[0] = conv_op->outputs[0];
    conv_op->inputs[0] = post_stb_op->outputs[0];
    post_stb_op->inputs[0] = stb_op->inputs[0];
  } else {
    bias_add_op->inputs[0] = conv_op->outputs[0];
    conv_op->inputs[0] = stb_op->inputs[0];
  }
  // TODO(mjmatthews): Connect bias directly into the Conv2D?

  // 3. DELETE LEFTOVER OPERATORS
  // ***************************************************************************
  DeleteOpAndArrays(model, bts_op);
  DeleteOpAndArrays(model, stb_op);
  if (has_pad_op) {
    DeleteOpAndArrays(model, pad_op);
  }

  return true;
}

::tensorflow::Status IdentifyDilatedConv::Run(Model* model,
                                              std::size_t op_index,
                                              bool* modified) {
  *modified = false;
  const auto it = model->operators.begin() + op_index;
  auto* stb_op = it->get();

  // 1. IDENTIFY OPERATORS
  // ***************************************************************************
  // SpaceToBatch Op.
  if (stb_op->type != OperatorType::kSpaceToBatchND) {
    return ::tensorflow::Status::OK();
  }
  if (stb_op->inputs.size() != 3) {
    return ::tensorflow::Status::OK();
  }
  CHECK_EQ(stb_op->outputs.size(), 1);
  // Extract the dilation factor from Input[1] of SpaceToBatch
  // TODO(mjmatthews): Support 2D dilation factors.
  const auto& block_shape_array = model->GetArray(stb_op->inputs[1]);
  if (!block_shape_array.buffer) {
    return ::tensorflow::Status::OK();
  }
  CHECK_EQ(block_shape_array.shape().dimensions_count(), 1);
  int dilation_factor =
      block_shape_array.Array::GetBuffer<ArrayDataType::kInt32>().data[0];

  // Expand Op
  auto* post_stb_op = GetOpWithInput(*model, stb_op->outputs[0]);
  if (!post_stb_op) {
    return ::tensorflow::Status::OK();
  }
  bool has_expand_op = false;
  if (post_stb_op->type == OperatorType::kExpandDims) {
    has_expand_op = true;
    CHECK_EQ(post_stb_op->inputs.size(), 2);
    CHECK_EQ(post_stb_op->outputs.size(), 1);
  }

  // Conv Op
  const string& input_of_conv_op =
      has_expand_op ? post_stb_op->outputs[0] : stb_op->outputs[0];
  auto* conv_base_op = GetOpWithInput(*model, input_of_conv_op);
  bool changed = false;
  if (conv_base_op->type == OperatorType::kConv) {
    changed = ResolveDilatedConv<ConvOperator>(model, conv_base_op, stb_op,
                                               post_stb_op, has_expand_op,
                                               dilation_factor);
    if (changed) {
      LOG(INFO) << "Replaced sub-network with Dilated Conv2D op outputting \""
                << conv_base_op->outputs[0] << "\".";
    }
  } else if (identify_depthwise_conv_ &&
             conv_base_op->type == OperatorType::kDepthwiseConv) {
    changed = ResolveDilatedConv<DepthwiseConvOperator>(
        model, conv_base_op, stb_op, post_stb_op, has_expand_op,
        dilation_factor);
    if (changed) {
      LOG(INFO)
          << "Replaced sub-netork with Dilated DepthwiseConv2D op outputting \""
          << conv_base_op->outputs[0] << "\".";
    }
  }

  *modified = changed;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
