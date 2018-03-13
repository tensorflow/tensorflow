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

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
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

bool IdentifyDilatedConv::Run(Model* model, std::size_t op_index) {
  const auto it = model->operators.begin() + op_index;
  auto* stb_op = it->get();

  // 1. IDENTIFY OPERATORS
  // ***************************************************************************
  // SpaceToBatch Op.
  if (stb_op->type != OperatorType::kSpaceToBatchND) {
    return false;
  }
  if (stb_op->inputs.size() != 3) {
    return false;
  }
  CHECK_EQ(stb_op->outputs.size(), 1);
  // Extract the dilation factor from Input[1] of SpaceToBatch
  // TODO(mjmatthews): Support 2D dilation factors.
  const auto& block_shape_array = model->GetArray(stb_op->inputs[1]);
  if (!block_shape_array.buffer) {
    return false;
  }
  CHECK_EQ(block_shape_array.shape().dimensions_count(), 1);
  int dilation_factor =
      block_shape_array.Array::GetBuffer<ArrayDataType::kInt32>().data[0];

  // Expand Op
  auto* post_stb_op = GetOpWithInput(*model, stb_op->outputs[0]);
  if (!post_stb_op) {
    return false;
  }
  bool has_expand_op = false;
  if (post_stb_op->type == OperatorType::kExpandDims) {
    has_expand_op = true;
    CHECK_EQ(post_stb_op->inputs.size(), 2);
    CHECK_EQ(post_stb_op->outputs.size(), 1);
  }

  // Conv Op
  ConvOperator* conv_op = dynamic_cast<ConvOperator*>(
      has_expand_op ? GetOpWithInput(*model, post_stb_op->outputs[0])
                    : GetOpWithInput(*model, stb_op->outputs[0]));
  if (!conv_op || conv_op->type != OperatorType::kConv) {
    return false;
  }
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

  LOG(INFO) << "Identified sub-network emulating dilated convolution.";

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
  // Order is important. Delete the output array first, then the op, then it's
  // redundant inputs.
  // BatchToSpace Op
  DeleteArrayIfUnused(bts_op->outputs[0], model);
  std::vector<string> bts_op_inputs = bts_op->inputs;
  model->operators.erase(FindOp(*model, bts_op));
  DeleteArrayIfUnused(bts_op_inputs[1], model);
  DeleteArrayIfUnused(bts_op_inputs[2], model);

  // Pad Op if present
  if (has_pad_op) {
    DeleteArrayIfUnused(pad_op->outputs[0], model);
    std::vector<string> pad_op_inputs = pad_op->inputs;
    model->operators.erase(FindOp(*model, pad_op));
    DeleteArrayIfUnused(pad_op_inputs[1], model);
  }

  // SpaceToBatch Op
  DeleteArrayIfUnused(stb_op->outputs[0], model);
  std::vector<string> stb_op_inputs = stb_op->inputs;
  model->operators.erase(FindOp(*model, stb_op));
  DeleteArrayIfUnused(stb_op_inputs[1], model);
  DeleteArrayIfUnused(stb_op_inputs[2], model);

  LOG(INFO) << "Replaced with Dilated Conv2D op outputting \""
            << conv_op->outputs[0] << "\".";
  return true;
}

}  // namespace toco
