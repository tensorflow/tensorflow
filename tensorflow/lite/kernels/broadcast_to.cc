/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/broadcast_to.h"

#include <string.h>

#include <cstdint>
#include <memory>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace broadcastto {

constexpr int kInputTensor = 0;
constexpr int kShapeTensor = 1;
constexpr int kOutputTensor = 0;
constexpr int kMaxDims = 8;

struct BroadcastToContext {
  BroadcastToContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, kInputTensor);
    shape = GetInput(context, node, kShapeTensor);
    output = GetOutput(context, node, kOutputTensor);
  }
  const TfLiteTensor* input;
  const TfLiteTensor* shape;
  TfLiteTensor* output;
};

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                BroadcastToContext* op_context) {
  // Ensures the shape is 1D tensor.
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context->shape), 1);

  // Ensure output dims is not less than input dims.
  int input_num_dims = NumDimensions(op_context->input);
  int output_num_dims = SizeOfDimension(op_context->shape, 0);
  TF_LITE_ENSURE_MSG(context, input_num_dims <= output_num_dims,
                     "Output shape must be broadcastable from input shape.");
  TF_LITE_ENSURE_MSG(context, output_num_dims <= kMaxDims,
                     "BroadcastTo only supports 1-8D tensor.");

  // Check if output shape is broadcastable from input shape.
  auto get_shape_data = [op_context](int i) -> int32_t {
    if (op_context->shape->type == kTfLiteInt32) {
      return GetTensorData<int32_t>(op_context->shape)[i];
    } else {
      return GetTensorData<int64_t>(op_context->shape)[i];
    }
  };

  int extending_dims = output_num_dims - input_num_dims;
  for (int idx = 0; idx < input_num_dims; ++idx) {
    TF_LITE_ENSURE_MSG(context,
                       (SizeOfDimension(op_context->input, idx) == 1 ||
                        SizeOfDimension(op_context->input, idx) ==
                            get_shape_data(extending_dims + idx)),
                       "Output shape must be broadcastable from input shape.");
  }
  // Resizing the shape of the output tensor.
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(output_num_dims);
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)>
      scoped_output_shape(output_shape, TfLiteIntArrayFree);
  for (int idx = 0; idx < output_num_dims; ++idx) {
    output_shape->data[idx] = get_shape_data(idx);
  }

  return context->ResizeTensor(context, op_context->output,
                               scoped_output_shape.release());
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, NumInputs(node) == 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TF_LITE_ENSURE_MSG(context,
                     (NumDimensions(GetInput(context, node, 0)) <= kMaxDims),
                     "BroadcastTo only supports 1-8D tensor.");

  BroadcastToContext op_context(context, node);
  TF_LITE_ENSURE(context, op_context.shape->type == kTfLiteInt32 ||
                              op_context.shape->type == kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);

  // Not yet support string type due to the use of memcopy with fixed size.
  TF_LITE_ENSURE(context, op_context.input->type != kTfLiteString);

  if (IsConstantOrPersistentTensor(op_context.shape)) {
    return ResizeOutputTensor(context, &op_context);
  }

  SetTensorToDynamic(op_context.output);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  BroadcastToContext op_context(context, node);
  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  }

  // BroadcastTo op support upto 8 dims, matching the support of Tensorflow.
  reference_ops::BroadcastTo<kMaxDims>(
      GetTensorShape(op_context.input), op_context.input->data.raw,
      GetTensorShape(op_context.output), op_context.output->data.raw,
      op_context.input->type);
  return kTfLiteOk;
}

}  // namespace broadcastto

TfLiteRegistration* Register_BROADCAST_TO() {
  static TfLiteRegistration r = {nullptr, nullptr, broadcastto::Prepare,
                                 broadcastto::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
