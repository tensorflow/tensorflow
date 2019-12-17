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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace micro {
namespace expand_dims {

constexpr int k_input_tensor = 0;
constexpr int k_axis_tensor = 1;
constexpr int k_output_tensor = 0;

// TODO(See TfLiteSqueezeParams): We can't have dynamic data, at least not yet.
// For now we will fix the maximum possible number of dimensions.
constexpr int max_num_dims = 8;

struct ExpandDimsContext {
  ExpandDimsContext(TfLiteContext* context, TfLiteNode* node)
      : input(GetInput(context, node, k_input_tensor)),
        axis(GetInput(context, node, k_axis_tensor)),
        output(GetOutput(context, node, k_output_tensor)) {}
  const TfLiteTensor* const input;
  const TfLiteTensor* const axis;
  TfLiteTensor* output;
};

TfLiteStatus ExpandTensorDim(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteTensor* input, int axis,
                             TfLiteTensor* output) {
  const TfLiteIntArray* input_dims = input->dims;
  TF_LITE_ENSURE(context, (0 < input_dims->size) &&
                              (input_dims->size <= (max_num_dims - 1)));
  if (axis < 0) {
    axis = input_dims->size + 1 + axis;
  }
  TF_LITE_ENSURE(context, (axis <= input_dims->size) && (axis >= 0));

  // Allocate new output_dims from node's temporaries buffer
  TfLiteIntArray* output_dims = node->temporaries;
  output_dims->size = input_dims->size + 1;
  for (int i = 0; i < output_dims->size; ++i) {
    if (i < axis) {
      output_dims->data[i] = input_dims->data[i];
    } else if (i == axis) {
      output_dims->data[i] = 1;
    } else {
      output_dims->data[i] = input_dims->data[i - 1];
    }
  }

  output->dims = output_dims;

  return kTfLiteOk;
}

TfLiteStatus GetAxisValueFromTensor(TfLiteContext* context,
                                    const TfLiteTensor* axis, int* axis_value) {
  TF_LITE_ENSURE_EQ(context, NumElements(axis), 1);
  switch (axis->type) {
    case kTfLiteInt32:
      *axis_value = *GetTensorData<int32_t>(axis);
      return kTfLiteOk;
    case kTfLiteInt64:
      *axis_value = *GetTensorData<int64_t>(axis);
      return kTfLiteOk;
    default:
      return kTfLiteError;
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  ExpandDimsContext op_context(context, node);
  int axis_value = 0;
  if (GetAxisValueFromTensor(context, op_context.axis, &axis_value) !=
      kTfLiteOk) {
    return kTfLiteError;
  }

  if (ExpandTensorDim(context, node, op_context.input, axis_value,
                      op_context.output) != kTfLiteOk) {
    return kTfLiteError;
  }
  // Just copy input to output.
  for (int i = 0; i < op_context.input->bytes; ++i) {
    op_context.output->data.raw[i] = op_context.input->data.raw[i];
  }
  return kTfLiteOk;
}

}  // namespace expand_dims

TfLiteRegistration* Register_EXPAND_DIMS() {
  static TfLiteRegistration r = {nullptr, nullptr, expand_dims::Prepare,
                                 expand_dims::Eval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
