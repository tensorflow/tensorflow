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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace micro {
namespace reshape {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus ReshapeOutput(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  // Tensorflow's Reshape allows one of the shape components to have the
  // special -1 value, meaning it will be calculated automatically based on the
  // input. Here we calculate what that dimension should be so that the number
  // of output elements in the same as the number of input elements.
  int num_input_elements = NumElements(input);
  TfLiteIntArray* output_shape = output->dims;

  if (NumInputs(node) == 1 &&  // Legacy scalar supported with params.
      output_shape->size == 1 && output_shape->data[0] == 0) {
    // Legacy tflite models use a shape parameter of [0] to indicate scalars,
    // so adjust accordingly. TODO(b/111614235): Allow zero-sized buffers during
    // toco conversion.
    output_shape->size = 0;
  }

  int num_output_elements = 1;
  int stretch_dim = -1;
  for (int i = 0; i < output_shape->size; ++i) {
    int value = output_shape->data[i];
    if (value == -1) {
      TF_LITE_ENSURE_EQ(context, stretch_dim, -1);
      stretch_dim = i;
    } else {
      num_output_elements *= value;
    }
  }
  if (stretch_dim != -1) {
    output_shape->data[stretch_dim] = num_input_elements / num_output_elements;
    num_output_elements *= output_shape->data[stretch_dim];
  }

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_EQ(context, num_input_elements, num_output_elements);
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, NumInputs(node) == 1 || NumInputs(node) == 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  if (ReshapeOutput(context, node) != kTfLiteOk) {
    return kTfLiteError;
  }

  for (size_t i = 0; i < input->bytes; ++i) {
    output->data.raw[i] = input->data.raw[i];
  }
  return kTfLiteOk;
}

}  // namespace reshape

TfLiteRegistration* Register_RESHAPE() {
  static TfLiteRegistration r = {nullptr, nullptr, reshape::Prepare,
                                 reshape::Eval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
