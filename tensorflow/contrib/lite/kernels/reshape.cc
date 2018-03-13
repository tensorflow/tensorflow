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
#include <string.h>
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace reshape {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteReshapeParams*>(node->builtin_data);

  // TODO(ahentz): we are often given a tensor with the shape but we only pay
  // attention to what the shape specified in 'params'.
  TF_LITE_ENSURE(context, NumInputs(node) == 1 || NumInputs(node) == 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Tensorflow's Reshape allows one of the shape components to have the
  // special -1 value, meaning it will be calculated automatically based on the
  // input. Here we calculate what that dimension should be so that the number
  // of output elements in the same as the number of input elements.
  int num_input_elements = 1;
  for (int i = 0; i < NumDimensions(input); ++i) {
    num_input_elements *= SizeOfDimension(input, i);
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(params->num_dimensions);
  int num_output_elements = 1;
  int stretch_dim = -1;
  for (int i = 0; i < params->num_dimensions; ++i) {
    int value = params->shape[i];
    if (value == -1) {
      TF_LITE_ENSURE_EQ(context, stretch_dim, -1);
      stretch_dim = i;
    } else {
      num_output_elements *= value;
      output_size->data[i] = value;
    }
  }
  if (stretch_dim != -1) {
    output_size->data[stretch_dim] = num_input_elements / num_output_elements;
    num_output_elements *= output_size->data[stretch_dim];
  }

  TF_LITE_ENSURE_EQ(context, num_input_elements, num_output_elements);
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  memcpy(output->data.raw, input->data.raw, input->bytes);

  return kTfLiteOk;
}

}  // namespace reshape

TfLiteRegistration* Register_RESHAPE() {
  static TfLiteRegistration r = {nullptr, nullptr, reshape::Prepare,
                                 reshape::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
