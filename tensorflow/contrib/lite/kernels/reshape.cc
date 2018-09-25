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
#include "tensorflow/contrib/lite/c/builtin_op_data.h"
#include "tensorflow/contrib/lite/c/c_api_internal.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace reshape {

constexpr int kInputTensor = 0;
constexpr int kShapeTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus ResizeOutput(TfLiteContext* context, TfLiteNode* node,
                          TfLiteIntArray* output_shape) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Tensorflow's Reshape allows one of the shape components to have the
  // special -1 value, meaning it will be calculated automatically based on the
  // input. Here we calculate what that dimension should be so that the number
  // of output elements in the same as the number of input elements.
  int num_input_elements = NumElements(input);

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

  TF_LITE_ENSURE_EQ(context, num_input_elements, num_output_elements);
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus ResizeOutputWithShapeTensor(TfLiteContext* context,
                                         TfLiteNode* node) {
  const TfLiteTensor* shape = GetInput(context, node, kShapeTensor);

  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(shape->dims->data[0]);
  for (int i = 0; i < output_shape->size; ++i) {
    output_shape->data[i] = shape->data.i32[i];
  }
  return ResizeOutput(context, node, output_shape);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteReshapeParams*>(node->builtin_data);

  TF_LITE_ENSURE(context, NumInputs(node) == 1 || NumInputs(node) == 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Attempt to use shape tensor if it exists.
  if (NumInputs(node) == 2) {
    const TfLiteTensor* shape = GetInput(context, node, kShapeTensor);
    // Check if the shape tensor is valid.
    if (shape->dims->size == 1 && shape->type == kTfLiteInt32) {
      // Set the output tensor as dynamic if the shape isn't constnat.
      if (!IsConstantTensor(shape)) {
        TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
        SetTensorToDynamic(output);
        return kTfLiteOk;
      }
      // Shape is constant. Resize now.
      return ResizeOutputWithShapeTensor(context, node);
    }
  }
  // The function is returned above this line if the shape tensor is usable.
  // Now fallback to the shape parameter in `TfLiteReshapeParams`.
  int num_dimensions = params->num_dimensions;
  if (num_dimensions == 1 && params->shape[0] == 0) {
    // Legacy tflite models use a shape parameter of [0] to indicate scalars,
    // so adjust accordingly. TODO(b/111614235): Allow zero-sized buffers during
    // toco conversion.
    num_dimensions = 0;
  }
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(num_dimensions);
  for (int i = 0; i < num_dimensions; ++i) {
    output_shape->data[i] = params->shape[i];
  }
  return ResizeOutput(context, node, output_shape);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputWithShapeTensor(context, node));
  }

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
