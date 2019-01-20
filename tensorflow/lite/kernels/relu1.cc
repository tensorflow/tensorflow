/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace relu1 {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TfLiteTensor* output = GetOutput(context, node, 0);
  output->type = input->type;
  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

// This is derived from lite/kernels/activations.cc.
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  const int elements = NumElements(input);
  const float* in = input->data.f;
  const float* in_end = in + elements;
  float* out = output->data.f;
  for (; in < in_end; ++in, ++out) {
    *out = std::min(std::max(0.f, *in), 1.f);
  }
  return kTfLiteOk;
}

}  // namespace relu1

TfLiteRegistration* Register_RELU_1() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 relu1::Prepare, relu1::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
