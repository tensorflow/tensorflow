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
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace rank {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  output->type = kTfLiteInt32;

  // Rank produces a 0-D int32 Tensor representing the rank of input.
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(0);
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 0);

  if (output->type == kTfLiteInt32) {
    int32_t* output_data = GetTensorData<int32_t>(output);
    *output_data = NumDimensions(input);
  } else {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace rank

TfLiteRegistration* Register_RANK() {
  static TfLiteRegistration r = {nullptr, nullptr, rank::Prepare, rank::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
