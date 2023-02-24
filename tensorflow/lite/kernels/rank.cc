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
#include <stdint.h>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace rank {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = kTfLiteInt32;

  // By design, the input shape is always known at the time of Prepare, even
  // if the preceding op that generates |input| is dynamic. Thus, we can
  // always compute the rank immediately, without waiting for Eval.
  SetTensorToPersistentRo(output);

  // Rank produces a 0-D int32 Tensor representing the rank of input.
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(0);
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_size));

  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 0);

  // Immediately propagate the known rank to the output tensor. This allows
  // downstream ops that rely on the value to use it during prepare.
  if (output->type == kTfLiteInt32) {
    int32_t* output_data = GetTensorData<int32_t>(output);
    *output_data = NumDimensions(input);
  } else {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
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
