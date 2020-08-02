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
#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace where {

constexpr int kInputConditionTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const TfLiteTensor* cond_tensor,
                                TfLiteTensor* output_tensor) {
  // Output tensor should have shape:
  // (num_true, cond_rank), where num_true denotes the number of true values
  // in condition.
  const RuntimeShape& cond_shape = GetTensorShape(cond_tensor);
  const int size = cond_shape.FlatSize();
  const int cond_rank = cond_shape.DimensionsCount();
  const bool* cond_data = GetTensorData<bool>(cond_tensor);

  int true_count = 0;
  for (int i = 0; i < size; ++i) {
    if (cond_data[i]) {
      true_count++;
    }
  }
  TfLiteIntArray* output_dims = TfLiteIntArrayCreate(2);
  output_dims->data[0] = true_count;
  output_dims->data[1] = cond_rank;
  return context->ResizeTensor(context, output_tensor, output_dims);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* cond_tensor =
      GetInput(context, node, kInputConditionTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  if (cond_tensor->type != kTfLiteBool) {
    context->ReportError(context,
                         "Condition tensor must be of type bool, but saw '%s'.",
                         TfLiteTypeGetName(cond_tensor->type));
    return kTfLiteError;
  }

  // As output will be a 2D tensor of indices, use int64 to be consistent with
  // tensorflow.
  output->type = kTfLiteInt64;

  // Exit early if cond is a non-const tensor. Set output tensor to dynamic so
  // output size can be determined in Eval.
  if (!IsConstantTensor(cond_tensor)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  return ResizeOutputTensor(context, cond_tensor, output);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* cond_tensor =
      GetInput(context, node, kInputConditionTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutputTensor(context, cond_tensor, output));
  }

  reference_ops::SelectTrueCoords(GetTensorShape(cond_tensor),
                                  GetTensorData<bool>(cond_tensor),
                                  GetTensorData<int64_t>(output));
  return kTfLiteOk;
}
}  // namespace where

TfLiteRegistration* Register_WHERE() {
  static TfLiteRegistration r = {/*init*/ nullptr, /*free*/ nullptr,
                                 where::Prepare, where::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
