/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_shift_left {
namespace {

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

template <typename DataType>
TfLiteStatus EvalImpl(const TfLiteTensor* operand1,
                      const TfLiteTensor* operand2, TfLiteTensor* result) {
  const int num_elements = NumElements(result);
  const DataType* input1 = GetTensorData<DataType>(operand1);
  const DataType* input2 = GetTensorData<DataType>(operand2);
  DataType* output = GetTensorData<DataType>(result);

  for (int i = 0; i < num_elements; ++i) {
    output[i] = input1[i] << input2[i];
  }

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);
  output->type = input1->type;
  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input1->dims));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TfLiteType data_type = input1->type;

  if (data_type == kTfLiteInt8) {
    return EvalImpl<int8_t>(input1, input2, output);
  } else if (data_type == kTfLiteInt16) {
    return EvalImpl<int16_t>(input1, input2, output);
  } else if (data_type == kTfLiteInt32) {
    return EvalImpl<int32_t>(input1, input2, output);
  } else {
    TF_LITE_KERNEL_LOG(context, "(Index Type: %s) currently not supported.\n",
                       TfLiteTypeGetName(data_type));
    return kTfLiteError;
  }
}

}  // namespace
}  // namespace stablehlo_shift_left

TfLiteRegistration* Register_STABLEHLO_SHIFT_LEFT() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 stablehlo_shift_left::Prepare,
                                 stablehlo_shift_left::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
