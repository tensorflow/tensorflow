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
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/lut.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace table {

constexpr int kInputTensor = 0;
constexpr int kTable = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* table;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kTable, &table));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE(context,
                 input->type == kTfLiteInt8 || input->type == kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, table->type);

  if (input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }

  TF_LITE_ENSURE_EQ(context, NumDimensions(table), 1);
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, NumElements(table), LUTSize<int8_t>());
  } else {
    TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt16);
    TF_LITE_ENSURE_EQ(context, NumElements(table), LUTSize<int16_t>());
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* table;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kTable, &table));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (input->type) {
    case kTfLiteInt8:
      reference_integer_ops::LookupTable(
          GetTensorData<int8_t>(input),
          MatchingFlatSize(GetTensorShape(input), GetTensorShape(output)),
          GetTensorData<int8_t>(table), GetTensorData<int8_t>(output));
      return kTfLiteOk;
    case kTfLiteInt16:
      reference_integer_ops::LookupTable(
          GetTensorData<int16_t>(input),
          MatchingFlatSize(GetTensorShape(input), GetTensorShape(output)),
          GetTensorData<int16_t>(table), GetTensorData<int16_t>(output));
      return kTfLiteOk;
    default:
      TF_LITE_UNSUPPORTED_TYPE(context, input->type, "Table");
  }

  return kTfLiteOk;
}

}  // namespace table

TfLiteRegistration* Register_TABLE() {
  static TfLiteRegistration r = {nullptr, nullptr, table::Prepare, table::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
