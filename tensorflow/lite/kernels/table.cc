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
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
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
  TF_LITE_ENSURE(context,
                 output->type == kTfLiteInt8 || output->type == kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, table->type);

  if (input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
  }
  if (output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }

  TF_LITE_ENSURE_EQ(context, NumDimensions(table), 1);
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, NumElements(table), lut_size<int8_t>());
  } else {
    TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt16);
    TF_LITE_ENSURE_EQ(context, NumElements(table), lut_size<int16_t>());
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <typename InputT, typename OutputT>
void Table(TfLiteContext* context, const TfLiteTensor* input,
           const TfLiteTensor* table, TfLiteTensor* output) {
  const InputT* input_data = GetTensorData<InputT>(input);
  const OutputT* table_data = GetTensorData<OutputT>(table);
  OutputT* output_data = GetTensorData<OutputT>(output);

  const int num_elements = NumElements(input);
  for (int i = 0; i < num_elements; i++) {
    // No need to rescale the input and output, the rescaling and its zero-point
    // are implicitly included into the table data during its generation.
    output_data[i] = lut_lookup(input_data[i], table_data);
  }
}

template <typename InputT>
TfLiteStatus EvalTable(TfLiteContext* context, const TfLiteTensor* input,
                       const TfLiteTensor* table, TfLiteTensor* output) {
  switch (output->type) {
    case kTfLiteInt8:
      Table<InputT, int8_t>(context, input, table, output);
      break;
    case kTfLiteInt16:
      Table<InputT, int16_t>(context, input, table, output);
      break;
    default:
      TF_LITE_UNSUPPORTED_TYPE(context, output->type, "Table");
  }

  return kTfLiteOk;
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
      return EvalTable<int8_t>(context, input, table, output);
    case kTfLiteInt16:
      return EvalTable<int16_t>(context, input, table, output);
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
