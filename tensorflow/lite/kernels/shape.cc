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
namespace shape {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

template <typename OutType>
void ExtractShape(const TfLiteTensor* input, OutType* output_data) {
  for (int i = 0; i < NumDimensions(input); ++i) {
    output_data[i] = SizeOfDimension(input, i);
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  auto* params = reinterpret_cast<TfLiteShapeParams*>(node->builtin_data);
  switch (params->out_type) {
    case kTfLiteInt32:
      output->type = kTfLiteInt32;
      break;
    case kTfLiteInt64:
      output->type = kTfLiteInt64;
      break;
    default:
      context->ReportError(context, "Unknown shape output data type: %d",
                           params->out_type);
      return kTfLiteError;
  }

  // Shape always produces a 1-dimensional output tensor, where each output
  // element is the length of the corresponding input tensor's dimension.
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(1);
  output_size->data[0] = NumDimensions(input);
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TFLITE_DCHECK_EQ(NumDimensions(output), 1);
  TFLITE_DCHECK_EQ(SizeOfDimension(output, 0), NumDimensions(input));

  switch (output->type) {
    case kTfLiteInt32:
      ExtractShape(input, GetTensorData<int32_t>(output));
      break;
    case kTfLiteInt64:
      ExtractShape(input, GetTensorData<int64_t>(output));
      break;
    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace shape

TfLiteRegistration* Register_SHAPE() {
  static TfLiteRegistration r = {nullptr, nullptr, shape::Prepare, shape::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
