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

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

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

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  auto* params = reinterpret_cast<TfLiteShapeParams*>(node->builtin_data);
  switch (params->out_type) {
    case kTfLiteInt32:
      output->type = kTfLiteInt32;
      break;
    case kTfLiteInt64:
      output->type = kTfLiteInt64;
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Unknown shape output data type: %d",
                         params->out_type);
      return kTfLiteError;
  }

  // By design, the input shape is always known at the time of Prepare, even
  // if the preceding op that generates |input| is dynamic. Thus, we can
  // always compute the shape immediately, without waiting for Eval.
  SetTensorToPersistentRo(output);

  // Shape always produces a 1-dimensional output tensor, where each output
  // element is the length of the corresponding input tensor's dimension.
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(1);
  output_size->data[0] = NumDimensions(input);
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_size));

  TFLITE_DCHECK_EQ(NumDimensions(output), 1);
  TFLITE_DCHECK_EQ(SizeOfDimension(output, 0), NumDimensions(input));

  // Immediately propagate the known shape to the output tensor. This allows
  // downstream ops that rely on the value to use it during prepare.
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

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
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
