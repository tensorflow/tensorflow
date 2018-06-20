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
#include <algorithm>
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"
#include "tensorflow/contrib/lite/string_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace cast {
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // TODO(ahentz): these two checks would make the new implementation
  // incompatible with some existing models, where params is not specified. It
  // is OK not to have them because toco would have set input and output types
  // to match the parameters.
  // auto* params = reinterpret_cast<TfLiteCastParams*>(node->builtin_data);
  // TF_LITE_ENSURE_EQ(context, input->type, params->in_data_type);
  // TF_LITE_ENSURE_EQ(context, output->type, params->out_data_type);

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <typename FromT, typename ToT>
void copyCast(const FromT* in, ToT* out, int num_elements) {
  std::transform(in, in + num_elements, out,
                 [](FromT a) { return static_cast<ToT>(a); });
}

template <typename FromT>
TfLiteStatus copyToTensor(const FromT* in, TfLiteTensor* out,
                          int num_elements) {
  switch (out->type) {
    case kTfLiteInt64:
      copyCast(in, out->data.i64, num_elements);
      break;
    case kTfLiteInt32:
      copyCast(in, out->data.i32, num_elements);
      break;
    case kTfLiteUInt8:
      copyCast(in, out->data.uint8, num_elements);
      break;
    case kTfLiteFloat32:
      copyCast(in, out->data.f, num_elements);
      break;
    case kTfLiteBool:
      copyCast(in, out->data.b, num_elements);
      break;
    default:
      // Unsupported type.
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const int num_elements = NumElements(input);
  TF_LITE_ENSURE_EQ(context, num_elements, NumElements(output));
  switch (input->type) {
    case kTfLiteInt64:
      return copyToTensor(input->data.i64, output, num_elements);
    case kTfLiteInt32:
      return copyToTensor(input->data.i32, output, num_elements);
    case kTfLiteUInt8:
      return copyToTensor(input->data.uint8, output, num_elements);
    case kTfLiteFloat32:
      return copyToTensor(input->data.f, output, num_elements);
    case kTfLiteBool:
      return copyToTensor(input->data.b, output, num_elements);
    default:
      // Unsupported type.
      return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace cast

TfLiteRegistration* Register_CAST() {
  static TfLiteRegistration r = {nullptr, nullptr, cast::Prepare, cast::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
