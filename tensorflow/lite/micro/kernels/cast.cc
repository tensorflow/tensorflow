/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  return kTfLiteOk;
}

template <typename FromT, typename ToT>
void copyCast(const FromT* in, ToT* out, int num_elements) {
  std::transform(in, in + num_elements, out,
                 [](FromT a) { return static_cast<ToT>(a); });
}

template <typename FromT>
TfLiteStatus copyToTensor(TfLiteContext* context, const FromT* in,
                          TfLiteEvalTensor* out, int num_elements) {
  switch (out->type) {
    case kTfLiteInt8:
      copyCast(in, out->data.int8, num_elements);
      break;
    case kTfLiteFloat32:
      copyCast(in, tflite::micro::GetTensorData<float>(out), num_elements);
      break;
    default:
      // Unsupported type.
      TF_LITE_KERNEL_LOG(context, "Output type %s (%d) not supported.",
                         TfLiteTypeGetName(out->type), out->type);
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  int num_elements = MatchingFlatSize(tflite::micro::GetTensorShape(input),
                                      tflite::micro::GetTensorShape(output));

  switch (input->type) {
    case kTfLiteInt8:
      return copyToTensor(context, input->data.int8, output, num_elements);
    case kTfLiteFloat32:
      return copyToTensor(context, tflite::micro::GetTensorData<float>(input),
                          output, num_elements);
    default:
      // Unsupported type.
      TF_LITE_KERNEL_LOG(context, "Input type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
  }
  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration Register_CAST() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
