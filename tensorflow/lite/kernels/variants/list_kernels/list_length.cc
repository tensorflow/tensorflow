/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"

namespace tflite {
namespace variants {
namespace ops {
namespace list_length {
namespace {

using ::tflite::variants::TensorArray;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &list_input));
  TF_LITE_ENSURE_TYPES_EQ(context, list_input->type, kTfLiteVariant);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, output->dims->size, 0);

  output->allocation_type = kTfLiteArenaRw;

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &list_input));
  TF_LITE_ENSURE_EQ(context, list_input->allocation_type, kTfLiteVariantObject);
  const TensorArray* const input_arr =
      reinterpret_cast<TensorArray*>(list_input->data.data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

  const int length = input_arr->NumElements();
  output->data.i32[0] = length;

  return kTfLiteOk;
}
}  // namespace
}  // namespace list_length

TfLiteRegistration* Register_LIST_LENGTH() {
  static TfLiteRegistration r = {nullptr, nullptr, list_length::Prepare,
                                 list_length::Eval};
  return &r;
}

}  // namespace ops
}  // namespace variants
}  // namespace tflite
