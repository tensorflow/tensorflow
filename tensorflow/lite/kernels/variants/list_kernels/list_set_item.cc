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
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace variants {
namespace ops {
namespace {

constexpr int kListInput = 0;
constexpr int kIndexInput = 1;
constexpr int kItemInput = 2;
constexpr int kOutput = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kListInput, &list_input));
  TF_LITE_ENSURE_TYPES_EQ(context, list_input->type, kTfLiteVariant);

  const TfLiteTensor* index_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kIndexInput, &index_input));
  TF_LITE_ENSURE_TYPES_EQ(context, index_input->type, kTfLiteInt32);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutput, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteVariant);
  output->allocation_type = kTfLiteVariantObject;

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kListInput, &list_input));
  TF_LITE_ENSURE_EQ(context, list_input->allocation_type, kTfLiteVariantObject);

  TensorArray* input_arr =
      reinterpret_cast<TensorArray*>(list_input->data.data);

  const TfLiteTensor* index_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kIndexInput, &index_input));
  const int* index_data = GetTensorData<int>(index_input);
  TF_LITE_ENSURE(context,
                 index_data != nullptr && index_input->bytes == sizeof(int));
  const int index = *index_data;
  TF_LITE_ENSURE(context, index >= 0);

  const TfLiteTensor* item_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kItemInput, &item_input));
  TF_LITE_ENSURE_TYPES_EQ(context, input_arr->ElementType(), item_input->type);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutput, &output));

  TensorArray* output_arr = static_cast<TensorArray*>(
      input_arr->CloneTo(static_cast<VariantData*>(output->data.data)));

  // TODO(b/288302706) Skip copy when tensor is used only once.
  TensorUniquePtr item_copy = BuildTfLiteTensor(
      item_input->type, BuildTfLiteArray(*item_input->dims), kTfLiteDynamic);
  TfLiteTensorCopy(item_input, item_copy.get());

  if (index >= output_arr->NumElements()) {
    output_arr->Resize(index + 1);
  }
  TF_LITE_ENSURE(context, output_arr->Set(index, std::move(item_copy)));
  output->data.data = static_cast<VariantData*>(output_arr);
  return kTfLiteOk;
}

}  // namespace
TfLiteRegistration* Register_LIST_SET_ITEM() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare, Eval};
  return &r;
}
}  // namespace ops
}  // namespace variants
}  // namespace tflite
