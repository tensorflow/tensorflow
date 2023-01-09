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
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource/lookup_interfaces.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {

namespace hashtable {

constexpr int kInputResourceIdTensor = 0;
constexpr int kKeyTensor = 1;
constexpr int kDefaultValueTensor = 2;
constexpr int kOutputTensor = 0;

TfLiteStatus PrepareHashtableFind(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input_resource_id_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputResourceIdTensor,
                                          &input_resource_id_tensor));
  TF_LITE_ENSURE_EQ(context, input_resource_id_tensor->type, kTfLiteResource);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_resource_id_tensor), 1);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(input_resource_id_tensor, 0), 1);

  const TfLiteTensor* default_value_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDefaultValueTensor,
                                          &default_value_tensor));

  const TfLiteTensor* key_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kKeyTensor, &key_tensor));
  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputTensor, &output_tensor));
  TF_LITE_ENSURE_EQ(context, default_value_tensor->type, output_tensor->type);
  TF_LITE_ENSURE(context, (key_tensor->type == kTfLiteInt64 &&
                           output_tensor->type == kTfLiteString) ||
                              (key_tensor->type == kTfLiteString &&
                               output_tensor->type == kTfLiteInt64));
  return context->ResizeTensor(context, output_tensor,
                               TfLiteIntArrayCopy(key_tensor->dims));
}

TfLiteStatus EvalHashtableFind(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_resource_id_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputResourceIdTensor,
                                          &input_resource_id_tensor));
  int resource_id = input_resource_id_tensor->data.i32[0];

  const TfLiteTensor* key_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kKeyTensor, &key_tensor));
  const TfLiteTensor* default_value_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDefaultValueTensor,
                                          &default_value_tensor));
  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output_tensor));

  Subgraph* subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto& resources = subgraph->resources();
  auto* lookup = resource::GetHashtableResource(&resources, resource_id);
  TF_LITE_ENSURE(context, lookup != nullptr);
  TF_LITE_ENSURE_STATUS(
      lookup->CheckKeyAndValueTypes(context, key_tensor, output_tensor));
  auto result =
      lookup->Lookup(context, key_tensor, output_tensor, default_value_tensor);
  return result;
}

}  // namespace hashtable

TfLiteRegistration* Register_HASHTABLE_FIND() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 hashtable::PrepareHashtableFind,
                                 hashtable::EvalHashtableFind};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
