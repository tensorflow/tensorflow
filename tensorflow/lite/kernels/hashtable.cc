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
#include <string>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource/lookup_interfaces.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace hashtable {

// The current hash table op returns a key of the hash table resource objects,
// shared by the context. Later, this implementation might be updated by sharing
// the actual reference of hash table objects in the tensor buffer.

static constexpr int kResourceHandleTensor = 0;

TfLiteStatus PrepareHashtable(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 0);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TF_LITE_ENSURE(context, node->builtin_data != nullptr);
  const auto* params =
      reinterpret_cast<const TfLiteHashtableParams*>(node->builtin_data);

  TF_LITE_ENSURE(context, (params->key_dtype == kTfLiteInt64 &&
                           params->value_dtype == kTfLiteString) ||
                              (params->key_dtype == kTfLiteString &&
                               params->value_dtype == kTfLiteInt64));

  TfLiteTensor* resource_handle_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kResourceHandleTensor,
                                           &resource_handle_tensor));
  TF_LITE_ENSURE_EQ(context, resource_handle_tensor->type, kTfLiteResource);
  size_t bytesRequired = sizeof(int32_t);

  // Realloc space for an integer handle value.
  TfLiteTensorRealloc(bytesRequired, resource_handle_tensor);
  resource_handle_tensor->bytes = bytesRequired;

  TfLiteIntArray* outputSize = TfLiteIntArrayCreate(1);
  outputSize->data[0] = 1;
  if (resource_handle_tensor->dims)
    TfLiteIntArrayFree(resource_handle_tensor->dims);
  resource_handle_tensor->dims = outputSize;

  return kTfLiteOk;
}

TfLiteStatus EvalHashtable(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->builtin_data != nullptr);
  const auto* params =
      reinterpret_cast<const TfLiteHashtableParams*>(node->builtin_data);

  const int32_t resource_id = params->table_id;

  TfLiteTensor* resource_handle_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kResourceHandleTensor,
                                           &resource_handle_tensor));
  *resource_handle_tensor->data.i32 = resource_id;

  Subgraph* subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto& resources = subgraph->resources();
  resource::CreateHashtableResourceIfNotAvailable(
      &resources, resource_id, params->key_dtype, params->value_dtype);
  return kTfLiteOk;
}

}  // namespace hashtable

TfLiteRegistration* Register_HASHTABLE() {
  static TfLiteRegistration r = {nullptr, nullptr, hashtable::PrepareHashtable,
                                 hashtable::EvalHashtable};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
