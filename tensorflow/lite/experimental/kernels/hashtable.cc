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

#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource/lookup_interfaces.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace ops {
namespace custom {
namespace hashtable {

constexpr int kResourceHandleTensor = 0;

// TODO(b/144728911): The following structure should be moved to
// builtin_op_data.h when it is ready to become a builtin op.
typedef struct {
  std::string table_name;
  TfLiteType key_dtype;
  TfLiteType value_dtype;
} TfLiteHashtableParams;

void* InitHashtable(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_CHECK(buffer != nullptr);

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  TfLiteHashtableParams* option = new TfLiteHashtableParams;
  option->table_name = m["table_name"].AsString().str();
  option->key_dtype = static_cast<TfLiteType>(m["key_dtype"].AsInt32());
  option->value_dtype = static_cast<TfLiteType>(m["value_dtype"].AsInt32());

  return option;
}

void FreeHashtable(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<TfLiteHashtableParams*>(buffer);
}

TfLiteStatus PrepareHashtable(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 0);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TF_LITE_ENSURE(context, node->user_data != nullptr);
  const auto* params =
      reinterpret_cast<const TfLiteHashtableParams*>(node->user_data);
  TF_LITE_ENSURE(context, !params->table_name.empty());
  TF_LITE_ENSURE(context, (params->key_dtype == kTfLiteInt32 ||
                           params->key_dtype == kTfLiteString));
  TF_LITE_ENSURE(context, (params->value_dtype == kTfLiteInt32 ||
                           params->value_dtype == kTfLiteString ||
                           params->value_dtype == kTfLiteFloat32));

  TfLiteTensor* resource_handle_tensor =
      GetOutput(context, node, kResourceHandleTensor);
  TF_LITE_ENSURE(context, resource_handle_tensor != nullptr);
  TF_LITE_ENSURE_EQ(context, resource_handle_tensor->type, kTfLiteInt32);
  TfLiteIntArray* outputSize = TfLiteIntArrayCreate(1);
  outputSize->data[0] = 1;
  return context->ResizeTensor(context, resource_handle_tensor, outputSize);
}

TfLiteStatus EvalHashtable(TfLiteContext* context, TfLiteNode* node) {
  const auto* params =
      reinterpret_cast<const TfLiteHashtableParams*>(node->user_data);

  // The resource id is generated based on the given table name.
  const int resource_id = std::hash<std::string>{}(params->table_name);

  TfLiteTensor* resource_handle_tensor =
      GetOutput(context, node, kResourceHandleTensor);
  auto* resource_handle_data =
      GetTensorData<std::int32_t>(resource_handle_tensor);
  resource_handle_data[0] = resource_id;

  Subgraph* subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto& resources = subgraph->resources();
  resource::CreateHashtableResourceIfNotAvailable(
      &resources, resource_id, params->key_dtype, params->value_dtype);
  return kTfLiteOk;
}

}  // namespace hashtable

TfLiteRegistration* Register_HASHTABLE() {
  static TfLiteRegistration r = {hashtable::InitHashtable,
                                 hashtable::FreeHashtable,
                                 hashtable::PrepareHashtable,
                                 hashtable::EvalHashtable,
                                 nullptr,
                                 BuiltinOperator_CUSTOM};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
