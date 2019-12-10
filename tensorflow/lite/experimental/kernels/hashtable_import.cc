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
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource/lookup_interfaces.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace ops {
namespace custom {

namespace hashtable {

constexpr int kInputResourceIdTensor = 0;
constexpr int kKeyTensor = 1;
constexpr int kValueTensor = 2;

TfLiteStatus PrepareHashtableImport(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 0);

  const TfLiteTensor* input_resource_id_tensor =
      GetInput(context, node, kInputResourceIdTensor);
  TF_LITE_ENSURE_EQ(context, input_resource_id_tensor->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_resource_id_tensor), 1);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(input_resource_id_tensor, 0), 1);

  const TfLiteTensor* key_tensor = GetInput(context, node, kKeyTensor);
  TF_LITE_ENSURE(context, (key_tensor->type == kTfLiteInt32 ||
                           key_tensor->type == kTfLiteString));

  const TfLiteTensor* value_tensor = GetInput(context, node, kValueTensor);
  TF_LITE_ENSURE(context, (value_tensor->type == kTfLiteInt32 ||
                           value_tensor->type == kTfLiteString ||
                           value_tensor->type == kTfLiteFloat32));
  // TODO(b/144731295): Tensorflow lookup ops support 1-D vector in storing
  // values.
  TF_LITE_ENSURE(context, HaveSameShapes(key_tensor, value_tensor));
  return kTfLiteOk;
}

TfLiteStatus EvalHashtableImport(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_resource_id_tensor =
      GetInput(context, node, kInputResourceIdTensor);
  const int resource_id = input_resource_id_tensor->data.i32[0];

  const TfLiteTensor* key_tensor = GetInput(context, node, kKeyTensor);
  const TfLiteTensor* value_tensor = GetInput(context, node, kValueTensor);

  Subgraph* subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto& resources = subgraph->resources();
  auto* lookup = resource::GetHashtableResource(&resources, resource_id);
  TF_LITE_ENSURE(context, lookup != nullptr);
  TF_LITE_ENSURE_STATUS(
      lookup->CheckKeyAndValueTypes(context, key_tensor, value_tensor));
  // The hashtable resource will only be initialized once, attempting to
  // initialize it multiple times will be a no-op.
  return lookup->Import(context, key_tensor, value_tensor);
}

}  // namespace hashtable

TfLiteRegistration* Register_HASHTABLE_IMPORT() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 hashtable::PrepareHashtableImport,
                                 hashtable::EvalHashtableImport,
                                 nullptr,
                                 BuiltinOperator_CUSTOM};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
