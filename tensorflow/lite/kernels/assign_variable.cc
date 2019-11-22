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

#include <string.h>

#include <memory>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource/resource_variable.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace custom {
namespace assign_variable {

constexpr int kInputVariableId = 0;
constexpr int kInputValue = 1;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  // TODO(b/137042749): TFLite infrastructure (converter, delegate) doesn't
  // fully support 0-output ops yet. Currently it works if we manually crfat
  // a TFLite graph that contains variable ops. Note:
  // * The TFLite Converter need to be changed to be able to produce an op
  //   with 0 output.
  // * The delegation code need to be changed to handle 0 output ops. However
  //   everything still works fine when variable ops aren't used.
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 0);

  const TfLiteTensor* input_resource_id_tensor =
      GetInput(context, node, kInputVariableId);
  TF_LITE_ENSURE_EQ(context, input_resource_id_tensor->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumElements(input_resource_id_tensor), 1);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  Subgraph* subgraph = reinterpret_cast<Subgraph*>(context->impl_);

  const TfLiteTensor* input_resource_id_tensor =
      GetInput(context, node, kInputVariableId);
  const TfLiteTensor* input_value_tensor = GetInput(context, node, kInputValue);

  int resource_id = input_resource_id_tensor->data.i32[0];
  auto& resources = subgraph->resources();
  resource::CreateResourceVariableIfNotAvailable(&resources, resource_id);
  auto* variable = resource::GetResourceVariable(&resources, resource_id);
  TF_LITE_ENSURE(context, variable != nullptr);
  variable->AssignFrom(input_value_tensor);

  return kTfLiteOk;
}

}  // namespace assign_variable

TfLiteRegistration* Register_ASSIGN_VARIABLE() {
  static TfLiteRegistration r = {nullptr, nullptr, assign_variable::Prepare,
                                 assign_variable::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
