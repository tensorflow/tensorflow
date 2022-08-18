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

#include <stdint.h>
#include <string.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource/resource_variable.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace read_variable {

constexpr int kInputVariableId = 0;
constexpr int kOutputValue = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor* input_resource_id_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputVariableId,
                                          &input_resource_id_tensor));
  TF_LITE_ENSURE(context, (input_resource_id_tensor->type == kTfLiteResource ||
                           input_resource_id_tensor->type == kTfLiteInt32));
  TF_LITE_ENSURE_EQ(context, NumElements(input_resource_id_tensor), 1);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputValue, &output));

  if (output->dims->size == 0) {
    // Currently there is no good way to differentiate between scalar and
    // unranked tensor, so we set the tensor's allocation type to dynamic in
    // both cases.
    SetTensorToDynamic(output);
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  Subgraph* subgraph = reinterpret_cast<Subgraph*>(context->impl_);

  const TfLiteTensor* input_resource_id_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputVariableId,
                                          &input_resource_id_tensor));
  int resource_id = input_resource_id_tensor->data.i32[0];
  auto& resources = subgraph->resources();
  auto* variable = resource::GetResourceVariable(&resources, resource_id);
  TF_LITE_ENSURE(context, variable != nullptr);

  TfLiteTensor* variable_tensor = variable->GetTensor();
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputValue, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, variable_tensor->type, output->type);
  // Only resize the output if the op produces dynamic output.
  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(
                                   context, output,
                                   TfLiteIntArrayCopy(variable_tensor->dims)));
  }
  memcpy(output->data.raw, variable_tensor->data.raw, output->bytes);

  return kTfLiteOk;
}

}  // namespace read_variable

TfLiteRegistration* Register_READ_VARIABLE() {
  static TfLiteRegistration r = {nullptr, nullptr, read_variable::Prepare,
                                 read_variable::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
