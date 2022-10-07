/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/delegates/compatibility/nnapi/nnapi_compatibility_lib.h"

#include <map>
#include <utility>
#include <vector>

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace tools {

using ::tflite::delegate::nnapi::NNAPIValidationFailure;

TfLiteStatus CheckCompatibility(
    TfLiteContext* context, int32_t runtime_feature_level,
    std::vector<int>* supported_nodes,
    std::map<int, std::vector<NNAPIValidationFailure>>* failures_by_node) {
  if (!context) {
    TFLITE_LOG_PROD_ONCE(TFLITE_LOG_ERROR, "Context is nullptr.");
    return kTfLiteError;
  }
  // Gets execution plan.
  TfLiteIntArray* execution_plan;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &execution_plan));

  // Validates compatibility for each node.
  for (int node_index : TfLiteIntArrayView(execution_plan)) {
    TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Node index: %d", node_index);
    TfLiteNode* node;
    TfLiteRegistration* registration;
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));
    std::vector<delegate::nnapi::NNAPIValidationFailure> map_failures;
    if (NNAPIDelegateKernel::Validate(
            context, registration, runtime_feature_level, node,
            /* is_accelerator_specified= */ true,
            /* vendor_plugin= */ nullptr, &map_failures)) {
      TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Built-in Code: %d",
                      registration->builtin_code);
      if (supported_nodes) {
        supported_nodes->push_back(node_index);
      }
    } else {
      if (failures_by_node) {
        (*failures_by_node)[node_index] = std::move(map_failures);
      }
    }
  }
  return kTfLiteOk;
}

}  // namespace tools
}  // namespace tflite
