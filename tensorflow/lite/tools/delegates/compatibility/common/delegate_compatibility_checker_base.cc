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

#include "tensorflow/lite/tools/delegates/compatibility/common/delegate_compatibility_checker_base.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite {
namespace tools {

absl::Status DelegateCompatibilityCheckerBase::checkModelCompatibilityOffline(
    tflite::FlatBufferModel* model_buffer, proto::CompatibilityResult* result) {
  auto model = tflite::GetModel(model_buffer);
  auto subgraphs = model->subgraphs();
  for (int i = 0; i < subgraphs->Length(); ++i) {
    const tflite::SubGraph* subgraph = subgraphs->Get(i);
    for (int j = 0; j < subgraph->operators()->Length(); ++j) {
      proto::OpCompatibilityResult* op_result =
          result->add_compatibility_results();
      op_result->set_subgraph_index_in_model(i);
      op_result->set_operator_index_in_subgraph(j);
      const tflite::Operator* op = subgraph->operators()->Get(j);
      const tflite::OperatorCode* op_code =
          model->operator_codes()->Get(op->opcode_index());
      auto status =
          checkOpCompatibilityOffline(op_code, op, subgraph, model, op_result);
    }
  }
  return absl::OkStatus();
}

absl::Status DelegateCompatibilityCheckerBase::checkModelCompatibilityOnline(
    tflite::FlatBufferModel* model_buffer, proto::CompatibilityResult* result) {
  auto model = tflite::GetModel(model_buffer);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  if (InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    return absl::InternalError("Unable to build the interpreter.");
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    return absl::InternalError("Unable to allocate tensors.");
  }
  for (int i = 0; i < interpreter->subgraphs_size(); ++i) {
    auto context = interpreter->subgraph(i)->context();
    if (!context) {
      return absl::InvalidArgumentError("Context is nullptr.");
    }
    // Gets execution plan.
    TfLiteIntArray* execution_plan;
    if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
      return absl::InternalError("Unable to get graph execution plan.");
    }

    // Validates compatibility for each node.
    for (int node_index : TfLiteIntArrayView(execution_plan)) {
      proto::OpCompatibilityResult* op_result =
          result->add_compatibility_results();
      TfLiteNode* node;
      TfLiteRegistration* registration;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Couldn't get node and registration at node index: ", node_index));
      }
      op_result->set_subgraph_index_in_model(i);
      op_result->set_operator_index_in_subgraph(node_index);
      auto status =
          checkOpCompatibilityOnline(context, node, registration, op_result);
    }
  }
  return absl::OkStatus();
}

absl::Status DelegateCompatibilityCheckerBase::checkOpCompatibilityOffline(
    const tflite::OperatorCode* op_code, const tflite::Operator* op,
    const tflite::SubGraph* subgraph, const tflite::Model* model,
    proto::OpCompatibilityResult* op_result) {
  OpSignature op_sig = tflite::GetOpSignature(op_code, op, subgraph, model);
  auto status = checkOpSigCompatibility(op_sig, op_result);
  if (op_sig.builtin_data) {
    free(op_sig.builtin_data);
  }
  return status;
}

}  // namespace tools
}  // namespace tflite
