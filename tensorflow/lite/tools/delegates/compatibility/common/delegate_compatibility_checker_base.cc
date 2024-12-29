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

#include <cstdlib>

#include "absl/status/status.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/delegates/compatibility/common/delegate_compatibility_checker_util.h"
#include "tensorflow/lite/tools/delegates/compatibility/protos/compatibility_result.pb.h"
#include "tensorflow/lite/tools/versioning/op_signature.h"

namespace tflite {
namespace tools {

absl::Status DelegateCompatibilityCheckerBase::checkModelCompatibilityOffline(
    tflite::FlatBufferModel* model_buffer, proto::CompatibilityResult* result) {
  auto model = model_buffer->GetModel();
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
      RETURN_IF_ERROR(
          checkOpCompatibilityOffline(op_code, op, subgraph, model, op_result));
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
