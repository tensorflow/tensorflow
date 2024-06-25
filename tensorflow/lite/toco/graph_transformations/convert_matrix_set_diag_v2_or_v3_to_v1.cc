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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

// V3 is only different from V2 because it has an extra attribute (align).
// This attribute doesn't affect V1 so we don't have to keep track of it here.
::tensorflow::Status ConvertMatrixSetDiagV2OrV3ToV1::Run(Model* model,
                                                         std::size_t op_index,
                                                         bool* modified) {
  *modified = false;
  auto it = model->operators.begin() + op_index;
  const auto* op = it->get();
  if (op->type != OperatorType::kMatrixSetDiagV2 &&
      op->type != OperatorType::kMatrixSetDiagV3) {
    return absl::OkStatus();
  }

  if (op->inputs.size() != 3) {
    return tensorflow::errors::InvalidArgument(
        "The input size of op %s should be 3", LogName(*op));
  }

  const auto& input_k = model->GetArray(op->inputs[2]);

  if (!input_k.buffer) {
    return absl::OkStatus();
  }

  if (input_k.GetBuffer<ArrayDataType::kInt32>().data.size() != 1) {
    return tensorflow::errors::InvalidArgument(
        "Array for argument k of op %s should contains exact one element",
        LogName(*op));
  }

  int k = input_k.GetBuffer<ArrayDataType::kInt32>().data[0];

  if (k != 0) {
    return tensorflow::errors::InvalidArgument(
        "parameter k of op ", LogName(*op),
        " is expected to be 0, other values are not supported currently");
  }

  auto* matrix_set_diag_op = new MatrixSetDiagOperator;
  matrix_set_diag_op->inputs.push_back(op->inputs[0]);
  matrix_set_diag_op->inputs.push_back(op->inputs[1]);
  matrix_set_diag_op->outputs.push_back(op->outputs[0]);

  AddMessageF("Replacing %s with %s", LogName(*op),
              LogName(*matrix_set_diag_op));

  // Replace the operator in the graph.
  model->operators.emplace(it, matrix_set_diag_op);
  DeleteOpAndArrays(model, op);

  *modified = true;
  return absl::OkStatus();
}

}  // namespace toco
