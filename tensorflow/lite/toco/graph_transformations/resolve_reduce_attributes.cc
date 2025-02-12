/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <cstddef>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

template <typename T>
bool ResolveAttributes(Model* model, T* op) {
  if (!op->axis.empty()) {
    // Attributes already resolved
    return false;
  }
  if (op->inputs.size() != 2) return false;
  if (!IsConstantParameterArray(*model, op->inputs[1])) return false;

  const Array& indices_array = model->GetArray(op->inputs[1]);
  if (!indices_array.has_shape()) return false;

  // It is ok for indices_array to have a shape for an empty tensor. In that
  // case, we don't bother setting 'axis'.
  if (indices_array.buffer->Length() == 0) return false;

  op->axis = indices_array.GetBuffer<ArrayDataType::kInt32>().data;
  return true;
}

::tensorflow::Status ResolveReduceAttributes::Run(Model* model,
                                                  std::size_t op_index,
                                                  bool* modified) {
  *modified = false;
  Operator* op = model->operators[op_index].get();
  switch (op->type) {
    case OperatorType::kMean:
      *modified = ResolveAttributes(model, static_cast<MeanOperator*>(op));
      return absl::OkStatus();
    case OperatorType::kSum:
      *modified =
          ResolveAttributes(model, static_cast<TensorFlowSumOperator*>(op));
      return absl::OkStatus();
    case OperatorType::kReduceProd:
      *modified =
          ResolveAttributes(model, static_cast<TensorFlowProdOperator*>(op));
      return absl::OkStatus();
    case OperatorType::kReduceMin:
      *modified =
          ResolveAttributes(model, static_cast<TensorFlowMinOperator*>(op));
      return absl::OkStatus();
    case OperatorType::kReduceMax:
      *modified =
          ResolveAttributes(model, static_cast<TensorFlowMaxOperator*>(op));
      return absl::OkStatus();
    case OperatorType::kAny:
      *modified =
          ResolveAttributes(model, static_cast<TensorFlowMaxOperator*>(op));
      return absl::OkStatus();
    default:
      return absl::OkStatus();
  }
}

}  // namespace toco
