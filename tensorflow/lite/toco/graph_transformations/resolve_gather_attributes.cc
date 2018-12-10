/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status ResolveGatherAttributes::Run(Model* model,
                                                  std::size_t op_index,
                                                  bool* modified) {
  *modified = false;
  auto* gather_op = model->operators[op_index].get();
  if (gather_op->type != OperatorType::kGather)
    return ::tensorflow::Status::OK();
  auto* op = static_cast<GatherOperator*>(gather_op);

  if (op->axis) {
    // Attributes already resolved
    return ::tensorflow::Status::OK();
  }
  if (op->inputs.size() != 3) return ::tensorflow::Status::OK();
  if (!IsConstantParameterArray(*model, op->inputs[2]))
    return ::tensorflow::Status::OK();

  const auto& indices_array = model->GetArray(op->inputs[2]);
  if (!indices_array.has_shape()) return ::tensorflow::Status::OK();
  const auto& axis_data = indices_array.GetBuffer<ArrayDataType::kInt32>().data;
  CHECK_EQ(axis_data.size(), 1)
      << "Multidimensional gather not supported on " << LogName(*op);
  op->axis = {axis_data[0]};

  // Drop the axis array as we no longer need it.
  DeleteArrayIfUsedOnce(op->inputs[2], model);
  op->inputs.resize(2);

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
