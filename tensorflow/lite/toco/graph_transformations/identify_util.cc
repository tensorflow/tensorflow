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

#include "tensorflow/lite/toco/graph_transformations/identify_util.h"

#include <string>

#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {
namespace util {

bool IsBinaryOp(const Operator* op, OperatorType optype,
                FusedActivationFunctionType act) {
  return op && op->type == optype && op->inputs.size() == 2 &&
         op->fused_activation_function == act;
}

bool CheckArrayIsScalarFloat(Model* model, const std::string& name, float val) {
  const auto& op_array = model->GetArray(name);
  if (!op_array.buffer || op_array.buffer->type != ArrayDataType::kFloat ||
      RequiredBufferSizeForShape(op_array.shape()) != 1) {
    return false;
  }
  const auto& op_data = op_array.GetBuffer<ArrayDataType::kFloat>().data;
  return op_data[0] == val;
}

int GetSingleScalarInputIndexOfBinaryOp(Model* model, const Operator* op,
                                        float val) {
  bool input0_is_scalar = CheckArrayIsScalarFloat(model, op->inputs[0], val);
  bool input1_is_scalar = CheckArrayIsScalarFloat(model, op->inputs[1], val);
  return input0_is_scalar == input1_is_scalar ? -1 : input0_is_scalar ? 0 : 1;
}

}  // namespace util
}  // namespace toco
