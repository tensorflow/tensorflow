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
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

template <ArrayDataType Type>
bool ComputeFillArray(Model* model, FillOperator* op) {
  const auto& val_array = model->GetArray(op->inputs[1]);
  auto& output_array = model->GetArray(op->outputs[0]);

  CHECK(val_array.data_type == Type);
  CHECK(output_array.data_type == Type);

  // Compute the array data
  std::vector<DataType<Type>>& data =
      output_array.GetMutableBuffer<Type>().data;
  data.resize(RequiredBufferSizeForShape(output_array.shape()));
  DataType<Type> fill_val = val_array.GetBuffer<Type>().data[0];
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = fill_val;
  }

  return true;
}

::tensorflow::Status ResolveConstantFill::Run(Model* model,
                                              std::size_t op_index,
                                              bool* modified) {
  *modified = false;
  const auto fill_it = model->operators.begin() + op_index;
  auto* base_op = fill_it->get();
  if (base_op->type != OperatorType::kFill) {
    return ::tensorflow::OkStatus();
  }
  auto* op = static_cast<FillOperator*>(base_op);

  CHECK_EQ(op->inputs.size(), 2);
  CHECK_EQ(op->outputs.size(), 1);

  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return ::tensorflow::OkStatus();
  }

  if (!output_array.has_shape()) {
    // Yield until the output shape has been set by PropagateFixedShapes
    return ::tensorflow::OkStatus();
  }

  const auto& val_array = model->GetArray(op->inputs[1]);
  if (!val_array.has_shape()) {
    // Yield until the value shape has been resolved.
    return ::tensorflow::OkStatus();
  }
  if (!IsConstantParameterArray(*model, op->inputs[1])) {
    // Yield until the value is constant.
    return ::tensorflow::OkStatus();
  }
  CHECK_EQ(RequiredBufferSizeForShape(val_array.shape()), 1);

  switch (output_array.data_type) {
    case ArrayDataType::kFloat:
      if (!ComputeFillArray<ArrayDataType::kFloat>(model, op)) {
        return ::tensorflow::OkStatus();
      }
      break;
    case ArrayDataType::kUint8:
      if (!ComputeFillArray<ArrayDataType::kUint8>(model, op)) {
        return ::tensorflow::OkStatus();
      }
      break;
    case ArrayDataType::kInt32:
      if (!ComputeFillArray<ArrayDataType::kInt32>(model, op)) {
        return ::tensorflow::OkStatus();
      }
      break;
    case ArrayDataType::kInt64:
      if (!ComputeFillArray<ArrayDataType::kInt64>(model, op)) {
        return ::tensorflow::OkStatus();
      }
      break;
    case ArrayDataType::kComplex64:
      if (!ComputeFillArray<ArrayDataType::kComplex64>(model, op)) {
        return ::tensorflow::OkStatus();
      }
      break;
    default:
      LOG(FATAL) << "Unsupported data type given to Fill op with output \""
                 << op->outputs[0] << "\"";
      break;
  }

  DeleteOpAndArrays(model, op);
  *modified = true;
  return ::tensorflow::OkStatus();
}

}  // namespace toco
