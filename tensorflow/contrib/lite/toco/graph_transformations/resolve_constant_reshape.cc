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
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

// Resolves a constant reshape operation by copying the buffer.
bool ResolveConstantReshape::Run(Model* model, std::size_t op_index) {
  auto it = model->operators.begin() + op_index;
  const auto* base_op = it->get();
  if (base_op->type != OperatorType::kReshape) {
    return false;
  }
  const auto* op = static_cast<const TensorFlowReshapeOperator*>(base_op);

  CHECK_EQ(op->inputs.size(), 2);
  CHECK_EQ(op->outputs.size(), 1);

  // We require constant inputs.
  if (!IsConstantParameterArray(*model, op->inputs[0]) ||
      !IsConstantParameterArray(*model, op->inputs[1])) {
    return false;
  }

  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes.
    return false;
  }
  if (!output_array.has_shape()) {
    // Yield until the output shape has been set by PropagateFixedShapes.
    return false;
  }

  const Array& input_array = model->GetArray(op->inputs[0]);
  if (!ShapesAgreeUpToExtending(input_array.shape(), output_array.shape())) {
    AddMessageF("Constant reshape is non-trivial (%s -> %s)",
                ShapeToString(input_array.shape()),
                ShapeToString(output_array.shape()));
    return false;
  }

  CHECK(!output_array.buffer);
  switch (input_array.data_type) {
    case ArrayDataType::kBool:
      CopyArrayBuffer<ArrayDataType::kBool>(input_array, &output_array);
      break;
    case ArrayDataType::kFloat:
      CopyArrayBuffer<ArrayDataType::kFloat>(input_array, &output_array);
      break;
    case ArrayDataType::kInt8:
      CopyArrayBuffer<ArrayDataType::kInt8>(input_array, &output_array);
      break;
    case ArrayDataType::kUint8:
      CopyArrayBuffer<ArrayDataType::kUint8>(input_array, &output_array);
      break;
    case ArrayDataType::kInt16:
      CopyArrayBuffer<ArrayDataType::kInt16>(input_array, &output_array);
      break;
    case ArrayDataType::kUint16:
      CopyArrayBuffer<ArrayDataType::kUint16>(input_array, &output_array);
      break;
    case ArrayDataType::kInt32:
      CopyArrayBuffer<ArrayDataType::kInt32>(input_array, &output_array);
      break;
    case ArrayDataType::kUint32:
      CopyArrayBuffer<ArrayDataType::kUint32>(input_array, &output_array);
      break;
    case ArrayDataType::kInt64:
      CopyArrayBuffer<ArrayDataType::kInt64>(input_array, &output_array);
      break;
    case ArrayDataType::kUint64:
      CopyArrayBuffer<ArrayDataType::kUint64>(input_array, &output_array);
      break;
    case ArrayDataType::kString:
      CopyArrayBuffer<ArrayDataType::kString>(input_array, &output_array);
      break;
    default:
      LOG(FATAL) << "Unsupported data type: "
                 << ArrayDataTypeName(input_array.data_type);
      return false;
  }

  AddMessageF("Resolving constant reshape of %s", LogName(*op));

  CopyMinMaxAndQuantizationRelatedFields(input_array, &output_array);

  // Erase input arrays if no longer used.
  for (const auto& input : op->inputs) {
    if (IsDiscardableArray(*model, input) &&
        CountOpsWithInput(*model, input) == 1) {
      model->EraseArray(input);
    }
  }

  // Erase the operator.
  model->operators.erase(it);
  return true;
}

}  // namespace toco
