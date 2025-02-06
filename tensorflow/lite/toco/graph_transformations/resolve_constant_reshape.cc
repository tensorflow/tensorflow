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

#include "absl/status/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

// Resolves a constant reshape operation by copying the buffer.
::tensorflow::Status ResolveConstantReshape::Run(Model* model,
                                                 std::size_t op_index,
                                                 bool* modified) {
  *modified = false;
  auto it = model->operators.begin() + op_index;
  const auto* base_op = it->get();
  if (base_op->type != OperatorType::kReshape) {
    return absl::OkStatus();
  }
  const auto* op = static_cast<const TensorFlowReshapeOperator*>(base_op);

  CHECK_EQ(op->inputs.size(), 2);
  CHECK_EQ(op->outputs.size(), 1);

  // We require constant inputs.
  if (!IsConstantParameterArray(*model, op->inputs[0]) ||
      !IsConstantParameterArray(*model, op->inputs[1])) {
    return absl::OkStatus();
  }

  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes.
    return absl::OkStatus();
  }
  if (!output_array.has_shape()) {
    // Yield until the output shape has been set by PropagateFixedShapes.
    return absl::OkStatus();
  }

  const Array& input_array = model->GetArray(op->inputs[0]);
  if (!ShapesAgreeUpToExtending(input_array.shape(), output_array.shape())) {
    AddMessageF("Constant reshape is non-trivial (%s -> %s)",
                ShapeToString(input_array.shape()),
                ShapeToString(output_array.shape()));
    return absl::OkStatus();
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
    case ArrayDataType::kComplex64:
      CopyArrayBuffer<ArrayDataType::kComplex64>(input_array, &output_array);
      break;
    default:
      LOG(FATAL) << "Unsupported data type: "
                 << ArrayDataTypeName(input_array.data_type);
      return absl::OkStatus();
  }

  AddMessageF("Resolving constant reshape of %s", LogName(*op));

  CopyMinMaxAndQuantizationRelatedFields(input_array, &output_array);

  DeleteOpAndArrays(model, op);
  *modified = true;
  return absl::OkStatus();
}

}  // namespace toco
