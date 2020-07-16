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
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

void RenameArray(Model* model, const std::string& oldname,
                 const std::string& desired_newname) {
  const std::string& newname = AvailableArrayName(*model, desired_newname);
  auto& arrays = model->GetMutableArrayMap();
  arrays[newname] = std::move(arrays[oldname]);
  arrays.erase(oldname);
  for (const auto& op : model->operators) {
    for (std::string& input : op->inputs) {
      if (input == oldname) {
        input = newname;
      }
    }
    for (std::string& output : op->outputs) {
      if (output == oldname) {
        output = newname;
      }
    }
  }
}

}  // namespace

// Reorder the elements of an input_array according to the input_axes_order and
// output_axes_order. Then adjust the shapes of the input and output arrays
// accordingly. Note that input_array must have a buffer (that is, it is a
// constant array).
template <typename T, ArrayDataType DataType>
void ReorderAxes(AxesOrder input_axes_order, AxesOrder output_axes_order,
                 const Array& input_array, Array* output_array) {
  DCHECK(input_array.buffer->type == DataType);
  DCHECK(!output_array->buffer);
  const auto& input_data = input_array.GetBuffer<DataType>().data;
  auto& output_data = output_array->GetMutableBuffer<DataType>().data;
  output_data.resize(RequiredBufferSizeForShape(output_array->shape()));
  // TODO(b/62904716) Shapes should be used directly.
  Shape input_shape = input_array.shape();
  Shape output_shape = output_array->shape();
  if (AxesCount(input_axes_order) == 2) {
    UnextendShape(&input_shape, 2);
    UnextendShape(&output_shape, 2);
  }
  ShuffleArray(input_shape, input_axes_order, output_axes_order, output_shape,
               input_data.data(), output_data.data());
  if (input_array.minmax) {
    output_array->GetOrCreateMinMax() = input_array.GetMinMax();
  }
  if (input_array.narrow_range) {
    output_array->narrow_range = true;
  }
}

::tensorflow::Status ResolveReorderAxes::Run(Model* model, std::size_t op_index,
                                             bool* modified) {
  *modified = false;
  auto it = model->operators.begin() + op_index;
  auto* op = it->get();
  if (op->type != OperatorType::kReorderAxes) {
    return ::tensorflow::Status::OK();
  }
  auto* reorder_op = static_cast<ReorderAxesOperator*>(op);

  // Intentionally copies, not references.
  const std::string input_array_name = reorder_op->inputs[0];
  const std::string output_array_name = reorder_op->outputs[0];

  auto& input_array = model->GetArray(input_array_name);
  auto& output_array = model->GetArray(output_array_name);
  if (!input_array.buffer) {
    return ::tensorflow::Status::OK();
  }
  // Yield until output dims have been resolved.
  if (!output_array.has_shape()) {
    return ::tensorflow::Status::OK();
  }
  // Reorder the input array dims and buffer data
  if (input_array.buffer->type == ArrayDataType::kFloat) {
    ReorderAxes<float, ArrayDataType::kFloat>(reorder_op->input_axes_order,
                                              reorder_op->output_axes_order,
                                              input_array, &output_array);
  } else if (input_array.buffer->type == ArrayDataType::kUint8) {
    // TODO(benoitjacob): This path seems unused.
    // ReorderAxes is only used when importing from
    // TensorFlow GraphDef, which does not support quantized nodes.
    ReorderAxes<uint8, ArrayDataType::kUint8>(reorder_op->input_axes_order,
                                              reorder_op->output_axes_order,
                                              input_array, &output_array);
  } else {
    LOG(FATAL) << "Cannot ReorderAxes unless input buffer is float or uint8.";
  }

  AddMessageF("Reordered axes for array %s", input_array_name);

  DeleteOpAndArrays(model, op);
  RenameArray(model, output_array_name, input_array_name);

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
