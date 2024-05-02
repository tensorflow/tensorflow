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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/quantization_util.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

// This inserts an operator whose output is a float array (name:
// flags.input_array()).  It has to wait for any existing operators that
// generate this output to be removed by graph transformations.  Note that there
// may be more than one operator that takes the input_array as their input, and
// that some of these may be removed by graph transformations.
bool AddDequantizeOperatorToInput(const std::string& input_name,
                                  const Operator* op,
                                  GraphTransformation* transformation,
                                  Model* model) {
  // An operator with the required output may be a dequantize operator already
  // created.  Alternatively it may be an operator that needs to be removed
  // because it is unused, in which case we wait for RemoveUnusedOp to do its
  // work.
  if (GetOpWithOutput(*model, input_name)) {
    return false;
  }

  // We only apply for the first operator if there is more than one.  This is
  // not strictly necessary for ordering correctness, since we insert the
  // dequant operator at the beginning of the op sequence, but it makes the
  // insertion more predictable (eg forward vs backwards operator sweep).
  if (CountOpsWithInput(*model, input_name) > 1) {
    if (op != GetFirstOpWithInput(*model, input_name)) {
      return false;
    }
  }

  auto& input_array = model->GetArray(input_name);
  if (input_array.data_type != ArrayDataType::kFloat) {
    return false;
  }

  if (input_array.final_data_type == input_array.data_type ||
      input_array.final_data_type == ArrayDataType::kNone) {
    return false;
  }

  const auto& dequantized_input_name =
      AvailableArrayName(*model, input_name + "_dequantized");
  for (auto& other_op : model->operators) {
    for (std::string& other_op_input : other_op->inputs) {
      if (other_op_input == input_name) {
        other_op_input = dequantized_input_name;
      }
    }
  }

  auto& dequantized_input_array =
      model->GetOrCreateArray(dequantized_input_name);
  auto* image_input_op = new DequantizeOperator;
  image_input_op->inputs = {input_name};
  image_input_op->outputs = {dequantized_input_name};
  model->operators.emplace(model->operators.begin(), image_input_op);

  dequantized_input_array.data_type = ArrayDataType::kFloat;
  const auto& input_minmax = input_array.GetMinMax();
  auto& dequantized_input_minmax = dequantized_input_array.GetOrCreateMinMax();
  dequantized_input_minmax = input_minmax;
  auto& input_qparams = input_array.GetOrCreateQuantizationParams();
  input_array.data_type = input_array.final_data_type;
  ChooseQuantizationParamsForArrayAndQuantizedDataType(
      input_array, input_array.data_type, &input_qparams);

  transformation->AddMessageF(
      "Created %s"
      " to handle quantized input image data, taking over existing"
      " mean_value and std_value flags. Cleared those flags.",
      LogName(*image_input_op));

  return true;
}

::tensorflow::Status MakeInitialDequantizeOperator::Run(Model* model,
                                                        std::size_t op_index,
                                                        bool* modified) {
  *modified = false;
  // This is effectively a transformation applied to edges.  We iterate over the
  // specified node (op) and proceed for input edges.
  const auto it = model->operators.begin() + op_index;
  const auto* op = it->get();
  bool change_made = false;
  for (auto& input : op->inputs) {
    for (auto& input_array : *model->flags.mutable_input_arrays()) {
      if (input_array.name() == input) {
        if (AddDequantizeOperatorToInput(input_array.name(), op, this, model)) {
          change_made = true;
          input_array.clear_mean_value();
          input_array.clear_std_value();
        }
      }
    }
  }
  *modified = change_made;
  return absl::OkStatus();
}

}  // namespace toco
