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

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/graph_transformations/remove_trivial_passthrough.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

template <ArrayDataType A>
void DequantizeBuffer(Array* array) {
  const auto old_data = array->GetBuffer<A>().data;
  array->buffer = nullptr;
  array->data_type = ArrayDataType::kFloat;
  auto& new_data = array->GetMutableBuffer<ArrayDataType::kFloat>().data;
  new_data.resize(old_data.size());
  const auto& qparams = array->GetQuantizationParams();
  for (int i = 0; i < old_data.size(); i++) {
    new_data[i] = qparams.scale * (old_data[i] - qparams.zero_point);
  }
}

std::vector<std::unique_ptr<Operator>>::iterator FindFirstOpWithInput(
    Model* model, const string& array_name) {
  for (auto it = model->operators.begin(); it != model->operators.end(); ++it) {
    for (const auto& input : it->get()->inputs) {
      if (input == array_name) {
        return it;
      }
    }
  }
  return model->operators.end();
}

void ClearArrayQuantizationParams(const string& array_name, Model* model) {
  auto* array = &model->GetArray(array_name);
  CHECK(array->quantization_params);
  for (auto& input_array : *model->flags.mutable_input_arrays()) {
    if (input_array.name() == array_name) {
      auto& qparams = *array->quantization_params;
      const double new_std_value = 1. / qparams.scale;
      const double new_mean_value = qparams.zero_point;
      if (input_array.has_std_value()) {
        CHECK_LE(std::abs(new_std_value - input_array.std_value()), 0.001);
      } else {
        input_array.set_std_value(new_std_value);
      }
      if (input_array.has_mean_value()) {
        CHECK_LE(std::abs(new_mean_value - input_array.mean_value()), 0.001);
      } else {
        input_array.set_mean_value(new_mean_value);
      }
    }
  }
  array->quantization_params = nullptr;
}

bool DequantizeArray(const string& array_name,
                     GraphTransformation* transformation, Model* model) {
  auto* array = &model->GetArray(array_name);
  if (!array->quantization_params) {
    return false;
  }
  transformation->AddMessageF("Dequantizing array: %s", array_name);

  // Dequantize any buffer
  if (array->buffer) {
    if (array->data_type == ArrayDataType::kUint8) {
      DequantizeBuffer<ArrayDataType::kUint8>(array);
    } else if (array->data_type == ArrayDataType::kInt32) {
      DequantizeBuffer<ArrayDataType::kInt32>(array);
    } else {
      LOG(FATAL) << "Unhandled data type";
    }
    CHECK(array->data_type == ArrayDataType::kFloat);
    CHECK(array->buffer->type == ArrayDataType::kFloat);

    // Clear quantization params, officially makes this a non-quantized array.
    ClearArrayQuantizationParams(array_name, model);
    return true;
  } else {
    array->data_type = ArrayDataType::kFloat;
  }

  // Clear quantization params, officially makes this a non-quantized array.
  ClearArrayQuantizationParams(array_name, model);

  if (array->buffer) {
    return true;
  }

  auto* op_outputting_array = GetOpWithOutput(*model, array_name);
  if (op_outputting_array) {
    if (op_outputting_array->type == OperatorType::kReshape) {
      return true;
    }
  }

  // If there was no minmax info, we can return now. Indeed,
  // the below only serves to create a FakeQuant node, but some arrays are
  // quantized without MinMax (see the CHECK above) and that corresponds to
  // places where a FakeQuant node is actually not wanted, because the
  // quantization params are meant to be inferred in another way (e.g. bias
  // vector for a Conv op, see their special-casing in quantize.cc).
  if (!array->minmax) {
    return true;
  }

  // Determine whether to insert a FakeQuant before or after
  // this array.
  bool must_insert_fakequant_before = false;
  bool must_insert_fakequant_after = false;
  if (IsInputArray(*model, array_name)) {
    must_insert_fakequant_after = true;
  }
  for (const string& output_array : model->flags.output_arrays()) {
    if (array_name == output_array) {
      must_insert_fakequant_before = true;
    }
  }
  for (const auto& rnn_state : model->flags.rnn_states()) {
    if (array_name == rnn_state.state_array()) {
      must_insert_fakequant_after = true;
    }
    if (array_name == rnn_state.back_edge_source_array()) {
      must_insert_fakequant_before = true;
    }
  }
  CHECK(!(must_insert_fakequant_before && must_insert_fakequant_after));

  // Create and insert the FakeQuant node
  auto* fakequant_op = new FakeQuantOperator;
  model->operators.emplace(FindFirstOpWithInput(model, array_name),
                           fakequant_op);
  const string& new_array_name = AvailableArrayName(*model, array_name);
  auto& new_array = model->GetOrCreateArray(new_array_name);
  new_array.data_type = ArrayDataType::kFloat;
  new_array.copy_shape(array->shape());
  new_array.GetOrCreateMinMax() = array->GetMinMax();
  fakequant_op->minmax.reset(new MinMax);
  *fakequant_op->minmax = array->GetMinMax();
  fakequant_op->narrow_range = array->narrow_range;
  if (must_insert_fakequant_before) {
    for (const auto& op : model->operators) {
      for (string& output : op->outputs) {
        if (output == array_name) {
          output = new_array_name;
        }
      }
    }
    fakequant_op->inputs = {new_array_name};
    fakequant_op->outputs = {array_name};
  } else {
    for (const auto& op : model->operators) {
      for (string& input : op->inputs) {
        if (input == array_name) {
          input = new_array_name;
        }
      }
    }
    fakequant_op->inputs = {array_name};
    fakequant_op->outputs = {new_array_name};
  }
  return true;
}

}  // namespace

bool Dequantize::Run(Model* model, std::size_t op_index) {
  const auto op_it = model->operators.begin() + op_index;
  auto* op = op_it->get();

  if (op->type == OperatorType::kDequantize) {
    auto& input_array = model->GetArray(op->inputs[0]);
    if (input_array.data_type == ArrayDataType::kFloat) {
      return false;
    }
    if (input_array.final_data_type != ArrayDataType::kFloat) {
      return false;
    }
    input_array.data_type = ArrayDataType::kFloat;
    input_array.quantization_params = nullptr;
    auto& output_array = model->GetArray(op->outputs[0]);
    output_array.data_type = ArrayDataType::kFloat;
    output_array.quantization_params = nullptr;
    return RemoveTrivialPassthroughOp(this, model, op_index);
  }

  std::vector<string> arrays;
  for (const string& input : op->inputs) {
    arrays.push_back(input);
  }
  for (const string& output : op->outputs) {
    arrays.push_back(output);
  }
  bool changed = false;
  for (const string& array : arrays) {
    if (!model->IsOptionalArray(array)) {
      changed |= DequantizeArray(array, this, model);
    }
  }

  return changed;
}

}  // namespace toco
