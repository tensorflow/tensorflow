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
#include "tensorflow/lite/toco/graph_transformations/quantization_util.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

bool SupportsMinMax(const Array& array) {
  return array.data_type == ArrayDataType::kFloat;
}

}  // namespace

// Propagates default min/max values to any operator input/output array that
// is missing them.
//
// When provided a set of min/max values for uint8 arrays this will rescale
// the values for other data types as required and preserving the floating point
// range within the new type.
::tensorflow::Status PropagateDefaultMinMax::Run(Model* model,
                                                 std::size_t op_index,
                                                 bool* modified) {
  *modified = false;
  const auto it = model->operators.begin() + op_index;
  const auto* op = it->get();

  bool did_change = false;

  for (const auto& input : op->inputs) {
    auto& input_array = model->GetArray(input);
    if (!input_array.minmax && !input_array.buffer &&
        SupportsMinMax(input_array)) {
      did_change |= SetArrayMinMax(input, &input_array);
    }
  }

  for (const auto& output : op->outputs) {
    auto& output_array = model->GetArray(output);
    if (!output_array.minmax && !output_array.buffer &&
        SupportsMinMax(output_array)) {
      did_change |= SetArrayMinMax(output, &output_array);
    }
  }

  *modified = did_change;
  return absl::OkStatus();
}

// Sets the min/max on the given array, adjusting the reference_minmax for the
// final data type of the array if it is already specified.
bool PropagateDefaultMinMax::SetArrayMinMax(const std::string& array_name,
                                            Array* array) {
  CHECK(!array->minmax);

  ArrayDataType quantized_data_type =
      GetQuantizedDataType(*array, ArrayDataType::kUint8);
  for (const auto& type_range : type_ranges_) {
    if (type_range.first == quantized_data_type) {
      array->GetOrCreateMinMax() = type_range.second;
      break;
    }
  }
  if (!array->minmax) {
    AddMessageF(
        "No defaults specified for quantized data type %s of array %s, "
        "skipping",
        ArrayDataTypeName(quantized_data_type), array_name);
    return false;
  }

  AddMessageF("Adding default minmax %g,%g to array %s when quantized as %s",
              array->GetMinMax().min, array->GetMinMax().max, array_name,
              ArrayDataTypeName(quantized_data_type));

  return true;
}

}  // namespace toco
