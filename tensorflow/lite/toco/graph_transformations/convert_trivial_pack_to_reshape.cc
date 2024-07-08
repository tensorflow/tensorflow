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

#include "absl/status/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

::tensorflow::Status ConvertTrivialPackToReshape::Run(Model* model,
                                                      std::size_t op_index,
                                                      bool* modified) {
  *modified = false;
  auto pack_it = model->operators.begin() + op_index;
  if (pack_it->get()->type != OperatorType::kPack) {
    return absl::OkStatus();
  }
  auto* pack_op = static_cast<PackOperator*>(pack_it->get());
  if (pack_op->inputs.size() > 1) {
    // Not trivial.
    return absl::OkStatus();
  }
  CHECK_EQ(pack_op->outputs.size(), 1);

  const auto& input_array = model->GetArray(pack_op->inputs[0]);
  if (!input_array.has_shape()) {
    // Yield until input dims have been resolved.
    return absl::OkStatus();
  }
  if (input_array.shape().dimensions_count() == 0) {
    // Input array cannot be 0-D.
    // (Unsure if this is TF behavior, but was required to get a test to pass.)
    return absl::OkStatus();
  }

  AddMessageF("Converting trivial %s to a reshape", LogName(*pack_op));

  // Note that we could convert to ExpandDims but toco prefers reshapes.
  auto* reshape_op = new TensorFlowReshapeOperator;
  reshape_op->inputs = {pack_op->inputs[0]};
  reshape_op->outputs = pack_op->outputs;

  // Create shape param.
  std::string shape_array_name =
      AvailableArrayName(*model, pack_op->outputs[0] + "_shape");
  Array& shape_array = model->GetOrCreateArray(shape_array_name);
  const int shape_array_dims = 1 + input_array.shape().dimensions_count();
  *(shape_array.mutable_shape()->mutable_dims()) = {shape_array_dims};
  reshape_op->inputs.push_back(shape_array_name);
  shape_array.data_type = ArrayDataType::kInt32;
  auto& shape_buffer = shape_array.GetMutableBuffer<ArrayDataType::kInt32>();

  // Insert '1' at the 'axis' dimension of the output shape.
  int index = 0;
  for (int dim = 0; dim < shape_array_dims; ++dim) {
    dim == pack_op->axis
        ? shape_buffer.data.push_back(1)
        : shape_buffer.data.push_back(input_array.shape().dims(index++));
  }

  // Replace the operator in the graph.
  model->operators.emplace(pack_it, reshape_op);
  DeleteOpAndArrays(model, pack_op);

  *modified = true;
  return absl::OkStatus();
}

}  // namespace toco
