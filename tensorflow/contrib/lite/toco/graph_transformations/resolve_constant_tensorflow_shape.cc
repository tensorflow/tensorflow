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
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ResolveConstantTensorFlowShape::Run(Model* model, std::size_t op_index) {
  const auto tfshape_it = model->operators.begin() + op_index;
  const auto* tfshape_base_op = tfshape_it->get();
  if (tfshape_base_op->type != OperatorType::kTensorFlowShape) {
    return false;
  }

  const auto* tfshape_op =
      static_cast<const TensorFlowShapeOperator*>(tfshape_base_op);

  const auto& input_array = model->GetArray(tfshape_op->inputs[0]);
  auto& output_array = model->GetArray(tfshape_op->outputs[0]);

  // Yield until the input array's shape has been resolved.
  if (!input_array.has_shape()) {
    return false;
  }

  // Create a buffer for the output array, making it a constant array, and
  // copy the input shape into the output buffer.
  CHECK(!output_array.buffer);
  auto& output_buffer = output_array.GetMutableBuffer<ArrayDataType::kInt32>();
  output_buffer.data = input_array.shape().dims();

  // Erase the input array if no longer used
  if (IsDiscardableArray(*model, tfshape_op->inputs[0]) &&
      CountOpsWithInput(*model, tfshape_op->inputs[0]) == 1) {
    model->arrays.erase(tfshape_op->inputs[0]);
  }
  model->operators.erase(tfshape_it);

  return true;
}

}  // namespace toco
