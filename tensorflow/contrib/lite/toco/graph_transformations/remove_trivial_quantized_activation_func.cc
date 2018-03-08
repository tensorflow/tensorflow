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
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/runtime/types.h"
#include "tensorflow/contrib/lite/toco/toco_types.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool RemoveTrivialQuantizedActivationFunc::Run(Model* model,
                                               std::size_t op_index) {
  const auto it = model->operators.begin() + op_index;
  auto* op = it->get();
  if (op->fused_activation_function != FusedActivationFunctionType::kRelu &&
      op->fused_activation_function != FusedActivationFunctionType::kRelu6) {
    return false;
  }
  const auto& output_array = model->GetArray(op->outputs[0]);
  if (!output_array.quantization_params) {
    return false;
  }
  if (output_array.data_type != ArrayDataType::kUint8) {
    return false;
  }
  const auto& quantization_params = output_array.GetQuantizationParams();

  bool has_nontrivial_min_bound = false;
  bool has_nontrivial_max_bound = false;

  if (op->fused_activation_function == FusedActivationFunctionType::kRelu ||
      op->fused_activation_function == FusedActivationFunctionType::kRelu6) {
    double lowest_representable_output =
        (0. - quantization_params.zero_point) * quantization_params.scale;
    if (lowest_representable_output < 0.) {
      has_nontrivial_min_bound = true;
      AddMessageF(
          "Quantized activation function is not trivial: "
          "the lowest representable output value %g"
          " less than the clamp min bound.",
          lowest_representable_output);
    }
  }
  if (op->fused_activation_function == FusedActivationFunctionType::kRelu6) {
    double highest_representable_output =
        (255. - quantization_params.zero_point) * quantization_params.scale;
    if (highest_representable_output > 6.) {
      has_nontrivial_max_bound = true;
      AddMessageF(
          "Quantized activation function is not trivial: "
          "the highest representable output value %g"
          " is greater than the clamp max bound.",
          highest_representable_output);
    }
  }

  if (has_nontrivial_min_bound || has_nontrivial_max_bound) {
    return false;
  }

  op->fused_activation_function = FusedActivationFunctionType::kNone;
  AddMessageF(
      "Removing trivial quantized activation function on %s"
      " because the output quantization parameters imply at least as tight"
      " a clamp anyway.",
      LogName(*op));
  return true;
}

}  // namespace toco
