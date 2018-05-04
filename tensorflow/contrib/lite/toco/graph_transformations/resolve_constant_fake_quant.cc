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
#include "tensorflow/contrib/lite/toco/graph_transformations/quantization_util.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ResolveConstantFakeQuant::Run(Model* model, std::size_t op_index) {
  const auto fakequant_it = model->operators.begin() + op_index;
  const auto* fakequant_base_op = fakequant_it->get();
  if (fakequant_base_op->type != OperatorType::kFakeQuant) {
    return false;
  }

  const auto* fakequant_op =
      static_cast<const FakeQuantOperator*>(fakequant_base_op);

  // Yield until the fakequant MinMax has been resolved.
  if (!fakequant_op->minmax) {
    return false;
  }

  // This transformation only applies when the input array is constant.
  if (!IsConstantParameterArray(*model, fakequant_op->inputs[0])) {
    return false;
  }

  const auto& input_array = model->GetArray(fakequant_op->inputs[0]);
  CHECK(input_array.data_type == ArrayDataType::kFloat);

  // Determine the final data type in the same way as PropagateFakeQuantNumBits.
  ArrayDataType quantized_data_type = input_array.final_data_type;
  if (!InferQuantizedDataTypeFromFakeQuant(*fakequant_op,
                                           &quantized_data_type)) {
    AddMessageF("Unsupported FakeQuant num_bits=%d", fakequant_op->num_bits);
    return false;
  }

  AddMessageF("Resolving constant %s", LogName(*fakequant_op));

  auto& output_array = model->GetArray(fakequant_op->outputs[0]);
  CHECK(input_array.data_type == ArrayDataType::kFloat);
  output_array.data_type = ArrayDataType::kFloat;

  // We'll set the final data type to what the fake quant indicates we should
  // have (and would have been set if this stayed around until
  // PropagateFakeQuantNumBits).
  if (propagate_fake_quant_num_bits()) {
    output_array.final_data_type = quantized_data_type;
  }

  CHECK(!output_array.buffer);
  const auto& input_buffer = input_array.GetBuffer<ArrayDataType::kFloat>();
  output_array.GetOrCreateMinMax() = *fakequant_op->minmax;
  auto& output_buffer = output_array.GetMutableBuffer<ArrayDataType::kFloat>();
  const int size = input_buffer.data.size();
  output_buffer.data.resize(size);
  QuantizationParams qparams;
  GetQuantizationParamsFromMinMax<ArrayDataType::kUint8>(*fakequant_op->minmax,
                                                         &qparams);
  for (int i = 0; i < size; i++) {
    const double src_val = input_buffer.data[i];
    const double unclamped_quantized_val =
        std::round(qparams.zero_point + src_val / qparams.scale);
    const double quantized_val =
        std::min(255., std::max(0., unclamped_quantized_val));
    const double dst_val = qparams.scale * (quantized_val - qparams.zero_point);
    output_buffer.data[i] = dst_val;
  }

  if (IsDiscardableArray(*model, fakequant_op->inputs[0]) &&
      CountOpsWithInput(*model, fakequant_op->inputs[0]) == 1) {
    model->EraseArray(fakequant_op->inputs[0]);
  }
  model->operators.erase(fakequant_it);

  return true;
}

}  // namespace toco
