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
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

template <ArrayDataType A>
void GetBoundsForQuantizedDataType(float* min, float* max) {
  using limits = std::numeric_limits<DataType<A>>;
  *min = limits::min();
  *max = limits::max();
}

void GetBoundsForQuantizedDataType(ArrayDataType quantized_data_type,
                                   float* min, float* max) {
  // It is important for matching accuracy between TF training and TFLite
  // inference, that the min and max values are float to match TF's
  // FakeQuantWithMinMaxVarsFunctor.
  switch (quantized_data_type) {
    case ArrayDataType::kUint8:
      return GetBoundsForQuantizedDataType<ArrayDataType::kUint8>(min, max);
    case ArrayDataType::kInt8:
      return GetBoundsForQuantizedDataType<ArrayDataType::kInt8>(min, max);
    case ArrayDataType::kUint16:
      return GetBoundsForQuantizedDataType<ArrayDataType::kUint16>(min, max);
    case ArrayDataType::kInt16:
      return GetBoundsForQuantizedDataType<ArrayDataType::kInt16>(min, max);
    case ArrayDataType::kUint32:
      return GetBoundsForQuantizedDataType<ArrayDataType::kUint32>(min, max);
    case ArrayDataType::kInt32:
      return GetBoundsForQuantizedDataType<ArrayDataType::kInt32>(min, max);
    case ArrayDataType::kUint64:
      return GetBoundsForQuantizedDataType<ArrayDataType::kUint64>(min, max);
    case ArrayDataType::kInt64:
      return GetBoundsForQuantizedDataType<ArrayDataType::kInt64>(min, max);
    default:
      LOG(FATAL) << "unhandled quantized data type";
  }
}

::tensorflow::Status ResolveConstantFakeQuant::Run(Model* model,
                                                   std::size_t op_index,
                                                   bool* modified) {
  *modified = false;
  const auto fakequant_it = model->operators.begin() + op_index;
  const auto* fakequant_base_op = fakequant_it->get();
  if (fakequant_base_op->type != OperatorType::kFakeQuant) {
    return ::tensorflow::Status::OK();
  }

  const auto* fakequant_op =
      static_cast<const FakeQuantOperator*>(fakequant_base_op);

  // Yield until the fakequant MinMax has been resolved.
  if (!fakequant_op->minmax) {
    return ::tensorflow::Status::OK();
  }

  // This transformation only applies when the input array is constant.
  if (!IsConstantParameterArray(*model, fakequant_op->inputs[0])) {
    return ::tensorflow::Status::OK();
  }

  const auto& input_array = model->GetArray(fakequant_op->inputs[0]);
  CHECK(input_array.data_type == ArrayDataType::kFloat);

  // Determine the final data type in the same way as PropagateFakeQuantNumBits.
  ArrayDataType quantized_data_type = input_array.final_data_type;
  if (!InferQuantizedDataTypeFromFakeQuant(*fakequant_op,
                                           &quantized_data_type)) {
    AddMessageF("Unsupported FakeQuant num_bits=%d", fakequant_op->num_bits);
    return ::tensorflow::Status::OK();
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
  ChooseQuantizationParamsForArrayAndQuantizedDataType(
      output_array, quantized_data_type, &qparams);
  float quantized_min, quantized_max;
  GetBoundsForQuantizedDataType(quantized_data_type, &quantized_min,
                                &quantized_max);
  if (fakequant_op->narrow_range) {
    quantized_min++;
    output_array.narrow_range = true;
  }

  // It is important for matching accuracy between TF training and TFLite
  // inference, that the following variables are float to match TF's
  // FakeQuantWithMinMaxVarsFunctor.
  const float scale = qparams.scale;
  const float nudged_min = (quantized_min - qparams.zero_point) * scale;
  const float nudged_max = (quantized_max - qparams.zero_point) * scale;
  tflite::FakeQuantizeArray(scale, nudged_min, nudged_max,
                            input_buffer.data.data(), output_buffer.data.data(),
                            size);
  DeleteOpAndArrays(model, fakequant_op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
