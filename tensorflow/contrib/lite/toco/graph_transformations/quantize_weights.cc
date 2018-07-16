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
#include <iterator>
#include <string>
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/graph_transformations/quantization_util.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"

namespace toco {

namespace {

// The minimum number of elements a weights array must have to be quantized
// by this transformation.
// TODO(suharshs): Make this minimum size configurable.
const int kWeightsMinSize = 1024;

// Gets the quantization params from the float array.
void GetQuantizationParamsFromArray(const Array& array,
                                    QuantizationParams* params) {
  const std::vector<float>& float_vals =
      array.GetBuffer<ArrayDataType::kFloat>().data;
  auto minmax = std::minmax_element(float_vals.begin(), float_vals.end());
  *params = tflite::ChooseQuantizationParams<uint8>(
      *minmax.first, *minmax.second, array.narrow_range);
}

}  // namespace

bool QuantizeWeights::Run(Model* model, std::size_t op_index) {
  const auto op_it = model->operators.begin() + op_index;
  Operator* op = op_it->get();

  // Get the weights tensor, if the current operator has one.
  int weights_index;
  if (op->type == OperatorType::kConv ||
      op->type == OperatorType::kDepthwiseConv ||
      op->type == OperatorType::kFullyConnected) {
    weights_index = 1;
  } else if (op->type == OperatorType::kLstmCell) {
    weights_index = LstmCellOperator::WEIGHTS_INPUT;
  } else {
    return false;
  }

  // Return early if the array isn't a constant param, this can happen in early
  // transformation passes until transpose operations following the weight array
  // are resolved.
  const string weights = op->inputs[weights_index];
  if (!IsConstantParameterArray(*model, weights)) {
    return false;
  }

  // Return early if the weight tensor is not type float.
  Array& weights_array = model->GetArray(weights);
  if (weights_array.data_type != ArrayDataType::kFloat) {
    return false;
  }

  // Return early if the tensor is too small. Small tensors don't take up too
  // much space and can result in bad quantization results.
  if (weights_array.GetBuffer<ArrayDataType::kFloat>().data.size() <
      kWeightsMinSize) {
    return false;
  }

  // Quantize the weight tensor to type kUint8.
  QuantizationParams params;
  GetQuantizationParamsFromArray(weights_array, &params);
  QuantizeArray(this, model, weights, ArrayDataType::kUint8, params);

  // Insert a Dequantize operation after the quantized weights tensor.
  auto* dequantize_op = new DequantizeOperator;
  model->operators.emplace(op_it, dequantize_op);

  // Create a new intermediate tensor to connect the Dequantize op to the
  // original op.
  const string dequantized_output =
      AvailableArrayName(*model, weights + "_dequantized");
  Array& dequantized_output_array = model->GetOrCreateArray(dequantized_output);
  dequantized_output_array.data_type = ArrayDataType::kFloat;

  // Connect up the new Dequantize op with the weights and original op.
  op->inputs[weights_index] = dequantized_output;
  dequantize_op->inputs = {weights};
  dequantize_op->outputs = {dequantized_output};

  return true;
}

}  // namespace toco
