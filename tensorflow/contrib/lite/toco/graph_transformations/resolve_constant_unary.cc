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
#include <string.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/runtime/types.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ResolveConstantUnaryOperator::Run(Model* model, std::size_t op_index) {
  const auto unary_it = model->operators.begin() + op_index;
  const auto* unary_op = unary_it->get();
  // Test for unary ops of types that we know how to resolve
  if (unary_op->type != OperatorType::kCast &&
      unary_op->type != OperatorType::kNeg &&
      unary_op->type != OperatorType::kTensorFlowRsqrt &&
      unary_op->type != OperatorType::kTensorFlowSqrt &&
      unary_op->type != OperatorType::kTensorFlowSquare &&
      unary_op->type != OperatorType::kTensorFlowSum &&
      unary_op->type != OperatorType::kTensorFlowMin &&
      unary_op->type != OperatorType::kTensorFlowMax &&
      unary_op->type != OperatorType::kTensorFlowReshape) {
    return false;
  }
  // Check if the input is a constant parameter.
  if (!IsConstantParameterArray(*model, unary_op->inputs[0])) {
    return false;
  }

  // if the unary op involves a tensor required by a rnn state, ignore it
  for (const auto& rnn_state : model->flags.rnn_states()) {
    if (unary_op->inputs[0] == rnn_state.back_edge_source_array()) {
      return false;
    }
    if (unary_op->inputs[0] == rnn_state.state_array()) {
      return false;
    }
  }

  auto& output_array = model->GetArray(unary_op->outputs[0]);
  if (!output_array.has_shape()) {
    // Yield until the output array dims have been resolved.
    return false;
  }

  // At the moment we don't want to care about fused activation functions.
  // The idea is that we should do the present constants-propagation before
  // activation functions get fused.
  if (unary_op->fused_activation_function !=
      FusedActivationFunctionType::kNone) {
    AddMessageF(
        "Not resolving constant %s "
        " because it has a fused activation function",
        LogName(*unary_op));
    return false;
  }

  const auto& input_array = model->GetArray(unary_op->inputs[0]);
  // We have already tested above for existence of buffers (synonymous to being
  // a constant param).
  CHECK(input_array.buffer);
  std::vector<DataType<ArrayDataType::kFloat>> const* input_float_data;
  if (unary_op->type == OperatorType::kCast) {
    CastOperator const* cast_op = static_cast<CastOperator const*>(unary_op);
    if (cast_op->dst_data_type != ArrayDataType::kFloat) {
      AddMessageF(
          "Not resolving constant %s because we currently only support casting "
          "to float",
          LogName(*unary_op));
      return false;
    }
    if (cast_op->src_data_type != input_array.buffer->type) {
      AddMessageF(
          "Not resolving constant %s because cast op source type does not "
          "match input type",
          LogName(*unary_op));
    }
  } else {
    if (input_array.buffer->type != ArrayDataType::kFloat) {
      return false;
    }
    input_float_data = &(input_array.GetBuffer<ArrayDataType::kFloat>().data);
  }

  // Create a float buffer on the output array, which are always constant.
  const Shape& output_shape = output_array.shape();
  const int output_dims_count = output_shape.dimensions_count();
  const int output_buffer_size = RequiredBufferSizeForShape(output_shape);
  auto& output_float_data =
      output_array.GetMutableBuffer<ArrayDataType::kFloat>().data;
  output_float_data.resize(output_buffer_size);

  const Shape& input_shape = input_array.shape();
  const int input_buffer_size = RequiredBufferSizeForShape(input_shape);
  if (unary_op->type == OperatorType::kCast) {
    for (int i = 0; i < output_buffer_size; i++) {
      float outval = 0.0f;
      if (input_array.buffer->type == ArrayDataType::kFloat) {
        outval = static_cast<float>(
            input_array.GetBuffer<ArrayDataType::kFloat>().data[i]);
      } else if (input_array.buffer->type == ArrayDataType::kUint8) {
        outval = static_cast<float>(
            input_array.GetBuffer<ArrayDataType::kUint8>().data[i]);
      } else if (input_array.buffer->type == ArrayDataType::kInt32) {
        outval = static_cast<float>(
            input_array.GetBuffer<ArrayDataType::kInt32>().data[i]);
      } else if (input_array.buffer->type == ArrayDataType::kInt64) {
        outval = static_cast<float>(
            input_array.GetBuffer<ArrayDataType::kInt64>().data[i]);
      } else {
        LOG(FATAL) << "Unsupported cast op input type";
      }
      output_float_data[i] = outval;
    }
  } else if (unary_op->type == OperatorType::kTensorFlowReshape) {
    CHECK(input_buffer_size == output_buffer_size);
    memcpy(output_float_data.data(), (*input_float_data).data(),
           output_buffer_size * sizeof(output_float_data[0]));
  } else if (unary_op->type == OperatorType::kTensorFlowSum) {
    // At the moment only full reduction across all dimensions is supported.
    float sum = 0.f;
    for (int i = 0; i < input_buffer_size; i++) {
      sum += (*input_float_data)[i];
    }
    for (int i = 0; i < output_buffer_size; ++i) {
      output_float_data[i] = sum;
    }
  } else if (unary_op->type == OperatorType::kTensorFlowMin) {
    // At the moment only full reduction across all dimensions is supported.
    // TODO(starka): Output should not be padded.
    for (int i = 0; i < output_dims_count; i++) {
      CHECK_EQ(output_shape.dims(i), 1);
    }
    float min = (*input_float_data)[0];
    for (int i = 0; i < input_buffer_size; i++) {
      min = std::min(min, (*input_float_data)[i]);
    }
    output_float_data[0] = min;
  } else if (unary_op->type == OperatorType::kTensorFlowMax) {
    // At the moment only full reduction across all dimensions is supported.
    // TODO(starka): Output should not be padded.
    for (int i = 0; i < output_dims_count; i++) {
      CHECK_EQ(output_shape.dims(i), 1);
    }
    float max = (*input_float_data)[0];
    for (int i = 0; i < input_buffer_size; i++) {
      max = std::max(max, (*input_float_data)[i]);
    }
    output_float_data[0] = max;
  } else if (unary_op->type == OperatorType::kNeg ||
             unary_op->type == OperatorType::kTensorFlowRsqrt ||
             unary_op->type == OperatorType::kTensorFlowSqrt ||
             unary_op->type == OperatorType::kTensorFlowSquare) {
    // Element-wise ops. Should have perfectly matching sizes here.
    for (int i = 0; i < output_dims_count; i++) {
      CHECK_EQ(output_shape.dims(i), input_shape.dims(i));
    }

    for (int i = 0; i < output_buffer_size; i++) {
      const float val = (*input_float_data)[i];
      float outval = 0.f;
      if (unary_op->type == OperatorType::kNeg) {
        outval = -val;
      } else if (unary_op->type == OperatorType::kTensorFlowRsqrt) {
        outval = 1.0f / std::sqrt(val);
      } else if (unary_op->type == OperatorType::kTensorFlowSqrt) {
        outval = std::sqrt(val);
      } else if (unary_op->type == OperatorType::kTensorFlowSquare) {
        outval = val * val;
      } else {
        LOG(FATAL) << "should not get here.";
      }
      output_float_data[i] = outval;
    }
  } else {
    LOG(FATAL) << "should not get here.";
  }
  for (const auto& input : unary_op->inputs) {
    if (CountOpsWithInput(*model, input) == 1) {
      model->EraseArray(input);
    }
  }
  AddMessageF("Resolved constant %s to the equivalent constant array",
              LogName(*unary_op));
  model->operators.erase(unary_it);
  return true;
}

}  // namespace toco
