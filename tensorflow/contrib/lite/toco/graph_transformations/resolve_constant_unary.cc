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

bool CopyMinMaxFromFirstInput(const Operator& op, Model* model) {
  auto& output_array = model->GetArray(op.outputs[0]);
  if (output_array.minmax) {
    return false;
  }
  const auto& input_array = model->GetArray(op.inputs[0]);
  if (!input_array.minmax) {
    return false;
  }
  const auto& input_minmax = input_array.GetMinMax();
  CHECK(!output_array.minmax);
  auto& output_minmax = output_array.GetOrCreateMinMax();
  output_minmax.min = input_minmax.min;
  output_minmax.max = input_minmax.max;
  return true;
}

bool ResolveConstantUnaryOperator::Run(Model* model, std::size_t op_index) {
  const auto unary_it = model->operators.begin() + op_index;
  const auto* unary_op = unary_it->get();
  // Test for unary ops of types that we know how to resolve.
  switch (unary_op->type) {
    case OperatorType::kCast:
    case OperatorType::kExp:
    case OperatorType::kLog:
    case OperatorType::kNeg:
    case OperatorType::kRsqrt:
    case OperatorType::kSqrt:
    case OperatorType::kSquare:
    case OperatorType::kSum:
    case OperatorType::kReduceMin:  //  Reduction Min
    case OperatorType::kReduceMax:  //  Reduction Max
    case OperatorType::kReshape:
    case OperatorType::kRelu6:
    case OperatorType::kRelu1:
    case OperatorType::kRelu:
      break;
    default:
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

  // The min-max is only copied for ops that copy data without arithmetic.
  // In future trivial transpose, etc, can be handled here.
  if (unary_op->type == OperatorType::kReshape) {
    CopyMinMaxFromFirstInput(*unary_op, model);
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
  } else if (unary_op->type == OperatorType::kReshape) {
    CHECK(input_buffer_size == output_buffer_size);
    output_float_data = *input_float_data;
  } else if (unary_op->type == OperatorType::kSum) {
    CHECK_EQ(unary_op->inputs.size(), 2) << "Sum needs 2 inputs";
    if (!IsConstantParameterArray(*model, unary_op->inputs[1])) {
      AddMessageF("Axis input is non-constant");
      return false;
    }
    auto& axis_array = model->GetArray(unary_op->inputs[1]);
    CHECK(axis_array.data_type == ArrayDataType::kInt32);
    int axis = axis_array.GetBuffer<ArrayDataType::kInt32>().data[0];
    CHECK_LT(axis, input_shape.dimensions_count()) << "Axis out of bounds";

    // We currently only handle reduction on axis 0.
    CHECK_EQ(axis, 0) << "Only reduction along axis 0 is supported";
    // We currently only handle 1-D and 2-D input tensors.
    CHECK_LE(input_shape.dimensions_count(), 2) << "Rank >2 not yet supported";
    // We only support keep_dims=true; shape prop will need to change otherwise.
    auto sum_op = static_cast<const TensorFlowSumOperator*>(unary_op);
    CHECK(sum_op->keep_dims) << "Only keep_dims=true is supported";

    std::vector<int> indices(input_shape.dimensions_count());
    for (int i = 0; i < input_shape.dims(1); ++i) {
      indices[1] = i;
      float sum = 0.f;
      for (int j = 0; j < input_shape.dims(0); ++j) {
        indices[0] = j;
        sum += (*input_float_data)[Offset(input_shape, indices)];
      }
      output_float_data[i] = sum;
    }
  } else if (unary_op->type == OperatorType::kReduceMin) {
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
  } else if (unary_op->type == OperatorType::kReduceMax) {
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
  } else if (unary_op->type == OperatorType::kExp ||
             unary_op->type == OperatorType::kNeg ||
             unary_op->type == OperatorType::kLog ||
             unary_op->type == OperatorType::kRsqrt ||
             unary_op->type == OperatorType::kSqrt ||
             unary_op->type == OperatorType::kSquare) {
    // Element-wise ops. Should have perfectly matching sizes here.
    for (int i = 0; i < output_dims_count; i++) {
      CHECK_EQ(output_shape.dims(i), input_shape.dims(i));
    }

    for (int i = 0; i < output_buffer_size; i++) {
      const float val = (*input_float_data)[i];
      float outval = 0.f;
      if (unary_op->type == OperatorType::kExp) {
        outval = std::exp(val);
      } else if (unary_op->type == OperatorType::kNeg) {
        outval = -val;
      } else if (unary_op->type == OperatorType::kLog) {
        outval = std::log(val);
      } else if (unary_op->type == OperatorType::kRsqrt) {
        outval = 1.0f / std::sqrt(val);
      } else if (unary_op->type == OperatorType::kSqrt) {
        outval = std::sqrt(val);
      } else if (unary_op->type == OperatorType::kSquare) {
        outval = val * val;
      } else {
        LOG(FATAL) << "should not get here.";
      }
      output_float_data[i] = outval;
    }
  } else if (unary_op->type == OperatorType::kRelu6 ||
             unary_op->type == OperatorType::kRelu1 ||
             unary_op->type == OperatorType::kRelu) {
    for (size_t i = 0; i < output_buffer_size; ++i) {
      const float value = (*input_float_data)[i];
      float new_value = 0.0f;
      switch (unary_op->type) {
        case OperatorType::kRelu: {
          static constexpr float kLower = 0;
          new_value = value < kLower ? kLower : value;
          break;
        }
        case OperatorType::kRelu1: {
          static constexpr float kUpper = 1;
          static constexpr float kLower = -1;
          new_value = value > kUpper ? kUpper : value < kLower ? kLower : value;
          break;
        }
        case OperatorType::kRelu6: {
          static constexpr float kUpper = 6;
          static constexpr float kLower = 0;
          new_value = value > kUpper ? kUpper : value < kLower ? kLower : value;
          break;
        }
        default:
          LOG(FATAL) << "Unsupported activation function "
                     << LogName(*unary_op);
          return false;
      }
      output_float_data[i] = new_value;
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
