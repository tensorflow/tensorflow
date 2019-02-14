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
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

void FuseAddOrSubParamsIntoFollowingAffine(Model* model, Operator* following_op,
                                           const Operator* add_or_sub_op,
                                           int index_of_constant_input) {
  CHECK(add_or_sub_op->type == OperatorType::kAdd ||
        add_or_sub_op->type == OperatorType::kSub);
  CHECK(index_of_constant_input == 0 || index_of_constant_input == 1);
  // If the op is a subtraction, the constant input should be the right hand
  // side.
  // This should have been checked before this point.
  CHECK(add_or_sub_op->type != OperatorType::kSub ||
        index_of_constant_input == 1);
  if (following_op->inputs.size() < 3) {
    LOG(FATAL) << "Missing bias parameter";
  }
  const auto& weights = model->GetArray(following_op->inputs[1]);
  auto& bias = model->GetArray(following_op->inputs[2]);
  bias.minmax = nullptr;
  const auto& operand =
      model->GetArray(add_or_sub_op->inputs[index_of_constant_input]);
  // We're only supporting the case of a scalar operand. Should have
  // been checked earlier.
  CHECK_EQ(RequiredBufferSizeForShape(operand.shape()), 1);

  const float scalar_operand =
      operand.GetBuffer<ArrayDataType::kFloat>().data[0];
  // At this point we reduce the case of subtraction to that of addition
  // by negating the operand.
  float add_scalar_operand = 0.f;
  if (add_or_sub_op->type == OperatorType::kAdd) {
    add_scalar_operand = scalar_operand;
  } else if (add_or_sub_op->type == OperatorType::kSub &&
             index_of_constant_input == 1) {
    add_scalar_operand = -scalar_operand;
  } else {
    LOG(FATAL) << "Should not get here";
  }
  // From here on we are fusing an addition. add_or_sub_op->type does not
  // matter anymore.

  const Shape& weights_shape = weights.shape();
  const Shape& bias_shape = bias.shape();
  const auto& weights_buffer = weights.GetBuffer<ArrayDataType::kFloat>();
  const float* const weights_data = weights_buffer.data.data();
  auto& bias_buffer = bias.GetMutableBuffer<ArrayDataType::kFloat>();
  float* const bias_data = bias_buffer.data.data();

  if (following_op->type == OperatorType::kConv ||
      following_op->type == OperatorType::kFullyConnected) {
    const int output_depth = weights_shape.dims(0);
    // TODO(b/62904716): Bias array should become 1-D when padding removed.
    CHECK_EQ(output_depth, bias_shape.dims(bias_shape.dimensions_count() - 1));
    const int weights_size = RequiredBufferSizeForShape(weights_shape);
    const int weights_per_depth = weights_size / output_depth;
    CHECK_EQ(weights_size, weights_per_depth * output_depth);

    for (int d = 0; d < output_depth; d++) {
      float accumulation = 0;
      for (int i = 0; i < weights_per_depth; i++) {
        accumulation +=
            add_scalar_operand * weights_data[d * weights_per_depth + i];
      }
      bias_data[d] += accumulation;
    }
  } else if (following_op->type == OperatorType::kDepthwiseConv) {
    const int output_depth =
        weights_shape.dims(weights_shape.dimensions_count() - 1);
    const int weights_size = RequiredBufferSizeForShape(weights_shape);
    const int weights_per_depth = weights_size / output_depth;
    CHECK_EQ(weights_size, weights_per_depth * output_depth);

    for (int c = 0; c < output_depth; c++) {
      float accumulation = 0;
      for (int k = 0; k < weights_per_depth; k++) {
        accumulation += add_scalar_operand * weights_data[k * output_depth + c];
      }
      bias_data[c] += accumulation;
    }
  } else {
    LOG(FATAL) << "Should not get here.";
  }
}

void FuseMulOrDivParamsIntoFollowingAffine(Model* model, Operator* following_op,
                                           const Operator* mul_or_div_op,
                                           int index_of_constant_input) {
  CHECK(mul_or_div_op->type == OperatorType::kMul ||
        mul_or_div_op->type == OperatorType::kDiv);
  CHECK(index_of_constant_input == 0 || index_of_constant_input == 1);
  // If the op is a division, the constant input should be the right hand side.
  // This should have been checked before this point.
  CHECK(mul_or_div_op->type != OperatorType::kDiv ||
        index_of_constant_input == 1);
  const auto& weights_name = following_op->inputs[1];
  const auto& bias_name = following_op->inputs[2];
  auto& weights = model->GetArray(weights_name);
  DropMinMax(model, weights_name);
  DropMinMax(model, bias_name);
  const auto& operand =
      model->GetArray(mul_or_div_op->inputs[index_of_constant_input]);
  // We're only supporting the case of a scalar operand. Should have
  // been checked earlier.
  CHECK_EQ(RequiredBufferSizeForShape(operand.shape()), 1);

  const float scalar_operand =
      operand.GetBuffer<ArrayDataType::kFloat>().data[0];

  float* weights_data =
      weights.GetMutableBuffer<ArrayDataType::kFloat>().data.data();
  const int weights_size = RequiredBufferSizeForShape(weights.shape());
  for (int i = 0; i < weights_size; i++) {
    if (mul_or_div_op->type == OperatorType::kMul) {
      weights_data[i] *= scalar_operand;
    } else if (mul_or_div_op->type == OperatorType::kDiv) {
      weights_data[i] /= scalar_operand;
    } else {
      LOG(FATAL) << "Should not get here";
    }
  }
}

}  // namespace

::tensorflow::Status FuseBinaryIntoFollowingAffine::Run(Model* model,
                                                        std::size_t op_index,
                                                        bool* modified) {
  *modified = false;
  const auto binary_it = model->operators.begin() + op_index;
  auto* binary_op = binary_it->get();
  if (binary_op->type != OperatorType::kAdd &&
      binary_op->type != OperatorType::kMul &&
      binary_op->type != OperatorType::kSub &&
      binary_op->type != OperatorType::kDiv) {
    return ::tensorflow::Status::OK();
  }

  CHECK_EQ(binary_op->inputs.size(), 2);

  // We only can fuse an binary when the two operands break down as follows:
  //   1. One operand is the (variable) output of a typical affine (linear plus
  //   bias)
  //      op of a finite list of possible types: at the moment Conv,
  //      DepthwiseConv and
  //      FullyConnected are supported.
  //   2. The other operand is a constant param array.
  const bool is_input_constant[2] = {
      IsConstantParameterArray(*model, binary_op->inputs[0]),
      IsConstantParameterArray(*model, binary_op->inputs[1]),
  };
  if (!is_input_constant[0] && !is_input_constant[1]) {
    // Neither input is constant, so nothing we can fuse into a constant.
    return ::tensorflow::Status::OK();
  }
  if (is_input_constant[0] && is_input_constant[1]) {
    // Both inputs are constants. That's a job for constants
    // propagation, not for us to handle here.
    return ::tensorflow::Status::OK();
  }
  const int index_of_constant_input = is_input_constant[0] ? 0 : 1;
  const int index_of_variable_input = is_input_constant[0] ? 1 : 0;
  CHECK(is_input_constant[index_of_constant_input]);
  CHECK(!is_input_constant[index_of_variable_input]);

  // For division, we can only fuse if the denominator is constant.
  if (binary_op->type == OperatorType::kDiv) {
    if (index_of_constant_input != 1) {
      AddMessageF("Not fusing %s because the denominator is not constant",
                  LogName(*binary_op));
      return ::tensorflow::Status::OK();
    }
  }

  const auto& operand_shape =
      model->GetArray(binary_op->inputs[index_of_constant_input]).shape();
  for (const auto& dim : operand_shape.dims()) {
    if (dim > 1) {
      AddMessageF(
          "Not fusing %s into the following affine op, because we only know "
          "how to do so when the constant operand is a scalar",
          LogName(*binary_op));
      return ::tensorflow::Status::OK();
    }
  }

  if (binary_op->fused_activation_function !=
      FusedActivationFunctionType::kNone) {
    AddMessageF("Not fusing %s because it has a fused activation function",
                LogName(*binary_op));
    return ::tensorflow::Status::OK();
  }

  if (CountOpsWithInput(*model, binary_op->outputs[0]) != 1) {
    AddMessageF("Not fusing %s because it's consumed by multiple ops",
                LogName(*binary_op));
    return ::tensorflow::Status::OK();
  }

  Operator* following_op = GetOpWithInput(*model, binary_op->outputs[0]);

  if (!following_op) {
    AddMessageF("Not fusing %s because it is not consumed by any op",
                LogName(*binary_op));
    return ::tensorflow::Status::OK();
  }

  if (following_op->type != OperatorType::kConv &&
      following_op->type != OperatorType::kFullyConnected &&
      following_op->type != OperatorType::kDepthwiseConv) {
    AddMessageF(
        "Not fusing %s because the following %s is not of one of the supported "
        "types",
        LogName(*binary_op), LogName(*following_op));
    return ::tensorflow::Status::OK();
  }

  if (following_op->inputs.size() < 3) {
    AddMessageF(
        "Not fusing %s because the following %s does not have a bias vector",
        LogName(*following_op), LogName(*binary_op));
    return ::tensorflow::Status::OK();
  }

  const auto& weights = model->GetArray(following_op->inputs[1]);
  const auto& bias = model->GetArray(following_op->inputs[2]);
  if (!weights.buffer || !bias.buffer) {
    AddMessageF(
        "Not fusing %s because the following %s has non-constant weights or "
        "bias arrays",
        LogName(*binary_op), LogName(*following_op));
    return ::tensorflow::Status::OK();
  }

  // Try to fuse the binary params into the following op's params
  if (binary_op->type == OperatorType::kAdd ||
      binary_op->type == OperatorType::kSub) {
    if (following_op->type == OperatorType::kConv) {
      if (static_cast<ConvOperator*>(following_op)->padding.type !=
          PaddingType::kValid) {
        AddMessageF(
            "Not fusing %s because the following %s does not use VALID padding",
            LogName(*binary_op), LogName(*following_op));
        return ::tensorflow::Status::OK();
      }
    }
    if (following_op->type == OperatorType::kDepthwiseConv) {
      if (static_cast<DepthwiseConvOperator*>(following_op)->padding.type !=
          PaddingType::kValid) {
        AddMessageF(
            "Not fusing %s because the following %s does not use VALID padding",
            LogName(*binary_op), LogName(*following_op));
        return ::tensorflow::Status::OK();
      }
    }
    FuseAddOrSubParamsIntoFollowingAffine(model, following_op, binary_op,
                                          index_of_constant_input);
  } else if (binary_op->type == OperatorType::kMul ||
             binary_op->type == OperatorType::kDiv) {
    FuseMulOrDivParamsIntoFollowingAffine(model, following_op, binary_op,
                                          index_of_constant_input);
  } else {
    LOG(FATAL) << "should not get here";
  }

  AddMessageF("Fusing %s into the following %s", LogName(*binary_op),
              LogName(*following_op));

  model->EraseArray(binary_op->outputs[0]);

  following_op->inputs[0] = binary_op->inputs[index_of_variable_input];
  const auto& old_constant_param_name =
      binary_op->inputs[index_of_constant_input];
  CHECK(IsConstantParameterArray(*model, old_constant_param_name));
  if (CountOpsWithInput(*model, old_constant_param_name) == 1) {
    model->EraseArray(old_constant_param_name);
  }
  model->operators.erase(binary_it);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
