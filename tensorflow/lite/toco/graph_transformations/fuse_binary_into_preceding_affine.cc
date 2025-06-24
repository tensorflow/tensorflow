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
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

int GetBiasIndex(const Operator& op) {
  if (op.type == OperatorType::kConv ||
      op.type == OperatorType::kFullyConnected ||
      op.type == OperatorType::kDepthwiseConv) {
    return 2;
  } else if (op.type == OperatorType::kTransposeConv) {
    return 3;
  }
  LOG(FATAL) << "Unhandled operator type";
  return 0;
}

void FuseAddOrSubParamsIntoPrecedingAffine(Model* model, Operator* preceding_op,
                                           const Operator* add_or_sub_op,
                                           int index_of_constant_input) {
  CHECK(add_or_sub_op->type == OperatorType::kAdd ||
        add_or_sub_op->type == OperatorType::kSub);
  CHECK(index_of_constant_input == 0 || index_of_constant_input == 1);
  if (preceding_op->inputs.size() < 3) {
    LOG(FATAL) << "Missing bias parameter";
  }
  const auto bias_ind = GetBiasIndex(*preceding_op);
  auto& bias = model->GetArray(preceding_op->inputs[bias_ind]);
  bias.minmax = nullptr;
  const auto& operand =
      model->GetArray(add_or_sub_op->inputs[index_of_constant_input]);

  const Shape& bias_shape = bias.shape();
  const Shape& operand_shape = operand.shape();
  auto& bias_buffer = bias.GetMutableBuffer<ArrayDataType::kFloat>();
  float* const bias_data = bias_buffer.data.data();
  const auto& operand_buffer = operand.GetBuffer<ArrayDataType::kFloat>();
  const float* const operand_data = operand_buffer.data.data();

  // TODO(b/62904716): Bias array should become 1-D when padding removed.
  const int depth = bias_shape.dims(bias_shape.dimensions_count() - 1);
  int operand_channel_increment = 0;
  if (operand_shape.dimensions_count() >= 1 &&
      operand_shape.dims(operand_shape.dimensions_count() - 1) ==
          bias_shape.dims(bias_shape.dimensions_count() - 1)) {
    operand_channel_increment = 1;
  } else if (operand_shape.dimensions_count() == 0 ||
             operand_shape.dims(operand_shape.dimensions_count() - 1) == 1) {
    operand_channel_increment = 0;
  } else {
    LOG(FATAL) << "Operand shape mismatch.";
  }

  enum class OpType { BiasPlusOperand, BiasMinusOperand, OperandMinusBias };

  const OpType optype = (add_or_sub_op->type == OperatorType::kAdd)
                            ? OpType::BiasPlusOperand
                            : (index_of_constant_input == 1)
                                  ? OpType::BiasMinusOperand
                                  : OpType::OperandMinusBias;

  int operand_channel = 0;
  for (int i = 0; i < depth; i++) {
    float& bias_val = bias_data[i];
    const float operand_val = operand_data[operand_channel];
    if (optype == OpType::BiasPlusOperand) {
      bias_val += operand_val;
    } else if (optype == OpType::BiasMinusOperand) {
      bias_val -= operand_val;
    } else if (optype == OpType::OperandMinusBias) {
      bias_val = operand_val - bias_val;
    } else {
      LOG(FATAL) << "Should not get here.";
    }
    operand_channel += operand_channel_increment;
  }
}

void FuseMulOrDivParamsIntoPrecedingAffine(Model* model, Operator* preceding_op,
                                           const Operator* mul_or_div_op,
                                           int index_of_constant_input) {
  CHECK(mul_or_div_op->type == OperatorType::kMul ||
        mul_or_div_op->type == OperatorType::kDiv);
  CHECK(index_of_constant_input == 0 || index_of_constant_input == 1);
  // If the op is a division, the constant input should be the right hand side.
  // This should have been checked before this point.
  CHECK(mul_or_div_op->type != OperatorType::kDiv ||
        index_of_constant_input == 1);
  if (preceding_op->inputs.size() < 3) {
    LOG(FATAL) << "Missing bias parameter";
  }
  const auto& weights_name = preceding_op->inputs[1];
  const auto bias_ind = GetBiasIndex(*preceding_op);
  const auto& bias_name = preceding_op->inputs[bias_ind];
  auto& weights = model->GetArray(weights_name);
  DropMinMax(model, weights_name);
  auto& bias = model->GetArray(bias_name);
  DropMinMax(model, bias_name);
  const auto& operand =
      model->GetArray(mul_or_div_op->inputs[index_of_constant_input]);

  const Shape& weights_shape = weights.shape();
  const Shape& bias_shape = bias.shape();
  const Shape& operand_shape = operand.shape();
  auto& weights_buffer = weights.GetMutableBuffer<ArrayDataType::kFloat>();
  float* const weights_data = weights_buffer.data.data();
  auto& bias_buffer = bias.GetMutableBuffer<ArrayDataType::kFloat>();
  float* const bias_data = bias_buffer.data.data();
  const auto& operand_buffer = operand.GetBuffer<ArrayDataType::kFloat>();
  const float* const operand_data = operand_buffer.data.data();

  // We support broadcasting the operand along the depth dimension,
  // when the operand's depth is 1.
  int operand_channel_increment = 0;
  if (operand_shape.dimensions_count() >= 1 &&
      operand_shape.dims(operand_shape.dimensions_count() - 1) ==
          bias_shape.dims(bias_shape.dimensions_count() - 1)) {
    operand_channel_increment = 1;
  } else if (operand_shape.dimensions_count() == 0 ||
             operand_shape.dims(operand_shape.dimensions_count() - 1) == 1) {
    operand_channel_increment = 0;
  } else {
    LOG(FATAL) << "Operand shape mismatch.";
  }

  int output_depth;

  if (preceding_op->type == OperatorType::kConv ||
      preceding_op->type == OperatorType::kFullyConnected ||
      preceding_op->type == OperatorType::kTransposeConv) {
    output_depth = weights_shape.dims(0);
  } else if (preceding_op->type == OperatorType::kDepthwiseConv) {
    output_depth = weights_shape.dims(weights_shape.dimensions_count() - 1);
  } else {
    LOG(FATAL) << "Should not get here";
  }

  const int weights_size = RequiredBufferSizeForShape(weights_shape);
  const int weights_per_depth = weights_size / output_depth;
  CHECK_EQ(weights_size, weights_per_depth * output_depth);

  int operand_channel = 0;
  for (int c = 0; c < output_depth; c++) {
    if (mul_or_div_op->type == OperatorType::kMul) {
      bias_data[c] *= operand_data[operand_channel];
    } else if (mul_or_div_op->type == OperatorType::kDiv) {
      bias_data[c] /= operand_data[operand_channel];
    } else {
      LOG(FATAL) << "Should not get here";
    }
    if (preceding_op->type == OperatorType::kConv ||
        preceding_op->type == OperatorType::kFullyConnected) {
      for (int i = 0; i < weights_per_depth; i++) {
        if (mul_or_div_op->type == OperatorType::kMul) {
          weights_data[c * weights_per_depth + i] *=
              operand_data[operand_channel];
        } else if (mul_or_div_op->type == OperatorType::kDiv) {
          weights_data[c * weights_per_depth + i] /=
              operand_data[operand_channel];
        } else {
          LOG(FATAL) << "Should not get here";
        }
      }
    } else if (preceding_op->type == OperatorType::kDepthwiseConv) {
      for (int k = 0; k < weights_per_depth; k++) {
        if (mul_or_div_op->type == OperatorType::kMul) {
          weights_data[k * output_depth + c] *= operand_data[operand_channel];
        } else if (mul_or_div_op->type == OperatorType::kDiv) {
          weights_data[k * output_depth + c] /= operand_data[operand_channel];
        } else {
          LOG(FATAL) << "Should not get here";
        }
      }
    } else {
      LOG(FATAL) << "Should not get here";
    }
    operand_channel += operand_channel_increment;
  }
}
}  // namespace

absl::Status FuseBinaryIntoPrecedingAffine::Run(Model* model,
                                                std::size_t op_index,
                                                bool* modified) {
  *modified = false;
  const auto binary_it = model->operators.begin() + op_index;
  const auto* binary_op = binary_it->get();
  if (binary_op->type != OperatorType::kAdd &&
      binary_op->type != OperatorType::kMul &&
      binary_op->type != OperatorType::kSub &&
      binary_op->type != OperatorType::kDiv) {
    return absl::OkStatus();
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
    return absl::OkStatus();
  }
  if (is_input_constant[0] && is_input_constant[1]) {
    // Both inputs are constants. That's a job for constants
    // propagation, not for us to handle here.
    return absl::OkStatus();
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
      return absl::OkStatus();
    }
  }

  Operator* preceding_op =
      GetOpWithOutput(*model, binary_op->inputs[index_of_variable_input]);
  if (!preceding_op) {
    AddMessageF("Not fusing %s because it is not the output of another op",
                LogName(*binary_op));
    return absl::OkStatus();
  }

  for (const std::string& output_array : model->flags.output_arrays()) {
    if (preceding_op->outputs[0] == output_array) {
      return absl::OkStatus();
    }
  }

  if (preceding_op->type != OperatorType::kConv &&
      preceding_op->type != OperatorType::kFullyConnected &&
      preceding_op->type != OperatorType::kDepthwiseConv &&
      preceding_op->type != OperatorType::kTransposeConv) {
    AddMessageF(
        "Not fusing %s because the preceding %s is not of one of the supported "
        "types",
        LogName(*binary_op), LogName(*preceding_op));
    return absl::OkStatus();
  }

  if (preceding_op->type == OperatorType::kTransposeConv &&
      binary_op->type != OperatorType::kAdd) {
    AddMessageF("Not fusing %s to preceding %s", LogName(*binary_op),
                LogName(*preceding_op));
    return absl::OkStatus();
  }

  if (preceding_op->fused_activation_function !=
      FusedActivationFunctionType::kNone) {
    AddMessageF(
        "Not fusing %s because the preceding %s has a fused activation "
        "function",
        LogName(*binary_op), LogName(*preceding_op));
    return absl::OkStatus();
  }

  if (preceding_op->inputs.size() < 3) {
    AddMessageF(
        "Not fusing %s because the preceding %s does not have a bias vector",
        LogName(*binary_op), LogName(*preceding_op));
    return absl::OkStatus();
  }

  const auto& weights_name = preceding_op->inputs[1];
  const auto bias_ind = GetBiasIndex(*preceding_op);
  const auto& bias_name = preceding_op->inputs[bias_ind];
  const auto& weights = model->GetArray(weights_name);
  const auto& bias = model->GetArray(bias_name);

  if (weights.data_type != ArrayDataType::kFloat ||
      bias.data_type != ArrayDataType::kFloat) {
    AddMessageF(
        "Not fusing %s into preceding %s because one of weights or bias array "
        "is not float (types are %s and %s)",
        LogName(*binary_op), LogName(*preceding_op),
        ArrayDataTypeName(weights.data_type),
        ArrayDataTypeName(bias.data_type));
    return absl::OkStatus();
  }

  const int count_ops_consuming_bias = CountOpsWithInput(*model, bias_name);
  const int count_ops_consuming_weights =
      CountOpsWithInput(*model, weights_name);

  if (binary_op->type == OperatorType::kAdd ||
      binary_op->type == OperatorType::kSub) {
    if (!bias.buffer) {
      AddMessageF(
          "Not fusing %s because the preceding %s has a non-constant bias "
          "array",
          LogName(*binary_op), LogName(*preceding_op));
      return absl::OkStatus();
    }
    if (count_ops_consuming_bias > 1) {
      AddMessageF(
          "Not fusing %s because the bias of the preceding %s is consumed by "
          "another op",
          LogName(*binary_op), LogName(*preceding_op));
      return absl::OkStatus();
    }
  } else {
    if (!weights.buffer || !bias.buffer) {
      AddMessageF(
          "Not fusing %s because the preceding %s has non-constant weights or "
          "bias arrays",
          LogName(*binary_op), LogName(*preceding_op));
      return absl::OkStatus();
    }
    if (count_ops_consuming_weights > 1 || count_ops_consuming_bias > 1) {
      AddMessageF(
          "Not fusing %s because the weights or bias of the preceding %s is "
          "consumed by another op",
          LogName(*binary_op), LogName(*preceding_op));
      return absl::OkStatus();
    }
  }

  int count_ops_consuming_output =
      CountOpsWithInput(*model, preceding_op->outputs[0]);
  DCHECK_GE(count_ops_consuming_output, 1);
  if (count_ops_consuming_output > 1) {
    AddMessageF(
        "Not fusing %s because the output of the preceding %s is consumed by "
        "another op",
        LogName(*binary_op), LogName(*preceding_op));
    return absl::OkStatus();
  }

  AddMessageF("Fusing %s into the preceding %s", LogName(*binary_op),
              LogName(*preceding_op));

  if (binary_op->type == OperatorType::kAdd ||
      binary_op->type == OperatorType::kSub) {
    FuseAddOrSubParamsIntoPrecedingAffine(model, preceding_op, binary_op,
                                          index_of_constant_input);
  } else if (binary_op->type == OperatorType::kMul ||
             binary_op->type == OperatorType::kDiv) {
    FuseMulOrDivParamsIntoPrecedingAffine(model, preceding_op, binary_op,
                                          index_of_constant_input);
  } else {
    LOG(FATAL) << "should not get here";
  }

  model->EraseArray(preceding_op->outputs[0]);
  preceding_op->outputs[0] = binary_op->outputs[0];
  preceding_op->fused_activation_function =
      binary_op->fused_activation_function;
  DeleteOpAndArrays(model, binary_op);
  *modified = true;
  return absl::OkStatus();
}

}  // namespace toco
