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

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

std::vector<std::unique_ptr<Operator>>::iterator FindOperator(
    Model* model, const Operator& op) {
  auto it = model->operators.begin();
  for (; it != model->operators.end(); ++it) {
    if (it->get() == &op) {
      break;
    }
  }
  return it;
}

bool ValidateSourceOp(const Model& model, const std::string& array_name,
                      OperatorType op_type, Operator** source_op) {
  if (op_type == OperatorType::kNone) {
    CHECK(!source_op);
  } else {
    CHECK(source_op);
    *source_op = GetOpWithOutput(model, array_name);
    if (*source_op == nullptr) {
      return false;
    }

    // Check that first operator, if connected, is of correct type
    if ((*source_op)->type != op_type) {
      return false;
    }
  }

  return true;
}

// Returns true if the given operator has exactly 1 input, and is connected to
// the given op_type.
// We use kNone to indicate an input unattached to an operator output. Usually
// these are the static input arrays.
bool MatchOperatorInputs(const Operator& op, const Model& model,
                         OperatorType op_type, Operator** connected_op) {
  // Check for required number of inputs
  if (op.inputs.size() != 1) {
    return false;
  }

  // Check if first input is disconnected/connected to an operator
  if (!ValidateSourceOp(model, op.inputs[0], op_type, connected_op)) {
    return false;
  }

  return true;
}

// Returns true if the given operator has exactly 2 inputs, which are connected
// to the given op_types.
// We use kNone to indicate an input unattached to an operator output. Usually
// these are the static input arrays.
bool MatchOperatorInputs(const Operator& op, const Model& model,
                         OperatorType a_op_type, Operator** a_op,
                         OperatorType b_op_type, Operator** b_op) {
  // Check for required number of inputs
  if (op.inputs.size() != 2) {
    return false;
  }

  // Check if first input is disconnected/connected to an operator
  if (!ValidateSourceOp(model, op.inputs[0], a_op_type, a_op)) {
    return false;
  }

  // Check if second input is disconnected/connected to an operator
  if (!ValidateSourceOp(model, op.inputs[1], b_op_type, b_op)) {
    return false;
  }

  return true;
}

// Returns true if the given operator has exactly 3 inputs, which are connected
// to the given op_types.
// We use kNone to indicate an input unattached to an operator output. Usually
// these are the static input arrays.
bool MatchOperatorInputs(const Operator& op, const Model& model,
                         OperatorType a_op_type, Operator** a_op,
                         OperatorType b_op_type, Operator** b_op,
                         OperatorType c_op_type, Operator** c_op) {
  // Check for required number of inputs
  if (op.inputs.size() != 3) {
    return false;
  }

  // Check if first input is disconnected/connected to an operator
  if (!ValidateSourceOp(model, op.inputs[0], a_op_type, a_op)) {
    return false;
  }

  // Check if second input is disconnected/connected to an operator
  if (!ValidateSourceOp(model, op.inputs[1], b_op_type, b_op)) {
    return false;
  }

  // Check if third input is disconnected/connected to an operator
  if (!ValidateSourceOp(model, op.inputs[2], c_op_type, c_op)) {
    return false;
  }

  return true;
}

}  // namespace

::tensorflow::Status IdentifyLstmCell::Run(Model* model, std::size_t op_index,
                                           bool* modified) {
  *modified = false;
  // This LSTM cell identification method is not invariant to commutation of
  // commutative operator inputs. For example, if input[0] and input[1] of the
  // final output multiplication were swapped, this method would not identify it
  // as an LSTM cell. This is OK in most cases, because
  // tf.rnn.contrib.BasicLSTMCell always generates LSTM cells the same way.

  // Final output multiply
  auto op_it = model->operators.begin() + op_index;
  Operator* final_output_mul = op_it->get();
  if (final_output_mul->type != OperatorType::kMul) {
    return ::tensorflow::Status::OK();
  }
  // final_output_mul->outputs[0] would be one of the two outputs of our
  // LstmCell. Exit if it does not already have a data type.
  // We won't be able to propagate data types through a fused LstmCell.
  if (model->GetArray(final_output_mul->outputs[0]).data_type ==
      ArrayDataType::kNone) {
    return ::tensorflow::Status::OK();
  }
  Operator *state_output_tanh, *fc_output_sig;
  if (!MatchOperatorInputs(*final_output_mul, *model, OperatorType::kTanh,
                           &state_output_tanh, OperatorType::kLogistic,
                           &fc_output_sig)) {
    return ::tensorflow::Status::OK();
  }
  // state_output_tanh->inputs[0] would be one of the two outputs of our
  // LstmCell. Exit if it does not already have a data type.
  // We won't be able to propagate data types through a fused LstmCell.
  if (model->GetArray(state_output_tanh->inputs[0]).data_type ==
      ArrayDataType::kNone) {
    return ::tensorflow::Status::OK();
  }

  // State output TanH
  // (We don't count an operator as ID'd until we verify it has the correct
  // operator types feeding into it.)
  Operator* state_combine_add;
  if (!MatchOperatorInputs(*state_output_tanh, *model, OperatorType::kAdd,
                           &state_combine_add)) {
    return ::tensorflow::Status::OK();
  }

  // State forget & remember addition
  Operator *state_forget_mul, *state_remember_mul;
  if (!MatchOperatorInputs(*state_combine_add, *model, OperatorType::kMul,
                           &state_forget_mul, OperatorType::kMul,
                           &state_remember_mul)) {
    return ::tensorflow::Status::OK();
  }
  const std::string prev_state = state_forget_mul->inputs[0];

  // State forget gate
  Operator* state_forget_sig;
  if (!MatchOperatorInputs(*state_forget_mul, *model, OperatorType::kNone,
                           nullptr, OperatorType::kLogistic,
                           &state_forget_sig)) {
    return ::tensorflow::Status::OK();
  }

  // State remember gate
  Operator *state_remember_sig, *state_info_tanh;
  if (!MatchOperatorInputs(*state_remember_mul, *model, OperatorType::kLogistic,
                           &state_remember_sig, OperatorType::kTanh,
                           &state_info_tanh)) {
    return ::tensorflow::Status::OK();
  }

  // State remember "information" activation function
  Operator* fc_output_split;
  if (!MatchOperatorInputs(*state_info_tanh, *model, OperatorType::kSplit,
                           &fc_output_split)) {
    return ::tensorflow::Status::OK();
  }
  // State remember gate activation function
  Operator* tmp;
  if (!MatchOperatorInputs(*state_remember_sig, *model, OperatorType::kSplit,
                           &tmp) ||
      (tmp != fc_output_split)) {
    return ::tensorflow::Status::OK();
  }
  // State forget gate activation function
  if (!MatchOperatorInputs(*state_forget_sig, *model, OperatorType::kSplit,
                           &tmp) ||
      (tmp != fc_output_split)) {
    return ::tensorflow::Status::OK();
  }
  // Fully connected output activation function
  if (!MatchOperatorInputs(*fc_output_sig, *model, OperatorType::kSplit,
                           &tmp) ||
      (tmp != fc_output_split)) {
    return ::tensorflow::Status::OK();
  }
  // Fully connected output split
  Operator* fully_connected;
  if (!MatchOperatorInputs(*fc_output_split, *model, OperatorType::kNone,
                           nullptr, OperatorType::kFullyConnected,
                           &fully_connected)) {
    return ::tensorflow::Status::OK();
  }

  // Fully connected op
  Operator* concat_inputs;
  if (!MatchOperatorInputs(*fully_connected, *model,
                           OperatorType::kConcatenation, &concat_inputs,
                           OperatorType::kNone, nullptr, OperatorType::kNone,
                           nullptr)) {
    return ::tensorflow::Status::OK();
  }

  if (static_cast<FullyConnectedOperator*>(fully_connected)->weights_format !=
      FullyConnectedWeightsFormat::kDefault) {
    // Not yet implemented: experimental shuffled weights in fused LSTM cell.
    return ::tensorflow::Status::OK();
  }

  // Emplace a new LSTM cell operator
  auto* lstm_cell_op = new LstmCellOperator;
  lstm_cell_op->inputs.resize(LstmCellOperator::NUM_INPUTS);
  lstm_cell_op->inputs[LstmCellOperator::DATA_INPUT] = concat_inputs->inputs[0];
  lstm_cell_op->inputs[LstmCellOperator::PREV_ACTIV_INPUT] =
      concat_inputs->inputs[1];
  lstm_cell_op->inputs[LstmCellOperator::WEIGHTS_INPUT] =
      fully_connected->inputs[1];
  lstm_cell_op->inputs[LstmCellOperator::BIASES_INPUT] =
      fully_connected->inputs[2];
  lstm_cell_op->inputs[LstmCellOperator::PREV_STATE_INPUT] = prev_state;
  lstm_cell_op->outputs.resize(LstmCellOperator::NUM_OUTPUTS);
  lstm_cell_op->outputs[LstmCellOperator::STATE_OUTPUT] =
      state_output_tanh->inputs[0];
  lstm_cell_op->outputs[LstmCellOperator::ACTIV_OUTPUT] =
      final_output_mul->outputs[0];
  model->operators.emplace(op_it, lstm_cell_op);
  AddMessageF("Creating %s replacing equivalent subgraph",
              LogName(*lstm_cell_op));

  // Create temp arrays used internally during runtime.
  const std::string base_name(FindLongestCommonPrefix(
      lstm_cell_op->outputs[LstmCellOperator::STATE_OUTPUT],
      lstm_cell_op->outputs[LstmCellOperator::ACTIV_OUTPUT]));
  const std::string& concat_temp_array_name =
      AvailableArrayName(*model, base_name + "concat_temp");
  auto& concat_temp_array = model->GetOrCreateArray(concat_temp_array_name);
  concat_temp_array.data_type =
      model->GetArray(concat_inputs->outputs[0]).data_type;
  lstm_cell_op->outputs[LstmCellOperator::CONCAT_TEMP] = concat_temp_array_name;
  const std::string& activ_temp_array_name =
      AvailableArrayName(*model, base_name + "activ_temp");
  auto& activ_temp_array = model->GetOrCreateArray(activ_temp_array_name);
  activ_temp_array.data_type =
      model->GetArray(fully_connected->outputs[0]).data_type;
  lstm_cell_op->outputs[LstmCellOperator::ACTIV_TEMP] = activ_temp_array_name;
  AddMessageF("Created temp outputs %s and %s on operator %s",
              concat_temp_array_name, activ_temp_array_name,
              LogName(*lstm_cell_op));

  DeleteOpAndArrays(model, final_output_mul);
  DeleteOpAndArrays(model, state_output_tanh);
  DeleteOpAndArrays(model, fc_output_sig);
  DeleteOpAndArrays(model, state_combine_add);
  DeleteOpAndArrays(model, state_forget_mul);
  DeleteOpAndArrays(model, state_remember_mul);
  DeleteOpAndArrays(model, state_forget_sig);
  DeleteOpAndArrays(model, state_info_tanh);
  DeleteOpAndArrays(model, state_remember_sig);
  DeleteOpAndArrays(model, fc_output_split);
  DeleteOpAndArrays(model, fully_connected);
  DeleteOpAndArrays(model, concat_inputs);

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
