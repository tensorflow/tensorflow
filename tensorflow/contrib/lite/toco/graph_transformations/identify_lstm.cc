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
#include "tensorflow/contrib/lite/toco/tooling_util.h"

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

bool GetStateArrayForBackEdge(const Model& model,
                              const string& back_edge_source_array,
                              string* state_array = nullptr) {
  for (const auto& rnn_state : model.flags.rnn_states()) {
    if (back_edge_source_array == rnn_state.back_edge_source_array()) {
      // Found LSTM cell output
      if (state_array) {
        *state_array = rnn_state.state_array();
      }
      return true;
    }
  }
  return false;
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
  Operator* x = GetOpWithOutput(model, op.inputs[0]);
  if ((op_type == OperatorType::kNone) && (x != nullptr)) {
    return false;
  }
  if ((op_type != OperatorType::kNone) && (x == nullptr)) {
    return false;
  }

  // Check that first operator, if connected, is of correct type
  if ((x != nullptr) && (x->type != op_type)) {
    return false;
  }

  // Successfully matched. Optionally return matching input operators.
  if (connected_op) {
    *connected_op = x;
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
  Operator* x = GetOpWithOutput(model, op.inputs[0]);
  if ((a_op_type == OperatorType::kNone) && (x != nullptr)) {
    return false;
  }
  if ((a_op_type != OperatorType::kNone) && (x == nullptr)) {
    return false;
  }

  // Check that first operator, if connected, is of correct type
  if ((x != nullptr) && (x->type != a_op_type)) {
    return false;
  }

  // Check if second input is disconnected/connected to an operator
  Operator* y = GetOpWithOutput(model, op.inputs[1]);
  if ((b_op_type == OperatorType::kNone) && (y != nullptr)) {
    return false;
  }
  if ((b_op_type != OperatorType::kNone) && (y == nullptr)) {
    return false;
  }

  // Check that second operator, if connected, is of correct type
  if ((y != nullptr) && (y->type != b_op_type)) {
    return false;
  }

  // Successfully matched. Optionally return matching input operators.
  if (a_op != nullptr) {
    *a_op = x;
  }
  if (b_op != nullptr) {
    *b_op = y;
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
  Operator* x = GetOpWithOutput(model, op.inputs[0]);
  if ((a_op_type == OperatorType::kNone) && (x != nullptr)) {
    return false;
  }
  if ((a_op_type != OperatorType::kNone) && (x == nullptr)) {
    return false;
  }

  // Check that first operator, if connected, is of correct type
  if ((x != nullptr) && (x->type != a_op_type)) {
    return false;
  }

  // Check if second input is disconnected/connected to an operator
  Operator* y = GetOpWithOutput(model, op.inputs[1]);
  if ((b_op_type == OperatorType::kNone) && (y != nullptr)) {
    return false;
  }
  if ((b_op_type != OperatorType::kNone) && (y == nullptr)) {
    return false;
  }

  // Check that second operator, if connected, is of correct type
  if ((y != nullptr) && (y->type != b_op_type)) {
    return false;
  }

  // Check if third input is disconnected/connected to an operator
  Operator* z = GetOpWithOutput(model, op.inputs[2]);
  if ((c_op_type == OperatorType::kNone) && (z != nullptr)) {
    return false;
  }
  if ((c_op_type != OperatorType::kNone) && (z == nullptr)) {
    return false;
  }

  // Check that third operator, if connected, is of correct type
  if ((z != nullptr) && (z->type != c_op_type)) {
    return false;
  }

  // Successfully matched. Optionally return matching input operators.
  if (a_op != nullptr) {
    *a_op = x;
  }
  if (b_op != nullptr) {
    *b_op = y;
  }
  if (c_op != nullptr) {
    *c_op = z;
  }
  return true;
}

}  // namespace

bool IdentifyLstmCell::Run(Model* model, std::size_t op_index) {
  // This LSTM cell identification method is not invariant to commutation of
  // commutative operator inputs. For example, if input[0] and input[1] of the
  // final output multiplication were swapped, this method would not identify it
  // as an LSTM cell. This is OK in most cases, because
  // tf.rnn.contrib.BasicLSTMCell always generates LSTM cells the same way.

  // Final output multiply
  auto op_it = model->operators.begin() + op_index;
  Operator* final_output_mul = op_it->get();
  if (final_output_mul->type != OperatorType::kMul) {
    return false;
  }
  Operator *state_output_tanh, *fc_output_sig;
  if (!MatchOperatorInputs(*final_output_mul, *model, OperatorType::kTanh,
                           &state_output_tanh, OperatorType::kLogistic,
                           &fc_output_sig)) {
    return false;
  }

  // State output TanH
  // (We don't count an operator as ID'd until we verify it has the correct
  // operator types feeding into it.)
  Operator* state_combine_add;
  if (!MatchOperatorInputs(*state_output_tanh, *model, OperatorType::kAdd,
                           &state_combine_add)) {
    return false;
  }
  string prev_state;
  if (!GetStateArrayForBackEdge(*model, state_output_tanh->inputs[0],
                                &prev_state)) {
    return false;
  }

  // State forget & remember addition
  Operator *state_forget_mul, *state_remember_mul;
  if (!MatchOperatorInputs(*state_combine_add, *model, OperatorType::kMul,
                           &state_forget_mul, OperatorType::kMul,
                           &state_remember_mul)) {
    return false;
  }
  if (state_forget_mul->inputs[0] != prev_state) {
    return false;
  }

  // State forget gate
  Operator* state_forget_sig;
  if (!MatchOperatorInputs(*state_forget_mul, *model, OperatorType::kNone,
                           nullptr, OperatorType::kLogistic,
                           &state_forget_sig)) {
    return false;
  }

  // State remember gate
  Operator *state_remember_sig, *state_info_tanh;
  if (!MatchOperatorInputs(*state_remember_mul, *model, OperatorType::kLogistic,
                           &state_remember_sig, OperatorType::kTanh,
                           &state_info_tanh)) {
    return false;
  }

  // State remember "information" activation function
  Operator* fc_output_split;
  if (!MatchOperatorInputs(*state_info_tanh, *model,
                           OperatorType::kTensorFlowSplit, &fc_output_split)) {
    return false;
  }
  // State remember gate activation function
  Operator* tmp;
  if (!MatchOperatorInputs(*state_remember_sig, *model,
                           OperatorType::kTensorFlowSplit, &tmp) ||
      (tmp != fc_output_split)) {
    return false;
  }
  // State forget gate activation function
  if (!MatchOperatorInputs(*state_forget_sig, *model,
                           OperatorType::kTensorFlowSplit, &tmp) ||
      (tmp != fc_output_split)) {
    return false;
  }
  // Fully connected output activation function
  if (!MatchOperatorInputs(*fc_output_sig, *model,
                           OperatorType::kTensorFlowSplit, &tmp) ||
      (tmp != fc_output_split)) {
    return false;
  }
  // Fully connected output split
  Operator* fully_connected;
  if (!MatchOperatorInputs(*fc_output_split, *model, OperatorType::kNone,
                           nullptr, OperatorType::kFullyConnected,
                           &fully_connected)) {
    return false;
  }

  // Fully connected op
  Operator* concat_inputs;
  if (!MatchOperatorInputs(*fully_connected, *model,
                           OperatorType::kConcatenation, &concat_inputs,
                           OperatorType::kNone, nullptr, OperatorType::kNone,
                           nullptr)) {
    return false;
  }

  if (static_cast<FullyConnectedOperator*>(fully_connected)
          ->experimental_shuffled_weights) {
    // Not yet implemented: experimental shuffled weights in fused LSTM cell.
    return false;
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
  const string base_name(FindLongestCommonPrefix(
      lstm_cell_op->outputs[LstmCellOperator::STATE_OUTPUT],
      lstm_cell_op->outputs[LstmCellOperator::ACTIV_OUTPUT]));
  const string& concat_temp_array_name =
      AvailableArrayName(*model, base_name + "concat_temp");
  model->GetOrCreateArray(concat_temp_array_name);
  lstm_cell_op->outputs[LstmCellOperator::CONCAT_TEMP] = concat_temp_array_name;
  const string& activ_temp_array_name =
      AvailableArrayName(*model, base_name + "activ_temp");
  model->GetOrCreateArray(activ_temp_array_name);
  lstm_cell_op->outputs[LstmCellOperator::ACTIV_TEMP] = activ_temp_array_name;
  AddMessageF("Created temp outputs %s and %s on operator %s",
              concat_temp_array_name, activ_temp_array_name,
              LogName(*lstm_cell_op));

  // Delete arrays and operators replaced by the LSTM cell operator. Order is
  // important - DeleteArrayIfUnused() only succeeds if dependent operators
  // have been removed first. Start at the output and work towards the input.
  model->operators.erase(FindOperator(model, *final_output_mul));
  DeleteArrayIfUnused(state_output_tanh->outputs[0], model);
  DeleteArrayIfUnused(fc_output_sig->outputs[0], model);
  model->operators.erase(FindOperator(model, *state_output_tanh));
  model->operators.erase(FindOperator(model, *fc_output_sig));
  model->operators.erase(FindOperator(model, *state_combine_add));
  DeleteArrayIfUnused(state_forget_mul->outputs[0], model);
  DeleteArrayIfUnused(state_remember_mul->outputs[0], model);
  model->operators.erase(FindOperator(model, *state_forget_mul));
  model->operators.erase(FindOperator(model, *state_remember_mul));
  DeleteArrayIfUnused(state_forget_sig->outputs[0], model);
  DeleteArrayIfUnused(state_info_tanh->outputs[0], model);
  DeleteArrayIfUnused(state_remember_sig->outputs[0], model);
  model->operators.erase(FindOperator(model, *state_forget_sig));
  model->operators.erase(FindOperator(model, *state_info_tanh));
  model->operators.erase(FindOperator(model, *state_remember_sig));
  DeleteArrayIfUnused(fc_output_split->outputs[0], model);
  DeleteArrayIfUnused(fc_output_split->outputs[1], model);
  DeleteArrayIfUnused(fc_output_split->outputs[2], model);
  DeleteArrayIfUnused(fc_output_split->outputs[3], model);
  string dims_array = fc_output_split->inputs[0];
  model->operators.erase(FindOperator(model, *fc_output_split));
  DeleteArrayIfUnused(dims_array, model);
  DeleteArrayIfUnused(fully_connected->outputs[0], model);
  model->operators.erase(FindOperator(model, *fully_connected));
  DeleteArrayIfUnused(concat_inputs->outputs[0], model);
  model->operators.erase(FindOperator(model, *concat_inputs));
  return true;
}

}  // namespace toco
