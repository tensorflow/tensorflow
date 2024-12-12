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
#include <algorithm>
#include <cstdio>
#include <iterator>
#include <memory>
#include <stack>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {
namespace {

std::vector<std::unique_ptr<Operator>>::iterator FindOperator(
    Model* model, const Operator& op) {
  return std::find_if(
      model->operators.begin(), model->operators.end(),
      [&op](const std::unique_ptr<Operator>& ptr) { return ptr.get() == &op; });
}

bool MatchTwoUnpackOps(const Operator& op, const Model& model,
                       Operator** fw_output, Operator** bw_output) {
  if (op.inputs.size() != 2) {
    return false;
  }

  *fw_output = GetOpWithOutput(model, op.inputs[0]);
  *bw_output = GetOpWithOutput(model, op.inputs[1]);
  if (*fw_output == nullptr || *bw_output == nullptr) {
    return false;
  }

  if ((*fw_output)->type != OperatorType::kUnpack ||
      (*bw_output)->type != OperatorType::kUnpack) {
    return false;
  }

  // TODO(renjieliu): Check the shapes are matching.

  return true;
}

bool MatchDynamicBidirectionalSequenceOutputs(Operator* op, const Model& model,
                                              Operator** fw_output,
                                              Operator** bw_output) {
  if (op->inputs.size() != 2) {
    return false;
  }

  // The concat op is already the fw_rnn_output.
  *fw_output = op;
  auto* reverse_output = GetOpWithOutput(model, op->inputs[1]);
  if (*fw_output == nullptr || reverse_output == nullptr) {
    return false;
  }

  if (reverse_output->type != OperatorType::kReverseV2 &&
      reverse_output->type != OperatorType::kReverseSequence) {
    return false;
  }

  *bw_output = reverse_output;

  return true;
}

bool FindUnidirectionalSequenceOp(const Model& model, const Operator& output_op,
                                  OperatorType operator_type,
                                  std::stack<Operator*>* sequence_ops,
                                  Operator** input_op) {
  Operator* op_it = nullptr;
  op_it = GetOpWithOutput(model, output_op.inputs[0]);
  if (op_it == nullptr) {
    return false;
  }

  while (op_it->type == operator_type) {
    sequence_ops->push(op_it);
    // Check the first input of the unidirectional sequence op.
    op_it = GetOpWithOutput(model, op_it->inputs[0]);
    if (op_it == nullptr) {
      return false;
    }
  }

  *input_op = op_it;
  return true;
}

bool CheckTwoUnidirectionalSequenceOpsAreValid(
    const Model& model, std::stack<Operator*> fw_unidirectional_sequence_ops,
    std::stack<Operator*> bw_unidirectional_sequence_ops,
    const Operator* first_fw_sequence_op_input,
    const Operator* first_bw_sequence_op_input, bool is_dynamic_rnn) {
  if (fw_unidirectional_sequence_ops.size() !=
          bw_unidirectional_sequence_ops.size() ||
      fw_unidirectional_sequence_ops.empty()) {
    return false;
  }

  // Fw & bw sequence ops are allowed to have different input shapes, but they
  // need to have the same data type.
  while (!fw_unidirectional_sequence_ops.empty()) {
    Operator* fw_sequence_op = fw_unidirectional_sequence_ops.top();
    Operator* bw_sequence_op = bw_unidirectional_sequence_ops.top();

    if (fw_sequence_op->inputs.size() != bw_sequence_op->inputs.size() ||
        fw_sequence_op->outputs.size() != bw_sequence_op->outputs.size())
      return false;

    // Make sure the inputs datatype matches.
    for (size_t i = 0; i < fw_sequence_op->inputs.size(); ++i) {
      const auto& fw_input_array_name = fw_sequence_op->inputs[i];
      const auto& bw_input_array_name = bw_sequence_op->inputs[i];
      if (model.HasArray(fw_input_array_name) &&
          model.HasArray(bw_input_array_name)) {
        if (model.GetArray(fw_input_array_name).data_type !=
            model.GetArray(bw_input_array_name).data_type)
          return false;
      }
    }

    // Make sure the outputs datatype matches.
    for (size_t i = 0; i < fw_sequence_op->outputs.size(); ++i) {
      const auto& fw_output_array_name = fw_sequence_op->outputs[i];
      const auto& bw_output_array_name = bw_sequence_op->outputs[i];
      if (model.HasArray(fw_output_array_name) &&
          model.HasArray(bw_output_array_name)) {
        if (model.GetArray(fw_output_array_name).data_type !=
            model.GetArray(bw_output_array_name).data_type)
          return false;
      }
    }

    fw_unidirectional_sequence_ops.pop();
    bw_unidirectional_sequence_ops.pop();
  }

  if (is_dynamic_rnn) {
    // For dynamic bidirectional sequence ops, bw_sequence will have a reverse
    // op.
    if (first_bw_sequence_op_input->type != OperatorType::kReverseV2 &&
        first_bw_sequence_op_input->type != OperatorType::kReverseSequence) {
      return false;
    }

    const auto* bw_real_input_op =
        GetOpWithOutput(model, first_bw_sequence_op_input->inputs[0]);
    if (first_fw_sequence_op_input != bw_real_input_op) {
      return false;
    }

  } else {
    // For static bidirectional sequence ops, we should have two pack ops.
    if (first_fw_sequence_op_input->type != OperatorType::kPack ||
        first_bw_sequence_op_input->type != OperatorType::kPack) {
      return false;
    }

    // fw_lstm & bw_lstm should point to the same input, but reversed sequence.
    for (size_t i = 0; i < first_fw_sequence_op_input->inputs.size(); ++i) {
      if (first_fw_sequence_op_input->inputs[i] !=
          first_bw_sequence_op_input
              ->inputs[first_fw_sequence_op_input->inputs.size() - i - 1]) {
        return false;
      }
    }
  }

  return true;
}

void ConstructBidirectionalSequenceOp(
    const Operator& fw_lstm_op, const Operator& bw_lstm_op, Model* model,
    BidirectionalSequenceLstmOperator** bi_op) {
  // TODO(renjieliu): Check the shapes & configurations are equal.
  constexpr int kBidirectionalSequenceLstmInputsCount = 47;
  constexpr int kFwLstmInputsStartIndex = 1;
  constexpr int kBwLstmInputsStartIndex = 18;
  constexpr int kFwInputActivationStartIndex = 35;
  constexpr int kBwInputActivationStartIndex = 37;
  constexpr int kAuxInputStartIndex = 39;
  (*bi_op)->inputs.reserve(kBidirectionalSequenceLstmInputsCount);
  const std::string& input_array_name =
      AvailableArrayName(*model, "bidirectional_sequence_lstm_input_0");
  model->GetOrCreateArray(input_array_name);
  // The input will be changed later.
  (*bi_op)->inputs.push_back(input_array_name);
  int i = 1;
  // Fill in the fw_lstm weights.
  for (; i < kBwLstmInputsStartIndex; ++i) {
    (*bi_op)->inputs.push_back(fw_lstm_op.inputs[i]);
  }

  // Fill in the bw_lstm weights. bidirectional lstm backward weights start
  // from 18.
  for (; i < kFwInputActivationStartIndex; ++i) {
    (*bi_op)->inputs.push_back(
        bw_lstm_op
            .inputs[i - (kBwLstmInputsStartIndex - kFwLstmInputsStartIndex)]);
  }

  // Fill in fw_lstm previous states.
  for (; i < kBwInputActivationStartIndex; ++i) {
    (*bi_op)->inputs.push_back(
        fw_lstm_op.inputs[i - (kFwInputActivationStartIndex -
                               kBwLstmInputsStartIndex)]);
  }

  // Fill in bw_lstm previous states.
  for (; i < kAuxInputStartIndex; ++i) {
    (*bi_op)->inputs.push_back(
        bw_lstm_op.inputs[i - (kBwInputActivationStartIndex -
                               kBwLstmInputsStartIndex)]);
  }

  // TODO(renjieliu): Deal with Auxiliary input and weights for 39 - 47.
  for (; i <= kBidirectionalSequenceLstmInputsCount; ++i) {
    const std::string& temp_array_name = AvailableArrayName(
        *model, "bidirectional_sequence_lstm_temp_" + std::to_string(i));
    model->CreateOptionalArray(temp_array_name);
    (*bi_op)->inputs.push_back(temp_array_name);
  }

  // Deal with outputs.
  (*bi_op)->outputs.reserve(2);
  const std::string& fw_output_array_name =
      AvailableArrayName(*model, "bidirectional_sequence_lstm_fw_output_0");
  const std::string& bw_output_array_name =
      AvailableArrayName(*model, "bidirectional_sequence_lstm_bw_output_0");
  model->GetOrCreateArray(fw_output_array_name);
  model->GetOrCreateArray(bw_output_array_name);
  (*bi_op)->outputs.push_back(fw_output_array_name);
  (*bi_op)->outputs.push_back(bw_output_array_name);
  (*bi_op)->merge_outputs = false;
}

void ConstructBidirectionalSequenceOp(
    const Operator& fw_rnn_op, const Operator& bw_rnn_op, Model* model,
    BidirectionalSequenceRnnOperator** bi_op) {
  // TODO(renjieliu): Check the shapes & configurations are equal.
  constexpr int kBidirectionalSequenceRnnInputsCount = 12;
  constexpr int kFwInputsStartIndex = 1;
  constexpr int kBwInputsStartIndex = 5;
  constexpr int kAuxInputsStartIndex = 9;
  (*bi_op)->inputs.reserve(kBidirectionalSequenceRnnInputsCount);
  const std::string& input_array_name =
      AvailableArrayName(*model, "bidirectional_sequence_rnn_input_0");
  model->GetOrCreateArray(input_array_name);
  // The input will be changed later.
  (*bi_op)->inputs.push_back(input_array_name);
  int i = 1;

  // Fill in the fw_rnn weights.
  for (; i < kBwInputsStartIndex; ++i) {
    (*bi_op)->inputs.push_back(fw_rnn_op.inputs[i]);
  }

  // Fill in the bw_rnn weights.
  for (; i < kAuxInputsStartIndex; ++i) {
    (*bi_op)->inputs.push_back(
        bw_rnn_op.inputs[i - (kBwInputsStartIndex - kFwInputsStartIndex)]);
  }

  // TODO(renjieliu): Deal with optional weights.
  for (; i < kBidirectionalSequenceRnnInputsCount; ++i) {
    const std::string& temp_array_name = AvailableArrayName(
        *model, "bidirectional_sequence_rnn_temp_" + std::to_string(i));
    model->CreateOptionalArray(temp_array_name);
    (*bi_op)->inputs.push_back(temp_array_name);
  }

  // Deal with outputs.
  (*bi_op)->outputs.reserve(2);
  const std::string& fw_output_array_name =
      AvailableArrayName(*model, "bidirectional_sequence_rnn_fw_output_0");
  const std::string& bw_output_array_name =
      AvailableArrayName(*model, "bidirectional_sequence_rnn_bw_output_0");
  model->GetOrCreateArray(fw_output_array_name);
  model->GetOrCreateArray(bw_output_array_name);
  (*bi_op)->outputs.push_back(fw_output_array_name);
  (*bi_op)->outputs.push_back(bw_output_array_name);
  (*bi_op)->merge_outputs = false;
}

template <typename T>
void GroupFwBwSequenceOps(Model* model, std::stack<Operator*> fw_sequence_ops,
                          std::stack<Operator*> bw_sequence_ops,
                          std::vector<T*>* bidirectional_sequence_ops) {
  while (!fw_sequence_ops.empty()) {
    Operator* fw_sequence_op = fw_sequence_ops.top();
    Operator* bw_sequence_op = bw_sequence_ops.top();
    T* bidirectional_sequence_op = new T;
    ConstructBidirectionalSequenceOp(*fw_sequence_op, *bw_sequence_op, model,
                                     &bidirectional_sequence_op);

    bidirectional_sequence_ops->push_back(bidirectional_sequence_op);
    fw_sequence_ops.pop();
    bw_sequence_ops.pop();
  }
}

template <typename T>
void RewireBidirectionalSequenceSequenceOpsConnections(
    OperatorType operator_type, const std::string& input_array_name,
    const std::vector<T*>& bidirectional_sequence_ops,
    std::vector<std::unique_ptr<Operator>>::iterator* op_it, Model* model) {
  int aux_input_index = -1;
  switch (operator_type) {
    case OperatorType::kBidirectionalSequenceLstm:
      aux_input_index = 39;
      break;
    case OperatorType::kBidirectionalSequenceRnn:
      aux_input_index = 9;
      break;
    default:
      // Should not reach here.
      DCHECK(false);
  }
  std::string cur_fw_input = input_array_name;
  std::string cur_bw_input = input_array_name;
  for (size_t i = 0; i < bidirectional_sequence_ops.size(); ++i) {
    DeleteArrayIfUnusedOutsideOfOp(bidirectional_sequence_ops[i]->inputs[0],
                                   bidirectional_sequence_ops[i], model);
    bidirectional_sequence_ops[i]->inputs[0] = cur_fw_input;
    if (i != 0) {
      DeleteArrayIfUnusedOutsideOfOp(
          bidirectional_sequence_ops[i]->inputs[aux_input_index],
          bidirectional_sequence_ops[i], model);
      bidirectional_sequence_ops[i]->inputs[aux_input_index] = cur_bw_input;
    }
    cur_fw_input = bidirectional_sequence_ops[i]->outputs[0];
    cur_bw_input = bidirectional_sequence_ops[i]->outputs[1];
    if (i != (bidirectional_sequence_ops.size() - 1)) {
      bidirectional_sequence_ops[i]->merge_outputs = false;
    } else {
      // TODO(renjieliu): We need to check whether the outputs of the last bidi
      // lstms needs merged outputs or not.
      bidirectional_sequence_ops[i]->merge_outputs = true;
      DeleteArrayIfUnused(bidirectional_sequence_ops[i]->outputs[1], model);
      bidirectional_sequence_ops[i]->outputs.pop_back();
    }
    model->operators.emplace(*op_it, bidirectional_sequence_ops[i]);
    *op_it += 1;
  }
}

template <typename T>
void RewireFinalUnpackOutputs(const UnpackOperator& original_unpack_operator,
                              UnpackOperator** final_unpack_operator,
                              T** final_bidi_sequence_operator, Model* model) {
  (*final_unpack_operator)
      ->inputs.push_back((*final_bidi_sequence_operator)->outputs[0]);
  (*final_unpack_operator)->axis = original_unpack_operator.axis;
  (*final_unpack_operator)->num = original_unpack_operator.num;

  for (size_t i = 0; i < original_unpack_operator.outputs.size(); ++i) {
    const std::string& output_array_name = original_unpack_operator.outputs[i];
    const std::string& final_unpack_output_array_name = AvailableArrayName(
        *model, "bidirectional_sequence_unpack_" + std::to_string(i));
    model->GetOrCreateArray(final_unpack_output_array_name);
    (*final_unpack_operator)->outputs.push_back(final_unpack_output_array_name);
    Operator* unpack_following_op = GetOpWithInput(*model, output_array_name);
    if (unpack_following_op != nullptr) {
      // If there's a following op after the unpack, it must be a concat op.
      DCHECK(unpack_following_op->type == OperatorType::kConcatenation);
      // For every output of the concat, rewire the outputs.
      for (const std::string& concat_output : unpack_following_op->outputs) {
        (*final_unpack_operator)->outputs[i] = concat_output;
      }
      // Remove the concat op.
      DeleteOpAndArrays(model, unpack_following_op);
    }
  }
}

void RemoveUnidirectionalSequenceOps(std::stack<Operator*> uni_sequence_ops,
                                     Model* model) {
  while (!uni_sequence_ops.empty()) {
    Operator* uni_sequence_op = uni_sequence_ops.top();
    DeleteOpAndArrays(model, uni_sequence_op);
    uni_sequence_ops.pop();
  }
}

template <typename T>
absl::Status GroupDynamicSequenceOps(Model* model, std::size_t op_index,
                                     OperatorType operator_type,
                                     bool* modified) {
  *modified = false;

  // We assume there's a concatenation right after the bidirectional sequence
  // ops, it may not be the case.
  auto op_it = model->operators.begin() + op_index;
  Operator* final_concat_op = op_it->get();
  if (final_concat_op->type != OperatorType::kConcatenation &&
      final_concat_op->type != OperatorType::kConcat &&
      final_concat_op->type != OperatorType::kConcatV2) {
    return absl::OkStatus();
  }

  // for bw, there will be a reverse op at the end.
  Operator *fw_sequence_output, *bw_sequence_output;
  if (!MatchDynamicBidirectionalSequenceOutputs(
          final_concat_op, *model, &fw_sequence_output, &bw_sequence_output)) {
    return absl::OkStatus();
  }

  // Find all upstream unidirectional sequence ops.
  std::stack<Operator*> fw_unidirectional_sequence_ops,
      bw_unidirectional_sequence_ops;
  OperatorType unidirectional_op_type;
  if (operator_type == OperatorType::kBidirectionalSequenceLstm) {
    unidirectional_op_type = OperatorType::kUnidirectionalSequenceLstm;
  } else {
    unidirectional_op_type = OperatorType::kUnidirectionalSequenceRnn;
  }
  Operator *first_fw_sequence_input, *first_bw_sequence_input;
  if (!FindUnidirectionalSequenceOp(
          *model, *fw_sequence_output, unidirectional_op_type,
          &fw_unidirectional_sequence_ops, &first_fw_sequence_input) ||
      !FindUnidirectionalSequenceOp(
          *model, *bw_sequence_output, unidirectional_op_type,
          &bw_unidirectional_sequence_ops, &first_bw_sequence_input)) {
    return absl::OkStatus();
  }

  if (!CheckTwoUnidirectionalSequenceOpsAreValid(
          *model, fw_unidirectional_sequence_ops,
          bw_unidirectional_sequence_ops, first_fw_sequence_input,
          first_bw_sequence_input, /*is_dynamic_rnn=*/true)) {
    return absl::OkStatus();
  }

  std::vector<T> bidirectional_sequence_ops;
  GroupFwBwSequenceOps(model, fw_unidirectional_sequence_ops,
                       bw_unidirectional_sequence_ops,
                       &bidirectional_sequence_ops);

  // Rewire the inputs & outputs.
  std::string current_input = first_fw_sequence_input->outputs[0];
  RewireBidirectionalSequenceSequenceOpsConnections(
      operator_type, current_input, bidirectional_sequence_ops, &op_it, model);

  // Change last bidirectional sequence rnn output to the concat output.
  bidirectional_sequence_ops[bidirectional_sequence_ops.size() - 1]
      ->outputs[0] = final_concat_op->outputs[0];

  // Delete unused ops.
  RemoveUnidirectionalSequenceOps(fw_unidirectional_sequence_ops, model);
  RemoveUnidirectionalSequenceOps(bw_unidirectional_sequence_ops, model);
  DeleteOpAndArrays(model, final_concat_op);
  // Only keep the fw lstm's input.
  DeleteOpAndArrays(model, first_bw_sequence_input);
  *modified = true;
  return absl::OkStatus();
}

}  // namespace

absl::Status GroupBidirectionalSequenceLstm::Run(Model* model,
                                                 std::size_t op_index,
                                                 bool* modified) {
  *modified = false;
  // Bidirectional sequence lstm will generate two separate unidirectional
  // sequence lstm ops, for static bidirectional sequence lstm, there will be
  // a concatenation op at very end; for dynamic bidirectional sequence lstm,
  // it is not guaranteed, but currently we do not support that.
  auto op_it = model->operators.begin() + op_index;
  Operator* final_concat_op = op_it->get();
  if (final_concat_op->type != OperatorType::kConcatenation &&
      final_concat_op->type != OperatorType::kConcat &&
      final_concat_op->type != OperatorType::kConcatV2) {
    return absl::OkStatus();
  }

  // Match fw unidirectional lstm outputs and bw unidirectional lstm outputs:
  // should be two unstack ops.
  Operator *fw_lstm_output, *bw_lstm_output;
  if (!MatchTwoUnpackOps(*final_concat_op, *model, &fw_lstm_output,
                         &bw_lstm_output)) {
    return absl::OkStatus();
  }

  // Find all upstream unidirectional lstm ops.
  std::stack<Operator*> fw_unidirectional_sequence_lstm_ops,
      bw_unidirectional_sequence_lstm_ops;
  Operator *first_fw_lstm_input, *first_bw_lstm_input;
  if (!FindUnidirectionalSequenceOp(
          *model, *fw_lstm_output, OperatorType::kUnidirectionalSequenceLstm,
          &fw_unidirectional_sequence_lstm_ops, &first_fw_lstm_input) ||
      !FindUnidirectionalSequenceOp(
          *model, *bw_lstm_output, OperatorType::kUnidirectionalSequenceLstm,
          &bw_unidirectional_sequence_lstm_ops, &first_bw_lstm_input)) {
    return absl::OkStatus();
  }

  if (!CheckTwoUnidirectionalSequenceOpsAreValid(
          *model, fw_unidirectional_sequence_lstm_ops,
          bw_unidirectional_sequence_lstm_ops, first_fw_lstm_input,
          first_bw_lstm_input, /*is_dynamic_rnn=*/false)) {
    return absl::OkStatus();
  }

  std::vector<BidirectionalSequenceLstmOperator*>
      bidirectional_sequence_lstm_ops;
  GroupFwBwSequenceOps(model, fw_unidirectional_sequence_lstm_ops,
                       bw_unidirectional_sequence_lstm_ops,
                       &bidirectional_sequence_lstm_ops);

  // Rewire the inputs & outputs.
  std::string current_input = first_fw_lstm_input->outputs[0];
  RewireBidirectionalSequenceSequenceOpsConnections(
      OperatorType::kBidirectionalSequenceLstm, current_input,
      bidirectional_sequence_lstm_ops, &op_it, model);

  // Insert a unpack op for the output.
  UnpackOperator* unpack_operator = new UnpackOperator;

  RewireFinalUnpackOutputs(
      static_cast<const UnpackOperator&>(*fw_lstm_output), &unpack_operator,
      &bidirectional_sequence_lstm_ops[bidirectional_sequence_lstm_ops.size() -
                                       1],
      model);
  model->operators.emplace(op_it, unpack_operator);

  // Delete unused ops.
  DeleteOpAndArrays(model, fw_lstm_output);
  DeleteOpAndArrays(model, bw_lstm_output);
  RemoveUnidirectionalSequenceOps(fw_unidirectional_sequence_lstm_ops, model);
  RemoveUnidirectionalSequenceOps(bw_unidirectional_sequence_lstm_ops, model);
  // Only keep the fw lstm's pack input.
  DeleteOpAndArrays(model, first_bw_lstm_input);
  *modified = true;
  return absl::OkStatus();
}

absl::Status GroupBidirectionalSequenceRnn::Run(Model* model,
                                                std::size_t op_index,
                                                bool* modified) {
  *modified = false;
  // Bidirectional sequence rnn will generate two separate unidirectional
  // sequence rnn ops, for static bidirectional sequence rnn, there will be
  // a concatenation op at very end; for dynamic bidirectional sequence rnn,
  // it is not guaranteed, but currently we do not support that.
  auto op_it = model->operators.begin() + op_index;
  Operator* final_concat_op = op_it->get();
  if (final_concat_op->type != OperatorType::kConcatenation &&
      final_concat_op->type != OperatorType::kConcat &&
      final_concat_op->type != OperatorType::kConcatV2) {
    return absl::OkStatus();
  }

  // Match fw unidirectional rnn outputs and bw unidirectional rnn outputs:
  // should be two unstack ops.
  Operator *fw_rnn_output, *bw_rnn_output;
  if (!MatchTwoUnpackOps(*final_concat_op, *model, &fw_rnn_output,
                         &bw_rnn_output)) {
    return absl::OkStatus();
  }

  // Find all upstream unidirectional rnn ops.
  std::stack<Operator*> fw_unidirectional_sequence_rnn_ops,
      bw_unidirectional_sequence_rnn_ops;
  Operator *first_fw_rnn_input, *first_bw_rnn_input;
  if (!FindUnidirectionalSequenceOp(
          *model, *fw_rnn_output, OperatorType::kUnidirectionalSequenceRnn,
          &fw_unidirectional_sequence_rnn_ops, &first_fw_rnn_input) ||
      !FindUnidirectionalSequenceOp(
          *model, *bw_rnn_output, OperatorType::kUnidirectionalSequenceRnn,
          &bw_unidirectional_sequence_rnn_ops, &first_bw_rnn_input)) {
    return absl::OkStatus();
  }

  if (!CheckTwoUnidirectionalSequenceOpsAreValid(
          *model, fw_unidirectional_sequence_rnn_ops,
          bw_unidirectional_sequence_rnn_ops, first_fw_rnn_input,
          first_bw_rnn_input, /*is_dynamic_rnn=*/false)) {
    return absl::OkStatus();
  }

  std::vector<BidirectionalSequenceRnnOperator*> bidirectional_sequence_rnn_ops;
  GroupFwBwSequenceOps(model, fw_unidirectional_sequence_rnn_ops,
                       bw_unidirectional_sequence_rnn_ops,
                       &bidirectional_sequence_rnn_ops);

  // Rewire the inputs & outputs.
  std::string current_input = first_fw_rnn_input->outputs[0];
  RewireBidirectionalSequenceSequenceOpsConnections(
      OperatorType::kBidirectionalSequenceRnn, current_input,
      bidirectional_sequence_rnn_ops, &op_it, model);

  // Insert a unpack op for the output.
  UnpackOperator* unpack_operator = new UnpackOperator;
  RewireFinalUnpackOutputs(
      static_cast<const UnpackOperator&>(*fw_rnn_output), &unpack_operator,
      &bidirectional_sequence_rnn_ops[bidirectional_sequence_rnn_ops.size() -
                                      1],
      model);
  model->operators.emplace(op_it, unpack_operator);

  // Delete unused ops.
  DeleteOpAndArrays(model, fw_rnn_output);
  DeleteOpAndArrays(model, bw_rnn_output);
  RemoveUnidirectionalSequenceOps(fw_unidirectional_sequence_rnn_ops, model);
  RemoveUnidirectionalSequenceOps(bw_unidirectional_sequence_rnn_ops, model);
  // Only keep the fw rnn's pack input.
  DeleteOpAndArrays(model, first_bw_rnn_input);
  *modified = true;
  return absl::OkStatus();
}

absl::Status GroupDynamicBidirectionalSequenceRnn::Run(Model* model,
                                                       std::size_t op_index,
                                                       bool* modified) {
  return GroupDynamicSequenceOps<BidirectionalSequenceRnnOperator*>(
      model, op_index, OperatorType::kBidirectionalSequenceRnn, modified);
}

absl::Status GroupDynamicBidirectionalSequenceLstm::Run(Model* model,
                                                        std::size_t op_index,
                                                        bool* modified) {
  return GroupDynamicSequenceOps<BidirectionalSequenceLstmOperator*>(
      model, op_index, OperatorType::kBidirectionalSequenceLstm, modified);
}

}  // namespace toco
