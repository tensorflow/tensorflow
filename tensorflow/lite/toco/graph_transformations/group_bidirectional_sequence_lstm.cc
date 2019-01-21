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
#include <cstdio>
#include <iterator>
#include <memory>
#include <stack>
#include <string>
#include <vector>

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

bool FindUnidirectionalSequenceLstmOp(const Model& model,
                                      const Operator& output_op,
                                      std::stack<Operator*>* lstm_ops,
                                      Operator** input_op) {
  Operator* op_it = nullptr;
  op_it = GetOpWithOutput(model, output_op.inputs[0]);
  if (op_it == nullptr) {
    return false;
  }

  while (op_it->type == OperatorType::kUnidirectionalSequenceLstm) {
    lstm_ops->push(op_it);
    // Check the first input of the unidirectional squence lstm op.
    op_it = GetOpWithOutput(model, op_it->inputs[0]);
    if (op_it == nullptr) {
      return false;
    }
  }

  *input_op = op_it;
  return true;
}

bool ConstructBidirectionalSequenceLstmOp(
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
  const string& input_array_name =
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
    const string& temp_array_name = AvailableArrayName(
        *model, "bidirectional_sequence_lstm_temp_" + std::to_string(i));
    model->CreateOptionalArray(temp_array_name);
    (*bi_op)->inputs.push_back(temp_array_name);
  }

  // Deal with outputs.
  (*bi_op)->outputs.reserve(2);
  const string& fw_output_array_name =
      AvailableArrayName(*model, "bidirectional_sequence_lstm_fw_output_0");
  const string& bw_output_array_name =
      AvailableArrayName(*model, "bidirectional_sequence_lstm_bw_output_0");
  model->GetOrCreateArray(fw_output_array_name);
  model->GetOrCreateArray(bw_output_array_name);
  (*bi_op)->outputs.push_back(fw_output_array_name);
  (*bi_op)->outputs.push_back(bw_output_array_name);
  (*bi_op)->merge_outputs = false;
  return true;
}

bool GroupFwBwLstmOps(
    Model* model, std::stack<Operator*> fw_lstm_ops,
    std::stack<Operator*> bw_lstm_ops,
    std::vector<BidirectionalSequenceLstmOperator*>* bidirectional_lstm_ops) {
  while (!fw_lstm_ops.empty()) {
    Operator* fw_lstm_op = fw_lstm_ops.top();
    Operator* bw_lstm_op = bw_lstm_ops.top();
    BidirectionalSequenceLstmOperator* bidirectional_sequence_lstm_op =
        new BidirectionalSequenceLstmOperator;
    if (!ConstructBidirectionalSequenceLstmOp(
            *fw_lstm_op, *bw_lstm_op, model, &bidirectional_sequence_lstm_op)) {
      return false;
    }

    bidirectional_lstm_ops->push_back(bidirectional_sequence_lstm_op);
    fw_lstm_ops.pop();
    bw_lstm_ops.pop();
  }
  return true;
}

void RewireBidirectionalSequenceLstmOpsConnections(
    const string& input_array_name,
    const std::vector<BidirectionalSequenceLstmOperator*>&
        bidirectional_sequence_lstm_ops,
    std::vector<std::unique_ptr<Operator>>::iterator* op_it, Model* model) {
  string cur_fw_input = input_array_name;
  string cur_bw_input = input_array_name;
  for (int i = 0; i < bidirectional_sequence_lstm_ops.size(); ++i) {
    DeleteArrayIfUsedOnce(bidirectional_sequence_lstm_ops[i]->inputs[0], model);
    bidirectional_sequence_lstm_ops[i]->inputs[0] = cur_fw_input;
    if (i != 0) {
      DeleteArrayIfUsedOnce(bidirectional_sequence_lstm_ops[i]->inputs[39],
                            model);
      bidirectional_sequence_lstm_ops[i]->inputs[39] = cur_bw_input;
    }
    cur_fw_input = bidirectional_sequence_lstm_ops[i]->outputs[0];
    cur_bw_input = bidirectional_sequence_lstm_ops[i]->outputs[1];
    if (i != (bidirectional_sequence_lstm_ops.size() - 1)) {
      bidirectional_sequence_lstm_ops[i]->merge_outputs = false;
    } else {
      // TODO(renjieliu): We need to check whether the outputs of the last bidi
      // lstms needs merged outputs or not.
      bidirectional_sequence_lstm_ops[i]->merge_outputs = true;
      DeleteArrayIfUnused(bidirectional_sequence_lstm_ops[i]->outputs[1],
                          model);
      bidirectional_sequence_lstm_ops[i]->outputs.pop_back();
    }
    model->operators.emplace(*op_it, bidirectional_sequence_lstm_ops[i]);
    *op_it += 1;
  }
}

void RewireFinalUnpackOutputs(
    const UnpackOperator& original_unpack_operator,
    UnpackOperator** final_unpack_operator,
    BidirectionalSequenceLstmOperator** final_bidi_lstm_operator,
    Model* model) {
  (*final_unpack_operator)
      ->inputs.push_back((*final_bidi_lstm_operator)->outputs[0]);
  (*final_unpack_operator)->axis = original_unpack_operator.axis;
  (*final_unpack_operator)->num = original_unpack_operator.num;

  for (int i = 0; i < original_unpack_operator.outputs.size(); ++i) {
    const string& output_array_name = original_unpack_operator.outputs[i];
    const string& final_unpack_output_array_name = AvailableArrayName(
        *model, "bidirectional_sequence_lstm_unpack_" + std::to_string(i));
    model->GetOrCreateArray(final_unpack_output_array_name);
    (*final_unpack_operator)->outputs.push_back(final_unpack_output_array_name);
    Operator* unpack_following_op = GetOpWithInput(*model, output_array_name);
    if (unpack_following_op != nullptr) {
      // If there's a following op after the unpack, it must be a concat op.
      DCHECK(unpack_following_op->type == OperatorType::kConcatenation);
      // For every output of the concat, rewire the outputs.
      for (const string& concat_output : unpack_following_op->outputs) {
        (*final_unpack_operator)->outputs[i] = concat_output;
      }
      // Remove the concat op.
      model->operators.erase(FindOperator(model, *unpack_following_op));
    }
  }
}

void RemoveUnpackOperator(const Operator& unpack_op, Model* model) {
  for (const string& output_array_name : unpack_op.outputs) {
    DeleteArrayIfUnused(output_array_name, model);
  }
  model->operators.erase(FindOperator(model, unpack_op));
}

void RemoveUnidirectionalSequenceLstmOps(std::stack<Operator*> uni_lstm_ops,
                                         Model* model) {
  while (!uni_lstm_ops.empty()) {
    Operator* uni_lstm_op = uni_lstm_ops.top();
    DeleteArrayIfUnused(uni_lstm_op->outputs[0], model);
    model->operators.erase(FindOperator(model, *uni_lstm_op));
    uni_lstm_ops.pop();
  }
}

}  // namespace

// TODO(renjieliu): Support graph generated by dynamic rnn as well.
::tensorflow::Status GroupBidirectionalSequenceLstm::Run(Model* model,
                                                         std::size_t op_index,
                                                         bool* modified) {
  *modified = false;
  // Bidirectional sequence lstm will generate two separate unidirectional
  // sequence lstm ops, for static bidirectional sequence lstm, there will be
  // a concatenation op at very end; for dynamic bidirectional squence lstm,
  // it is not guaranteed, but currently we do not support that.
  auto op_it = model->operators.begin() + op_index;
  Operator* final_concat_op = op_it->get();
  if (final_concat_op->type != OperatorType::kConcatenation &&
      final_concat_op->type != OperatorType::kConcat &&
      final_concat_op->type != OperatorType::kConcatV2) {
    return ::tensorflow::Status::OK();
  }

  // Match fw unidirectional lstm outputs and bw unidirectional lstm outputs:
  // should be two unstack ops.
  Operator *fw_lstm_output, *bw_lstm_output;
  if (!MatchTwoUnpackOps(*final_concat_op, *model, &fw_lstm_output,
                         &bw_lstm_output)) {
    return ::tensorflow::Status::OK();
  }

  // Find all upstream unidirectional lstm ops.
  std::stack<Operator*> fw_unidirectional_sequence_lstm_ops,
      bw_unidirectional_sequence_lstm_ops;
  Operator *first_fw_lstm_input, *first_bw_lstm_input;
  if (!FindUnidirectionalSequenceLstmOp(*model, *fw_lstm_output,
                                        &fw_unidirectional_sequence_lstm_ops,
                                        &first_fw_lstm_input) ||
      !FindUnidirectionalSequenceLstmOp(*model, *bw_lstm_output,
                                        &bw_unidirectional_sequence_lstm_ops,
                                        &first_bw_lstm_input)) {
    return ::tensorflow::Status::OK();
  }

  if (fw_unidirectional_sequence_lstm_ops.size() !=
          bw_unidirectional_sequence_lstm_ops.size() ||
      fw_unidirectional_sequence_lstm_ops.empty()) {
    return ::tensorflow::Status::OK();
  }

  // For static bidirectional sequence lstm, we should have two pack ops.
  if (first_fw_lstm_input->type != OperatorType::kPack ||
      first_bw_lstm_input->type != OperatorType::kPack) {
    return ::tensorflow::Status::OK();
  }

  // fw_lstm & bw_lstm should point to the same input, but reversed sequence.
  for (int i = 0; i < first_fw_lstm_input->inputs.size(); ++i) {
    if (first_fw_lstm_input->inputs[i] !=
        first_bw_lstm_input
            ->inputs[first_fw_lstm_input->inputs.size() - i - 1]) {
      return ::tensorflow::Status::OK();
    }
  }

  std::vector<BidirectionalSequenceLstmOperator*>
      bidirectional_sequence_lstm_ops;
  if (!GroupFwBwLstmOps(model, fw_unidirectional_sequence_lstm_ops,
                        bw_unidirectional_sequence_lstm_ops,
                        &bidirectional_sequence_lstm_ops)) {
    return ::tensorflow::Status::OK();
  }

  // Rewire the inputs & outputs.
  string current_input = first_fw_lstm_input->outputs[0];
  RewireBidirectionalSequenceLstmOpsConnections(
      current_input, bidirectional_sequence_lstm_ops, &op_it, model);

  // Insert a unpack op for the output.
  UnpackOperator* unpack_operator = new UnpackOperator;

  RewireFinalUnpackOutputs(
      static_cast<const UnpackOperator&>(*fw_lstm_output), &unpack_operator,
      &bidirectional_sequence_lstm_ops[bidirectional_sequence_lstm_ops.size() -
                                       1],
      model);
  model->operators.emplace(op_it, unpack_operator);

  // Delete unused ops.
  RemoveUnpackOperator(*fw_lstm_output, model);
  RemoveUnpackOperator(*bw_lstm_output, model);
  RemoveUnidirectionalSequenceLstmOps(fw_unidirectional_sequence_lstm_ops,
                                      model);
  RemoveUnidirectionalSequenceLstmOps(bw_unidirectional_sequence_lstm_ops,
                                      model);
  // Only keep the fw lstm's pack input.
  DeleteArrayIfUnused(first_bw_lstm_input->outputs[0], model);
  model->operators.erase(FindOperator(model, *first_bw_lstm_input));
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
