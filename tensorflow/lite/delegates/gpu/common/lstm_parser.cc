/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/lstm_parser.h"

#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/types/any.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder_helper.h"
#include "tensorflow/lite/delegates/gpu/common/object_reader.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/lstm_shared.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace gpu {
namespace {

Value* CreateNewSimilarValue(GraphFloat32* graph, const Value* old_value) {
  Value* new_value = graph->NewValue();
  new_value->quant_params = old_value->quant_params;
  new_value->tensor.shape = old_value->tensor.shape;
  new_value->tensor.type = old_value->tensor.type;
  new_value->tensor.ref = -1;
  return new_value;
}

absl::Status SetFullyConnectedWeights(int weights_tensor_id,
                                      ObjectReader* reader,
                                      FullyConnectedAttributes* attr) {
  Tensor<HW, DataType::FLOAT32> weights;
  RETURN_IF_ERROR(reader->ReadTensor(weights_tensor_id, &weights));
  attr->weights.data = std::move(weights.data);
  attr->weights.id = weights.id;
  attr->weights.shape.o = weights.shape.h;
  attr->weights.shape.h = 1;
  attr->weights.shape.w = 1;
  attr->weights.shape.i = weights.shape.w;
  return absl::OkStatus();
}

bool HasTensor(const TfLiteNode* node, const int index) {
  return (index < node->inputs->size) &&
         (node->inputs->data[index] != kTfLiteOptionalTensor);
}

bool HasCifg(const TfLiteNode* node) {
  return !HasTensor(
      node, tflite::ops::builtin::lstm::full::kInputToInputWeightsTensor);
}

bool HasPeephole(const TfLiteNode* node) {
  // Use forget weights to detect peephole instead of input weights as input
  // weights may be missing for cifg.
  return HasTensor(
      node, tflite::ops::builtin::lstm::full::kCellToForgetWeightsTensor);
}

bool HasNormalization(const TfLiteNode* node) {
  return HasTensor(
      node,
      tflite::ops::builtin::lstm::full::kForgetLayerNormCoefficientsTensor);
}

bool HasProjection(const TfLiteNode* node) {
  return HasTensor(node,
                   tflite::ops::builtin::lstm::full::kProjectionWeightsTensor);
}

// Builds subgraph for a single LSTM gate.
// Returns a Value representing the gate's output.
// High-level parameters:
//   - Has normalization (if true: provide normalization weights).
//   - Has peephole connection (if true: provide peephole weights).
//   - Which activation function to use.
// Note: no support for aux input.
//
// Implements the following:
// (*: matrix multiply, .*: elementwise multiply, +: elementwise add):
//   temp = input_weights * input_tensor + recurrent_weights * output_state;
//   if (peephole):
//     temp += peephole_weights .* cell_state;
//   if (layer normalization):
//     gate = activate(normalization_weights .* mean_stddev_norm(temp) + bias);
//   else:
//     gate = activate(temp + bias);
//
absl::Status BuildLstmGate(GraphFloat32* graph, ObjectReader* reader,
                           Value* output_state, Value* cell_state,
                           int input_weight_id, int recurrent_weight_id,
                           int cell_weight_id, int bias_id,
                           int normalization_weight_id,
                           const TfLiteFusedActivation activation,
                           bool has_peephole, bool has_normalization,
                           Value** gate_out) {
  Value* input_times_weights = CreateNewSimilarValue(graph, cell_state);
  {
    // #1 matrix multiplication: input_weights * input_tensor
    // If has no normalization, also adds bias.
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::FULLY_CONNECTED);
    FullyConnectedAttributes fc_attr;
    RETURN_IF_ERROR(
        SetFullyConnectedWeights(input_weight_id, reader, &fc_attr));
    if (!has_normalization) {
      RETURN_IF_ERROR(reader->ReadTensor(bias_id, &(fc_attr.bias)));
    }
    node->operation.attributes = std::move(fc_attr);
    RETURN_IF_ERROR(
        reader->AddInput(node, tflite::ops::builtin::lstm::full::kInputTensor));
    RETURN_IF_ERROR(graph->SetProducer(node->id, input_times_weights->id));
  }

  Value* output_state_times_weights = CreateNewSimilarValue(graph, cell_state);
  {
    // #2 matrix multiplication: recurrent_weights * output_state
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::FULLY_CONNECTED);
    FullyConnectedAttributes fc_attr;
    RETURN_IF_ERROR(
        SetFullyConnectedWeights(recurrent_weight_id, reader, &fc_attr));
    node->operation.attributes = std::move(fc_attr);
    RETURN_IF_ERROR(graph->AddConsumer(node->id, output_state->id));
    RETURN_IF_ERROR(
        graph->SetProducer(node->id, output_state_times_weights->id));
  }

  Value* cell_state_times_weights;
  if (has_peephole) {
    // #3 elementwise multiplication: cell_weight .* cell_state
    cell_state_times_weights = CreateNewSimilarValue(graph, cell_state);
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::MUL);
    ElementwiseAttributes attr;
    Tensor<Linear, DataType::FLOAT32> weights;
    RETURN_IF_ERROR(reader->ReadTensor(cell_weight_id, &weights));
    attr.param = std::move(weights);
    node->operation.attributes = std::move(attr);
    RETURN_IF_ERROR(graph->AddConsumer(node->id, cell_state->id));
    RETURN_IF_ERROR(graph->SetProducer(node->id, cell_state_times_weights->id));
  }

  Value* gate_before_normalization = CreateNewSimilarValue(graph, cell_state);
  Node* add_node = graph->NewNode();
  {
    // #4 elementwise addition: #1 + #2 + #3
    add_node->operation.type = ToString(OperationType::ADD);
    RETURN_IF_ERROR(graph->AddConsumer(add_node->id, input_times_weights->id));
    RETURN_IF_ERROR(
        graph->AddConsumer(add_node->id, output_state_times_weights->id));
    if (has_peephole) {
      RETURN_IF_ERROR(
          graph->AddConsumer(add_node->id, cell_state_times_weights->id));
    }
    RETURN_IF_ERROR(
        graph->SetProducer(add_node->id, gate_before_normalization->id));
  }

  if (!has_normalization) {
    // #5 Activation function: activate(temp + bias)
    // Bias is added in node #1.
    RETURN_IF_ERROR(MaybeFuseActivation(activation, graph, add_node));
    *gate_out = gate_before_normalization;
    return absl::OkStatus();
  }

  Value* normalized_gate =
      CreateNewSimilarValue(graph, gate_before_normalization);
  {
    // #6 Normalization: normalize(temp)
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::MEAN_STDDEV_NORMALIZATION);
    RETURN_IF_ERROR(
        graph->AddConsumer(node->id, gate_before_normalization->id));
    RETURN_IF_ERROR(graph->SetProducer(node->id, normalized_gate->id));
  }
  Value* reweighted_normalized_gate =
      CreateNewSimilarValue(graph, normalized_gate);
  {
    // #7 Elementwise multiplication: norm_weights .* #6
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::MUL);
    ElementwiseAttributes attr;
    Tensor<Linear, DataType::FLOAT32> norm_weights;
    RETURN_IF_ERROR(reader->ReadTensor(normalization_weight_id, &norm_weights));
    attr.param = std::move(norm_weights);
    node->operation.attributes = std::move(attr);
    RETURN_IF_ERROR(graph->AddConsumer(node->id, normalized_gate->id));
    RETURN_IF_ERROR(
        graph->SetProducer(node->id, reweighted_normalized_gate->id));
  }
  Value* gate = CreateNewSimilarValue(graph, reweighted_normalized_gate);
  {
    // #8 Elementwise add: #7 + bias
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::ADD);
    ElementwiseAttributes attr;
    Tensor<Linear, DataType::FLOAT32> bias;
    RETURN_IF_ERROR(reader->ReadTensor(bias_id, &bias));
    attr.param = std::move(bias);
    node->operation.attributes = std::move(attr);
    RETURN_IF_ERROR(
        graph->AddConsumer(node->id, reweighted_normalized_gate->id));
    RETURN_IF_ERROR(graph->SetProducer(node->id, gate->id));

    // #9: Activation function
    RETURN_IF_ERROR(MaybeFuseActivation(activation, graph, node));
  }
  *gate_out = gate;
  return absl::OkStatus();
}

// Builds subgraph for LSTM cell state update.
// Returns a Value representing the updated cell state.
// High-level parameters:
//  - clip: if > 0, clamp the resulting cell state to [-clip, +clip].
//
// Implements the following:
// (*: matrix multiply, .*: elementwise multiply, +: elementwise add):
//
//   cell_state_new = clip(forget_gate .* cell_state + input_gate .* cell_gate);
//
absl::Status BuildCellStateUpdate(GraphFloat32* graph, ObjectReader* reader,
                                  Value* forget_gate, Value* input_gate,
                                  Value* cell_gate, float cell_clip,
                                  Value** cell_state_new) {
  Value* cell_state;
  RETURN_IF_ERROR(reader->ReadValue(
      tflite::ops::builtin::lstm::full::kCellStateTensor, &cell_state));
  Value* cell_state_contrib = CreateNewSimilarValue(graph, cell_gate);
  {
    // #1 elementwise multiplication: forget_gate .* cell_state
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::MUL);
    RETURN_IF_ERROR(graph->AddConsumer(node->id, forget_gate->id));
    RETURN_IF_ERROR(graph->AddConsumer(node->id, cell_state->id));
    RETURN_IF_ERROR(graph->SetProducer(node->id, cell_state_contrib->id));
  }
  Value* cell_gate_contrib = CreateNewSimilarValue(graph, cell_gate);
  {
    // #2 elementwise multiplication: input_gate .* cell_gate
    // Note, with CIFG input_gate is equal to 1-forget_gate.
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::MUL);
    RETURN_IF_ERROR(graph->AddConsumer(node->id, input_gate->id));
    RETURN_IF_ERROR(graph->AddConsumer(node->id, cell_gate->id));
    RETURN_IF_ERROR(graph->SetProducer(node->id, cell_gate_contrib->id));
  }
  Value* new_cell_state = CreateNewSimilarValue(graph, cell_gate);
  {
    // #3 elementwise add: #1 + #2
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::ADD);
    RETURN_IF_ERROR(graph->AddConsumer(node->id, cell_state_contrib->id));
    RETURN_IF_ERROR(graph->AddConsumer(node->id, cell_gate_contrib->id));
    RETURN_IF_ERROR(graph->SetProducer(node->id, new_cell_state->id));
  }

  if (cell_clip <= 0.0f) {
    *cell_state_new = new_cell_state;
    return absl::OkStatus();
  }

  Value* max_clipped_state = CreateNewSimilarValue(graph, new_cell_state);
  {
    // #4 elementwise minimum: min(#3, clip)
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::MINIMUM);
    ElementwiseAttributes attr;
    attr.param = cell_clip;
    node->operation.attributes = std::move(attr);
    RETURN_IF_ERROR(graph->AddConsumer(node->id, new_cell_state->id));
    RETURN_IF_ERROR(graph->SetProducer(node->id, max_clipped_state->id));
  }
  Value* clipped_cell_state = CreateNewSimilarValue(graph, max_clipped_state);
  {
    // #5 elementwise maximum: max(#4, -clip)
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::MAXIMUM);
    ElementwiseAttributes attr;
    attr.param = -cell_clip;
    node->operation.attributes = std::move(attr);
    RETURN_IF_ERROR(graph->AddConsumer(node->id, max_clipped_state->id));
    RETURN_IF_ERROR(graph->SetProducer(node->id, clipped_cell_state->id));
  }
  *cell_state_new = clipped_cell_state;
  return absl::OkStatus();
}

// Build subgraph for LSTM output state update.
// Returns value representing the updated output state.
// High-level parameters:
//   - Has projection (if true, provide projection_weights).
//   - Has projection bias (only with projection).
//   - clip: clamp the projection output to [-clip, clip].
//   - Which activation function to use.
// Note the updated output state does not depend on the old output state
// directly, only through the output gate.
//
// Implements the following:
// (*: matrix multiply, .*: elementwise multiply, +: elementwise add):
//
//   temp = output_gate .* activate(cell_state);
//   if (projection):
//     output_state_new = clip(projection_weights * temp + projection_bias);
//   else:
//     output_state_new = temp;
//
absl::Status BuildOutputStateUpdate(GraphFloat32* graph, ObjectReader* reader,
                                    Value* output_state, Value* output_gate,
                                    Value* cell_state,
                                    TfLiteFusedActivation activation,
                                    bool has_projection, float proj_clip,
                                    Value** output_state_new) {
  Value* activated_state = CreateNewSimilarValue(graph, cell_state);
  {
    // #1 activation: activate(cell_state)
    Node* node = graph->NewNode();
    switch (activation) {
      case kTfLiteActTanh:
        node->operation.type = ToString(OperationType::TANH);
        break;
      case kTfLiteActSigmoid:
        node->operation.type = ToString(OperationType::SIGMOID);
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported activation: ", activation));
    }
    RETURN_IF_ERROR(graph->AddConsumer(node->id, cell_state->id));
    RETURN_IF_ERROR(graph->SetProducer(node->id, activated_state->id));
  }

  Value* new_output_state = CreateNewSimilarValue(graph, cell_state);
  {
    // #2 elementwise multiplication: output_gate .* #1
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::MUL);
    RETURN_IF_ERROR(graph->AddConsumer(node->id, activated_state->id));
    RETURN_IF_ERROR(graph->AddConsumer(node->id, output_gate->id));
    RETURN_IF_ERROR(graph->SetProducer(node->id, new_output_state->id));
  }

  if (!has_projection) {
    *output_state_new = new_output_state;
    return absl::OkStatus();
  }

  Value* projected_output_state = CreateNewSimilarValue(graph, output_state);
  {
    // #3 matrix multiplication: projection_weights * #2 + projection_bias
    Node* node = graph->NewNode();
    FullyConnectedAttributes fc_attr;
    RETURN_IF_ERROR(SetFullyConnectedWeights(
        tflite::ops::builtin::lstm::full::kProjectionWeightsTensor, reader,
        &fc_attr));
    // Projection bias is optional
    reader
        ->ReadTensor(tflite::ops::builtin::lstm::full::kProjectionBiasTensor,
                     &(fc_attr.bias))
        .IgnoreError();
    node->operation.attributes = std::move(fc_attr);
    node->operation.type = ToString(OperationType::FULLY_CONNECTED);
    RETURN_IF_ERROR(graph->AddConsumer(node->id, new_output_state->id));
    RETURN_IF_ERROR(graph->SetProducer(node->id, projected_output_state->id));
  }

  if (proj_clip <= 0.0f) {
    *output_state_new = projected_output_state;
    return absl::OkStatus();
  }

  Value* max_clipped_state =
      CreateNewSimilarValue(graph, projected_output_state);
  {
    // #4 elementwise minimum: min(#3, clip)
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::MINIMUM);
    ElementwiseAttributes attr;
    attr.param = proj_clip;
    node->operation.attributes = std::move(attr);
    RETURN_IF_ERROR(graph->AddConsumer(node->id, projected_output_state->id));
    RETURN_IF_ERROR(graph->SetProducer(node->id, max_clipped_state->id));
  }
  Value* clipped_output_state = CreateNewSimilarValue(graph, max_clipped_state);
  {
    // #5 elementwise maximum: max(#4, -clip)
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::MAXIMUM);
    ElementwiseAttributes attr;
    attr.param = -proj_clip;
    node->operation.attributes = std::move(attr);
    RETURN_IF_ERROR(graph->AddConsumer(node->id, max_clipped_state->id));
    RETURN_IF_ERROR(graph->SetProducer(node->id, clipped_output_state->id));
  }
  *output_state_new = clipped_output_state;
  return absl::OkStatus();
}

}  // namespace

// Build subgraph for a single LSTM OP.
// Returns a mapping for the used variable tensors' updated Values.
//
// High-level parameters:
//   - Has CIFG:
//       If false, calculate input_gate regularly.
//       If true, calculate input_gate to 1-forget_gate.
//   - Has peephole: see BuildLstmGate. Applies to all gates.
//   - Has normalization: see BuildLstmGate. Applies to all gates.
//   - Has projection, projection_bias, proj_clip: see BuildOutputStateUpdate
//   - Which activation to use:
//       Applies to only cell gate and output state update.
//       Other gates always use Sigmoid.
//
absl::Status ParseLSTMAttributes(
    const TfLiteNode* tflite_node, const TfLiteRegistration* registration,
    GraphFloat32* graph, ObjectReader* reader, const TfLiteLSTMParams* params,
    absl::flat_hash_map<int, ValueId>* new_variable_input_values) {
  const bool has_cifg = HasCifg(tflite_node);
  const bool has_peephole = HasPeephole(tflite_node);
  const bool has_normalization = HasNormalization(tflite_node);
  const bool has_projection = HasProjection(tflite_node);

  Value* old_cell_state;
  RETURN_IF_ERROR(reader->ReadValue(
      tflite::ops::builtin::lstm::full::kCellStateTensor, &old_cell_state));

  if (old_cell_state->tensor.shape.b != 1) {
    return absl::InvalidArgumentError(
        "Batched execution is not supported for LSTM");
  }

  Value* old_output_state;
  RETURN_IF_ERROR(reader->ReadValue(
      tflite::ops::builtin::lstm::full::kOutputStateTensor, &old_output_state));

  Value* forget_gate;
  RETURN_IF_ERROR(BuildLstmGate(
      graph, reader, old_output_state, old_cell_state,
      tflite::ops::builtin::lstm::full::kInputToForgetWeightsTensor,
      tflite::ops::builtin::lstm::full::kRecurrentToForgetWeightsTensor,
      tflite::ops::builtin::lstm::full::kCellToForgetWeightsTensor,
      tflite::ops::builtin::lstm::full::kForgetGateBiasTensor,
      tflite::ops::builtin::lstm::full::kForgetLayerNormCoefficientsTensor,
      kTfLiteActSigmoid, has_peephole, has_normalization, &forget_gate));

  Value* input_gate;
  if (has_cifg) {
    // When using cifg, input_gate is computed as (1 - forget_gate).
    Node* node = graph->NewNode();
    input_gate = CreateNewSimilarValue(graph, forget_gate);

    node->operation.type = ToString(OperationType::SUB);
    ElementwiseAttributes attr;
    attr.param = 1.0f;
    attr.runtime_tensor_is_second = true;
    node->operation.attributes = std::move(attr);
    RETURN_IF_ERROR(graph->AddConsumer(node->id, forget_gate->id));
    RETURN_IF_ERROR(graph->SetProducer(node->id, input_gate->id));
  } else {
    RETURN_IF_ERROR(BuildLstmGate(
        graph, reader, old_output_state, old_cell_state,
        tflite::ops::builtin::lstm::full::kInputToInputWeightsTensor,
        tflite::ops::builtin::lstm::full::kRecurrentToInputWeightsTensor,
        tflite::ops::builtin::lstm::full::kCellToInputWeightsTensor,
        tflite::ops::builtin::lstm::full::kInputGateBiasTensor,
        tflite::ops::builtin::lstm::full::kInputLayerNormCoefficientsTensor,
        kTfLiteActSigmoid, has_peephole, has_normalization, &input_gate));
  }

  // Cell state will not have peephole connections to itself
  Value* cell_gate;
  RETURN_IF_ERROR(BuildLstmGate(
      graph, reader, old_output_state, old_cell_state,
      tflite::ops::builtin::lstm::full::kInputToCellWeightsTensor,
      tflite::ops::builtin::lstm::full::kRecurrentToCellWeightsTensor,
      /*cell_weight_id=*/-1,
      tflite::ops::builtin::lstm::full::kCellGateBiasTensor,
      tflite::ops::builtin::lstm::full::kCellLayerNormCoefficientsTensor,
      params->activation, /*has_peephole=*/false, has_normalization,
      &cell_gate));

  Value* new_cell_state;
  RETURN_IF_ERROR(BuildCellStateUpdate(graph, reader, forget_gate, input_gate,
                                       cell_gate, params->cell_clip,
                                       &new_cell_state));

  Value* output_gate;
  RETURN_IF_ERROR(BuildLstmGate(
      graph, reader, old_output_state, new_cell_state,
      tflite::ops::builtin::lstm::full::kInputToOutputWeightsTensor,
      tflite::ops::builtin::lstm::full::kRecurrentToOutputWeightsTensor,
      tflite::ops::builtin::lstm::full::kCellToOutputWeightsTensor,
      tflite::ops::builtin::lstm::full::kOutputGateBiasTensor,
      tflite::ops::builtin::lstm::full::kOutputLayerNormCoefficientsTensor,
      kTfLiteActSigmoid, has_peephole, has_normalization, &output_gate));

  Value* new_output_state;
  RETURN_IF_ERROR(BuildOutputStateUpdate(graph, reader, old_output_state,
                                         output_gate, new_cell_state,
                                         params->activation, has_projection,
                                         params->proj_clip, &new_output_state));

  {
    // Copy updated output state to output.
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::COPY);
    RETURN_IF_ERROR(graph->AddConsumer(node->id, new_output_state->id));
    RETURN_IF_ERROR(reader->AddOutput(
        node, tflite::ops::builtin::lstm::full::kOutputTensor));
  }

  new_variable_input_values->clear();
  new_variable_input_values->emplace(
      tflite::ops::builtin::lstm::full::kCellStateTensor, new_cell_state->id);
  new_variable_input_values->emplace(
      tflite::ops::builtin::lstm::full::kOutputStateTensor,
      new_output_state->id);
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
