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

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/contrib/lite/c/builtin_op_data.h"
#include "tensorflow/contrib/lite/c/c_api_internal.h"
#include "tensorflow/contrib/lite/kernels/activation_functor.h"
#include "tensorflow/contrib/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace bidirectional_sequence_lstm {

// Input Tensors of size {max_time, n_batch, n_input}
constexpr int kInputTensor = 0;

// Forward LSTM cell tensors.
// Input weight tensors of size: {n_cell, n_input}
constexpr int kFwInputToInputWeightsTensor = 1;  // Optional
constexpr int kFwInputToForgetWeightsTensor = 2;
constexpr int kFwInputToCellWeightsTensor = 3;
constexpr int kFwInputToOutputWeightsTensor = 4;

// Recurrent weight tensors of size {n_cell, n_output}
constexpr int kFwRecurrentToInputWeightsTensor = 5;  // Optional
constexpr int kFwRecurrentToForgetWeightsTensor = 6;
constexpr int kFwRecurrentToCellWeightsTensor = 7;
constexpr int kFwRecurrentToOutputWeightsTensor = 8;

// Peephole weights tensors of size {n_cell}, representing a diagonal matrix.
constexpr int kFwCellToInputWeightsTensor = 9;    // Optional
constexpr int kFwCellToForgetWeightsTensor = 10;  // Optional
constexpr int kFwCellToOutputWeightsTensor = 11;  // Optional

// Gates bias tensors of size {n_cell}
constexpr int kFwInputGateBiasTensor = 12;  // Optional
constexpr int kFwForgetGateBiasTensor = 13;
constexpr int kFwCellGateBiasTensor = 14;
constexpr int kFwOutputGateBiasTensor = 15;

// Projection weight tensor of size {n_output, n_cell}
constexpr int kFwProjectionWeightsTensor = 16;  // Optional
// Projection bias tensor of size {n_output}
constexpr int kFwProjectionBiasTensor = 17;  // Optional

// Backward LSTM cell tensors.
// Input weight tensors of size: {n_cell, n_input}
constexpr int kBwInputToInputWeightsTensor = 18;  // Optional
constexpr int kBwInputToForgetWeightsTensor = 19;
constexpr int kBwInputToCellWeightsTensor = 20;
constexpr int kBwInputToOutputWeightsTensor = 21;

// Recurrent weight tensors of size {n_cell, n_output}
constexpr int kBwRecurrentToInputWeightsTensor = 22;  // Optional
constexpr int kBwRecurrentToForgetWeightsTensor = 23;
constexpr int kBwRecurrentToCellWeightsTensor = 24;
constexpr int kBwRecurrentToOutputWeightsTensor = 25;

// Peephole weights tensors of size {n_cell}, representing a diagonal matrix.
constexpr int kBwCellToInputWeightsTensor = 26;   // Optional
constexpr int kBwCellToForgetWeightsTensor = 27;  // Optional
constexpr int kBwCellToOutputWeightsTensor = 28;  // Optional

// Gates bias tensors of size {n_cell}
constexpr int kBwInputGateBiasTensor = 29;  // Optional
constexpr int kBwForgetGateBiasTensor = 30;
constexpr int kBwCellGateBiasTensor = 31;
constexpr int kBwOutputGateBiasTensor = 32;

// Projection weight tensor of size {n_output, n_cell}
constexpr int kBwProjectionWeightsTensor = 33;  // Optional
// Projection bias tensor of size {n_output}
constexpr int kBwProjectionBiasTensor = 34;  // Optional

// Stateful input tensors that are variables and will be modified by the Op.
// Activation state tensors of size {n_batch, n_output}
constexpr int kFwInputActivationStateTensor = 35;
// Cell state tensors of size {n_batch, n_cell}
constexpr int kFwInputCellStateTensor = 36;
// Activation state tensors of size {n_batch, n_output}
constexpr int kBwInputActivationStateTensor = 37;
// Cell state tensors of size {n_batch, n_cell}
constexpr int kBwInputCellStateTensor = 38;

// Auxiliary input and weights when stacking.
constexpr int kAuxInputTensor = 39;  // Optional
// Forward weights.
constexpr int kFwAuxInputToInputWeightsTensor = 40;   // Optional
constexpr int kFwAuxInputToForgetWeightsTensor = 41;  // Optional
constexpr int kFwAuxInputToCellWeightsTensor = 42;    // Optional
constexpr int kFwAuxInputToOutputWeightsTensor = 43;  // Optional
// Backward weights.
constexpr int kBwAuxInputToInputWeightsTensor = 44;   // Optional
constexpr int kBwAuxInputToForgetWeightsTensor = 45;  // Optional
constexpr int kBwAuxInputToCellWeightsTensor = 46;    // Optional
constexpr int kBwAuxInputToOutputWeightsTensor = 47;  // Optional

// Output tensors.
constexpr int kFwOutputTensor = 0;
constexpr int kBwOutputTensor = 1;

// Temporary tensors.
enum TemporaryTensor {
  // Scratch buffers for input, forget, etc. gates
  kFwScratchBuffer = 0,
  kBwScratchBuffer = 1,
  // Quantized tensors needed for the hybrid kernel.
  kInputQuantized = 2,
  kAuxInputQuantized = 3,  // Quantized tensor needed for auxiliary input.
  kFwActivationStateQuantized = 4,
  kBwActivationStateQuantized = 5,
  kFwCellStateQuantized = 6,
  kBwCellStateQuantized = 7,
  kScalingFactors = 8,
  kProductScalingFactors = 9,
  kRecoveredCellWeights = 10,
  kNumTemporaryTensors = 11
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* scratch_tensor_index = new int;
  context->AddTensors(context, kNumTemporaryTensors, scratch_tensor_index);
  return scratch_tensor_index;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<int*>(buffer);
}

// Check that input tensor dimensions matches with each other.
TfLiteStatus CheckLstmTensorDimensions(
    TfLiteContext* context, TfLiteNode* node, int n_input, int n_output,
    int n_cell, int input_to_input_weights_tensor,
    int input_to_forget_weights_tensor, int input_to_cell_weights_tensor,
    int input_to_output_weights_tensor, int recurrent_to_input_weights_tensor,
    int recurrent_to_forget_weights_tensor,
    int recurrent_to_cell_weights_tensor,
    int recurrent_to_output_weights_tensor, int cell_to_input_weights_tensor,
    int cell_to_forget_weights_tensor, int cell_to_output_weights_tensor,
    int input_gate_bias_tensor, int forget_gate_bias_tensor,
    int cell_gate_bias_tensor, int output_gate_bias_tensor,
    int projection_weights_tensor, int projection_bias_tensor) {
  const auto* params = reinterpret_cast<TfLiteLSTMParams*>(node->builtin_data);

  // Making sure clipping parameters have valid values.
  // == 0 means no clipping
  //  > 0 means clipping
  TF_LITE_ENSURE(context, params->cell_clip >= 0);
  TF_LITE_ENSURE(context, params->proj_clip >= 0);

  const TfLiteTensor* input_to_input_weights =
      GetOptionalInputTensor(context, node, input_to_input_weights_tensor);
  if (input_to_input_weights) {
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->data[1], n_input);
  }

  const TfLiteTensor* input_to_forget_weights =
      GetInput(context, node, input_to_forget_weights_tensor);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->data[1], n_input);

  const TfLiteTensor* input_to_cell_weights =
      GetInput(context, node, input_to_cell_weights_tensor);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->data[1], n_input);

  const TfLiteTensor* recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, recurrent_to_input_weights_tensor);
  if (recurrent_to_input_weights) {
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->data[0],
                      n_cell);
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->data[1],
                      n_output);
  }

  const TfLiteTensor* recurrent_to_forget_weights =
      GetInput(context, node, recurrent_to_forget_weights_tensor);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->data[0],
                    n_cell);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->data[1],
                    n_output);

  const TfLiteTensor* recurrent_to_cell_weights =
      GetInput(context, node, recurrent_to_cell_weights_tensor);
  TF_LITE_ENSURE_EQ(context, recurrent_to_cell_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_cell_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, recurrent_to_cell_weights->dims->data[1],
                    n_output);

  // We make sure the input-gate's parameters are either both present (regular
  // LSTM) or not at all (CIFG-LSTM).
  const bool cifg_weights_all_or_none =
      ((input_to_input_weights != nullptr) &&
       (recurrent_to_input_weights != nullptr)) ||
      ((input_to_input_weights == nullptr) &&
       (recurrent_to_input_weights == nullptr));
  TF_LITE_ENSURE(context, cifg_weights_all_or_none == true);

  const TfLiteTensor* cell_to_input_weights =
      GetOptionalInputTensor(context, node, cell_to_input_weights_tensor);
  if (cell_to_input_weights) {
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->data[0], n_cell);
  }

  const TfLiteTensor* cell_to_forget_weights =
      GetOptionalInputTensor(context, node, cell_to_forget_weights_tensor);
  if (cell_to_forget_weights) {
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->data[0], n_cell);
  }

  const TfLiteTensor* cell_to_output_weights =
      GetOptionalInputTensor(context, node, cell_to_output_weights_tensor);
  if (cell_to_output_weights) {
    TF_LITE_ENSURE_EQ(context, cell_to_output_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_output_weights->dims->data[0], n_cell);
  }

  // Making sure the peephole weights are there all or none.
  const bool use_cifg = (input_to_input_weights == nullptr);
  const bool peephole_weights_all_or_none =
      ((cell_to_input_weights != nullptr || use_cifg) &&
       (cell_to_forget_weights != nullptr) &&
       (cell_to_output_weights != nullptr)) ||
      ((cell_to_input_weights == nullptr) &&
       (cell_to_forget_weights == nullptr) &&
       (cell_to_output_weights == nullptr));
  TF_LITE_ENSURE(context, peephole_weights_all_or_none == true);

  // Make sure the input gate bias is present only when not a CIFG-LSTM.
  const TfLiteTensor* input_gate_bias =
      GetOptionalInputTensor(context, node, input_gate_bias_tensor);
  if (use_cifg) {
    TF_LITE_ENSURE_EQ(context, input_gate_bias, nullptr);
  } else {
    TF_LITE_ENSURE_EQ(context, input_gate_bias->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, input_gate_bias->dims->data[0], n_cell);
  }

  const TfLiteTensor* forget_gate_bias =
      GetInput(context, node, forget_gate_bias_tensor);
  TF_LITE_ENSURE_EQ(context, forget_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, forget_gate_bias->dims->data[0], n_cell);

  const TfLiteTensor* cell_bias =
      GetInput(context, node, cell_gate_bias_tensor);
  TF_LITE_ENSURE_EQ(context, cell_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, cell_bias->dims->data[0], n_cell);

  const TfLiteTensor* output_gate_bias =
      GetInput(context, node, output_gate_bias_tensor);
  TF_LITE_ENSURE_EQ(context, output_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, output_gate_bias->dims->data[0], n_cell);

  const TfLiteTensor* projection_weights =
      GetOptionalInputTensor(context, node, projection_weights_tensor);
  if (projection_weights) {
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[0], n_output);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[1], n_cell);
  }

  const TfLiteTensor* projection_bias =
      GetOptionalInputTensor(context, node, projection_bias_tensor);
  if (projection_bias) {
    TF_LITE_ENSURE_EQ(context, projection_bias->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, projection_bias->dims->data[0], n_output);
  }

  // Making sure the projection tensors are consistent:
  // 1) If projection weight is not present, then projection bias should not be
  // present.
  // 2) If projection weight is present, then projection bias is optional.
  // TODO(ghodrat): make sure this is correct.
  const bool projecton_tensors_consistent =
      ((projection_weights != nullptr) || (projection_bias == nullptr));
  TF_LITE_ENSURE(context, projecton_tensors_consistent == true);

  return kTfLiteOk;
}

TfLiteStatus CheckInputTensorDimensions(TfLiteContext* context,
                                        TfLiteNode* node, int n_input,
                                        int n_output, int n_cell) {
  TF_LITE_ENSURE_OK(
      context,
      CheckLstmTensorDimensions(
          context, node, n_input, n_output, n_cell,
          kFwInputToInputWeightsTensor, kFwInputToForgetWeightsTensor,
          kFwInputToCellWeightsTensor, kFwInputToOutputWeightsTensor,
          kFwRecurrentToInputWeightsTensor, kFwRecurrentToForgetWeightsTensor,
          kFwRecurrentToCellWeightsTensor, kFwRecurrentToOutputWeightsTensor,
          kFwCellToInputWeightsTensor, kFwCellToForgetWeightsTensor,
          kFwCellToOutputWeightsTensor, kFwInputGateBiasTensor,
          kFwForgetGateBiasTensor, kFwCellGateBiasTensor,
          kFwOutputGateBiasTensor, kFwProjectionWeightsTensor,
          kFwProjectionBiasTensor));

  TF_LITE_ENSURE_OK(
      context,
      CheckLstmTensorDimensions(
          context, node, n_input, n_output, n_cell,
          kBwInputToInputWeightsTensor, kBwInputToForgetWeightsTensor,
          kBwInputToCellWeightsTensor, kBwInputToOutputWeightsTensor,
          kBwRecurrentToInputWeightsTensor, kBwRecurrentToForgetWeightsTensor,
          kBwRecurrentToCellWeightsTensor, kBwRecurrentToOutputWeightsTensor,
          kBwCellToInputWeightsTensor, kBwCellToForgetWeightsTensor,
          kBwCellToOutputWeightsTensor, kBwInputGateBiasTensor,
          kBwForgetGateBiasTensor, kBwCellGateBiasTensor,
          kBwOutputGateBiasTensor, kBwProjectionWeightsTensor,
          kBwProjectionBiasTensor));

  // Check if Forward and Backward tensors match along required dimensions.
  return kTfLiteOk;
}

// Resize the output and scratch tensors based on the sizes of the input
// tensors. Also check that the size of the input tensors match each other.
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  int* scratch_tensor_index = reinterpret_cast<int*>(node->user_data);

  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 48);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 2);

  // Inferring batch size, number of outputs and sequence length and
  // number of cells from the input tensors.
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, input->dims->size, 3);
  const int max_time = input->dims->data[0];
  const int n_batch = input->dims->data[1];
  const int n_input = input->dims->data[2];

  const TfLiteTensor* fw_input_to_output_weights =
      GetInput(context, node, kFwInputToOutputWeightsTensor);
  const int n_fw_cell = fw_input_to_output_weights->dims->data[0];
  TF_LITE_ENSURE_EQ(context, fw_input_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, fw_input_to_output_weights->dims->data[1],
                    n_input);

  const TfLiteTensor* fw_recurrent_to_output_weights =
      GetInput(context, node, kFwRecurrentToOutputWeightsTensor);
  TF_LITE_ENSURE_EQ(context, fw_recurrent_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, fw_recurrent_to_output_weights->dims->data[0],
                    n_fw_cell);
  const int n_fw_output = fw_recurrent_to_output_weights->dims->data[1];

  // Check that input tensor dimensions matches with each other.
  TF_LITE_ENSURE_OK(
      context, CheckInputTensorDimensions(context, node, n_input, n_fw_output,
                                          n_fw_cell));

  // Get (optional) auxiliary inputs and weights.
  const TfLiteTensor* aux_input =
      GetOptionalInputTensor(context, node, kAuxInputTensor);
  const TfLiteTensor* fw_aux_input_to_input_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToInputWeightsTensor);
  const TfLiteTensor* fw_aux_input_to_forget_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToForgetWeightsTensor);
  const TfLiteTensor* fw_aux_input_to_cell_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToCellWeightsTensor);
  const TfLiteTensor* fw_aux_input_to_output_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToOutputWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_input_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToInputWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_forget_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToForgetWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_cell_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToCellWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_output_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToOutputWeightsTensor);

  const bool aux_inputs_all_or_none =
      ((aux_input != nullptr) && (fw_aux_input_to_cell_weights != nullptr) &&
       (fw_aux_input_to_forget_weights != nullptr) &&
       (fw_aux_input_to_output_weights != nullptr) &&
       (bw_aux_input_to_cell_weights != nullptr) &&
       (bw_aux_input_to_forget_weights != nullptr) &&
       (bw_aux_input_to_output_weights != nullptr)) ||
      ((fw_aux_input_to_cell_weights == nullptr) &&
       (fw_aux_input_to_forget_weights == nullptr) &&
       (fw_aux_input_to_output_weights == nullptr) &&
       (bw_aux_input_to_cell_weights == nullptr) &&
       (bw_aux_input_to_forget_weights == nullptr) &&
       (bw_aux_input_to_output_weights == nullptr));
  TF_LITE_ENSURE(context, aux_inputs_all_or_none);
  const bool has_aux_input = (aux_input != nullptr);

  if (has_aux_input) {
    // Check that aux_input has the same dimensions (except last) as the input.
    TF_LITE_ASSERT_EQ(aux_input->dims->data[0], input->dims->data[0]);
    TF_LITE_ASSERT_EQ(aux_input->dims->data[1], input->dims->data[1]);
  }

  // Get the pointer to output, activation_state and cell_state buffer tensors.
  TfLiteTensor* fw_output = GetOutput(context, node, kFwOutputTensor);
  TfLiteTensor* fw_activation_state =
      GetVariableInput(context, node, kFwInputActivationStateTensor);
  TfLiteTensor* fw_cell_state =
      GetVariableInput(context, node, kFwInputCellStateTensor);

  // Check the shape of input state tensors.
  // These tensor may be 1D or 2D. It's fine as long as the total size is
  // correct.
  TF_LITE_ENSURE_EQ(context, NumElements(fw_activation_state),
                    n_batch * n_fw_output);
  TF_LITE_ENSURE_EQ(context, NumElements(fw_cell_state), n_batch * n_fw_cell);

  // Resize the output tensors.
  TfLiteIntArray* fw_output_size = TfLiteIntArrayCreate(3);
  fw_output_size->data[0] = max_time;
  fw_output_size->data[1] = n_batch;
  fw_output_size->data[2] = n_fw_output;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, fw_output, fw_output_size));

  // The weights are of consistent type, so it suffices to check one.
  const bool is_hybrid_op = (fw_input_to_output_weights->type == kTfLiteUInt8);

  TfLiteIntArrayFree(node->temporaries);
  if (is_hybrid_op) {
    node->temporaries = TfLiteIntArrayCreate(kNumTemporaryTensors);
  } else {
    node->temporaries = TfLiteIntArrayCreate(2);  // the two scratch buffers.
  }
  // Create a scratch buffer tensor.
  node->temporaries->data[kFwScratchBuffer] = *scratch_tensor_index;
  TfLiteTensor* fw_scratch_buffer =
      GetTemporary(context, node, kFwScratchBuffer);
  fw_scratch_buffer->type = input->type;
  fw_scratch_buffer->allocation_type = kTfLiteArenaRw;

  const TfLiteTensor* fw_input_to_input_weights =
      GetOptionalInputTensor(context, node, kFwInputToInputWeightsTensor);
  if (has_aux_input) {
    TF_LITE_ENSURE_EQ(context, fw_aux_input_to_input_weights->dims->data[0],
                      fw_input_to_input_weights->dims->data[0]);
  }
  const bool fw_use_cifg = (fw_input_to_input_weights == nullptr);
  TfLiteIntArray* fw_scratch_buffer_size = TfLiteIntArrayCreate(2);
  fw_scratch_buffer_size->data[0] = n_batch;
  if (fw_use_cifg) {
    // Reserving space for Cell, Forget, Output gates
    fw_scratch_buffer_size->data[1] = n_fw_cell * 3;
  } else {
    // Reserving space for Input, Cell, Forget, Output gates
    fw_scratch_buffer_size->data[1] = n_fw_cell * 4;
  }
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, fw_scratch_buffer,
                                                   fw_scratch_buffer_size));
  // Same for the backward cell.
  const TfLiteTensor* bw_input_to_output_weights =
      GetInput(context, node, kBwInputToOutputWeightsTensor);
  const int n_bw_cell = bw_input_to_output_weights->dims->data[0];
  TF_LITE_ENSURE_EQ(context, bw_input_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, bw_input_to_output_weights->dims->data[1],
                    n_input);

  const TfLiteTensor* bw_recurrent_to_output_weights =
      GetInput(context, node, kBwRecurrentToOutputWeightsTensor);
  TF_LITE_ENSURE_EQ(context, bw_recurrent_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, bw_recurrent_to_output_weights->dims->data[0],
                    n_bw_cell);
  const int n_bw_output = bw_recurrent_to_output_weights->dims->data[1];

  // Check that input tensor dimensions matches with each other.
  TF_LITE_ENSURE_OK(
      context, CheckInputTensorDimensions(context, node, n_input, n_bw_output,
                                          n_bw_cell));

  // Get the pointer to output, activation_state and cell_state buffer tensors.
  TfLiteTensor* bw_output = GetOutput(context, node, kBwOutputTensor);
  TfLiteTensor* bw_activation_state =
      GetVariableInput(context, node, kBwInputActivationStateTensor);
  TfLiteTensor* bw_cell_state =
      GetVariableInput(context, node, kBwInputCellStateTensor);

  // Resize the output tensors.
  TfLiteIntArray* bw_output_size = TfLiteIntArrayCreate(3);
  bw_output_size->data[0] = max_time;
  bw_output_size->data[1] = n_batch;
  bw_output_size->data[2] = n_bw_output;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, bw_output, bw_output_size));

  // Check the shape of input state tensors.
  // These tensor may be 1D or 2D. It's fine as long as the total size is
  // correct.
  TF_LITE_ENSURE_EQ(context, NumElements(bw_activation_state),
                    n_batch * n_bw_output);
  TF_LITE_ENSURE_EQ(context, NumElements(bw_cell_state), n_batch * n_bw_cell);

  // Create a scratch buffer tensor.
  node->temporaries->data[kBwScratchBuffer] =
      *(scratch_tensor_index) + kBwScratchBuffer;
  TfLiteTensor* bw_scratch_buffer =
      GetTemporary(context, node, kBwScratchBuffer);
  bw_scratch_buffer->type = input->type;
  bw_scratch_buffer->allocation_type = kTfLiteArenaRw;

  const TfLiteTensor* bw_input_to_input_weights =
      GetOptionalInputTensor(context, node, kBwInputToInputWeightsTensor);
  if (has_aux_input) {
    TF_LITE_ENSURE_EQ(context, bw_aux_input_to_input_weights->dims->data[0],
                      bw_input_to_input_weights->dims->data[0]);
  }
  const bool bw_use_cifg = (bw_input_to_input_weights == nullptr);
  TfLiteIntArray* bw_scratch_buffer_size = TfLiteIntArrayCreate(2);
  bw_scratch_buffer_size->data[0] = n_batch;
  if (bw_use_cifg) {
    // Reserving space for Cell, Forget, Output gates
    bw_scratch_buffer_size->data[1] = n_bw_cell * 3;
  } else {
    // Reserving space for Input, Cell, Forget, Output gates
    bw_scratch_buffer_size->data[1] = n_bw_cell * 4;
  }
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, bw_scratch_buffer,
                                                   bw_scratch_buffer_size));
  if (is_hybrid_op) {
    // Allocate temporary tensors to store quantized values of input, aux_input
    // (if present), activation_state and cell_state tensors.
    node->temporaries->data[kInputQuantized] =
        *scratch_tensor_index + kInputQuantized;
    TfLiteTensor* input_quantized =
        GetTemporary(context, node, kInputQuantized);
    input_quantized->type = kTfLiteUInt8;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }

    if (has_aux_input) {
      node->temporaries->data[kAuxInputQuantized] =
          *scratch_tensor_index + kAuxInputQuantized;
      TfLiteTensor* aux_input_quantized =
          GetTemporary(context, node, kAuxInputQuantized);
      aux_input_quantized->type = kTfLiteUInt8;
      aux_input_quantized->allocation_type = kTfLiteArenaRw;
      if (!TfLiteIntArrayEqual(aux_input_quantized->dims, aux_input->dims)) {
        TfLiteIntArray* aux_input_quantized_size =
            TfLiteIntArrayCopy(aux_input->dims);
        TF_LITE_ENSURE_OK(context,
                          context->ResizeTensor(context, aux_input_quantized,
                                                aux_input_quantized_size));
      }
    }

    node->temporaries->data[kFwActivationStateQuantized] =
        *scratch_tensor_index + kFwActivationStateQuantized;
    TfLiteTensor* fw_activation_state_quantized =
        GetTemporary(context, node, kFwActivationStateQuantized);
    fw_activation_state_quantized->type = kTfLiteUInt8;
    fw_activation_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(fw_activation_state_quantized->dims,
                             fw_activation_state->dims)) {
      TfLiteIntArray* fw_activation_state_quantized_size =
          TfLiteIntArrayCopy(fw_activation_state->dims);
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, fw_activation_state_quantized,
                                         fw_activation_state_quantized_size));
    }
    node->temporaries->data[kBwActivationStateQuantized] =
        *scratch_tensor_index + kBwActivationStateQuantized;
    TfLiteTensor* bw_activation_state_quantized =
        GetTemporary(context, node, kBwActivationStateQuantized);
    bw_activation_state_quantized->type = kTfLiteUInt8;
    bw_activation_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(bw_activation_state_quantized->dims,
                             bw_activation_state->dims)) {
      TfLiteIntArray* bw_activation_state_quantized_size =
          TfLiteIntArrayCopy(bw_activation_state->dims);
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, bw_activation_state_quantized,
                                         bw_activation_state_quantized_size));
    }
    node->temporaries->data[kFwCellStateQuantized] =
        *scratch_tensor_index + kFwCellStateQuantized;
    TfLiteTensor* fw_cell_state_quantized =
        GetTemporary(context, node, kFwCellStateQuantized);
    fw_cell_state_quantized->type = kTfLiteUInt8;
    fw_cell_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(fw_cell_state_quantized->dims,
                             fw_cell_state->dims)) {
      TfLiteIntArray* fw_cell_state_quantized_size =
          TfLiteIntArrayCopy(fw_cell_state->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, fw_cell_state_quantized,
                                              fw_cell_state_quantized_size));
    }
    node->temporaries->data[kBwCellStateQuantized] =
        *scratch_tensor_index + kBwCellStateQuantized;
    TfLiteTensor* bw_cell_state_quantized =
        GetTemporary(context, node, kBwCellStateQuantized);
    bw_cell_state_quantized->type = kTfLiteUInt8;
    bw_cell_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(bw_cell_state_quantized->dims,
                             bw_cell_state->dims)) {
      TfLiteIntArray* bw_cell_state_quantized_size =
          TfLiteIntArrayCopy(bw_cell_state->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, bw_cell_state_quantized,
                                              bw_cell_state_quantized_size));
    }

    // Allocate temporary tensors to store scaling factors and product scaling
    // factors. The latter is a convenience storage which allows to quantize
    // a vector once (which produces the scaling factors) and multiply it with
    // different matrices (which requires multiplying the scaling factors with
    // the scaling factor of the matrix).
    node->temporaries->data[kScalingFactors] =
        *scratch_tensor_index + kScalingFactors;
    TfLiteTensor* scaling_factors =
        GetTemporary(context, node, kScalingFactors);
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;
    TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
    scaling_factors_size->data[0] = n_batch;
    if (!TfLiteIntArrayEqual(scaling_factors->dims, scaling_factors_size)) {
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }
    node->temporaries->data[kProductScalingFactors] =
        *scratch_tensor_index + kProductScalingFactors;
    TfLiteTensor* prod_scaling_factors =
        GetTemporary(context, node, kProductScalingFactors);
    prod_scaling_factors->type = kTfLiteFloat32;
    prod_scaling_factors->allocation_type = kTfLiteArenaRw;
    TfLiteIntArray* prod_scaling_factors_size = TfLiteIntArrayCreate(1);
    prod_scaling_factors_size->data[0] = n_batch;
    if (!TfLiteIntArrayEqual(prod_scaling_factors->dims,
                             prod_scaling_factors_size)) {
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, prod_scaling_factors,
                                              prod_scaling_factors_size));
    }

    // Allocate a temporary tensor to store the recovered cell weights. Since
    // this is used for diagonal matrices, only need to store n_cell values.
    node->temporaries->data[kRecoveredCellWeights] =
        *scratch_tensor_index + kRecoveredCellWeights;
    TfLiteTensor* recovered_cell_weights =
        GetTemporary(context, node, kRecoveredCellWeights);
    recovered_cell_weights->type = kTfLiteFloat32;
    recovered_cell_weights->allocation_type = kTfLiteArenaRw;
    TfLiteIntArray* recovered_cell_weights_size = TfLiteIntArrayCreate(1);
    recovered_cell_weights_size->data[0] = n_fw_cell;
    if (!TfLiteIntArrayEqual(recovered_cell_weights->dims,
                             recovered_cell_weights_size)) {
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, recovered_cell_weights,
                                              recovered_cell_weights_size));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalFloat(
    const TfLiteTensor* input, const TfLiteTensor* input_to_input_weights,
    const TfLiteTensor* input_to_forget_weights,
    const TfLiteTensor* input_to_cell_weights,
    const TfLiteTensor* input_to_output_weights,
    const TfLiteTensor* recurrent_to_input_weights,
    const TfLiteTensor* recurrent_to_forget_weights,
    const TfLiteTensor* recurrent_to_cell_weights,
    const TfLiteTensor* recurrent_to_output_weights,
    const TfLiteTensor* cell_to_input_weights,
    const TfLiteTensor* cell_to_forget_weights,
    const TfLiteTensor* cell_to_output_weights, const TfLiteTensor* aux_input,
    const TfLiteTensor* aux_input_to_input_weights,
    const TfLiteTensor* aux_input_to_forget_weights,
    const TfLiteTensor* aux_input_to_cell_weights,
    const TfLiteTensor* aux_input_to_output_weights,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    const TfLiteLSTMParams* params, bool forward_sequence,
    TfLiteTensor* scratch_buffer, TfLiteTensor* activation_state,
    TfLiteTensor* cell_state, TfLiteTensor* output) {
  const int max_time = input->dims->data[0];
  const int n_batch = input->dims->data[1];
  const int n_input = input->dims->data[2];
  const int aux_input_size = (aux_input) ? aux_input->dims->data[2] : 0;

  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Since we have already checked that weights are all there or none, we can
  // check the existense of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights == nullptr);
  const bool use_peephole = (cell_to_output_weights != nullptr);

  // Index the scratch buffers pointers to the global scratch buffer.
  float* input_gate_scratch = nullptr;
  float* cell_scratch = nullptr;
  float* forget_gate_scratch = nullptr;
  float* output_gate_scratch = nullptr;
  if (use_cifg) {
    cell_scratch = scratch_buffer->data.f;
    forget_gate_scratch = scratch_buffer->data.f + n_cell * n_batch;
    output_gate_scratch = scratch_buffer->data.f + 2 * n_cell * n_batch;
  } else {
    input_gate_scratch = scratch_buffer->data.f;
    cell_scratch = scratch_buffer->data.f + n_cell * n_batch;
    forget_gate_scratch = scratch_buffer->data.f + 2 * n_cell * n_batch;
    output_gate_scratch = scratch_buffer->data.f + 3 * n_cell * n_batch;
  }

  // Check optional tensors, the respective pointers can be null.
  const float* input_to_input_weights_ptr =
      (use_cifg) ? nullptr : input_to_input_weights->data.f;
  const float* recurrent_to_input_weights_ptr =
      (use_cifg) ? nullptr : recurrent_to_input_weights->data.f;
  const float* input_gate_bias_ptr =
      (use_cifg) ? nullptr : input_gate_bias->data.f;
  const float* cell_to_input_weights_ptr =
      (use_peephole && !use_cifg) ? cell_to_input_weights->data.f : nullptr;
  const float* cell_to_forget_weights_ptr =
      (use_peephole) ? cell_to_forget_weights->data.f : nullptr;
  const float* cell_to_output_weights_ptr =
      (use_peephole) ? cell_to_output_weights->data.f : nullptr;
  const float* projection_weights_ptr =
      (projection_weights == nullptr) ? nullptr : projection_weights->data.f;
  const float* projection_bias_ptr =
      (projection_bias == nullptr) ? nullptr : projection_bias->data.f;

  float* aux_input_ptr = nullptr;
  float* aux_input_to_input_weights_ptr = nullptr;
  float* aux_input_to_forget_weights_ptr = nullptr;
  float* aux_input_to_cell_weights_ptr = nullptr;
  float* aux_input_to_output_weights_ptr = nullptr;
  if (aux_input_size > 0) {
    aux_input_ptr = aux_input->data.f;
    aux_input_to_input_weights_ptr = aux_input_to_input_weights->data.f;
    aux_input_to_forget_weights_ptr = aux_input_to_forget_weights->data.f;
    aux_input_to_cell_weights_ptr = aux_input_to_cell_weights->data.f;
    aux_input_to_output_weights_ptr = aux_input_to_output_weights->data.f;
  }

  // Loop through the sequence.
  const int input_step = n_batch * n_input;
  const int output_step = n_batch * n_output;
  for (int t = 0; t < max_time; t++) {
    // If this is the forward_sequence, step forward, otherwise step backwards.
    const int t_rel = forward_sequence ? t : max_time - t - 1;
    const float* input_ptr = input->data.f + t_rel * input_step;
    float* output_ptr_time = output->data.f + t_rel * output_step;

    kernel_utils::LstmStepWithAuxInput(
        input_ptr, input_to_input_weights_ptr, input_to_forget_weights->data.f,
        input_to_cell_weights->data.f, input_to_output_weights->data.f,
        aux_input_ptr, aux_input_to_input_weights_ptr,
        aux_input_to_forget_weights_ptr, aux_input_to_cell_weights_ptr,
        aux_input_to_output_weights_ptr, recurrent_to_input_weights_ptr,
        recurrent_to_forget_weights->data.f, recurrent_to_cell_weights->data.f,
        recurrent_to_output_weights->data.f, cell_to_input_weights_ptr,
        cell_to_forget_weights_ptr, cell_to_output_weights_ptr,
        input_gate_bias_ptr, forget_gate_bias->data.f, cell_bias->data.f,
        output_gate_bias->data.f, projection_weights_ptr, projection_bias_ptr,
        params, n_batch, n_cell, n_input, aux_input_size, n_output,
        activation_state->data.f, cell_state->data.f, input_gate_scratch,
        forget_gate_scratch, cell_scratch, output_gate_scratch,
        output_ptr_time);
  }
  return kTfLiteOk;
}

TfLiteStatus EvalHybrid(
    const TfLiteTensor* input, const TfLiteTensor* input_to_input_weights,
    const TfLiteTensor* input_to_forget_weights,
    const TfLiteTensor* input_to_cell_weights,
    const TfLiteTensor* input_to_output_weights,
    const TfLiteTensor* recurrent_to_input_weights,
    const TfLiteTensor* recurrent_to_forget_weights,
    const TfLiteTensor* recurrent_to_cell_weights,
    const TfLiteTensor* recurrent_to_output_weights,
    const TfLiteTensor* cell_to_input_weights,
    const TfLiteTensor* cell_to_forget_weights,
    const TfLiteTensor* cell_to_output_weights, const TfLiteTensor* aux_input,
    const TfLiteTensor* aux_input_to_input_weights,
    const TfLiteTensor* aux_input_to_forget_weights,
    const TfLiteTensor* aux_input_to_cell_weights,
    const TfLiteTensor* aux_input_to_output_weights,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    const TfLiteLSTMParams* params, bool forward_sequence,
    TfLiteTensor* scratch_buffer, TfLiteTensor* scaling_factors,
    TfLiteTensor* prod_scaling_factors, TfLiteTensor* recovered_cell_weights,
    TfLiteTensor* input_quantized, TfLiteTensor* aux_input_quantized,
    TfLiteTensor* output_state_quantized, TfLiteTensor* cell_state_quantized,
    TfLiteTensor* output_state, TfLiteTensor* cell_state,
    TfLiteTensor* output) {
  const int max_time = input->dims->data[0];
  const int n_batch = input->dims->data[1];
  const int n_input = input->dims->data[2];
  const int aux_input_size = (aux_input) ? aux_input->dims->data[2] : 0;
  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to get the condition.
  const bool use_cifg = (input_to_input_weights == nullptr);
  const bool use_peephole = (cell_to_output_weights != nullptr);

  float* input_gate_scratch = nullptr;
  float* cell_scratch = nullptr;
  float* forget_gate_scratch = nullptr;
  float* output_gate_scratch = nullptr;
  if (use_cifg) {
    cell_scratch = scratch_buffer->data.f;
    forget_gate_scratch = scratch_buffer->data.f + n_cell * n_batch;
    output_gate_scratch = scratch_buffer->data.f + 2 * n_cell * n_batch;
  } else {
    input_gate_scratch = scratch_buffer->data.f;
    cell_scratch = scratch_buffer->data.f + n_cell * n_batch;
    forget_gate_scratch = scratch_buffer->data.f + 2 * n_cell * n_batch;
    output_gate_scratch = scratch_buffer->data.f + 3 * n_cell * n_batch;
  }

  // Check optional tensors, the respective pointers can be null.
  int8_t* input_to_input_weights_ptr = nullptr;
  float input_to_input_weights_scale = 1.0f;
  int8_t* recurrent_to_input_weights_ptr = nullptr;
  float recurrent_to_input_weights_scale = 1.0f;
  float* input_gate_bias_ptr = nullptr;
  if (!use_cifg) {
    input_to_input_weights_ptr =
        reinterpret_cast<int8_t*>(input_to_input_weights->data.uint8);
    recurrent_to_input_weights_ptr =
        reinterpret_cast<int8_t*>(recurrent_to_input_weights->data.uint8);
    input_gate_bias_ptr = input_gate_bias->data.f;
    input_to_input_weights_scale = input_to_input_weights->params.scale;
    recurrent_to_input_weights_scale = recurrent_to_input_weights->params.scale;
  }

  int8_t* cell_to_input_weights_ptr = nullptr;
  int8_t* cell_to_forget_weights_ptr = nullptr;
  int8_t* cell_to_output_weights_ptr = nullptr;
  float cell_to_input_weights_scale = 1.0f;
  float cell_to_forget_weights_scale = 1.0f;
  float cell_to_output_weights_scale = 1.0f;
  if (use_peephole) {
    if (!use_cifg) {
      cell_to_input_weights_ptr =
          reinterpret_cast<int8_t*>(cell_to_input_weights->data.uint8);
      cell_to_input_weights_scale = cell_to_input_weights->params.scale;
    }
    cell_to_forget_weights_ptr =
        reinterpret_cast<int8_t*>(cell_to_forget_weights->data.uint8);
    cell_to_output_weights_ptr =
        reinterpret_cast<int8_t*>(cell_to_output_weights->data.uint8);
    cell_to_forget_weights_scale = cell_to_forget_weights->params.scale;
    cell_to_output_weights_scale = cell_to_output_weights->params.scale;
  }

  const int8_t* projection_weights_ptr =
      (projection_weights == nullptr)
          ? nullptr
          : reinterpret_cast<int8_t*>(projection_weights->data.uint8);
  const float projection_weights_scale =
      (projection_weights == nullptr) ? 1.0f : projection_weights->params.scale;
  const float* projection_bias_ptr =
      (projection_bias == nullptr) ? nullptr : projection_bias->data.f;

  // Required tensors, pointers are non-null.
  const int8_t* input_to_forget_weights_ptr =
      reinterpret_cast<int8_t*>(input_to_forget_weights->data.uint8);
  const float input_to_forget_weights_scale =
      input_to_forget_weights->params.scale;
  const int8_t* input_to_cell_weights_ptr =
      reinterpret_cast<int8_t*>(input_to_cell_weights->data.uint8);
  const float input_to_cell_weights_scale = input_to_cell_weights->params.scale;
  const int8_t* input_to_output_weights_ptr =
      reinterpret_cast<int8_t*>(input_to_output_weights->data.uint8);
  const float input_to_output_weights_scale =
      input_to_output_weights->params.scale;
  const int8_t* recurrent_to_forget_weights_ptr =
      reinterpret_cast<int8_t*>(recurrent_to_forget_weights->data.uint8);
  const float recurrent_to_forget_weights_scale =
      recurrent_to_forget_weights->params.scale;
  const int8_t* recurrent_to_cell_weights_ptr =
      reinterpret_cast<int8_t*>(recurrent_to_cell_weights->data.uint8);
  const float recurrent_to_cell_weights_scale =
      recurrent_to_cell_weights->params.scale;
  const int8_t* recurrent_to_output_weights_ptr =
      reinterpret_cast<int8_t*>(recurrent_to_output_weights->data.uint8);
  const float recurrent_to_output_weights_scale =
      recurrent_to_output_weights->params.scale;
  const float* forget_gate_bias_ptr = forget_gate_bias->data.f;
  const float* cell_bias_ptr = cell_bias->data.f;
  const float* output_gate_bias_ptr = output_gate_bias->data.f;

  float* output_state_ptr = output_state->data.f;
  float* cell_state_ptr = cell_state->data.f;

  // Temporary storage for quantized values and scaling factors.
  int8_t* quantized_input_ptr =
      reinterpret_cast<int8_t*>(input_quantized->data.uint8);
  int8_t* quantized_aux_input_ptr =
      (aux_input_quantized == nullptr)
          ? nullptr
          : reinterpret_cast<int8_t*>(aux_input_quantized->data.uint8);
  int8_t* quantized_output_state_ptr =
      reinterpret_cast<int8_t*>(output_state_quantized->data.uint8);
  int8_t* quantized_cell_state_ptr =
      reinterpret_cast<int8_t*>(cell_state_quantized->data.uint8);
  float* scaling_factors_ptr = scaling_factors->data.f;
  float* prod_scaling_factors_ptr = prod_scaling_factors->data.f;
  float* recovered_cell_weights_ptr = recovered_cell_weights->data.f;

  // Auxiliary input and weights.
  float* aux_input_ptr = nullptr;
  int8_t* aux_input_to_input_weights_ptr = nullptr;
  int8_t* aux_input_to_forget_weights_ptr = nullptr;
  int8_t* aux_input_to_cell_weights_ptr = nullptr;
  int8_t* aux_input_to_output_weights_ptr = nullptr;
  float aux_input_to_input_weights_scale = 0.0f;
  float aux_input_to_forget_weights_scale = 0.0f;
  float aux_input_to_cell_weights_scale = 0.0f;
  float aux_input_to_output_weights_scale = 0.0f;
  if (aux_input_size > 0) {
    aux_input_ptr = aux_input->data.f;
    aux_input_to_input_weights_ptr =
        reinterpret_cast<int8_t*>(aux_input_to_input_weights->data.uint8);
    aux_input_to_forget_weights_ptr =
        reinterpret_cast<int8_t*>(aux_input_to_forget_weights->data.uint8);
    aux_input_to_cell_weights_ptr =
        reinterpret_cast<int8_t*>(aux_input_to_cell_weights->data.uint8);
    aux_input_to_output_weights_ptr =
        reinterpret_cast<int8_t*>(aux_input_to_output_weights->data.uint8);
    aux_input_to_input_weights_scale = aux_input_to_input_weights->params.scale;
    aux_input_to_forget_weights_scale =
        aux_input_to_forget_weights->params.scale;
    aux_input_to_cell_weights_scale = aux_input_to_cell_weights->params.scale;
    aux_input_to_output_weights_scale =
        aux_input_to_output_weights->params.scale;
  }

  // Feed the sequence into the LSTM step-by-step.
  const int input_step = n_batch * n_input;
  const int output_step = n_batch * n_output;
  for (int t = 0; t < max_time; t++) {
    // If this is the forward_sequence, step forward, otherwise step backwards.
    const int t_rel = forward_sequence ? t : max_time - t - 1;
    const float* input_ptr = input->data.f + t_rel * input_step;
    float* output_ptr = output->data.f + t_rel * output_step;

    kernel_utils::LstmStepWithAuxInput(
        input_ptr, input_to_input_weights_ptr, input_to_input_weights_scale,
        input_to_forget_weights_ptr, input_to_forget_weights_scale,
        input_to_cell_weights_ptr, input_to_cell_weights_scale,
        input_to_output_weights_ptr, input_to_output_weights_scale,
        aux_input_ptr, aux_input_to_input_weights_ptr,
        aux_input_to_input_weights_scale, aux_input_to_forget_weights_ptr,
        aux_input_to_forget_weights_scale, aux_input_to_cell_weights_ptr,
        aux_input_to_cell_weights_scale, aux_input_to_output_weights_ptr,
        aux_input_to_output_weights_scale, recurrent_to_input_weights_ptr,
        recurrent_to_input_weights_scale, recurrent_to_forget_weights_ptr,
        recurrent_to_forget_weights_scale, recurrent_to_cell_weights_ptr,
        recurrent_to_cell_weights_scale, recurrent_to_output_weights_ptr,
        recurrent_to_output_weights_scale, cell_to_input_weights_ptr,
        cell_to_input_weights_scale, cell_to_forget_weights_ptr,
        cell_to_forget_weights_scale, cell_to_output_weights_ptr,
        cell_to_output_weights_scale, input_gate_bias_ptr, forget_gate_bias_ptr,
        cell_bias_ptr, output_gate_bias_ptr, projection_weights_ptr,
        projection_weights_scale, projection_bias_ptr, params, n_batch, n_cell,
        n_input, aux_input_size, n_output, input_gate_scratch,
        forget_gate_scratch, cell_scratch, output_gate_scratch,
        scaling_factors_ptr, prod_scaling_factors_ptr,
        recovered_cell_weights_ptr, quantized_input_ptr,
        quantized_aux_input_ptr, quantized_output_state_ptr,
        quantized_cell_state_ptr, output_state_ptr, cell_state_ptr, output_ptr);
  }

  return kTfLiteOk;
}

// The LSTM Op engine.
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* params = reinterpret_cast<TfLiteLSTMParams*>(node->builtin_data);

  // Input tensor.
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);

  // Tensors for the forward cell.
  const TfLiteTensor* fw_input_to_input_weights =
      GetOptionalInputTensor(context, node, kFwInputToInputWeightsTensor);
  const TfLiteTensor* fw_input_to_forget_weights =
      GetInput(context, node, kFwInputToForgetWeightsTensor);
  const TfLiteTensor* fw_input_to_cell_weights =
      GetInput(context, node, kFwInputToCellWeightsTensor);
  const TfLiteTensor* fw_input_to_output_weights =
      GetInput(context, node, kFwInputToOutputWeightsTensor);

  const TfLiteTensor* fw_recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, kFwRecurrentToInputWeightsTensor);
  const TfLiteTensor* fw_recurrent_to_forget_weights =
      GetInput(context, node, kFwRecurrentToForgetWeightsTensor);
  const TfLiteTensor* fw_recurrent_to_cell_weights =
      GetInput(context, node, kFwRecurrentToCellWeightsTensor);
  const TfLiteTensor* fw_recurrent_to_output_weights =
      GetInput(context, node, kFwRecurrentToOutputWeightsTensor);

  const TfLiteTensor* fw_cell_to_input_weights =
      GetOptionalInputTensor(context, node, kFwCellToInputWeightsTensor);
  const TfLiteTensor* fw_cell_to_forget_weights =
      GetOptionalInputTensor(context, node, kFwCellToForgetWeightsTensor);
  const TfLiteTensor* fw_cell_to_output_weights =
      GetOptionalInputTensor(context, node, kFwCellToOutputWeightsTensor);

  const TfLiteTensor* fw_input_gate_bias =
      GetOptionalInputTensor(context, node, kFwInputGateBiasTensor);
  const TfLiteTensor* fw_forget_gate_bias =
      GetInput(context, node, kFwForgetGateBiasTensor);
  const TfLiteTensor* fw_cell_bias =
      GetInput(context, node, kFwCellGateBiasTensor);
  const TfLiteTensor* fw_output_gate_bias =
      GetInput(context, node, kFwOutputGateBiasTensor);

  const TfLiteTensor* fw_projection_weights =
      GetOptionalInputTensor(context, node, kFwProjectionWeightsTensor);
  const TfLiteTensor* fw_projection_bias =
      GetOptionalInputTensor(context, node, kFwProjectionBiasTensor);

  TfLiteTensor* fw_activation_state =
      GetVariableInput(context, node, kFwInputActivationStateTensor);
  TfLiteTensor* fw_cell_state =
      GetVariableInput(context, node, kFwInputCellStateTensor);
  TfLiteTensor* fw_output = GetOutput(context, node, kFwOutputTensor);

  // Tensors for the backward cell.
  const TfLiteTensor* bw_input_to_input_weights =
      GetOptionalInputTensor(context, node, kBwInputToInputWeightsTensor);
  const TfLiteTensor* bw_input_to_forget_weights =
      GetInput(context, node, kBwInputToForgetWeightsTensor);
  const TfLiteTensor* bw_input_to_cell_weights =
      GetInput(context, node, kBwInputToCellWeightsTensor);
  const TfLiteTensor* bw_input_to_output_weights =
      GetInput(context, node, kBwInputToOutputWeightsTensor);

  const TfLiteTensor* bw_recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, kBwRecurrentToInputWeightsTensor);
  const TfLiteTensor* bw_recurrent_to_forget_weights =
      GetInput(context, node, kBwRecurrentToForgetWeightsTensor);
  const TfLiteTensor* bw_recurrent_to_cell_weights =
      GetInput(context, node, kBwRecurrentToCellWeightsTensor);
  const TfLiteTensor* bw_recurrent_to_output_weights =
      GetInput(context, node, kBwRecurrentToOutputWeightsTensor);

  const TfLiteTensor* bw_cell_to_input_weights =
      GetOptionalInputTensor(context, node, kBwCellToInputWeightsTensor);
  const TfLiteTensor* bw_cell_to_forget_weights =
      GetOptionalInputTensor(context, node, kBwCellToForgetWeightsTensor);
  const TfLiteTensor* bw_cell_to_output_weights =
      GetOptionalInputTensor(context, node, kBwCellToOutputWeightsTensor);

  const TfLiteTensor* bw_input_gate_bias =
      GetOptionalInputTensor(context, node, kBwInputGateBiasTensor);
  const TfLiteTensor* bw_forget_gate_bias =
      GetInput(context, node, kBwForgetGateBiasTensor);
  const TfLiteTensor* bw_cell_bias =
      GetInput(context, node, kBwCellGateBiasTensor);
  const TfLiteTensor* bw_output_gate_bias =
      GetInput(context, node, kBwOutputGateBiasTensor);

  const TfLiteTensor* bw_projection_weights =
      GetOptionalInputTensor(context, node, kBwProjectionWeightsTensor);
  const TfLiteTensor* bw_projection_bias =
      GetOptionalInputTensor(context, node, kBwProjectionBiasTensor);

  // State tensors.
  TfLiteTensor* bw_activation_state =
      GetVariableInput(context, node, kBwInputActivationStateTensor);
  TfLiteTensor* bw_cell_state =
      GetVariableInput(context, node, kBwInputCellStateTensor);
  TfLiteTensor* bw_output = GetOutput(context, node, kBwOutputTensor);

  // Temporary tensors.
  TfLiteTensor* fw_scratch_buffer =
      GetTemporary(context, node, kFwScratchBuffer);
  TfLiteTensor* bw_scratch_buffer =
      GetTemporary(context, node, kBwScratchBuffer);

  // (Optional) auxiliary inputs.
  const TfLiteTensor* aux_input =
      GetOptionalInputTensor(context, node, kAuxInputTensor);
  const TfLiteTensor* fw_aux_input_to_input_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToInputWeightsTensor);
  const TfLiteTensor* fw_aux_input_to_forget_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToForgetWeightsTensor);
  const TfLiteTensor* fw_aux_input_to_cell_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToCellWeightsTensor);
  const TfLiteTensor* fw_aux_input_to_output_weights =
      GetOptionalInputTensor(context, node, kFwAuxInputToOutputWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_input_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToInputWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_forget_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToForgetWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_cell_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToCellWeightsTensor);
  const TfLiteTensor* bw_aux_input_to_output_weights =
      GetOptionalInputTensor(context, node, kBwAuxInputToOutputWeightsTensor);

  switch (fw_input_to_output_weights->type) {
    case kTfLiteFloat32: {
      TfLiteStatus fw_pass_status = EvalFloat(
          input, fw_input_to_input_weights, fw_input_to_forget_weights,
          fw_input_to_cell_weights, fw_input_to_output_weights,
          fw_recurrent_to_input_weights, fw_recurrent_to_forget_weights,
          fw_recurrent_to_cell_weights, fw_recurrent_to_output_weights,
          fw_cell_to_input_weights, fw_cell_to_forget_weights,
          fw_cell_to_output_weights, aux_input, fw_aux_input_to_input_weights,
          fw_aux_input_to_forget_weights, fw_aux_input_to_cell_weights,
          fw_aux_input_to_output_weights, fw_input_gate_bias,
          fw_forget_gate_bias, fw_cell_bias, fw_output_gate_bias,
          fw_projection_weights, fw_projection_bias, params,
          /*forward_sequence=*/true, fw_scratch_buffer, fw_activation_state,
          fw_cell_state, fw_output);
      TF_LITE_ENSURE_OK(context, fw_pass_status);

      TfLiteStatus bw_pass_status = EvalFloat(
          input, bw_input_to_input_weights, bw_input_to_forget_weights,
          bw_input_to_cell_weights, bw_input_to_output_weights,
          bw_recurrent_to_input_weights, bw_recurrent_to_forget_weights,
          bw_recurrent_to_cell_weights, bw_recurrent_to_output_weights,
          bw_cell_to_input_weights, bw_cell_to_forget_weights,
          bw_cell_to_output_weights, aux_input, bw_aux_input_to_input_weights,
          bw_aux_input_to_forget_weights, bw_aux_input_to_cell_weights,
          bw_aux_input_to_output_weights, bw_input_gate_bias,
          bw_forget_gate_bias, bw_cell_bias, bw_output_gate_bias,
          bw_projection_weights, bw_projection_bias, params,
          /*forward_sequence=*/false, bw_scratch_buffer, bw_activation_state,
          bw_cell_state, bw_output);
      TF_LITE_ENSURE_OK(context, bw_pass_status);
      return kTfLiteOk;
    }
    case kTfLiteUInt8: {
      TfLiteTensor* input_quantized =
          GetTemporary(context, node, kInputQuantized);
      TfLiteTensor* aux_input_quantized =
          GetTemporary(context, node, kAuxInputQuantized);
      TfLiteTensor* fw_activation_state_quantized =
          GetTemporary(context, node, kFwActivationStateQuantized);
      TfLiteTensor* bw_activation_state_quantized =
          GetTemporary(context, node, kBwActivationStateQuantized);
      TfLiteTensor* fw_cell_state_quantized =
          GetTemporary(context, node, kFwCellStateQuantized);
      TfLiteTensor* bw_cell_state_quantized =
          GetTemporary(context, node, kBwCellStateQuantized);
      TfLiteTensor* scaling_factors =
          GetTemporary(context, node, kScalingFactors);
      TfLiteTensor* prod_scaling_factors =
          GetTemporary(context, node, kProductScalingFactors);
      TfLiteTensor* recovered_cell_weights =
          GetTemporary(context, node, kRecoveredCellWeights);

      TfLiteStatus fw_pass_status = EvalHybrid(
          input, fw_input_to_input_weights, fw_input_to_forget_weights,
          fw_input_to_cell_weights, fw_input_to_output_weights,
          fw_recurrent_to_input_weights, fw_recurrent_to_forget_weights,
          fw_recurrent_to_cell_weights, fw_recurrent_to_output_weights,
          fw_cell_to_input_weights, fw_cell_to_forget_weights,
          fw_cell_to_output_weights, aux_input, fw_aux_input_to_input_weights,
          fw_aux_input_to_forget_weights, fw_aux_input_to_cell_weights,
          fw_aux_input_to_output_weights, fw_input_gate_bias,
          fw_forget_gate_bias, fw_cell_bias, fw_output_gate_bias,
          fw_projection_weights, fw_projection_bias, params,
          /*forward_sequence=*/true, fw_scratch_buffer, scaling_factors,
          prod_scaling_factors, recovered_cell_weights, input_quantized,
          aux_input_quantized, fw_activation_state_quantized,
          fw_cell_state_quantized, fw_activation_state, fw_cell_state,
          fw_output);
      TF_LITE_ENSURE_OK(context, fw_pass_status);

      TfLiteStatus bw_pass_status = EvalHybrid(
          input, bw_input_to_input_weights, bw_input_to_forget_weights,
          bw_input_to_cell_weights, bw_input_to_output_weights,
          bw_recurrent_to_input_weights, bw_recurrent_to_forget_weights,
          bw_recurrent_to_cell_weights, bw_recurrent_to_output_weights,
          bw_cell_to_input_weights, bw_cell_to_forget_weights,
          bw_cell_to_output_weights, aux_input, fw_aux_input_to_input_weights,
          fw_aux_input_to_forget_weights, fw_aux_input_to_cell_weights,
          fw_aux_input_to_output_weights, bw_input_gate_bias,
          bw_forget_gate_bias, bw_cell_bias, bw_output_gate_bias,
          bw_projection_weights, bw_projection_bias, params,
          /*forward_sequence=*/false, bw_scratch_buffer, scaling_factors,
          prod_scaling_factors, recovered_cell_weights, input_quantized,
          aux_input_quantized, bw_activation_state_quantized,
          bw_cell_state_quantized, bw_activation_state, bw_cell_state,
          bw_output);
      TF_LITE_ENSURE_OK(context, bw_pass_status);
      return kTfLiteOk;
    }
    default:
      context->ReportError(context, "Type %d is not currently supported.",
                           fw_input_to_output_weights->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace bidirectional_sequence_lstm

TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_LSTM() {
  static TfLiteRegistration r = {
      bidirectional_sequence_lstm::Init, bidirectional_sequence_lstm::Free,
      bidirectional_sequence_lstm::Prepare, bidirectional_sequence_lstm::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
