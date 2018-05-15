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

#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
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

// Output tensors.
constexpr int kFwOutputStateTensor = 0;
constexpr int kFwCellStateTensor = 1;
constexpr int kFwOutputTensor = 2;

constexpr int kBwOutputStateTensor = 3;
constexpr int kBwCellStateTensor = 4;
constexpr int kBwOutputTensor = 5;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* scratch_tensor_index = new int;
  context->AddTensors(context, 2, scratch_tensor_index);
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
  auto* params = reinterpret_cast<TfLiteLSTMParams*>(node->builtin_data);

  // Making sure clipping parameters have valid values.
  // == 0 means no clipping
  //  > 0 means clipping
  TF_LITE_ENSURE(context, params->cell_clip >= 0);
  TF_LITE_ENSURE(context, params->proj_clip >= 0);

  TfLiteTensor* input_to_input_weights =
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

  TfLiteTensor* recurrent_to_input_weights =
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

  TfLiteTensor* cell_to_input_weights =
      GetOptionalInputTensor(context, node, cell_to_input_weights_tensor);
  if (cell_to_input_weights) {
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->data[0], n_cell);
  }

  TfLiteTensor* cell_to_forget_weights =
      GetOptionalInputTensor(context, node, cell_to_forget_weights_tensor);
  if (cell_to_forget_weights) {
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->data[0], n_cell);
  }

  TfLiteTensor* cell_to_output_weights =
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
  TfLiteTensor* input_gate_bias =
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

  TfLiteTensor* projection_weights =
      GetOptionalInputTensor(context, node, projection_weights_tensor);
  if (projection_weights) {
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[0], n_output);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[1], n_cell);
  }

  TfLiteTensor* projection_bias =
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
  CheckLstmTensorDimensions(
      context, node, n_input, n_output, n_cell, kFwInputToInputWeightsTensor,
      kFwInputToForgetWeightsTensor, kFwInputToCellWeightsTensor,
      kFwInputToOutputWeightsTensor, kFwRecurrentToInputWeightsTensor,
      kFwRecurrentToForgetWeightsTensor, kFwRecurrentToCellWeightsTensor,
      kFwRecurrentToOutputWeightsTensor, kFwCellToInputWeightsTensor,
      kFwCellToForgetWeightsTensor, kFwCellToOutputWeightsTensor,
      kFwInputGateBiasTensor, kFwForgetGateBiasTensor, kFwCellGateBiasTensor,
      kFwOutputGateBiasTensor, kFwProjectionWeightsTensor,
      kFwProjectionBiasTensor);

  CheckLstmTensorDimensions(
      context, node, n_input, n_output, n_cell, kBwInputToInputWeightsTensor,
      kBwInputToForgetWeightsTensor, kBwInputToCellWeightsTensor,
      kBwInputToOutputWeightsTensor, kBwRecurrentToInputWeightsTensor,
      kBwRecurrentToForgetWeightsTensor, kBwRecurrentToCellWeightsTensor,
      kBwRecurrentToOutputWeightsTensor, kBwCellToInputWeightsTensor,
      kBwCellToForgetWeightsTensor, kBwCellToOutputWeightsTensor,
      kBwInputGateBiasTensor, kBwForgetGateBiasTensor, kBwCellGateBiasTensor,
      kBwOutputGateBiasTensor, kBwProjectionWeightsTensor,
      kBwProjectionBiasTensor);

  // Check if Forward and Backward tensors match along required dimensions.
  return kTfLiteOk;
}

// Resize the output, state and scratch tensors based on the sizes of the input
// tensors. Also check that the size of the input tensors match each other.
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  int* scratch_tensor_index = reinterpret_cast<int*>(node->user_data);

  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 35);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 6);

  // Inferring batch size, number of outputs and sequence length and
  // number of cells from the input tensors.
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input->dims->size > 1);
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
  CheckInputTensorDimensions(context, node, n_input, n_fw_output, n_fw_cell);

  // Get the pointer to output, state and scratch buffer tensors.
  TfLiteTensor* fw_output = GetOutput(context, node, kFwOutputTensor);
  TfLiteTensor* fw_output_state =
      GetOutput(context, node, kFwOutputStateTensor);
  TfLiteTensor* fw_cell_state = GetOutput(context, node, kFwCellStateTensor);

  // Resize the output, output_state and cell_state tensors.
  TfLiteIntArray* fw_output_size = TfLiteIntArrayCreate(3);
  fw_output_size->data[0] = max_time;
  fw_output_size->data[1] = n_batch;
  fw_output_size->data[2] = n_fw_output;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, fw_output, fw_output_size));

  TfLiteIntArray* fw_output_state_size = TfLiteIntArrayCreate(2);
  fw_output_state_size->data[0] = n_batch;
  fw_output_state_size->data[1] = n_fw_output;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, fw_output_state,
                                                   fw_output_state_size));

  TfLiteIntArray* fw_cell_size = TfLiteIntArrayCreate(2);
  fw_cell_size->data[0] = n_batch;
  fw_cell_size->data[1] = n_fw_cell;
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, fw_cell_state, fw_cell_size));

  // Create a scratch buffer tensor.
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(2);
  node->temporaries->data[0] = *scratch_tensor_index;
  TfLiteTensor* fw_scratch_buffer = GetTemporary(context, node, /*index=*/0);
  fw_scratch_buffer->type = input->type;
  fw_scratch_buffer->allocation_type = kTfLiteArenaRw;

  // Mark state tensors as persistent tensors.
  fw_output_state->allocation_type = kTfLiteArenaRwPersistent;
  fw_cell_state->allocation_type = kTfLiteArenaRwPersistent;

  TfLiteTensor* fw_input_to_input_weights =
      GetOptionalInputTensor(context, node, kFwInputToInputWeightsTensor);
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
  CheckInputTensorDimensions(context, node, n_input, n_bw_output, n_bw_cell);

  // Get the pointer to output, output_state and cell_state buffer tensors.
  TfLiteTensor* bw_output = GetOutput(context, node, kBwOutputTensor);
  TfLiteTensor* bw_output_state =
      GetOutput(context, node, kBwOutputStateTensor);
  TfLiteTensor* bw_cell_state = GetOutput(context, node, kBwCellStateTensor);

  // Resize the output, output_state and cell_state tensors.
  TfLiteIntArray* bw_output_size = TfLiteIntArrayCreate(3);
  bw_output_size->data[0] = max_time;
  bw_output_size->data[1] = n_batch;
  bw_output_size->data[2] = n_bw_output;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, bw_output, bw_output_size));

  TfLiteIntArray* bw_output_state_size = TfLiteIntArrayCreate(2);
  bw_output_state_size->data[0] = n_batch;
  bw_output_state_size->data[1] = n_bw_output;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, bw_output_state,
                                                   bw_output_state_size));

  TfLiteIntArray* bw_cell_size = TfLiteIntArrayCreate(2);
  bw_cell_size->data[0] = n_batch;
  bw_cell_size->data[1] = n_bw_cell;
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, bw_cell_state, bw_cell_size));

  // Create a scratch buffer tensor.
  node->temporaries->data[1] = *(scratch_tensor_index) + 1;
  TfLiteTensor* bw_scratch_buffer = GetTemporary(context, node, /*index=*/1);
  bw_scratch_buffer->type = input->type;
  bw_scratch_buffer->allocation_type = kTfLiteArenaRw;

  // Mark state tensors as persistent tensors.
  bw_output_state->allocation_type = kTfLiteArenaRwPersistent;
  bw_cell_state->allocation_type = kTfLiteArenaRwPersistent;

  TfLiteTensor* bw_input_to_input_weights =
      GetOptionalInputTensor(context, node, kBwInputToInputWeightsTensor);
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
  return kTfLiteOk;
}

// The LSTM Op engine.
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteLSTMParams*>(node->builtin_data);

  // Input tensor.
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const int max_time = input->dims->data[0];
  const int n_batch = input->dims->data[1];
  const int n_input = input->dims->data[2];

  // Tensors for the forward cell.
  TfLiteTensor* fw_input_to_input_weights =
      GetOptionalInputTensor(context, node, kFwInputToInputWeightsTensor);
  const TfLiteTensor* fw_input_to_forget_weights =
      GetInput(context, node, kFwInputToForgetWeightsTensor);
  const TfLiteTensor* fw_input_to_cell_weights =
      GetInput(context, node, kFwInputToCellWeightsTensor);
  const TfLiteTensor* fw_input_to_output_weights =
      GetInput(context, node, kFwInputToOutputWeightsTensor);

  TfLiteTensor* fw_recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, kFwRecurrentToInputWeightsTensor);
  const TfLiteTensor* fw_recurrent_to_forget_weights =
      GetInput(context, node, kFwRecurrentToForgetWeightsTensor);
  const TfLiteTensor* fw_recurrent_to_cell_weights =
      GetInput(context, node, kFwRecurrentToCellWeightsTensor);
  const TfLiteTensor* fw_recurrent_to_output_weights =
      GetInput(context, node, kFwRecurrentToOutputWeightsTensor);

  TfLiteTensor* fw_cell_to_input_weights =
      GetOptionalInputTensor(context, node, kFwCellToInputWeightsTensor);
  TfLiteTensor* fw_cell_to_forget_weights =
      GetOptionalInputTensor(context, node, kFwCellToForgetWeightsTensor);
  TfLiteTensor* fw_cell_to_output_weights =
      GetOptionalInputTensor(context, node, kFwCellToOutputWeightsTensor);

  TfLiteTensor* fw_input_gate_bias =
      GetOptionalInputTensor(context, node, kFwInputGateBiasTensor);
  const TfLiteTensor* fw_forget_gate_bias =
      GetInput(context, node, kFwForgetGateBiasTensor);
  const TfLiteTensor* fw_cell_bias =
      GetInput(context, node, kFwCellGateBiasTensor);
  const TfLiteTensor* fw_output_gate_bias =
      GetInput(context, node, kFwOutputGateBiasTensor);

  TfLiteTensor* fw_projection_weights =
      GetOptionalInputTensor(context, node, kFwProjectionWeightsTensor);
  TfLiteTensor* fw_projection_bias =
      GetOptionalInputTensor(context, node, kFwProjectionBiasTensor);

  TfLiteTensor* fw_output_state =
      GetOutput(context, node, kFwOutputStateTensor);
  TfLiteTensor* fw_cell_state = GetOutput(context, node, kFwCellStateTensor);
  TfLiteTensor* fw_output = GetOutput(context, node, kFwOutputTensor);

  // Tensors for the backward cell.
  TfLiteTensor* bw_input_to_input_weights =
      GetOptionalInputTensor(context, node, kBwInputToInputWeightsTensor);
  const TfLiteTensor* bw_input_to_forget_weights =
      GetInput(context, node, kBwInputToForgetWeightsTensor);
  const TfLiteTensor* bw_input_to_cell_weights =
      GetInput(context, node, kBwInputToCellWeightsTensor);
  const TfLiteTensor* bw_input_to_output_weights =
      GetInput(context, node, kBwInputToOutputWeightsTensor);

  TfLiteTensor* bw_recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, kBwRecurrentToInputWeightsTensor);
  const TfLiteTensor* bw_recurrent_to_forget_weights =
      GetInput(context, node, kBwRecurrentToForgetWeightsTensor);
  const TfLiteTensor* bw_recurrent_to_cell_weights =
      GetInput(context, node, kBwRecurrentToCellWeightsTensor);
  const TfLiteTensor* bw_recurrent_to_output_weights =
      GetInput(context, node, kBwRecurrentToOutputWeightsTensor);

  TfLiteTensor* bw_cell_to_input_weights =
      GetOptionalInputTensor(context, node, kBwCellToInputWeightsTensor);
  TfLiteTensor* bw_cell_to_forget_weights =
      GetOptionalInputTensor(context, node, kBwCellToForgetWeightsTensor);
  TfLiteTensor* bw_cell_to_output_weights =
      GetOptionalInputTensor(context, node, kBwCellToOutputWeightsTensor);

  TfLiteTensor* bw_input_gate_bias =
      GetOptionalInputTensor(context, node, kBwInputGateBiasTensor);
  const TfLiteTensor* bw_forget_gate_bias =
      GetInput(context, node, kBwForgetGateBiasTensor);
  const TfLiteTensor* bw_cell_bias =
      GetInput(context, node, kBwCellGateBiasTensor);
  const TfLiteTensor* bw_output_gate_bias =
      GetInput(context, node, kBwOutputGateBiasTensor);

  TfLiteTensor* bw_projection_weights =
      GetOptionalInputTensor(context, node, kBwProjectionWeightsTensor);
  TfLiteTensor* bw_projection_bias =
      GetOptionalInputTensor(context, node, kBwProjectionBiasTensor);

  TfLiteTensor* bw_output_state =
      GetOutput(context, node, kBwOutputStateTensor);
  TfLiteTensor* bw_cell_state = GetOutput(context, node, kBwCellStateTensor);
  TfLiteTensor* bw_output = GetOutput(context, node, kBwOutputTensor);

  // n_cell and n_output will be the same size when there is no projection.
  const int n_fw_cell = fw_input_to_output_weights->dims->data[0];
  const int n_fw_output = fw_recurrent_to_output_weights->dims->data[1];

  // Since we have already checked that weights are all there or none, we can
  // check the existense of only one to the get the condition.
  const bool fw_use_cifg = (fw_input_to_input_weights == nullptr);
  const bool fw_use_peephole = (fw_cell_to_output_weights != nullptr);

  // Index the scratch buffers pointers to the global scratch buffer.
  TfLiteTensor* fw_scratch_buffer =
      &context->tensors[node->temporaries->data[0]];
  float* fw_input_gate_scratch = nullptr;
  float* fw_cell_scratch = nullptr;
  float* fw_forget_gate_scratch = nullptr;
  float* fw_output_gate_scratch = nullptr;
  if (fw_use_cifg) {
    fw_cell_scratch = fw_scratch_buffer->data.f;
    fw_forget_gate_scratch = fw_scratch_buffer->data.f + n_fw_cell * n_batch;
    fw_output_gate_scratch =
        fw_scratch_buffer->data.f + 2 * n_fw_cell * n_batch;
  } else {
    fw_input_gate_scratch = fw_scratch_buffer->data.f;
    fw_cell_scratch = fw_scratch_buffer->data.f + n_fw_cell * n_batch;
    fw_forget_gate_scratch =
        fw_scratch_buffer->data.f + 2 * n_fw_cell * n_batch;
    fw_output_gate_scratch =
        fw_scratch_buffer->data.f + 3 * n_fw_cell * n_batch;
  }

  // Check optional tensors, the respective pointers can be null.
  const float* fw_input_to_input_weights_ptr =
      (fw_use_cifg) ? nullptr : fw_input_to_input_weights->data.f;
  const float* fw_recurrent_to_input_weights_ptr =
      (fw_use_cifg) ? nullptr : fw_recurrent_to_input_weights->data.f;
  const float* fw_input_gate_bias_ptr =
      (fw_use_cifg) ? nullptr : fw_input_gate_bias->data.f;
  const float* fw_cell_to_input_weights_ptr =
      (fw_use_peephole && !fw_use_cifg) ? fw_cell_to_input_weights->data.f
                                        : nullptr;
  const float* fw_cell_to_forget_weights_ptr =
      (fw_use_peephole) ? fw_cell_to_forget_weights->data.f : nullptr;
  const float* fw_cell_to_output_weights_ptr =
      (fw_use_peephole) ? fw_cell_to_output_weights->data.f : nullptr;
  const float* fw_projection_weights_ptr = (fw_projection_weights == nullptr)
                                               ? nullptr
                                               : fw_projection_weights->data.f;
  const float* fw_projection_bias_ptr =
      (fw_projection_bias == nullptr) ? nullptr : fw_projection_bias->data.f;

  // Loop through the sequence.
  for (int t = 0; t < max_time; t++) {
    const float* input_ptr_batch = input->data.f + t * n_batch * n_input;
    float* output_ptr_time = fw_output->data.f + t * n_batch * n_fw_output;

    kernel_utils::LstmStep(
        input_ptr_batch, fw_input_to_input_weights_ptr,
        fw_input_to_forget_weights->data.f, fw_input_to_cell_weights->data.f,
        fw_input_to_output_weights->data.f, fw_recurrent_to_input_weights_ptr,
        fw_recurrent_to_forget_weights->data.f,
        fw_recurrent_to_cell_weights->data.f,
        fw_recurrent_to_output_weights->data.f, fw_cell_to_input_weights_ptr,
        fw_cell_to_forget_weights_ptr, fw_cell_to_output_weights_ptr,
        fw_input_gate_bias_ptr, fw_forget_gate_bias->data.f,
        fw_cell_bias->data.f, fw_output_gate_bias->data.f,
        fw_projection_weights_ptr, fw_projection_bias_ptr, params, n_batch,
        n_fw_cell, n_input, n_fw_output, fw_output_state->data.f,
        fw_cell_state->data.f, fw_input_gate_scratch, fw_forget_gate_scratch,
        fw_cell_scratch, fw_output_gate_scratch, output_ptr_time);
  }

  // n_cell and n_output will be the same size when there is no projection.
  const int n_bw_cell = bw_input_to_output_weights->dims->data[0];
  const int n_bw_output = bw_recurrent_to_output_weights->dims->data[1];

  // Since we have already checked that weights are all there or none, we can
  // check the existense of only one to the get the condition.
  const bool bw_use_cifg = (bw_input_to_input_weights == nullptr);
  const bool bw_use_peephole = (bw_cell_to_output_weights != nullptr);

  // Index the scratch buffers pointers to the global scratch buffer.
  TfLiteTensor* bw_scratch_buffer =
      &context->tensors[node->temporaries->data[1]];
  float* bw_input_gate_scratch = nullptr;
  float* bw_cell_scratch = nullptr;
  float* bw_forget_gate_scratch = nullptr;
  float* bw_output_gate_scratch = nullptr;
  if (bw_use_cifg) {
    bw_cell_scratch = bw_scratch_buffer->data.f;
    bw_forget_gate_scratch = bw_scratch_buffer->data.f + n_bw_cell * n_batch;
    bw_output_gate_scratch =
        bw_scratch_buffer->data.f + 2 * n_bw_cell * n_batch;
  } else {
    bw_input_gate_scratch = bw_scratch_buffer->data.f;
    bw_cell_scratch = bw_scratch_buffer->data.f + n_bw_cell * n_batch;
    bw_forget_gate_scratch =
        bw_scratch_buffer->data.f + 2 * n_bw_cell * n_batch;
    bw_output_gate_scratch =
        bw_scratch_buffer->data.f + 3 * n_bw_cell * n_batch;
  }

  // Check optional tensors, the respective pointers can be null.
  const float* bw_input_to_input_weights_ptr =
      (bw_use_cifg) ? nullptr : bw_input_to_input_weights->data.f;
  const float* bw_recurrent_to_input_weights_ptr =
      (bw_use_cifg) ? nullptr : bw_recurrent_to_input_weights->data.f;
  const float* bw_input_gate_bias_ptr =
      (bw_use_cifg) ? nullptr : bw_input_gate_bias->data.f;
  const float* bw_cell_to_input_weights_ptr =
      (bw_use_peephole && !bw_use_cifg) ? bw_cell_to_input_weights->data.f
                                        : nullptr;
  const float* bw_cell_to_forget_weights_ptr =
      (bw_use_peephole) ? bw_cell_to_forget_weights->data.f : nullptr;
  const float* bw_cell_to_output_weights_ptr =
      (bw_use_peephole) ? bw_cell_to_output_weights->data.f : nullptr;
  const float* bw_projection_weights_ptr = (bw_projection_weights == nullptr)
                                               ? nullptr
                                               : bw_projection_weights->data.f;
  const float* bw_projection_bias_ptr =
      (bw_projection_bias == nullptr) ? nullptr : bw_projection_bias->data.f;

  // Loop through the sequence backwards.
  for (int t = max_time - 1; t >= 0; t--) {
    const float* input_ptr_batch = input->data.f + t * n_batch * n_input;
    float* output_ptr_time = bw_output->data.f + t * n_batch * n_bw_output;

    kernel_utils::LstmStep(
        input_ptr_batch, bw_input_to_input_weights_ptr,
        bw_input_to_forget_weights->data.f, bw_input_to_cell_weights->data.f,
        bw_input_to_output_weights->data.f, bw_recurrent_to_input_weights_ptr,
        bw_recurrent_to_forget_weights->data.f,
        bw_recurrent_to_cell_weights->data.f,
        bw_recurrent_to_output_weights->data.f, bw_cell_to_input_weights_ptr,
        bw_cell_to_forget_weights_ptr, bw_cell_to_output_weights_ptr,
        bw_input_gate_bias_ptr, bw_forget_gate_bias->data.f,
        bw_cell_bias->data.f, bw_output_gate_bias->data.f,
        bw_projection_weights_ptr, bw_projection_bias_ptr, params, n_batch,
        n_bw_cell, n_input, n_bw_output, bw_output_state->data.f,
        bw_cell_state->data.f, bw_input_gate_scratch, bw_forget_gate_scratch,
        bw_cell_scratch, bw_output_gate_scratch, output_ptr_time);
  }

  // Backward step.
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
