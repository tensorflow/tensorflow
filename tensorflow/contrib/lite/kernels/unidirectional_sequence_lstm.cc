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
#include "tensorflow/contrib/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace unidirectional_sequence_lstm {

// Input Tensors of size {max_time, n_batch, n_input}
constexpr int kInputTensor = 0;

// Input weight tensors of size: {n_cell, n_input}
constexpr int kInputToInputWeightsTensor = 1;  // Optional
constexpr int kInputToForgetWeightsTensor = 2;
constexpr int kInputToCellWeightsTensor = 3;
constexpr int kInputToOutputWeightsTensor = 4;

// Recurrent weight tensors of size {n_cell, n_output}
constexpr int kRecurrentToInputWeightsTensor = 5;  // Optional
constexpr int kRecurrentToForgetWeightsTensor = 6;
constexpr int kRecurrentToCellWeightsTensor = 7;
constexpr int kRecurrentToOutputWeightsTensor = 8;

// Peephole weights tensors of size {n_cell}, representing a diagonal matrix.
constexpr int kCellToInputWeightsTensor = 9;    // Optional
constexpr int kCellToForgetWeightsTensor = 10;  // Optional
constexpr int kCellToOutputWeightsTensor = 11;  // Optional

// Gates bias tensors of size {n_cell}
constexpr int kInputGateBiasTensor = 12;  // Optional
constexpr int kForgetGateBiasTensor = 13;
constexpr int kCellGateBiasTensor = 14;
constexpr int kOutputGateBiasTensor = 15;

// Projection weight tensor of size {n_output, n_cell}
constexpr int kProjectionWeightsTensor = 16;  // Optional
// Projection bias tensor of size {n_output}
constexpr int kProjectionBiasTensor = 17;  // Optional

// Output tensors.
constexpr int kScratchBufferTensor = 0;
constexpr int kOutputStateTensor = 1;
constexpr int kCellStateTensor = 2;
constexpr int kOutputTensor = 3;

// Check that input tensor dimensions matches with each other.
TfLiteStatus CheckInputTensorDimensions(TfLiteContext* context,
                                        TfLiteNode* node, int n_input,
                                        int n_output, int n_cell) {
  auto* params = reinterpret_cast<TfLiteLSTMParams*>(node->builtin_data);

  // Making sure clipping parameters have valid values.
  // == 0 means no clipping
  //  > 0 means clipping
  TF_LITE_ENSURE(context, params->cell_clip >= 0);
  TF_LITE_ENSURE(context, params->proj_clip >= 0);

  TfLiteTensor* input_to_input_weights =
      GetOptionalInputTensor(context, node, kInputToInputWeightsTensor);
  if (input_to_input_weights) {
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->data[1], n_input);
  }

  TfLiteTensor* input_to_forget_weights =
      GetInput(context, node, kInputToForgetWeightsTensor);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->data[1], n_input);

  TfLiteTensor* input_to_cell_weights =
      GetInput(context, node, kInputToCellWeightsTensor);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->data[1], n_input);

  TfLiteTensor* recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, kRecurrentToInputWeightsTensor);
  if (recurrent_to_input_weights) {
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->data[0],
                      n_cell);
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->data[1],
                      n_output);
  }

  TfLiteTensor* recurrent_to_forget_weights =
      GetInput(context, node, kRecurrentToForgetWeightsTensor);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->data[0],
                    n_cell);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->data[1],
                    n_output);

  TfLiteTensor* recurrent_to_cell_weights =
      GetInput(context, node, kRecurrentToCellWeightsTensor);
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
      GetOptionalInputTensor(context, node, kCellToInputWeightsTensor);
  if (cell_to_input_weights) {
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->data[0], n_cell);
  }

  TfLiteTensor* cell_to_forget_weights =
      GetOptionalInputTensor(context, node, kCellToForgetWeightsTensor);
  if (cell_to_forget_weights) {
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->data[0], n_cell);
  }

  TfLiteTensor* cell_to_output_weights =
      GetOptionalInputTensor(context, node, kCellToOutputWeightsTensor);
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
      GetOptionalInputTensor(context, node, kInputGateBiasTensor);
  if (use_cifg) {
    TF_LITE_ENSURE_EQ(context, input_gate_bias, nullptr);
  } else {
    TF_LITE_ENSURE_EQ(context, input_gate_bias->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, input_gate_bias->dims->data[0], n_cell);
  }

  TfLiteTensor* forget_gate_bias =
      GetInput(context, node, kForgetGateBiasTensor);
  TF_LITE_ENSURE_EQ(context, forget_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, forget_gate_bias->dims->data[0], n_cell);

  TfLiteTensor* cell_bias = GetInput(context, node, kCellGateBiasTensor);
  TF_LITE_ENSURE_EQ(context, cell_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, cell_bias->dims->data[0], n_cell);

  TfLiteTensor* output_gate_bias =
      GetInput(context, node, kOutputGateBiasTensor);
  TF_LITE_ENSURE_EQ(context, output_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, output_gate_bias->dims->data[0], n_cell);

  TfLiteTensor* projection_weights =
      GetOptionalInputTensor(context, node, kProjectionWeightsTensor);
  if (projection_weights) {
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[0], n_output);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[1], n_cell);
  }

  TfLiteTensor* projection_bias =
      GetOptionalInputTensor(context, node, kProjectionBiasTensor);
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

// Resize the output, state and scratch tensors based on the sizes of the input
// tensors. Also check that the size of the input tensors match each other.
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 18);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 4);

  // Inferring batch size, number of outputs and sequence length and
  // number of cells from the input tensors.
  TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input->dims->size > 1);
  const int max_time = input->dims->data[0];
  const int n_batch = input->dims->data[1];
  const int n_input = input->dims->data[2];

  TfLiteTensor* input_to_output_weights =
      GetInput(context, node, kInputToOutputWeightsTensor);
  const int n_cell = input_to_output_weights->dims->data[0];
  TF_LITE_ENSURE_EQ(context, input_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_output_weights->dims->data[1], n_input);

  TfLiteTensor* recurrent_to_output_weights =
      GetInput(context, node, kRecurrentToOutputWeightsTensor);
  TF_LITE_ENSURE_EQ(context, recurrent_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_output_weights->dims->data[0],
                    n_cell);
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Check that input tensor dimensions matches with each other.
  CheckInputTensorDimensions(context, node, n_input, n_output, n_cell);

  // Get the pointer to output, state and scratch buffer tensors.
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TfLiteTensor* output_state = GetOutput(context, node, kOutputStateTensor);
  TfLiteTensor* cell_state = GetOutput(context, node, kCellStateTensor);
  // TODO(ghodrat): Modify this as soon as we have a finalized method for
  // scratch buffers.
  TfLiteTensor* scratch_buffer = GetOutput(context, node, kScratchBufferTensor);

  // Resize the output and output_state tensors.
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(3);
  output_size->data[0] = max_time;
  output_size->data[1] = n_batch;
  output_size->data[2] = n_output;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size));

  TfLiteIntArray* output_state_size = TfLiteIntArrayCreate(2);
  output_state_size->data[0] = n_batch;
  output_state_size->data[1] = n_output;
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, output_state, output_state_size));

  // Resize the scratch buffer tensor.
  TfLiteIntArray* cell_size = TfLiteIntArrayCreate(2);
  cell_size->data[0] = n_batch;
  cell_size->data[1] = n_cell;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, cell_state, cell_size));

  // Mark state tensors as persistent tensors.
  output_state->allocation_type = kTfLiteArenaRwPersistent;
  cell_state->allocation_type = kTfLiteArenaRwPersistent;

  TfLiteTensor* input_to_input_weights =
      GetOptionalInputTensor(context, node, kInputToInputWeightsTensor);
  const bool use_cifg = (input_to_input_weights == nullptr);
  if (use_cifg) {
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(2);
    scratch_buffer_size->data[0] = n_batch;
    // Reserving space for Cell, Forget, Output gates
    scratch_buffer_size->data[1] = n_cell * 3;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  } else {
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(2);
    scratch_buffer_size->data[0] = n_batch;
    // Reserving space for Input, Cell, Forget, Output gates
    scratch_buffer_size->data[1] = n_cell * 4;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }
  return kTfLiteOk;
}

// The LSTM Op engine.
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteLSTMParams*>(node->builtin_data);
  TfLiteTensor* input = GetInput(context, node, kInputTensor);

  TfLiteTensor* input_to_input_weights =
      GetOptionalInputTensor(context, node, kInputToInputWeightsTensor);
  TfLiteTensor* input_to_forget_weights =
      GetInput(context, node, kInputToForgetWeightsTensor);
  TfLiteTensor* input_to_cell_weights =
      GetInput(context, node, kInputToCellWeightsTensor);
  TfLiteTensor* input_to_output_weights =
      GetInput(context, node, kInputToOutputWeightsTensor);

  TfLiteTensor* recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, kRecurrentToInputWeightsTensor);
  TfLiteTensor* recurrent_to_forget_weights =
      GetInput(context, node, kRecurrentToForgetWeightsTensor);
  TfLiteTensor* recurrent_to_cell_weights =
      GetInput(context, node, kRecurrentToCellWeightsTensor);
  TfLiteTensor* recurrent_to_output_weights =
      GetInput(context, node, kRecurrentToOutputWeightsTensor);

  TfLiteTensor* cell_to_input_weights =
      GetOptionalInputTensor(context, node, kCellToInputWeightsTensor);
  TfLiteTensor* cell_to_forget_weights =
      GetOptionalInputTensor(context, node, kCellToForgetWeightsTensor);
  TfLiteTensor* cell_to_output_weights =
      GetOptionalInputTensor(context, node, kCellToOutputWeightsTensor);

  TfLiteTensor* input_gate_bias =
      GetOptionalInputTensor(context, node, kInputGateBiasTensor);
  TfLiteTensor* forget_gate_bias =
      GetInput(context, node, kForgetGateBiasTensor);
  TfLiteTensor* cell_bias = GetInput(context, node, kCellGateBiasTensor);
  TfLiteTensor* output_gate_bias =
      GetInput(context, node, kOutputGateBiasTensor);

  TfLiteTensor* projection_weights =
      GetOptionalInputTensor(context, node, kProjectionWeightsTensor);
  TfLiteTensor* projection_bias =
      GetOptionalInputTensor(context, node, kProjectionBiasTensor);

  TfLiteTensor* output_state = GetOutput(context, node, kOutputStateTensor);
  TfLiteTensor* cell_state = GetOutput(context, node, kCellStateTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  const int max_time = input->dims->data[0];
  const int n_batch = input->dims->data[1];
  const int n_input = input->dims->data[2];
  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Since we have already checked that weights are all there or none, we can
  // check the existense of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights == nullptr);
  const bool use_peephole = (cell_to_output_weights != nullptr);

  // Index the scratch buffers pointers to the global scratch buffer.
  TfLiteTensor* scratch_buffer = GetOutput(context, node, kScratchBufferTensor);
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

  for (int t = 0; t < max_time; t++) {
    const float* input_ptr_time = input->data.f + t * n_batch * n_input;
    // Initialize scratch buffers with bias.
    if (!use_cifg) {
      tensor_utils::VectorBatchVectorAssign(input_gate_bias->data.f, n_cell,
                                            n_batch, input_gate_scratch);
    }
    tensor_utils::VectorBatchVectorAssign(forget_gate_bias->data.f, n_cell,
                                          n_batch, forget_gate_scratch);
    tensor_utils::VectorBatchVectorAssign(cell_bias->data.f, n_cell, n_batch,
                                          cell_scratch);
    tensor_utils::VectorBatchVectorAssign(output_gate_bias->data.f, n_cell,
                                          n_batch, output_gate_scratch);

    // For each batch and cell: compute input_weight * input.
    if (!use_cifg) {
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          input_to_input_weights->data.f, n_cell, n_input, input_ptr_time,
          n_batch, input_gate_scratch, /*result_stride=*/1);
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        input_to_forget_weights->data.f, n_cell, n_input, input_ptr_time,
        n_batch, forget_gate_scratch, /*result_stride=*/1);
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        input_to_cell_weights->data.f, n_cell, n_input, input_ptr_time, n_batch,
        cell_scratch, /*result_stride=*/1);
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        input_to_output_weights->data.f, n_cell, n_input, input_ptr_time,
        n_batch, output_gate_scratch, /*result_stride=*/1);

    // For each batch and cell: compute recurrent_weight * output_state.
    if (!use_cifg) {
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          recurrent_to_input_weights->data.f, n_cell, n_output,
          output_state->data.f, n_batch, input_gate_scratch,
          /*result_stride=*/1);
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        recurrent_to_forget_weights->data.f, n_cell, n_output,
        output_state->data.f, n_batch, forget_gate_scratch,
        /*result_stride=*/1);
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        recurrent_to_cell_weights->data.f, n_cell, n_output,
        output_state->data.f, n_batch, cell_scratch, /*result_stride=*/1);
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        recurrent_to_output_weights->data.f, n_cell, n_output,
        output_state->data.f, n_batch, output_gate_scratch,
        /*result_stride=*/1);

    // For each batch and cell: update input gate.
    if (!use_cifg) {
      if (use_peephole) {
        tensor_utils::VectorBatchVectorCwiseProductAccumulate(
            cell_to_input_weights->data.f, n_cell, cell_state->data.f, n_batch,
            input_gate_scratch);
      }
      tensor_utils::ApplySigmoidToVector(input_gate_scratch, n_cell * n_batch,
                                         input_gate_scratch);
    }

    // For each batch and cell: update forget gate.
    if (use_peephole) {
      tensor_utils::VectorBatchVectorCwiseProductAccumulate(
          cell_to_forget_weights->data.f, n_cell, cell_state->data.f, n_batch,
          forget_gate_scratch);
    }
    tensor_utils::ApplySigmoidToVector(forget_gate_scratch, n_cell * n_batch,
                                       forget_gate_scratch);

    // For each batch and cell: update the cell.
    tensor_utils::VectorVectorCwiseProduct(forget_gate_scratch,
                                           cell_state->data.f, n_batch * n_cell,
                                           cell_state->data.f);
    tensor_utils::ApplyActivationToVector(cell_scratch, n_batch * n_cell,
                                          params->activation, cell_scratch);
    if (use_cifg) {
      tensor_utils::Sub1Vector(forget_gate_scratch, n_batch * n_cell,
                               forget_gate_scratch);
      tensor_utils::VectorVectorCwiseProductAccumulate(
          cell_scratch, forget_gate_scratch, n_batch * n_cell,
          cell_state->data.f);
    } else {
      tensor_utils::VectorVectorCwiseProductAccumulate(
          cell_scratch, input_gate_scratch, n_batch * n_cell,
          cell_state->data.f);
    }
    if (params->cell_clip > 0.0) {
      tensor_utils::ClipVector(cell_state->data.f, n_batch * n_cell,
                               params->cell_clip, cell_state->data.f);
    }

    // For each batch and cell: update the output gate.
    if (use_peephole) {
      tensor_utils::VectorBatchVectorCwiseProductAccumulate(
          cell_to_output_weights->data.f, n_cell, cell_state->data.f, n_batch,
          output_gate_scratch);
    }
    tensor_utils::ApplySigmoidToVector(output_gate_scratch, n_batch * n_cell,
                                       output_gate_scratch);
    tensor_utils::ApplyActivationToVector(cell_state->data.f, n_batch * n_cell,
                                          params->activation, cell_scratch);
    tensor_utils::VectorVectorCwiseProduct(output_gate_scratch, cell_scratch,
                                           n_batch * n_cell,
                                           output_gate_scratch);

    // For each batch: update the projection and output_state.
    const bool use_projection_weight = (projection_weights != nullptr);
    const bool use_projection_bias = (projection_bias != nullptr);
    float* output_ptr_time = output->data.f + t * n_batch * n_output;
    if (use_projection_weight) {
      if (use_projection_bias) {
        tensor_utils::VectorBatchVectorAssign(projection_bias->data.f, n_output,
                                              n_batch, output_ptr_time);
      } else {
        tensor_utils::ZeroVector(output_ptr_time, n_batch * n_output);
      }
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          projection_weights->data.f, n_output, n_cell, output_gate_scratch,
          n_batch, output_ptr_time, /*result_stride=*/1);
      if (params->proj_clip > 0.0) {
        tensor_utils::ClipVector(output_ptr_time, n_batch * n_output,
                                 params->proj_clip, output_ptr_time);
      }
    } else {
      tensor_utils::CopyVector(output_gate_scratch, n_batch * n_output,
                               output_ptr_time);
    }
    tensor_utils::CopyVector(output_ptr_time, n_batch * n_output,
                             output_state->data.f);
  }
  return kTfLiteOk;
}

}  // namespace unidirectional_sequence_lstm

TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_LSTM() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 unidirectional_sequence_lstm::Prepare,
                                 unidirectional_sequence_lstm::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
