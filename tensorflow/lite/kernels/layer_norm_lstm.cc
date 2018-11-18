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

// Layer Normalization LSTM op that applies normalization by mean and standard
// deviation to the activation of the LSTM layers. Please see
// https://arxiv.org/abs/1607.06450 for details.
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace layer_norm_lstm {

// Struct to hold Layer Norm LSTM option data.
struct OpData {
  TfLiteFusedActivation activation;
  float cell_clip;
  float proj_clip;
  int scratch_tensor_index;
};

// Input Tensors of size {n_batch, n_input}
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

// Layer norm weights tensors of size {n_cell}, representing a diagonal matrix.
constexpr int kInputLayerNormWeightsTensor = 12;
constexpr int kForgetLayerNormWeightsTensor = 13;
constexpr int kCellLayerNormWeightsTensor = 14;
constexpr int kOutputLayerNormWeightsTensor = 15;

// Gates bias tensors of size {n_cell}
constexpr int kInputGateBiasTensor = 16;  // Optional
constexpr int kForgetGateBiasTensor = 17;
constexpr int kCellGateBiasTensor = 18;
constexpr int kOutputGateBiasTensor = 19;

// Projection weight tensor of size {n_output, n_cell}
constexpr int kProjectionWeightsTensor = 20;  // Optional
// Projection bias tensor of size {n_output}
constexpr int kProjectionBiasTensor = 21;  // Optional

// State tensors.
constexpr int kInputActivationStateTensor = 22;
constexpr int kInputCellStateTensor = 23;

// Output tensor.
constexpr int kOutputTensor = 0;

// Total number of scratch tensors for hybrid Op.
constexpr int kTensorsToAdd = 7;

// Small float to avoid divergence during calculation of deviation.
const float kLayerNormEpsilon = 1e-8;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;

  // Turn custom option data into flexbuffer map format.
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  // Get activation function, cell_clip and proj_clip from the flexbuffer.
  // TODO(b/113824099): make activation more generic.
  assert(m["fused_activation_function"].ToString() == "TANH");
  data->activation = kTfLiteActTanh;
  data->cell_clip = m["cell_clip"].AsFloat();
  data->proj_clip = m["proj_clip"].AsFloat();

  // Populate scratch_tensor_index.
  context->AddTensors(context, /*tensors_to_add=*/kTensorsToAdd,
                      &data->scratch_tensor_index);
  return data;
}

// Check that input tensor dimensions matches with each other.
TfLiteStatus CheckInputTensorDimensions(TfLiteContext* context,
                                        TfLiteNode* node, int n_input,
                                        int n_output, int n_cell) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  // Making sure clipping parameters have valid values.
  // == 0 means no clipping
  //  > 0 means clipping
  TF_LITE_ENSURE(context, op_data->cell_clip >= 0);
  TF_LITE_ENSURE(context, op_data->proj_clip >= 0);

  const TfLiteTensor* input_to_input_weights =
      GetOptionalInputTensor(context, node, kInputToInputWeightsTensor);
  if (input_to_input_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->data[1], n_input);
  }

  const TfLiteTensor* input_to_forget_weights =
      GetInput(context, node, kInputToForgetWeightsTensor);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->data[1], n_input);

  const TfLiteTensor* input_to_cell_weights =
      GetInput(context, node, kInputToCellWeightsTensor);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->data[1], n_input);

  const TfLiteTensor* recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, kRecurrentToInputWeightsTensor);
  if (recurrent_to_input_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->data[0],
                      n_cell);
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->data[1],
                      n_output);
  }

  const TfLiteTensor* recurrent_to_forget_weights =
      GetInput(context, node, kRecurrentToForgetWeightsTensor);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->data[0],
                    n_cell);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->data[1],
                    n_output);

  const TfLiteTensor* recurrent_to_cell_weights =
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

  const TfLiteTensor* cell_to_input_weights =
      GetOptionalInputTensor(context, node, kCellToInputWeightsTensor);
  if (cell_to_input_weights) {
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->data[0], n_cell);
  }

  const TfLiteTensor* cell_to_forget_weights =
      GetOptionalInputTensor(context, node, kCellToForgetWeightsTensor);
  if (cell_to_forget_weights) {
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->data[0], n_cell);
  }

  const TfLiteTensor* cell_to_output_weights =
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

  // Making sure layer norm weights are not null and have the right dimension.
  const TfLiteTensor* input_layer_norm_weights =
      GetInput(context, node, kInputLayerNormWeightsTensor);
  TF_LITE_ENSURE(context, input_layer_norm_weights != nullptr);
  TF_LITE_ENSURE_EQ(context, input_layer_norm_weights->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, input_layer_norm_weights->dims->data[0], n_cell);

  const TfLiteTensor* forget_layer_norm_weights =
      GetInput(context, node, kForgetLayerNormWeightsTensor);
  TF_LITE_ENSURE(context, forget_layer_norm_weights != nullptr);
  TF_LITE_ENSURE_EQ(context, forget_layer_norm_weights->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, forget_layer_norm_weights->dims->data[0], n_cell);

  const TfLiteTensor* cell_layer_norm_weights =
      GetInput(context, node, kCellLayerNormWeightsTensor);
  TF_LITE_ENSURE(context, cell_layer_norm_weights != nullptr);
  TF_LITE_ENSURE_EQ(context, cell_layer_norm_weights->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, cell_layer_norm_weights->dims->data[0], n_cell);

  const TfLiteTensor* output_layer_norm_weights =
      GetInput(context, node, kOutputLayerNormWeightsTensor);
  TF_LITE_ENSURE(context, output_layer_norm_weights != nullptr);
  TF_LITE_ENSURE_EQ(context, output_layer_norm_weights->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, output_layer_norm_weights->dims->data[0], n_cell);

  // Make sure the input gate bias is present only when not a CIFG-LSTM.
  const TfLiteTensor* input_gate_bias =
      GetOptionalInputTensor(context, node, kInputGateBiasTensor);
  if (use_cifg) {
    TF_LITE_ENSURE_EQ(context, input_gate_bias, nullptr);
  } else {
    TF_LITE_ENSURE_EQ(context, input_gate_bias->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, input_gate_bias->dims->data[0], n_cell);
  }

  const TfLiteTensor* forget_gate_bias =
      GetInput(context, node, kForgetGateBiasTensor);
  TF_LITE_ENSURE_EQ(context, forget_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, forget_gate_bias->dims->data[0], n_cell);

  const TfLiteTensor* cell_bias = GetInput(context, node, kCellGateBiasTensor);
  TF_LITE_ENSURE_EQ(context, cell_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, cell_bias->dims->data[0], n_cell);

  const TfLiteTensor* output_gate_bias =
      GetInput(context, node, kOutputGateBiasTensor);
  TF_LITE_ENSURE_EQ(context, output_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, output_gate_bias->dims->data[0], n_cell);

  const TfLiteTensor* projection_weights =
      GetOptionalInputTensor(context, node, kProjectionWeightsTensor);
  if (projection_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[0], n_output);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[1], n_cell);
  }

  const TfLiteTensor* projection_bias =
      GetOptionalInputTensor(context, node, kProjectionBiasTensor);
  if (projection_bias != nullptr) {
    TF_LITE_ENSURE_EQ(context, projection_bias->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, projection_bias->dims->data[0], n_output);
  }

  // Making sure the projection tensors are consistent:
  // 1) If projection weight is not present, then projection bias should not be
  // present.
  // 2) If projection weight is present, then projection bias is optional.
  const bool projection_tensors_consistent =
      ((projection_weights != nullptr) || (projection_bias == nullptr));
  TF_LITE_ENSURE(context, projection_tensors_consistent == true);

  return kTfLiteOk;
}

// Resize the output, state tensors based on the sizes of the input tensors.
// Allocate a temporary scratch tensor. Also check that the sizes of the input
// tensors match each other.
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 24);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Inferring batch size, number of outputs and number of cells from the
  // input tensors.
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE(context, input->dims->size > 1);
  const int n_batch = input->dims->data[0];
  const int n_input = input->dims->data[1];

  const TfLiteTensor* input_to_output_weights =
      GetInput(context, node, kInputToOutputWeightsTensor);
  const int n_cell = input_to_output_weights->dims->data[0];
  TF_LITE_ENSURE_EQ(context, input_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_output_weights->dims->data[1], n_input);

  const TfLiteTensor* recurrent_to_output_weights =
      GetInput(context, node, kRecurrentToOutputWeightsTensor);
  TF_LITE_ENSURE_EQ(context, recurrent_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_output_weights->dims->data[0],
                    n_cell);
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Check that input tensor dimensions matches with each other.
  TF_LITE_ENSURE_OK(context, CheckInputTensorDimensions(context, node, n_input,
                                                        n_output, n_cell));

  // Get the pointer to output, activation_state and cell_state tensors.
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  const TfLiteTensor* activation_state =
      GetInput(context, node, kInputActivationStateTensor);
  const TfLiteTensor* cell_state =
      GetInput(context, node, kInputCellStateTensor);

  // Check the shape of input state tensors.
  // These tensor may be 1D or 2D. It's fine as long as the total size is
  // correct.
  TF_LITE_ENSURE_EQ(context, NumElements(activation_state), n_batch * n_output);
  TF_LITE_ENSURE_EQ(context, NumElements(cell_state), n_batch * n_cell);
  // Resize the output tensors.
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(2);
  output_size->data[0] = n_batch;
  output_size->data[1] = n_output;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size));

  // The weights are of consistent type, so it suffices to check one.
  const bool is_hybrid_op = (input_to_output_weights->type == kTfLiteUInt8 &&
                             input->type == kTfLiteFloat32);

  TfLiteIntArrayFree(node->temporaries);
  if (is_hybrid_op) {
    node->temporaries = TfLiteIntArrayCreate(7);
  } else {
    node->temporaries = TfLiteIntArrayCreate(1);
  }
  node->temporaries->data[0] = op_data->scratch_tensor_index;

  // Create a scratch buffer tensor.
  TfLiteTensor* scratch_buffer = GetTemporary(context, node, /*index=*/0);
  scratch_buffer->type = input->type;
  scratch_buffer->allocation_type = kTfLiteArenaRw;

  const TfLiteTensor* input_to_input_weights =
      GetOptionalInputTensor(context, node, kInputToInputWeightsTensor);
  const bool use_cifg = (input_to_input_weights == nullptr);
  TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(2);
  scratch_buffer_size->data[0] = n_batch;
  if (use_cifg) {
    // Reserving space for Cell, Forget, Output gates
    scratch_buffer_size->data[1] = n_cell * 3;
  } else {
    // Reserving space for Input, Cell, Forget, Output gates
    scratch_buffer_size->data[1] = n_cell * 4;
  }
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                   scratch_buffer_size));

  if (is_hybrid_op) {
    // Allocate temporary tensors to store quantized values of input,
    // activation_state and cell_state tensors.
    node->temporaries->data[1] = op_data->scratch_tensor_index + 1;
    TfLiteTensor* input_quantized = GetTemporary(context, node, /*index=*/1);
    input_quantized->type = kTfLiteUInt8;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }
    node->temporaries->data[2] = op_data->scratch_tensor_index + 2;
    TfLiteTensor* activation_state_quantized =
        GetTemporary(context, node, /*index=*/2);
    activation_state_quantized->type = kTfLiteUInt8;
    activation_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(activation_state_quantized->dims,
                             activation_state->dims)) {
      TfLiteIntArray* activation_state_quantized_size =
          TfLiteIntArrayCopy(activation_state->dims);
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, activation_state_quantized,
                                         activation_state_quantized_size));
    }
    node->temporaries->data[3] = op_data->scratch_tensor_index + 3;
    TfLiteTensor* cell_state_quantized =
        GetTemporary(context, node, /*index=*/3);
    cell_state_quantized->type = kTfLiteUInt8;
    cell_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(cell_state_quantized->dims, cell_state->dims)) {
      TfLiteIntArray* cell_state_quantized_size =
          TfLiteIntArrayCopy(cell_state->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, cell_state_quantized,
                                              cell_state_quantized_size));
    }

    // Allocate temporary tensors to store scaling factors and product scaling
    // factors. The latter is a convenience storage which allows to quantize
    // a vector once (which produces the scaling factors) and multiply it with
    // different matrices (which requires multiplying the scaling factors with
    // the scaling factor of the matrix).
    node->temporaries->data[4] = op_data->scratch_tensor_index + 4;
    TfLiteTensor* scaling_factors = GetTemporary(context, node, /*index=*/4);
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;
    int scaling_dims[1] = {n_batch};
    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
      TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
      scaling_factors_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }
    node->temporaries->data[5] = op_data->scratch_tensor_index + 5;
    TfLiteTensor* prod_scaling_factors =
        GetTemporary(context, node, /*index=*/5);
    prod_scaling_factors->type = kTfLiteFloat32;
    prod_scaling_factors->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(prod_scaling_factors->dims, 1,
                                   scaling_dims)) {
      TfLiteIntArray* prod_scaling_factors_size = TfLiteIntArrayCreate(1);
      prod_scaling_factors_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, prod_scaling_factors,
                                              prod_scaling_factors_size));
    }

    // Allocate a temporary tensor to store the recovered weights. Since
    // this is used for diagonal matrices, only need to store n_cell values.
    node->temporaries->data[6] = op_data->scratch_tensor_index + 6;
    TfLiteTensor* recovered_weights = GetTemporary(context, node, /*index=*/6);
    recovered_weights->type = kTfLiteFloat32;
    recovered_weights->allocation_type = kTfLiteArenaRw;
    int recovered_dims[1] = {n_cell};
    if (!TfLiteIntArrayEqualsArray(recovered_weights->dims, 1,
                                   recovered_dims)) {
      TfLiteIntArray* recovered_weights_size = TfLiteIntArrayCreate(1);
      recovered_weights_size->data[0] = n_cell;
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, recovered_weights,
                                              recovered_weights_size));
    }
  }
  return kTfLiteOk;
}

void LayerNormLstmStep(
    const float* input_ptr_batch, const float* input_to_input_weights_ptr,
    const float* input_to_forget_weights_ptr,
    const float* input_to_cell_weights_ptr,
    const float* input_to_output_weights_ptr,
    const float* recurrent_to_input_weights_ptr,
    const float* recurrent_to_forget_weights_ptr,
    const float* recurrent_to_cell_weights_ptr,
    const float* recurrent_to_output_weights_ptr,
    const float* cell_to_input_weights_ptr,
    const float* cell_to_forget_weights_ptr,
    const float* cell_to_output_weights_ptr,
    const float* input_layer_norm_weight_ptr,
    const float* forget_layer_norm_weight_ptr,
    const float* cell_layer_norm_weight_ptr,
    const float* output_layer_norm_weight_ptr, const float* input_gate_bias_ptr,
    const float* forget_gate_bias_ptr, const float* cell_bias_ptr,
    const float* output_gate_bias_ptr, const float* projection_weights_ptr,
    const float* projection_bias_ptr, float cell_clip, float proj_clip,
    const TfLiteFusedActivation& activation, int n_batch, int n_cell,
    int n_input, int n_output, float* output_state_ptr, float* cell_state_ptr,
    float* input_gate_scratch, float* forget_gate_scratch, float* cell_scratch,
    float* output_gate_scratch, float* output_ptr_batch) {
  // Since we have already checked that weights are all there or none, we can
  // check the existense of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights_ptr == nullptr);
  const bool use_peephole = (cell_to_output_weights_ptr != nullptr);

  // Initialize scratch buffers with 0.
  if (!use_cifg) {
    tensor_utils::ZeroVector(input_gate_scratch, n_cell * n_batch);
  }
  tensor_utils::ZeroVector(forget_gate_scratch, n_cell * n_batch);
  tensor_utils::ZeroVector(cell_scratch, n_cell * n_batch);
  tensor_utils::ZeroVector(output_gate_scratch, n_cell * n_batch);

  // For each batch and cell: compute input_weight * input.
  if (!use_cifg) {
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        input_to_input_weights_ptr, n_cell, n_input, input_ptr_batch, n_batch,
        input_gate_scratch, /*result_stride=*/1);
  }

  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      input_to_forget_weights_ptr, n_cell, n_input, input_ptr_batch, n_batch,
      forget_gate_scratch, /*result_stride=*/1);
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      input_to_cell_weights_ptr, n_cell, n_input, input_ptr_batch, n_batch,
      cell_scratch, /*result_stride=*/1);
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      input_to_output_weights_ptr, n_cell, n_input, input_ptr_batch, n_batch,
      output_gate_scratch, /*result_stride=*/1);

  // For each batch and cell: compute recurrent_weight * output_state.
  if (!use_cifg) {
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        recurrent_to_input_weights_ptr, n_cell, n_output, output_state_ptr,
        n_batch, input_gate_scratch, /*result_stride=*/1);
  }
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      recurrent_to_forget_weights_ptr, n_cell, n_output, output_state_ptr,
      n_batch, forget_gate_scratch,
      /*result_stride=*/1);
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      recurrent_to_cell_weights_ptr, n_cell, n_output, output_state_ptr,
      n_batch, cell_scratch, /*result_stride=*/1);
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      recurrent_to_output_weights_ptr, n_cell, n_output, output_state_ptr,
      n_batch, output_gate_scratch,
      /*result_stride=*/1);

  // For each batch and cell: update input gate.
  if (!use_cifg) {
    if (use_peephole) {
      tensor_utils::VectorBatchVectorCwiseProductAccumulate(
          cell_to_input_weights_ptr, n_cell, cell_state_ptr, n_batch,
          input_gate_scratch);
    }
    tensor_utils::MeanStddevNormalization(input_gate_scratch,
                                          input_gate_scratch, n_cell, n_batch,
                                          kLayerNormEpsilon);
    tensor_utils::VectorBatchVectorCwiseProduct(input_layer_norm_weight_ptr,
                                                n_cell, input_gate_scratch,
                                                n_batch, input_gate_scratch);
    tensor_utils::VectorBatchVectorAdd(input_gate_bias_ptr, n_cell, n_batch,
                                       input_gate_scratch);
    tensor_utils::ApplySigmoidToVector(input_gate_scratch, n_cell * n_batch,
                                       input_gate_scratch);
  }

  // For each batch and cell: update forget gate.
  if (use_peephole) {
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        cell_to_forget_weights_ptr, n_cell, cell_state_ptr, n_batch,
        forget_gate_scratch);
  }
  tensor_utils::MeanStddevNormalization(forget_gate_scratch,
                                        forget_gate_scratch, n_cell, n_batch,
                                        kLayerNormEpsilon);
  tensor_utils::VectorBatchVectorCwiseProduct(forget_layer_norm_weight_ptr,
                                              n_cell, forget_gate_scratch,
                                              n_batch, forget_gate_scratch);
  tensor_utils::VectorBatchVectorAdd(forget_gate_bias_ptr, n_cell, n_batch,
                                     forget_gate_scratch);
  tensor_utils::ApplySigmoidToVector(forget_gate_scratch, n_cell * n_batch,
                                     forget_gate_scratch);

  // For each batch and cell: update the cell.
  tensor_utils::MeanStddevNormalization(cell_scratch, cell_scratch, n_cell,
                                        n_batch, kLayerNormEpsilon);
  tensor_utils::VectorBatchVectorCwiseProduct(
      cell_layer_norm_weight_ptr, n_cell, cell_scratch, n_batch, cell_scratch);
  tensor_utils::VectorBatchVectorAdd(cell_bias_ptr, n_cell, n_batch,
                                     cell_scratch);
  tensor_utils::VectorVectorCwiseProduct(forget_gate_scratch, cell_state_ptr,
                                         n_batch * n_cell, cell_state_ptr);
  tensor_utils::ApplyActivationToVector(cell_scratch, n_batch * n_cell,
                                        activation, cell_scratch);
  if (use_cifg) {
    tensor_utils::Sub1Vector(forget_gate_scratch, n_batch * n_cell,
                             forget_gate_scratch);
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_scratch, forget_gate_scratch, n_batch * n_cell, cell_state_ptr);
  } else {
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_scratch, input_gate_scratch, n_batch * n_cell, cell_state_ptr);
  }
  if (cell_clip > 0.0) {
    tensor_utils::ClipVector(cell_state_ptr, n_batch * n_cell, cell_clip,
                             cell_state_ptr);
  }

  // For each batch and cell: update the output gate.
  if (use_peephole) {
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        cell_to_output_weights_ptr, n_cell, cell_state_ptr, n_batch,
        output_gate_scratch);
  }
  tensor_utils::MeanStddevNormalization(output_gate_scratch,
                                        output_gate_scratch, n_cell, n_batch,
                                        kLayerNormEpsilon);
  tensor_utils::VectorBatchVectorCwiseProduct(output_layer_norm_weight_ptr,
                                              n_cell, output_gate_scratch,
                                              n_batch, output_gate_scratch);
  tensor_utils::VectorBatchVectorAdd(output_gate_bias_ptr, n_cell, n_batch,
                                     output_gate_scratch);
  tensor_utils::ApplySigmoidToVector(output_gate_scratch, n_batch * n_cell,
                                     output_gate_scratch);
  tensor_utils::ApplyActivationToVector(cell_state_ptr, n_batch * n_cell,
                                        activation, cell_scratch);
  tensor_utils::VectorVectorCwiseProduct(output_gate_scratch, cell_scratch,
                                         n_batch * n_cell, output_gate_scratch);

  // For each batch: update the projection and output_state.
  const bool use_projection_weight = (projection_weights_ptr != nullptr);
  const bool use_projection_bias = (projection_bias_ptr != nullptr);
  if (use_projection_weight) {
    if (use_projection_bias) {
      tensor_utils::VectorBatchVectorAssign(projection_bias_ptr, n_output,
                                            n_batch, output_ptr_batch);
    } else {
      tensor_utils::ZeroVector(output_ptr_batch, n_batch * n_output);
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        projection_weights_ptr, n_output, n_cell, output_gate_scratch, n_batch,
        output_ptr_batch, /*result_stride=*/1);
    if (proj_clip > 0.0) {
      tensor_utils::ClipVector(output_ptr_batch, n_batch * n_output, proj_clip,
                               output_ptr_batch);
    }
  } else {
    tensor_utils::CopyVector(output_gate_scratch, n_batch * n_output,
                             output_ptr_batch);
  }
  tensor_utils::CopyVector(output_ptr_batch, n_batch * n_output,
                           output_state_ptr);
}

void LayerNormLstmStep(
    const float* input_ptr_batch, const int8_t* input_to_input_weights_ptr,
    float input_to_input_weights_scale,
    const int8_t* input_to_forget_weights_ptr,
    float input_to_forget_weights_scale,
    const int8_t* input_to_cell_weights_ptr, float input_to_cell_weights_scale,
    const int8_t* input_to_output_weights_ptr,
    float input_to_output_weights_scale,
    const int8_t* recurrent_to_input_weights_ptr,
    float recurrent_to_input_weights_scale,
    const int8_t* recurrent_to_forget_weights_ptr,
    float recurrent_to_forget_weights_scale,
    const int8_t* recurrent_to_cell_weights_ptr,
    float recurrent_to_cell_weights_scale,
    const int8_t* recurrent_to_output_weights_ptr,
    float recurrent_to_output_weights_scale,
    const int8_t* cell_to_input_weights_ptr, float cell_to_input_weights_scale,
    const int8_t* cell_to_forget_weights_ptr,
    float cell_to_forget_weights_scale,
    const int8_t* cell_to_output_weights_ptr,
    float cell_to_output_weights_scale,
    const float* input_layer_norm_weight_ptr,
    const float* forget_layer_norm_weight_ptr,
    const float* cell_layer_norm_weight_ptr,
    const float* output_layer_norm_weight_ptr, const float* input_gate_bias_ptr,
    const float* forget_gate_bias_ptr, const float* cell_bias_ptr,
    const float* output_gate_bias_ptr, const int8_t* projection_weights_ptr,
    float projection_weights_scale, const float* projection_bias_ptr,
    float cell_clip, float proj_clip, const TfLiteFusedActivation& activation,
    int n_batch, int n_cell, int n_input, int n_output,
    float* input_gate_scratch, float* forget_gate_scratch, float* cell_scratch,
    float* output_gate_scratch, float* scaling_factors,
    float* product_scaling_factors, float* recovered_weights,
    int8_t* quantized_input_ptr_batch, int8_t* quantized_output_state_ptr,
    int8_t* quantized_cell_state_ptr, float* output_state_ptr,
    float* cell_state_ptr, float* output_ptr_batch) {
  // Since we have already checked that weights are all there or none, we can
  // check the existense of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights_ptr == nullptr);
  const bool use_peephole = (cell_to_output_weights_ptr != nullptr);

  // Initialize scratch buffers with 0.
  if (!use_cifg) {
    tensor_utils::ZeroVector(input_gate_scratch, n_cell * n_batch);
  }
  tensor_utils::ZeroVector(forget_gate_scratch, n_cell * n_batch);
  tensor_utils::ZeroVector(cell_scratch, n_cell * n_batch);
  tensor_utils::ZeroVector(output_gate_scratch, n_cell * n_batch);

  if (!tensor_utils::IsZeroVector(input_ptr_batch, n_batch * n_input)) {
    // Save quantization and matmul computation for all zero input.
    float unused_min, unused_max;
    for (int b = 0; b < n_batch; ++b) {
      const int offset = b * n_input;
      tensor_utils::SymmetricQuantizeFloats(
          input_ptr_batch + offset, n_input, quantized_input_ptr_batch + offset,
          &unused_min, &unused_max, &scaling_factors[b]);
    }
    // For each batch and cell: compute input_weight * input.
    if (!use_cifg) {
      for (int b = 0; b < n_batch; ++b) {
        product_scaling_factors[b] =
            scaling_factors[b] * input_to_input_weights_scale;
      }
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          input_to_input_weights_ptr, n_cell, n_input,
          quantized_input_ptr_batch, product_scaling_factors, n_batch,
          input_gate_scratch, /*result_stride=*/1);
    }

    for (int b = 0; b < n_batch; ++b) {
      product_scaling_factors[b] =
          scaling_factors[b] * input_to_forget_weights_scale;
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        input_to_forget_weights_ptr, n_cell, n_input, quantized_input_ptr_batch,
        product_scaling_factors, n_batch, forget_gate_scratch,
        /*result_stride=*/1);

    for (int b = 0; b < n_batch; ++b) {
      product_scaling_factors[b] =
          scaling_factors[b] * input_to_cell_weights_scale;
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        input_to_cell_weights_ptr, n_cell, n_input, quantized_input_ptr_batch,
        product_scaling_factors, n_batch, cell_scratch, /*result_stride=*/1);

    for (int b = 0; b < n_batch; ++b) {
      product_scaling_factors[b] =
          scaling_factors[b] * input_to_output_weights_scale;
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        input_to_output_weights_ptr, n_cell, n_input, quantized_input_ptr_batch,
        product_scaling_factors, n_batch, output_gate_scratch,
        /*result_stride=*/1);
  }

  if (!tensor_utils::IsZeroVector(output_state_ptr, n_batch * n_output)) {
    // Save quantization and matmul computation for all zero input.
    float unused_min, unused_max;
    for (int b = 0; b < n_batch; ++b) {
      const int offset = b * n_output;
      tensor_utils::SymmetricQuantizeFloats(output_state_ptr + offset, n_output,
                                            quantized_output_state_ptr + offset,
                                            &unused_min, &unused_max,
                                            &scaling_factors[b]);
    }
    // For each batch and cell: compute recurrent_weight * output_state.
    if (!use_cifg) {
      for (int b = 0; b < n_batch; ++b) {
        product_scaling_factors[b] =
            scaling_factors[b] * recurrent_to_input_weights_scale;
      }
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          recurrent_to_input_weights_ptr, n_cell, n_output,
          quantized_output_state_ptr, product_scaling_factors, n_batch,
          input_gate_scratch, /*result_stride=*/1);
    }

    for (int b = 0; b < n_batch; ++b) {
      product_scaling_factors[b] =
          scaling_factors[b] * recurrent_to_forget_weights_scale;
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        recurrent_to_forget_weights_ptr, n_cell, n_output,
        quantized_output_state_ptr, product_scaling_factors, n_batch,
        forget_gate_scratch, /*result_stride=*/1);

    for (int b = 0; b < n_batch; ++b) {
      product_scaling_factors[b] =
          scaling_factors[b] * recurrent_to_cell_weights_scale;
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        recurrent_to_cell_weights_ptr, n_cell, n_output,
        quantized_output_state_ptr, product_scaling_factors, n_batch,
        cell_scratch, /*result_stride=*/1);

    for (int b = 0; b < n_batch; ++b) {
      product_scaling_factors[b] =
          scaling_factors[b] * recurrent_to_output_weights_scale;
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        recurrent_to_output_weights_ptr, n_cell, n_output,
        quantized_output_state_ptr, product_scaling_factors, n_batch,
        output_gate_scratch, /*result_stride=*/1);
  }

  // Save quantization and matmul computation for all zero input.
  bool is_cell_state_all_zeros =
      tensor_utils::IsZeroVector(cell_state_ptr, n_batch * n_cell);

  // For each batch and cell: update input gate.
  if (!use_cifg) {
    if (use_peephole && !is_cell_state_all_zeros) {
      tensor_utils::VectorScalarMultiply(cell_to_input_weights_ptr, n_cell,
                                         cell_to_input_weights_scale,
                                         recovered_weights);
      tensor_utils::VectorBatchVectorCwiseProductAccumulate(
          recovered_weights, n_cell, cell_state_ptr, n_batch,
          input_gate_scratch);
    }
    tensor_utils::MeanStddevNormalization(input_gate_scratch,
                                          input_gate_scratch, n_cell, n_batch,
                                          kLayerNormEpsilon);
    tensor_utils::VectorBatchVectorCwiseProduct(input_layer_norm_weight_ptr,
                                                n_cell, input_gate_scratch,
                                                n_batch, input_gate_scratch);
    tensor_utils::VectorBatchVectorAdd(input_gate_bias_ptr, n_cell, n_batch,
                                       input_gate_scratch);
    tensor_utils::ApplySigmoidToVector(input_gate_scratch, n_cell * n_batch,
                                       input_gate_scratch);
  }

  // For each batch and cell: update forget gate.
  if (use_peephole && !is_cell_state_all_zeros) {
    tensor_utils::VectorScalarMultiply(cell_to_forget_weights_ptr, n_cell,
                                       cell_to_forget_weights_scale,
                                       recovered_weights);
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        recovered_weights, n_cell, cell_state_ptr, n_batch,
        forget_gate_scratch);
  }
  tensor_utils::MeanStddevNormalization(forget_gate_scratch,
                                        forget_gate_scratch, n_cell, n_batch,
                                        kLayerNormEpsilon);
  tensor_utils::VectorBatchVectorCwiseProduct(forget_layer_norm_weight_ptr,
                                              n_cell, forget_gate_scratch,
                                              n_batch, forget_gate_scratch);
  tensor_utils::VectorBatchVectorAdd(forget_gate_bias_ptr, n_cell, n_batch,
                                     forget_gate_scratch);
  tensor_utils::ApplySigmoidToVector(forget_gate_scratch, n_cell * n_batch,
                                     forget_gate_scratch);

  // For each batch and cell: update the cell.
  tensor_utils::MeanStddevNormalization(cell_scratch, cell_scratch, n_cell,
                                        n_batch, kLayerNormEpsilon);
  tensor_utils::VectorBatchVectorCwiseProduct(
      cell_layer_norm_weight_ptr, n_cell, cell_scratch, n_batch, cell_scratch);
  tensor_utils::VectorBatchVectorAdd(cell_bias_ptr, n_cell, n_batch,
                                     cell_scratch);
  tensor_utils::VectorVectorCwiseProduct(forget_gate_scratch, cell_state_ptr,
                                         n_batch * n_cell, cell_state_ptr);
  tensor_utils::ApplyActivationToVector(cell_scratch, n_batch * n_cell,
                                        activation, cell_scratch);
  if (use_cifg) {
    tensor_utils::Sub1Vector(forget_gate_scratch, n_batch * n_cell,
                             forget_gate_scratch);
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_scratch, forget_gate_scratch, n_batch * n_cell, cell_state_ptr);
  } else {
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_scratch, input_gate_scratch, n_batch * n_cell, cell_state_ptr);
  }
  if (cell_clip > 0.0) {
    tensor_utils::ClipVector(cell_state_ptr, n_batch * n_cell, cell_clip,
                             cell_state_ptr);
  }

  is_cell_state_all_zeros =
      tensor_utils::IsZeroVector(cell_state_ptr, n_batch * n_cell);
  // For each batch and cell: update the output gate.
  if (use_peephole && !is_cell_state_all_zeros) {
    tensor_utils::VectorScalarMultiply(cell_to_output_weights_ptr, n_cell,
                                       cell_to_output_weights_scale,
                                       recovered_weights);
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        recovered_weights, n_cell, cell_state_ptr, n_batch,
        output_gate_scratch);
  }
  tensor_utils::MeanStddevNormalization(output_gate_scratch,
                                        output_gate_scratch, n_cell, n_batch,
                                        kLayerNormEpsilon);
  tensor_utils::VectorBatchVectorCwiseProduct(output_layer_norm_weight_ptr,
                                              n_cell, output_gate_scratch,
                                              n_batch, output_gate_scratch);
  tensor_utils::VectorBatchVectorAdd(output_gate_bias_ptr, n_cell, n_batch,
                                     output_gate_scratch);
  tensor_utils::ApplySigmoidToVector(output_gate_scratch, n_batch * n_cell,
                                     output_gate_scratch);
  tensor_utils::ApplyActivationToVector(cell_state_ptr, n_batch * n_cell,
                                        activation, cell_scratch);
  tensor_utils::VectorVectorCwiseProduct(output_gate_scratch, cell_scratch,
                                         n_batch * n_cell, output_gate_scratch);

  // For each batch: update the projection and output_state.
  const bool use_projection_weight = (projection_weights_ptr != nullptr);
  const bool use_projection_bias = (projection_bias_ptr != nullptr);
  if (use_projection_weight) {
    if (use_projection_bias) {
      tensor_utils::VectorBatchVectorAssign(projection_bias_ptr, n_output,
                                            n_batch, output_ptr_batch);
    } else {
      tensor_utils::ZeroVector(output_ptr_batch, n_batch * n_output);
    }
    if (!tensor_utils::IsZeroVector(output_gate_scratch, n_batch * n_cell)) {
      // Save quantization and matmul computation for all zero input.
      float unused_min, unused_max;
      for (int b = 0; b < n_batch; ++b) {
        const int offset = b * n_cell;
        tensor_utils::SymmetricQuantizeFloats(
            output_gate_scratch + offset, n_cell,
            quantized_cell_state_ptr + offset, &unused_min, &unused_max,
            &scaling_factors[b]);
      }
      for (int b = 0; b < n_batch; ++b) {
        product_scaling_factors[b] =
            scaling_factors[b] * projection_weights_scale;
      }
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          projection_weights_ptr, n_output, n_cell, quantized_cell_state_ptr,
          product_scaling_factors, n_batch, output_ptr_batch,
          /*result_stride=*/1);
    }
    if (proj_clip > 0.0) {
      tensor_utils::ClipVector(output_ptr_batch, n_batch * n_output, proj_clip,
                               output_ptr_batch);
    }
  } else {
    tensor_utils::CopyVector(output_gate_scratch, n_batch * n_output,
                             output_ptr_batch);
  }
  tensor_utils::CopyVector(output_ptr_batch, n_batch * n_output,
                           output_state_ptr);
}

// The LayerNormLSTM Op engine.
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
    const TfLiteTensor* cell_to_output_weights,
    const TfLiteTensor* input_layer_norm_weights,
    const TfLiteTensor* forget_layer_norm_weights,
    const TfLiteTensor* cell_layer_norm_weights,
    const TfLiteTensor* output_layer_norm_weights,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    float cell_clip, float proj_clip, const TfLiteFusedActivation& activation,
    TfLiteTensor* scratch_buffer, TfLiteTensor* activation_state,
    TfLiteTensor* cell_state, TfLiteTensor* output) {
  const int n_batch = input->dims->data[0];
  const int n_input = input->dims->data[1];
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

  // Required tensors, pointers are non-null.
  const float* input_ptr_batch = input->data.f;
  const float* input_to_forget_weights_ptr = input_to_forget_weights->data.f;
  const float* input_to_cell_weights_ptr = input_to_cell_weights->data.f;
  const float* input_to_output_weights_ptr = input_to_output_weights->data.f;
  const float* recurrent_to_forget_weights_ptr =
      recurrent_to_forget_weights->data.f;
  const float* recurrent_to_cell_weights_ptr =
      recurrent_to_cell_weights->data.f;
  const float* recurrent_to_output_weights_ptr =
      recurrent_to_output_weights->data.f;
  const float* input_layer_norm_weight_ptr = input_layer_norm_weights->data.f;
  const float* forget_layer_norm_weight_ptr = forget_layer_norm_weights->data.f;
  const float* cell_layer_norm_weight_ptr = cell_layer_norm_weights->data.f;
  const float* output_layer_norm_weight_ptr = output_layer_norm_weights->data.f;
  const float* forget_gate_bias_ptr = forget_gate_bias->data.f;
  const float* cell_bias_ptr = cell_bias->data.f;
  const float* output_gate_bias_ptr = output_gate_bias->data.f;

  float* activation_state_ptr = activation_state->data.f;
  float* cell_state_ptr = cell_state->data.f;
  float* output_ptr_batch = output->data.f;

  LayerNormLstmStep(
      input_ptr_batch, input_to_input_weights_ptr, input_to_forget_weights_ptr,
      input_to_cell_weights_ptr, input_to_output_weights_ptr,
      recurrent_to_input_weights_ptr, recurrent_to_forget_weights_ptr,
      recurrent_to_cell_weights_ptr, recurrent_to_output_weights_ptr,
      cell_to_input_weights_ptr, cell_to_forget_weights_ptr,
      cell_to_output_weights_ptr, input_layer_norm_weight_ptr,
      forget_layer_norm_weight_ptr, cell_layer_norm_weight_ptr,
      output_layer_norm_weight_ptr, input_gate_bias_ptr, forget_gate_bias_ptr,
      cell_bias_ptr, output_gate_bias_ptr, projection_weights_ptr,
      projection_bias_ptr, cell_clip, proj_clip, activation, n_batch, n_cell,
      n_input, n_output, activation_state_ptr, cell_state_ptr,
      input_gate_scratch, forget_gate_scratch, cell_scratch,
      output_gate_scratch, output_ptr_batch);

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
    const TfLiteTensor* cell_to_output_weights,
    const TfLiteTensor* input_layer_norm_weights,
    const TfLiteTensor* forget_layer_norm_weights,
    const TfLiteTensor* cell_layer_norm_weights,
    const TfLiteTensor* output_layer_norm_weights,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    float cell_clip, float proj_clip, const TfLiteFusedActivation& activation,
    TfLiteTensor* scratch_buffer, TfLiteTensor* scaling_factors,
    TfLiteTensor* prod_scaling_factors, TfLiteTensor* recovered_weights,
    TfLiteTensor* input_quantized, TfLiteTensor* activation_state_quantized,
    TfLiteTensor* cell_state_quantized, TfLiteTensor* activation_state,
    TfLiteTensor* cell_state, TfLiteTensor* output) {
  const int n_batch = input->dims->data[0];
  const int n_input = input->dims->data[1];
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
  const float* input_ptr_batch = input->data.f;
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
  const float* input_layer_norm_weight_ptr = input_layer_norm_weights->data.f;
  const float* forget_layer_norm_weight_ptr = forget_layer_norm_weights->data.f;
  const float* cell_layer_norm_weight_ptr = cell_layer_norm_weights->data.f;
  const float* output_layer_norm_weight_ptr = output_layer_norm_weights->data.f;
  const float* forget_gate_bias_ptr = forget_gate_bias->data.f;
  const float* cell_bias_ptr = cell_bias->data.f;
  const float* output_gate_bias_ptr = output_gate_bias->data.f;

  float* activation_state_ptr = activation_state->data.f;
  float* cell_state_ptr = cell_state->data.f;
  float* output_ptr_batch = output->data.f;

  // Temporary storage for quantized values and scaling factors.
  int8_t* quantized_input_ptr =
      reinterpret_cast<int8_t*>(input_quantized->data.uint8);
  int8_t* quantized_activation_state_ptr =
      reinterpret_cast<int8_t*>(activation_state_quantized->data.uint8);
  int8_t* quantized_cell_state_ptr =
      reinterpret_cast<int8_t*>(cell_state_quantized->data.uint8);
  float* scaling_factors_ptr = scaling_factors->data.f;
  float* prod_scaling_factors_ptr = prod_scaling_factors->data.f;
  float* recovered_weights_ptr = recovered_weights->data.f;

  LayerNormLstmStep(
      input_ptr_batch, input_to_input_weights_ptr, input_to_input_weights_scale,
      input_to_forget_weights_ptr, input_to_forget_weights_scale,
      input_to_cell_weights_ptr, input_to_cell_weights_scale,
      input_to_output_weights_ptr, input_to_output_weights_scale,
      recurrent_to_input_weights_ptr, recurrent_to_input_weights_scale,
      recurrent_to_forget_weights_ptr, recurrent_to_forget_weights_scale,
      recurrent_to_cell_weights_ptr, recurrent_to_cell_weights_scale,
      recurrent_to_output_weights_ptr, recurrent_to_output_weights_scale,
      cell_to_input_weights_ptr, cell_to_input_weights_scale,
      cell_to_forget_weights_ptr, cell_to_forget_weights_scale,
      cell_to_output_weights_ptr, cell_to_output_weights_scale,
      input_layer_norm_weight_ptr, forget_layer_norm_weight_ptr,
      cell_layer_norm_weight_ptr, output_layer_norm_weight_ptr,
      input_gate_bias_ptr, forget_gate_bias_ptr, cell_bias_ptr,
      output_gate_bias_ptr, projection_weights_ptr, projection_weights_scale,
      projection_bias_ptr, cell_clip, proj_clip, activation, n_batch, n_cell,
      n_input, n_output, input_gate_scratch, forget_gate_scratch, cell_scratch,
      output_gate_scratch, scaling_factors_ptr, prod_scaling_factors_ptr,
      recovered_weights_ptr, quantized_input_ptr,
      quantized_activation_state_ptr, quantized_cell_state_ptr,
      activation_state_ptr, cell_state_ptr, output_ptr_batch);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);

  const TfLiteTensor* input_to_input_weights =
      GetOptionalInputTensor(context, node, kInputToInputWeightsTensor);
  const TfLiteTensor* input_to_forget_weights =
      GetInput(context, node, kInputToForgetWeightsTensor);
  const TfLiteTensor* input_to_cell_weights =
      GetInput(context, node, kInputToCellWeightsTensor);
  const TfLiteTensor* input_to_output_weights =
      GetInput(context, node, kInputToOutputWeightsTensor);

  const TfLiteTensor* recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, kRecurrentToInputWeightsTensor);
  const TfLiteTensor* recurrent_to_forget_weights =
      GetInput(context, node, kRecurrentToForgetWeightsTensor);
  const TfLiteTensor* recurrent_to_cell_weights =
      GetInput(context, node, kRecurrentToCellWeightsTensor);
  const TfLiteTensor* recurrent_to_output_weights =
      GetInput(context, node, kRecurrentToOutputWeightsTensor);

  const TfLiteTensor* cell_to_input_weights =
      GetOptionalInputTensor(context, node, kCellToInputWeightsTensor);
  const TfLiteTensor* cell_to_forget_weights =
      GetOptionalInputTensor(context, node, kCellToForgetWeightsTensor);
  const TfLiteTensor* cell_to_output_weights =
      GetOptionalInputTensor(context, node, kCellToOutputWeightsTensor);

  const TfLiteTensor* input_layer_norm_weights =
      GetInput(context, node, kInputLayerNormWeightsTensor);
  const TfLiteTensor* forget_layer_norm_weights =
      GetInput(context, node, kForgetLayerNormWeightsTensor);
  const TfLiteTensor* cell_layer_norm_weights =
      GetInput(context, node, kCellLayerNormWeightsTensor);
  const TfLiteTensor* output_layer_norm_weights =
      GetInput(context, node, kOutputLayerNormWeightsTensor);

  const TfLiteTensor* input_gate_bias =
      GetOptionalInputTensor(context, node, kInputGateBiasTensor);
  const TfLiteTensor* forget_gate_bias =
      GetInput(context, node, kForgetGateBiasTensor);
  const TfLiteTensor* cell_bias = GetInput(context, node, kCellGateBiasTensor);
  const TfLiteTensor* output_gate_bias =
      GetInput(context, node, kOutputGateBiasTensor);

  const TfLiteTensor* projection_weights =
      GetOptionalInputTensor(context, node, kProjectionWeightsTensor);
  const TfLiteTensor* projection_bias =
      GetOptionalInputTensor(context, node, kProjectionBiasTensor);

  // Index the scratch buffers pointers to the global scratch buffer.
  TfLiteTensor* scratch_buffer = GetTemporary(context, node, /*index=*/0);

  TfLiteTensor* activation_state =
      &context->tensors[node->inputs->data[kInputActivationStateTensor]];
  TfLiteTensor* cell_state =
      &context->tensors[node->inputs->data[kInputCellStateTensor]];

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input_to_output_weights->type) {
    case kTfLiteFloat32: {
      return EvalFloat(input, input_to_input_weights, input_to_forget_weights,
                       input_to_cell_weights, input_to_output_weights,
                       recurrent_to_input_weights, recurrent_to_forget_weights,
                       recurrent_to_cell_weights, recurrent_to_output_weights,
                       cell_to_input_weights, cell_to_forget_weights,
                       cell_to_output_weights, input_layer_norm_weights,
                       forget_layer_norm_weights, cell_layer_norm_weights,
                       output_layer_norm_weights, input_gate_bias,
                       forget_gate_bias, cell_bias, output_gate_bias,
                       projection_weights, projection_bias, op_data->cell_clip,
                       op_data->proj_clip, op_data->activation, scratch_buffer,
                       activation_state, cell_state, output);
    }
    case kTfLiteUInt8: {
      TfLiteTensor* input_quantized = GetTemporary(context, node, /*index=*/1);
      TfLiteTensor* activation_state_quantized =
          GetTemporary(context, node, /*index=*/2);
      TfLiteTensor* cell_state_quantized =
          GetTemporary(context, node, /*index=*/3);
      TfLiteTensor* scaling_factors = GetTemporary(context, node, /*index=*/4);
      TfLiteTensor* prod_scaling_factors =
          GetTemporary(context, node, /*index=*/5);
      TfLiteTensor* recovered_weights =
          GetTemporary(context, node, /*index=*/6);
      return EvalHybrid(
          input, input_to_input_weights, input_to_forget_weights,
          input_to_cell_weights, input_to_output_weights,
          recurrent_to_input_weights, recurrent_to_forget_weights,
          recurrent_to_cell_weights, recurrent_to_output_weights,
          cell_to_input_weights, cell_to_forget_weights, cell_to_output_weights,
          input_layer_norm_weights, forget_layer_norm_weights,
          cell_layer_norm_weights, output_layer_norm_weights, input_gate_bias,
          forget_gate_bias, cell_bias, output_gate_bias, projection_weights,
          projection_bias, op_data->cell_clip, op_data->proj_clip,
          op_data->activation, scratch_buffer, scaling_factors,
          prod_scaling_factors, recovered_weights, input_quantized,
          activation_state_quantized, cell_state_quantized, activation_state,
          cell_state, output);
    }
    default:
      context->ReportError(context, "Type %d is not currently supported.",
                           input_to_output_weights->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

}  // namespace layer_norm_lstm

TfLiteRegistration* Register_LAYER_NORM_LSTM() {
  static TfLiteRegistration r = {layer_norm_lstm::Init, layer_norm_lstm::Free,
                                 layer_norm_lstm::Prepare,
                                 layer_norm_lstm::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
