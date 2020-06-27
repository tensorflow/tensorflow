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

#include <math.h>

#include <cstddef>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/lstm_eval.h"
#include "tensorflow/lite/kernels/lstm_shared.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace unidirectional_sequence_lstm {

struct OpData {
  // If the lstm is layer norm.
  bool is_layer_norm_lstm;
  // The scratch tensor index.
  int scratch_tensor_index;
  bool compute_row_sums = false;
};

// Temporary tensors
enum TemporaryTensor {
  kScratchBuffer = 0,
  kInputQuantized = 1,
  kOutputStateQuantized = 2,
  kCellStateQuantized = 3,
  kInputScalingFactors = 4,
  kOutputStateScalingFactors = 5,
  kProductScalingFactors = 6,
  kRecoveredCellWeights = 7,
  kAccumScratch = 8,
  kInputZeroPoints = 9,
  kOutputStateZeroPoints = 10,
  kRowSums = 11,
  kNumTemporaryTensors = 12,
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  context->AddTensors(context, kNumTemporaryTensors,
                      &op_data->scratch_tensor_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

// Check that input tensor dimensions matches with each other.
TfLiteStatus CheckInputTensorDimensions(TfLiteContext* context,
                                        TfLiteNode* node, int n_input,
                                        int n_output, int n_cell,
                                        bool is_layer_norm_lstm) {
  const auto* params = reinterpret_cast<TfLiteLSTMParams*>(node->builtin_data);

  // Making sure clipping parameters have valid values.
  // == 0 means no clipping
  //  > 0 means clipping
  TF_LITE_ENSURE(context, params->cell_clip >= 0);
  TF_LITE_ENSURE(context, params->proj_clip >= 0);

  const TfLiteTensor* input_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kInputToInputWeightsTensor);
  if (input_to_input_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->data[1], n_input);
  }

  const TfLiteTensor* input_to_forget_weights =
      GetInput(context, node, lstm::full::kInputToForgetWeightsTensor);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->data[1], n_input);

  const TfLiteTensor* input_to_cell_weights =
      GetInput(context, node, lstm::full::kInputToCellWeightsTensor);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->data[1], n_input);

  const TfLiteTensor* recurrent_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kRecurrentToInputWeightsTensor);
  if (recurrent_to_input_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->data[0],
                      n_cell);
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->data[1],
                      n_output);
  }

  const TfLiteTensor* recurrent_to_forget_weights =
      GetInput(context, node, lstm::full::kRecurrentToForgetWeightsTensor);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->data[0],
                    n_cell);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->data[1],
                    n_output);

  const TfLiteTensor* recurrent_to_cell_weights =
      GetInput(context, node, lstm::full::kRecurrentToCellWeightsTensor);
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

  const TfLiteTensor* cell_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kCellToInputWeightsTensor);
  if (cell_to_input_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->data[0], n_cell);
  }

  const TfLiteTensor* cell_to_forget_weights = GetOptionalInputTensor(
      context, node, lstm::full::kCellToForgetWeightsTensor);
  if (cell_to_forget_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->data[0], n_cell);
  }

  const TfLiteTensor* cell_to_output_weights = GetOptionalInputTensor(
      context, node, lstm::full::kCellToOutputWeightsTensor);
  if (cell_to_output_weights != nullptr) {
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
      GetOptionalInputTensor(context, node, lstm::full::kInputGateBiasTensor);
  if (use_cifg) {
    TF_LITE_ENSURE_EQ(context, input_gate_bias, nullptr);
  } else {
    TF_LITE_ENSURE_EQ(context, input_gate_bias->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, input_gate_bias->dims->data[0], n_cell);
  }

  const TfLiteTensor* forget_gate_bias =
      GetInput(context, node, lstm::full::kForgetGateBiasTensor);
  TF_LITE_ENSURE_EQ(context, forget_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, forget_gate_bias->dims->data[0], n_cell);

  const TfLiteTensor* cell_gate_bias =
      GetInput(context, node, lstm::full::kCellGateBiasTensor);
  TF_LITE_ENSURE_EQ(context, cell_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, cell_gate_bias->dims->data[0], n_cell);

  const TfLiteTensor* output_gate_bias =
      GetInput(context, node, lstm::full::kOutputGateBiasTensor);
  TF_LITE_ENSURE_EQ(context, output_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, output_gate_bias->dims->data[0], n_cell);

  const TfLiteTensor* projection_weights = GetOptionalInputTensor(
      context, node, lstm::full::kProjectionWeightsTensor);
  if (projection_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[0], n_output);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[1], n_cell);
  }

  const TfLiteTensor* projection_bias =
      GetOptionalInputTensor(context, node, lstm::full::kProjectionBiasTensor);
  if (projection_bias != nullptr) {
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

  if (is_layer_norm_lstm) {
    const TfLiteTensor* input_layer_norm_coefficients = GetOptionalInputTensor(
        context, node, lstm::full::kInputLayerNormCoefficientsTensor);
    if (use_cifg) {
      TF_LITE_ENSURE_EQ(context, input_layer_norm_coefficients, nullptr);
    } else {
      TF_LITE_ENSURE(context, input_layer_norm_coefficients != nullptr);
      TF_LITE_ENSURE_EQ(context, input_layer_norm_coefficients->dims->size, 1);
      TF_LITE_ENSURE_EQ(context, input_layer_norm_coefficients->dims->data[0],
                        n_cell);
      TF_LITE_ENSURE_TYPES_EQ(context, input_layer_norm_coefficients->type,
                              kTfLiteFloat32);
    }

    const TfLiteTensor* forget_layer_norm_coefficients =
        GetInput(context, node, lstm::full::kForgetLayerNormCoefficientsTensor);
    TF_LITE_ENSURE(context, forget_layer_norm_coefficients != nullptr);
    TF_LITE_ENSURE_EQ(context, forget_layer_norm_coefficients->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, forget_layer_norm_coefficients->dims->data[0],
                      n_cell);
    TF_LITE_ENSURE_TYPES_EQ(context, forget_layer_norm_coefficients->type,
                            kTfLiteFloat32);

    const TfLiteTensor* cell_layer_norm_coefficients =
        GetInput(context, node, lstm::full::kCellLayerNormCoefficientsTensor);
    TF_LITE_ENSURE(context, cell_layer_norm_coefficients != nullptr);
    TF_LITE_ENSURE_EQ(context, cell_layer_norm_coefficients->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_layer_norm_coefficients->dims->data[0],
                      n_cell);
    TF_LITE_ENSURE_TYPES_EQ(context, cell_layer_norm_coefficients->type,
                            kTfLiteFloat32);

    const TfLiteTensor* output_layer_norm_coefficients =
        GetInput(context, node, lstm::full::kOutputLayerNormCoefficientsTensor);
    TF_LITE_ENSURE(context, output_layer_norm_coefficients != nullptr);
    TF_LITE_ENSURE_EQ(context, output_layer_norm_coefficients->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, output_layer_norm_coefficients->dims->data[0],
                      n_cell);
    TF_LITE_ENSURE_TYPES_EQ(context, output_layer_norm_coefficients->type,
                            kTfLiteFloat32);
  }

  return kTfLiteOk;
}

// Resize the output and  state tensors based on the sizes of the input tensors.
// Allocate a temporary scratch tensor. Also check that the sizes of the input
// tensors match each other.
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  const int scratch_tensor_index = op_data->scratch_tensor_index;

  // Check we have all the inputs and outputs we need.
  bool is_layer_norm_lstm = false;
  if (node->inputs->size == 24) {
    const TfLiteTensor* forget_layer_norm_coefficients = GetOptionalInputTensor(
        context, node, lstm::full::kForgetLayerNormCoefficientsTensor);
    if (forget_layer_norm_coefficients == nullptr) {
      is_layer_norm_lstm = false;
    } else {
      is_layer_norm_lstm = true;
    }
  } else if (node->inputs->size == 20) {
    // This is deprecated and is only kept here for backward compatibility.
    is_layer_norm_lstm = false;
  } else {
    context->ReportError(
        context, "The LSTM Full kernel expects 20 or 24 inputs. Got %d inputs",
        node->inputs->size);
    return kTfLiteError;
  }
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  op_data->is_layer_norm_lstm = is_layer_norm_lstm;

  // Inferring batch size, number of outputs and sequence length and
  // number of cells from the input tensors.
  const TfLiteTensor* input = GetInput(context, node, lstm::full::kInputTensor);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE(context, input->dims->size > 1);
  const auto* params =
      reinterpret_cast<TfLiteUnidirectionalSequenceLSTMParams*>(
          node->builtin_data);
  const bool time_major = params->time_major;
  const int n_batch = time_major ? input->dims->data[1] : input->dims->data[0];
  const int n_input = input->dims->data[2];

  const TfLiteTensor* input_to_output_weights =
      GetInput(context, node, lstm::full::kInputToOutputWeightsTensor);
  const int n_cell = input_to_output_weights->dims->data[0];
  TF_LITE_ENSURE_EQ(context, input_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_output_weights->dims->data[1], n_input);

  const TfLiteTensor* recurrent_to_output_weights =
      GetInput(context, node, lstm::full::kRecurrentToOutputWeightsTensor);
  TF_LITE_ENSURE_EQ(context, recurrent_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_output_weights->dims->data[0],
                    n_cell);
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Check that input tensor dimensions matches with each other.
  TF_LITE_ENSURE_OK(context,
                    CheckInputTensorDimensions(context, node, n_input, n_output,
                                               n_cell, is_layer_norm_lstm));

  // Get the pointer to output, output_state and cell_state buffer tensors.
  TfLiteTensor* output = GetOutput(context, node, lstm::full::kOutputTensor);

  TfLiteTensor* output_state =
      GetVariableInput(context, node, lstm::full::kOutputStateTensor);
  TF_LITE_ENSURE(context, output_state != nullptr);
  TfLiteTensor* cell_state =
      GetVariableInput(context, node, lstm::full::kCellStateTensor);
  TF_LITE_ENSURE(context, cell_state != nullptr);

  // Check the shape of input state tensors.
  // These tensor may be 1D or 2D. It's fine as long as the total size is
  // correct.
  TF_LITE_ENSURE_EQ(context, NumElements(output_state), n_batch * n_output);
  TF_LITE_ENSURE_EQ(context, NumElements(cell_state), n_batch * n_cell);

  // Resize the output tensors.
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input->dims);
  output_size->data[input->dims->size - 1] = n_output;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size));

  TfLiteIntArrayFree(node->temporaries);
  if (IsHybridOp(input, input_to_output_weights)) {
    node->temporaries = TfLiteIntArrayCreate(kNumTemporaryTensors);
  } else {
    node->temporaries = TfLiteIntArrayCreate(1);
  }
  node->temporaries->data[kScratchBuffer] =
      scratch_tensor_index + kScratchBuffer;

  // Create a scratch buffer tensor.
  TfLiteTensor* scratch_buffer = GetTemporary(context, node, kScratchBuffer);
  scratch_buffer->type = input->type;
  scratch_buffer->allocation_type = kTfLiteArenaRw;

  const TfLiteTensor* input_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kInputToInputWeightsTensor);
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

  if (IsHybridOp(input, input_to_output_weights)) {
    op_data->compute_row_sums = true;
    // Allocate temporary tensors to store quantized values of input,
    // output_state and cell_state tensors.
    node->temporaries->data[kInputQuantized] =
        scratch_tensor_index + kInputQuantized;
    TfLiteTensor* input_quantized =
        GetTemporary(context, node, kInputQuantized);
    input_quantized->type = input_to_output_weights->type;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }
    node->temporaries->data[kOutputStateQuantized] =
        scratch_tensor_index + kOutputStateQuantized;
    TfLiteTensor* output_state_quantized =
        GetTemporary(context, node, kOutputStateQuantized);
    output_state_quantized->type = input_to_output_weights->type;
    output_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(output_state_quantized->dims,
                             output_state->dims)) {
      TfLiteIntArray* output_state_quantized_size =
          TfLiteIntArrayCopy(output_state->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, output_state_quantized,
                                              output_state_quantized_size));
    }
    node->temporaries->data[kCellStateQuantized] =
        scratch_tensor_index + kCellStateQuantized;
    TfLiteTensor* cell_state_quantized =
        GetTemporary(context, node, kCellStateQuantized);
    cell_state_quantized->type = input_to_output_weights->type;
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
    node->temporaries->data[kInputScalingFactors] =
        op_data->scratch_tensor_index + kInputScalingFactors;
    TfLiteTensor* input_sf = GetTemporary(context, node, kInputScalingFactors);
    input_sf->type = kTfLiteFloat32;
    input_sf->allocation_type = kTfLiteArenaRw;
    int scaling_dims[1] = {n_batch};
    if (!TfLiteIntArrayEqualsArray(input_sf->dims, 1, scaling_dims)) {
      TfLiteIntArray* input_sf_size = TfLiteIntArrayCreate(1);
      input_sf_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, input_sf, input_sf_size));
    }
    node->temporaries->data[kOutputStateScalingFactors] =
        op_data->scratch_tensor_index + kOutputStateScalingFactors;
    TfLiteTensor* output_state_sf =
        GetTemporary(context, node, kOutputStateScalingFactors);
    output_state_sf->type = kTfLiteFloat32;
    output_state_sf->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(output_state_sf->dims, 1, scaling_dims)) {
      TfLiteIntArray* output_state_sf_size = TfLiteIntArrayCreate(1);
      output_state_sf_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_state_sf,
                                                       output_state_sf_size));
    }
    node->temporaries->data[kProductScalingFactors] =
        scratch_tensor_index + kProductScalingFactors;
    TfLiteTensor* prod_scaling_factors =
        GetTemporary(context, node, kProductScalingFactors);
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

    // Allocate a temporary tensor to store the recovered cell weights. Since
    // this is used for diagonal matrices, only need to store n_cell values.
    node->temporaries->data[kRecoveredCellWeights] =
        scratch_tensor_index + kRecoveredCellWeights;
    TfLiteTensor* recovered_cell_weights =
        GetTemporary(context, node, kRecoveredCellWeights);
    recovered_cell_weights->type = kTfLiteFloat32;
    recovered_cell_weights->allocation_type = kTfLiteArenaRw;
    int recovered_cell_dims[1] = {n_cell};
    if (!TfLiteIntArrayEqualsArray(recovered_cell_weights->dims, 1,
                                   recovered_cell_dims)) {
      TfLiteIntArray* recovered_cell_weights_size = TfLiteIntArrayCreate(1);
      recovered_cell_weights_size->data[0] = n_cell;
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, recovered_cell_weights,
                                              recovered_cell_weights_size));
    }

    // Allocate a temporary tensor to store the accumulated int32 values.
    node->temporaries->data[kAccumScratch] =
        scratch_tensor_index + kAccumScratch;
    TfLiteTensor* accum_scratch = GetTemporary(context, node, kAccumScratch);
    accum_scratch->type = kTfLiteInt32;
    accum_scratch->allocation_type = kTfLiteArenaRw;
    int accum_scratch_dims[2] = {n_cell, n_batch};
    if (!TfLiteIntArrayEqualsArray(accum_scratch->dims, 2,
                                   accum_scratch_dims)) {
      TfLiteIntArray* accum_size = TfLiteIntArrayCreate(2);
      accum_size->data[0] = n_cell;
      accum_size->data[1] = n_batch;
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, accum_scratch, accum_size));
    }
    node->temporaries->data[kInputZeroPoints] =
        op_data->scratch_tensor_index + kInputZeroPoints;
    TfLiteTensor* input_zp = GetTemporary(context, node, kInputZeroPoints);
    input_zp->type = kTfLiteFloat32;
    input_zp->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(input_zp->dims, 1, scaling_dims)) {
      TfLiteIntArray* input_zp_size = TfLiteIntArrayCreate(1);
      input_zp_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, input_zp, input_zp_size));
    }
    node->temporaries->data[kOutputStateZeroPoints] =
        op_data->scratch_tensor_index + kOutputStateZeroPoints;
    TfLiteTensor* output_state_zp =
        GetTemporary(context, node, kOutputStateZeroPoints);
    output_state_zp->type = kTfLiteFloat32;
    output_state_zp->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(output_state_zp->dims, 1, scaling_dims)) {
      TfLiteIntArray* output_state_zp_size = TfLiteIntArrayCreate(1);
      output_state_zp_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_state_zp,
                                                       output_state_zp_size));
    }
    node->temporaries->data[kRowSums] = scratch_tensor_index + kRowSums;
    TfLiteTensor* row_sums = GetTemporary(context, node, kRowSums);
    row_sums->type = kTfLiteInt32;
    row_sums->allocation_type = kTfLiteArenaRwPersistent;
    int row_sums_rows = use_cifg ? 6 : 8;
    const TfLiteTensor* projection_weights = GetOptionalInputTensor(
        context, node, lstm::full::kProjectionWeightsTensor);
    if (projection_weights != nullptr) {
      row_sums_rows += ceil(static_cast<float>(n_output) / n_cell);
    }
    int row_sums_dims[2] = {row_sums_rows, n_cell};
    if (!TfLiteIntArrayEqualsArray(row_sums->dims, 2, row_sums_dims)) {
      TfLiteIntArray* row_sums_size = TfLiteIntArrayCreate(2);
      row_sums_size->data[0] = row_sums_dims[0];
      row_sums_size->data[1] = row_sums_dims[1];
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, row_sums, row_sums_size));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* params =
      reinterpret_cast<TfLiteUnidirectionalSequenceLSTMParams*>(
          node->builtin_data);
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  const bool is_layer_norm_lstm = op_data->is_layer_norm_lstm;
  const bool time_major = params->time_major;
  const TfLiteTensor* input = GetInput(context, node, lstm::full::kInputTensor);

  const TfLiteTensor* input_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kInputToInputWeightsTensor);
  const TfLiteTensor* input_to_forget_weights =
      GetInput(context, node, lstm::full::kInputToForgetWeightsTensor);
  const TfLiteTensor* input_to_cell_weights =
      GetInput(context, node, lstm::full::kInputToCellWeightsTensor);
  const TfLiteTensor* input_to_output_weights =
      GetInput(context, node, lstm::full::kInputToOutputWeightsTensor);

  const TfLiteTensor* recurrent_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kRecurrentToInputWeightsTensor);
  const TfLiteTensor* recurrent_to_forget_weights =
      GetInput(context, node, lstm::full::kRecurrentToForgetWeightsTensor);
  const TfLiteTensor* recurrent_to_cell_weights =
      GetInput(context, node, lstm::full::kRecurrentToCellWeightsTensor);
  const TfLiteTensor* recurrent_to_output_weights =
      GetInput(context, node, lstm::full::kRecurrentToOutputWeightsTensor);

  const TfLiteTensor* cell_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kCellToInputWeightsTensor);
  const TfLiteTensor* cell_to_forget_weights = GetOptionalInputTensor(
      context, node, lstm::full::kCellToForgetWeightsTensor);
  const TfLiteTensor* cell_to_output_weights = GetOptionalInputTensor(
      context, node, lstm::full::kCellToOutputWeightsTensor);

  const TfLiteTensor* input_gate_bias =
      GetOptionalInputTensor(context, node, lstm::full::kInputGateBiasTensor);
  const TfLiteTensor* forget_gate_bias =
      GetInput(context, node, lstm::full::kForgetGateBiasTensor);
  const TfLiteTensor* cell_gate_bias =
      GetInput(context, node, lstm::full::kCellGateBiasTensor);
  const TfLiteTensor* output_gate_bias =
      GetInput(context, node, lstm::full::kOutputGateBiasTensor);

  const TfLiteTensor* projection_weights = GetOptionalInputTensor(
      context, node, lstm::full::kProjectionWeightsTensor);
  const TfLiteTensor* projection_bias =
      GetOptionalInputTensor(context, node, lstm::full::kProjectionBiasTensor);

  // Index the scratch buffers pointers to the global scratch buffer.
  TfLiteTensor* scratch_buffer = GetTemporary(context, node, kScratchBuffer);

  TfLiteTensor* output_state =
      GetVariableInput(context, node, lstm::full::kOutputStateTensor);
  TF_LITE_ENSURE(context, output_state != nullptr);
  TfLiteTensor* cell_state =
      GetVariableInput(context, node, lstm::full::kCellStateTensor);
  TF_LITE_ENSURE(context, cell_state != nullptr);

  const TfLiteTensor* input_layer_norm_coefficients =
      is_layer_norm_lstm
          ? GetOptionalInputTensor(
                context, node, lstm::full::kInputLayerNormCoefficientsTensor)
          : nullptr;
  const TfLiteTensor* forget_layer_norm_coefficients =
      is_layer_norm_lstm
          ? GetInput(context, node,
                     lstm::full::kForgetLayerNormCoefficientsTensor)
          : nullptr;
  const TfLiteTensor* cell_layer_norm_coefficients =
      is_layer_norm_lstm
          ? GetInput(context, node,
                     lstm::full::kCellLayerNormCoefficientsTensor)
          : nullptr;
  const TfLiteTensor* output_layer_norm_coefficients =
      is_layer_norm_lstm
          ? GetInput(context, node,
                     lstm::full::kOutputLayerNormCoefficientsTensor)
          : nullptr;

  TfLiteTensor* output = GetOutput(context, node, lstm::full::kOutputTensor);

  // Copy out the LSTM specific params so they can be passed in the function.
  TfLiteLSTMParams lstm_params;
  lstm_params.activation = params->activation;
  lstm_params.cell_clip = params->cell_clip;
  lstm_params.proj_clip = params->proj_clip;
  lstm_params.asymmetric_quantize_inputs = params->asymmetric_quantize_inputs;

  switch (input_to_output_weights->type) {
    case kTfLiteFloat32: {
      return lstm_eval::EvalFloat(
          input, input_to_input_weights, input_to_forget_weights,
          input_to_cell_weights, input_to_output_weights,
          recurrent_to_input_weights, recurrent_to_forget_weights,
          recurrent_to_cell_weights, recurrent_to_output_weights,
          cell_to_input_weights, cell_to_forget_weights, cell_to_output_weights,
          input_layer_norm_coefficients, forget_layer_norm_coefficients,
          cell_layer_norm_coefficients, output_layer_norm_coefficients,
          /*aux_input=*/nullptr,
          /*aux_input_to_input_weights=*/nullptr,
          /*aux_input_to_forget_weights=*/nullptr,
          /*aux_input_to_cell_weights=*/nullptr,
          /*aux_input_to_output_weights=*/nullptr, input_gate_bias,
          forget_gate_bias, cell_gate_bias, output_gate_bias,
          projection_weights, projection_bias, &lstm_params,
          /*forward_sequence=*/true, time_major,
          /*output_offset=*/0, scratch_buffer, output_state, cell_state,
          output);
    }
    case kTfLiteUInt8:
    case kTfLiteInt8: {
      OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
      TfLiteTensor* row_sums = GetTemporary(context, node, kRowSums);
      const int row_sums_size = row_sums->dims->data[0];
      return lstm_eval::EvalHybrid(
          input, input_to_input_weights, input_to_forget_weights,
          input_to_cell_weights, input_to_output_weights,
          recurrent_to_input_weights, recurrent_to_forget_weights,
          recurrent_to_cell_weights, recurrent_to_output_weights,
          cell_to_input_weights, cell_to_forget_weights, cell_to_output_weights,
          input_layer_norm_coefficients, forget_layer_norm_coefficients,
          cell_layer_norm_coefficients, output_layer_norm_coefficients,
          /*aux_input=*/nullptr,
          /*aux_input_to_input_weights=*/nullptr,
          /*aux_input_to_forget_weights=*/nullptr,
          /*aux_input_to_cell_weights=*/nullptr,
          /*aux_input_to_output_weights=*/nullptr, input_gate_bias,
          forget_gate_bias, cell_gate_bias, output_gate_bias,
          projection_weights, projection_bias, &lstm_params,
          /*forward_sequence=*/true, time_major,
          /*output_offset=*/0, scratch_buffer,
          GetTemporary(context, node, kInputScalingFactors),
          /*aux_input_sf=*/nullptr,
          GetTemporary(context, node, kOutputStateScalingFactors),
          GetTemporary(context, node, kProductScalingFactors),
          GetTemporary(context, node, kRecoveredCellWeights),
          GetTemporary(context, node, kInputQuantized),
          /*aux_input_quantized=*/nullptr,
          GetTemporary(context, node, kOutputStateQuantized),
          GetTemporary(context, node, kCellStateQuantized), output_state,
          cell_state, GetTemporary(context, node, kAccumScratch), output,
          GetTemporary(context, node, kInputZeroPoints),
          /*aux_input_zp=*/nullptr,
          GetTemporary(context, node, kOutputStateZeroPoints), row_sums,
          row_sums_size, &op_data->compute_row_sums,
          CpuBackendContext::GetFromContext(context));
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s is not currently supported.",
                         TfLiteTypeGetName(input_to_output_weights->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace unidirectional_sequence_lstm

TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_LSTM() {
  static TfLiteRegistration r = {unidirectional_sequence_lstm::Init,
                                 unidirectional_sequence_lstm::Free,
                                 unidirectional_sequence_lstm::Prepare,
                                 unidirectional_sequence_lstm::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
