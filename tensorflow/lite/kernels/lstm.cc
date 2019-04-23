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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/activation_functor.h"
#include "tensorflow/lite/kernels/gemm_support.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/lstm_eval.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace lstm {

struct OpData {
  // Which kernel type to use. Full kernel (24 inputs) or basic kernel (5
  // inputs).
  // Please note the 20-input full kernel is deprecated and only kept
  // here for backward compatibility.
  TfLiteLSTMKernelType kernel_type;

  // If the lstm is layer norm.
  bool is_layer_norm_lstm;

  // These fields are only used by full kernel.
  int activation_state_tensor_index;
  int cell_state_tensor_index;
  int scratch_tensor_index;
};

// For full inputs kernel (24-inputs).
// Please note the 20-input full kernel is deprecated and only kept
// here for backward compatibility.
namespace full {

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

// Gates bias tensors of size {n_cell}
constexpr int kInputGateBiasTensor = 12;  // Optional
constexpr int kForgetGateBiasTensor = 13;
constexpr int kCellGateBiasTensor = 14;
constexpr int kOutputGateBiasTensor = 15;

// Projection weight tensor of size {n_output, n_cell}
constexpr int kProjectionWeightsTensor = 16;  // Optional
// Projection bias tensor of size {n_output}
constexpr int kProjectionBiasTensor = 17;  // Optional

// These state tensors are defined as variable tensors, and will be modified by
// this op.
constexpr int kInputActivationStateTensor = 18;
constexpr int kInputCellStateTensor = 19;

// Layer norm coefficient tensors of size {n_cell}, representing a diagonal
// matrix.
constexpr int kInputLayerNormCoefficientsTensor = 20;   // Optional
constexpr int kForgetLayerNormCoefficientsTensor = 21;  // Optional
constexpr int kCellLayerNormCoefficientsTensor = 22;    // Optional
constexpr int kOutputLayerNormCoefficientsTensor = 23;  // Optional

// Output tensors.
constexpr int kOutputTensor = 0;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  op_data->kernel_type = kTfLiteLSTMFullKernel;
  context->AddTensors(context, /*tensors_to_add=*/7,
                      &op_data->scratch_tensor_index);
  return op_data;
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

  const TfLiteTensor* input_to_input_weights =
      GetOptionalInputTensor(context, node, kInputToInputWeightsTensor);
  const bool use_cifg = (input_to_input_weights == nullptr);
  if (!use_cifg) {
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
  // TODO(ghodrat): make sure this is correct.
  const bool projection_tensors_consistent =
      ((projection_weights != nullptr) || (projection_bias == nullptr));
  TF_LITE_ENSURE(context, projection_tensors_consistent == true);

  if (is_layer_norm_lstm) {
    const TfLiteTensor* input_layer_norm_coefficients = GetOptionalInputTensor(
        context, node, kInputLayerNormCoefficientsTensor);
    if (use_cifg) {
      TF_LITE_ENSURE_EQ(context, input_layer_norm_coefficients, nullptr);
    } else {
      TF_LITE_ENSURE(context, input_layer_norm_coefficients != nullptr);
      TF_LITE_ENSURE_EQ(context, input_layer_norm_coefficients->dims->size, 1);
      TF_LITE_ENSURE_EQ(context, input_layer_norm_coefficients->dims->data[0],
                        n_cell);
    }

    const TfLiteTensor* forget_layer_norm_coefficients =
        GetInput(context, node, kForgetLayerNormCoefficientsTensor);
    TF_LITE_ENSURE(context, forget_layer_norm_coefficients != nullptr);
    TF_LITE_ENSURE_EQ(context, forget_layer_norm_coefficients->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, forget_layer_norm_coefficients->dims->data[0],
                      n_cell);

    const TfLiteTensor* cell_layer_norm_coefficients =
        GetInput(context, node, kCellLayerNormCoefficientsTensor);
    TF_LITE_ENSURE(context, cell_layer_norm_coefficients != nullptr);
    TF_LITE_ENSURE_EQ(context, cell_layer_norm_coefficients->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_layer_norm_coefficients->dims->data[0],
                      n_cell);

    const TfLiteTensor* output_layer_norm_coefficients =
        GetInput(context, node, kOutputLayerNormCoefficientsTensor);
    TF_LITE_ENSURE(context, output_layer_norm_coefficients != nullptr);
    TF_LITE_ENSURE_EQ(context, output_layer_norm_coefficients->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, output_layer_norm_coefficients->dims->data[0],
                      n_cell);
  }

  return kTfLiteOk;
}

// Resize the output, state tensors based on the sizes of the input tensors.
// Allocate a temporary scratch tensor. Also check that the sizes of the input
// tensors match each other.
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  // Logic for determining regular lstm and layer norm lstm:
  // input_size, forget_gate_layer_norm_tensor (20) null? is_layer_norm?
  // 20,         N/A,                                     No.
  // 24,         null,                                    No.
  // 24,         not null,                                Yes.
  // 20-inputs lstm are deprecated and is only kept here for backward
  // compatibility.
  if (node->inputs->size == 24) {
    const TfLiteTensor* forget_layer_norm_coefficients = GetOptionalInputTensor(
        context, node, kForgetLayerNormCoefficientsTensor);
    if (forget_layer_norm_coefficients == nullptr) {
      op_data->is_layer_norm_lstm = false;
    } else {
      op_data->is_layer_norm_lstm = true;
    }
  } else if (node->inputs->size == 20) {
    // This is deprecated and is only kept here for backward compatibility.
    op_data->is_layer_norm_lstm = false;
  } else {
    context->ReportError(
        context, "The LSTM Full kernel expects 20 or 24 inputs. Got %d inputs",
        node->inputs->size);
    return kTfLiteError;
  }

  const bool is_layer_norm_lstm = op_data->is_layer_norm_lstm;
  op_data->activation_state_tensor_index =
      node->inputs->data[kInputActivationStateTensor];
  op_data->cell_state_tensor_index = node->inputs->data[kInputCellStateTensor];

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
  TF_LITE_ENSURE_OK(context,
                    CheckInputTensorDimensions(context, node, n_input, n_output,
                                               n_cell, is_layer_norm_lstm));

  // Get the pointer to output, activation_state and cell_state tensors.
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TfLiteTensor* activation_state =
      &context->tensors[op_data->activation_state_tensor_index];
  TfLiteTensor* cell_state =
      &context->tensors[op_data->cell_state_tensor_index];

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
  const bool is_hybrid_op = IsHybridOp(input, input_to_output_weights);

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
    input_quantized->type = input_to_output_weights->type;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }
    node->temporaries->data[2] = op_data->scratch_tensor_index + 2;
    TfLiteTensor* activation_state_quantized =
        GetTemporary(context, node, /*index=*/2);
    activation_state_quantized->type = input_to_output_weights->type;
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

    // Allocate a temporary tensor to store the recovered cell weights. Since
    // this is used for diagonal matrices, only need to store n_cell values.
    node->temporaries->data[6] = op_data->scratch_tensor_index + 6;
    TfLiteTensor* recovered_cell_weights =
        GetTemporary(context, node, /*index=*/6);
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
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* params = reinterpret_cast<TfLiteLSTMParams*>(node->builtin_data);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  const bool is_layer_norm_lstm = op_data->is_layer_norm_lstm;

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

  const TfLiteTensor* input_layer_norm_coefficients =
      is_layer_norm_lstm ? GetOptionalInputTensor(
                               context, node, kInputLayerNormCoefficientsTensor)
                         : nullptr;
  const TfLiteTensor* forget_layer_norm_coefficients =
      is_layer_norm_lstm
          ? GetInput(context, node, kForgetLayerNormCoefficientsTensor)
          : nullptr;
  const TfLiteTensor* cell_layer_norm_coefficients =
      is_layer_norm_lstm
          ? GetInput(context, node, kCellLayerNormCoefficientsTensor)
          : nullptr;
  const TfLiteTensor* output_layer_norm_coefficients =
      is_layer_norm_lstm
          ? GetInput(context, node, kOutputLayerNormCoefficientsTensor)
          : nullptr;

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
      &context->tensors[op_data->activation_state_tensor_index];
  TfLiteTensor* cell_state =
      &context->tensors[op_data->cell_state_tensor_index];

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // TODO(mirkov): add a check that weights are all uint8s or all floats.
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
          forget_gate_bias, cell_bias, output_gate_bias, projection_weights,
          projection_bias, params, /*forward_sequence=*/true,
          /*time_major=*/true,
          /*output_offset=*/0, scratch_buffer, activation_state, cell_state,
          output);
    }
    case kTfLiteUInt8:
    case kTfLiteInt8: {
      TfLiteTensor* input_quantized = GetTemporary(context, node, /*index=*/1);
      TfLiteTensor* activation_state_quantized =
          GetTemporary(context, node, /*index=*/2);
      TfLiteTensor* cell_state_quantized =
          GetTemporary(context, node, /*index=*/3);
      TfLiteTensor* scaling_factors = GetTemporary(context, node, /*index=*/4);
      TfLiteTensor* prod_scaling_factors =
          GetTemporary(context, node, /*index=*/5);
      TfLiteTensor* recovered_cell_weights =
          GetTemporary(context, node, /*index=*/6);
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
          forget_gate_bias, cell_bias, output_gate_bias, projection_weights,
          projection_bias, params, /*forward_sequence=*/true,
          /*time_major=*/true, /*output_offset=*/0, scratch_buffer,
          scaling_factors, prod_scaling_factors, recovered_cell_weights,
          input_quantized,
          /*aux_input_quantized=*/nullptr, activation_state_quantized,
          cell_state_quantized, activation_state, cell_state, output);
    }
    default:
      context->ReportError(context, "Type %d is not currently supported.",
                           input_to_output_weights->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace full

// For basic kernel (5-inputs).
namespace basic {

enum InputTensor {
  kInputData = 0,
  kInputPrevActivation = 1,
  kInputWeights = 2,
  kInputBiases = 3,
  kInputPrevState = 4,
  kInputNum = 5,
};

enum OutputTensor {
  kOutputActivation = 0,
  kOutputState = 1,
  kOutputConcatTemp = 2,
  kOutputActivationTemp = 3,
  kOutputNum = 4,
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  op_data->kernel_type = kTfLiteLSTMBasicKernel;
  // `scratch_tensor_index` is unused in this kernel.
  op_data->scratch_tensor_index = -1;
  return op_data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->inputs->size == kInputNum);
  TF_LITE_ENSURE(context, node->outputs->size == kOutputNum);

  const TfLiteTensor* input = GetInput(context, node, kInputData);
  const TfLiteTensor* prev_activation =
      GetInput(context, node, kInputPrevActivation);
  const TfLiteTensor* weights = GetInput(context, node, kInputWeights);
  const TfLiteTensor* bias = GetInput(context, node, kInputBiases);
  const TfLiteTensor* prev_state = GetInput(context, node, kInputPrevState);

  TF_LITE_ENSURE_EQ(context, input->dims->size, 2);
  const int num_batches = input->dims->data[0];
  const int input_depth = input->dims->data[1];

  TF_LITE_ENSURE_EQ(context, prev_activation->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, prev_activation->dims->data[0], num_batches);
  const int activation_depth = prev_activation->dims->data[1];
  const int total_depth = input_depth + activation_depth;

  TF_LITE_ENSURE_EQ(context, weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, weights->dims->data[0], 4 * activation_depth);
  TF_LITE_ENSURE_EQ(context, weights->dims->data[1], total_depth);

  TF_LITE_ENSURE_EQ(context, bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, bias->dims->data[0], 4 * activation_depth);

  TF_LITE_ENSURE_EQ(context, prev_state->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, prev_state->dims->data[0], num_batches);
  TF_LITE_ENSURE_EQ(context, prev_state->dims->data[1], activation_depth);

  TfLiteTensor* activation_out = GetOutput(context, node, kOutputActivation);
  TfLiteTensor* state_out = GetOutput(context, node, kOutputState);
  TfLiteTensor* concat_temp = GetOutput(context, node, kOutputConcatTemp);
  TfLiteTensor* activation_temp =
      GetOutput(context, node, kOutputActivationTemp);

  TF_LITE_ENSURE_OK(context, context->ResizeTensor(
                                 context, activation_out,
                                 TfLiteIntArrayCopy(prev_activation->dims)));
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, state_out,
                                     TfLiteIntArrayCopy(prev_state->dims)));

  TfLiteIntArray* concat_temp_size = TfLiteIntArrayCreate(2);
  concat_temp_size->data[0] = num_batches;
  concat_temp_size->data[1] = total_depth;
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, concat_temp, concat_temp_size));
  TfLiteIntArray* activation_temp_size = TfLiteIntArrayCreate(2);
  activation_temp_size->data[0] = num_batches;
  activation_temp_size->data[1] = 4 * activation_depth;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, activation_temp,
                                                   activation_temp_size));

  // Set the state tensors as persistent.
  for (auto index : {kInputPrevActivation, kInputPrevState}) {
    TfLiteTensor* tensor = &context->tensors[node->inputs->data[index]];
    tensor->allocation_type = kTfLiteArenaRwPersistent;
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputData);
  const TfLiteTensor* prev_activation =
      GetInput(context, node, kInputPrevActivation);
  const TfLiteTensor* weights = GetInput(context, node, kInputWeights);
  const TfLiteTensor* bias = GetInput(context, node, kInputBiases);
  const TfLiteTensor* prev_state = GetInput(context, node, kInputPrevState);

  TfLiteTensor* activation_out = GetOutput(context, node, kOutputActivation);
  TfLiteTensor* state_out = GetOutput(context, node, kOutputState);
  TfLiteTensor* concat_temp = GetOutput(context, node, kOutputConcatTemp);
  TfLiteTensor* activation_temp =
      GetOutput(context, node, kOutputActivationTemp);

  if (input->type == kTfLiteFloat32 &&
      prev_activation->type == kTfLiteFloat32 &&
      weights->type == kTfLiteFloat32 && bias->type == kTfLiteFloat32 &&
      prev_state->type == kTfLiteFloat32 && state_out->type == kTfLiteFloat32 &&
      activation_out->type == kTfLiteFloat32 &&
      concat_temp->type == kTfLiteFloat32 &&
      activation_temp->type == kTfLiteFloat32) {
    tflite::LstmCellParams op_params;
    // Float LSTM cell does not need parameters to be set: leave untouched.
    optimized_ops::LstmCell(
        op_params,
        // Inputs.
        GetTensorShape(input), GetTensorData<float>(input),
        GetTensorShape(prev_activation), GetTensorData<float>(prev_activation),
        GetTensorShape(weights), GetTensorData<float>(weights),
        GetTensorShape(bias), GetTensorData<float>(bias),
        GetTensorShape(prev_state), GetTensorData<float>(prev_state),
        // Outputs.
        GetTensorShape(state_out), GetTensorData<float>(state_out),
        GetTensorShape(activation_out), GetTensorData<float>(activation_out),
        GetTensorShape(concat_temp), GetTensorData<float>(concat_temp),
        GetTensorShape(activation_temp), GetTensorData<float>(activation_temp));
  } else if (input->type == kTfLiteUInt8 &&
             prev_activation->type == kTfLiteUInt8 &&
             weights->type == kTfLiteUInt8 && bias->type == kTfLiteInt32 &&
             prev_state->type == kTfLiteInt16 &&
             state_out->type == kTfLiteInt16 &&
             activation_out->type == kTfLiteUInt8 &&
             concat_temp->type == kTfLiteUInt8 &&
             activation_temp->type == kTfLiteInt16) {
    gemmlowp::GemmContext* gemm_context = gemm_support::GetFromContext(context);
    int state_scale_log2_rounded;
    if (!CheckedLog2(state_out->params.scale, &state_scale_log2_rounded)) {
      context->ReportError(
          context,
          "The internal state of a LSTM cell must have a power-of-two scale.");
      return kTfLiteError;
    }
    const int state_integer_bits = 15 + state_scale_log2_rounded;
    if (state_integer_bits != 4) {
      context->ReportError(context,
                           "The only case of quantized LstmCell currently "
                           "supported is with StateIntegerBits==4");
      return kTfLiteError;
    }

    double real_accum_multiplier = 4096 * bias->params.scale;
    int32 accum_multiplier;
    int accum_shift;
    tflite::QuantizeMultiplier(real_accum_multiplier, &accum_multiplier,
                               &accum_shift);
    tflite::LstmCellParams op_params;
    op_params.weights_zero_point = weights->params.zero_point;
    op_params.accum_multiplier = accum_multiplier;
    op_params.accum_shift = accum_shift;
    optimized_ops::LstmCell<4>(
        op_params,
        // Inputs.
        GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(prev_activation),
        GetTensorData<uint8_t>(prev_activation), GetTensorShape(weights),
        GetTensorData<uint8_t>(weights), GetTensorShape(bias),
        GetTensorData<int32_t>(bias), GetTensorShape(prev_state),
        GetTensorData<int16_t>(prev_state),
        // Outputs.
        GetTensorShape(state_out), GetTensorData<int16_t>(state_out),
        GetTensorShape(activation_out), GetTensorData<uint8_t>(activation_out),
        GetTensorShape(concat_temp), GetTensorData<uint8_t>(concat_temp),
        GetTensorShape(activation_temp),
        GetTensorData<int16_t>(activation_temp), gemm_context);
  } else {
    context->ReportError(context,
                         "Unsupported combination of data types for LstmCell");
    return kTfLiteError;
  }

  // TODO(ycling): Investigate if this copy can be avoided with the 5-inputs
  // LSTM kernel.
  memcpy(prev_activation->data.raw, activation_out->data.raw,
         activation_out->bytes);
  memcpy(prev_state->data.raw, state_out->data.raw, state_out->bytes);

  return kTfLiteOk;
}

}  // namespace basic

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  gemm_support::IncrementUsageCounter(context);

  const auto* params = reinterpret_cast<const TfLiteLSTMParams*>(buffer);
  switch (params->kernel_type) {
    case kTfLiteLSTMFullKernel:
      return full::Init(context, buffer, length);
    case kTfLiteLSTMBasicKernel:
      return basic::Init(context, buffer, length);
    default:
      return nullptr;
  }
  return nullptr;
}
void Free(TfLiteContext* context, void* buffer) {
  gemm_support::DecrementUsageCounter(context);

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const auto* op_data = reinterpret_cast<const OpData*>(node->user_data);
  switch (op_data->kernel_type) {
    case kTfLiteLSTMFullKernel:
      return full::Prepare(context, node);
    case kTfLiteLSTMBasicKernel:
      return basic::Prepare(context, node);
    default:
      return kTfLiteError;
  }
  return kTfLiteError;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* op_data = reinterpret_cast<const OpData*>(node->user_data);
  switch (op_data->kernel_type) {
    case kTfLiteLSTMFullKernel:
      return full::Eval(context, node);
    case kTfLiteLSTMBasicKernel:
      return basic::Eval(context, node);
    default:
      return kTfLiteError;
  }
  return kTfLiteError;
}

}  // namespace lstm

TfLiteRegistration* Register_LSTM() {
  static TfLiteRegistration r = {lstm::Init, lstm::Free, lstm::Prepare,
                                 lstm::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
