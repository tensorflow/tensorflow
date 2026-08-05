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
#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace unidirectional_sequence_rnn {

namespace {

// Holds kernel-specific runtime state and scratch tensor index allocations
// across invocations, including whether row sums need to be computed.
struct OpData {
  int scratch_tensor_index;
  bool compute_row_sums = false;
};

// Encapsulates the extracted tensor dimensions (batch size, max time,
// number of units, and input size) for the unidirectional sequence RNN op.
struct RnnDimensions {
  int batch_size;
  int max_time;
  int num_units;
  int input_size;
};

// Helper function that extracts and returns the standardized RNN dimensions
// from the input tensor, input weights, and op parameters.
inline RnnDimensions GetRnnDimensions(const TfLiteTensor* input,
                                      const TfLiteTensor* input_weights,
                                      const TfLiteSequenceRNNParams* params) {
  const bool time_major = params->time_major;
  return {
      /*batch_size=*/time_major ? input->dims->data[1] : input->dims->data[0],
      /*max_time=*/time_major ? input->dims->data[0] : input->dims->data[1],
      /*num_units=*/input_weights->dims->data[0],
      /*input_size=*/input->dims->data[2]};
}

}  // namespace

// Input tensors.
constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kRecurrentWeightsTensor = 2;
constexpr int kBiasTensor = 3;
constexpr int kHiddenStateTensor = 4;

// Output tensor.
constexpr int kOutputTensor = 0;

// Temporary (scratch) tensors.
constexpr int kInputQuantizedScratchTensor = 0;
constexpr int kHiddenStateQuantizedScratchTensor = 1;
constexpr int kScalingFactorsScratchTensor = 2;
constexpr int kAccumScratchTensor = 3;
constexpr int kZeroPointsScratchTensor = 4;
constexpr int kRowSumsScratchTensor = 5;

// Initializes the op data and allocates scratch tensor indices.
void* Init(TfLiteContext* context, const char* /*buffer*/, size_t /*length*/) {
  auto* op_data = new OpData();
  context->AddTensors(context, /*tensors_to_add=*/6,
                      &op_data->scratch_tensor_index);
  return op_data;
}

// Frees the op data allocated in Init.
void Free(TfLiteContext* /*context*/, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

// Verifies tensor dimensions and types, and allocates temporary scratch
// tensors.
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 5);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* input_weights;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kWeightsTensor, &input_weights));
  const TfLiteTensor* recurrent_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, kRecurrentWeightsTensor, &recurrent_weights));
  const TfLiteTensor* bias;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kBiasTensor, &bias));
  const TfLiteTensor* hidden_state;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kHiddenStateTensor, &hidden_state));

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 3);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_weights), 2);
  TF_LITE_ENSURE_EQ(context, NumDimensions(recurrent_weights), 2);
  TF_LITE_ENSURE_EQ(context, NumDimensions(bias), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(hidden_state), 2);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  TF_LITE_ENSURE(context, node->builtin_data != nullptr);
  auto* params = static_cast<TfLiteSequenceRNNParams*>(node->builtin_data);
  const RnnDimensions rnn_dims = GetRnnDimensions(input, input_weights, params);
  const bool time_major = params->time_major;
  const int batch_size = rnn_dims.batch_size;
  const int max_time = rnn_dims.max_time;
  const int num_units = rnn_dims.num_units;
  TF_LITE_ENSURE_EQ(context, rnn_dims.input_size, input_weights->dims->data[1]);
  TF_LITE_ENSURE_EQ(context, input_weights->dims->data[0], bias->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, recurrent_weights->dims->data[0],
                    bias->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, recurrent_weights->dims->data[1],
                    bias->dims->data[0]);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, input_weights->type,
                          recurrent_weights->type);
  TF_LITE_ENSURE_EQ(context, hidden_state->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, hidden_state->dims->data[1], num_units);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Resize output.
  TfLiteIntArray* output_size_array = TfLiteIntArrayCreate(3);
  output_size_array->data[0] = time_major ? max_time : batch_size;
  output_size_array->data[1] = time_major ? batch_size : max_time;
  output_size_array->data[2] = num_units;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size_array));

  const bool is_hybrid = IsHybridOp(input, input_weights);

  // Allocate temporary tensors to store quantized values of input and
  // hidden_state tensors.
  if (is_hybrid) {
    auto* op_data = static_cast<OpData*>(node->user_data);
    op_data->compute_row_sums = true;
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(6);
    for (int i = 0; i < 6; ++i) {
      node->temporaries->data[i] = op_data->scratch_tensor_index + i;
    }
    TfLiteTensor* input_quantized;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, kInputQuantizedScratchTensor,
                                  &input_quantized));
    input_quantized->type = input_weights->type;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }
    TfLiteTensor* hidden_state_quantized;
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node, kHiddenStateQuantizedScratchTensor,
                         &hidden_state_quantized));
    hidden_state_quantized->type = input_weights->type;
    hidden_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(hidden_state_quantized->dims,
                             hidden_state->dims)) {
      TfLiteIntArray* hidden_state_quantized_size =
          TfLiteIntArrayCopy(hidden_state->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, hidden_state_quantized,
                                              hidden_state_quantized_size));
    }
    TfLiteTensor* scaling_factors;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, kScalingFactorsScratchTensor,
                                  &scaling_factors));
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;
    int scaling_dims[1] = {batch_size};
    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
      TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
      scaling_factors_size->data[0] = batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }
    TfLiteTensor* accum_scratch;
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node, kAccumScratchTensor, &accum_scratch));
    accum_scratch->type = kTfLiteInt32;
    accum_scratch->allocation_type = kTfLiteArenaRw;
    int accum_scratch_dims[2] = {num_units, batch_size};
    if (!TfLiteIntArrayEqualsArray(accum_scratch->dims, 2,
                                   accum_scratch_dims)) {
      TfLiteIntArray* accum_scratch_size = TfLiteIntArrayCreate(2);
      accum_scratch_size->data[0] = accum_scratch_dims[0];
      accum_scratch_size->data[1] = accum_scratch_dims[1];
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, accum_scratch,
                                                       accum_scratch_size));
    }
    TfLiteTensor* zero_points;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, kZeroPointsScratchTensor,
                                       &zero_points));
    zero_points->type = kTfLiteInt32;
    zero_points->allocation_type = kTfLiteArenaRw;
    int zero_points_dims[1] = {batch_size};
    if (!TfLiteIntArrayEqualsArray(zero_points->dims, 1, zero_points_dims)) {
      TfLiteIntArray* zero_points_size = TfLiteIntArrayCreate(1);
      zero_points_size->data[0] = batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, zero_points,
                                                       zero_points_size));
    }
    TfLiteTensor* row_sums;
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node, kRowSumsScratchTensor, &row_sums));
    row_sums->type = kTfLiteInt32;
    row_sums->allocation_type = kTfLiteArenaRwPersistent;
    int row_sums_dims[2] = {2, num_units};
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

// Evaluates the unidirectional sequence RNN op using float32 weights and
// inputs, unrolling the sequence over time and updating the hidden state.
// Returns kTfLiteOk on success.
TfLiteStatus EvalFloat(const TfLiteTensor* input,
                       const TfLiteTensor* input_weights,
                       const TfLiteTensor* recurrent_weights,
                       const TfLiteTensor* bias,
                       const TfLiteSequenceRNNParams* params,
                       TfLiteTensor* hidden_state, TfLiteTensor* output) {
  const float* bias_ptr = GetTensorData<float>(bias);

  const RnnDimensions rnn_dims = GetRnnDimensions(input, input_weights, params);
  const bool time_major = params->time_major;
  const int batch_size = rnn_dims.batch_size;
  const int max_time = rnn_dims.max_time;
  const int num_units = rnn_dims.num_units;
  const int input_size = rnn_dims.input_size;

  const float* input_weights_ptr = GetTensorData<float>(input_weights);
  const float* recurrent_weights_ptr = GetTensorData<float>(recurrent_weights);

  if (time_major) {
    float* hidden_state_ptr_batch = GetTensorData<float>(hidden_state);
    const float* batch_input_ptr = GetTensorData<float>(input);
    float* batch_output_ptr = GetTensorData<float>(output);
    for (int s = 0; s < max_time; s++) {
      const float* input_ptr_batch =
          batch_input_ptr + s * input_size * batch_size;
      float* output_ptr_batch = batch_output_ptr + s * num_units * batch_size;

      kernel_utils::RnnBatchStep(
          input_ptr_batch, input_weights_ptr, recurrent_weights_ptr, bias_ptr,
          input_size, num_units, batch_size, num_units, params->activation,
          hidden_state_ptr_batch, output_ptr_batch);
    }
  } else {
    for (int b = 0; b < batch_size; b++) {
      float* hidden_state_ptr_batch =
          GetTensorData<float>(hidden_state) + b * num_units;
      const float* batch_input_ptr =
          GetTensorData<float>(input) + b * input_size * max_time;
      float* batch_output_ptr =
          GetTensorData<float>(output) + b * num_units * max_time;
      for (int s = 0; s < max_time; s++) {
        const float* input_ptr_batch = batch_input_ptr + s * input_size;
        float* output_ptr_batch = batch_output_ptr + s * num_units;

        kernel_utils::RnnBatchStep(
            input_ptr_batch, input_weights_ptr, recurrent_weights_ptr, bias_ptr,
            input_size, num_units, /*batch_size=*/1, num_units,
            params->activation, hidden_state_ptr_batch, output_ptr_batch);
      }
    }
  }
  return kTfLiteOk;
}

// Evaluates the unidirectional sequence RNN op using hybrid quantized weights
// and float inputs, dynamically quantizing inputs and unrolling the sequence
// over time. Returns kTfLiteOk on success.
TfLiteStatus EvalHybrid(
    const TfLiteTensor* input, const TfLiteTensor* input_weights,
    const TfLiteTensor* recurrent_weights, const TfLiteTensor* bias,
    const TfLiteSequenceRNNParams* params, TfLiteTensor* input_scratch,
    TfLiteTensor* hidden_state_scratch, TfLiteTensor* scaling_factors,
    TfLiteTensor* hidden_state, TfLiteTensor* output, TfLiteTensor* zero_points,
    TfLiteTensor* accum_scratch, TfLiteTensor* row_sums,
    bool* compute_row_sums) {
  const RnnDimensions rnn_dims = GetRnnDimensions(input, input_weights, params);
  const bool time_major = params->time_major;
  const int batch_size = rnn_dims.batch_size;
  const int max_time = rnn_dims.max_time;
  const int num_units = rnn_dims.num_units;
  const int input_size = rnn_dims.input_size;

  const float* bias_ptr = GetTensorData<float>(bias);

  const int8_t* input_weights_ptr = GetTensorData<int8_t>(input_weights);
  const int8_t* recurrent_weights_ptr =
      GetTensorData<int8_t>(recurrent_weights);
  int8_t* quantized_input_ptr = GetTensorData<int8_t>(input_scratch);
  int8_t* quantized_hidden_state_ptr =
      GetTensorData<int8_t>(hidden_state_scratch);

  const float input_weights_scale = input_weights->params.scale;
  const float recurrent_weights_scale = recurrent_weights->params.scale;
  float* scaling_factors_ptr = GetTensorData<float>(scaling_factors);
  int32_t* accum_scratch_ptr = GetTensorData<int32_t>(accum_scratch);
  int32_t* zero_points_ptr = nullptr;
  int32_t* row_sums_ptr = nullptr;

  if (params->asymmetric_quantize_inputs) {
    zero_points_ptr = GetTensorData<int32_t>(zero_points);
    row_sums_ptr = GetTensorData<int32_t>(row_sums);
  }

  if (time_major) {
    float* hidden_state_ptr_batch = GetTensorData<float>(hidden_state);
    const float* batch_input_ptr = GetTensorData<float>(input);
    float* batch_output_ptr = GetTensorData<float>(output);
    for (int s = 0; s < max_time; s++) {
      const float* input_ptr_batch =
          batch_input_ptr + s * input_size * batch_size;
      float* output_ptr_batch = batch_output_ptr + s * num_units * batch_size;

      kernel_utils::RnnBatchStep(
          input_ptr_batch, input_weights_ptr, input_weights_scale,
          recurrent_weights_ptr, recurrent_weights_scale, bias_ptr, input_size,
          num_units, batch_size, num_units, params->activation,
          quantized_input_ptr, quantized_hidden_state_ptr, scaling_factors_ptr,
          hidden_state_ptr_batch, output_ptr_batch,
          params->asymmetric_quantize_inputs, zero_points_ptr,
          accum_scratch_ptr, row_sums_ptr, compute_row_sums);
    }
  } else {
    for (int b = 0; b < batch_size; b++) {
      float* hidden_state_ptr_batch =
          GetTensorData<float>(hidden_state) + b * num_units;
      const float* batch_input_ptr =
          GetTensorData<float>(input) + b * input_size * max_time;
      float* batch_output_ptr =
          GetTensorData<float>(output) + b * num_units * max_time;
      for (int s = 0; s < max_time; s++) {
        const float* input_ptr_batch = batch_input_ptr + s * input_size;
        float* output_ptr_batch = batch_output_ptr + s * num_units;
        kernel_utils::RnnBatchStep(
            input_ptr_batch, input_weights_ptr, input_weights_scale,
            recurrent_weights_ptr, recurrent_weights_scale, bias_ptr,
            input_size, num_units, /*batch_size=*/1, num_units,
            params->activation, quantized_input_ptr, quantized_hidden_state_ptr,
            scaling_factors_ptr, hidden_state_ptr_batch, output_ptr_batch,
            params->asymmetric_quantize_inputs, zero_points_ptr,
            accum_scratch_ptr, row_sums_ptr, compute_row_sums);
      }
    }
  }
  return kTfLiteOk;
}

// Evaluates the unidirectional sequence RNN op, dispatching to float or hybrid
// implementations based on weight type. Returns kTfLiteOk on success.
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->builtin_data != nullptr);
  auto* params = static_cast<TfLiteSequenceRNNParams*>(node->builtin_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* input_weights;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kWeightsTensor, &input_weights));
  const TfLiteTensor* recurrent_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, kRecurrentWeightsTensor, &recurrent_weights));
  const TfLiteTensor* bias;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kBiasTensor, &bias));
  // The hidden_state is a variable input tensor that can be modified.
  TfLiteTensor* hidden_state =
      GetVariableInput(context, node, kHiddenStateTensor);
  TF_LITE_ENSURE(context, hidden_state != nullptr);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (input_weights->type) {
    case kTfLiteFloat32:
      return EvalFloat(input, input_weights, recurrent_weights, bias, params,
                       hidden_state, output);
    case kTfLiteUInt8:
    case kTfLiteInt8: {
      // TODO: b/512604697 - Implement eval with quantized inputs as well.
      auto* op_data = static_cast<OpData*>(node->user_data);
      TfLiteTensor* input_quantized;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, kInputQuantizedScratchTensor,
                                    &input_quantized));
      TfLiteTensor* hidden_state_quantized;
      TF_LITE_ENSURE_OK(
          context,
          GetTemporarySafe(context, node, kHiddenStateQuantizedScratchTensor,
                           &hidden_state_quantized));
      TfLiteTensor* scaling_factors;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, kScalingFactorsScratchTensor,
                                    &scaling_factors));
      TfLiteTensor* accum_scratch;
      TF_LITE_ENSURE_OK(
          context,
          GetTemporarySafe(context, node, kAccumScratchTensor, &accum_scratch));
      TfLiteTensor* zero_points;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, kZeroPointsScratchTensor,
                                    &zero_points));
      TfLiteTensor* row_sums;
      TF_LITE_ENSURE_OK(
          context,
          GetTemporarySafe(context, node, kRowSumsScratchTensor, &row_sums));
      return EvalHybrid(input, input_weights, recurrent_weights, bias, params,
                        input_quantized, hidden_state_quantized,
                        scaling_factors, hidden_state, output, zero_points,
                        accum_scratch, row_sums, &op_data->compute_row_sums);
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input_weights->type));
      return kTfLiteError;
  }
}

}  // namespace unidirectional_sequence_rnn

TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_RNN() {
  static TfLiteRegistration r = {
      unidirectional_sequence_rnn::Init, unidirectional_sequence_rnn::Free,
      unidirectional_sequence_rnn::Prepare, unidirectional_sequence_rnn::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
