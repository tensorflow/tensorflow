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

// SVDF op that compresses a fully connected op via low-rank matrix
// factorization. See https://research.google.com/pubs/archive/43813.pdf for
// details.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/contrib/lite/c/builtin_op_data.h"
#include "tensorflow/contrib/lite/c/c_api_internal.h"
#include "tensorflow/contrib/lite/kernels/activation_functor.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace svdf {

namespace {

struct OpData {
  int scratch_tensor_index;
  bool float_weights_time_initialized;

  int activation_state_tensor_index;
};

static inline void ApplyTimeWeightsBiasAndActivation(
    int batch_size, int memory_size, int num_filters, int num_units, int rank,
    const TfLiteTensor* weights_time, const TfLiteTensor* bias,
    TfLiteFusedActivation activation, TfLiteTensor* activation_state,
    TfLiteTensor* scratch, TfLiteTensor* output) {
  // Compute matmul(state, weights_time).
  // The right most column is used to save temporary output (with the size of
  // num_filters). This is achieved by starting at activation_state->data.f,
  // and having the stride equal to memory_size.
  for (int b = 0; b < batch_size; ++b) {
    float* state_ptr_batch =
        activation_state->data.f + b * memory_size * num_filters;
    float* scratch_ptr_batch = scratch->data.f + b * num_filters;
    tensor_utils::BatchVectorBatchVectorDotProduct(
        weights_time->data.f, state_ptr_batch, memory_size, num_filters,
        scratch_ptr_batch, /*result_stride=*/1);
  }

  // Initialize output with bias if provided.
  if (bias) {
    tensor_utils::VectorBatchVectorAssign(bias->data.f, num_units, batch_size,
                                          output->data.f);
  } else {
    tensor_utils::ZeroVector(output->data.f, batch_size * num_units);
  }

  // Reduction sum.
  for (int b = 0; b < batch_size; ++b) {
    float* output_ptr_batch = output->data.f + b * num_units;
    float* scratch_ptr_batch = scratch->data.f + b * num_filters;
    tensor_utils::ReductionSumVector(scratch_ptr_batch, output_ptr_batch,
                                     num_units, rank);
  }

  // Apply activation.
  for (int b = 0; b < batch_size; ++b) {
    float* output_ptr_batch = output->data.f + b * num_units;
    tensor_utils::ApplyActivationToVector(output_ptr_batch, num_units,
                                          activation, output_ptr_batch);
  }

  // Left shift the activation_state to make room for next cycle's activation.
  // TODO(alanchiao): explore collapsing this into a single loop.
  for (int b = 0; b < batch_size; ++b) {
    float* state_ptr_batch =
        activation_state->data.f + b * memory_size * num_filters;
    for (int f = 0; f < num_filters; ++f) {
      tensor_utils::VectorShiftLeft(state_ptr_batch, memory_size,
                                    /*shift_value=*/0.0f);
      state_ptr_batch += memory_size;
    }
  }
}

}  // namespace

// Input tensors.
constexpr int kInputTensor = 0;
constexpr int kWeightsFeatureTensor = 1;
constexpr int kWeightsTimeTensor = 2;
constexpr int kBiasTensor = 3;
// This is a variable tensor, and will be modified by this op.
constexpr int kInputActivationStateTensor = 4;

// Output tensor.
constexpr int kOutputTensor = 0;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  op_data->float_weights_time_initialized = false;
  context->AddTensors(context, /*tensors_to_add=*/4,
                      &op_data->scratch_tensor_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  int scratch_tensor_index = op_data->scratch_tensor_index;

  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 5);
  op_data->activation_state_tensor_index =
      node->inputs->data[kInputActivationStateTensor];

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights_feature =
      GetInput(context, node, kWeightsFeatureTensor);
  const TfLiteTensor* weights_time =
      GetInput(context, node, kWeightsTimeTensor);

  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  const int rank = params->rank;
  const int batch_size = input->dims->data[0];
  const int num_filters = weights_feature->dims->data[0];
  TF_LITE_ASSERT_EQ(num_filters % rank, 0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];
  TF_LITE_ASSERT_EQ(input->dims->data[1], weights_feature->dims->data[1]);
  TF_LITE_ASSERT_EQ(weights_time->dims->data[0], num_filters);

  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  if (bias) {
    TF_LITE_ASSERT_EQ(bias->dims->data[0], num_units);
  }

  TfLiteTensor* activation_state =
      &context->tensors[op_data->activation_state_tensor_index];
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Check the shape of input state tensors.
  TF_LITE_ENSURE_EQ(context, NumDimensions(activation_state), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(activation_state, 0), batch_size);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(activation_state, 1),
                    memory_size * num_filters);

  // Resize output.
  TfLiteIntArray* output_size_array = TfLiteIntArrayCreate(2);
  output_size_array->data[0] = batch_size;
  output_size_array->data[1] = num_units;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size_array));

  // The weights are of consistent type, so it suffices to check one.
  const bool is_hybrid_op =
      (input->type == kTfLiteFloat32 && weights_feature->type == kTfLiteUInt8);

  // Resize scratch.
  TfLiteIntArrayFree(node->temporaries);
  if (is_hybrid_op) {
    node->temporaries = TfLiteIntArrayCreate(4);
  } else {
    node->temporaries = TfLiteIntArrayCreate(1);
  }
  node->temporaries->data[0] = scratch_tensor_index;

  TfLiteIntArray* scratch_size_array = TfLiteIntArrayCreate(2);
  scratch_size_array->data[0] = batch_size;
  scratch_size_array->data[1] = num_filters;

  TfLiteTensor* scratch_tensor = GetTemporary(context, node, /*index=*/0);
  scratch_tensor->type = input->type;
  scratch_tensor->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_tensor,
                                                   scratch_size_array));

  if (is_hybrid_op) {
    // Tell interpreter to allocate temporary tensors to store quantized values
    // of input tensors.
    node->temporaries->data[1] = scratch_tensor_index + 1;
    TfLiteTensor* input_quantized = GetTemporary(context, node, /*index=*/1);
    input_quantized->type = kTfLiteUInt8;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }

    // Tell interpreter to allocate temporary tensors to store scaling factors.
    node->temporaries->data[2] = scratch_tensor_index + 2;
    TfLiteTensor* scaling_factors = GetTemporary(context, node, /*index=*/2);
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;
    TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
    scaling_factors_size->data[0] = batch_size;
    if (!TfLiteIntArrayEqual(scaling_factors->dims, scaling_factors_size)) {
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }

    // Used to store dequantized weights_time matrix for hybrid computation of
    // matmul(activation_state, weights_time), which occurs in floating point.
    node->temporaries->data[3] = scratch_tensor_index + 3;
    TfLiteTensor* float_weights_time = GetTemporary(context, node, /*index=*/3);
    float_weights_time->type = kTfLiteFloat32;
    // Persistent so that we can compute the dequantized weights only once.
    float_weights_time->allocation_type = kTfLiteArenaRwPersistent;
    if (!TfLiteIntArrayEqual(float_weights_time->dims, weights_time->dims)) {
      TfLiteIntArray* float_weights_time_size =
          TfLiteIntArrayCopy(weights_time->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, float_weights_time,
                                              float_weights_time_size));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       const TfLiteTensor* input,
                       const TfLiteTensor* weights_feature,
                       const TfLiteTensor* weights_time,
                       const TfLiteTensor* bias, const TfLiteSVDFParams* params,
                       TfLiteTensor* scratch, TfLiteTensor* state,
                       TfLiteTensor* output) {
  const int rank = params->rank;
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  const int num_filters = weights_feature->dims->data[0];
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  // Clear the activation (state left most column).
  // TODO(ghodrat): Add a test which initialize activation_state with invalid
  // values in left most column and make sure it passes.
  for (int b = 0; b < batch_size; ++b) {
    float* state_ptr_batch = state->data.f + b * memory_size * num_filters;
    for (int c = 0; c < num_filters; ++c) {
      float* state_ptr = state_ptr_batch + c * memory_size;
      state_ptr[memory_size - 1] = 0.0f;
    }
  }

  // Compute conv1d(inputs, weights_feature).
  // The state right most column is used to save current cycle activation. This
  // is achieved by starting at state->data.f[memory_size - 1] and having the
  // stride equal to memory_size.
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      weights_feature->data.f, num_filters, input_size, input->data.f,
      batch_size, &state->data.f[memory_size - 1], memory_size);

  ApplyTimeWeightsBiasAndActivation(batch_size, memory_size, num_filters,
                                    num_units, rank, weights_time, bias,
                                    params->activation, state, scratch, output);
  return kTfLiteOk;
}

TfLiteStatus EvalHybrid(
    TfLiteContext* context, TfLiteNode* node, const TfLiteTensor* input,
    const TfLiteTensor* weights_feature, const TfLiteTensor* weights_time,
    const TfLiteTensor* bias, const TfLiteSVDFParams* params,
    TfLiteTensor* scratch, TfLiteTensor* scaling_factors,
    TfLiteTensor* input_quantized, TfLiteTensor* state, TfLiteTensor* output) {
  const int rank = params->rank;
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  const int num_filters = weights_feature->dims->data[0];
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  // Initialize the pointer to input.
  const float* input_ptr_batch = input->data.f;

  // Initialize the pointer to storage for quantized values and
  // scaling factors.
  int8_t* quantized_input_ptr_batch =
      reinterpret_cast<int8_t*>(input_quantized->data.uint8);

  float* scaling_factors_ptr = scaling_factors->data.f;

  // Other initializations.
  const int8_t* weights_feature_ptr =
      reinterpret_cast<int8_t*>(weights_feature->data.uint8);
  const float weights_feature_scale = weights_feature->params.scale;

  // Clear the activation (state left most column).
  // TODO(ghodrat): Add a test which initialize state with invalid values in
  // the left most column and make sure it passes.
  for (int b = 0; b < batch_size; ++b) {
    float* state_ptr_batch = state->data.f + b * memory_size * num_filters;
    for (int c = 0; c < num_filters; ++c) {
      float* state_ptr = state_ptr_batch + c * memory_size;
      state_ptr[memory_size - 1] = 0.0;
    }
  }

  if (!tensor_utils::IsZeroVector(input_ptr_batch, batch_size * input_size)) {
    // Quantize input from float to int8.
    float unused_min, unused_max;
    for (int b = 0; b < batch_size; ++b) {
      const int offset = b * input_size;
      tensor_utils::SymmetricQuantizeFloats(
          input_ptr_batch + offset, input_size,
          quantized_input_ptr_batch + offset, &unused_min, &unused_max,
          &scaling_factors_ptr[b]);
      scaling_factors_ptr[b] *= weights_feature_scale;
    }

    // Compute conv1d(inputs, weights_feature).
    // The rightmost column of state is used to save the current cycle
    // activation.
    // This is achieved by starting at state->data.f[memory_size - 1]
    // and having the stride equal to memory_size.
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        weights_feature_ptr, num_filters, input_size, quantized_input_ptr_batch,
        scaling_factors_ptr, batch_size, &state->data.f[memory_size - 1],
        memory_size);
  }

  // TODO(alanchiao): can optimize hybrid case ~5% by unrolling loop in applying
  // time weights so that the inner loop multiplies eight elements at a time.
  ApplyTimeWeightsBiasAndActivation(batch_size, memory_size, num_filters,
                                    num_units, rank, weights_time, bias,
                                    params->activation, state, scratch, output);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights_feature =
      GetInput(context, node, kWeightsFeatureTensor);
  const TfLiteTensor* weights_time =
      GetInput(context, node, kWeightsTimeTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);

  TfLiteTensor* scratch = GetTemporary(context, node, /*index=*/0);

  TfLiteTensor* activation_state =
      &context->tensors[op_data->activation_state_tensor_index];
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (weights_feature->type) {
    case kTfLiteFloat32: {
      return EvalFloat(context, node, input, weights_feature, weights_time,
                       bias, params, scratch, activation_state, output);
      break;
    }
    case kTfLiteUInt8: {
      TfLiteTensor* input_quantized = GetTemporary(context, node, /*index=*/1);
      TfLiteTensor* scaling_factors = GetTemporary(context, node, /*index=*/2);
      TfLiteTensor* float_weights_time =
          GetTemporary(context, node, /*index=*/3);

      // Dequantize weights time.
      // TODO(alanchiao): this dequantization initialization only needs to
      // happen once per model and should theoretically be placed in either Init
      // or Prepare. However, TFLite doesn't allocate float_weights_time until
      // the Eval function.
      // TODO(alanchiao): refactor logic out into dequantize function.
      if (!op_data->float_weights_time_initialized) {
        const float dequantization_scale = weights_time->params.scale;
        const int8_t* weights_time_ptr =
            reinterpret_cast<int8_t*>(weights_time->data.uint8);
        for (int i = 0; i < NumElements(float_weights_time); ++i) {
          float_weights_time->data.f[i] =
              weights_time_ptr[i] * dequantization_scale;
        }
        op_data->float_weights_time_initialized = true;
      }
      return EvalHybrid(context, node, input, weights_feature,
                        float_weights_time, bias, params, scratch,
                        scaling_factors, input_quantized, activation_state,
                        output);
      break;
    }
    default:
      context->ReportError(context, "Type %d not currently supported.",
                           weights_feature->type);
      return kTfLiteError;
  }
}

}  // namespace svdf

TfLiteRegistration* Register_SVDF() {
  static TfLiteRegistration r = {svdf::Init, svdf::Free, svdf::Prepare,
                                 svdf::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
