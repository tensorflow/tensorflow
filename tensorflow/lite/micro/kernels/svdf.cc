/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/activation_utils.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace svdf {
namespace {

// These constants represent constants specific to the hotword "OK G" model.
// They exist until (b/132070898) is fixed.
constexpr int kScratchTensorMaxSize = 64;

struct OpData {
  int32 effective_scale_1_a;
  int32 effective_scale_2_a;
  // b versions of each scale are kept at int since the numbers are just the
  // shift value - typically between [-32, 32].
  int effective_scale_1_b;
  int effective_scale_2_b;
};

/**
 * This version of SVDF is specific to TFLite Micro. It contains the following
 * differences between the TFLite version:
 *
 * 1.) Scratch tensor allocation - scratch tensors must be known ahead of time
 * for the Micro interpreter.
 * 2.) Output dimensions - the TFLite version determines output size and runtime
 * and resizes the output tensor. Micro runtime does not support tensor
 * resizing.
 */

static inline void ApplyTimeWeightsBiasAndActivation(
    int batch_size, int memory_size, int num_filters, int num_units, int rank,
    const TfLiteTensor* weights_time, const TfLiteTensor* bias,
    TfLiteFusedActivation activation, TfLiteTensor* activation_state,
    TfLiteTensor* scratch, TfLiteTensor* output) {
  // Compute matmul(activation_state, weights_time).
  // The rightmost column is used to save temporary output (with the size of
  // num_filters). This is achieved by starting at
  // GetTensorData<float>(activation_state), and having the stride equal to
  // memory_size.
  for (int b = 0; b < batch_size; ++b) {
    // Perform batched vector dot product:
    float* scratch_ptr_batch = GetTensorData<float>(scratch) + b * num_filters;
    const float* vector1_ptr = GetTensorData<float>(weights_time);
    const float* vector2_ptr =
        GetTensorData<float>(activation_state) + b * memory_size * num_filters;
    for (int i = 0; i < num_filters; ++i) {
      *scratch_ptr_batch = 0.f;
      for (int j = 0; j < memory_size; ++j) {
        *scratch_ptr_batch += *vector1_ptr++ * *vector2_ptr++;
      }
      scratch_ptr_batch++;
    }
  }

  // Initialize output with bias if provided.
  if (bias) {
    // VectorBatchVectorAssign
    const float* bias_data = GetTensorData<float>(bias);
    float* output_data = GetTensorData<float>(output);
    for (int i = 0; i < batch_size; ++i) {
      float* output_ptr = output_data + i * num_units;
      const float* bias_ptr = bias_data;
      for (int j = 0; j < num_units; ++j) {
        *output_ptr++ = *bias_ptr++;
      }
    }
  } else {
    float* output_data = GetTensorData<float>(output);
    for (int i = 0; i < batch_size * num_units; ++i) {
      *output_data++ = 0.0f;
    }
  }

  // Reduction sum.
  for (int b = 0; b < batch_size; ++b) {
    float* output_ptr_batch = GetTensorData<float>(output) + b * num_units;
    float* scratch_ptr_batch = GetTensorData<float>(scratch) + b * num_filters;

    // Reduction sum vector
    for (int i = 0; i < num_units; ++i) {
      for (int j = 0; j < rank; j++) {
        output_ptr_batch[i] += *scratch_ptr_batch++;
      }
    }
  }

  // Apply activation.
  for (int b = 0; b < batch_size; ++b) {
    float* output_ptr_batch = GetTensorData<float>(output) + b * num_units;
    for (int i = 0; i < num_units; ++i) {
      *output_ptr_batch = ActivationValFloat(activation, *output_ptr_batch);
      ++output_ptr_batch;
    }
  }

  // Left shift the activation_state to make room for next cycle's activation.
  // TODO(alanchiao): explore collapsing this into a single loop.
  for (int b = 0; b < batch_size; ++b) {
    float* state_ptr_batch =
        GetTensorData<float>(activation_state) + b * memory_size * num_filters;
    for (int f = 0; f < num_filters; ++f) {
      // Shift the vector left:
      float* batch_ptr = state_ptr_batch;
      float* batch_start = state_ptr_batch + 1;
      float* batch_end = state_ptr_batch + memory_size;
      while (batch_start != batch_end) {
        *batch_ptr++ = *batch_start++;
      }
      state_ptr_batch[memory_size - 1] = 0.0f;
      state_ptr_batch += memory_size;
    }
  }
}

inline void EvalFloatSVDF(TfLiteContext* context, TfLiteNode* node,
                          const TfLiteTensor* input,
                          const TfLiteTensor* weights_feature,
                          const TfLiteTensor* weights_time,
                          const TfLiteTensor* bias,
                          const TfLiteSVDFParams* params, TfLiteTensor* scratch,
                          TfLiteTensor* activation_state,
                          TfLiteTensor* output) {
  const int rank = params->rank;
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  const int num_filters = weights_feature->dims->data[0];
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  // Clear the activation (activation_state's leftmost column).
  // TODO(ghodrat): Add a test which initialize activation_state with invalid
  // values in leftmost column and make sure it passes.
  for (int b = 0; b < batch_size; ++b) {
    float* state_ptr_batch =
        GetTensorData<float>(activation_state) + b * memory_size * num_filters;
    for (int c = 0; c < num_filters; ++c) {
      float* state_ptr = state_ptr_batch + c * memory_size;
      state_ptr[memory_size - 1] = 0.0f;
    }
  }

  // Compute conv1d(inputs, weights_feature).
  // The activation_state's rightmost column is used to save current cycle
  // activation. This is achieved by starting at
  // GetTensorData<float>(activation_state)[memory_size - 1] and having the
  // stride equal to memory_size.

  // Perform batched matrix vector multiply accumulate operation:
  const float* matrix = GetTensorData<float>(weights_feature);
  const float* vector = GetTensorData<float>(input);
  float* result = &GetTensorData<float>(activation_state)[memory_size - 1];
  float* result_in_batch = result;
  for (int i = 0; i < batch_size; ++i) {
    const float* matrix_ptr = matrix;
    for (int j = 0; j < num_filters; ++j) {
      float dot_prod = 0.0f;
      const float* vector_in_batch = vector + i * input_size;
      for (int k = 0; k < input_size; ++k) {
        dot_prod += *matrix_ptr++ * *vector_in_batch++;
      }
      *result_in_batch += dot_prod;
      result_in_batch += memory_size;
    }
  }

  ApplyTimeWeightsBiasAndActivation(
      batch_size, memory_size, num_filters, num_units, rank, weights_time, bias,
      params->activation, activation_state, scratch, output);
}

void EvalIntegerSVDF(
    TfLiteContext* context, TfLiteNode* node, const TfLiteTensor* input_tensor,
    const TfLiteTensor* weights_feature_tensor,
    const TfLiteTensor* weights_time_tensor, const TfLiteTensor* bias_tensor,
    const TfLiteSVDFParams* params, TfLiteTensor* activation_state_tensor,
    TfLiteTensor* output_tensor, int32_t scale_1_a, int scale_1_b,
    int32_t scale_2_a, int scale_2_b, int32_t input_zp, int32_t output_zp) {
  const int n_rank = params->rank;
  const int n_batch = input_tensor->dims->data[0];
  const int n_input = input_tensor->dims->data[1];
  const int n_filter = weights_feature_tensor->dims->data[0];
  const int n_unit = n_filter / n_rank;
  const int n_memory = weights_time_tensor->dims->data[1];

  // TODO(b/132070898): Move these temp variables to the new scratch buffer API
  // when ready.
  int32_t scratch_tensor[kScratchTensorMaxSize];
  int32_t scratch_output_tensor[kScratchTensorMaxSize];

  // Rewrite last bit of state.
  {
    for (int b = 0; b < n_batch; ++b) {
      int16_t* state_ptr_batch =
          GetTensorData<int16_t>(activation_state_tensor) +
          b * n_memory * n_filter;
      for (int c = 0; c < n_filter; ++c) {
        int16_t* state_ptr = state_ptr_batch + c * n_memory;
        state_ptr[n_memory - 1] = 0;
      }
    }
  }

  // Feature matmul.
  {
    int16_t* state = GetTensorData<int16_t>(activation_state_tensor);
    const int8_t* input = GetTensorData<int8_t>(input_tensor);
    const int8_t* weight_feature =
        GetTensorData<int8_t>(weights_feature_tensor);
    const int32_t output_max = std::numeric_limits<int16_t>::max();
    const int32_t output_min = std::numeric_limits<int16_t>::min();
    int16_t* result_in_batch = state + (n_memory - 1);
    for (int b = 0; b < n_batch; b++) {
      const int8_t* matrix_ptr = weight_feature;
      for (int r = 0; r < n_filter; r++) {
        int32_t dot_prod = 0;
        const int8_t* vector_in_batch = input + b * n_input;
        for (int c = 0; c < n_input; c++) {
          dot_prod += *matrix_ptr++ * (*vector_in_batch++ - input_zp);
        }
        dot_prod =
            MultiplyByQuantizedMultiplier(dot_prod, scale_1_a, scale_1_b);
        dot_prod = std::min(std::max(output_min, dot_prod), output_max);
        *result_in_batch = dot_prod;
        result_in_batch += n_memory;
      }
    }
  }

  // Time.
  {
    for (int b = 0; b < n_batch; ++b) {
      int32_t* scratch_ptr_batch = scratch_tensor + b * n_filter;

      // Perform batched vector dot product:
      const int16_t* vector1_ptr = GetTensorData<int16_t>(weights_time_tensor);
      const int16_t* vector2_ptr =
          GetTensorData<int16_t>(activation_state_tensor) +
          b * n_memory * n_filter;

      for (int i = 0; i < n_filter; i++) {
        *scratch_ptr_batch = 0;
        for (int j = 0; j < n_memory; j++) {
          *scratch_ptr_batch += *vector1_ptr++ * *vector2_ptr++;
        }
        scratch_ptr_batch++;
      }
    }
  }

  // Reduce, add bias, rescale, activation.
  {
    // Add bias.
    if (bias_tensor) {
      // Vector batch assign:
      const int32_t* bias_data = GetTensorData<int32_t>(bias_tensor);
      for (int i = 0; i < n_batch; ++i) {
        int32_t* output_ptr = scratch_output_tensor + i * n_unit;
        const int32_t* bias_ptr = bias_data;
        for (int j = 0; j < n_unit; ++j) {
          *output_ptr++ = *bias_ptr++;
        }
      }
    } else {
      int32_t* output_ptr = scratch_output_tensor;
      for (int i = 0; i < n_batch * n_unit; ++i) {
        *output_ptr++ = 0;
      }
    }

    // Reduce.
    for (int b = 0; b < n_batch; ++b) {
      int32_t* output_temp_ptr = scratch_output_tensor + b * n_unit;
      int32_t* scratch_ptr_batch = scratch_tensor + b * n_filter;

      // Reduction sum vector
      for (int i = 0; i < n_unit; ++i) {
        for (int j = 0; j < n_rank; ++j) {
          output_temp_ptr[i] += *scratch_ptr_batch++;
        }
      }
    }

    // Rescale.
    const int32_t output_max = std::numeric_limits<int8_t>::max();
    const int32_t output_min = std::numeric_limits<int8_t>::min();
    for (int i = 0; i < n_batch * n_unit; ++i) {
      int32_t x1 = scratch_output_tensor[i];
      int32_t x2 = MultiplyByQuantizedMultiplier(x1, scale_2_a, scale_2_b);
      int32_t x3 = x2 + output_zp;
      int32_t x4 = std::min(std::max(output_min, x3), output_max);
      GetTensorData<int8_t>(output_tensor)[i] = static_cast<int8_t>(x4);
    }
  }

  // Shift state.
  {
    for (int b = 0; b < n_batch; ++b) {
      int16_t* state_ptr_batch =
          GetTensorData<int16_t>(activation_state_tensor) +
          b * n_memory * n_filter;
      for (int f = 0; f < n_filter; ++f) {
        // Shift the vector left:
        int16_t* batch_ptr = state_ptr_batch;
        int16_t* batch_start = state_ptr_batch + 1;
        int16_t* batch_end = state_ptr_batch + n_memory;
        while (batch_start != batch_end) {
          *batch_ptr++ = *batch_start++;
        }
        state_ptr_batch[n_memory - 1] = 0;
        state_ptr_batch += n_memory;
      }
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
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);

  // Validate Tensor Inputs (dtype depends on quantization):
  // [0] = Input, {2, batch_size, input_size}
  // [1] = Weights Feature, {2, num_filters, input_size}
  // [2] = Weights Time, {2, num_filters, memory_size}
  // [3] = Bias (optional), {1, num_units}
  // [4] = Activation State (variable),
  //         {2, batch_size, memory_size * num_filters}

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights_feature =
      GetInput(context, node, kWeightsFeatureTensor);
  const TfLiteTensor* weights_time =
      GetInput(context, node, kWeightsTimeTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* activation_state =
      &context->tensors[node->inputs->data[kInputActivationStateTensor]];

  // Define input constants based on input tensor definition above:
  const int rank = params->rank;
  const int input_size = input->dims->data[1];
  const int batch_size = input->dims->data[0];
  const int num_filters = weights_feature->dims->data[0];
  TF_LITE_ENSURE_EQ(context, num_filters % rank, 0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  const bool is_full_integer = input->type == kTfLiteInt8;

  // Validate Input Tensor:
  TF_LITE_ENSURE(context,
                 input->type == kTfLiteFloat32 || input->type == kTfLiteInt8);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 2);

  // Validate Tensor Output:
  // [0] = float/int8, {2, batch_size, num_units}
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 2);
  TF_LITE_ENSURE_EQ(context, output->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, output->dims->data[1], num_units);

  // Validate Weights Feature Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights_feature), 2);
  TF_LITE_ENSURE_EQ(context, weights_feature->dims->data[1], input_size);

  // Validate Weights Time Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights_time), 2);
  TF_LITE_ENSURE_EQ(context, weights_time->dims->data[0], num_filters);
  TF_LITE_ENSURE_EQ(context, weights_time->dims->data[1], memory_size);

  // Validate Optional Bias Input Tensor:
  if (bias) {
    TF_LITE_ENSURE_EQ(context, bias->dims->data[0], num_units);
  }

  // Validate Activation State Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(activation_state), 2);
  TF_LITE_ENSURE_EQ(context, activation_state->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, activation_state->dims->data[1],
                    memory_size * num_filters);

  if (is_full_integer) {
    TF_LITE_ENSURE_EQ(context, node->inputs->size, 5);

    TF_LITE_ENSURE_EQ(context, weights_feature->type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(context, weights_time->type, kTfLiteInt16);

    if (bias) {
      TF_LITE_ENSURE_EQ(context, bias->type, kTfLiteInt32);
    }

    TF_LITE_ENSURE_EQ(context, activation_state->type, kTfLiteInt16);

    // Validate Scratch Tensors:
    // [0] = (shared - see float block below for usage)
    // [1] = Output Temp, int8_t, {2, num_units, batch_size}
    // TODO(b/132070898): Scratch values are used as stack variables in
    // EvalIntegerSVDF().

    // Validate output tensor:
    TF_LITE_ENSURE_EQ(context, output->type, kTfLiteInt8);
  } else {
    TF_LITE_ENSURE_EQ(context, node->inputs->size, 6);

    // Validate Input Tensor dtypes:
    TF_LITE_ENSURE_EQ(context, weights_feature->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, weights_time->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, activation_state->type, kTfLiteFloat32);

    if (bias) {
      TF_LITE_ENSURE_EQ(context, bias->type, kTfLiteFloat32);
    }

    // Validate shared Scratch Tensor:
    // [0] = Holds dot-product of time-forward calculations in
    //       ApplyTimeWeightsBiasAndActivation():
    //         float/int32, {2, batch_size, num_filters}
    // TODO(b/132070898): Use input tensor as variable until scratch tensor
    // allocation has been implemented (b/132070898) TfLiteTensor*
    // scratch_tensor = GetTemporary(context, node, 0);
    TfLiteTensor* scratch_tensor = &context->tensors[node->inputs->data[5]];
    TF_LITE_ENSURE_EQ(context, scratch_tensor->type, kTfLiteFloat32);

    TF_LITE_ENSURE_EQ(context, NumDimensions(scratch_tensor), 2);
    TF_LITE_ENSURE_EQ(context, scratch_tensor->dims->data[0], batch_size);
    TF_LITE_ENSURE_EQ(context, scratch_tensor->dims->data[1], num_filters);

    // Full-float SVDF only uses the one shared scratch tensor (see above for
    // usage).
    // TODO(b/132070898): Use input tensor as variable until scratch tensor
    // allocation has been implemented.
    // TF_LITE_ENSURE_EQ(context, node->temporaries->size, 1);
    TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights_feature =
      GetInput(context, node, kWeightsFeatureTensor);
  const TfLiteTensor* weights_time =
      GetInput(context, node, kWeightsTimeTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* activation_state =
      &context->tensors[node->inputs->data[kInputActivationStateTensor]];
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  const bool is_full_integer = input->type == kTfLiteInt8;

  switch (weights_feature->type) {
    case kTfLiteFloat32: {
      // TODO(b/132070898): Use input tensor as variable until scratch tensor
      // allocation has been implemented. TfLiteTensor* scratch =
      // GetTemporary(context, node, /*index=*/0);
      TfLiteTensor* scratch = &context->tensors[node->inputs->data[5]];
      EvalFloatSVDF(context, node, input, weights_feature, weights_time, bias,
                    params, scratch, activation_state, output);
      return kTfLiteOk;
      break;
    }

    case kTfLiteInt8: {
      if (is_full_integer) {
        // TODO(b/132070898): Store these values in ::Prepare() instead of
        // ::Eval():
        // Calculate effective scales.
        OpData op_data;
        auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
            input->quantization.params);
        auto* weights_feature_params =
            reinterpret_cast<TfLiteAffineQuantization*>(
                weights_feature->quantization.params);
        auto* state_params = reinterpret_cast<TfLiteAffineQuantization*>(
            activation_state->quantization.params);
        auto* weight_time_params = reinterpret_cast<TfLiteAffineQuantization*>(
            weights_time->quantization.params);
        auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
            output->quantization.params);
        const double effective_scale_1 =
            static_cast<double>(input_params->scale->data[0] *
                                weights_feature_params->scale->data[0] /
                                state_params->scale->data[0]);
        const double effective_scale_2 = static_cast<double>(
            state_params->scale->data[0] * weight_time_params->scale->data[0] /
            output_params->scale->data[0]);
        QuantizeMultiplier(effective_scale_1, &op_data.effective_scale_1_a,
                           &op_data.effective_scale_1_b);
        QuantizeMultiplier(effective_scale_2, &op_data.effective_scale_2_a,
                           &op_data.effective_scale_2_b);

        TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActRelu);
        EvalIntegerSVDF(
            context, node, input, weights_feature, weights_time, bias, params,
            activation_state, output, op_data.effective_scale_1_a,
            op_data.effective_scale_1_b, op_data.effective_scale_2_a,
            op_data.effective_scale_2_b, input->params.zero_point,
            output->params.zero_point);
        return kTfLiteOk;
      }
      break;
    }

    default:
      context->ReportError(context, "Type %s not currently supported.",
                           TfLiteTypeGetName(weights_feature->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace svdf

TfLiteRegistration* Register_SVDF() {
  static TfLiteRegistration r = {svdf::Init, svdf::Free, svdf::Prepare,
                                 svdf::Eval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
