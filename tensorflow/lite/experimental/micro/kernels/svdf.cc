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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/kernels/activation_utils.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace micro {
namespace svdf {
namespace {

// TODO(kreeger): upstream these reference methods into
// `lite/kernels/reference/svdf.h`

static inline void ApplyTimeWeightsBiasAndActivation(
    int batch_size, int memory_size, int num_filters, int num_units, int rank,
    const TfLiteTensor* weights_time, const TfLiteTensor* bias,
    TfLiteFusedActivation activation, TfLiteTensor* activation_state,
    TfLiteTensor* scratch, TfLiteTensor* output) {
  // Compute matmul(state, weights_time).
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
    // TODO(kreeger): doc me - VectorBatchVectorAssign
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
    const float* input_vector_ptr = scratch_ptr_batch;
    for (int i = 0; i < num_units; ++i) {
      for (int j = 0; j < rank; j++) {
        output_ptr_batch[i] += *input_vector_ptr++;
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
                          TfLiteTensor* state, TfLiteTensor* output) {
  const int rank = params->rank;
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  const int num_filters = weights_feature->dims->data[0];
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  // Clear the activation (state's leftmost column).
  // TODO(ghodrat): Add a test which initialize activation_state with invalid
  // values in leftmost column and make sure it passes.
  for (int b = 0; b < batch_size; ++b) {
    float* state_ptr_batch =
        GetTensorData<float>(state) + b * memory_size * num_filters;
    for (int c = 0; c < num_filters; ++c) {
      float* state_ptr = state_ptr_batch + c * memory_size;
      state_ptr[memory_size - 1] = 0.0f;
    }
  }

  // Compute conv1d(inputs, weights_feature).
  // The state's rightmost column is used to save current cycle activation. This
  // is achieved by starting at GetTensorData<float>(state)[memory_size - 1] and
  // having the stride equal to memory_size.

  // Perform batched matrix vector multiply accumulate operation:
  const float* matrix = GetTensorData<float>(weights_feature);
  const float* vector = GetTensorData<float>(input);
  float* result = &GetTensorData<float>(state)[memory_size - 1];
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

  ApplyTimeWeightsBiasAndActivation(batch_size, memory_size, num_filters,
                                    num_units, rank, weights_time, bias,
                                    params->activation, state, scratch, output);
}

}  // namespace

// Input tensors.
constexpr int kInputTensor = 0;
constexpr int kWeightsFeatureTensor = 1;
constexpr int kWeightsTimeTensor = 2;
constexpr int kBiasTensor = 3;
// This is a variable tensor, and will be modified by this op.
constexpr int kInputActivationStateTensor = 4;
constexpr int kScratchTensorEvalFloat = 5;

// Output tensor.
constexpr int kOutputTensor = 0;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // auto op_data = new OpData();
  // // TODO(kreeger): Handle hybrid quant b/137786105
  // op_data->float_weights_time_initialized = false;
  // return op_data;
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);

  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 5);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights_feature =
      GetInput(context, node, kWeightsFeatureTensor);
  const TfLiteTensor* weights_time =
      GetInput(context, node, kWeightsTimeTensor);

  // The weights are of consistent type, so it suffices to check one.
  const bool is_hybrid_op = IsHybridOp(input, weights_feature);

  // TODO(kreeger): Handle hybrid quant b/137786105
  // Note: only needs 4 scratch tensors when is_hybrid_op, only 1 otherwise.
  int scratch_tensor_index = kScratchTensorEvalFloat;
  TF_LITE_ENSURE_EQ(context, node->temporaries->size, 1);

  // TODO(kreeger): Handle this case for full quant svdf b/139435798
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  const int rank = params->rank;
  const int batch_size = input->dims->data[0];
  const int num_filters = weights_feature->dims->data[0];
  TF_LITE_ENSURE_EQ(context, num_filters % rank, 0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];
  TF_LITE_ENSURE_EQ(context, input->dims->data[1],
                    weights_feature->dims->data[1]);
  TF_LITE_ENSURE_EQ(context, weights_time->dims->data[0], num_filters);

  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  if (bias) {
    TF_LITE_ENSURE_EQ(context, bias->dims->data[0], num_units);
  }

  const int activation_state_tensor_index =
      node->inputs->data[kInputActivationStateTensor];
  TfLiteTensor* activation_state =
      &context->tensors[activation_state_tensor_index];

  // Check the shape of input state tensors.
  TF_LITE_ENSURE_EQ(context, NumDimensions(activation_state), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(activation_state, 0), batch_size);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(activation_state, 1),
                    memory_size * num_filters);

  node->temporaries->data[0] = scratch_tensor_index;

  if (is_hybrid_op) {
    // TODO(kreeger): Handle hybrid quant b/137786105
    return kTfLiteError;
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

  TfLiteTensor* scratch = GetTemporary(context, node, /*index=*/0);

  const int activation_state_tensor_index =
      node->inputs->data[kInputActivationStateTensor];
  TfLiteTensor* activation_state =
      &context->tensors[activation_state_tensor_index];
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (weights_feature->type) {
    case kTfLiteFloat32: {
      EvalFloatSVDF(context, node, input, weights_feature, weights_time, bias,
                    params, scratch, activation_state, output);
      return kTfLiteOk;
      break;
    }
    default:
      // TODO(kreeger): Handle hybrid quant b/137786105
      // TODO(kreeger): Handle this case for full quant svdf b/139435798
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
