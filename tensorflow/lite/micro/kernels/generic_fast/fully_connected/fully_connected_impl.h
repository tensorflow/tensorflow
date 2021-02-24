/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_FULLY_CONNECTED_FULLY_CONNECTED_IMPL_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_FULLY_CONNECTED_FULLY_CONNECTED_IMPL_H_

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/generic_fast/fully_connected/fully_connected_op_data.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

/*
 * Calculates the OpData which stores all important metadata about the kernel
 * and parameters.
 */
TfLiteStatus CalculateOpData(TfLiteContext* context,
                             TfLiteFullyConnectedParams* params,
                             TfLiteType data_type, const TfLiteTensor* input,
                             const TfLiteTensor* weights,
                             const TfLiteTensor* bias, TfLiteTensor* output,
                             OpData* data) {
  TfLiteStatus status = kTfLiteOk;
  if (data_type != kTfLiteFloat32) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, weights, bias, output, &real_multiplier));
    int exponent;
    QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
    data->output_shift = -exponent;
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
  }
  return status;
}

/*
 * Precomputes a factor from the weights and offsets, which needs to be
 * calculated only once, not in every invocation.
 */
template <typename T>
void PrecomputeSumOfWeightsFactor(const int32_t* bias, const T* weights,
                                  int32_t* sum_of_weights_factor, int cols,
                                  int rows, int32_t weights_offset,
                                  int32_t input_offset) {
  for (int row = 0; row < rows; row++) {
    int32_t sum_of_weights = 0;
    for (int col = 0; col < cols; col++) {
      sum_of_weights += weights[col];
    }
    weights += cols;
    sum_of_weights_factor[row] =
        (sum_of_weights + cols * weights_offset) * input_offset;
    if (bias) {
      sum_of_weights_factor[row] += bias[row];
    }
  }
}

/*
 * Init function is called once at the beginning to initialize kernels and
 * allocate memory.
 */
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  void* raw = context->AllocatePersistentBuffer(context, sizeof(OpData));
  OpData* data = reinterpret_cast<OpData*>(raw);
  *data = {};
  return raw;
}

void Free(TfLiteContext* context, void* buffer) {}

/*
 * Evaluation function. Called in every invocation.
 */
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  OpData* opData = reinterpret_cast<OpData*>(node->user_data);

  return opData->eval_function(context, params, opData, input, weights, bias,
                               output);
}

}  // namespace
}  // namespace tflite

#endif // TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_FULLY_CONNECTED_FULLY_CONNECTED_IMPL_H_
