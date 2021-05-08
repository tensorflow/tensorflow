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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_FULLY_CONNECTED_FULLY_CONNECTED_OPS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_FULLY_CONNECTED_FULLY_CONNECTED_OPS_H_

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/generic_fast/fully_connected/fully_connected_op_data.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

template <typename T>
TfLiteStatus EvalQuantized(TfLiteContext* context,
                           TfLiteFullyConnectedParams* params, OpData* opData,
                           const TfLiteTensor* input,
                           const TfLiteTensor* weights,
                           const TfLiteTensor* bias, TfLiteTensor* output) {
  // Get input info
  const T* input_data = GetTensorData<T>(input);

  // Get weights info
  const T* weights_data = GetTensorData<T>(weights);
  const int32_t weights_offset = -weights->params.zero_point;
  RuntimeShape weights_shape = GetTensorShape(weights);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);
  const int weights_dim_count = weights_shape.DimensionsCount();
  const int accum_depth = weights_shape.Dims(weights_dim_count - 1);

  // Get output info
  T* output_data = GetTensorData<T>(output);
  const int32_t output_offset = output->params.zero_point;
  RuntimeShape output_shape = GetTensorShape(output);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);
  const int32_t output_multiplier = opData->output_multiplier;
  const int output_shift = -opData->output_shift;
  const int32_t output_activation_min = opData->output_activation_min;
  const int32_t output_activation_max = opData->output_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  TFLITE_DCHECK_LE(output_depth, weights_shape.Dims(weights_dim_count - 2));

  // Get factor pre-computed in the Prepare-phase
  const int32_t* sum_of_weights_factor = opData->sum_of_weights_factor;

  for (int b = 0; b < batches; ++b) {
    // Pre-compute factor for this output-batch
    int32_t sum_of_inputs_factor = 0;
    if (weights_offset != 0) {
      for (int d = 0; d < accum_depth; ++d) {
        sum_of_inputs_factor += input_data[d];
      }
      sum_of_inputs_factor *= weights_offset;
    }
    // Calculate output-nodes using pre-computed factors
    KernelCore<T>::run(output_data, input_data, weights_data,
                       sum_of_weights_factor, sum_of_inputs_factor, accum_depth,
                       output_depth, output_offset, output_multiplier,
                       output_shift, output_activation_min,
                       output_activation_max);
    output_data += output_depth;
    input_data += accum_depth;
  }
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedInt8(TfLiteContext* context,
                               TfLiteFullyConnectedParams* params,
                               OpData* opData, const TfLiteTensor* input,
                               const TfLiteTensor* weights,
                               const TfLiteTensor* bias, TfLiteTensor* output) {
  return EvalQuantized<int8_t>(context, params, opData, input, weights, bias,
                               output);
}

TfLiteStatus EvalQuantizedUInt8(TfLiteContext* context,
                                TfLiteFullyConnectedParams* params,
                                OpData* opData, const TfLiteTensor* input,
                                const TfLiteTensor* weights,
                                const TfLiteTensor* bias,
                                TfLiteTensor* output) {
  return EvalQuantized<uint8_t>(context, params, opData, input, weights, bias,
                                output);
}

TfLiteStatus EvalQuantizedUint8WithOutputInt16(
    TfLiteContext* context, TfLiteFullyConnectedParams* params, OpData* opData,
    const TfLiteTensor* input, const TfLiteTensor* weights,
    const TfLiteTensor* bias, TfLiteTensor* output) {
  tflite::FullyConnectedParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.weights_offset = -weights->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.output_multiplier = opData->output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = -opData->output_shift;
  op_params.quantized_activation_min = opData->output_activation_min;
  op_params.quantized_activation_max = opData->output_activation_max;
  reference_ops::FullyConnected(
      op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
      GetTensorShape(weights), GetTensorData<uint8_t>(weights),
      GetTensorShape(bias), GetTensorData<int32_t>(bias),
      GetTensorShape(output), GetTensorData<int16_t>(output));
  return kTfLiteOk;
}

TfLiteStatus EvalFloat(TfLiteContext* context,
                       TfLiteFullyConnectedParams* params, OpData* opData,
                       const TfLiteTensor* input, const TfLiteTensor* weights,
                       const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  tflite::FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  tflite::reference_ops::FullyConnected(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(weights), GetTensorData<float>(weights),
      GetTensorShape(bias), GetTensorData<float>(bias), GetTensorShape(output),
      GetTensorData<float>(output));
  return kTfLiteOk;
}

}  // namespace
}  // namespace tflite

#endif // TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_FULLY_CONNECTED_FULLY_CONNECTED_OPS_H_
