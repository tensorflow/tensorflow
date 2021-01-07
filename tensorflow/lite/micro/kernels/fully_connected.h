/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_FULLY_CONNECTED_H_

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

struct OpDataFullyConnectedReference {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // The index of the temporary tensor where the quantized inputs are cached.
  int input_quantized_index;
  // Cached zero point values of tensors.
  int32_t input_zero_point;
  int32_t filter_zero_point;
  int32_t output_zero_point;

  // Returns a FullyConnectedParams struct with all the parameters needed for a
  // quantized fully connected computation.
  FullyConnectedParams ToQuantizedParams() const {
    FullyConnectedParams op_params;
    op_params.input_offset = -input_zero_point;
    op_params.weights_offset = -filter_zero_point;
    op_params.output_offset = output_zero_point;
    op_params.output_multiplier = output_multiplier;
    op_params.output_shift = output_shift;
    op_params.quantized_activation_min = output_activation_min;
    op_params.quantized_activation_max = output_activation_max;
    return op_params;
  }
};

extern const int kFullyConnectedInputTensor;
extern const int kFullyConnectedWeightsTensor;
extern const int kFullyConnectedBiasTensor;
extern const int kFullyConnectedOutputTensor;

TfLiteStatus CalculateOpDataFullyConnectedReference(
    TfLiteContext* context, TfLiteFusedActivation activation,
    TfLiteType data_type, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output,
    OpDataFullyConnectedReference* data);

void* InitFullyConnectedReference(TfLiteContext* context, const char* buffer,
                                  size_t length);

TfLiteStatus EvalFloatFullyConnectedReference(
    TfLiteContext* context, TfLiteNode* node, TfLiteFusedActivation activation,
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output);

TfLiteStatus EvalQuantizedInt8FullyConnectedReference(
    TfLiteContext* context, TfLiteNode* node,
    const OpDataFullyConnectedReference& data, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output);

TfLiteStatus EvalQuantizedFullyConnectedReference(
    TfLiteContext* context, TfLiteNode* node,
    const OpDataFullyConnectedReference& data, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output);

// This is the most generic TfLiteRegistration. The actual supported types may
// still be target dependent. The only requirement is that every implementation
// (reference or optimized) must define this function.
TfLiteRegistration Register_FULLY_CONNECTED();

#if defined(CMSIS_NN) || defined(ARDUINO)
// The Arduino is a special case where we use the CMSIS kernels, but because of
// the current approach to building for Arduino, we do not support -DCMSIS_NN as
// part of the build. As a result, we use defined(ARDUINO) as proxy for the
// CMSIS kernels for this one special case.

// Returns a TfLiteRegistration struct for cmsis-nn kernel variant that only
// supports int8.
TfLiteRegistration Register_FULLY_CONNECTED_INT8();

#else
// Note that while this block gets used for both reference and optimized kernels
// that do not have any specialized implementations, the only goal here is to
// define fallback implementation that allow reference kernels to still be used
// from applications that call a more specific kernel variant.

inline TfLiteRegistration Register_FULLY_CONNECTED_INT8() {
  return Register_FULLY_CONNECTED();
}

#endif
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_FULLY_CONNECTED_H_
