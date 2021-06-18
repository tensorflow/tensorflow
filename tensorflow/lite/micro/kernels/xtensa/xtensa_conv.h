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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_CONV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_CONV_H_

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/conv.h"

namespace tflite {
struct XtensaConvOpData {
  OpDataConv reference_op_data;

#if defined(FUSION_F1) || defined(HIFI5)
  int scratch_tensor_index;
#endif  // defined(FUSION_F1) || defined(HIFI5)
};

#if defined(HIFIMINI)
void ConvEvalHifiMini(const ConvParams& params,
                      const int32_t* output_multiplier,
                      const int32_t* output_shift,
                      const RuntimeShape& input_shape, const int8_t* input_data,
                      const RuntimeShape& filter_shape,
                      const int8_t* filter_data, const RuntimeShape& bias_shape,
                      const int32_t* bias_data,
                      const RuntimeShape& output_shape, int8_t* output_data);

void Conv1x32Input32x32FilterHifiMini(
    const int input_offset, const int output_offset,
    const int quantized_activation_min, const int quantized_activation_max,
    const int32_t* output_multiplier, const int32_t* output_shift,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data);

#elif defined(FUSION_F1) || defined(HIFI5)
TfLiteStatus ConvPrepareHifi(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus ConvEvalHifi(TfLiteContext* context, TfLiteNode* node,
                          const TfLiteConvParams& params,
                          const XtensaConvOpData& data,
                          const TfLiteEvalTensor* input,
                          const TfLiteEvalTensor* filter,
                          const TfLiteEvalTensor* bias,
                          TfLiteEvalTensor* output);
#endif

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_CONV_H_
