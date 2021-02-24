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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_CONV_CONV_OPS_UINT8_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_CONV_CONV_OPS_UINT8_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/generic_fast/conv/conv_core.h"
#include "tensorflow/lite/micro/kernels/generic_fast/conv/conv_op_data.h"

namespace tflite {
namespace {

TfLiteStatus EvalConvUInt8WithoutPadding(TfLiteConvParams* params, OpData* data,
                                         const TfLiteEvalTensor* input,
                                         const TfLiteEvalTensor* filter,
                                         const TfLiteEvalTensor* bias,
                                         TfLiteEvalTensor* output,
                                         TfLiteContext* context) {
  return EvalConvWithoutPadding<uint8_t>(params, data, input, filter, bias,
                                         output, context);
}

TfLiteStatus EvalConvUInt8WithPadding(TfLiteConvParams* params, OpData* data,
                                      const TfLiteEvalTensor* input,
                                      const TfLiteEvalTensor* filter,
                                      const TfLiteEvalTensor* bias,
                                      TfLiteEvalTensor* output,
                                      TfLiteContext* context) {
  return EvalConvWithPadding<uint8_t>(params, data, input, filter, bias, output,
                                      context);
}

/*
 * Whether or not we need padding/no-padding variants depends on the target.
 * Ideally we just define NoPadding as an alias of Padding variant if a single
 * variant is o.k. however, this would be horribly non-portable/toolchain
 * dependent so we live with this slightly clunky despatcher.
 */

TfLiteStatus EvalConvUInt8(TfLiteConvParams* params, OpData* data,
                           const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* filter,
                           const TfLiteEvalTensor* bias,
                           TfLiteEvalTensor* output, TfLiteContext* context) {
  bool use_padding =
      (data->padding.height != 0 || data->padding.width != 0 ||
       data->padding.height_offset != 0 || data->padding.width_offset != 0);
  if (use_padding) {
    return EvalConvUInt8WithPadding(params, data, input, filter, bias, output,
                                    context);
  } else {
    return EvalConvUInt8WithoutPadding(params, data, input, filter, bias,
                                       output, context);
  }
}

}  // namespace
}  // namespace tflite

#endif /* TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_CONV_CONV_OPS_UINT8_H_ */
