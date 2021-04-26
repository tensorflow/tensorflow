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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_DEPTHWISE_CONV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_DEPTHWISE_CONV_H_

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/conv.h"

namespace tflite {

extern const int kDepthwiseConvInputTensor;
extern const int kDepthwiseConvWeightsTensor;
extern const int kDepthwiseConvBiasTensor;
extern const int kDepthwiseConvOutputTensor;
extern const int kDepthwiseConvQuantizedDimension;

// Returns a DepthwiseParams struct with all the parameters needed for a
// float computation.
DepthwiseParams DepthwiseConvParamsFloat(
    const TfLiteDepthwiseConvParams& params, const OpDataConv& data);

// Returns a DepthwiseParams struct with all the parameters needed for a
// quantized computation.
DepthwiseParams DepthwiseConvParamsQuantized(
    const TfLiteDepthwiseConvParams& params, const OpDataConv& data);

TfLiteStatus CalculateOpDataDepthwiseConv(
    TfLiteContext* context, TfLiteNode* node,
    const TfLiteDepthwiseConvParams& params, int width, int height,
    int filter_width, int filter_height, int out_width, int out_height,
    const TfLiteType data_type, OpDataConv* data);

TfLiteStatus DepthwiseConvPrepare(TfLiteContext* context, TfLiteNode* node);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_DEPTHWISE_CONV_H_
