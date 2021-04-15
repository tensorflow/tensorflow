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
/*
* Copyright (c) 2020 Cadence Design Systems Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_DEPTHWISE_CONV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_DEPTHWISE_CONV_H_

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "conv.h"

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
