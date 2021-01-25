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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_CONV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_CONV_H_

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

TfLiteStatus InvokeConv(TfLiteTensor* tensors, int tensors_size,
                        int output_length, TfLiteConvParams* conv_params,
                        TfLiteRegistration registration, float* output_data);

TfLiteStatus InvokeConv(TfLiteTensor* tensors, int tensors_size,
                        int output_length, TfLiteConvParams* conv_params,
                        TfLiteRegistration registration, int8_t* output_data);

TfLiteStatus InvokeConv(TfLiteTensor* tensors, int tensors_size,
                        int output_length, TfLiteConvParams* conv_params,
                        TfLiteRegistration registration, uint8_t* output_data);

TfLiteStatus ValidateConvGoldens(TfLiteTensor* tensors, int tensors_size,
                                 const float* expected_output_data,
                                 int output_length,
                                 TfLiteConvParams* conv_params,
                                 TfLiteRegistration registration,
                                 float* output_data, float tolerance = 1e-5);

TfLiteStatus ValidateConvGoldens(TfLiteTensor* tensors, int tensors_size,
                                 const int8_t* expected_output_data,
                                 int output_length,
                                 TfLiteConvParams* conv_params,
                                 TfLiteRegistration registration,
                                 int8_t* output_data, float tolerance = 1e-5);

TfLiteStatus ValidateConvGoldens(TfLiteTensor* tensors, int tensors_size,
                                 const uint8_t* expected_output_data,
                                 int output_length,
                                 TfLiteConvParams* conv_params,
                                 TfLiteRegistration registration,
                                 uint8_t* output_data, float tolerance = 1e-5);

TfLiteStatus TestConvFloat(const int* input_dims_data, const float* input_data,
                           const int* filter_dims_data,
                           const float* filter_data, const int* bias_dims_data,
                           const float* bias_data, const int* output_dims_data,
                           const float* expected_output_data,
                           TfLiteConvParams* conv_params,
                           TfLiteRegistration registration, float* output_data);

TfLiteStatus TestConvQuantizedPerLayer(
    const int* input_dims_data, const float* input_data,
    uint8_t* input_quantized, float input_scale, const int* filter_dims_data,
    const float* filter_data, uint8_t* filter_quantized, float filter_scale,
    const int* bias_dims_data, const float* bias_data, int32_t* bias_quantized,
    const int* output_dims_data, const float* expected_output_data,
    uint8_t* expected_output_quantized, float output_scale,
    TfLiteConvParams* conv_params, TfLiteRegistration registration,
    uint8_t* output_data);

TfLiteStatus TestConvQuantizedPerChannel(
    const int* input_dims_data, const float* input_data,
    int8_t* input_quantized, float input_scale, int input_zero_point,
    const int* filter_dims_data, const float* filter_data,
    int8_t* filter_data_quantized, const int* bias_dims_data,
    const float* bias_data, int32_t* bias_data_quantized, float* bias_scales,
    int* bias_zero_points, const int* output_dims_data,
    const float* expected_output_data, int8_t* expected_output_data_quantized,
    float output_scale, int output_zero_point, TfLiteConvParams* conv_params,
    TfLiteRegistration registration, int8_t* output_data);

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_CONV_H_
