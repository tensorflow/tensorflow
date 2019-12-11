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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_
#define TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_

#include <stdint.h>

#include "tensorflow/lite/c/common.h"

namespace tflite {

// Returns number of elements in the shape array.

int ElementCount(const TfLiteIntArray& dims);

uint8_t FloatToAsymmetricQuantizedUInt8(const float value, const float scale,
                                        const int zero_point);

uint8_t FloatToSymmetricQuantizedUInt8(const float value, const float scale);

int8_t FloatToAsymmetricQuantizedInt8(const float value, const float scale,
                                      const int zero_point);

int8_t FloatToSymmetricQuantizedInt8(const float value, const float scale);

// Converts a float value into a signed thirty-two-bit quantized value.  Note
// that values close to max int and min int may see significant error due to
// a lack of floating point granularity for large values.
int32_t FloatToSymmetricQuantizedInt32(const float value, const float scale);

// Helper methods to quantize arrays of floats to the desired format.
//
// There are several key flavors of quantization in TfLite:
//        asymmetric symmetric  per channel
// int8  |     X    |    X    |     X      |
// uint8 |     X    |    X    |            |
// int32 |          |    X    |     X      |
//
// The per-op quantizaiton spec can be found here:
// https://www.tensorflow.org/lite/performance/quantization_spec

void AsymmetricQuantize(const float* input, int8_t* output, int num_elements,
                        float scale, int zero_point = 0);

void AsymmetricQuantize(const float* input, uint8_t* output, int num_elements,
                        float scale, int zero_point = 128);

void SymmetricQuantize(const float* input, int32_t* output, int num_elements,
                       float scale);

void SymmetricPerChannelQuantize(const float* input, int32_t* output,
                                 int num_elements, int num_channels,
                                 float* scales);

void SignedSymmetricPerChannelQuantize(const float* values,
                                       TfLiteIntArray* dims,
                                       int quantized_dimension,
                                       int8_t* quantized_values,
                                       float* scaling_factor);

void SignedSymmetricQuantize(const float* values, TfLiteIntArray* dims,
                             int8_t* quantized_values, float* scaling_factor);

void SymmetricQuantize(const float* values, TfLiteIntArray* dims,
                       uint8_t* quantized_values, float* scaling_factor);

void SymmetricDequantize(const int8_t* values, const int size,
                         const float dequantization_scale,
                         float* dequantized_values);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_
