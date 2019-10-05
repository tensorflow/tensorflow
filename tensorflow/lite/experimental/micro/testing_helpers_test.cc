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

#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(CreateQuantizedBiasTensor) {
  float input_scale = 0.5;
  float weight_scale = 0.5;
  const int tensor_size = 12;
  int dims_arr[] = {4, 2, 3, 2, 1};
  const char* tensor_name = "test_tensor";
  int32_t quantized[tensor_size];
  float pre_quantized[] = {-10, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 10};
  int32_t expected_quantized_values[] = {-40, -20, -16, -12, -8, -4,
                                         0,   4,   8,   12,  16, 40};
  TfLiteIntArray* dims = tflite::testing::IntArrayFromInts(dims_arr);

  TfLiteTensor result = tflite::testing::CreateQuantizedBiasTensor(
      pre_quantized, quantized, dims, input_scale, weight_scale, tensor_name);

  TF_LITE_MICRO_EXPECT_EQ(result.bytes, tensor_size * sizeof(int32_t));
  TF_LITE_MICRO_EXPECT_EQ(result.dims, dims);
  TF_LITE_MICRO_EXPECT_EQ(result.name, tensor_name);
  TF_LITE_MICRO_EXPECT_EQ(result.params.scale, input_scale * weight_scale);
  for (int i = 0; i < tensor_size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(expected_quantized_values[i], result.data.i32[i]);
  }
}

TF_LITE_MICRO_TEST(CreatePerChannelQuantizedBiasTensor) {
  float input_scale = 0.5;
  float weight_scales[] = {0.5, 1, 2, 4};
  const int tensor_size = 12;
  const int channels = 4;
  int dims_arr[] = {4, 4, 3, 1, 1};
  const char* tensor_name = "test_tensor";
  int32_t quantized[tensor_size];
  float scales[channels + 1];
  int zero_points[] = {4, 0, 0, 0, 0};
  float pre_quantized[] = {-10, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 10};
  int32_t expected_quantized_values[] = {-40, -20, -16, -6, -4, -2,
                                         0,   1,   2,   2,  2,  5};
  TfLiteIntArray* dims = tflite::testing::IntArrayFromInts(dims_arr);

  TfLiteAffineQuantization quant;
  TfLiteTensor result = tflite::testing::CreatePerChannelQuantizedBiasTensor(
      pre_quantized, quantized, dims, input_scale, weight_scales, scales,
      zero_points, &quant, 0, tensor_name);

  // Values in scales array start at index 1 since index 0 is dedicated to
  // tracking the tensor size.
  for (int i = 0; i < channels; i++) {
    TF_LITE_MICRO_EXPECT_EQ(scales[i + 1], input_scale * weight_scales[i]);
  }

  TF_LITE_MICRO_EXPECT_EQ(result.bytes, tensor_size * sizeof(int32_t));
  TF_LITE_MICRO_EXPECT_EQ(result.dims, dims);
  TF_LITE_MICRO_EXPECT_EQ(result.name, tensor_name);
  for (int i = 0; i < tensor_size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(expected_quantized_values[i], result.data.i32[i]);
  }
}

TF_LITE_MICRO_TEST(CreateSymmetricPerChannelQuantizedTensor) {
  const int tensor_size = 12;
  const int channels = 2;
  const int dims_arr[] = {4, channels, 3, 2, 1};
  const char* tensor_name = "test_tensor";
  int8_t quantized[12];
  const float pre_quantized[] = {-127, -55, -4, -3, -2, -1,
                                 0,    1,   2,  3,  4,  63.5};
  const int8_t expected_quantized_values[] = {-127, -55, -4, -3, -2, -1,
                                              0,    2,   4,  6,  8,  127};
  float expected_scales[] = {1.0, 0.5};
  TfLiteIntArray* dims = tflite::testing::IntArrayFromInts(dims_arr);

  int zero_points[channels + 1];
  float scales[channels + 1];
  TfLiteAffineQuantization quant;
  TfLiteTensor result =
      tflite::testing::CreateSymmetricPerChannelQuantizedTensor(
          pre_quantized, quantized, dims, scales, zero_points, &quant, 0,
          "test_tensor");

  TF_LITE_MICRO_EXPECT_EQ(result.bytes, tensor_size * sizeof(int8_t));
  TF_LITE_MICRO_EXPECT_EQ(result.dims, dims);
  TF_LITE_MICRO_EXPECT_EQ(result.name, tensor_name);
  TfLiteFloatArray* result_scales =
      static_cast<TfLiteAffineQuantization*>(result.quantization.params)->scale;
  for (int i = 0; i < channels; i++) {
    TF_LITE_MICRO_EXPECT_EQ(result_scales->data[i], expected_scales[i]);
  }
  for (int i = 0; i < tensor_size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(expected_quantized_values[i], result.data.int8[i]);
  }
}

TF_LITE_MICRO_TESTS_END
