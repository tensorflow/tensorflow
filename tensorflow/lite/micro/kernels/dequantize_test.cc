/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

template <typename T>
void ValidateDequantizeGoldens(TfLiteTensor* tensors, int tensors_size,
                               const T* expected_output_data, T* output_data,
                               int output_length, float tolerance = 1e-5) {
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_DEQUANTIZE();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i], 0.001f);
  }
}

template <typename T>
void TestDequantizeToFloat(int* input_dims_data, const float* input_data,
                           T* input_data_quantized, float scale, int zero_point,
                           int* output_dims_data,
                           const float* expected_output_data,
                           float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_length = ElementCount(*output_dims);

  // 1 input, 1 output.
  const int tensors_size = 2;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims, scale,
                            zero_point),
      CreateTensor(output_data, output_dims),
  };

  ValidateDequantizeGoldens(tensors, tensors_size, expected_output_data,
                            output_data, output_length);
}

template <typename T>
void TestDequantizeToInt32(int* input_dims_data, const float* input_data,
                           T* input_data_quantized, float input_scale,
                           int input_zero_point, int* output_dims_data,
                           const int32_t* expected_output_data,
                           float output_scale, int output_zero_point,
                           int32_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_length = ElementCount(*output_dims);

  // 1 input, 1 output.
  const int tensors_size = 2;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateTensor(output_data, output_dims),
  };

  tensors[1].params.scale = output_scale;
  tensors[1].params.zero_point = output_zero_point;

  ValidateDequantizeGoldens(tensors, tensors_size, expected_output_data,
                            output_data, output_length);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(DequantizeOpTestUint8) {
  const int length = 10;
  int dims[] = {2, 5, 2};
  const float values[] = {-63.5, -63,  -62.5, -62,  -61.5,
                          62,    62.5, 63,    63.5, 64};
  const float scale = 0.5;
  const int zero_point = 127;
  uint8_t input_quantized[length];
  float output[length];
  tflite::testing::TestDequantizeToFloat(dims, values, input_quantized, scale,
                                         zero_point, dims, values, output);
}

TF_LITE_MICRO_TEST(DequantizeOpTestInt8) {
  const int length = 10;
  int dims[] = {2, 5, 2};
  const float values[] = {-63.5, -63,  -62.5, -62,  -61.5,
                          62,    62.5, 63,    63.5, 64};
  const float scale = 0.5;
  const int zero_point = -1;
  int8_t input_quantized[length];
  float output[length];
  tflite::testing::TestDequantizeToFloat(dims, values, input_quantized, scale,
                                         zero_point, dims, values, output);
}

TF_LITE_MICRO_TEST(DequantizeOpTestInt16) {
  const int length = 10;
  int dims[] = {2, 5, 2};
  const float values[] = {-63.5, -63,  -62.5, -62,  -61.5,
                          62,    62.5, 63,    63.5, 64};
  const float scale = 0.5;
  const int zero_point = -1;
  int16_t input_quantized[length];
  float output[length];
  tflite::testing::TestDequantizeToFloat(dims, values, input_quantized, scale,
                                         zero_point, dims, values, output);
}

TF_LITE_MICRO_TESTS_END
