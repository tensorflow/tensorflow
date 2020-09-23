/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

// Simple test data for 2x2x10 input 2x3x10 weights.
const int simple_input_size = 20;
const int simple_input_dims[] = {2, 2, 10};
const float simple_input_data[] = {
    1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
    1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
};
const int simple_weights_size = 30;
const int simple_weights_dims[] = {2, 3, 10};
const float simple_weights_data[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
};
const int simple_bias_dims[] = {1, 3};
const float simple_bias_data[] = {1, 2, 3};
const float simple_golden[] = {
    24, 25, 26, 58, 59, 60,
};
const int simple_output_size = 6;
const int simple_output_dims[] = {2, 2, 3};

// Test data for 2x2x10 input 2x3x10 weights with negative outputs to test relu.
const int relu_input_size = 20;
const int relu_input_dims[] = {2, 2, 10};
const float relu_input_data[] = {
    1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
    1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
};
const int relu_weights_size = 30;
const int relu_weights_dims[] = {2, 3, 10};
const float relu_weights_data[] = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   // u = 0
    -1, -2, -3, -4, -5, -6, -7, -8, -9, -10,  // u = 1
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   // u = 2
};
const int relu_bias_dims[] = {1, 3};
const float relu_bias_data[] = {1, -2, 3};
const float relu_golden[] = {
    24, 0, 26, 58, 0, 60,
};
const int relu_output_size = 6;
const int relu_output_dims[] = {2, 2, 3};

template <typename T>
TfLiteStatus ValidateFullyConnectedGoldens(
    TfLiteTensor* tensors, const int tensors_size,
    const TfLiteFusedActivation activation, const float tolerance,
    const int output_len, const T* golden, T* output_data) {
  TfLiteFullyConnectedParams builtin_data = {
      activation, kTfLiteFullyConnectedWeightsFormatDefault, false, false};

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration =
      ops::micro::Register_FULLY_CONNECTED();
  micro::KernelRunner runner(
      registration, tensors, tensors_size, inputs_array, outputs_array,
      reinterpret_cast<void*>(&builtin_data), micro_test::reporter);

  TfLiteStatus status = runner.InitAndPrepare();
  if (status != kTfLiteOk) {
    return status;
  }

  status = runner.Invoke();
  if (status != kTfLiteOk) {
    return status;
  }

  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], tolerance);
  }
  return kTfLiteOk;
}

TfLiteStatus TestFullyConnectedFloat(
    const int* input_dims_data, const float* input_data,
    const int* weights_dims_data, const float* weights_data,
    const int* bias_dims_data, const float* bias_data, const float* golden,
    const int* output_dims_data, TfLiteFusedActivation activation,
    float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInts(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims),
      CreateFloatTensor(weights_data, weights_dims),
      CreateFloatTensor(bias_data, bias_dims),
      CreateFloatTensor(output_data, output_dims),
  };

  return ValidateFullyConnectedGoldens(tensors, tensors_size, activation, 1e-5f,
                                       output_dims_count, golden, output_data);
}

template <typename T>
TfLiteStatus TestFullyConnectedQuantized(
    const int* input_dims_data, const float* input_data, T* input_quantized,
    const float input_scale, const int input_zero_point,
    const int* weights_dims_data, const float* weights_data,
    T* weights_quantized, const float weights_scale, const int* bias_dims_data,
    const float* bias_data, int32_t* bias_quantized, const float* golden,
    T* golden_quantized, const int* output_dims_data, const float output_scale,
    const int output_zero_point, TfLiteFusedActivation activation,
    T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInts(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(weights_data, weights_quantized, weights_dims,
                            weights_scale, 0),
      CreateQuantizedBiasTensor(bias_data, bias_quantized, bias_dims,
                                input_scale, weights_scale),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  AsymmetricQuantize(golden, golden_quantized, output_dims_count, output_scale,
                     output_zero_point);

  return ValidateFullyConnectedGoldens(tensors, tensors_size, activation, 0.0f,
                                       output_dims_count, golden_quantized,
                                       output_data);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleTest) {
  float output_data[tflite::testing::simple_output_size];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedFloat(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data,
          tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data,
          tflite::testing::simple_bias_dims, tflite::testing::simple_bias_data,
          tflite::testing::simple_golden, tflite::testing::simple_output_dims,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedUInt8) {
  const float input_scale = 1.0f;
  const int input_zero_point = 127;
  const float weights_scale = 1.0f;
  const float output_scale = 0.5f;
  const int output_zero_point = 127;

  uint8_t input_quantized[tflite::testing::simple_input_size];
  uint8_t weights_quantized[tflite::testing::simple_weights_size];
  int32_t bias_quantized[tflite::testing::simple_output_size];
  uint8_t golden_quantized[tflite::testing::simple_output_size];
  uint8_t output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data, input_quantized, input_scale,
          input_zero_point, tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data, weights_quantized,
          weights_scale, tflite::testing::simple_bias_dims,
          tflite::testing::simple_bias_data, bias_quantized,
          tflite::testing::simple_golden, golden_quantized,
          tflite::testing::simple_output_dims, output_scale, output_zero_point,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedInt8) {
  const float input_scale = 1.0f;
  const int input_zero_point = -1;
  const float weights_scale = 1.0f;
  const float output_scale = 0.5f;
  const int output_zero_point = -1;

  int8_t input_quantized[tflite::testing::simple_input_size];
  int8_t weights_quantized[tflite::testing::simple_weights_size];
  int32_t bias_quantized[tflite::testing::simple_output_size];
  int8_t golden_quantized[tflite::testing::simple_output_size];
  int8_t output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::simple_input_dims,
          tflite::testing::simple_input_data, input_quantized, input_scale,
          input_zero_point, tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data, weights_quantized,
          weights_scale, tflite::testing::simple_bias_dims,
          tflite::testing::simple_bias_data, bias_quantized,
          tflite::testing::simple_golden, golden_quantized,
          tflite::testing::simple_output_dims, output_scale, output_zero_point,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestRelu) {
  float output_data[tflite::testing::relu_output_size];
  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedFloat(
          tflite::testing::relu_input_dims, tflite::testing::relu_input_data,
          tflite::testing::relu_weights_dims,
          tflite::testing::relu_weights_data, tflite::testing::relu_bias_dims,
          tflite::testing::relu_bias_data, tflite::testing::relu_golden,
          tflite::testing::relu_output_dims, kTfLiteActRelu, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedUInt8Relu) {
  const float input_scale = 1.0f;
  const int input_zero_point = 127;
  const float weights_scale = 1.0f;
  const float output_scale = 0.5f;
  const int output_zero_point = 0;

  uint8_t input_quantized[tflite::testing::relu_input_size];
  uint8_t weights_quantized[tflite::testing::relu_weights_size];
  int32_t bias_quantized[tflite::testing::relu_output_size];
  uint8_t golden_quantized[tflite::testing::relu_output_size];
  uint8_t output_data[tflite::testing::relu_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::relu_input_dims, tflite::testing::relu_input_data,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::relu_weights_dims,
          tflite::testing::relu_weights_data, weights_quantized, weights_scale,
          tflite::testing::relu_bias_dims, tflite::testing::relu_bias_data,
          bias_quantized, tflite::testing::relu_golden, golden_quantized,
          tflite::testing::relu_output_dims, output_scale, output_zero_point,
          kTfLiteActRelu, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedInt8Relu) {
  const float input_scale = 1.0f;
  const int input_zero_point = -1;
  const float weights_scale = 1.0f;
  const float output_scale = 0.5f;
  const int output_zero_point = -128;

  int8_t input_quantized[tflite::testing::relu_input_size];
  int8_t weights_quantized[tflite::testing::relu_weights_size];
  int32_t bias_quantized[tflite::testing::relu_output_size];
  int8_t golden_quantized[tflite::testing::relu_output_size];
  int8_t output_data[tflite::testing::relu_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          tflite::testing::relu_input_dims, tflite::testing::relu_input_data,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::relu_weights_dims,
          tflite::testing::relu_weights_data, weights_quantized, weights_scale,
          tflite::testing::relu_bias_dims, tflite::testing::relu_bias_data,
          bias_quantized, tflite::testing::relu_golden, golden_quantized,
          tflite::testing::relu_output_dims, output_scale, output_zero_point,
          kTfLiteActRelu, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTest4DInput) {
  const int input_dims_4d[] = {4, 1, 1, 2, 10};

  float output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedFloat(
          input_dims_4d, tflite::testing::simple_input_data,
          tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data,
          tflite::testing::simple_bias_dims, tflite::testing::simple_bias_data,
          tflite::testing::simple_golden, tflite::testing::simple_output_dims,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTest4DInputQuantizedUInt8) {
  const float input_scale = 1.0f;
  const int input_zero_point = 127;
  const float weights_scale = 1.0f;
  const float output_scale = 0.5f;
  const int output_zero_point = 127;

  const int input_dims_4d[] = {4, 1, 1, 2, 10};

  uint8_t input_quantized[tflite::testing::simple_input_size];
  uint8_t weights_quantized[tflite::testing::simple_weights_size];
  int32_t bias_quantized[tflite::testing::simple_output_size];
  uint8_t golden_quantized[tflite::testing::simple_output_size];
  uint8_t output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          input_dims_4d, tflite::testing::simple_input_data, input_quantized,
          input_scale, input_zero_point, tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data, weights_quantized,
          weights_scale, tflite::testing::simple_bias_dims,
          tflite::testing::simple_bias_data, bias_quantized,
          tflite::testing::simple_golden, golden_quantized,
          tflite::testing::simple_output_dims, output_scale, output_zero_point,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TEST(SimpleTest4DInputQuantizedInt8) {
  const float input_scale = 1.0f;
  const int input_zero_point = -1;
  const float weights_scale = 1.0f;
  const float output_scale = 0.5f;
  const int output_zero_point = -1;

  const int input_dims_4d[] = {4, 1, 1, 2, 10};

  int8_t input_quantized[tflite::testing::simple_input_size];
  int8_t weights_quantized[tflite::testing::simple_weights_size];
  int32_t bias_quantized[tflite::testing::simple_output_size];
  int8_t golden_quantized[tflite::testing::simple_output_size];
  int8_t output_data[tflite::testing::simple_output_size];

  TF_LITE_MICRO_EXPECT_EQ(
      tflite::testing::TestFullyConnectedQuantized(
          input_dims_4d, tflite::testing::simple_input_data, input_quantized,
          input_scale, input_zero_point, tflite::testing::simple_weights_dims,
          tflite::testing::simple_weights_data, weights_quantized,
          weights_scale, tflite::testing::simple_bias_dims,
          tflite::testing::simple_bias_data, bias_quantized,
          tflite::testing::simple_golden, golden_quantized,
          tflite::testing::simple_output_dims, output_scale, output_zero_point,
          kTfLiteActNone, output_data),
      kTfLiteOk);
}

TF_LITE_MICRO_TESTS_END
