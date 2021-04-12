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

#include <limits>
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// min/max are used to compute scale, zero-point, compare tolerance
template <typename T>
struct TestLeakyReluParams {
  // general parameters
  float alpha;  // alpha multiplier

  // quantization parameters
  float data_min;   // input and output data minimum value
  float data_max;   // input and output data maximum value
  T* input_data;    // quantized input storage
  T* output_data;   // quantized output storage
  float tolerance;  // output vs expected value tolerance
};

void ExecuteLeakyReluTest(const float alpha, const int tensors_count,
                          TfLiteTensor* tensors) {
  TfLiteLeakyReluParams builtin_data = {};
  builtin_data.alpha = alpha;

  constexpr int kInputArrayData[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
  constexpr int kOutputArrayData[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  const TfLiteRegistration registration = tflite::Register_LEAKY_RELU();
  micro::KernelRunner runner(registration, tensors, tensors_count, inputs_array,
                             outputs_array, static_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

template <typename T>
void TestLeakyRelu(const TestLeakyReluParams<T>& params,
                   const int* input_dims_data, const T* input_data,
                   const int* expected_dims, const T* expected_data,
                   T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  TfLiteTensor tensors[] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
  };
  constexpr int tensors_count = std::extent<decltype(tensors)>::value;
  ExecuteLeakyReluTest(params.alpha, tensors_count, tensors);

  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_EQ(expected_data[i], output_data[i]);
  }
}

template <typename T>
void TestLeakyReluQuantized(const TestLeakyReluParams<T>& params,
                            const int* input_dims_data, const float* input_data,
                            const int* expected_dims,
                            const float* expected_data, float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  const float scale = ScaleFromMinMax<T>(params.data_min, params.data_max);
  const int zero_point =
      ZeroPointFromMinMax<T>(params.data_min, params.data_max);

  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input_data, params.input_data, input_dims, scale,
                            zero_point),
      CreateQuantizedTensor(params.output_data, output_dims, scale, zero_point),
  };
  constexpr int kTensorsCount = std::extent<decltype(tensors)>::value;

  ExecuteLeakyReluTest(params.alpha, kTensorsCount, tensors);

  Dequantize(params.output_data, output_count, scale, zero_point, output_data);
  const float kTolerance = params.tolerance;
  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTolerance);
  }
}

// Our fixed-point math function implementations have roughly 12 bits of
// accuracy, when specialized to 16-bit fixed-point arithmetic.
// That is purely an implementation compromise, it would have been possible
// to get closer to 16 bits of accuracy but that would be more expensive,
// and not needed for our purposes as ultimately the output is either
// immediately down-quantized to 8 bits, or will typically be at the output
// of the surrounding LSTM cell.
// So we can require roughly 2^-12 accuracy when the output is 16-bit, and
// we can more or less expect the full 2^-8 accuracy when the output is 8-bit.
//
// However, the representable output interval is often [-1, 1]  (it has to be
// for tanh, and even for logistic, when we implement it in fixed-point, we
// typically have to do so on such a symmetric interval, e.g. ARM NEON only
// has signed fixed-point arithmetic (SQRDMULH)).  As the width of [-1, 1]
// is 2, our representable values are often diluted by a factor of 2, whence
// the factor of 2 below.
const float kQuantizedTolerance = 2 * (1. / 256);

template <typename integer_dtype>
void QuantizedActivationsOpTestLeakyRelu() {
  constexpr int kDims[] = {2, 5, 5};
  constexpr float kInput[] = {
      -5.0f, -4.6f, -4.2f, -3.8f, -3.4f,  // Row 1
      -3.0f, -2.6f, -2.2f, -1.8f, -1.4f,  // Row 2
      -1.0f, -0.6f, -0.2f, 0.2f,  0.6f,   // Row 3
      1.0f,  1.4f,  1.8f,  2.2f,  2.6f,   // Row 4
      3.0f,  3.4f,  3.8f,  4.2f,  4.6f,   // Row 5
  };
  constexpr float kExpect[] = {
      -0.50f, -0.46f, -0.42f, -0.38f, -0.34f,  // Row 1
      -0.30f, -0.26f, -0.22f, -0.18f, -0.14f,  // Row 2
      -0.10f, -0.06f, -0.02f, 0.20f,  0.60f,   // Row 3
      1.00f,  1.40f,  1.80f,  2.20f,  2.60f,   // Row 4
      3.00f,  3.40f,  3.80f,  4.20f,  4.60f,   // Row 5
  };
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  // setup quantization storage and parameters
  integer_dtype q_output_data[kOutputCount];
  integer_dtype q_input_data[kOutputCount];
  constexpr float kMin = -1;
  constexpr float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
  TestLeakyReluParams<integer_dtype> params = {};
  params.alpha = 0.1f;
  params.data_min = 5 * kMin;
  params.data_max = 5 * kMax;
  params.input_data = q_input_data;
  params.output_data = q_output_data;
  params.tolerance = kQuantizedTolerance * 5;

  TestLeakyReluQuantized(params, kDims, kInput, kDims, kExpect, output_data);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(QuantizedActivationsOpTestLeakyReluInt8_1) {
  constexpr int kDims[] = {2, 2, 3};
  constexpr float kInput[] = {0.0f, 1.0f, 3.0f, 1.0f, -1.0f, -2.0f};
  constexpr float kExpect[] = {0.0f, 1.0f, 3.0f, 1.0f, -0.5f, -1.0f};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  // setup quantization storage and parameters
  int8_t q_output_data[kOutputCount];
  int8_t q_input_data[kOutputCount];
  constexpr float kMin = -1;
  constexpr float kMax = 127.f / 128.f;
  tflite::testing::TestLeakyReluParams<int8_t> params = {};
  params.alpha = 0.5f;
  params.data_min = 8 * kMin;
  params.data_max = 8 * kMax;
  params.input_data = q_input_data;
  params.output_data = q_output_data;
  params.tolerance = tflite::testing::kQuantizedTolerance * 8;

  tflite::testing::TestLeakyReluQuantized(params, kDims, kInput, kDims, kExpect,
                                          output_data);
}

TF_LITE_MICRO_TEST(QuantizedActivationsOpTestLeakyReluInt8_2) {
  tflite::testing::QuantizedActivationsOpTestLeakyRelu<int8_t>();
}

TF_LITE_MICRO_TEST(FloatActivationsOpTestLeakyRelu) {
  constexpr int kDims[] = {2, 2, 3};
  constexpr float kInput[] = {0.0f, 1.0f, 3.0f, 1.0f, -1.0f, -2.0f};
  constexpr float kExpect[] = {0.0f, 1.0f, 3.0f, 1.0f, -0.5f, -1.0f};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::TestLeakyReluParams<float> params = {};
  params.alpha = 0.5f;

  tflite::testing::TestLeakyRelu(params, kDims, kInput, kDims, kExpect,
                                 output_data);
}

TF_LITE_MICRO_TESTS_END
