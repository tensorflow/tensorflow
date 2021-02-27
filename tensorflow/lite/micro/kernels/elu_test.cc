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
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// min/max are used to compute scale, zero-point
template <typename T>
struct TestEluParams {
  // quantization parameters
  float data_min;   // input and output data minimum value
  float data_max;   // input and output data maximum value
  T* input_data;    // quantized input storage
  T* output_data;   // quantized output storage
  float tolerance;  // output vs expected value tolerance
};

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
constexpr float kQuantizedTolerance = 2 * (1. / 256);

void ExecuteEluTest(TfLiteTensor* tensors, int tensors_count) {
  constexpr int kInputArrayData[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
  constexpr int kOutputArrayData[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  const TfLiteRegistration registration = tflite::Register_ELU();
  micro::KernelRunner runner(registration, tensors, tensors_count, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

template <typename T>
void TestElu(const int* input_dims_data, const T* input_data,
             const int* expected_dims, const T* expected_data, T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  TfLiteTensor tensors[] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
  };
  constexpr int tensors_count = std::extent<decltype(tensors)>::value;
  ExecuteEluTest(tensors, tensors_count);

  constexpr float kTolerance = 1e-5;
  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTolerance);
  }
}

template <typename T>
void TestEluQuantized(const TestEluParams<T>& params,
                      const int* input_dims_data, const float* input_data,
                      const int* expected_dims, const float* expected_data,
                      float* output_data) {
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

  ExecuteEluTest(tensors, kTensorsCount);

  Dequantize(params.output_data, output_count, scale, zero_point, output_data);
  const float kTolerance = params.tolerance;
  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTolerance);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatActivationsOpTestElu) {
  constexpr int kDims[] = {4, 1, 2, 4, 1};
  constexpr float kInput[] = {
      0, -6, 2,  -4,    //
      3, -2, 10, -0.1,  //
  };
  constexpr float kExpect[] = {
      0.0, -0.997521, 2.0,  -0.981684,   //
      3.0, -0.864665, 10.0, -0.0951626,  //
  };
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestElu(kDims, kInput, kDims, kExpect, output_data);
}

TF_LITE_MICRO_TEST(QuantizedActivationsOpTestEluInt8) {
  constexpr int kDims[] = {4, 1, 2, 4, 1};
  constexpr float kInput[] = {
      0, -6, 2, -4,    //
      3, -2, 6, -0.1,  //
  };
  constexpr float kExpect[] = {
      0,   -1.0,   2.0, -1,      //
      3.0, -0.875, 6.0, -0.125,  //
  };
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  // setup quantization storage and parameters
  int8_t q_output_data[kOutputCount];
  int8_t q_input_data[kOutputCount];
  constexpr float kMin = -1;
  constexpr float kMax = 127.f / 128.f;
  tflite::testing::TestEluParams<int8_t> params = {};
  params.data_min = 8 * kMin;
  params.data_max = 8 * kMax;
  params.input_data = q_input_data;
  params.output_data = q_output_data;
  params.tolerance = tflite::testing::kQuantizedTolerance;

  tflite::testing::TestEluQuantized(params, kDims, kInput, kDims, kExpect,
                                    output_data);
}

TF_LITE_MICRO_TESTS_END
