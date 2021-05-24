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

#include <random>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

void GenerateUniformRandomVector(int size, float min, float max,
                                 std::minstd_rand* random_engine,
                                 float* result) {
  // Never use std::uniform_*_distribution in tests, it's
  // implementation-defined. Likewise, don't use std::default_random_engine,
  // implementation-defined. Implementation-defined is bad because it means that
  // any toolchain update or new platform may run into test failures.
  // std::minstd_rand is a standard instantiation of
  // std::linear_congruential_engine, the cheapest generator in c++11 stdlib,
  // it's good enough here.
  for (int i = 0; i < size; i++) {
    // We don't care whether the `max` value may ever be produced exactly.
    // It may actually be thanks to rounding, as std::minstd_rand::modulus
    // is 2^31 - 1 is greater than the inverse float epsilon.
    float random_value_scaled_0_1 =
        (*random_engine)() *
        (1.0f / static_cast<float>(std::minstd_rand::modulus));
    result[i] = min + (max - min) * random_value_scaled_0_1;
  }
}

void EvalTestReferenceHardSwish(int size, float* input, float* result) {
  for (int i = 0; i < size; i++) {
    const float in = input[i];
    result[i] = in * std::min(6.0f, std::max(0.0f, in + 3)) * (1.0f / 6.0f);
  }
}

template <typename T>
void TestHardSwishQuantized(int size, const T* output_data,
                            T* input_data_quantized, float* dequantized_output,
                            float input_min, float input_max, float output_min,
                            float output_max, std::minstd_rand* random_engine,
                            float* float_input_values,
                            float* float_ref_output_values) {
  int input_dims_data[] = {2, 1, size};
  int output_dims_data[] = {2, 1, size};
  const float input_scale = ScaleFromMinMax<T>(input_min, input_max);
  const int input_zero_point = ZeroPointFromMinMax<T>(input_min, input_max);
  const float output_scale = ScaleFromMinMax<T>(output_min, output_max);
  const int output_zero_point = ZeroPointFromMinMax<T>(output_min, output_max);

  // The numerical error for any 8bit quantized function is at least one half
  // times the quantization step: 0.5 * (kOutMax - kOutMin) / 256.
  // To that we add again the quantization step (kOutMax - kOutMin) / 256
  // to allow for an off-by-one rounding error.
  const float kTolerance =
      std::max(input_max - input_min, output_max - output_min) * (1.5f / 256.f);

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  TF_LITE_MICRO_EXPECT_EQ(output_elements_count, size);

  GenerateUniformRandomVector(size, input_min, input_max, random_engine,
                              float_input_values);
  EvalTestReferenceHardSwish(size, float_input_values, float_ref_output_values);
  for (int i = 0; i < size; i++) {
    float val = float_ref_output_values[i];
    float_ref_output_values[i] =
        std::min(output_max, std::max(output_min, val));
  }

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(float_input_values, input_data_quantized,
                            input_dims, input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_HARD_SWISH();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  Dequantize<T>(output_data, output_elements_count, output_scale,
                output_zero_point, dequantized_output);

  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(float_ref_output_values[i], dequantized_output[i],
                              kTolerance);
  }
}

template <typename T>
void TestHardSwishQuantizedBias(const int size, const T* output_data,
                                T* input_data_quantized,
                                float* dequantized_output, float input_min,
                                float input_max, float output_min,
                                float output_max, float tolerated_bias,
                                float* float_input_values,
                                float* float_ref_output_values) {
  const float input_scale = ScaleFromMinMax<T>(input_min, input_max);
  const float output_scale = ScaleFromMinMax<T>(output_min, output_max);

  const int input_zero_point = ZeroPointFromMinMax<T>(input_min, input_max);
  const int output_zero_point = ZeroPointFromMinMax<T>(output_min, output_max);

  const float max_scale = std::max(output_scale, input_scale);

  // In this bias-focused test case, no need for randomly generated input
  // values.
  TF_LITE_MICRO_EXPECT_LE(input_min, -3.0f);
  TF_LITE_MICRO_EXPECT_GE(input_max, 3.0f);
  const int quantized_input_negative_three = std::round(
      std::numeric_limits<T>::min() + (-3.0f - input_min) / input_scale);
  const int quantized_input_positive_three = std::round(
      std::numeric_limits<T>::min() + (3.0f - input_min) / input_scale);

  for (int i = quantized_input_negative_three;
       i < size && i <= quantized_input_positive_three; i++) {
    float_input_values[i] =
        input_min + (i - std::numeric_limits<T>::min()) * input_scale;
  }

  EvalTestReferenceHardSwish(size, float_input_values, float_ref_output_values);
  for (int i = 0; i < size; i++) {
    float val = float_ref_output_values[i];
    float_ref_output_values[i] =
        std::min(output_max, std::max(output_min, val));
  }

  int input_dims_data[] = {2, 1, size};
  int output_dims_data[] = {2, 1, size};

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  TF_LITE_MICRO_EXPECT_EQ(output_elements_count, size);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(float_input_values, input_data_quantized,
                            input_dims, input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_HARD_SWISH();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  Dequantize<T>(output_data, output_elements_count, output_scale,
                output_zero_point, dequantized_output);

  float sum_diff = 0;
  for (int i = 0; i < size; i++) {
    sum_diff += dequantized_output[i] - float_ref_output_values[i];
  }
  const float bias = sum_diff / (size * max_scale);
  TF_LITE_MICRO_EXPECT_LE(std::abs(bias), tolerated_bias);
}

void TestHardSwishFloat(const int size, float* output_data,
                        std::minstd_rand* random_engine,
                        float* float_input_values,
                        float* float_ref_output_values) {
  const float kMin = -10.0f;
  const float kMax = 10.0f;
  GenerateUniformRandomVector(size, kMin, kMax, random_engine,
                              float_input_values);

  EvalTestReferenceHardSwish(size, float_input_values, float_ref_output_values);

  int input_dims_data[] = {1, size};
  int output_dims_data[] = {1, size};

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  TF_LITE_MICRO_EXPECT_EQ(output_elements_count, size);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(float_input_values, input_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_HARD_SWISH();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(float_ref_output_values[i], output_data[i],
                              1e-5f);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleHardSwishTestFloat) {
  std::minstd_rand random_engine;
  constexpr int size = 100;
  float output_data[size] = {0.f};
  float input_values[size] = {0.f};
  float output_values[size] = {0.f};

  tflite::testing::TestHardSwishFloat(size, output_data, &random_engine,
                                      input_values, output_values);
}

TF_LITE_MICRO_TEST(SimpleHardSwishTestInt8) {
  std::minstd_rand random_engine;
  constexpr int pairs = 4, one_pair = 2;
  constexpr int size = 101;
  constexpr float minmax_pairs[pairs][one_pair] = {
      {0.f, 1.f}, {-2.f, 1.f}, {-5.f, 10.f}, {-40.f, 60.f}};
  int8_t output_data[size] = {0};
  int8_t input_data_quantized[size] = {0};
  float dequantized_output[size] = {0.f};
  float input_values[size] = {0.f};
  float output_values[size] = {0.f};

  for (int x = 0; x < pairs; x++) {
    for (int y = 0; y < pairs; y++) {
      float input_min = minmax_pairs[x][0];
      float input_max = minmax_pairs[x][1];
      float output_min = minmax_pairs[y][0];
      float output_max = minmax_pairs[y][1];

      tflite::testing::TestHardSwishQuantized<int8_t>(
          size, output_data, input_data_quantized, dequantized_output,
          input_min, input_max, output_min, output_max, &random_engine,
          input_values, output_values);
    }
  }
}

TF_LITE_MICRO_TEST(SimpleHardSwishTestUint8) {
  std::minstd_rand random_engine;
  constexpr int size = 99;
  constexpr int pairs = 4, one_pair = 2;
  constexpr float minmax_pairs[pairs][one_pair] = {
      {0.f, 1.f}, {-2.f, 1.f}, {-5.f, 10.f}, {-40.f, 60.f}};
  uint8_t output_data[size] = {0};
  uint8_t input_data_quantized[size] = {0};
  float dequantized_output[size] = {0.f};
  float input_values[size] = {0.f};
  float output_values[size] = {0.f};

  for (int x = 0; x < pairs; x++) {
    for (int y = 0; y < pairs; y++) {
      float input_min = minmax_pairs[x][0];
      float input_max = minmax_pairs[x][1];
      float output_min = minmax_pairs[y][0];
      float output_max = minmax_pairs[y][1];

      tflite::testing::TestHardSwishQuantized<uint8_t>(
          size, output_data, input_data_quantized, dequantized_output,
          input_min, input_max, output_min, output_max, &random_engine,
          input_values, output_values);
    }
  }
}

// See the comment in the reference implementation of quantized HardSwish:
// A numerical issue significantly affecting ImageNet classification accuracy
// with MobileNet v3 is only observable at the scale of HardSwish unit tests
// if we monitor specifically bias. This testcase is extracted from one of the
// HardSwish nodes in that MobileNet v3 that exhibited this issue.
TF_LITE_MICRO_TEST(SimpleHardSwishTestQuantizedBias) {
  constexpr int size = 43;
  uint8_t output_data[size] = {0};
  uint8_t input_data_quantized[size] = {0};
  float dequantized_output[size] = {0.f};
  float input_values[size] = {0.f};
  float output_values[size] = {0.f};

  tflite::testing::TestHardSwishQuantizedBias<uint8_t>(
      size, output_data, input_data_quantized, dequantized_output, -11.654928f,
      25.036512f, -0.3905796f, 24.50887f, 0.035, input_values, output_values);
}

TF_LITE_MICRO_TESTS_END
