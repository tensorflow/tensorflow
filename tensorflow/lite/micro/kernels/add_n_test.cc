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

#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kMaxInputTensors = 3;
constexpr int kMaxOutputTensors = 1;

void ExecuteAddN(TfLiteTensor* tensors, int tensors_count) {
  int input_array_data[kMaxInputTensors + kMaxOutputTensors] = {tensors_count -
                                                                1};
  for (int i = 1; i < tensors_count; i++) {
    input_array_data[i] = i - 1;
  }
  TfLiteIntArray* inputs_array = IntArrayFromInts(input_array_data);
  int kOutputArrayData[] = {1, tensors_count - 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  const TfLiteRegistration registration = tflite::Register_ADD_N();
  micro::KernelRunner runner(registration, tensors, tensors_count, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

template <typename T>
void TestAddN(int* input_dims_data, const T* const* input_data,
              int input_data_count, int* expected_dims, const T* expected_data,
              T* output_data) {
  TF_LITE_MICRO_EXPECT_LE(input_data_count, kMaxInputTensors);

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  TfLiteTensor tensors[kMaxInputTensors + kMaxOutputTensors] = {};
  for (int i = 0; i < input_data_count; i++) {
    tensors[i] = CreateTensor(input_data[i], input_dims);
  }
  tensors[input_data_count] = CreateTensor(output_data, output_dims);

  ExecuteAddN(tensors, input_data_count + 1);

  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_EQ(expected_data[i], output_data[i]);
  }
}

// min/max are used to compute scale, zero-point, compare tolerance
template <typename T, int kNumInputs, int kOutputSize>
struct TestQuantParams {
  float data_min;                         // input and output data minimum value
  float data_max;                         // input and output data maximum value
  T input_data[kNumInputs][kOutputSize];  // quantized input storage
  T output_data[kOutputSize];             // quantized output storage
};

// for quantized Add, the error shouldn't exceed step
template <typename T>
float GetTolerance(float min, float max) {
  float kQuantizedStep =
      2.0f * (max - min) /
      (std::numeric_limits<T>::max() - std::numeric_limits<T>::min());
  return kQuantizedStep;
}

template <typename T, int kNumInputs, int kOutputSize>
void TestAddNQuantized(TestQuantParams<T, kNumInputs, kOutputSize>* params,
                       int* input_dims_data, const float* const* input_data,
                       int* expected_dims, const float* expected_data,
                       float* output_data) {
  TF_LITE_MICRO_EXPECT_LE(kNumInputs, kMaxInputTensors);

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);

  const float scale = ScaleFromMinMax<T>(params->data_min, params->data_max);
  const int zero_point =
      ZeroPointFromMinMax<T>(params->data_min, params->data_max);

  TfLiteTensor tensors[kMaxInputTensors + kMaxOutputTensors] = {};
  for (int i = 0; i < kNumInputs; i++) {
    tensors[i] = CreateQuantizedTensor(input_data[i], params->input_data[i],
                                       input_dims, scale, zero_point);
  }
  tensors[kNumInputs] = CreateQuantizedTensor(params->output_data, output_dims,
                                              scale, zero_point);

  ExecuteAddN(tensors, kNumInputs + 1);

  Dequantize(params->output_data, kOutputSize, scale, zero_point, output_data);
  const float kTolerance = GetTolerance<T>(params->data_min, params->data_max);
  for (int i = 0; i < kOutputSize; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTolerance);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatAddNOpAddMultipleTensors) {
  int kDims[] = {4, 1, 2, 2, 1};
  constexpr float kInput1[] = {-2.0, 0.2, 0.7, 0.8};
  constexpr float kInput2[] = {0.1, 0.2, 0.3, 0.5};
  constexpr float kInput3[] = {0.5, 0.1, 0.1, 0.2};
  constexpr float kExpect[] = {-1.4, 0.5, 1.1, 1.5};
  const float* kInputs[tflite::testing::kMaxInputTensors] = {
      kInput1,
      kInput2,
      kInput3,
  };
  constexpr int kInputCount = std::extent<decltype(kInputs)>::value;
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestAddN(kDims, kInputs, kInputCount, kDims, kExpect,
                            output_data);
}

TF_LITE_MICRO_TEST(Int8AddNOpAddMultipleTensors) {
  int kDims[] = {4, 1, 2, 2, 1};
  constexpr float kInput1[] = {-2.0, 0.2, 0.7, 0.8};
  constexpr float kInput2[] = {0.1, 0.2, 0.3, 0.5};
  constexpr float kInput3[] = {0.5, 0.1, 0.1, 0.2};
  constexpr float kExpect[] = {-1.4, 0.5, 1.1, 1.5};
  const float* kInputs[tflite::testing::kMaxInputTensors] = {
      kInput1,
      kInput2,
      kInput3,
  };
  constexpr int kInputCount = std::extent<decltype(kInputs)>::value;
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestQuantParams<int8_t, kInputCount, kOutputCount> params =
      {};
  params.data_min = -3.0;
  params.data_max = 3.0;

  tflite::testing::TestAddNQuantized<int8_t, kInputCount, kOutputCount>(
      &params, kDims, kInputs, kDims, kExpect, output_data);
}

TF_LITE_MICRO_TESTS_END
