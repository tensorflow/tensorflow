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

struct CumSumTestParams {
  bool exclusive = false;
  bool reverse = false;
  int32_t axis = std::numeric_limits<int32_t>::max();
};

void ExecuteCumSumTest(CumSumTestParams& test_params, TfLiteTensor* tensors,
                       int tensors_count) {
  int kInputArrayData[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
  int kOutputArrayData[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  TfLiteCumsumParams params;
  params.exclusive = test_params.exclusive;
  params.reverse = test_params.reverse;

  const TfLiteRegistration registration = tflite::Register_CUMSUM();
  micro::KernelRunner runner(registration, tensors, tensors_count, inputs_array,
                             outputs_array, static_cast<void*>(&params));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

template <typename T>
void TestCumSum(CumSumTestParams& test_params, int* input_dims_data,
                const T* input_data, int* expected_dims, const T* expected_data,
                T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  int axis_dims_data[] = {1, 1};
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  const int32_t axis_data[] = {test_params.axis};

  TfLiteTensor tensors[] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(axis_data, axis_dims),
      CreateTensor(output_data, output_dims),
  };
  constexpr int tensors_count = std::extent<decltype(tensors)>::value;
  ExecuteCumSumTest(test_params, tensors, tensors_count);

  constexpr float kTolerance = 1e-5;
  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTolerance);
  }
}

// min/max are used to compute scale, zero-point, compare tolerance
template <typename T, int kOutputSize>
struct TestQuantParams {
  float data_min;              // input and output data minimum value
  float data_max;              // input and output data maximum value
  T input_data[kOutputSize];   // quantized input storage
  T output_data[kOutputSize];  // quantized output storage
};

// for quantized int, the error shouldn't exceed step
template <typename T>
float GetTolerance(float min, float max) {
  float kQuantizedStep =
      2.0f * (max - min) /
      (std::numeric_limits<T>::max() - std::numeric_limits<T>::min());
  return kQuantizedStep;
}

template <typename T, int kOutputSize>
void TestCumSumQuantized(CumSumTestParams& test_params,
                         TestQuantParams<T, kOutputSize>* params,
                         int* input_dims_data, const float* input_data,
                         int* expected_dims, const float* expected_data,
                         float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);

  int axis_dims_data[] = {1, 1};
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  const int32_t axis_data[] = {test_params.axis};

  const float scale = ScaleFromMinMax<T>(params->data_min, params->data_max);
  const int zero_point =
      ZeroPointFromMinMax<T>(params->data_min, params->data_max);

  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input_data, params->input_data, input_dims, scale,
                            zero_point),
      CreateTensor(axis_data, axis_dims),
      CreateQuantizedTensor(params->output_data, output_dims, scale,
                            zero_point),
  };

  constexpr int tensors_count = std::extent<decltype(tensors)>::value;
  ExecuteCumSumTest(test_params, tensors, tensors_count);

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

TF_LITE_MICRO_TEST(CumSumOpTestSimpleTest) {
  int kDims[] = {2, 2, 4};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr float kExpect[] = {1, 3, 6, 10, 5, 11, 18, 26};

  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::CumSumTestParams test_params;
  test_params.axis = 1;

  tflite::testing::TestCumSum(test_params, kDims, kInput, kDims, kExpect,
                              output_data);
}

TF_LITE_MICRO_TEST(CumSumOpTestSimpleAxis0Test) {
  int kDims[] = {2, 2, 4};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr float kExpect[] = {1, 2, 3, 4, 6, 8, 10, 12};

  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::CumSumTestParams test_params;
  test_params.axis = 0;

  tflite::testing::TestCumSum(test_params, kDims, kInput, kDims, kExpect,
                              output_data);
}

TF_LITE_MICRO_TEST(CumSumOpTestSimple1DTest) {
  int kDims[] = {1, 8};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr float kExpect[] = {1, 3, 6, 10, 15, 21, 28, 36};

  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::CumSumTestParams test_params;
  test_params.axis = 0;

  tflite::testing::TestCumSum(test_params, kDims, kInput, kDims, kExpect,
                              output_data);
}

TF_LITE_MICRO_TEST(CumSumOpTestSimpleReverseTest) {
  int kDims[] = {2, 2, 4};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr float kExpect[] = {10, 9, 7, 4, 26, 21, 15, 8};

  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::CumSumTestParams test_params;
  test_params.axis = 1;
  test_params.reverse = true;

  tflite::testing::TestCumSum(test_params, kDims, kInput, kDims, kExpect,
                              output_data);
}

TF_LITE_MICRO_TEST(CumSumOpTestSimpleExclusiveTest) {
  int kDims[] = {2, 2, 4};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr float kExpect[] = {0, 1, 3, 6, 0, 5, 11, 18};

  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::CumSumTestParams test_params;
  test_params.axis = 1;
  test_params.exclusive = true;

  tflite::testing::TestCumSum(test_params, kDims, kInput, kDims, kExpect,
                              output_data);
}

TF_LITE_MICRO_TEST(CumSumOpTestSimpleReverseExclusiveTest) {
  int kDims[] = {2, 2, 4};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr float kExpect[] = {9, 7, 4, 0, 21, 15, 8, 0};

  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::CumSumTestParams test_params;
  test_params.axis = -1;
  test_params.exclusive = true;
  test_params.reverse = true;

  tflite::testing::TestCumSum(test_params, kDims, kInput, kDims, kExpect,
                              output_data);
}

TF_LITE_MICRO_TEST(CumSumOpTestSimpleTestInt8) {
  int kDims[] = {2, 2, 4};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr float kExpect[] = {1, 3, 6, 10, 5, 11, 18, 26};

  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::CumSumTestParams test_params;
  test_params.axis = 1;

  tflite::testing::TestQuantParams<int8_t, kOutputCount> params = {};
  params.data_min = -26.0f;
  params.data_max = 26.0f;

  tflite::testing::TestCumSumQuantized<int8_t, kOutputCount>(
      test_params, &params, kDims, kInput, kDims, kExpect, output_data);
}

TF_LITE_MICRO_TEST(CumSumOpTestSimpleAxis0TestInt8) {
  int kDims[] = {2, 2, 4};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr float kExpect[] = {1, 2, 3, 4, 6, 8, 10, 12};

  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::CumSumTestParams test_params;
  test_params.axis = 0;

  tflite::testing::TestQuantParams<int8_t, kOutputCount> params = {};
  params.data_min = -12.0f;
  params.data_max = 12.0f;

  tflite::testing::TestCumSumQuantized<int8_t, kOutputCount>(
      test_params, &params, kDims, kInput, kDims, kExpect, output_data);
}

TF_LITE_MICRO_TEST(CumSumOpTestSimple1DTestInt8) {
  int kDims[] = {1, 8};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr float kExpect[] = {1, 3, 6, 10, 15, 21, 28, 36};

  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::CumSumTestParams test_params;
  test_params.axis = 0;

  tflite::testing::TestQuantParams<int8_t, kOutputCount> params = {};
  params.data_min = -36.0f;
  params.data_max = 36.0f;

  tflite::testing::TestCumSumQuantized<int8_t, kOutputCount>(
      test_params, &params, kDims, kInput, kDims, kExpect, output_data);
}

TF_LITE_MICRO_TEST(CumSumOpTestSimpleReverseTestInt8) {
  int kDims[] = {2, 2, 4};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr float kExpect[] = {10, 9, 7, 4, 26, 21, 15, 8};

  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::CumSumTestParams test_params;
  test_params.axis = 1;
  test_params.reverse = true;

  tflite::testing::TestQuantParams<int8_t, kOutputCount> params = {};
  params.data_min = -26.0f;
  params.data_max = 26.0f;

  tflite::testing::TestCumSumQuantized<int8_t, kOutputCount>(
      test_params, &params, kDims, kInput, kDims, kExpect, output_data);
}

TF_LITE_MICRO_TEST(CumSumOpTestSimpleExclusiveTestInt8) {
  int kDims[] = {2, 2, 4};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr float kExpect[] = {0, 1, 3, 6, 0, 5, 11, 18};

  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::CumSumTestParams test_params;
  test_params.axis = 1;
  test_params.exclusive = true;

  tflite::testing::TestQuantParams<int8_t, kOutputCount> params = {};
  params.data_min = -18.0f;
  params.data_max = 18.0f;

  tflite::testing::TestCumSumQuantized<int8_t, kOutputCount>(
      test_params, &params, kDims, kInput, kDims, kExpect, output_data);
}

TF_LITE_MICRO_TEST(CumSumOpTestSimpleReverseExclusiveTestInt8) {
  int kDims[] = {2, 2, 4};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr float kExpect[] = {9, 7, 4, 0, 21, 15, 8, 0};

  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::CumSumTestParams test_params;
  test_params.axis = -1;
  test_params.exclusive = true;
  test_params.reverse = true;

  tflite::testing::TestQuantParams<int8_t, kOutputCount> params = {};
  params.data_min = -21.0f;
  params.data_max = 21.0f;

  tflite::testing::TestCumSumQuantized<int8_t, kOutputCount>(
      test_params, &params, kDims, kInput, kDims, kExpect, output_data);
}
TF_LITE_MICRO_TESTS_END
