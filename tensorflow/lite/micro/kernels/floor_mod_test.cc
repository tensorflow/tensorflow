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

#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

void ExecuteFloorModTest(TfLiteTensor* tensors, int tensors_count) {
  int kInputArrayData[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
  int kOutputArrayData[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  const TfLiteRegistration registration = tflite::Register_FLOOR_MOD();
  micro::KernelRunner runner(registration, tensors, tensors_count, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

template <typename T>
void TestFloorMod(int* input1_dims_data, const T* input1_data,
                  int* input2_dims_data, const T* input2_data,
                  int* expected_dims, const T* expected_data, T* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  TfLiteTensor tensors[] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };
  constexpr int tensors_count = std::extent<decltype(tensors)>::value;

  ExecuteFloorModTest(tensors, tensors_count);

  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_EQ(expected_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloorModFloatSimple) {
  int kDims[] = {4, 1, 2, 2, 1};
  constexpr float kInput1[] = {10, 9, 11, 3};
  constexpr float kInput2[] = {2, 2, 3, 4};
  constexpr float kExpect[] = {0, 1, 2, 3};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestFloorMod(kDims, kInput1, kDims, kInput2, kDims, kExpect,
                                output_data);
}

TF_LITE_MICRO_TEST(FloorModFloatNegativeValue) {
  int kDims[] = {4, 1, 2, 2, 1};
  constexpr float kInput1[] = {10, -9, -11, 7};
  constexpr float kInput2[] = {2, 2, -3, -4};
  constexpr float kExpect[] = {0, 1, -2, -1};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestFloorMod(kDims, kInput1, kDims, kInput2, kDims, kExpect,
                                output_data);
}

TF_LITE_MICRO_TEST(FloorModFloatBroadcast) {
  int kDims1[] = {4, 1, 2, 2, 1};
  int kDims2[] = {1, 1};
  constexpr float kInput1[] = {10, -9, -11, 7};
  constexpr float kInput2[] = {-3};
  constexpr float kExpect[] = {-2, 0, -2, -2};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestFloorMod(kDims1, kInput1, kDims2, kInput2, kDims1,
                                kExpect, output_data);
}

TF_LITE_MICRO_TESTS_END
