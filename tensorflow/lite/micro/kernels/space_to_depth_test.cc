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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

using tflite::ElementCount;
using tflite::testing::CreateTensor;
using tflite::testing::IntArrayFromInts;

namespace {

void ExpectEq(TfLiteIntArray* a, TfLiteIntArray* b) {
  for (int i = 0; i < a->size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(a->data[i], b->data[i]);
  }
}

template <typename T>
void ExpectNear(const T a[], const T b[], int size, float tolerance = 1e-5) {
  for (int i = 0; i < size; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(a[i], b[i], tolerance);
  }
}

template <typename T>
constexpr int ArrayLength(const T&) {
  return std::extent<T>::value;
}

template <typename T>
struct SpaceToDepthTest {
  int* input_dims;
  const T* input_data;
  int block_size;
  int* expect_dims;
  const T* expect_data;
  T* output_data;
};

template <typename T>
void TestSpaceToDepth(const SpaceToDepthTest<T>& args) {
  TfLiteIntArray* input_dims = IntArrayFromInts(args.input_dims);
  constexpr int kOutputDims = 4;
  int output_dims_data[] = {kOutputDims, 0, 0, 0, 0};
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  TfLiteTensor tensors[] = {CreateTensor(args.input_data, input_dims),
                            CreateTensor(args.output_data, output_dims)};

  const TfLiteRegistration registration = tflite::Register_SPACE_TO_DEPTH();
  constexpr int tensor_count = ArrayLength(tensors);
  constexpr int kInputIndex = 0;
  int input_indexes_data[] = {1, kInputIndex};
  TfLiteIntArray* input_indexes = IntArrayFromInts(input_indexes_data);
  constexpr int kOutputIndex = 1;
  int output_indexes_data[] = {1, kOutputIndex};
  TfLiteIntArray* output_indexes = IntArrayFromInts(output_indexes_data);
  TfLiteSpaceToDepthParams op_params = {};
  op_params.block_size = args.block_size;

  tflite::micro::KernelRunner runner(registration, tensors, tensor_count,
                                     input_indexes, output_indexes,
                                     static_cast<void*>(&op_params));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  const TfLiteTensor* output_tensor = &tensors[kOutputIndex];
  TfLiteIntArray* expect_dims = IntArrayFromInts(args.expect_dims);
  ExpectEq(output_tensor->dims, expect_dims);
  ExpectNear(args.output_data, args.expect_data, ElementCount(*expect_dims));
}

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SpaceToDepth_Float32_1222) {
  using value_type = float;
  SpaceToDepthTest<value_type> test;

  int input_dims[] = {4, 1, 2, 2, 2};
  test.input_dims = input_dims;
  constexpr value_type kInputData[] = {1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1};
  test.input_data = kInputData;

  test.block_size = 2;

  int expect_dims[] = {4, 1, 1, 1, 8};
  test.expect_dims = expect_dims;
  test.expect_data = kInputData;

  constexpr int kExpectElements = ArrayLength(kInputData);
  value_type output_data[kExpectElements];
  test.output_data = output_data;

  TestSpaceToDepth(test);
}

TF_LITE_MICRO_TEST(SpaceToDepth_Int8_1221) {
  using value_type = int8_t;
  SpaceToDepthTest<value_type> test;

  int input_dims[] = {4, 1, 2, 2, 1};
  test.input_dims = input_dims;
  constexpr value_type kInputData[] = {1, 2, 3, 4};
  test.input_data = kInputData;

  test.block_size = 2;

  int expect_dims[] = {4, 1, 1, 1, 4};
  test.expect_dims = expect_dims;
  test.expect_data = kInputData;

  constexpr int kExpectElements = ArrayLength(kInputData);
  value_type output_data[kExpectElements];
  test.output_data = output_data;

  TestSpaceToDepth(test);
}

TF_LITE_MICRO_TEST(SpaceToDepth_Int8_1223) {
  using value_type = int8_t;
  SpaceToDepthTest<value_type> test;

  int input_dims[] = {4, 1, 2, 2, 3};
  test.input_dims = input_dims;
  constexpr value_type kInputData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  test.input_data = kInputData;

  test.block_size = 2;

  int expect_dims[] = {4, 1, 1, 1, 12};
  test.expect_dims = expect_dims;
  test.expect_data = kInputData;

  constexpr int kExpectElements = ArrayLength(kInputData);
  value_type output_data[kExpectElements];
  test.output_data = output_data;

  TestSpaceToDepth(test);
}

TF_LITE_MICRO_TEST(SpaceToDepth_Int8_1441) {
  using value_type = int8_t;
  SpaceToDepthTest<value_type> test;

  int input_dims[] = {4, 1, 4, 4, 1};
  test.input_dims = input_dims;
  constexpr value_type kInputData[] = {1, 2,  5,  6,  3,  4,  7,  8,
                                       9, 10, 13, 14, 11, 12, 15, 16};
  test.input_data = kInputData;

  test.block_size = 2;

  int expect_dims[] = {4, 1, 2, 2, 4};
  test.expect_dims = expect_dims;
  constexpr value_type kExpectData[] = {1, 2,  3,  4,  5,  6,  7,  8,
                                        9, 10, 11, 12, 13, 14, 15, 16};
  test.expect_data = kExpectData;

  constexpr int kExpectElements = ArrayLength(kInputData);
  value_type output_data[kExpectElements];
  test.output_data = output_data;

  TestSpaceToDepth(test);
}

TF_LITE_MICRO_TESTS_END
