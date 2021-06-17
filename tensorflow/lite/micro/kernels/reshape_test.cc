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

#include <stdint.h>

#include <initializer_list>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// TODO(b/162356196): Cleanup this unit test more.

template <typename T>
void ValidateReshapeGoldens(TfLiteTensor* tensors, int tensors_size,
                            TfLiteIntArray* inputs_array,
                            TfLiteIntArray* outputs_array,
                            const T* expected_output,
                            const size_t expected_output_len,
                            int* expected_dims, const size_t expected_dims_len,
                            bool expect_failure) {
  const TfLiteRegistration registration =
      tflite::ops::micro::Register_RESHAPE();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  if (expect_failure) {
    TF_LITE_MICRO_EXPECT_NE(kTfLiteOk, runner.InitAndPrepare());
    return;
  }

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  TfLiteTensor* output_tensor = &tensors[outputs_array->data[0]];
  const T* output_data = GetTensorData<T>(output_tensor);
  for (size_t i = 0; i < expected_output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output[i], output_data[i], 1e-5f);
  }
  TF_LITE_MICRO_EXPECT_EQ(expected_dims_len,
                          static_cast<size_t>(output_tensor->dims->size));
  for (size_t i = 0; i < expected_dims_len; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_dims[i], output_tensor->dims->data[i]);
  }
}
template <typename T>
void TestReshapeWithShape(TfLiteTensor* input_tensor,
                          TfLiteTensor* shape_tensor,
                          TfLiteTensor* output_tensor, const T* expected_output,
                          const size_t expected_output_len, int* expected_dims,
                          const size_t expected_dims_len, bool expect_failure) {
  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size];
  tensors[0] = *input_tensor;
  tensors[1] = *shape_tensor;
  tensors[2] = *output_tensor;

  int inputs_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_data);
  int outputs_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_data);

  ValidateReshapeGoldens(tensors, tensors_size, inputs_array, outputs_array,
                         expected_output, expected_output_len, expected_dims,
                         expected_dims_len, expect_failure);
}

// If expected output is empty, the test is expected to fail.
template <typename T>
void TestReshapeWithoutShape(TfLiteTensor* input_tensor,
                             TfLiteTensor* output_tensor,
                             const T* expected_output,
                             const size_t expected_output_len,
                             int* expected_dims, const size_t expected_dims_len,
                             bool expect_failure) {
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size];
  tensors[0] = *input_tensor;
  tensors[1] = *output_tensor;

  int inputs_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_data);
  int outputs_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_data);

  ValidateReshapeGoldens(tensors, tensors_size, inputs_array, outputs_array,
                         expected_output, expected_output_len, expected_dims,
                         expected_dims_len, expect_failure);
}

void TestReshape(int* input_dims_data, const float* input_data,
                 int* shape_dims_data, const int32_t* shape_data,
                 int* output_dims_data, float* output_data,
                 const float* expected_output, const size_t expected_output_len,
                 int* expected_dims, const size_t expected_dims_len,
                 bool expect_failure = false) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* shape_dims = IntArrayFromInts(shape_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  TfLiteTensor input_tensor = CreateTensor(input_data, input_dims);
  TfLiteTensor shape_tensor = CreateTensor(shape_data, shape_dims);
  TfLiteTensor output_tensor = CreateTensor(output_data, output_dims);

  TestReshapeWithShape(&input_tensor, &shape_tensor, &output_tensor,
                       expected_output, expected_output_len, expected_dims,
                       expected_dims_len, expect_failure);
}

template <typename T>
void TestReshapeQuantized(int* input_dims_data, const T* input_data,
                          int* shape_dims_data, const int32_t* shape_data,
                          int* output_dims_data, T* output_data,
                          const T* expected_output,
                          const size_t expected_output_len, int* expected_dims,
                          const size_t expected_dims_len,
                          bool expect_failure = false) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* shape_dims = IntArrayFromInts(shape_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  TfLiteTensor input_tensor = CreateQuantizedTensor(
      input_data, input_dims, /*scale=*/1.f, /*zero_point=*/0);
  TfLiteTensor shape_tensor = CreateTensor(shape_data, shape_dims);
  TfLiteTensor output_tensor = CreateQuantizedTensor(
      output_data, output_dims, /*scale=*/1.f, /*zero_point=*/0);

  TestReshapeWithShape(&input_tensor, &shape_tensor, &output_tensor,
                       expected_output, expected_output_len, expected_dims,
                       expected_dims_len, expect_failure);
}
}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(ReshapeWithMismatchedDimensionsShouldFail) {
  float output_data[32];
  int input_dims[] = {4, 1, 2, 4, 1};
  const float input_data[] = {3};
  int shape_dims[] = {1, 2};
  const int32_t shape_int32[] = {2, 1};
  int output_dims[] = {2, 2, 1};
  const int golden_output_len = 0;
  const float golden_output[] = {};
  const int golden_dims_len = 0;
  int golden_dims[] = {};
  tflite::testing::TestReshape(
      input_dims, input_data, shape_dims, shape_int32, output_dims, output_data,
      golden_output, golden_output_len, golden_dims, golden_dims_len, true);
}

TF_LITE_MICRO_TEST(ReshapeWithTooManyDimensionsShouldFail) {
  float output_data[32];
  int input_dims[] = {9, 1, 1, 2, 1, 1, 1, 1, 1, 1};
  const float input[] = {3, 2};
  int shape_dims[] = {1, 9};
  const int32_t shape_int32[] = {1, 1, 1, 1, 1, 1, 1, 1, 2};
  int output_dims[] = {9, 1, 1, 1, 1, 1, 1, 1, 1, 2};
  const int golden_output_len = 2;
  const float golden_output[] = {3, 2};
  const int golden_dims_len = 9;
  int golden_dims[] = {1, 1, 1, 1, 1, 1, 1, 1, 2};
  tflite::testing::TestReshape(
      input_dims, input, shape_dims, shape_int32, output_dims, output_data,
      golden_output, golden_output_len, golden_dims, golden_dims_len, false);
}

TF_LITE_MICRO_TEST(ReshapeWithTooManySpecialDimensionsShouldFail) {
  float output_data[32];
  int input_dims[] = {4, 1, 2, 4, 11};
  const float input[] = {3};
  int shape_dims[] = {1, 4};
  const int32_t shape_int32[] = {-1, -1, 2, 4};
  int output_dims[] = {4, -1, -1, 2, 4};
  const int golden_output_len = 2;
  const float golden_output[] = {};
  const int golden_dims_len = 9;
  int golden_dims[] = {};
  tflite::testing::TestReshape(
      input_dims, input, shape_dims, shape_int32, output_dims, output_data,
      golden_output, golden_output_len, golden_dims, golden_dims_len, true);
}

// Create the model with a 2x2 shape. Processing still works because the new
// shape ends up being hardcoded as a flat vector.
TF_LITE_MICRO_TEST(ReshapeWithInvalidShapeShouldFail) {
  int input_dims_data[] = {3, 1, 2, 2};
  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInts(input_dims_data);
  const float input_data[] = {3.0f};
  auto input_tensor = tflite::testing::CreateTensor(input_data, input_dims);
  float output_data[4];
  int output_dims_data[6] = {2, 2, 1, 2, 2, 1};
  TfLiteIntArray* output_dims =
      tflite::testing::IntArrayFromInts(output_dims_data);
  auto output_tensor = tflite::testing::CreateTensor(output_data, output_dims);
  const int expected_output[] = {};
  const int expected_output_len = 0;
  int expected_dims[] = {};
  const int expected_dims_len = 0;
  tflite::testing::TestReshapeWithoutShape(
      &input_tensor, &output_tensor, expected_output, expected_output_len,
      expected_dims, expected_dims_len, true);
}

TF_LITE_MICRO_TEST(ReshapeWithRegularShapesShouldSucceed) {
  float output_data_float[32];
  int8_t output_data_int8[32];
  uint8_t output_data_uint8[32];
  int input_dims[] = {4, 1, 2, 4, 1};
  const float input_float[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const int8_t input_int8[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint8_t input_uint8[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int shape_dims[] = {1, 3};
  const int32_t shape_int32[] = {2, 2, 2};
  int output_dims[] = {3, 2, 2, 2};
  const int golden_output_len = 8;
  const float golden_output_float[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const int8_t golden_output_int8[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint8_t golden_output_uint8[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const int golden_dims_len = 3;
  int golden_dims[] = {2, 2, 2};
  tflite::testing::TestReshape(input_dims, input_float, shape_dims, shape_int32,
                               output_dims, output_data_float,
                               golden_output_float, golden_output_len,
                               golden_dims, golden_dims_len, false);
  tflite::testing::TestReshapeQuantized(
      input_dims, input_int8, shape_dims, shape_int32, output_dims,
      output_data_int8, golden_output_int8, golden_output_len, golden_dims,
      golden_dims_len, false);
  tflite::testing::TestReshapeQuantized(
      input_dims, input_uint8, shape_dims, shape_int32, output_dims,
      output_data_uint8, golden_output_uint8, golden_output_len, golden_dims,
      golden_dims_len, false);
}

// Stretch is not supported with TF Micro
TF_LITE_MICRO_TEST(ReshapeWithStretchDimensionShouldSucceed) {
  float output_data_float[32];
  int8_t output_data_int8[32];
  uint8_t output_data_uint8[32];
  int input_dims[] = {4, 1, 2, 4, 1};
  const float input_float[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const int8_t input_int8[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint8_t input_uint8[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int shape_dims[] = {1, 3};
  const int32_t shape_int32[] = {2, 1, -1};
  int output_dims[] = {3, 2, 1, -1};
  const int golden_output_len = 8;
  const float golden_output_float[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const int8_t golden_output_int8[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint8_t golden_output_uint8[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const int golden_dims_len = 3;
  int golden_dims[] = {2, 1, 4};
  tflite::testing::TestReshape(input_dims, input_float, shape_dims, shape_int32,
                               output_dims, output_data_float,
                               golden_output_float, golden_output_len,
                               golden_dims, golden_dims_len, false);
  tflite::testing::TestReshapeQuantized(
      input_dims, input_int8, shape_dims, shape_int32, output_dims,
      output_data_int8, golden_output_int8, golden_output_len, golden_dims,
      golden_dims_len, false);
  tflite::testing::TestReshapeQuantized(
      input_dims, input_uint8, shape_dims, shape_int32, output_dims,
      output_data_uint8, golden_output_uint8, golden_output_len, golden_dims,
      golden_dims_len, false);
}

// Empty shape indicates scalar output.
TF_LITE_MICRO_TEST(ReshapeWithScalarOutputShouldSucceed) {
  float output_data_float[4];
  int8_t output_data_int8[4];
  uint8_t output_data_uint8[4];
  int input_dims[] = {1, 1};
  const float input_float[] = {3};
  const int8_t input_int8[] = {3};
  const uint8_t input_uint8[] = {3};
  int shape_dims[] = {0};
  const int32_t shape_int32[] = {};
  int output_dims[] = {0};
  const int golden_output_len = 1;
  const float golden_output_float[] = {3};
  const int8_t golden_output_int8[] = {3};
  const uint8_t golden_output_uint8[] = {3};
  const int golden_dims_len = 0;
  int golden_dims[] = {};
  tflite::testing::TestReshape(input_dims, input_float, shape_dims, shape_int32,
                               output_dims, output_data_float,
                               golden_output_float, golden_output_len,
                               golden_dims, golden_dims_len, false);
  tflite::testing::TestReshapeQuantized(
      input_dims, input_int8, shape_dims, shape_int32, output_dims,
      output_data_int8, golden_output_int8, golden_output_len, golden_dims,
      golden_dims_len, false);
  tflite::testing::TestReshapeQuantized(
      input_dims, input_uint8, shape_dims, shape_int32, output_dims,
      output_data_uint8, golden_output_uint8, golden_output_len, golden_dims,
      golden_dims_len, false);
}

// Some old models specify '[0]' as the new shape, indicating that both input
// and output are scalars.
TF_LITE_MICRO_TEST(ReshapeWithLegacyScalarOutputShouldSucceed) {
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;

  int input_dims_data[] = {1, 1};
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  const float input_data[] = {3.0f};
  auto input_tensor = CreateTensor(input_data, input_dims);

  float output_data[1];
  int output_dims_data[2] = {1, 0};
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  auto output_tensor = CreateTensor(output_data, output_dims);

  int shape_dims_data[] = {1, 0};
  TfLiteIntArray* shape_dims = IntArrayFromInts(shape_dims_data);

  const int32_t shape_data[] = {0};
  auto shape_tensor = tflite::testing::CreateTensor(shape_data, shape_dims);
  const float expected_output_with_shape[] = {};
  const int expected_output_with_shape_len = 0;
  const float expected_output_no_shape[] = {3};
  const int expected_output_no_shape_len = 1;
  int expected_dims[] = {};
  const int expected_dims_len = 0;
  tflite::testing::TestReshapeWithShape<float>(
      &input_tensor, &shape_tensor, &output_tensor, expected_output_with_shape,
      expected_output_with_shape_len, expected_dims, expected_dims_len, true);

  tflite::testing::TestReshapeWithoutShape<float>(
      &input_tensor, &output_tensor, expected_output_no_shape,
      expected_output_no_shape_len, expected_dims, expected_dims_len, false);
}

TF_LITE_MICRO_TESTS_END
