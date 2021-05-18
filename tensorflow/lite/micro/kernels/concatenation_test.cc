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
#include <initializer_list>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

void TestConcatenateTwoInputs(int* input1_dims_data, const float* input1_data,
                              int* input2_dims_data, const float* input2_data,
                              int axis, int* output_dims_data,
                              const float* expected_output_data,
                              float* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int input_size = 2;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {CreateTensor(input1_data, input1_dims),
                                        CreateTensor(input2_data, input2_dims),
                                        CreateTensor(output_data, output_dims)};

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteConcatenationParams builtin_data = {
      .axis = axis,
      .activation = kTfLiteActNone  // Only activation supported in this impl
  };

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_CONCATENATION();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  const int output_dims_count = ElementCount(*output_dims);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i], 1e-5f);
  }
}

void TestConcatenateQuantizedTwoInputs(
    int* input1_dims_data, const uint8_t* input1_data, int* input2_dims_data,
    const uint8_t* input2_data, const float input_scale,
    const int input_zero_point, int axis, int* output_dims_data,
    const uint8_t* expected_output_data, const float output_scale,
    const int output_zero_point, uint8_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int input_size = 2;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_dims, input_scale,
                            input_zero_point),
      CreateQuantizedTensor(input2_data, input2_dims, input_scale,
                            input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point)};

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteConcatenationParams builtin_data = {
      .axis = axis,
      .activation = kTfLiteActNone  // Only activation supported in this impl
  };

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_CONCATENATION();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  const int output_dims_count = ElementCount(*output_dims);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TwoInputsAllAxesCombinations) {
  // Concatenate the same two input tensors along all possible axes.

  int input_shape[] = {2, 2, 3};
  const float input1_value[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const float input2_value[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  // expected output when concatenating on axis 0
  int output_shape_axis0[] = {2, 4, 3};
  const float output_value_axis0[] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                      7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  // expected output when concatenating on axis 1
  int output_shape_axis1[] = {2, 2, 6};
  const float output_value_axis1[] = {1.0f, 2.0f, 3.0f, 7.0f,  8.0f,  9.0f,
                                      4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f};

  float output_data[12];

  // Axis = 0
  tflite::testing::TestConcatenateTwoInputs(
      input_shape, input1_value, input_shape, input2_value, /* axis */ 0,
      output_shape_axis0, output_value_axis0, output_data);

  // Axis = -2 (equivalent to axis = 0)
  tflite::testing::TestConcatenateTwoInputs(
      input_shape, input1_value, input_shape, input2_value, /* axis */ -2,
      output_shape_axis0, output_value_axis0, output_data);

  // Axis = 1
  tflite::testing::TestConcatenateTwoInputs(
      input_shape, input1_value, input_shape, input2_value, /* axis */ 1,
      output_shape_axis1, output_value_axis1, output_data);

  // Axis = -1 (equivalent to axis = 1)
  tflite::testing::TestConcatenateTwoInputs(
      input_shape, input1_value, input_shape, input2_value, /* axis */ -1,
      output_shape_axis1, output_value_axis1, output_data);
}

TF_LITE_MICRO_TEST(TwoInputsQuantizedUint8) {
  const int axis = 2;
  int input_shape[] = {3, 2, 1, 2};
  int output_shape[] = {3, 2, 1, 4};

  const float input_scale = 0.1f;
  const int input_zero_point = 127;
  const float output_scale = 0.1f;
  const int output_zero_point = 127;

  const uint8_t input1_values[] = {137, 157, 167, 197};

  const uint8_t input2_values[] = {138, 158, 168, 198};

  const uint8_t output_value[] = {
      137, 157, 138, 158, 167, 197, 168, 198,
  };

  uint8_t output_data[8];
  tflite::testing::TestConcatenateQuantizedTwoInputs(
      input_shape, input1_values, input_shape, input2_values, input_scale,
      input_zero_point, axis, output_shape, output_value, output_scale,
      output_zero_point, output_data);
}

TF_LITE_MICRO_TEST(ThreeDimensionalTwoInputsDifferentShapes) {
  const int axis = 1;

  int input1_shape[] = {3, 2, 1, 2};
  int input2_shape[] = {3, 2, 3, 2};
  int output_shape[] = {3, 2, 4, 2};

  const float input1_values[] = {1.0f, 3.0f, 4.0f, 7.0f};
  const float input2_values[] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  const float output_values[] = {1.0f, 3.0f,  1.0f,  2.0f, 3.0f, 4.0f,
                                 5.0f, 6.0f,  4.0f,  7.0f, 7.0f, 8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f};

  float output_data[16];
  tflite::testing::TestConcatenateTwoInputs(
      input1_shape, input1_values, input2_shape, input2_values, axis,
      output_shape, output_values, output_data);
}

TF_LITE_MICRO_TESTS_END
