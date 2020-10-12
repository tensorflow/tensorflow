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

void TestConcatenateTwoInputs(const int* input1_dims_data,
                              const float* input1_data,
                              const int* input2_dims_data,
                              const float* input2_data, int axis,
                              const int* output_dims_data,
                              const float* expected_output_data,
                              float* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int input_size = 2;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input1_data, input1_dims),
      CreateFloatTensor(input2_data, input2_dims),
      CreateFloatTensor(output_data, output_dims)};

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
  micro::KernelRunner runner(
      registration, tensors, tensors_size, inputs_array, outputs_array,
      reinterpret_cast<void*>(&builtin_data), micro_test::reporter);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  const int output_dims_count = ElementCount(*output_dims);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i], 1e-5f);
  }
}

void TestConcatenateQuantizedTwoInputs(
    const int* input1_dims_data, const uint8_t* input1_data, float input1_min,
    float input1_max, const int* input2_dims_data, const uint8_t* input2_data,
    float input2_min, float input2_max, int axis, bool fixed_point_scaling,
    const int* output_dims_data, const uint8_t* expected_output_data,
    float output_min, float output_max, uint8_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int input_size = 2;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(
          input1_data, input1_dims,
          ScaleFromMinMax<uint8_t>(input1_min, input1_max),
          ZeroPointFromMinMax<uint8_t>(input1_min, input1_max)),
      CreateQuantizedTensor(
          input2_data, input2_dims,
          ScaleFromMinMax<uint8_t>(input2_min, input2_max),
          ZeroPointFromMinMax<uint8_t>(input2_min, input2_max)),
      CreateQuantizedTensor(
          output_data, output_dims,
          ScaleFromMinMax<uint8_t>(output_min, output_max),
          ZeroPointFromMinMax<uint8_t>(output_min, output_max))};

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteConcatenationParams builtin_data = {
      .axis = axis,
      .activation = kTfLiteActNone,  // Only activation supported in this impl
      .fixed_point_scaling = fixed_point_scaling};

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_CONCATENATION();
  micro::KernelRunner runner(
      registration, tensors, tensors_size, inputs_array, outputs_array,
      reinterpret_cast<void*>(&builtin_data), micro_test::reporter);

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

  const int input_shape[] = {2, 2, 3};
  const float input1_value[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const float input2_value[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  // expected output when concatenating on axis 0
  const int output_shape_axis0[] = {2, 4, 3};
  const float output_value_axis0[] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                      7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  // expected output when concatenating on axis 1
  const int output_shape_axis1[] = {2, 2, 6};
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
  using tflite::testing::F2Q;

  const int axis = 2;
  const bool fixed_point_scaling = false;
  const int input_shape[] = {3, 2, 1, 2};
  const int output_shape[] = {3, 2, 1, 4};

  const float input_min = -12.7f;
  const float input_max = 12.8f;
  const float output_min = -12.7f;
  const float output_max = 12.8f;

  const uint8_t input1_values[] = {
      F2Q(1.0, input_min, input_max),
      F2Q(3.0, input_min, input_max),
      F2Q(4.0, input_min, input_max),
      F2Q(7.0, input_min, input_max),
  };

  const uint8_t input2_values[] = {
      F2Q(1.1, input_min, input_max),
      F2Q(3.1, input_min, input_max),
      F2Q(4.1, input_min, input_max),
      F2Q(7.1, input_min, input_max),
  };

  const uint8_t output_values[] = {
      F2Q(1.0, output_min, output_max), F2Q(3.0, output_min, output_max),
      F2Q(1.1, output_min, output_max), F2Q(3.1, output_min, output_max),
      F2Q(4.0, output_min, output_max), F2Q(7.0, output_min, output_max),
      F2Q(4.1, output_min, output_max), F2Q(7.1, output_min, output_max),
  };

  uint8_t output_data[8];
  tflite::testing::TestConcatenateQuantizedTwoInputs(
      input_shape, input1_values, input_min, input_max, input_shape,
      input2_values, input_min, input_max, axis, fixed_point_scaling,
      output_shape, output_values, output_min, output_max, output_data);
}

TF_LITE_MICRO_TEST(TwoInputsQuantizedUint8MixedRange) {
  using tflite::testing::F2Q;

  const int axis = 2;
  const int input_shape[] = {3, 2, 1, 2};
  const int output_shape[] = {3, 2, 1, 4};

  const float input1_min = -12.7f;
  const float input1_max = 12.8f;
  const float input2_min = -7.1f;
  const float input2_max = 7.2f;
  const float output_min = -12.7f;
  const float output_max = 12.8f;

  const uint8_t input1_values[] = {
      F2Q(1.0, input1_min, input1_max),
      F2Q(3.0, input1_min, input1_max),
      F2Q(4.0, input1_min, input1_max),
      F2Q(7.0, input1_min, input1_max),
  };

  const uint8_t input2_values[] = {
      F2Q(1.1, input2_min, input2_max),
      F2Q(3.1, input2_min, input2_max),
      F2Q(4.1, input2_min, input2_max),
      F2Q(7.1, input2_min, input2_max),
  };

  const uint8_t output_values[] = {
      F2Q(1.0, output_min, output_max), F2Q(3.0, output_min, output_max),
      F2Q(1.1, output_min, output_max), F2Q(3.1, output_min, output_max),
      F2Q(4.0, output_min, output_max), F2Q(7.0, output_min, output_max),
      F2Q(4.1, output_min, output_max), F2Q(7.1, output_min, output_max),
  };

  uint8_t output_data[8];
  tflite::testing::TestConcatenateQuantizedTwoInputs(
      input_shape, input1_values, input1_min, input1_max, input_shape,
      input2_values, input2_min, input2_max, axis,
      false /* fixed_point_scaling */, output_shape, output_values, output_min,
      output_max, output_data);

  tflite::testing::TestConcatenateQuantizedTwoInputs(
      input_shape, input1_values, input1_min, input1_max, input_shape,
      input2_values, input2_min, input2_max, axis,
      true /* fixed_point_scaling */, output_shape, output_values, output_min,
      output_max, output_data);
}

TF_LITE_MICRO_TEST(ThreeDimensionalTwoInputsDifferentShapes) {
  const int axis = 1;

  const int input1_shape[] = {3, 2, 1, 2};
  const int input2_shape[] = {3, 2, 3, 2};
  const int output_shape[] = {3, 2, 4, 2};

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
