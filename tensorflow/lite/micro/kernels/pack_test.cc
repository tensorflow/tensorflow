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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

template <typename T>
void ValidatePackGoldens(TfLiteTensor* tensors, int tensors_size,
                         TfLitePackParams params, TfLiteIntArray* inputs_array,
                         TfLiteIntArray* outputs_array, const T* golden,
                         int output_len, float tolerance, T* output) {
  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output_len; ++i) {
    output[i] = 23;
  }

  const TfLiteRegistration registration = tflite::ops::micro::Register_PACK();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, reinterpret_cast<void*>(&params));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output[i], tolerance);
  }
}

void TestPackTwoInputsFloat(const int* input1_dims_data,
                            const float* input1_data,
                            const int* input2_dims_data,
                            const float* input2_data, int axis,
                            const int* output_dims_data,
                            const float* expected_output_data,
                            float* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int input_size = 2;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {CreateTensor(input1_data, input1_dims),
                                        CreateTensor(input2_data, input2_dims),
                                        CreateTensor(output_data, output_dims)};

  TfLitePackParams builtin_data = {
      .values_count = 2,
      .axis = axis,
  };
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  ValidatePackGoldens(tensors, tensors_size, builtin_data, inputs_array,
                      outputs_array, expected_output_data, output_dims_count,
                      1e-5f, output_data);
}

void TestPackThreeInputsFloat(
    const int* input1_dims_data, const float* input1_data,
    const int* input2_dims_data, const float* input2_data,
    const int* input3_dims_data, const float* input3_data, int axis,
    const int* output_dims_data, const float* expected_output_data,
    float* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* input3_dims = IntArrayFromInts(input3_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int input_size = 3;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {CreateTensor(input1_data, input1_dims),
                                        CreateTensor(input2_data, input2_dims),
                                        CreateTensor(input3_data, input3_dims),
                                        CreateTensor(output_data, output_dims)};

  TfLitePackParams builtin_data = {
      .values_count = 3,
      .axis = axis,
  };
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  ValidatePackGoldens(tensors, tensors_size, builtin_data, inputs_array,
                      outputs_array, expected_output_data, output_dims_count,
                      1e-5f, output_data);
}

void TestPackTwoInputsQuantized(const int* input1_dims_data,
                                const uint8_t* input1_data,
                                const int* input2_dims_data,
                                const uint8_t* input2_data, int axis,
                                const int* output_dims_data,
                                const uint8_t* expected_output_data,
                                uint8_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int input_size = 2;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      // CreateQuantizedTensor needs scale/zero_point values as input, but these
      // values don't matter as to the functionality of PACK, so just set as 1.0
      // and 128.
      CreateQuantizedTensor(input1_data, input1_dims, 1.0, 128),
      CreateQuantizedTensor(input2_data, input2_dims, 1.0, 128),
      CreateQuantizedTensor(output_data, output_dims, 1.0, 128)};

  TfLitePackParams builtin_data = {
      .values_count = 2,
      .axis = axis,
  };
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  ValidatePackGoldens(tensors, tensors_size, builtin_data, inputs_array,
                      outputs_array, expected_output_data, output_dims_count,
                      1e-5f, output_data);
}

void TestPackTwoInputsQuantized32(const int* input1_dims_data,
                                  const int32_t* input1_data,
                                  const int* input2_dims_data,
                                  const int32_t* input2_data, int axis,
                                  const int* output_dims_data,
                                  const int32_t* expected_output_data,
                                  int32_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int input_size = 2;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {CreateTensor(input1_data, input1_dims),
                                        CreateTensor(input2_data, input2_dims),
                                        CreateTensor(output_data, output_dims)};

  TfLitePackParams builtin_data = {
      .values_count = 2,
      .axis = axis,
  };
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  ValidatePackGoldens(tensors, tensors_size, builtin_data, inputs_array,
                      outputs_array, expected_output_data, output_dims_count,
                      1e-5f, output_data);
}

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(PackFloatThreeInputs) {
  const int input_shape[] = {1, 2};
  const int output_shape[] = {2, 3, 2};
  const float input1_values[] = {1, 4};
  const float input2_values[] = {2, 5};
  const float input3_values[] = {3, 6};
  const float golden[] = {1, 4, 2, 5, 3, 6};
  const int axis = 0;
  constexpr int output_dims_count = 6;
  float output_data[output_dims_count];

  tflite::testing::TestPackThreeInputsFloat(
      input_shape, input1_values, input_shape, input2_values, input_shape,
      input3_values, axis, output_shape, golden, output_data);
}

TF_LITE_MICRO_TEST(PackFloatThreeInputsDifferentAxis) {
  const int input_shape[] = {1, 2};
  const int output_shape[] = {2, 2, 3};
  const float input1_values[] = {1, 4};
  const float input2_values[] = {2, 5};
  const float input3_values[] = {3, 6};
  const float golden[] = {1, 2, 3, 4, 5, 6};
  const int axis = 1;
  constexpr int output_dims_count = 6;
  float output_data[output_dims_count];

  tflite::testing::TestPackThreeInputsFloat(
      input_shape, input1_values, input_shape, input2_values, input_shape,
      input3_values, axis, output_shape, golden, output_data);
}

TF_LITE_MICRO_TEST(PackFloatThreeInputsNegativeAxis) {
  const int input_shape[] = {1, 2};
  const int output_shape[] = {2, 2, 3};
  const float input1_values[] = {1, 4};
  const float input2_values[] = {2, 5};
  const float input3_values[] = {3, 6};
  const float golden[] = {1, 2, 3, 4, 5, 6};
  const int axis = -1;
  constexpr int output_dims_count = 6;
  float output_data[output_dims_count];

  tflite::testing::TestPackThreeInputsFloat(
      input_shape, input1_values, input_shape, input2_values, input_shape,
      input3_values, axis, output_shape, golden, output_data);
}

TF_LITE_MICRO_TEST(PackFloatMultilDimensions) {
  const int input_shape[] = {2, 2, 3};
  const int output_shape[] = {3, 2, 2, 3};
  const float input1_values[] = {1, 2, 3, 4, 5, 6};
  const float input2_values[] = {7, 8, 9, 10, 11, 12};
  const float golden[] = {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12};
  const int axis = 1;
  constexpr int output_dims_count = 12;
  float output_data[output_dims_count];

  tflite::testing::TestPackTwoInputsFloat(input_shape, input1_values,
                                          input_shape, input2_values, axis,
                                          output_shape, golden, output_data);
}

TF_LITE_MICRO_TEST(PackQuantizedMultilDimensions) {
  const int input_shape[] = {2, 2, 3};
  const int output_shape[] = {3, 2, 2, 3};
  const uint8_t input1_values[] = {1, 2, 3, 4, 5, 6};
  const uint8_t input2_values[] = {7, 8, 9, 10, 11, 12};
  const uint8_t golden[] = {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12};
  const int axis = 1;
  constexpr int output_dims_count = 12;
  uint8_t output_data[output_dims_count];

  tflite::testing::TestPackTwoInputsQuantized(
      input_shape, input1_values, input_shape, input2_values, axis,
      output_shape, golden, output_data);
}

TF_LITE_MICRO_TEST(PackQuantized32MultilDimensions) {
  const int input_shape[] = {2, 2, 3};
  const int output_shape[] = {3, 2, 2, 3};
  const int32_t input1_values[] = {1, 2, 3, 4, 5, 6};
  const int32_t input2_values[] = {7, 8, 9, 10, 11, 12};
  const int32_t golden[] = {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12};
  const int axis = 1;
  constexpr int output_dims_count = 12;
  int32_t output_data[output_dims_count];

  tflite::testing::TestPackTwoInputsQuantized32(
      input_shape, input1_values, input_shape, input2_values, axis,
      output_shape, golden, output_data);
}

TF_LITE_MICRO_TESTS_END
