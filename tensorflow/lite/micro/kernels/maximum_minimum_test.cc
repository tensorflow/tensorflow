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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

void TestMaxMinFloat(const TfLiteRegistration& registration,
                     int* input1_dims_data, const float* input1_data,
                     int* input2_dims_data, const float* input2_data,
                     const float* expected_output_data, int* output_dims_data,
                     float* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i], 1e-5f);
  }
}

void TestMaxMinQuantized(const TfLiteRegistration& registration,
                         int* input1_dims_data, const uint8_t* input1_data,
                         float const input1_scale, const int input1_zero_point,
                         int* input2_dims_data, const uint8_t* input2_data,
                         const float input2_scale, const int input2_zero_point,
                         const uint8_t* expected_output_data,
                         const float output_scale, const int output_zero_point,
                         int* output_dims_data, uint8_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_dims, input1_scale,
                            input1_zero_point),
      CreateQuantizedTensor(input2_data, input2_dims, input2_scale,
                            input2_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

void TestMaxMinQuantizedInt32(const TfLiteRegistration& registration,
                              int* input1_dims_data, const int32_t* input1_data,
                              int* input2_dims_data, const int32_t* input2_data,
                              const int32_t* expected_output_data,
                              int* output_dims_data, int32_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatTest) {
  int dims[] = {3, 3, 1, 2};
  const float data1[] = {1.0, 0.0, -1.0, 11.0, -2.0, -1.44};
  const float data2[] = {-1.0, 0.0, 1.0, 12.0, -3.0, -1.43};
  const float golden_max[] = {1.0, 0.0, 1.0, 12.0, -2.0, -1.43};
  const float golden_min[] = {-1.0, 0.0, -1.0, 11.0, -3.0, -1.44};
  float output_data[6];

  tflite::testing::TestMaxMinFloat(tflite::ops::micro::Register_MAXIMUM(), dims,
                                   data1, dims, data2, golden_max, dims,
                                   output_data);

  tflite::testing::TestMaxMinFloat(tflite::ops::micro::Register_MINIMUM(), dims,
                                   data1, dims, data2, golden_min, dims,
                                   output_data);
}

TF_LITE_MICRO_TEST(Uint8Test) {
  int dims[] = {3, 3, 1, 2};
  const uint8_t data1[] = {1, 0, 2, 11, 2, 23};
  const uint8_t data2[] = {0, 0, 1, 12, 255, 1};
  const uint8_t golden_max[] = {1, 0, 2, 12, 255, 23};
  const uint8_t golden_min[] = {0, 0, 1, 11, 2, 1};

  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;

  uint8_t output_data[6];

  tflite::testing::TestMaxMinQuantized(
      tflite::ops::micro::Register_MAXIMUM(), dims, data1, input_scale,
      input_zero_point, dims, data2, input_scale, input_zero_point, golden_max,
      output_scale, output_zero_point, dims, output_data);

  tflite::testing::TestMaxMinQuantized(
      tflite::ops::micro::Register_MINIMUM(), dims, data1, input_scale,
      input_zero_point, dims, data2, input_scale, input_zero_point, golden_min,
      output_scale, output_zero_point, dims, output_data);
}

TF_LITE_MICRO_TEST(FloatWithBroadcastTest) {
  int dims[] = {3, 3, 1, 2};
  int dims_scalar[] = {1, 2};
  const float data1[] = {1.0, 0.0, -1.0, -2.0, -1.44, 11.0};
  const float data2[] = {0.5, 2.0};
  const float golden_max[] = {1.0, 2.0, 0.5, 2.0, 0.5, 11.0};
  const float golden_min[] = {0.5, 0.0, -1.0, -2.0, -1.44, 2.0};
  float output_data[6];

  tflite::testing::TestMaxMinFloat(tflite::ops::micro::Register_MAXIMUM(), dims,
                                   data1, dims_scalar, data2, golden_max, dims,
                                   output_data);

  tflite::testing::TestMaxMinFloat(tflite::ops::micro::Register_MINIMUM(), dims,
                                   data1, dims_scalar, data2, golden_min, dims,
                                   output_data);
}

TF_LITE_MICRO_TEST(Int32WithBroadcastTest) {
  int dims[] = {3, 3, 1, 2};
  int dims_scalar[] = {1, 1};
  const int32_t data1[] = {1, 0, -1, -2, 3, 11};
  const int32_t data2[] = {2};
  const int32_t golden_max[] = {2, 2, 2, 2, 3, 11};
  const int32_t golden_min[] = {1, 0, -1, -2, 2, 2};
  int32_t output_data[6];

  tflite::testing::TestMaxMinQuantizedInt32(
      tflite::ops::micro::Register_MAXIMUM(), dims, data1, dims_scalar, data2,
      golden_max, dims, output_data);

  tflite::testing::TestMaxMinQuantizedInt32(
      tflite::ops::micro::Register_MINIMUM(), dims, data1, dims_scalar, data2,
      golden_min, dims, output_data);
}

TF_LITE_MICRO_TESTS_END
