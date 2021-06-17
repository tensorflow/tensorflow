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
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// The Logistic kernel assumes an output in the range [0, 1.0], leading to these
// quantization parameters.
const float quantized_output_scale = 1.0 / 255.0;
const int quantized_output_zero_point_int8 = -128;

const int flat_size_basic = 10;
int shape_basic[] = {2, 2, 5};
const float input_data_basic[] = {1, 2, 3, 4, 5, -1, -2, -3, -4, -5};
const float golden_basic[] = {0.73105858, 0.88079708, 0.95257413, 0.98201379,
                              0.99330715, 0.26894142, 0.11920292, 0.04742587,
                              0.01798621, 0.00669285};

const int flat_size_wide_range = 10;
int shape_wide_range[] = {2, 1, 5};
const float input_data_wide_range[]{
    1.0, 2.0, 3.0, 4.0, 93.0, -1.0, -2.0, -3.0, -4.0, -93.0,
};
const float golden_wide_range[] = {
    0.73105858, 0.88079708, 0.95257413, 0.98201379, 1.0,
    0.26894142, 0.11920292, 0.04742587, 0.01798621, 0.0,
};

template <typename T>
void ValidateLogisticGoldens(TfLiteTensor* tensors, const int tensor_count,
                             T* output_data, const T* golden,
                             int output_dims_count, float tolerance) {
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_LOGISTIC();
  micro::KernelRunner runner(registration, tensors, tensor_count, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], tolerance);
  }
}

void TestLogisticFloat(int* input_dims_data, const float* input_data,
                       const float* golden, int* output_dims_data,
                       float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
  };

  ValidateLogisticGoldens(tensors, tensors_size, output_data, golden,
                          output_elements_count, 1e-5);
}

template <typename T>
void TestLogisticQuantized(int* input_dims_data, const float* input_data,
                           T* input_quantized, const float input_scale,
                           const int input_zero_point, const float* golden,
                           T* golden_quantized, int* output_dims_data,
                           const float output_scale,
                           const int output_zero_point, int8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  tflite::Quantize(golden, golden_quantized, output_elements_count,
                   output_scale, output_zero_point);
  ValidateLogisticGoldens(tensors, tensors_size, output_data, golden_quantized,
                          output_elements_count, 1.0);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LogisticFloatBasicShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_basic];
  tflite::testing::TestLogisticFloat(
      tflite::testing::shape_basic, tflite::testing::input_data_basic,
      tflite::testing::golden_basic, tflite::testing::shape_basic, output_data);
}

TF_LITE_MICRO_TEST(LogisticQuantizedInt8BasicShouldMatchGolden) {
  const float input_scale = 0.1;
  const int input_zero_point = 0;
  int8_t input_quantized[tflite::testing::flat_size_basic];
  int8_t golden_quantized[tflite::testing::flat_size_basic];
  int8_t output_data[tflite::testing::flat_size_basic];

  tflite::testing::TestLogisticQuantized(
      tflite::testing::shape_basic, tflite::testing::input_data_basic,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::golden_basic, golden_quantized,
      tflite::testing::shape_basic, tflite::testing::quantized_output_scale,
      tflite::testing::quantized_output_zero_point_int8, output_data);
}

TF_LITE_MICRO_TEST(LogisticFloatWideRangeShouldMatchGolden) {
  float output_data[tflite::testing::flat_size_wide_range];
  tflite::testing::TestLogisticFloat(
      tflite::testing::shape_wide_range, tflite::testing::input_data_wide_range,
      tflite::testing::golden_wide_range, tflite::testing::shape_wide_range,
      output_data);
}

TF_LITE_MICRO_TEST(LogisticQuantizedInt8WideRangeShouldMatchGolden) {
  const float input_scale = 1.0;
  const int input_zero_point = 0;
  int8_t input_quantized[tflite::testing::flat_size_wide_range];
  int8_t golden_quantized[tflite::testing::flat_size_wide_range];
  int8_t output_data[tflite::testing::flat_size_wide_range];

  tflite::testing::TestLogisticQuantized(
      tflite::testing::shape_wide_range, tflite::testing::input_data_wide_range,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::golden_wide_range, golden_quantized,
      tflite::testing::shape_wide_range,
      tflite::testing::quantized_output_scale,
      tflite::testing::quantized_output_zero_point_int8, output_data);
}

TF_LITE_MICRO_TESTS_END
