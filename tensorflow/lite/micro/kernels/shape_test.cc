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

void ValidateShape(TfLiteTensor* tensors, const int tensor_count,
                   int32_t* output_data, const int32_t* expected_output,
                   int output_dims_count) {
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = tflite::Register_SHAPE();
  micro::KernelRunner runner(registration, tensors, tensor_count, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output[i], output_data[i]);
  }
}

void TestShape(const int* input_dims_data, const float* input_data,
               const int* output_dims_data, const int32_t* expected_output_data,
               int32_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims, true),
  };

  ValidateShape(tensors, tensors_size, output_data, expected_output_data,
                output_dims_count);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestShape0) {
  int input_shape[] = {1, 5};
  float input_values[] = {1, 3, 1, 3, 5};
  int output_dims[] = {1, 1};  // this is actually input_shapes shape
  int32_t expected_output_data[] = {5};
  int32_t output_data[1];

  tflite::testing::TestShape(input_shape, input_values, output_dims,
                             expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(TestShape1) {
  int input_shape[] = {2, 4, 3};
  float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int output_dims[] = {2, 1, 1};
  int32_t expected_output_data[] = {4, 3};
  int32_t output_data[2];

  tflite::testing::TestShape(input_shape, input_values, output_dims,
                             expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(TestShape2) {
  int input_shape[] = {2, 12, 1};
  float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int output_dims[] = {2, 1, 1};
  int32_t expected_output_data[] = {12, 1};
  int32_t output_data[2];

  tflite::testing::TestShape(input_shape, input_values, output_dims,
                             expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(TestShape3) {
  int input_shape[] = {2, 2, 6};
  float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int output_dims[] = {2, 1, 1};
  int32_t expected_output_data[] = {2, 6};
  int32_t output_data[2];

  tflite::testing::TestShape(input_shape, input_values, output_dims,
                             expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(TestShape4) {
  int input_shape[] = {2, 2, 2, 3};
  float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int output_dims[] = {3, 1, 1, 1};
  int32_t expected_output_data[] = {2, 2, 3};
  int32_t output_data[3];

  tflite::testing::TestShape(input_shape, input_values, output_dims,
                             expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(TestShape5) {
  int input_shape[] = {1, 1};
  float input_values[] = {1};
  int output_dims[] = {1, 1};
  int32_t expected_output_data[] = {1};
  int32_t output_data[1];

  tflite::testing::TestShape(input_shape, input_values, output_dims,
                             expected_output_data, output_data);
}

TF_LITE_MICRO_TESTS_END
