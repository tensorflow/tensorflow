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
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

template <typename T>
void TestZerosLike(const int* input_dims_data, const T* input_data,
                   const T* expected_output_data, T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(input_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = Register_ZEROS_LIKE();
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

TF_LITE_MICRO_TEST(TestZerosLikeFloat) {
  float output_data[6];
  const int input_dims[] = {2, 2, 3};
  const float input_values[] = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
  const float golden[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  tflite::testing::TestZerosLike<float>(input_dims, input_values, golden,
                                        output_data);
}

TF_LITE_MICRO_TEST(TestZerosLikeInt8) {
  int8_t output_data[6];
  const int input_dims[] = {3, 1, 2, 3};
  const int8_t input_values[] = {-2, -1, 0, 1, 2, 3};
  const int8_t golden[] = {0, 0, 0, 0, 0, 0};
  tflite::testing::TestZerosLike<int8_t>(input_dims, input_values, golden,
                                         output_data);
}

TF_LITE_MICRO_TEST(TestZerosLikeInt32) {
  int32_t output_data[4];
  const int input_dims[] = {4, 1, 2, 2, 1};
  const int32_t input_values[] = {-2, -1, 0, 3};
  const int32_t golden[] = {0, 0, 0, 0};
  tflite::testing::TestZerosLike<int32_t>(input_dims, input_values, golden,
                                          output_data);
}

TF_LITE_MICRO_TEST(TestZerosLikeInt64) {
  int64_t output_data[4];
  const int input_dims[] = {4, 1, 2, 2, 1};
  const int64_t input_values[] = {-2, -1, 0, 3};
  const int64_t golden[] = {0, 0, 0, 0};
  tflite::testing::TestZerosLike<int64_t>(input_dims, input_values, golden,
                                          output_data);
}

TF_LITE_MICRO_TESTS_END
