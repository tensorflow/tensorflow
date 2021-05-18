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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

void TestLogicalOp(const TfLiteRegistration& registration,
                   int* input1_dims_data, const bool* input1_data,
                   int* input2_dims_data, const bool* input2_data,
                   int* output_dims_data, const bool* expected_output_data,
                   bool* output_data) {
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

  TF_LITE_MICRO_EXPECT_EQ(output_dims_count, 4);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LogicalOr) {
  int shape[] = {4, 1, 1, 1, 4};
  const bool input1[] = {true, false, false, true};
  const bool input2[] = {true, false, true, false};
  const bool golden[] = {true, false, true, true};
  bool output_data[4];
  tflite::testing::TestLogicalOp(tflite::ops::micro::Register_LOGICAL_OR(),
                                 shape, input1, shape, input2, shape, golden,
                                 output_data);
}

TF_LITE_MICRO_TEST(BroadcastLogicalOr) {
  int input1_shape[] = {4, 1, 1, 1, 4};
  const bool input1[] = {true, false, false, true};
  int input2_shape[] = {4, 1, 1, 1, 1};
  const bool input2[] = {false};
  const bool golden[] = {true, false, false, true};
  bool output_data[4];
  tflite::testing::TestLogicalOp(tflite::ops::micro::Register_LOGICAL_OR(),
                                 input1_shape, input1, input2_shape, input2,
                                 input1_shape, golden, output_data);
}

TF_LITE_MICRO_TEST(LogicalAnd) {
  int shape[] = {4, 1, 1, 1, 4};
  const bool input1[] = {true, false, false, true};
  const bool input2[] = {true, false, true, false};
  const bool golden[] = {true, false, false, false};
  bool output_data[4];
  tflite::testing::TestLogicalOp(tflite::ops::micro::Register_LOGICAL_AND(),
                                 shape, input1, shape, input2, shape, golden,
                                 output_data);
}

TF_LITE_MICRO_TEST(BroadcastLogicalAnd) {
  int input1_shape[] = {4, 1, 1, 1, 4};
  const bool input1[] = {true, false, false, true};
  int input2_shape[] = {4, 1, 1, 1, 1};
  const bool input2[] = {true};
  const bool golden[] = {true, false, false, true};
  bool output_data[4];
  tflite::testing::TestLogicalOp(tflite::ops::micro::Register_LOGICAL_AND(),
                                 input1_shape, input1, input2_shape, input2,
                                 input1_shape, golden, output_data);
}

TF_LITE_MICRO_TESTS_END
