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
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void TestLogicalOp(tflite::BuiltinOperator op,
                   std::initializer_list<int> input1_dims_data,
                   std::initializer_list<bool> input1_data,
                   std::initializer_list<int> input2_dims_data,
                   std::initializer_list<bool> input2_data,
                   std::initializer_list<int> output_dims_data,
                   std::initializer_list<bool> expected_output_data,
                   bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInitializer(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInitializer(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateBoolTensor(input1_data, input1_dims, "input1_tensor"),
      CreateBoolTensor(input2_data, input2_dims, "input2_tensor"),
      CreateBoolTensor(output_data, output_dims, "output_tensor"),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration = resolver.FindOp(op);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteIntArray* inputs_array = IntArrayFromInitializer({2, 0, 1});
  TfLiteIntArray* outputs_array = IntArrayFromInitializer({1, 2});
  TfLiteIntArray* temporaries_array = IntArrayFromInitializer({0});

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = nullptr;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));

  TF_LITE_MICRO_EXPECT_EQ(output_dims_count, 4);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data.begin()[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LogicalOr) {
  bool output_data[4];
  tflite::testing::TestLogicalOp(
      tflite::BuiltinOperator_LOGICAL_OR,           // operator
      {4, 1, 1, 1, 4}, {true, false, false, true},  // input1
      {4, 1, 1, 1, 4}, {true, false, true, false},  // input2
      {4, 1, 1, 1, 4}, {true, false, true, true},   // expected output
      output_data);
}

TF_LITE_MICRO_TEST(BroadcastLogicalOr) {
  bool output_data[4];
  tflite::testing::TestLogicalOp(
      tflite::BuiltinOperator_LOGICAL_OR,           // operator
      {4, 1, 1, 1, 4}, {true, false, false, true},  // input1
      {4, 1, 1, 1, 1}, {false},                     // input2
      {4, 1, 1, 1, 4}, {true, false, false, true},  // expected output
      output_data);
}

TF_LITE_MICRO_TEST(LogicalAnd) {
  bool output_data[4];
  tflite::testing::TestLogicalOp(
      tflite::BuiltinOperator_LOGICAL_AND,           // operator
      {4, 1, 1, 1, 4}, {true, false, false, true},   // input1
      {4, 1, 1, 1, 4}, {true, false, true, false},   // input2
      {4, 1, 1, 1, 4}, {true, false, false, false},  // expected output
      output_data);
}

TF_LITE_MICRO_TEST(BroadcastLogicalAnd) {
  bool output_data[4];
  tflite::testing::TestLogicalOp(
      tflite::BuiltinOperator_LOGICAL_AND,          // operator
      {4, 1, 1, 1, 4}, {true, false, false, true},  // input1
      {4, 1, 1, 1, 1}, {true},                      // input2
      {4, 1, 1, 1, 4}, {true, false, false, true},  // expected output
      output_data);
}

TF_LITE_MICRO_TESTS_END
