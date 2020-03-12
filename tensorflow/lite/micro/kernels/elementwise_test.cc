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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {

void TestElementwiseFloat(tflite::BuiltinOperator op,
                          std::initializer_list<int> input_dims_data,
                          std::initializer_list<float> input_data,
                          std::initializer_list<int> output_dims_data,
                          std::initializer_list<float> expected_output_data,
                          float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor")};

  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output_dims_count; ++i) {
    output_data[i] = 23;
  }

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(op, /* version= */ 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, 0);
  }
  auto inputs_array_data = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInitializer(inputs_array_data);
  auto outputs_array_data = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInitializer(outputs_array_data);
  auto temporaries_array_data = {0};
  TfLiteIntArray* temporaries_array =
      IntArrayFromInitializer(temporaries_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data.begin()[i], output_data[i],
                              1e-5f);
  }
}

void TestElementwiseBool(tflite::BuiltinOperator op,
                         std::initializer_list<int> input_dims_data,
                         std::initializer_list<bool> input_data,
                         std::initializer_list<int> output_dims_data,
                         std::initializer_list<bool> expected_output_data,
                         bool* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateBoolTensor(input_data, input_dims, "input_tensor"),
      CreateBoolTensor(output_data, output_dims, "output_tensor")};

  // Place false in the uninitialized output buffer.
  for (int i = 0; i < output_dims_count; ++i) {
    output_data[i] = false;
  }

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(op, /* version= */ 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, 0);
  }
  TfLiteIntArray* inputs_array = IntArrayFromInitializer({1, 0});
  TfLiteIntArray* outputs_array = IntArrayFromInitializer({1, 1});
  TfLiteIntArray* temporaries_array = IntArrayFromInitializer({0});

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data.begin()[i], output_data[i]);
  }
}

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(Abs) {
  constexpr int output_dims_count = 4;
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(
      tflite::BuiltinOperator_ABS,  // ABS operator
      {2, 2, 2},                    // Input shape
      {0.01, -0.01, 10, -10},       // Input values
      {2, 2, 2},                    // Output shape
      {0.01, 0.01, 10, 10},         // Output values
      output_data);
}

TF_LITE_MICRO_TEST(Sin) {
  constexpr int output_dims_count = 4;
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(
      tflite::BuiltinOperator_SIN,    // SIN operator
      {2, 2, 2},                      // Input shape
      {0, 3.1415926, -3.1415926, 1},  // Input values
      {2, 2, 2},                      // Output shape
      {0, 0, 0, 0.84147},             // Output values
      output_data);
}

TF_LITE_MICRO_TEST(Cos) {
  constexpr int output_dims_count = 4;
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(
      tflite::BuiltinOperator_COS,    // COS operator
      {2, 2, 2},                      // Input shape
      {0, 3.1415926, -3.1415926, 1},  // Input values
      {2, 2, 2},                      // Output shape
      {1, -1, -1, 0.54030},           // Output values
      output_data);
}

TF_LITE_MICRO_TEST(Log) {
  constexpr int output_dims_count = 4;
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(
      tflite::BuiltinOperator_LOG,    // LOG operator
      {2, 2, 2},                      // Input shape
      {1, 2.7182818, 0.5, 2},         // Input values
      {2, 2, 2},                      // Output shape
      {0, 1, -0.6931472, 0.6931472},  // Output values
      output_data);
}

TF_LITE_MICRO_TEST(Sqrt) {
  constexpr int output_dims_count = 4;
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(
      tflite::BuiltinOperator_SQRT,  // SQRT operator
      {2, 2, 2},                     // Input shape
      {0, 1, 2, 4},                  // Input values
      {2, 2, 2},                     // Output shape
      {0, 1, 1.41421, 2},            // Output values
      output_data);
}

TF_LITE_MICRO_TEST(Rsqrt) {
  constexpr int output_dims_count = 4;
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(
      tflite::BuiltinOperator_RSQRT,  // RSQRT operator
      {2, 2, 2},                      // Input shape
      {1, 2, 4, 9},                   // Input values
      {2, 2, 2},                      // Output shape
      {1, 0.7071, 0.5, 0.33333},      // Output values
      output_data);
}

TF_LITE_MICRO_TEST(Square) {
  constexpr int output_dims_count = 4;
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(
      tflite::BuiltinOperator_SQUARE,  // SQARE operator
      {2, 2, 2},                       // Input shape
      {1, 2, 0.5, -3.0},               // Input values
      {2, 2, 2},                       // Output shape
      {1, 4.0, 0.25, 9.0},             // Output values
      output_data);
}

TF_LITE_MICRO_TEST(LogicalNot) {
  constexpr int output_dims_count = 4;
  bool output_data[output_dims_count];
  tflite::testing::TestElementwiseBool(
      tflite::BuiltinOperator_LOGICAL_NOT,  // Logical NOT operator
      {2, 2, 2},                            // Input shape
      {true, false, false, true},           // Input values
      {2, 2, 2},                            // Output shape
      {false, true, true, false},           // Output values
      output_data);
}

TF_LITE_MICRO_TESTS_END
