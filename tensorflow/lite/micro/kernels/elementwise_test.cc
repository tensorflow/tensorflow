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
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {

void TestElementwiseFloat(tflite::BuiltinOperator op,
                          const int* input_dims_data, const float* input_data,
                          const int* output_dims_data,
                          const float* expected_output_data,
                          float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims),
      CreateFloatTensor(output_data, output_dims)};

  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output_dims_count; ++i) {
    output_data[i] = 23;
  }

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration = resolver.FindOp(op);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, 0);
  }
  static int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  static int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.user_data = user_data;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i], 1e-5f);
  }
}

void TestElementwiseBool(tflite::BuiltinOperator op, const int* input_dims_data,
                         const bool* input_data, const int* output_dims_data,
                         const bool* expected_output_data, bool* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateBoolTensor(input_data, input_dims),
      CreateBoolTensor(output_data, output_dims)};

  // Place false in the uninitialized output buffer.
  for (int i = 0; i < output_dims_count; ++i) {
    output_data[i] = false;
  }

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration = resolver.FindOp(op);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, 0);
  }

  const int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  const int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.user_data = user_data;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(Abs) {
  constexpr int output_dims_count = 4;
  const int shape[] = {2, 2, 2};
  const float input[] = {0.01, -0.01, 10, -10};
  const float golden[] = {0.01, 0.01, 10, 10};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::BuiltinOperator_ABS, shape,
                                        input, shape, golden, output_data);
}

TF_LITE_MICRO_TEST(Sin) {
  constexpr int output_dims_count = 4;
  const int shape[] = {2, 2, 2};
  const float input[] = {0, 3.1415926, -3.1415926, 1};
  const float golden[] = {0, 0, 0, 0.84147};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::BuiltinOperator_SIN, shape,
                                        input, shape, golden, output_data);
}

TF_LITE_MICRO_TEST(Cos) {
  constexpr int output_dims_count = 4;
  const int shape[] = {2, 2, 2};
  const float input[] = {0, 3.1415926, -3.1415926, 1};
  const float golden[] = {1, -1, -1, 0.54030};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::BuiltinOperator_COS, shape,
                                        input, shape, golden, output_data);
}

TF_LITE_MICRO_TEST(Log) {
  constexpr int output_dims_count = 4;
  const int shape[] = {2, 2, 2};
  const float input[] = {1, 2.7182818, 0.5, 2};
  const float golden[] = {0, 1, -0.6931472, 0.6931472};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::BuiltinOperator_LOG, shape,
                                        input, shape, golden, output_data);
}

TF_LITE_MICRO_TEST(Sqrt) {
  constexpr int output_dims_count = 4;
  const int shape[] = {2, 2, 2};
  const float input[] = {0, 1, 2, 4};
  const float golden[] = {0, 1, 1.41421, 2};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::BuiltinOperator_SQRT, shape,
                                        input, shape, golden, output_data);
}

TF_LITE_MICRO_TEST(Rsqrt) {
  constexpr int output_dims_count = 4;
  const int shape[] = {2, 2, 2};
  const float input[] = {1, 2, 4, 9};
  const float golden[] = {1, 0.7071, 0.5, 0.33333};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::BuiltinOperator_RSQRT, shape,
                                        input, shape, golden, output_data);
}

TF_LITE_MICRO_TEST(Square) {
  constexpr int output_dims_count = 4;
  const int shape[] = {2, 2, 2};
  const float input[] = {1, 2, 0.5, -3.0};
  const float golden[] = {1, 4.0, 0.25, 9.0};
  float output_data[output_dims_count];
  tflite::testing::TestElementwiseFloat(tflite::BuiltinOperator_SQUARE, shape,
                                        input, shape, golden, output_data);
}

TF_LITE_MICRO_TEST(LogicalNot) {
  constexpr int output_dims_count = 4;
  const int shape[] = {2, 2, 2};
  const bool input[] = {true, false, false, true};
  const bool golden[] = {false, true, true, false};
  bool output_data[output_dims_count];
  tflite::testing::TestElementwiseBool(tflite::BuiltinOperator_LOGICAL_NOT,
                                       shape, input, shape, golden,
                                       output_data);
}

TF_LITE_MICRO_TESTS_END
