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
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void TestLogisticFloat(std::initializer_list<int> input_dims_data,
                       std::initializer_list<float> input_data,
                       std::initializer_list<float> expected_output_data,
                       std::initializer_list<int> output_dims_data,
                       float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims),
      CreateFloatTensor(output_data, output_dims),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_LOGISTIC);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = nullptr;
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
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
  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data.begin()[i], output_data[i],
                              1e-5f);
  }
}

void TestLogisticInt8(std::initializer_list<int> input_dims_data,
                      std::initializer_list<int8_t> input_data, float input_min,
                      float input_max,
                      std::initializer_list<int8_t> expected_output_data,
                      std::initializer_list<int> output_dims_data,
                      float output_min, float output_max, int8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, input_min, input_max),
      CreateQuantizedTensor(output_data, output_dims, output_min, output_max),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_LOGISTIC);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = nullptr;
  size_t init_data_size = 1;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
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
  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data.begin()[i], output_data[i],
                              1);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleTestFloat) {
  const int output_elements_count = 10;
  float output_data[output_elements_count];
  tflite::testing::TestLogisticFloat({2, 1, 5},  // Input shape.
                                     {
                                         1.0,
                                         2.0,
                                         3.0,
                                         4.0,
                                         93.0,
                                         -1.0,
                                         -2.0,
                                         -3.0,
                                         -4.0,
                                         -93.0,
                                     },
                                     {
                                         // Expected results.
                                         0.73105858,
                                         0.88079708,
                                         0.95257413,
                                         0.98201379,
                                         1.0,
                                         0.26894142,
                                         0.11920292,
                                         0.04742587,
                                         0.01798621,
                                         0.0,
                                     },
                                     {2, 1, 5},  // Output shape.
                                     output_data);
}

TF_LITE_MICRO_TEST(SimpleTestInt8) {
  using tflite::testing::F2QS;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float output_min = 0.0f;
  const float output_max = (255.0f / 256.0f);

  const int output_elements_count = 10;
  int8_t output_data[output_elements_count];
  tflite::testing::TestLogisticInt8(
      {2, 1, output_elements_count},  // Input shape.
      {F2QS(1.0, input_min, input_max), F2QS(2.0, input_min, input_max),
       F2QS(3.0, input_min, input_max), F2QS(4.0, input_min, input_max),
       F2QS(5.0, input_min, input_max), F2QS(-1.0, input_min, input_max),
       F2QS(-2.0, input_min, input_max), F2QS(-3.0, input_min, input_max),
       F2QS(-4.0, input_min, input_max), F2QS(-5.0, input_min, input_max)},
      input_min, input_max,  // Input quantized range.
      {                      // Expected results.
       F2QS(0.73105858, output_min, output_max),
       F2QS(0.88079708, output_min, output_max),
       F2QS(0.95257413, output_min, output_max),
       F2QS(0.98201379, output_min, output_max),
       F2QS(0.99330715, output_min, output_max),
       F2QS(0.26894142, output_min, output_max),
       F2QS(0.11920292, output_min, output_max),
       F2QS(0.04742587, output_min, output_max),
       F2QS(0.01798621, output_min, output_max),
       F2QS(0.00669285, output_min, output_max)},
      {2, 1, output_elements_count},  // Output shape.
      output_min, output_max,         // Output quantized range.
      output_data);
}

TF_LITE_MICRO_TESTS_END
