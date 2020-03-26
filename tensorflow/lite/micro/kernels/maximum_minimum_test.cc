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
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void TestMaxMinFloat(tflite::BuiltinOperator op,
                     std::initializer_list<int> input1_dims_data,
                     std::initializer_list<float> input1_data,
                     std::initializer_list<int> input2_dims_data,
                     std::initializer_list<float> input2_data,
                     std::initializer_list<float> expected_output_data,
                     std::initializer_list<int> output_dims_data,
                     float* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInitializer(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInitializer(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input1_data, input1_dims, "input1_tensor"),
      CreateFloatTensor(input2_data, input2_dims, "input2_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration = resolver.FindOp(op, 1);
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

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data.begin()[i], output_data[i],
                              1e-5);
  }
}

void TestMaxMinQuantized(
    tflite::BuiltinOperator op, std::initializer_list<int> input1_dims_data,
    std::initializer_list<uint8_t> input1_data, float input1_min,
    float input1_max, std::initializer_list<int> input2_dims_data,
    std::initializer_list<uint8_t> input2_data, float input2_min,
    float input2_max, std::initializer_list<uint8_t> expected_output_data,
    float output_min, float output_max,
    std::initializer_list<int> output_dims_data, uint8_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInitializer(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInitializer(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_dims, "input1_tensor",
                            input1_min, input1_max),
      CreateQuantizedTensor(input2_data, input2_dims, "input2_tensor",
                            input2_min, input2_max),
      CreateQuantizedTensor(output_data, output_dims, "output_tensor",
                            output_min, output_max),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration = resolver.FindOp(op, 1);
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

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data.begin()[i], output_data[i]);
  }
}

void TestMaxMinQuantizedInt32(
    tflite::BuiltinOperator op, std::initializer_list<int> input1_dims_data,
    std::initializer_list<int32_t> input1_data, float input1_scale,
    std::initializer_list<int> input2_dims_data,
    std::initializer_list<int32_t> input2_data, float input2_scale,
    std::initializer_list<int32_t> expected_output_data, float output_scale,
    std::initializer_list<int> output_dims_data, int32_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInitializer(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInitializer(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantized32Tensor(input1_data, input1_dims, "input1_tensor",
                              input1_scale),
      CreateQuantized32Tensor(input2_data, input2_dims, "input2_tensor",
                              input2_scale),
      CreateQuantized32Tensor(output_data, output_dims, "output_tensor",
                              output_scale),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration = resolver.FindOp(op, 1);
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

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data.begin()[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatTest) {
  std::initializer_list<float> data1 = {1.0, 0.0, -1.0, 11.0, -2.0, -1.44};
  std::initializer_list<float> data2 = {-1.0, 0.0, 1.0, 12.0, -3.0, -1.43};
  float output_data[6];

  tflite::testing::TestMaxMinFloat(
      tflite::BuiltinOperator_MAXIMUM, {3, 3, 1, 2},
      data1,                               // input1 shape and data
      {3, 3, 1, 2}, data2,                 // input2 shape and data
      {1.0, 0.0, 1.0, 12.0, -2.0, -1.43},  // expected output
      {3, 3, 1, 2}, output_data);          // output shape and data buffer

  tflite::testing::TestMaxMinFloat(
      tflite::BuiltinOperator_MINIMUM, {3, 3, 1, 2},
      data1,                                 // input1 shape and data
      {3, 3, 1, 2}, data2,                   // input2 shape and data
      {-1.0, 0.0, -1.0, 11.0, -3.0, -1.44},  // expected output
      {3, 3, 1, 2}, output_data);            // output shape and data buffer
}

TF_LITE_MICRO_TEST(Uint8Test) {
  std::initializer_list<uint8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<uint8_t> data2 = {0, 0, 1, 12, 255, 1};
  const float input1_min = -63.5;
  const float input1_max = 64;
  const float input2_min = -63.5;
  const float input2_max = 64;
  const float output_min = -63.5;
  const float output_max = 64;

  uint8_t output_data[6];

  tflite::testing::TestMaxMinQuantized(
      tflite::BuiltinOperator_MAXIMUM,
      // input1 shape, data and bounds
      {3, 3, 1, 2}, data1, input1_min, input1_max,
      // input2 shape, data and bounds
      {3, 3, 1, 2}, data2, input2_min, input2_max,
      // expected output
      {1, 0, 2, 12, 255, 23},
      // output bounds, shape and data buffer
      output_min, output_max, {3, 3, 1, 2}, output_data);

  tflite::testing::TestMaxMinQuantized(
      tflite::BuiltinOperator_MINIMUM,
      // input1 shape, data and bounds
      {3, 3, 1, 2}, data1, input1_min, input1_max,
      // input2 shape, data and bounds
      {3, 3, 1, 2}, data2, input2_min, input2_max,
      // expected output
      {0, 0, 1, 11, 2, 1},
      // output bounds, shape and data buffer
      output_min, output_max, {3, 3, 1, 2}, output_data);
}

TF_LITE_MICRO_TEST(FloatWithBroadcastTest) {
  std::initializer_list<float> data1 = {1.0, 0.0, -1.0, -2.0, -1.44, 11.0};
  std::initializer_list<float> data2 = {0.5, 2.0};
  float output_data[6];

  tflite::testing::TestMaxMinFloat(
      tflite::BuiltinOperator_MAXIMUM, {3, 3, 1, 2},
      data1,                            // input1 shape and data
      {1, 2}, data2,                    // input2 shape and data
      {1.0, 2.0, 0.5, 2.0, 0.5, 11.0},  // expected output
      {3, 3, 1, 2}, output_data);       // output shape and data buffer

  tflite::testing::TestMaxMinFloat(
      tflite::BuiltinOperator_MINIMUM, {3, 3, 1, 2},
      data1,                               // input1 shape and data
      {1, 2}, data2,                       // input2 shape and data
      {0.5, 0.0, -1.0, -2.0, -1.44, 2.0},  // expected output
      {3, 3, 1, 2}, output_data);          // output shape and data buffer
}

TF_LITE_MICRO_TEST(Int32WithBroadcastTest) {
  const float input1_scale = 0.5;
  const float input2_scale = 0.5;
  const float output_scale = 0.5;
  std::initializer_list<int32_t> data1 = {1, 0, -1, -2, 3, 11};
  std::initializer_list<int32_t> data2 = {2};
  int32_t output_data[6];

  tflite::testing::TestMaxMinQuantizedInt32(
      tflite::BuiltinOperator_MAXIMUM,
      // input1 shape, data and scale
      {3, 3, 1, 2}, data1, input1_scale,
      // input2 shape, data and scale
      {1, 1}, data2, input2_scale,
      // expected output
      {2, 2, 2, 2, 3, 11},
      // output scale, shape and data buffer
      output_scale, {3, 3, 1, 2}, output_data);

  tflite::testing::TestMaxMinQuantizedInt32(
      tflite::BuiltinOperator_MINIMUM,
      // input1 shape, data and scale
      {3, 3, 1, 2}, data1, input1_scale,
      // input2 shape, data and scale
      {1, 1}, data2, input2_scale,
      // expected output
      {1, 0, -1, -2, 2, 2},
      // output scale, shape and data buffer
      output_scale, {3, 3, 1, 2}, output_data);
}

TF_LITE_MICRO_TESTS_END
