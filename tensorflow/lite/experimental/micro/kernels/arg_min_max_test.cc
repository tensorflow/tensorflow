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
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/simple_tensor_allocator.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

// If expected output is empty, the test is expected to fail.
void TestArgMinMax(TfLiteTensor* input_tensor, TfLiteTensor* axis_tensor,
                   TfLiteTensor* output_tensor,
                   std::initializer_list<int> expected_output_data,
                   bool using_min = false) {
  const int output_dims_count = ElementCount(*output_tensor->dims);
  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      *input_tensor,
      *axis_tensor,
      *output_tensor,
  };
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);
  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration;
  if (using_min) {
    registration = resolver.FindOp(tflite::BuiltinOperator_ARG_MIN, 1);
  } else {
    registration = resolver.FindOp(tflite::BuiltinOperator_ARG_MAX, 1);
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, init_data_size);
  }
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
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
  if (!expected_output_data.size()) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                            registration->invoke(&context, &node));
    return;
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data.begin()[i],
                              output_tensor->data.i32[i], 1e-5f);
  }
}
}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(GetMaxArgFloat) {
  int32_t output_data[1];
  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInitializer({4, 1, 1, 1, 4});
  auto input_tensor = tflite::testing::CreateFloatTensor(
      {0.1, 0.9, 0.7, 0.3}, input_dims, "input_tensor");
  auto axis_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      {3}, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "axis_tensor");
  auto output_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      output_data, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "output_tensor");
  tflite::testing::TestArgMinMax(&input_tensor, &axis_tensor, &output_tensor,
                                 {1});
}

TF_LITE_MICRO_TEST(GetMaxArgUInt8) {
  using tflite::testing::F2Q;
  int32_t output_data[1];
  float input_min = 0;
  float input_max = 15.9375;
  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInitializer({4, 1, 1, 1, 4});
  auto input_data = {
      F2Q(1., input_min, input_max), F2Q(9., input_min, input_max),
      F2Q(7., input_min, input_max), F2Q(3., input_min, input_max)};
  auto input_tensor = tflite::testing::CreateQuantizedTensor(
      input_data, input_dims, "input_tensor", input_min, input_max);
  auto axis_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      {3}, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "axis_tensor");
  auto output_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      output_data, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "output_tensor");
  tflite::testing::TestArgMinMax(&input_tensor, &axis_tensor, &output_tensor,
                                 {1});
}

TF_LITE_MICRO_TEST(GetMaxArgInt8) {
  int32_t output_data[1];
  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInitializer({4, 1, 1, 1, 4});
  std::initializer_list<int8_t> input_data = {1, 9, 7, 3};
  auto input_tensor = tflite::testing::CreateTensor<int8_t, kTfLiteInt8>(
      input_data, input_dims, "input_tensor");
  auto axis_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      {3}, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "axis_tensor");
  auto output_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      output_data, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "output_tensor");
  tflite::testing::TestArgMinMax(&input_tensor, &axis_tensor, &output_tensor,
                                 {1});
}

TF_LITE_MICRO_TEST(GetMaxArgInt32) {
  using tflite::testing::F2Q32;
  int32_t output_data[1];
  float input_min = 0;
  float input_max = 31.9375;
  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInitializer({4, 1, 1, 1, 4});
  auto input_data = {
      F2Q32(1, input_min, input_max), F2Q32(9, input_min, input_max),
      F2Q32(7, input_min, input_max), F2Q32(3, input_min, input_max)};
  auto input_tensor = tflite::testing::CreateQuantized32Tensor(
      input_data, input_dims, "input_tensor", input_min, input_max);
  auto axis_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      {3}, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "axis_tensor");
  auto output_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      output_data, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "output_tensor");
  tflite::testing::TestArgMinMax(&input_tensor, &axis_tensor, &output_tensor,
                                 {});  // Expects {1} if supported.
}

TF_LITE_MICRO_TEST(GetMaxArgMulDimensions) {
  using tflite::testing::F2Q;
  int32_t output_data[2];
  float input_min = 0;
  float input_max = 15.9375;
  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInitializer({4, 1, 1, 2, 4});
  auto input_data = {
      F2Q(1, input_min, input_max), F2Q(2, input_min, input_max),
      F2Q(7, input_min, input_max), F2Q(8, input_min, input_max),
      F2Q(1, input_min, input_max), F2Q(9, input_min, input_max),
      F2Q(7, input_min, input_max), F2Q(3, input_min, input_max)};
  auto input_tensor = tflite::testing::CreateQuantizedTensor(
      input_data, input_dims, "input_tensor", input_min, input_max);
  auto axis_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      {3}, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "axis_tensor");
  auto output_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      output_data, tflite::testing::IntArrayFromInitializer({3, 1, 1, 2}),
      "output_tensor");
  tflite::testing::TestArgMinMax(&input_tensor, &axis_tensor, &output_tensor,
                                 {3, 1});
}

TF_LITE_MICRO_TEST(GetMaxArgNegativeAxis) {
  using tflite::testing::F2Q;
  int32_t output_data[4];
  float input_min = 0;
  float input_max = 15.9375;
  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInitializer({4, 1, 1, 2, 4});
  auto input_data = {
      F2Q(1, input_min, input_max), F2Q(2, input_min, input_max),
      F2Q(7, input_min, input_max), F2Q(8, input_min, input_max),
      F2Q(1, input_min, input_max), F2Q(9, input_min, input_max),
      F2Q(7, input_min, input_max), F2Q(3, input_min, input_max)};
  auto input_tensor = tflite::testing::CreateQuantizedTensor(
      input_data, input_dims, "input_tensor", input_min, input_max);
  auto axis_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      {-2}, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "axis_tensor");
  auto output_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      output_data, tflite::testing::IntArrayFromInitializer({3, 1, 1, 4}),
      "output_tensor");
  tflite::testing::TestArgMinMax(&input_tensor, &axis_tensor, &output_tensor,
                                 {0, 1, 0, 0});
}

TF_LITE_MICRO_TEST(GetMaxArgOutput64) {
  using tflite::testing::F2Q;
  int64_t output_data[2];
  float input_min = 0;
  float input_max = 15.9375;
  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInitializer({4, 1, 1, 2, 4});
  auto input_data = {
      F2Q(10, input_min, input_max), F2Q(2, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(8, input_min, input_max),
      F2Q(1, input_min, input_max),  F2Q(9, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(3, input_min, input_max)};
  auto input_tensor = tflite::testing::CreateQuantizedTensor(
      input_data, input_dims, "input_tensor", input_min, input_max);
  auto axis_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      {3}, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "axis_tensor");
  auto output_tensor = tflite::testing::CreateTensor<int64_t, kTfLiteInt64>(
      output_data, tflite::testing::IntArrayFromInitializer({3, 1, 1, 2}),
      "output_tensor");
  tflite::testing::TestArgMinMax(&input_tensor, &axis_tensor, &output_tensor,
                                 {});  // Expects {0, 1} if supported.
}

TF_LITE_MICRO_TEST(GetMaxArgAxis64) {
  using tflite::testing::F2Q;
  int32_t output_data[2];
  float input_min = 0;
  float input_max = 15.9375;
  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInitializer({4, 1, 1, 2, 4});
  auto input_data = {
      F2Q(10, input_min, input_max), F2Q(2, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(8, input_min, input_max),
      F2Q(1, input_min, input_max),  F2Q(9, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(3, input_min, input_max)};
  auto input_tensor = tflite::testing::CreateQuantizedTensor(
      input_data, input_dims, "input_tensor", input_min, input_max);
  auto axis_tensor = tflite::testing::CreateTensor<int64_t, kTfLiteInt64>(
      {3}, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "axis_tensor");
  auto output_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      output_data, tflite::testing::IntArrayFromInitializer({3, 1, 1, 2}),
      "output_tensor");
  tflite::testing::TestArgMinMax(&input_tensor, &axis_tensor, &output_tensor,
                                 {});  // Expects {0, 1} if supported.
}

TF_LITE_MICRO_TEST(GetMinArgFloat) {
  int32_t output_data[1];
  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInitializer({4, 1, 1, 1, 4});
  auto input_tensor = tflite::testing::CreateFloatTensor(
      {0.1, 0.9, 0.7, 0.3}, input_dims, "input_tensor");
  auto axis_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      {3}, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "axis_tensor");
  auto output_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      output_data, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "output_tensor");
  tflite::testing::TestArgMinMax(&input_tensor, &axis_tensor, &output_tensor,
                                 {0}, true);
}

TF_LITE_MICRO_TEST(GetMinArgUInt8) {
  using tflite::testing::F2Q;
  float input_min = 0;
  float input_max = 15.9375;
  int32_t output_data[1];
  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInitializer({4, 1, 1, 1, 4});
  // Getting weird error when defining input_data directly in
  // CreateQuantizedTensor. So I have to define it ahead.
  auto input_data = {
      F2Q(1.0, input_min, input_max), F2Q(9.0, input_min, input_max),
      F2Q(7.0, input_min, input_max), F2Q(3.0, input_min, input_max)};
  auto input_tensor = tflite::testing::CreateQuantizedTensor(
      input_data, input_dims, "input_tensor", input_min, input_max);
  auto axis_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      {3}, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "axis_tensor");
  auto output_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      output_data, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "output_tensor");
  tflite::testing::TestArgMinMax(&input_tensor, &axis_tensor, &output_tensor,
                                 {0}, true);
}

TF_LITE_MICRO_TEST(GetMinArgInt8) {
  int32_t output_data[1];
  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInitializer({4, 1, 1, 1, 4});
  std::initializer_list<int8_t> input_data = {1, 9, 7, 3};
  auto input_tensor = tflite::testing::CreateTensor<int8_t, kTfLiteInt8>(
      input_data, input_dims, "input_tensor");
  auto axis_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      {3}, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "axis_tensor");
  auto output_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      output_data, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "output_tensor");
  tflite::testing::TestArgMinMax(&input_tensor, &axis_tensor, &output_tensor,
                                 {0}, true);
}

TF_LITE_MICRO_TEST(GetMinArgMulDimensions) {
  using tflite::testing::F2Q;
  float input_min = 0;
  float input_max = 15.9375;
  int32_t output_data[2];
  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInitializer({4, 1, 1, 2, 4});
  auto input_data = {
      F2Q(1, input_min, input_max), F2Q(2, input_min, input_max),
      F2Q(7, input_min, input_max), F2Q(8, input_min, input_max),
      F2Q(1, input_min, input_max), F2Q(9, input_min, input_max),
      F2Q(7, input_min, input_max), F2Q(3, input_min, input_max)};
  auto input_tensor = tflite::testing::CreateQuantizedTensor(
      input_data, input_dims, "input_tensor", input_min, input_max);
  auto axis_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      {3}, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "axis_tensor");
  auto output_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      output_data, tflite::testing::IntArrayFromInitializer({3, 1, 1, 2}),
      "output_tensor");
  tflite::testing::TestArgMinMax(&input_tensor, &axis_tensor, &output_tensor,
                                 {0, 0}, true);
}

TF_LITE_MICRO_TEST(GetMinArgOutput64) {
  using tflite::testing::F2Q;
  float input_min = 0;
  float input_max = 15.9375;
  int64_t output_data[2];
  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInitializer({4, 1, 1, 2, 4});
  auto input_data = {
      F2Q(10, input_min, input_max), F2Q(2, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(8, input_min, input_max),
      F2Q(1, input_min, input_max),  F2Q(9, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(3, input_min, input_max)};
  auto input_tensor = tflite::testing::CreateQuantizedTensor(
      input_data, input_dims, "input_tensor", input_min, input_max);
  auto axis_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      {3}, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "axis_tensor");
  auto output_tensor = tflite::testing::CreateTensor<int64_t, kTfLiteInt64>(
      output_data, tflite::testing::IntArrayFromInitializer({3, 1, 1, 2}),
      "output_tensor");
  tflite::testing::TestArgMinMax(&input_tensor, &axis_tensor, &output_tensor,
                                 {}, true);  // Expects {1, 0} if supported.
}

TF_LITE_MICRO_TEST(GetMinArgAxis64) {
  using tflite::testing::F2Q;
  float input_min = 0;
  float input_max = 15.9375;
  int32_t output_data[2];
  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInitializer({4, 1, 1, 2, 4});
  auto input_data = {
      F2Q(10, input_min, input_max), F2Q(2, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(8, input_min, input_max),
      F2Q(1, input_min, input_max),  F2Q(9, input_min, input_max),
      F2Q(7, input_min, input_max),  F2Q(3, input_min, input_max)};
  auto input_tensor = tflite::testing::CreateQuantizedTensor(
      input_data, input_dims, "input_tensor", input_min, input_max);
  auto axis_tensor = tflite::testing::CreateTensor<int64_t, kTfLiteInt64>(
      {3}, tflite::testing::IntArrayFromInitializer({3, 1, 1, 1}),
      "axis_tensor");
  auto output_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      output_data, tflite::testing::IntArrayFromInitializer({3, 1, 1, 2}),
      "output_tensor");
  tflite::testing::TestArgMinMax(&input_tensor, &axis_tensor, &output_tensor,
                                 {}, true);  // Expects {1, 0} if supported
}

TF_LITE_MICRO_TESTS_END
