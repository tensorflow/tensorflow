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
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void TestPreluFloat(std::initializer_list<int> input_dims_data,
                    std::initializer_list<float> input_data,
                    std::initializer_list<int> alpha_dims_data,
                    std::initializer_list<float> alpha_data,
                    std::initializer_list<float> expected_output_data,
                    std::initializer_list<int> output_dims_data,
                    float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* alpha_dims = IntArrayFromInitializer(alpha_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateFloatTensor(alpha_data, alpha_dims, "alpha_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_PRELU, 1);
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
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data.begin()[i], output_data[i],
                              1e-5f);
  }
}

void TestPreluQuantized(std::initializer_list<int> input_dims_data,
                        std::initializer_list<uint8_t> input_data,
                        float input_min, float input_max,
                        std::initializer_list<int> alpha_dims_data,
                        std::initializer_list<uint8_t> alpha_data,
                        float alpha_min, float alpha_max,
                        std::initializer_list<uint8_t> expected_output_data,
                        std::initializer_list<int> output_dims_data,
                        float output_min, float output_max,
                        uint8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* alpha_dims = IntArrayFromInitializer(alpha_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, "input_tensor", input_min,
                            input_max),
      CreateQuantizedTensor(alpha_data, alpha_dims, "alpha_tensor", alpha_min,
                            alpha_max),
      CreateQuantizedTensor(output_data, output_dims, "output_tensor",
                            output_min, output_max),
  };
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_PRELU, 1);
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
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data.begin()[i], output_data[i]);
  }
}
}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatPreluActivationsOpTest) {
  const int output_dims_count = 12;
  float output_data[output_dims_count];
  tflite::testing::TestPreluFloat({1, 2, 2, 3},  // input shape
                                  {
                                      0.0f, 0.0f, 0.0f,     // Row 1, Column 1
                                      1.0f, 1.0f, 1.0f,     // Row 1, Column 2
                                      -1.0f, -1.0f, -1.0f,  // Row 2, Column 1
                                      -2.0f, -2.0f, -2.0f,  // Row 1, Column 2
                                  },
                                  {1, 1, 1, 3},        // alpha shape
                                  {0.0f, 1.0f, 2.0f},  // alpha values
                                  {
                                      0.0f, 0.0f, 0.0f,    // Row 1, Column 1
                                      1.0f, 1.0f, 1.0f,    // Row 1, Column 2
                                      0.0f, -1.0f, -2.0f,  // Row 2, Column 1
                                      0.0f, -2.0f, -4.0f,  // Row 1, Column 2
                                  },
                                  {1, 2, 2, 3},  // output shape
                                  output_data);
}

TF_LITE_MICRO_TEST(QuantizedPreluActivationsOpTest) {
  using tflite::testing::F2Q;
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  const float kAlphaMin = -0.5f;
  const float kAlphaMax = 0.5f;
  const int output_dims_count = 12;
  uint8_t output_data[output_dims_count];
  tflite::testing::TestPreluQuantized(
      {1, 2, 2, 3},  // input shape
      {F2Q(0.0f, kMin, kMax), F2Q(0.0f, kMin, kMax), F2Q(0.0f, kMin, kMax),
       F2Q(0.5f, kMin, kMax), F2Q(0.5f, kMin, kMax), F2Q(0.5f, kMin, kMax),
       F2Q(-1.0f, kMin, kMax), F2Q(-1.0f, kMin, kMax), F2Q(-1.0f, kMin, kMax),
       F2Q(-0.25f, kMin, kMax), F2Q(-0.25f, kMin, kMax),
       F2Q(-0.25f, kMin, kMax)},
      kMin, kMax, {1, 1, 1, 3},  // alpha shape
      {F2Q(0.0f, kMin, kMax), F2Q(0.5f, kMin, kMax), F2Q(-0.5f, kMin, kMax)},
      kMin, kMax,
      {F2Q(0.0f, kMin, kMax), F2Q(0.0f, kMin, kMax), F2Q(0.0f, kMin, kMax),
       F2Q(0.5f, kMin, kMax), F2Q(0.5f, kMin, kMax), F2Q(0.5f, kMin, kMax),
       F2Q(0.0f, kMin, kMax), F2Q(-0.5f, kMin, kMax), F2Q(0.5f, kMin, kMax),
       F2Q(0.0f, kMin, kMax), F2Q(-0.125f, kMin, kMax),
       F2Q(0.125f, kMin, kMax)},
      {1, 2, 2, 3},  // output shape
      kMin, kMax, output_data);
}

TF_LITE_MICRO_TESTS_END
