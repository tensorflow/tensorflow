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
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void TestReluFloat(std::initializer_list<int> input_dims_data,
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
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU, 1);
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

void TestRelu6Float(std::initializer_list<int> input_dims_data,
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
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU6, 1);
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

void TestReluUint8(std::initializer_list<int> input_dims_data,
                   std::initializer_list<uint8_t> input_data,
                   const float input_min, const float input_max,
                   std::initializer_list<uint8_t> expected_output_data,
                   std::initializer_list<int> output_dims_data,
                   const float output_min, const float output_max,
                   uint8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, "input_tensor", input_min,
                            input_max),
      CreateQuantizedTensor(output_data, output_dims, "output_tensor",
                            output_min, output_max),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU, 1);
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

void TestRelu6Uint8(std::initializer_list<int> input_dims_data,
                    std::initializer_list<uint8_t> input_data,
                    const float input_min, const float input_max,
                    std::initializer_list<uint8_t> expected_output_data,
                    std::initializer_list<int> output_dims_data,
                    const float output_min, const float output_max,
                    uint8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, "input_tensor", input_min,
                            input_max),
      CreateQuantizedTensor(output_data, output_dims, "output_tensor",
                            output_min, output_max),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU6, 1);
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

void TestReluInt8(std::initializer_list<int> input_dims_data,
                  std::initializer_list<int8_t> input_data,
                  const float input_min, const float input_max,
                  std::initializer_list<int8_t> expected_output_data,
                  std::initializer_list<int> output_dims_data,
                  const float output_min, const float output_max,
                  int8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, "input_tensor", input_min,
                            input_max),
      CreateQuantizedTensor(output_data, output_dims, "output_tensor",
                            output_min, output_max),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU, 1);
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

void TestRelu6Int8(std::initializer_list<int> input_dims_data,
                   std::initializer_list<int8_t> input_data,
                   const float input_min, const float input_max,
                   std::initializer_list<int8_t> expected_output_data,
                   std::initializer_list<int> output_dims_data,
                   const float output_min, const float output_max,
                   int8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, "input_tensor", input_min,
                            input_max),
      CreateQuantizedTensor(output_data, output_dims, "output_tensor",
                            output_min, output_max),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU6, 1);
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

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleReluTestFloat) {
  const int output_elements_count = 10;
  float output_data[output_elements_count];
  tflite::testing::TestReluFloat({2, 1, 5},  // Input shape.
                                 {
                                     1.0,
                                     2.0,
                                     3.0,
                                     4.0,
                                     5.0,
                                     -1.0,
                                     -2.0,
                                     -3.0,
                                     -4.0,
                                     -5.0,
                                 },
                                 {
                                     // Expected results.
                                     1.0,
                                     2.0,
                                     3.0,
                                     4.0,
                                     5.0,
                                     0.0,
                                     0.0,
                                     0.0,
                                     0.0,
                                     0.0,
                                 },
                                 {2, 1, 5},  // Output shape.
                                 output_data);
}

TF_LITE_MICRO_TEST(SimpleRelu6TestFloat) {
  const int output_elements_count = 10;
  float output_data[output_elements_count];
  tflite::testing::TestRelu6Float({2, 1, 5},  // Input shape.
                                  {
                                      4.0,
                                      5.0,
                                      6.0,
                                      7.0,
                                      8.0,
                                      -4.0,
                                      -5.0,
                                      -6.0,
                                      -7.0,
                                      -8.0,
                                  },
                                  {
                                      // Expected results.
                                      4.0,
                                      5.0,
                                      6.0,
                                      6.0,
                                      6.0,
                                      0.0,
                                      0.0,
                                      0.0,
                                      0.0,
                                      0.0,
                                  },
                                  {2, 1, 5},  // Output shape.
                                  output_data);
}

TF_LITE_MICRO_TEST(SimpleReluTestUint8) {
  using tflite::testing::F2Q;

  const float input_min = -15.9375;
  const float input_max = 15.9375;
  const float output_min = -15.9375;
  const float output_max = 15.9375;

  const int output_elements_count = 10;
  uint8_t output_data[output_elements_count];
  tflite::testing::TestReluUint8(
      {2, 1, 5},  // Input shape.
      {F2Q(1.0, input_min, input_max), F2Q(2.0, input_min, input_max),
       F2Q(3.0, input_min, input_max), F2Q(4.0, input_min, input_max),
       F2Q(5.0, input_min, input_max), F2Q(-1.0, input_min, input_max),
       F2Q(-2.0, input_min, input_max), F2Q(-3.0, input_min, input_max),
       F2Q(-4.0, input_min, input_max), F2Q(-5.0, input_min, input_max)},
      input_min, input_max,
      {// Expected results.
       F2Q(1.0, input_min, input_max), F2Q(2.0, input_min, input_max),
       F2Q(3.0, input_min, input_max), F2Q(4.0, input_min, input_max),
       F2Q(5.0, input_min, input_max), F2Q(0.0, input_min, input_max),
       F2Q(0.0, input_min, input_max), F2Q(0.0, input_min, input_max),
       F2Q(0.0, input_min, input_max), F2Q(0.0, input_min, input_max)},
      {2, 1, 5},  // Output shape.
      output_min, output_max, output_data);
}

TF_LITE_MICRO_TEST(SimpleRelu6TestUint8) {
  using tflite::testing::F2Q;

  const float input_min = -15.9375;
  const float input_max = 15.9375;
  const float output_min = -15.9375;
  const float output_max = 15.9375;

  const int output_elements_count = 10;
  uint8_t output_data[output_elements_count];
  tflite::testing::TestRelu6Uint8(
      {2, 1, 5},  // Input shape.
      {F2Q(4.0, input_min, input_max), F2Q(5.0, input_min, input_max),
       F2Q(6.0, input_min, input_max), F2Q(7.0, input_min, input_max),
       F2Q(8.0, input_min, input_max), F2Q(-1.0, input_min, input_max),
       F2Q(-2.0, input_min, input_max), F2Q(-3.0, input_min, input_max),
       F2Q(-4.0, input_min, input_max), F2Q(-5.0, input_min, input_max)},
      input_min, input_max,
      {// Expected results.
       F2Q(4.0, input_min, input_max), F2Q(5.0, input_min, input_max),
       F2Q(6.0, input_min, input_max), F2Q(6.0, input_min, input_max),
       F2Q(6.0, input_min, input_max), F2Q(0.0, input_min, input_max),
       F2Q(0.0, input_min, input_max), F2Q(0.0, input_min, input_max),
       F2Q(0.0, input_min, input_max), F2Q(0.0, input_min, input_max)},
      {2, 1, 5},  // Output shape.
      output_min, output_max, output_data);
}

TF_LITE_MICRO_TEST(SimpleReluTestInt8) {
  using tflite::testing::F2QS;

  const float input_min = -15.9375;
  const float input_max = 15.9375;
  const float output_min = -15.9375;
  const float output_max = 15.9375;

  const int output_elements_count = 10;
  int8_t output_data[output_elements_count];
  tflite::testing::TestReluInt8(
      {2, 1, 5},  // Input shape.
      {F2QS(1.0, input_min, input_max), F2QS(2.0, input_min, input_max),
       F2QS(3.0, input_min, input_max), F2QS(4.0, input_min, input_max),
       F2QS(5.0, input_min, input_max), F2QS(-1.0, input_min, input_max),
       F2QS(-2.0, input_min, input_max), F2QS(-3.0, input_min, input_max),
       F2QS(-4.0, input_min, input_max), F2QS(-5.0, input_min, input_max)},
      input_min, input_max,
      {// Expected results.
       F2QS(1.0, input_min, input_max), F2QS(2.0, input_min, input_max),
       F2QS(3.0, input_min, input_max), F2QS(4.0, input_min, input_max),
       F2QS(5.0, input_min, input_max), F2QS(0.0, input_min, input_max),
       F2QS(0.0, input_min, input_max), F2QS(0.0, input_min, input_max),
       F2QS(0.0, input_min, input_max), F2QS(0.0, input_min, input_max)},
      {2, 1, 5},  // Output shape.
      output_min, output_max, output_data);
}

TF_LITE_MICRO_TEST(SimpleRelu6TestInt8) {
  using tflite::testing::F2QS;

  const float input_min = -15.9375;
  const float input_max = 15.9375;
  const float output_min = -15.9375;
  const float output_max = 15.9375;

  const int output_elements_count = 10;
  int8_t output_data[output_elements_count];
  tflite::testing::TestRelu6Int8(
      {2, 1, 5},  // Input shape.
      {F2QS(4.0, input_min, input_max), F2QS(5.0, input_min, input_max),
       F2QS(6.0, input_min, input_max), F2QS(7.0, input_min, input_max),
       F2QS(8.0, input_min, input_max), F2QS(-1.0, input_min, input_max),
       F2QS(-2.0, input_min, input_max), F2QS(-3.0, input_min, input_max),
       F2QS(-4.0, input_min, input_max), F2QS(-5.0, input_min, input_max)},
      input_min, input_max,
      {// Expected results.
       F2QS(4.0, input_min, input_max), F2QS(5.0, input_min, input_max),
       F2QS(6.0, input_min, input_max), F2QS(6.0, input_min, input_max),
       F2QS(6.0, input_min, input_max), F2QS(0.0, input_min, input_max),
       F2QS(0.0, input_min, input_max), F2QS(0.0, input_min, input_max),
       F2QS(0.0, input_min, input_max), F2QS(0.0, input_min, input_max)},
      {2, 1, 5},  // Output shape.
      output_min, output_max, output_data);
}

TF_LITE_MICRO_TESTS_END
