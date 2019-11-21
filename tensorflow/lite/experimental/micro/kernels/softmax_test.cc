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
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void TestSoftmaxFloat(std::initializer_list<int> input_dims_data,
                      std::initializer_list<float> input_data,
                      std::initializer_list<float> expected_output_data,
                      std::initializer_list<int> output_dims_data,
                      float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
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
      resolver.FindOp(tflite::BuiltinOperator_SOFTMAX, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteSoftmaxParams builtin_data = {1.0f};
  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
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

void TestSoftmaxQuantized(std::initializer_list<int> input_dims_data,
                          std::initializer_list<uint8_t> input_data,
                          float input_min, float input_max,
                          std::initializer_list<uint8_t> expected_output_data,
                          std::initializer_list<int> output_dims_data,
                          float output_min, float output_max,
                          uint8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

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
      resolver.FindOp(tflite::BuiltinOperator_SOFTMAX, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteSoftmaxParams builtin_data = {1.0f};
  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
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

void TestSoftmaxQuantizedSigned(
    std::initializer_list<int> input_dims_data,
    std::initializer_list<int8_t> input_data, float input_min, float input_max,
    std::initializer_list<int8_t> expected_output_data,
    std::initializer_list<int> output_dims_data, float output_min,
    float output_max, int8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

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
      resolver.FindOp(tflite::BuiltinOperator_SOFTMAX, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteSoftmaxParams builtin_data = {1.0f};
  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
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

TF_LITE_MICRO_TEST(SimpleTest) {
  const int output_dims_count = 10;
  float output_data[output_dims_count];
  tflite::testing::TestSoftmaxFloat(  //
      {2, 2, 5},                      // Input shape.
      {
          1.0, 2.0, 3.0, 4.0, 5.0,       // b = 0
          -1.0, -2.0, -3.0, -4.0, -5.0,  // b = 0
      },
      {
          // Expected results.
          0.011656231,
          0.031684921,
          0.086128544,
          0.234121657,
          0.636408647,
          0.636408647,
          0.234121657,
          0.086128544,
          0.031684921,
          0.011656231,
      },
      {2, 2, 5},  // Output shape.
      output_data);
}

TF_LITE_MICRO_TEST(SimpleTestQuantized) {
  using tflite::testing::F2Q;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float output_min = 0.0f;
  const float output_max = (255.0f / 256.0f);
  const int output_dims_count = 5;
  uint8_t output_data[output_dims_count];
  tflite::testing::TestSoftmaxQuantized(  //
      {2, 1, 5},                          // Input shape.
      {
          F2Q(1.0, input_min, input_max),
          F2Q(2.0, input_min, input_max),
          F2Q(3.0, input_min, input_max),
          F2Q(4.0, input_min, input_max),
          F2Q(5.0, input_min, input_max),
      },
      input_min, input_max,  // Input quantized range.
      {
          // Expected results.
          F2Q(0.011656231, output_min, output_max),
          F2Q(0.031684921, output_min, output_max),
          F2Q(0.086128544, output_min, output_max),
          F2Q(0.234121657, output_min, output_max),
          F2Q(0.636408647, output_min, output_max),
      },
      {2, 1, 5},               // Output shape.
      output_min, output_max,  // Output quantized range.
      output_data);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedSigned) {
  using tflite::testing::F2QS;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float output_min = 0.0f;
  const float output_max = (255.0f / 256.0f);
  const int output_dims_count = 10;
  int8_t output_data[output_dims_count];
  tflite::testing::TestSoftmaxQuantizedSigned(  //
      {2, 2, 5},                                // Input shape.
      {                                         // b = 0
       F2QS(-3.0, input_min, input_max), F2QS(5.0, input_min, input_max),
       F2QS(-7.0, input_min, input_max), F2QS(9.0, input_min, input_max),
       F2QS(-11.0, input_min, input_max),
       // b = 1
       F2QS(1.0, input_min, input_max), F2QS(2.0, input_min, input_max),
       F2QS(3.0, input_min, input_max), F2QS(4.0, input_min, input_max),
       F2QS(5.0, input_min, input_max)},
      input_min, input_max,  // Input quantized range.
      {
          // Expected results.
          // b = 0
          F2QS(0.000006933, output_min, output_max),
          F2QS(0.017986099, output_min, output_max),
          F2QS(0.000000110, output_min, output_max),
          F2QS(0.982007754, output_min, output_max),
          F2QS(0.000000002, output_min, output_max),
          // b = 1
          F2QS(0.011656231, output_min, output_max),
          F2QS(0.031684921, output_min, output_max),
          F2QS(0.086128544, output_min, output_max),
          F2QS(0.234121657, output_min, output_max),
          F2QS(0.636408647, output_min, output_max),
      },
      {2, 2, 5},               // Output shape.
      output_min, output_max,  // Output quantized range.
      output_data);
}

TF_LITE_MICRO_TESTS_END
