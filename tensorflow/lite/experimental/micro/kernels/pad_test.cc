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
#include "tensorflow/lite/experimental/micro/simple_tensor_allocator.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void TestPadFloat(BuiltinOperator op,
                  std::initializer_list<int> input_dims_data,
                  std::initializer_list<float> input_data,
                  std::initializer_list<int> padding_dims_data,
                  std::initializer_list<int32_t> padding_data,
                  std::initializer_list<int> const_dims_data,
                  std::initializer_list<float> const_data,
                  std::initializer_list<float> expected_output_data,
                  std::initializer_list<int> output_dims_data,
                  float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* padding_dims = IntArrayFromInitializer(padding_dims_data);
  TfLiteIntArray* const_dims = IntArrayFromInitializer(const_dims_data);

  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateIntTensor(padding_data, padding_dims, "padding_tensor"),
      CreateFloatTensor(const_data, const_dims, "const_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration = nullptr;
  if (op == tflite::BuiltinOperator_PAD) {
    registration = resolver.FindOp(tflite::BuiltinOperator_PAD, 1);
  } else if (op == tflite::BuiltinOperator_PADV2) {
    registration = resolver.FindOp(tflite::BuiltinOperator_PADV2, 1);
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteSoftmaxParams builtin_data = {1.0f};
  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
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
                          std::initializer_list<int> padding_dims_data,
                          std::initializer_list<int32_t> padding_data,
                          std::initializer_list<int> const_dims_data,
                          std::initializer_list<uint8_t> const_data,
                          std::initializer_list<uint8_t> expected_output_data,
                          std::initializer_list<int> output_dims_data,
                          float output_min, float output_max,
                          uint8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* padding_dims = IntArrayFromInitializer(padding_dims_data);
  TfLiteIntArray* const_dims = IntArrayFromInitializer(const_dims_data);

  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, "input_tensor", input_min,
                            input_max),
      CreateIntTensor(padding_data, padding_dims, "padding_tensor"),
      CreateQuantizedTensor(const_data, const_dims, "const_tensor", input_min,
                            input_max),
      CreateQuantizedTensor(output_data, output_dims, "output_tensor",
                            output_min, output_max),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_PAD, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteSoftmaxParams builtin_data = {1.0f};
  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
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

TF_LITE_MICRO_TEST(SimplePADTest) {
  const int output_dims_count = 10;
  float output_data[output_dims_count];
  tflite::testing::TestPadFloat(                     //
      tflite::BuiltinOperator_PAD, {4, 1, 2, 2, 1},  // Input shape.
      {
          1.0, 2.0, 3.0, 4.0,  // b = 0
      },
      {2, 4, 2},                         // Pad shape.
      {1, 1, 0, 0, 1, 1, 0, 0}, {1, 1},  // const shape.
      {
          0.0,
      },
      {
          // Expected results.
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0,
          0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      },
      {4, 3, 2, 4, 1},  // Output shape.
      output_data);
}

TF_LITE_MICRO_TEST(SimplePADV2Test) {
  const int output_dims_count = 10;
  float output_data[output_dims_count];
  tflite::testing::TestPadFloat(                       //
      tflite::BuiltinOperator_PADV2, {4, 1, 2, 2, 1},  // Input shape.
      {
          1.0, 2.0, 3.0, 4.0,  // b = 0
      },
      {2, 4, 2},                         // Pad shape.
      {0, 0, 1, 1, 1, 1, 0, 0}, {1, 1},  // const shape.
      {
          5.0,
      },
      {
          // Expected results.
          5,
          5,
          5,
          5,
          5,
          1,
          2,
          5,
          5,
          3,
          4,
          5,
          5,
          5,
          5,
          5,
      },
      {4, 1, 4, 4, 1},  // Output shape.
      output_data);
}

TF_LITE_MICRO_TEST(SimpleTestQuantized) {
  using tflite::testing::F2Q;
  const float input_min = -1.0f;
  const float input_max = 1.0f;
  const float output_min = -1.0f;
  const float output_max = 1.0f;
  const int output_dims_count = 5;
  uint8_t output_data[output_dims_count];
  tflite::testing::TestSoftmaxQuantized(  //
      {4, 1, 2, 2, 1},                    // Input shape.
      {
          F2Q(-0.8, input_min, input_max),
          F2Q(0.2, input_min, input_max),
          F2Q(0.9, input_min, input_max),
          F2Q(0.7, input_min, input_max),
      },
      input_min, input_max,              // Input quantized range.
      {2, 4, 2},                         // Pad shape.
      {0, 0, 1, 1, 1, 1, 0, 0}, {1, 1},  // const shape.
      {F2Q(0.0, input_min, input_max)},
      {
          // Expected results.
          127,
          127,
          127,
          127,
          127,
          25,
          153,
          127,
          127,
          242,
          216,
          127,
          127,
          127,
          127,
          127,
      },
      {4, 1, 4, 4, 1},         // Output shape.
      output_min, output_max,  // Output quantized range.
      output_data);
}
TF_LITE_MICRO_TESTS_END
