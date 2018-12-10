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
#include "tensorflow/lite/experimental/micro/kernels/test_utils.h"
#include "tensorflow/lite/experimental/micro/simple_tensor_allocator.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

void TestFullyConnectedFloat(std::initializer_list<int> input_dims_data,
                             std::initializer_list<float> input_data,
                             std::initializer_list<int> weights_dims_data,
                             std::initializer_list<float> weights_data,
                             std::initializer_list<int> bias_dims_data,
                             std::initializer_list<float> bias_data,
                             std::initializer_list<float> expected_output_data,
                             std::initializer_list<int> output_dims_data,
                             TfLiteFusedActivation activation,
                             float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInitializer(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInitializer(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateFloatTensor(weights_data, weights_dims, "weights_tensor"),
      CreateFloatTensor(bias_data, bias_dims, "bias_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_FULLY_CONNECTED, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteFullyConnectedParams builtin_data = {
      activation,
      kTfLiteFullyConnectedWeightsFormatDefault,
  };
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

void TestFullyConnectedQuantized(
    std::initializer_list<int> input_dims_data,
    std::initializer_list<uint8_t> input_data, float input_min, float input_max,
    std::initializer_list<int> weights_dims_data,
    std::initializer_list<uint8_t> weights_data, float weights_min,
    float weights_max, std::initializer_list<int> bias_dims_data,
    std::initializer_list<int32_t> bias_data, float bias_min, float bias_max,
    std::initializer_list<uint8_t> expected_output_data,
    std::initializer_list<int> output_dims_data, float output_min,
    float output_max, TfLiteFusedActivation activation, uint8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInitializer(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInitializer(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, "input_tensor", input_min,
                            input_max),
      CreateQuantizedTensor(weights_data, weights_dims, "weights_tensor",
                            weights_min, weights_max),
      CreateQuantized32Tensor(bias_data, bias_dims, "bias_tensor", bias_min,
                              bias_max),
      CreateQuantizedTensor(output_data, output_dims, "output_tensor",
                            output_min, output_max),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_FULLY_CONNECTED, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteFullyConnectedParams builtin_data = {
      activation,
      kTfLiteFullyConnectedWeightsFormatDefault,
  };
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

TF_LITE_MICRO_TEST(SimpleTest) {
  const int output_dims_count = 6;
  float output_data[output_dims_count];
  tflite::testing::TestFullyConnectedFloat(  //
      {2, 2, 10},                            // Input shape.
      {
          1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
          1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
      },
      {2, 3, 10},  // Weights shape.
      {
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
      },
      {1, 3},  // Bias shape.
      {
          1, 2, 3,  // Bias values.
      },
      {
          24, 25, 26, 58, 59, 60,  // Expected results.
      },
      {2, 2, 3},  // Output shape.
      kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTest2) {
  const int output_dims_count = 6;
  float output_data[output_dims_count];
  tflite::testing::TestFullyConnectedFloat(  //
      {2, 2, 2},                             // Input shape.
      {
          1, 2,  // b = 0
          2, 1,  // b = 1
      },
      {2, 1, 2},  // Weights shape.
      {
          2, 4,  // u = 0
      },
      {1, 1},  // Bias shape.
      {
          1,  // Bias values.
      },
      {
          11, 9,  // Expected results.
      },
      {2, 2, 1},  // Output shape.
      kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestRelu) {
  const int output_dims_count = 6;
  float output_data[output_dims_count];
  tflite::testing::TestFullyConnectedFloat(  //
      {2, 2, 10},                            // Input shape.
      {
          1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
          1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
      },
      {2, 3, 10},  // Weights shape.
      {
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   // u = 0
          -1, -2, -3, -4, -5, -6, -7, -8, -9, -10,  // u = 1
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   // u = 2
      },
      {1, 3},  // Bias shape.
      {
          1, -2, 3,  // Bias values.
      },
      {
          24, 0, 26, 58, 0, 60,  // Expected results.
      },
      {2, 2, 3},  // Output shape.
      kTfLiteActRelu, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestQuantized) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -63.5f;
  const float weights_max = 64.0f;
  const float bias_min = 0.0f;
  const float bias_max = 64.0f * (1 << 24);
  const float output_min = -127.0f;
  const float output_max = 128.0f;
  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized(  //
      {2, 2, 10},                                // Input shape.
      {
          // Input values.
          F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
          F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
          F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
          F2Q(7, input_min, input_max),  F2Q(8, input_min, input_max),
          F2Q(-9, input_min, input_max), F2Q(-10, input_min, input_max),
          F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
          F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
          F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
          F2Q(7, input_min, input_max),  F2Q(-8, input_min, input_max),
          F2Q(9, input_min, input_max),  F2Q(-10, input_min, input_max),
      },
      input_min, input_max,  // Input quantization range.
      {2, 3, 10},            // Weights shape.
      {
          // Weight values.
          F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
          F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
          F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
          F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
          F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
          F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
          F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
          F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
          F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
          F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
          F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
          F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
          F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
          F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
          F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
      },
      weights_min, weights_max,  // Weights quantization range.
      {1, 3},                    // Bias shape.
      {
          F2Q32(1, bias_min, bias_max),
          F2Q32(2, bias_min, bias_max),
          F2Q32(3, bias_min, bias_max),
      },
      bias_min, bias_max,  // Bias quantization range.
      {
          // Expected results.
          F2Q(24, output_min, output_max),
          F2Q(25, output_min, output_max),
          F2Q(26, output_min, output_max),
          F2Q(58, output_min, output_max),
          F2Q(59, output_min, output_max),
          F2Q(60, output_min, output_max),
      },
      {2, 2, 3},               // Output shape.
      output_min, output_max,  // Output quantization range.
      kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedRelu) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -63.5f;
  const float weights_max = 64.0f;
  const float bias_min = 0.0f;
  const float bias_max = 64.0f * (1 << 24);
  const float output_min = -127.0f;
  const float output_max = 128.0f;
  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized(  //
      {2, 2, 10},                                // Input shape.
      {
          // Input values.
          F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
          F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
          F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
          F2Q(7, input_min, input_max),  F2Q(8, input_min, input_max),
          F2Q(-9, input_min, input_max), F2Q(-10, input_min, input_max),
          F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
          F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
          F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
          F2Q(7, input_min, input_max),  F2Q(-8, input_min, input_max),
          F2Q(9, input_min, input_max),  F2Q(-10, input_min, input_max),
      },
      input_min, input_max,  // Input quantization range.
      {2, 3, 10},            // Weights shape.
      {
          // Weight values.
          F2Q(1, weights_min, weights_max),  F2Q(2, weights_min, weights_max),
          F2Q(3, weights_min, weights_max),  F2Q(4, weights_min, weights_max),
          F2Q(5, weights_min, weights_max),  F2Q(6, weights_min, weights_max),
          F2Q(7, weights_min, weights_max),  F2Q(8, weights_min, weights_max),
          F2Q(9, weights_min, weights_max),  F2Q(10, weights_min, weights_max),
          F2Q(-1, weights_min, weights_max), F2Q(-2, weights_min, weights_max),
          F2Q(-3, weights_min, weights_max), F2Q(-4, weights_min, weights_max),
          F2Q(-5, weights_min, weights_max), F2Q(-6, weights_min, weights_max),
          F2Q(-7, weights_min, weights_max), F2Q(-8, weights_min, weights_max),
          F2Q(-9, weights_min, weights_max), F2Q(-10, weights_min, weights_max),
          F2Q(1, weights_min, weights_max),  F2Q(2, weights_min, weights_max),
          F2Q(3, weights_min, weights_max),  F2Q(4, weights_min, weights_max),
          F2Q(5, weights_min, weights_max),  F2Q(6, weights_min, weights_max),
          F2Q(7, weights_min, weights_max),  F2Q(8, weights_min, weights_max),
          F2Q(9, weights_min, weights_max),  F2Q(10, weights_min, weights_max),
      },
      weights_min, weights_max,  // Weights quantization range.
      {1, 3},                    // Bias shape.
      {
          F2Q32(1, bias_min, bias_max),
          F2Q32(0, bias_min, bias_max),
          F2Q32(3, bias_min, bias_max),
      },
      bias_min, bias_max,  // Bias quantization range.
      {
          // Expected results.
          F2Q(24, output_min, output_max),
          F2Q(0, output_min, output_max),
          F2Q(26, output_min, output_max),
          F2Q(58, output_min, output_max),
          F2Q(0, output_min, output_max),
          F2Q(60, output_min, output_max),
      },
      {2, 2, 3},               // Output shape.
      output_min, output_max,  // Output quantization range.
      kTfLiteActRelu, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedOutputMultiplierGreaterThan1) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -127.0f;
  const float input_max = 128.0f;
  const float weights_min = -127.0f;
  const float weights_max = 128.0f;
  const float bias_min = 0.0f;
  const float bias_max = 256.0f * (1 << 24);
  const float output_min = -63.5f;
  const float output_max = 64.0f;
  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized(  //
      {2, 2, 10},                                // Input shape.
      {
          // Input values.
          F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
          F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
          F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
          F2Q(7, input_min, input_max),  F2Q(8, input_min, input_max),
          F2Q(-9, input_min, input_max), F2Q(-10, input_min, input_max),
          F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
          F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
          F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
          F2Q(7, input_min, input_max),  F2Q(-8, input_min, input_max),
          F2Q(9, input_min, input_max),  F2Q(-10, input_min, input_max),
      },
      input_min, input_max,  // Input quantization range.
      {2, 3, 10},            // Weights shape.
      {
          // Weight values.
          F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
          F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
          F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
          F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
          F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
          F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
          F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
          F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
          F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
          F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
          F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
          F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
          F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
          F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
          F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
      },
      weights_min, weights_max,  // Weights quantization range.
      {1, 3},                    // Bias shape.
      {
          F2Q32(1, bias_min, bias_max),
          F2Q32(2, bias_min, bias_max),
          F2Q32(3, bias_min, bias_max),
      },
      bias_min, bias_max,  // Bias quantization range.
      {
          // Expected results.
          F2Q(24, output_min, output_max),
          F2Q(25, output_min, output_max),
          F2Q(26, output_min, output_max),
          F2Q(58, output_min, output_max),
          F2Q(59, output_min, output_max),
          F2Q(60, output_min, output_max),
      },
      {2, 2, 3},               // Output shape.
      output_min, output_max,  // Output quantization range.
      kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTest4DInput) {
  const int output_dims_count = 6;
  float output_data[output_dims_count];
  tflite::testing::TestFullyConnectedFloat(  //
      {4, 1, 1, 5, 1},                       // Input shape.
      {
          1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
          1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
      },
      {2, 3, 10},  // Weights shape.
      {
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
      },
      {1, 3},  // Bias shape.
      {
          1, 2, 3,  // Bias values.
      },
      {
          24, 25, 26, 58, 59, 60,  // Expected results.
      },
      {2, 2, 3},  // Output shape.
      kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTest4DInputQuantized) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -63.5f;
  const float weights_max = 64.0f;
  const float bias_min = 0.0f;
  const float bias_max = 64.0f * (1 << 24);
  const float output_min = -127.0f;
  const float output_max = 128.0f;
  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized(  //
      {4, 1, 1, 5, 1},                           // Input shape.
      {
          // Input values.
          F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
          F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
          F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
          F2Q(7, input_min, input_max),  F2Q(8, input_min, input_max),
          F2Q(-9, input_min, input_max), F2Q(-10, input_min, input_max),
          F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
          F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
          F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
          F2Q(7, input_min, input_max),  F2Q(-8, input_min, input_max),
          F2Q(9, input_min, input_max),  F2Q(-10, input_min, input_max),
      },
      input_min, input_max,  // Input quantization range.
      {2, 3, 10},            // Weights shape.
      {
          // Weight values.
          F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
          F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
          F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
          F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
          F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
          F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
          F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
          F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
          F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
          F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
          F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
          F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
          F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
          F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
          F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
      },
      weights_min, weights_max,  // Weights quantization range.
      {1, 3},                    // Bias shape.
      {
          F2Q32(1, bias_min, bias_max),
          F2Q32(2, bias_min, bias_max),
          F2Q32(3, bias_min, bias_max),
      },
      bias_min, bias_max,  // Bias quantization range.
      {
          // Expected results.
          F2Q(24, output_min, output_max),
          F2Q(25, output_min, output_max),
          F2Q(26, output_min, output_max),
          F2Q(58, output_min, output_max),
          F2Q(59, output_min, output_max),
          F2Q(60, output_min, output_max),
      },
      {2, 2, 3},               // Output shape.
      output_min, output_max,  // Output quantization range.
      kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTest4DInputQuantizedOutputMultiplierGreaterThan1) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -127.0f;
  const float input_max = 128.0f;
  const float weights_min = -127.0f;
  const float weights_max = 128.0f;
  const float bias_min = 0.0f;
  const float bias_max = 256.0f * (1 << 24);
  const float output_min = -63.5f;
  const float output_max = 64.0f;
  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized(  //
      {4, 1, 1, 5, 1},                           // Input shape.
      {
          // Input values.
          F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
          F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
          F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
          F2Q(7, input_min, input_max),  F2Q(8, input_min, input_max),
          F2Q(-9, input_min, input_max), F2Q(-10, input_min, input_max),
          F2Q(1, input_min, input_max),  F2Q(2, input_min, input_max),
          F2Q(3, input_min, input_max),  F2Q(4, input_min, input_max),
          F2Q(5, input_min, input_max),  F2Q(6, input_min, input_max),
          F2Q(7, input_min, input_max),  F2Q(-8, input_min, input_max),
          F2Q(9, input_min, input_max),  F2Q(-10, input_min, input_max),
      },
      input_min, input_max,  // Input quantization range.
      {2, 3, 10},            // Weights shape.
      {
          // Weight values.
          F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
          F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
          F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
          F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
          F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
          F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
          F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
          F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
          F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
          F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
          F2Q(1, weights_min, weights_max), F2Q(2, weights_min, weights_max),
          F2Q(3, weights_min, weights_max), F2Q(4, weights_min, weights_max),
          F2Q(5, weights_min, weights_max), F2Q(6, weights_min, weights_max),
          F2Q(7, weights_min, weights_max), F2Q(8, weights_min, weights_max),
          F2Q(9, weights_min, weights_max), F2Q(10, weights_min, weights_max),
      },
      weights_min, weights_max,  // Weights quantization range.
      {1, 3},                    // Bias shape.
      {
          F2Q32(1, bias_min, bias_max),
          F2Q32(2, bias_min, bias_max),
          F2Q32(3, bias_min, bias_max),
      },
      bias_min, bias_max,  // Bias quantization range.
      {
          // Expected results.
          F2Q(24, output_min, output_max),
          F2Q(25, output_min, output_max),
          F2Q(26, output_min, output_max),
          F2Q(58, output_min, output_max),
          F2Q(59, output_min, output_max),
          F2Q(60, output_min, output_max),
      },
      {2, 2, 3},               // Output shape.
      output_min, output_max,  // Output quantization range.
      kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TESTS_END
