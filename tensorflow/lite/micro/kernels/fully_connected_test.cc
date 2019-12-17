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

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void TestFullyConnectedFloat(
    const int* input_dims_data, const float* input_data,
    const int* weights_dims_data, const float* weights_data,
    const int* bias_dims_data, const float* bias_data,
    const float* expected_output_data, const int* output_dims_data,
    TfLiteFusedActivation activation, float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInts(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
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
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i], 1e-5f);
  }
}

template <typename T>
void TestFullyConnectedQuantized(
    const int* input_dims_data, const T* input_data, const float input_min,
    const float input_max, const int* weights_dims_data, const T* weights_data,
    const float weights_min, const float weights_max, const int* bias_dims_data,
    const int32_t* bias_data, const float bias_scale,
    const T* expected_output_data, const int* output_dims_data,
    const float output_min, const float output_max,
    TfLiteFusedActivation activation, T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* weights_dims = IntArrayFromInts(weights_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, "input_tensor", input_min,
                            input_max),
      CreateQuantizedTensor(weights_data, weights_dims, "weights_tensor",
                            weights_min, weights_max),
      CreateQuantized32Tensor(bias_data, bias_dims, "bias_tensor", bias_scale),
      CreateQuantizedTensor(output_data, output_dims, "output_tensor",
                            output_min, output_max),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_FULLY_CONNECTED, 4);
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
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleTest) {
  const int input_dims_data[] = {2, 2, 10};
  const float input_data[] = {
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  };
  const int weights_dims_data[] = {2, 3, 10};
  const float weights_data[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  };
  const int bias_dims_data[] = {1, 3};
  const float bias_data[] = {1, 2, 3};
  const float expected_output_data[] = {
      24, 25, 26, 58, 59, 60,
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  float output_data[output_dims_count];
  tflite::testing::TestFullyConnectedFloat(
      input_dims_data, input_data, weights_dims_data, weights_data,
      bias_dims_data, bias_data, expected_output_data, output_dims_data,
      kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTest2) {
  const int input_dims_data[] = {2, 2, 2};
  const float input_data[] = {
      1, 2,  // b = 0
      2, 1,  // b = 1
  };
  const int weights_dims_data[] = {2, 1, 2};
  const float weights_data[] = {
      2, 4,  // u = 0
  };
  const int bias_dims_data[] = {1, 1};
  const float bias_data[] = {1};
  const float expected_output_data[] = {
      11,
      9,
  };
  const int output_dims_data[] = {2, 2, 1};

  const int output_dims_count = 6;
  float output_data[output_dims_count];
  tflite::testing::TestFullyConnectedFloat(
      input_dims_data, input_data, weights_dims_data, weights_data,
      bias_dims_data, bias_data, expected_output_data, output_dims_data,
      kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestRelu) {
  const int input_dims_data[] = {2, 2, 10};
  const float input_data[] = {
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  };
  const int weights_dims_data[] = {2, 3, 10};
  const float weights_data[] = {
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   // u = 0
      -1, -2, -3, -4, -5, -6, -7, -8, -9, -10,  // u = 1
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   // u = 2
  };
  const int bias_dims_data[] = {1, 3};
  const float bias_data[] = {1, -2, 3};
  const float expected_output_data[] = {
      24, 0, 26, 58, 0, 60,
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  float output_data[output_dims_count];
  tflite::testing::TestFullyConnectedFloat(
      input_dims_data, input_data, weights_dims_data, weights_data,
      bias_dims_data, bias_data, expected_output_data, output_dims_data,
      kTfLiteActRelu, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedUInt8) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -63.5f;
  const float weights_max = 64.0f;
  const float bias_scale = 0.25f;
  const float output_min = -127.0f;
  const float output_max = 128.0f;

  const int input_dims_data[] = {2, 2, 10};
  const uint8_t input_data[] = {
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
  };
  const int weights_dims_data[] = {2, 3, 10};
  const uint8_t weights_data[] = {
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
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const uint8_t expected_output_data[] = {
      F2Q(24, output_min, output_max), F2Q(25, output_min, output_max),
      F2Q(26, output_min, output_max), F2Q(58, output_min, output_max),
      F2Q(59, output_min, output_max), F2Q(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized<uint8_t>(
      input_dims_data, input_data, input_min, input_max, weights_dims_data,
      weights_data, weights_min, weights_max, bias_dims_data, bias_data,
      bias_scale, expected_output_data, output_dims_data, output_min,
      output_max, kTfLiteActNone, output_data);
}

// TODO(b/138811455): Fix code duplication in micro tests
TF_LITE_MICRO_TEST(SimpleTestQuantizedInt8) {
  using tflite::testing::F2Q32;
  using tflite::testing::F2QS;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -63.5f;
  const float weights_max = 64.0f;
  const float bias_scale = 0.25f;
  const float output_min = -127.0f;
  const float output_max = 128.0f;

  const int input_dims_data[] = {2, 2, 10};
  const int8_t input_data[] = {
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(8, input_min, input_max),
      F2QS(-9, input_min, input_max), F2QS(-10, input_min, input_max),
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(-8, input_min, input_max),
      F2QS(9, input_min, input_max),  F2QS(-10, input_min, input_max),
  };
  const int weights_dims_data[] = {2, 3, 10};
  const int8_t weights_data[] = {
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const int8_t expected_output_data[] = {
      F2QS(24, output_min, output_max), F2QS(25, output_min, output_max),
      F2QS(26, output_min, output_max), F2QS(58, output_min, output_max),
      F2QS(59, output_min, output_max), F2QS(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  int8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized<int8_t>(
      input_dims_data, input_data, input_min, input_max, weights_dims_data,
      weights_data, weights_min, weights_max, bias_dims_data, bias_data,
      bias_scale, expected_output_data, output_dims_data, output_min,
      output_max, kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedUInt8Relu) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -63.5f;
  const float weights_max = 64.0f;
  const float bias_scale = 0.25f;
  const float output_min = -127.0f;
  const float output_max = 128.0f;

  const int input_dims_data[] = {2, 2, 10};
  const uint8_t input_data[] = {
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
  };
  const int weights_dims_data[] = {2, 3, 10};
  const uint8_t weights_data[] = {
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
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(0, bias_scale),
      F2Q32(3, bias_scale),
  };
  const uint8_t expected_output_data[] = {
      F2Q(24, output_min, output_max), F2Q(0, output_min, output_max),
      F2Q(26, output_min, output_max), F2Q(58, output_min, output_max),
      F2Q(0, output_min, output_max),  F2Q(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized<uint8_t>(
      input_dims_data, input_data, input_min, input_max, weights_dims_data,
      weights_data, weights_min, weights_max, bias_dims_data, bias_data,
      bias_scale, expected_output_data, output_dims_data, output_min,
      output_max, kTfLiteActRelu, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedInt8Relu) {
  using tflite::testing::F2Q32;
  using tflite::testing::F2QS;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -63.5f;
  const float weights_max = 64.0f;
  const float bias_scale = 0.25f;
  const float output_min = -127.0f;
  const float output_max = 128.0f;

  const int input_dims_data[] = {2, 2, 10};
  const int8_t input_data[] = {
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(8, input_min, input_max),
      F2QS(-9, input_min, input_max), F2QS(-10, input_min, input_max),
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(-8, input_min, input_max),
      F2QS(9, input_min, input_max),  F2QS(-10, input_min, input_max),
  };
  const int weights_dims_data[] = {2, 3, 10};
  const int8_t weights_data[] = {
      F2QS(1, weights_min, weights_max),  F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max),  F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max),  F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max),  F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max),  F2QS(10, weights_min, weights_max),
      F2QS(-1, weights_min, weights_max), F2QS(-2, weights_min, weights_max),
      F2QS(-3, weights_min, weights_max), F2QS(-4, weights_min, weights_max),
      F2QS(-5, weights_min, weights_max), F2QS(-6, weights_min, weights_max),
      F2QS(-7, weights_min, weights_max), F2QS(-8, weights_min, weights_max),
      F2QS(-9, weights_min, weights_max), F2QS(-10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max),  F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max),  F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max),  F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max),  F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max),  F2QS(10, weights_min, weights_max),
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(0, bias_scale),
      F2Q32(3, bias_scale),
  };
  const int8_t expected_output_data[] = {
      F2QS(24, output_min, output_max), F2QS(0, output_min, output_max),
      F2QS(26, output_min, output_max), F2QS(58, output_min, output_max),
      F2QS(0, output_min, output_max),  F2QS(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  int8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized<int8_t>(
      input_dims_data, input_data, input_min, input_max, weights_dims_data,
      weights_data, weights_min, weights_max, bias_dims_data, bias_data,
      bias_scale, expected_output_data, output_dims_data, output_min,
      output_max, kTfLiteActRelu, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedUInt8OutputMultiplierGreaterThan1) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -127.0f;
  const float input_max = 128.0f;
  const float weights_min = -127.0f;
  const float weights_max = 128.0f;
  const float bias_scale = 1.0f;
  const float output_min = -63.5f;
  const float output_max = 64.0f;

  const int input_dims_data[] = {2, 2, 10};
  const uint8_t input_data[] = {
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
  };
  const int weights_dims_data[] = {2, 3, 10};
  const uint8_t weights_data[] = {
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
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const uint8_t expected_output_data[] = {
      F2Q(24, output_min, output_max), F2Q(25, output_min, output_max),
      F2Q(26, output_min, output_max), F2Q(58, output_min, output_max),
      F2Q(59, output_min, output_max), F2Q(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized<uint8_t>(
      input_dims_data, input_data, input_min, input_max, weights_dims_data,
      weights_data, weights_min, weights_max, bias_dims_data, bias_data,
      bias_scale, expected_output_data, output_dims_data, output_min,
      output_max, kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedInt8OutputMultiplierGreaterThan1) {
  using tflite::testing::F2Q32;
  using tflite::testing::F2QS;

  const float input_min = -127.0f;
  const float input_max = 128.0f;
  const float weights_min = -127.0f;
  const float weights_max = 128.0f;
  const float bias_scale = 1.0f;
  const float output_min = -63.5f;
  const float output_max = 64.0f;

  const int input_dims_data[] = {2, 2, 10};
  const int8_t input_data[] = {
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(8, input_min, input_max),
      F2QS(-9, input_min, input_max), F2QS(-10, input_min, input_max),
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(-8, input_min, input_max),
      F2QS(9, input_min, input_max),  F2QS(-10, input_min, input_max),
  };
  const int weights_dims_data[] = {2, 3, 10};
  const int8_t weights_data[] = {
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const int8_t expected_output_data[] = {
      F2QS(24, output_min, output_max), F2QS(25, output_min, output_max),
      F2QS(26, output_min, output_max), F2QS(58, output_min, output_max),
      F2QS(59, output_min, output_max), F2QS(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  int8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized<int8_t>(
      input_dims_data, input_data, input_min, input_max, weights_dims_data,
      weights_data, weights_min, weights_max, bias_dims_data, bias_data,
      bias_scale, expected_output_data, output_dims_data, output_min,
      output_max, kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTest4DInput) {
  const int input_dims_data[] = {4, 1, 1, 5, 1};
  const float input_data[] = {
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  };
  const int weights_dims_data[] = {2, 3, 10};
  const float weights_data[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  };
  const int bias_dims_data[] = {1, 3};
  const float bias_data[] = {1, 2, 3};
  const float expected_output_data[] = {
      24, 25, 26, 58, 59, 60,  // Expected results.
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  float output_data[output_dims_count];
  tflite::testing::TestFullyConnectedFloat(
      input_dims_data, input_data, weights_dims_data, weights_data,
      bias_dims_data, bias_data, expected_output_data, output_dims_data,
      kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTest4DInputQuantizedUInt8) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -63.5f;
  const float weights_max = 64.0f;
  const float bias_scale = 0.25f;
  const float output_min = -127.0f;
  const float output_max = 128.0f;

  const int input_dims_data[] = {4, 1, 1, 5, 1};
  const uint8_t input_data[] = {
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
  };
  const int weights_dims_data[] = {2, 3, 10};
  const uint8_t weights_data[] = {
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
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const uint8_t expected_output_data[] = {
      F2Q(24, output_min, output_max), F2Q(25, output_min, output_max),
      F2Q(26, output_min, output_max), F2Q(58, output_min, output_max),
      F2Q(59, output_min, output_max), F2Q(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized<uint8_t>(
      input_dims_data, input_data, input_min, input_max, weights_dims_data,
      weights_data, weights_min, weights_max, bias_dims_data, bias_data,
      bias_scale, expected_output_data, output_dims_data, output_min,
      output_max, kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTest4DInputQuantizedInt8) {
  using tflite::testing::F2Q32;
  using tflite::testing::F2QS;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float weights_min = -63.5f;
  const float weights_max = 64.0f;
  const float bias_scale = 0.25f;
  const float output_min = -127.0f;
  const float output_max = 128.0f;

  const int input_dims_data[] = {4, 1, 1, 5, 1};
  const int8_t input_data[] = {
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(8, input_min, input_max),
      F2QS(-9, input_min, input_max), F2QS(-10, input_min, input_max),
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(-8, input_min, input_max),
      F2QS(9, input_min, input_max),  F2QS(-10, input_min, input_max),
  };
  const int weights_dims_data[] = {2, 3, 10};
  const int8_t weights_data[] = {
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const int8_t expected_output_data[] = {
      F2QS(24, output_min, output_max), F2QS(25, output_min, output_max),
      F2QS(26, output_min, output_max), F2QS(58, output_min, output_max),
      F2QS(59, output_min, output_max), F2QS(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  int8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized<int8_t>(
      input_dims_data, input_data, input_min, input_max, weights_dims_data,
      weights_data, weights_min, weights_max, bias_dims_data, bias_data,
      bias_scale, expected_output_data, output_dims_data, output_min,
      output_max, kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(
    SimpleTest4DInputQuantizedUInt8OutputMultiplierGreaterThan1) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -127.0f;
  const float input_max = 128.0f;
  const float weights_min = -127.0f;
  const float weights_max = 128.0f;
  const float bias_scale = 1.0f;
  const float output_min = -63.5f;
  const float output_max = 64.0f;

  const int input_dims_data[] = {4, 1, 1, 5, 1};
  const uint8_t input_data[] = {
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
  };
  const int weights_dims_data[] = {2, 3, 10};
  const uint8_t weights_data[] = {
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
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const uint8_t expected_output_data[] = {
      F2Q(24, output_min, output_max), F2Q(25, output_min, output_max),
      F2Q(26, output_min, output_max), F2Q(58, output_min, output_max),
      F2Q(59, output_min, output_max), F2Q(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  uint8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized<uint8_t>(
      input_dims_data, input_data, input_min, input_max, weights_dims_data,
      weights_data, weights_min, weights_max, bias_dims_data, bias_data,
      bias_scale, expected_output_data, output_dims_data, output_min,
      output_max, kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTest4DInputQuantizedInt8OutputMultiplierGreaterThan1) {
  using tflite::testing::F2Q32;
  using tflite::testing::F2QS;

  const float input_min = -127.0f;
  const float input_max = 128.0f;
  const float weights_min = -127.0f;
  const float weights_max = 128.0f;
  const float bias_scale = 1.0f;
  const float output_min = -63.5f;
  const float output_max = 64.0f;

  const int input_dims_data[] = {4, 1, 1, 5, 1};
  const int8_t input_data[] = {
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(8, input_min, input_max),
      F2QS(-9, input_min, input_max), F2QS(-10, input_min, input_max),
      F2QS(1, input_min, input_max),  F2QS(2, input_min, input_max),
      F2QS(3, input_min, input_max),  F2QS(4, input_min, input_max),
      F2QS(5, input_min, input_max),  F2QS(6, input_min, input_max),
      F2QS(7, input_min, input_max),  F2QS(-8, input_min, input_max),
      F2QS(9, input_min, input_max),  F2QS(-10, input_min, input_max),
  };
  const int weights_dims_data[] = {2, 3, 10};
  const int8_t weights_data[] = {
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
      F2QS(1, weights_min, weights_max), F2QS(2, weights_min, weights_max),
      F2QS(3, weights_min, weights_max), F2QS(4, weights_min, weights_max),
      F2QS(5, weights_min, weights_max), F2QS(6, weights_min, weights_max),
      F2QS(7, weights_min, weights_max), F2QS(8, weights_min, weights_max),
      F2QS(9, weights_min, weights_max), F2QS(10, weights_min, weights_max),
  };
  const int bias_dims_data[] = {1, 3};
  const int32_t bias_data[] = {
      F2Q32(1, bias_scale),
      F2Q32(2, bias_scale),
      F2Q32(3, bias_scale),
  };
  const int8_t expected_output_data[] = {
      F2QS(24, output_min, output_max), F2QS(25, output_min, output_max),
      F2QS(26, output_min, output_max), F2QS(58, output_min, output_max),
      F2QS(59, output_min, output_max), F2QS(60, output_min, output_max),
  };
  const int output_dims_data[] = {2, 2, 3};

  const int output_dims_count = 6;
  int8_t output_data[output_dims_count];
  tflite::testing::TestFullyConnectedQuantized<int8_t>(
      input_dims_data, input_data, input_min, input_max, weights_dims_data,
      weights_data, weights_min, weights_max, bias_dims_data, bias_data,
      bias_scale, expected_output_data, output_dims_data, output_min,
      output_max, kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TESTS_END
