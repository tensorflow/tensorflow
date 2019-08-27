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

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void TestAddFloat(const int* input1_dims_data, const float* input1_data,
                  const int* input2_dims_data, const float* input2_data,
                  const int* output_dims_data, const float* expected_output,
                  TfLiteFusedActivation activation, float* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input1_data, input1_dims, "input1_tensor"),
      CreateFloatTensor(input2_data, input2_dims, "input2_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(::tflite::BuiltinOperator_ADD, 1);

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteAddParams builtin_data;
  builtin_data.activation = activation;

  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  const size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
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

  const int output_dims_count = ElementCount(*output_dims);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output[i], output_data[i], 1e-5f);
  }
}

void TestAddFloat(std::initializer_list<int> input1_dims_data,
                  std::initializer_list<float> input1_data,
                  std::initializer_list<int> input2_dims_data,
                  std::initializer_list<float> input2_data,
                  std::initializer_list<int> output_dims_data,
                  std::initializer_list<float> expected_output,
                  TfLiteFusedActivation activation, float* output_data) {
  TestAddFloat(input1_dims_data.begin(), input1_data.begin(),
               input2_dims_data.begin(), input2_data.begin(),
               output_dims_data.begin(), expected_output.begin(), activation,
               output_data);
}

template <typename integer_dtype>
void TestAddQuantized(const int* input1_dims_data,
                      const integer_dtype* input1_data, float input1_min,
                      float input1_max, const int* input2_dims_data,
                      const integer_dtype* input2_data, float input2_min,
                      float input2_max, const int* output_dims_data,
                      const integer_dtype* expected_output, float output_min,
                      float output_max, TfLiteFusedActivation activation,
                      integer_dtype* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

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
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(::tflite::BuiltinOperator_ADD, 1);

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteAddParams builtin_data;
  builtin_data.activation = activation;

  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  const size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
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

  const int output_dims_count = ElementCount(*output_dims);
  for (int i = 0; i < output_dims_count; ++i) {
    // For quantized Add, the maximum error should be one step.
    TF_LITE_MICRO_EXPECT_NEAR(expected_output[i], output_data[i], 1);
  }
}

template <typename integer_dtype>
void TestAddQuantized(std::initializer_list<int> input1_dims_data,
                      std::initializer_list<integer_dtype> input1_data,
                      float input1_min, float input1_max,
                      std::initializer_list<int> input2_dims_data,
                      std::initializer_list<integer_dtype> input2_data,
                      float input2_min, float input2_max,
                      std::initializer_list<int> output_dims_data,
                      std::initializer_list<integer_dtype> expected_output,
                      float output_min, float output_max,
                      TfLiteFusedActivation activation,
                      integer_dtype* output_data) {
  TestAddQuantized<integer_dtype>(
      input1_dims_data.begin(), input1_data.begin(), input1_min, input1_max,
      input2_dims_data.begin(), input2_data.begin(), input2_min, input2_max,
      output_dims_data.begin(), expected_output.begin(), output_min, output_max,
      activation, output_data);
}

// Quantization helpers.
template <typename integer_dtype>
integer_dtype Quantize(const float value, const float min, const float max);

template <>
uint8_t Quantize<uint8_t>(const float value, const float min, const float max) {
  return tflite::testing::F2Q(value, min, max);
}

template <>
int8_t Quantize<int8_t>(const float value, const float min, const float max) {
  return tflite::testing::F2QS(value, min, max);
}

// Quantized tests are defined here for templatizing.
template <typename integer_dtype>
void QuantizedAddNoActivation() {
  const int output_dims_count = 4;
  integer_dtype output_data[output_dims_count];

  const float kMin = -1.0;
  const float kMax = 1.0;

#define Q(x) Quantize<integer_dtype>((x), kMin, kMax)
  const int input_shape[] = {4, 1, 2, 2, 1};
  constexpr int num_test_cases = 3;
  constexpr int num_values = 4;
  const integer_dtype input1_values[num_test_cases][num_values] = {
      {Q(0.1), Q(0.2), Q(0.3), Q(0.4)},
      {Q(-0.8), Q(0.2), Q(0.4), Q(0.7)},
      {Q(-0.8), Q(0.2), Q(0.7), Q(0.3)},
  };
  const integer_dtype input2_values[num_test_cases][num_values] = {
      {Q(0.6), Q(0.4), Q(0.3), Q(0.1)},
      {Q(0.6), Q(0.4), Q(0.5), Q(-0.8)},
      {Q(0.6), Q(0.4), Q(-0.8), Q(0.5)},
  };
  const integer_dtype expected_output[num_test_cases][num_values] = {
      {Q(0.7), Q(0.6), Q(0.6), Q(0.5)},
      {Q(-0.2), Q(0.6), Q(0.9), Q(-0.1)},
      {Q(-0.2), Q(0.6), Q(-0.1), Q(0.8)},
  };
#undef Q

  for (int i = 0; i < num_test_cases; ++i) {
    TestAddQuantized<integer_dtype>(
        input_shape, input1_values[i], kMin, kMax,    // Input 1
        input_shape, input2_values[i], kMin, kMax,    // Input 2
        input_shape, expected_output[i], kMin, kMax,  // Output
        kTfLiteActNone, output_data);
  }
}

template <typename integer_dtype>
void QuantizedAddActivationRelu1() {
  const int output_dims_count = 4;
  integer_dtype output_data[output_dims_count];

  const float kMin = -1.0;
  const float kMax = 1.0;

#define Q(x) Quantize<integer_dtype>((x), kMin, kMax)
  const int input_shape[] = {4, 1, 2, 2, 1};
  constexpr int num_test_cases = 3;
  constexpr int num_values = 4;
  const integer_dtype input1_values[num_test_cases][num_values] = {
      {Q(-0.8), Q(0.2), Q(0.9), Q(0.7)},
      {Q(-0.8), Q(-0.9), Q(0.7), Q(0.3)},
  };
  const integer_dtype input2_values[num_test_cases][num_values] = {
      {Q(0.6), Q(0.4), Q(0.9), Q(-0.8)},
      {Q(0.6), Q(-0.7), Q(-0.8), Q(0.5)},
  };
  const integer_dtype expected_output[num_test_cases][num_values] = {
      {Q(-0.2), Q(0.6), Q(1.0), Q(-0.1)},
      {Q(-0.2), Q(-1.0), Q(-0.1), Q(0.8)},
  };
#undef Q

  for (int i = 0; i < num_test_cases; ++i) {
    TestAddQuantized<integer_dtype>(
        input_shape, input1_values[i], kMin, kMax,    // Input 1
        input_shape, input2_values[i], kMin, kMax,    // Input 2
        input_shape, expected_output[i], kMin, kMax,  // Output
        kTfLiteActRelu1, output_data);
  }
}

template <typename integer_dtype>
void QuantizedAddVariousInputShapes() {
  const int output_dims_count = 6;
  integer_dtype output_data[output_dims_count];

  const float kMin = -3.0;
  const float kMax = 3.0;

#define Q(x) Quantize<integer_dtype>((x), kMin, kMax)
  const integer_dtype input1_values[] = {Q(-2.0), Q(0.2), Q(0.7),
                                         Q(0.8),  Q(1.1), Q(2.0)};
  const integer_dtype input2_values[] = {Q(0.1), Q(0.3), Q(0.3),
                                         Q(0.5), Q(1.1), Q(0.1)};
  const integer_dtype expected_output[] = {Q(-1.9), Q(0.5), Q(1.0),
                                           Q(1.3),  Q(2.2), Q(2.1)};
#undef Q

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  const int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  for (int i = 0; i < num_shapes; ++i) {
    TestAddQuantized<integer_dtype>(
        test_shapes[i], input1_values, kMin, kMax,    // Input 1
        test_shapes[i], input2_values, kMin, kMax,    // Input 2
        test_shapes[i], expected_output, kMin, kMax,  // Output
        kTfLiteActNone, output_data);
  }
}

template <typename integer_dtype>
void QuantizedAddWithScalarBroadcast() {
  const int output_dims_count = 6;
  integer_dtype output_data[output_dims_count];

  const float kMin = -3.0;
  const float kMax = 3.0;

#define Q(x) Quantize<integer_dtype>((x), kMin, kMax)
  const integer_dtype input1_values[] = {Q(-2.0), Q(0.2), Q(0.7),
                                         Q(0.8),  Q(1.1), Q(2.0)};
  const int input2_shape[] = {0};
  const integer_dtype input2_values[] = {Q(0.1)};
  const integer_dtype expected_output[] = {Q(-1.9), Q(0.3), Q(0.8),
                                           Q(0.9),  Q(1.2), Q(2.1)};
#undef Q

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  const int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  for (int i = 0; i < num_shapes; ++i) {
    tflite::testing::TestAddQuantized(
        test_shapes[i], input1_values, kMin, kMax,    // Input 1
        input2_shape, input2_values, kMin, kMax,      // Input 2
        test_shapes[i], expected_output, kMin, kMax,  // Output
        kTfLiteActNone, output_data);
  }
}

template <typename integer_dtype>
void QuantizedAddWithMixedBroadcast() {
  const int output_dims_count = 36;
  integer_dtype output_data[output_dims_count];

  const float kMin = -3.0;
  const float kMax = 3.0;

  constexpr int num_shapes = 4;

#define Q(x) Quantize<integer_dtype>((x), kMin, kMax)
  const int input1_shape[] = {4, 2, 3, 1, 2};
  const integer_dtype input1_values[] = {Q(-0.3), Q(2.3),  Q(0.9), Q(0.5),
                                         Q(0.8),  Q(-1.1), Q(1.2), Q(2.8),
                                         Q(-1.6), Q(0.0),  Q(0.7), Q(-2.2)};
  const integer_dtype input2_values[] = {Q(0.2), Q(0.3), Q(-0.4),
                                         Q(0.5), Q(1.0), Q(0.9)};
  const integer_dtype expected_outputs[num_shapes][output_dims_count] = {
      {Q(-0.1), Q(2.6),  Q(-0.7), Q(2.8), Q(0.7),  Q(3.0),  Q(1.1), Q(0.8),
       Q(0.5),  Q(1.0),  Q(1.9),  Q(1.4), Q(1.0),  Q(-0.8), Q(0.4), Q(-0.6),
       Q(1.8),  Q(-0.2), Q(1.4),  Q(3.0), Q(0.8),  Q(3.0),  Q(2.2), Q(3.0),
       Q(-1.4), Q(0.3),  Q(-2.0), Q(0.5), Q(-0.6), Q(0.9),  Q(0.9), Q(-1.9),
       Q(0.3),  Q(-1.7), Q(1.7),  Q(-1.3)},
      {Q(-0.1), Q(2.6), Q(0.5), Q(1.0), Q(1.8), Q(-0.2), Q(1.4), Q(3.0),
       Q(-2.0), Q(0.5), Q(1.7), Q(-1.3)},
      {Q(-0.1), Q(2.5),  Q(0.0),  Q(2.6), Q(-0.7), Q(1.9),  Q(1.1), Q(0.7),
       Q(1.2),  Q(0.8),  Q(0.5),  Q(0.1), Q(1.0),  Q(-0.9), Q(1.1), Q(-0.8),
       Q(0.4),  Q(-1.5), Q(1.7),  Q(3.0), Q(2.2),  Q(3.0),  Q(2.1), Q(3.0),
       Q(-1.1), Q(0.5),  Q(-0.6), Q(1.0), Q(-0.7), Q(0.9),  Q(1.2), Q(-1.7),
       Q(1.7),  Q(-1.2), Q(1.6),  Q(-1.3)},
      {Q(-0.1), Q(2.5), Q(1.2), Q(0.8), Q(0.4), Q(-1.5), Q(1.7), Q(3.0),
       Q(-0.6), Q(1.0), Q(1.6), Q(-1.3)},
  };
#undef Q

  constexpr int max_shape_size = 5;
  const int input2_shapes[num_shapes][max_shape_size] = {
      {4, 1, 1, 3, 2},
      {4, 1, 3, 1, 2},
      {4, 2, 1, 3, 1},
      {4, 2, 3, 1, 1},
  };
  const int output_shapes[num_shapes][max_shape_size] = {
      {4, 2, 3, 3, 2},
      {4, 2, 3, 1, 2},
      {4, 2, 3, 3, 2},
      {4, 2, 3, 1, 2},
  };

  for (int i = 0; i < num_shapes; ++i) {
    tflite::testing::TestAddQuantized(
        input1_shape, input1_values, kMin, kMax,            // Input 1
        input2_shapes[i], input2_values, kMin, kMax,        // Input 2
        output_shapes[i], expected_outputs[i], kMin, kMax,  // Output
        kTfLiteActNone, output_data);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatAddNoActivation) {
  const int output_dims_count = 4;
  float output_data[output_dims_count];
  tflite::testing::TestAddFloat(
      {4, 1, 2, 2, 1}, {-2.0, 0.2, 0.7, 0.8},  // Input1
      {4, 1, 2, 2, 1}, {0.1, 0.2, 0.3, 0.5},   // Input2
      {4, 1, 2, 2, 1}, {-1.9, 0.4, 1.0, 1.3},  // Expected output
      kTfLiteActNone,                          // No activation
      output_data);                            // Output buffer
}

TF_LITE_MICRO_TEST(FloatAddActivationRelu1) {
  const int output_dims_count = 4;
  float output_data[output_dims_count];
  tflite::testing::TestAddFloat(
      {4, 1, 2, 2, 1}, {-2.0, 0.2, 0.7, 0.8},  // Input1
      {4, 1, 2, 2, 1}, {0.1, 0.2, 0.3, 0.5},   // Input2
      {4, 1, 2, 2, 1}, {-1.0, 0.4, 1.0, 1.0},  // Expected output
      kTfLiteActRelu1,                         // RELU -1 to 1 activation
      output_data);                            // Output buffer
}

TF_LITE_MICRO_TEST(FloatAddVariousInputShapes) {
  const int output_dims_count = 6;
  float output_data[output_dims_count];

  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
  const float input2_values[] = {0.1, 0.2, 0.3, 0.5, 1.1, 0.1};
  const float expected_output[] = {-1.9, 0.4, 1.0, 1.3, 2.2, 2.1};

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  const int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  for (int i = 0; i < num_shapes; ++i) {
    tflite::testing::TestAddFloat(test_shapes[i], input1_values, test_shapes[i],
                                  input2_values, test_shapes[i],
                                  expected_output, kTfLiteActNone, output_data);
  }
}

TF_LITE_MICRO_TEST(FloatAddWithScalarBroadcast) {
  const int output_dims_count = 6;
  float output_data[output_dims_count];

  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
  const int input2_shape[] = {0};
  const float input2_values[] = {0.1};
  const float expected_output[] = {-1.9, 0.3, 0.8, 0.9, 1.2, 2.1};

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  const int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  for (int i = 0; i < num_shapes; ++i) {
    tflite::testing::TestAddFloat(test_shapes[i], input1_values, input2_shape,
                                  input2_values, test_shapes[i],
                                  expected_output, kTfLiteActNone, output_data);
  }
}

TF_LITE_MICRO_TEST(QuantizedAddNoActivationUint8) {
  tflite::testing::QuantizedAddNoActivation<uint8_t>();
}

TF_LITE_MICRO_TEST(QuantizedAddNoActivationInt8) {
  tflite::testing::QuantizedAddNoActivation<int8_t>();
}

TF_LITE_MICRO_TEST(QuantizedAddActivationRelu1Uint8) {
  tflite::testing::QuantizedAddActivationRelu1<uint8_t>();
}

TF_LITE_MICRO_TEST(QuantizedAddActivationRelu1Int8) {
  tflite::testing::QuantizedAddActivationRelu1<int8_t>();
}

TF_LITE_MICRO_TEST(QuantizedAddVariousInputShapesUint8) {
  tflite::testing::QuantizedAddVariousInputShapes<uint8_t>();
}

TF_LITE_MICRO_TEST(QuantizedAddVariousInputShapesInt8) {
  tflite::testing::QuantizedAddVariousInputShapes<int8_t>();
}

TF_LITE_MICRO_TEST(QuantizedAddWithScalarBroadcastUint8) {
  tflite::testing::QuantizedAddWithScalarBroadcast<uint8_t>();
}

TF_LITE_MICRO_TEST(QuantizedAddWithScalarBroadcastInt8) {
  tflite::testing::QuantizedAddWithScalarBroadcast<int8_t>();
}

TF_LITE_MICRO_TEST(QuantizedAddWithMixedBroadcastUint8) {
  tflite::testing::QuantizedAddWithMixedBroadcast<uint8_t>();
}

TF_LITE_MICRO_TEST(QuantizedAddWithMixedBroadcastInt8) {
  tflite::testing::QuantizedAddWithMixedBroadcast<int8_t>();
}

TF_LITE_MICRO_TESTS_END
