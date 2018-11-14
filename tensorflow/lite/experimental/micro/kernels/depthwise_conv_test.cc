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

void TestDepthwiseConvFloat(std::initializer_list<int> input_dims_data,
                            std::initializer_list<float> input_data,
                            std::initializer_list<int> filter_dims_data,
                            std::initializer_list<float> filter_data,
                            std::initializer_list<int> bias_dims_data,
                            std::initializer_list<float> bias_data,
                            std::initializer_list<float> expected_output_data,
                            std::initializer_list<int> output_dims_data,
                            TfLiteFusedActivation activation,
                            float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInitializer(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInitializer(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateFloatTensor(filter_data, filter_dims, "filter_tensor"),
      CreateFloatTensor(bias_data, bias_dims, "bias_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_DEPTHWISE_CONV_2D, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  int input_depth = input_dims->data[3];
  int output_depth = filter_dims->data[3];
  int depth_mul = output_depth / input_depth;
  TfLiteDepthwiseConvParams builtin_data = {
      kTfLitePaddingValid, 1, 1, depth_mul, activation,
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

void TestDepthwiseConvQuantized(
    std::initializer_list<int> input_dims_data,
    std::initializer_list<uint8_t> input_data, float input_min, float input_max,
    std::initializer_list<int> filter_dims_data,
    std::initializer_list<uint8_t> filter_data, float filter_min,
    float filter_max, std::initializer_list<int> bias_dims_data,
    std::initializer_list<int32_t> bias_data, float bias_min, float bias_max,
    std::initializer_list<uint8_t> expected_output_data,
    std::initializer_list<int> output_dims_data, float output_min,
    float output_max, TfLiteFusedActivation activation, uint8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInitializer(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInitializer(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, "input_tensor", input_min,
                            input_max),
      CreateQuantizedTensor(filter_data, filter_dims, "filter_tensor",
                            filter_min, filter_max),
      CreateQuantized32Tensor(bias_data, bias_dims, "bias_tensor", bias_min,
                              bias_max),
      CreateQuantizedTensor(output_data, output_dims, "output_tensor",
                            output_min, output_max),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_DEPTHWISE_CONV_2D, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  int input_depth = input_dims->data[3];
  int output_depth = filter_dims->data[3];
  int depth_mul = output_depth / input_depth;
  TfLiteDepthwiseConvParams builtin_data = {
      kTfLitePaddingValid, 1, 1, depth_mul, activation,
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
  const int output_dims_count = 8;
  float output_data[output_dims_count];
  tflite::testing::TestDepthwiseConvFloat(  //
      {4, 1, 3, 2, 2},                      // Input shape.
      {
          1, 2, 7, 8,    // Input values.
          3, 4, 9, 10,   //
          5, 6, 11, 12,  //
      },
      {4, 1, 2, 2, 4},  // Filters shape.
      {
          1, 2, 3, 4,        // Filters values.
          -9, 10, -11, 12,   //
          5, 6, 7, 8,        //
          13, -14, 15, -16,  //
      },
      {1, 4},  // Bias shape.
      {
          1, 2, 3, 4,  // Bias values.
      },
      {
          71, -34, 99, -20,  // Expected results.
          91, -26, 127, -4,  //
      },
      {4, 1, 2, 1, 4},  // Output shape.
      kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestQuantized) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float filter_min = -63.5f;
  const float filter_max = 64.0f;
  const float bias_min = 0.0f;
  const float bias_max = 64.0f * (1 << 24);
  const float output_min = -127.0f;
  const float output_max = 128.0f;
  const int output_dims_count = 8;
  uint8_t output_data[output_dims_count];

  tflite::testing::TestDepthwiseConvQuantized(  //
      {4, 1, 3, 2, 2},                          // Input shape.
      {
          // Input values.
          F2Q(1, input_min, input_max),
          F2Q(2, input_min, input_max),
          F2Q(7, input_min, input_max),
          F2Q(8, input_min, input_max),
          F2Q(3, input_min, input_max),
          F2Q(4, input_min, input_max),
          F2Q(9, input_min, input_max),
          F2Q(10, input_min, input_max),
          F2Q(5, input_min, input_max),
          F2Q(6, input_min, input_max),
          F2Q(11, input_min, input_max),
          F2Q(12, input_min, input_max),
      },
      input_min, input_max,  // Input quantization range.
      {4, 1, 2, 2, 4},       // Filter shape.
      {
          // Filter values.
          F2Q(1, filter_min, filter_max),
          F2Q(2, filter_min, filter_max),
          F2Q(3, filter_min, filter_max),
          F2Q(4, filter_min, filter_max),
          F2Q(-9, filter_min, filter_max),
          F2Q(10, filter_min, filter_max),
          F2Q(-11, filter_min, filter_max),
          F2Q(12, filter_min, filter_max),
          F2Q(5, filter_min, filter_max),
          F2Q(6, filter_min, filter_max),
          F2Q(7, filter_min, filter_max),
          F2Q(8, filter_min, filter_max),
          F2Q(13, filter_min, filter_max),
          F2Q(-14, filter_min, filter_max),
          F2Q(15, filter_min, filter_max),
          F2Q(-16, filter_min, filter_max),
      },
      filter_min, filter_max,  // Filter quantization range.
      {1, 4},                  // Bias shape.
      {
          // Bias values.
          F2Q32(1, bias_min, bias_max),
          F2Q32(2, bias_min, bias_max),
          F2Q32(3, bias_min, bias_max),
          F2Q32(4, bias_min, bias_max),
      },
      bias_min, bias_max,  // Bias quantization range.
      {
          // Expected results.
          F2Q(71, output_min, output_max),
          F2Q(-34, output_min, output_max),
          F2Q(99, output_min, output_max),
          F2Q(-20, output_min, output_max),
          F2Q(91, output_min, output_max),
          F2Q(-26, output_min, output_max),
          F2Q(127, output_min, output_max),
          F2Q(-4, output_min, output_max),
      },
      {4, 1, 2, 1, 4},         // Output shape.
      output_min, output_max,  // Output quantization range.
      kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestRelu) {
  const int output_dims_count = 8;
  float output_data[output_dims_count];
  tflite::testing::TestDepthwiseConvFloat(  //
      {4, 1, 3, 2, 2},                      // Input shape.
      {
          1, 2, 7, 8,    // Input values.
          3, 4, 9, 10,   //
          5, 6, 11, 12,  //
      },
      {4, 1, 2, 2, 4},  // Filters shape.
      {
          1, 2, 3, 4,        // Filters values.
          -9, 10, -11, 12,   //
          5, 6, 7, 8,        //
          13, -14, 15, -16,  //
      },
      {1, 4},  // Bias shape.
      {
          1, 2, 3, 4,  // Bias values.
      },
      {
          71, 0, 99, 0,   // Expected results.
          91, 0, 127, 0,  //
      },
      {4, 1, 2, 1, 4},  // Output shape.
      kTfLiteActRelu, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestReluQuantized) {
  using tflite::testing::F2Q;
  using tflite::testing::F2Q32;

  const float input_min = -63.5f;
  const float input_max = 64.0f;
  const float filter_min = -63.5f;
  const float filter_max = 64.0f;
  const float bias_min = 0.0f;
  const float bias_max = 64.0f * (1 << 24);
  const float output_min = -127.0f;
  const float output_max = 128.0f;
  const int output_dims_count = 8;
  uint8_t output_data[output_dims_count];

  tflite::testing::TestDepthwiseConvQuantized(  //
      {4, 1, 3, 2, 2},                          // Input shape.
      {
          // Input values.
          F2Q(1, input_min, input_max),
          F2Q(2, input_min, input_max),
          F2Q(7, input_min, input_max),
          F2Q(8, input_min, input_max),
          F2Q(3, input_min, input_max),
          F2Q(4, input_min, input_max),
          F2Q(9, input_min, input_max),
          F2Q(10, input_min, input_max),
          F2Q(5, input_min, input_max),
          F2Q(6, input_min, input_max),
          F2Q(11, input_min, input_max),
          F2Q(12, input_min, input_max),
      },
      input_min, input_max,  // Input quantization range.
      {4, 1, 2, 2, 4},       // Filter shape.
      {
          // Filter values.
          F2Q(1, filter_min, filter_max),
          F2Q(2, filter_min, filter_max),
          F2Q(3, filter_min, filter_max),
          F2Q(4, filter_min, filter_max),
          F2Q(-9, filter_min, filter_max),
          F2Q(10, filter_min, filter_max),
          F2Q(-11, filter_min, filter_max),
          F2Q(12, filter_min, filter_max),
          F2Q(5, filter_min, filter_max),
          F2Q(6, filter_min, filter_max),
          F2Q(7, filter_min, filter_max),
          F2Q(8, filter_min, filter_max),
          F2Q(13, filter_min, filter_max),
          F2Q(-14, filter_min, filter_max),
          F2Q(15, filter_min, filter_max),
          F2Q(-16, filter_min, filter_max),
      },
      filter_min, filter_max,  // Filter quantization range.
      {1, 4},                  // Bias shape.
      {
          // Bias values.
          F2Q32(1, bias_min, bias_max),
          F2Q32(2, bias_min, bias_max),
          F2Q32(3, bias_min, bias_max),
          F2Q32(4, bias_min, bias_max),
      },
      bias_min, bias_max,  // Bias quantization range.
      {
          // Expected results.
          F2Q(71, output_min, output_max),
          F2Q(0, output_min, output_max),
          F2Q(99, output_min, output_max),
          F2Q(0, output_min, output_max),
          F2Q(91, output_min, output_max),
          F2Q(0, output_min, output_max),
          F2Q(127, output_min, output_max),
          F2Q(0, output_min, output_max),
      },
      {4, 1, 2, 1, 4},         // Output shape.
      output_min, output_max,  // Output quantization range.
      kTfLiteActRelu, output_data);
}

TF_LITE_MICRO_TESTS_END
