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
#include "tensorflow/lite/experimental/micro/simple_tensor_allocator.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void TestAveragePoolingFloat(std::initializer_list<int> input_dims_data,
                             std::initializer_list<float> input_data,
                             const int filter_height, const int filter_width,
                             const int stride_height, const int stride_width,
                             std::initializer_list<float> expected_output_data,
                             std::initializer_list<int> output_dims_data,
                             TfLitePadding padding,
                             TfLiteFusedActivation activation,
                             float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

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
      resolver.FindOp(tflite::BuiltinOperator_AVERAGE_POOL_2D, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteConvParams builtin_data = {padding,      stride_width,  stride_height,
                                   filter_width, filter_height, activation};
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

void TestAveragePoolingUint8(
    std::initializer_list<int> input_dims_data,
    std::initializer_list<uint8_t> input_data, const float input_min,
    const float input_max, const int filter_height, const int filter_width,
    const int stride_height, const int stride_width,
    std::initializer_list<uint8_t> expected_output_data,
    std::initializer_list<int> output_dims_data, float output_min,
    float output_max, TfLitePadding padding, TfLiteFusedActivation activation,
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
      resolver.FindOp(tflite::BuiltinOperator_AVERAGE_POOL_2D, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteConvParams builtin_data = {padding,      stride_width,  stride_height,
                                   filter_width, filter_height, activation};
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

}  // namespace

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleAveragePoolTestFloat) {
  float output_data[2];
  tflite::testing::TestAveragePoolingFloat({4, 1, 2, 4, 1},  // Input shape
                                           {                 // Input values
                                            0., 6., 2., 4., 3., 2., 10., 7.},
                                           2, 2,  // filter width, filter height
                                           2, 2,  // stride width, stride height
                                           {
                                               // Output values
                                               2.75,
                                               5.75,
                                           },
                                           {4, 1, 1, 2, 1},  // Output shape
                                           kTfLitePaddingValid, kTfLiteActNone,
                                           output_data);
}

TF_LITE_MICRO_TEST(SimpleAveragePoolTestUint8) {
  using tflite::testing::F2Q;

  const float input_min = -15.9375;
  const float input_max = 15.9375;
  const float output_min = -15.9375;
  const float output_max = 15.9375;
  uint8_t output_data[2];
  tflite::testing::TestAveragePoolingUint8(
      {4, 1, 2, 4, 1},  // Input shape
      {
          // Input values
          F2Q(0., input_min, input_max),
          F2Q(-6., input_min, input_max),
          F2Q(2., input_min, input_max),
          F2Q(4., input_min, input_max),
          F2Q(3., input_min, input_max),
          F2Q(2., input_min, input_max),
          F2Q(-10., input_min, input_max),
          F2Q(7., input_min, input_max),
      },
      input_min, input_max,  // input quantization range
      2, 2,                  // filter width, filter height
      2, 2,                  // stride width, stride height
      {
          // Output values
          F2Q(0., output_min, output_max),
          F2Q(0.75, output_min, output_max),
      },
      {4, 1, 1, 2, 1},         // Output shape
      output_min, output_max,  // output quantization range
      kTfLitePaddingValid, kTfLiteActRelu, output_data);
}

TF_LITE_MICRO_TESTS_END
