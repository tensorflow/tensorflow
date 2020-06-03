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
#include <initializer_list>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void TestConcatenateTwoInputs(std::initializer_list<int> input1_dims_data,
                              std::initializer_list<float> input1_data,
                              std::initializer_list<int> input2_dims_data,
                              std::initializer_list<float> input2_data,
                              int axis,
                              std::initializer_list<int> output_dims_data,
                              std::initializer_list<float> expected_output_data,
                              float* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInitializer(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInitializer(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);

  constexpr int input_size = 2;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input1_data, input1_dims, "input1_tensor"),
      CreateFloatTensor(input2_data, input2_dims, "input2_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor")};

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_CONCATENATION);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteConcatenationParams builtin_data = {
      .axis = axis,
      .activation = kTfLiteActNone  // Only activation supported in this impl
  };

  TfLiteIntArray* inputs_array = IntArrayFromInitializer({2, 0, 1});
  TfLiteIntArray* outputs_array = IntArrayFromInitializer({1, 2});
  TfLiteIntArray* temporaries_array = IntArrayFromInitializer({0});

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = nullptr;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));

  const int output_dims_count = ElementCount(*output_dims);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data.begin()[i], output_data[i],
                              1e-5f);
  }
}

void TestConcatenateQuantizedTwoInputs(
    std::initializer_list<int> input1_dims_data,
    std::initializer_list<uint8_t> input1_data,
    std::initializer_list<int> input2_dims_data,
    std::initializer_list<uint8_t> input2_data, float input_min,
    float input_max, int axis, std::initializer_list<int> output_dims_data,
    std::initializer_list<uint8_t> expected_output_data, float output_min,
    float output_max, uint8_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInitializer(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInitializer(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);

  constexpr int input_size = 2;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_dims, "input1_tensor",
                            input_min, input_max),
      CreateQuantizedTensor(input2_data, input2_dims, "input2_tensor",
                            input_min, input_max),
      CreateQuantizedTensor(output_data, output_dims, "output_tensor",
                            output_min, output_max)};

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_CONCATENATION);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteConcatenationParams builtin_data = {
      .axis = axis,
      .activation = kTfLiteActNone  // Only activation supported in this impl
  };

  TfLiteIntArray* inputs_array = IntArrayFromInitializer({2, 0, 1});
  TfLiteIntArray* outputs_array = IntArrayFromInitializer({1, 2});
  TfLiteIntArray* temporaries_array = IntArrayFromInitializer({0});

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = nullptr;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));

  const int output_dims_count = ElementCount(*output_dims);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data.begin()[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TwoInputsAllAxesCombinations) {
  // Concatenate the same two input tensors along all possible axes.

  auto input_shape = {2, 2, 3};
  auto input1_value = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto input2_value = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  // expected output when concatenating on axis 0
  auto output_shape_axis0 = {2, 4, 3};
  auto output_value_axis0 = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                             7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  // expected output when concatenating on axis 1
  auto output_shape_axis1 = {2, 2, 6};
  auto output_value_axis1 = {1.0f, 2.0f, 3.0f, 7.0f,  8.0f,  9.0f,
                             4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f};

  float output_data[12];

  // Axis = 0
  tflite::testing::TestConcatenateTwoInputs(
      input_shape, input1_value, input_shape, input2_value, /* axis */ 0,
      output_shape_axis0, output_value_axis0, output_data);

  // Axis = -2 (equivalent to axis = 0)
  tflite::testing::TestConcatenateTwoInputs(
      input_shape, input1_value, input_shape, input2_value, /* axis */ -2,
      output_shape_axis0, output_value_axis0, output_data);

  // Axis = 1
  tflite::testing::TestConcatenateTwoInputs(
      input_shape, input1_value, input_shape, input2_value, /* axis */ 1,
      output_shape_axis1, output_value_axis1, output_data);

  // Axis = -1 (equivalent to axis = 1)
  tflite::testing::TestConcatenateTwoInputs(
      input_shape, input1_value, input_shape, input2_value, /* axis */ -1,
      output_shape_axis1, output_value_axis1, output_data);
}

TF_LITE_MICRO_TEST(TwoInputsQuantizedUint8) {
  using tflite::testing::F2Q;

  const int axis = 2;
  auto input_shape = {3, 2, 1, 2};
  auto output_shape = {3, 2, 1, 4};

  const float input_min = -12.7f;
  const float input_max = 12.8f;
  const float output_min = -12.7f;
  const float output_max = 12.8f;

  auto input1_value = {
      F2Q(1.0, input_min, input_max),
      F2Q(3.0, input_min, input_max),
      F2Q(4.0, input_min, input_max),
      F2Q(7.0, input_min, input_max),
  };

  auto input2_value = {
      F2Q(1.1, input_min, input_max),
      F2Q(3.1, input_min, input_max),
      F2Q(4.1, input_min, input_max),
      F2Q(7.1, input_min, input_max),
  };

  std::initializer_list<uint8_t> output_value = {
      137, 157, 138, 158, 167, 197, 168, 198,
  };

  uint8_t output_data[8];
  tflite::testing::TestConcatenateQuantizedTwoInputs(
      input_shape, input1_value, input_shape, input2_value, input_min,
      input_max, axis, output_shape, output_value, output_min, output_max,
      output_data);
}

TF_LITE_MICRO_TEST(ThreeDimensionalTwoInputsDifferentShapes) {
  const int axis = 1;

  auto input1_shape = {3, 2, 1, 2};
  auto input2_shape = {3, 2, 3, 2};
  auto output_shape = {3, 2, 4, 2};

  auto input1_value = {1.0f, 3.0f, 4.0f, 7.0f};
  auto input2_value = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                       7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  auto output_value = {1.0f, 3.0f, 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                       4.0f, 7.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  float output_data[16];
  tflite::testing::TestConcatenateTwoInputs(
      input1_shape, input1_value, input2_shape, input2_value, axis,
      output_shape, output_value, output_data);
}

TF_LITE_MICRO_TESTS_END
