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
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {

void TestUnpackThreeOutputsFloat(
    std::initializer_list<int> input_dims_data,
    std::initializer_list<float> input_data, int axis,
    std::initializer_list<int> output1_dims_data,
    std::initializer_list<float> expected_output1_data,
    std::initializer_list<int> output2_dims_data,
    std::initializer_list<float> expected_output2_data,
    std::initializer_list<int> output3_dims_data,
    std::initializer_list<float> expected_output3_data, float* output1_data,
    float* output2_data, float* output3_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInitializer(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInitializer(output2_dims_data);
  TfLiteIntArray* output3_dims = IntArrayFromInitializer(output3_dims_data);
  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);
  const int output3_dims_count = ElementCount(*output3_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 3;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims),
      CreateFloatTensor(output1_data, output1_dims),
      CreateFloatTensor(output2_data, output2_dims),
      CreateFloatTensor(output3_data, output3_dims)};

  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output1_dims_count; ++i) {
    output1_data[i] = 23;
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    output2_data[i] = 23;
  }

  for (int i = 0; i < output3_dims_count; ++i) {
    output3_data[i] = 23;
  }

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_UNPACK);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteUnpackParams builtin_data = {
      .num = 3,
      .axis = axis,
  };

  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, 0);
  }
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {3, 1, 2, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output1_data.begin()[i], output1_data[i],
                              1e-5f);
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output2_data.begin()[i], output2_data[i],
                              1e-5f);
  }

  for (int i = 0; i < output3_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output3_data.begin()[i], output3_data[i],
                              1e-5f);
  }
}

void TestUnpackOneOutputFloat(std::initializer_list<int> input_dims_data,
                              std::initializer_list<float> input_data, int axis,
                              std::initializer_list<int> output_dims_data,
                              std::initializer_list<float> expected_output_data,
                              float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 1;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims),
      CreateFloatTensor(output_data, output_dims)};

  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output_dims_count; ++i) {
    output_data[i] = 23;
  }

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_UNPACK);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteUnpackParams builtin_data = {
      .num = 1,
      .axis = axis,
  };

  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, 0);
  }
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;

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

void TestUnpackThreeOutputsQuantized(
    std::initializer_list<int> input_dims_data,
    std::initializer_list<uint8_t> input_data, int axis,
    std::initializer_list<int> output1_dims_data,
    std::initializer_list<uint8_t> expected_output1_data,
    std::initializer_list<int> output2_dims_data,
    std::initializer_list<uint8_t> expected_output2_data,
    std::initializer_list<int> output3_dims_data,
    std::initializer_list<uint8_t> expected_output3_data, uint8_t* output1_data,
    uint8_t* output2_data, uint8_t* output3_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInitializer(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInitializer(output2_dims_data);
  TfLiteIntArray* output3_dims = IntArrayFromInitializer(output3_dims_data);
  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);
  const int output3_dims_count = ElementCount(*output3_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 3;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      // CreateQuantizedTensor needs min/max values as input, but these values
      // don't matter as to the functionality of UNPACK, so just set as 0
      // and 10.
      CreateQuantizedTensor(input_data, input_dims, 0, 10),
      CreateQuantizedTensor(output1_data, output1_dims, 0, 10),
      CreateQuantizedTensor(output2_data, output2_dims, 0, 10),
      CreateQuantizedTensor(output3_data, output3_dims, 0, 10)};

  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output1_dims_count; ++i) {
    output1_data[i] = 23;
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    output2_data[i] = 23;
  }

  for (int i = 0; i < output3_dims_count; ++i) {
    output3_data[i] = 23;
  }

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_UNPACK);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteUnpackParams builtin_data = {
      .num = 3,
      .axis = axis,
  };

  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, 0);
  }
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {3, 1, 2, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output1_data.begin()[i], output1_data[i]);
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output2_data.begin()[i], output2_data[i]);
  }

  for (int i = 0; i < output3_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output3_data.begin()[i], output3_data[i]);
  }
}

void TestUnpackThreeOutputsQuantized32(
    std::initializer_list<int> input_dims_data,
    std::initializer_list<int32_t> input_data, int axis,
    std::initializer_list<int> output1_dims_data,
    std::initializer_list<int32_t> expected_output1_data,
    std::initializer_list<int> output2_dims_data,
    std::initializer_list<int32_t> expected_output2_data,
    std::initializer_list<int> output3_dims_data,
    std::initializer_list<int32_t> expected_output3_data, int32_t* output1_data,
    int32_t* output2_data, int32_t* output3_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInitializer(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInitializer(output2_dims_data);
  TfLiteIntArray* output3_dims = IntArrayFromInitializer(output3_dims_data);
  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);
  const int output3_dims_count = ElementCount(*output3_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 3;
  constexpr int tensors_size = input_size + output_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantized32Tensor(input_data, input_dims, 1.0),
      CreateQuantized32Tensor(output1_data, output1_dims, 1.0),
      CreateQuantized32Tensor(output2_data, output2_dims, 1.0),
      CreateQuantized32Tensor(output3_data, output3_dims, 1.0)};

  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output1_dims_count; ++i) {
    output1_data[i] = 23;
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    output2_data[i] = 23;
  }

  for (int i = 0; i < output3_dims_count; ++i) {
    output3_data[i] = 23;
  }

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_UNPACK);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteUnpackParams builtin_data = {
      .num = 3,
      .axis = axis,
  };

  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, 0);
  }
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {3, 1, 2, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output1_data.begin()[i], output1_data[i]);
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output2_data.begin()[i], output2_data[i]);
  }

  for (int i = 0; i < output3_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output3_data.begin()[i], output3_data[i]);
  }
}

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(UnpackFloatThreeOutputs) {
  constexpr int output1_dims_count = 2;
  constexpr int output2_dims_count = 2;
  constexpr int output3_dims_count = 2;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  float output3_data[output3_dims_count];
  tflite::testing::TestUnpackThreeOutputsFloat(
      {2, 3, 2},           // Input shape
      {1, 2, 3, 4, 5, 6},  // Input values
      0, {1, 2},           // Output1 shape
      {1, 2},              // Output1 values
      {1, 2},              // Output2 shape
      {3, 4},              // Output2 values
      {1, 2},              // Output3 shape
      {5, 6},              // Output3 values
      output1_data, output2_data, output3_data);
}

TF_LITE_MICRO_TEST(UnpackFloatThreeOutputsNegativeAxisTwo) {
  constexpr int output1_dims_count = 2;
  constexpr int output2_dims_count = 2;
  constexpr int output3_dims_count = 2;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  float output3_data[output3_dims_count];
  tflite::testing::TestUnpackThreeOutputsFloat(
      {2, 3, 2},           // Input shape
      {1, 2, 3, 4, 5, 6},  // Input values
      -2, {1, 2},          // Output1 shape
      {1, 2},              // Output1 values
      {1, 2},              // Output2 shape
      {3, 4},              // Output2 values
      {1, 2},              // Output3 shape
      {5, 6},              // Output3 values
      output1_data, output2_data, output3_data);
}

TF_LITE_MICRO_TEST(UnpackFloatOneOutput) {
  constexpr int output_dims_count = 6;
  float output_data[output_dims_count];
  tflite::testing::TestUnpackOneOutputFloat(
      {2, 1, 6},           // Input shape
      {1, 2, 3, 4, 5, 6},  // Input values
      0, {1, 6},           // Output shape
      {1, 2, 3, 4, 5, 6},  // Output values
      output_data);
}

TF_LITE_MICRO_TEST(UnpackQuantizedThreeOutputs) {
  constexpr int output1_dims_count = 2;
  constexpr int output2_dims_count = 2;
  constexpr int output3_dims_count = 2;
  uint8_t output1_data[output1_dims_count];
  uint8_t output2_data[output2_dims_count];
  uint8_t output3_data[output3_dims_count];
  tflite::testing::TestUnpackThreeOutputsQuantized(
      {2, 3, 2},           // Input shape
      {1, 2, 3, 4, 5, 6},  // Input values
      0, {1, 2},           // Output1 shape
      {1, 2},              // Output1 values
      {1, 2},              // Output2 shape
      {3, 4},              // Output2 values
      {1, 2},              // Output3 shape
      {5, 6},              // Output3 values
      output1_data, output2_data, output3_data);
}

TF_LITE_MICRO_TEST(UnpackQuantized32ThreeOutputs) {
  constexpr int output1_dims_count = 2;
  constexpr int output2_dims_count = 2;
  constexpr int output3_dims_count = 2;
  int32_t output1_data[output1_dims_count];
  int32_t output2_data[output2_dims_count];
  int32_t output3_data[output3_dims_count];
  tflite::testing::TestUnpackThreeOutputsQuantized32(
      {2, 3, 2},           // Input shape
      {1, 2, 3, 4, 5, 6},  // Input values
      0, {1, 2},           // Output1 shape
      {1, 2},              // Output1 values
      {1, 2},              // Output2 shape
      {3, 4},              // Output2 values
      {1, 2},              // Output3 shape
      {5, 6},              // Output3 values
      output1_data, output2_data, output3_data);
}

TF_LITE_MICRO_TESTS_END
