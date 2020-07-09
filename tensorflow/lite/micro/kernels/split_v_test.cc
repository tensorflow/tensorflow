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
#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {

void TestSplitVThreeOutputsFloat(
    std::initializer_list<int> input_dims_data,
    std::initializer_list<float> input_data,
    std::initializer_list<int> axis_dims_data,
    std::initializer_list<int32_t> axis_data,
    std::initializer_list<int> split_dims_data,
    std::initializer_list<int32_t> split_data,
    std::initializer_list<int> output1_dims_data,
    std::initializer_list<float> expected_output1_data,
    std::initializer_list<int> output2_dims_data,
    std::initializer_list<float> expected_output2_data,
    std::initializer_list<int> output3_dims_data,
    std::initializer_list<float> expected_output3_data, float* output1_data,
    float* output2_data, float* output3_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInitializer(axis_dims_data);
  TfLiteIntArray* split_dims = IntArrayFromInitializer(split_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInitializer(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInitializer(output2_dims_data);
  TfLiteIntArray* output3_dims = IntArrayFromInitializer(output3_dims_data);

  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);
  const int output3_dims_count = ElementCount(*output3_dims);

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

  constexpr int input_size = 1;
  constexpr int axis_size = 1;
  constexpr int split_size = 1;
  constexpr int output_size = 3;

  TfLiteContext context;
  constexpr int tensors_size =
      input_size + output_size + axis_size + split_size;

  // first input tensor is data
  // second is size_splits
  // third is axis
  // then come outputs

  TfLiteTensor tensors[tensors_size] = {
      // inputs
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateQuantized32Tensor(split_data, split_dims, "size_splits", 1.0),
      CreateQuantized32Tensor(axis_data, axis_dims, "axis_tensor", 1.0),

      // outputs
      CreateFloatTensor(output1_data, output1_dims, "output1_tensor"),
      CreateFloatTensor(output2_data, output2_dims, "output2_tensor"),
      CreateFloatTensor(output3_data, output3_dims, "output3_tensor")

  };
  tensors[2].allocation_type = kTfLiteMmapRo;

  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_SPLIT_V, /* version= */ 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  void* user_data = nullptr;
  TfLiteSplitVParams builtin;
  builtin.num_splits = 3;
  TfLiteIntArray* inputs_array = IntArrayFromInitializer({3, 0, 1, 2});
  TfLiteIntArray* outputs_array = IntArrayFromInitializer({3, 3, 4, 5});
  TfLiteIntArray* temporaries_array = IntArrayFromInitializer({0});

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;

  node.user_data = nullptr;
  node.builtin_data = reinterpret_cast<void*>(&builtin);
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

void TestSplitVTwoOutputsFloat(
    std::initializer_list<int> input_dims_data,
    std::initializer_list<float> input_data,
    std::initializer_list<int> axis_dims_data,
    std::initializer_list<int32_t> axis_data,
    std::initializer_list<int> split_dims_data,
    std::initializer_list<int32_t> size_splits_data,
    std::initializer_list<int> output1_dims_data,
    std::initializer_list<float> expected_output1_data,
    std::initializer_list<int> output2_dims_data,
    std::initializer_list<float> expected_output2_data, float* output1_data,
    float* output2_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInitializer(axis_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInitializer(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInitializer(output2_dims_data);
  TfLiteIntArray* size_splits_dims = IntArrayFromInitializer(split_dims_data);

  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 2;
  constexpr int axis_size = 1;
  constexpr int split_size = 1;
  constexpr int tensors_size =
      input_size + output_size + axis_size + split_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateQuantized32Tensor(size_splits_data, size_splits_dims, "size_splits",
                              1.0),
      CreateQuantized32Tensor(axis_data, axis_dims, "axis_tensor", 1.0),
      CreateFloatTensor(output1_data, output1_dims, "output1_tensor"),
      CreateFloatTensor(output2_data, output2_dims, "output2_tensor")};

  // Currently only support constant axis tensor.
  tensors[0].allocation_type = kTfLiteMmapRo;
  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output1_dims_count; ++i) {
    output1_data[i] = 23;
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    output2_data[i] = 23;
  }

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_SPLIT_V, /* version= */ 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteSplitVParams builtin_data;
  builtin_data.num_splits = 2;

  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, 0);
  }
  TfLiteIntArray* inputs_array = IntArrayFromInitializer({3, 0, 1, 2});
  TfLiteIntArray* outputs_array = IntArrayFromInitializer({2, 3, 4});
  TfLiteIntArray* temporaries_array = IntArrayFromInitializer({0});

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

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output1_data.begin()[i], output1_data[i],
                              1e-5f);
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output2_data.begin()[i], output2_data[i],
                              1e-5f);
  }
}
void TestSplitVEightOutputsFloat(
    std::initializer_list<int> input_dims_data,
    std::initializer_list<float> input_data,
    std::initializer_list<int> axis_dims_data,
    std::initializer_list<int32_t> axis_data,
    std::initializer_list<int> split_dims_data,
    std::initializer_list<int32_t> size_splits_data,
    std::initializer_list<int> output1_dims_data,
    std::initializer_list<float> expected_output1_data,
    std::initializer_list<int> output2_dims_data,
    std::initializer_list<float> expected_output2_data,
    std::initializer_list<int> output3_dims_data,
    std::initializer_list<float> expected_output3_data,
    std::initializer_list<int> output4_dims_data,
    std::initializer_list<float> expected_output4_data,
    std::initializer_list<int> output5_dims_data,
    std::initializer_list<float> expected_output5_data,
    std::initializer_list<int> output6_dims_data,
    std::initializer_list<float> expected_output6_data,
    std::initializer_list<int> output7_dims_data,
    std::initializer_list<float> expected_output7_data,
    std::initializer_list<int> output8_dims_data,
    std::initializer_list<float> expected_output8_data,

    float* output1_data, float* output2_data, float* output3_data,
    float* output4_data, float* output5_data, float* output6_data,
    float* output7_data, float* output8_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInitializer(axis_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInitializer(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInitializer(output2_dims_data);
  TfLiteIntArray* output3_dims = IntArrayFromInitializer(output3_dims_data);
  TfLiteIntArray* output4_dims = IntArrayFromInitializer(output4_dims_data);
  TfLiteIntArray* output5_dims = IntArrayFromInitializer(output5_dims_data);
  TfLiteIntArray* output6_dims = IntArrayFromInitializer(output6_dims_data);
  TfLiteIntArray* output7_dims = IntArrayFromInitializer(output7_dims_data);
  TfLiteIntArray* output8_dims = IntArrayFromInitializer(output8_dims_data);

  TfLiteIntArray* size_splits_dims = IntArrayFromInitializer(split_dims_data);

  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);
  const int output3_dims_count = ElementCount(*output3_dims);
  const int output4_dims_count = ElementCount(*output4_dims);
  const int output5_dims_count = ElementCount(*output5_dims);
  const int output6_dims_count = ElementCount(*output6_dims);
  const int output7_dims_count = ElementCount(*output7_dims);
  const int output8_dims_count = ElementCount(*output8_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 8;
  constexpr int axis_size = 1;
  constexpr int split_size = 1;
  constexpr int tensors_size =
      input_size + output_size + axis_size + split_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateQuantized32Tensor(size_splits_data, size_splits_dims, "size_splits",
                              1.0),
      CreateQuantized32Tensor(axis_data, axis_dims, "axis_tensor", 1.0),
      CreateFloatTensor(output1_data, output1_dims, "output1_tensor"),
      CreateFloatTensor(output2_data, output2_dims, "output2_tensor"),
      CreateFloatTensor(output3_data, output3_dims, "output3_tensor"),
      CreateFloatTensor(output4_data, output4_dims, "output4_tensor"),
      CreateFloatTensor(output5_data, output5_dims, "output5_tensor"),
      CreateFloatTensor(output6_data, output6_dims, "output6_tensor"),
      CreateFloatTensor(output7_data, output7_dims, "output7_tensor"),
      CreateFloatTensor(output8_data, output8_dims, "output8_tensor")};

  // Currently only support constant axis tensor.
  tensors[0].allocation_type = kTfLiteMmapRo;
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

  for (int i = 0; i < output4_dims_count; ++i) {
    output4_data[i] = 23;
  }
  for (int i = 0; i < output5_dims_count; ++i) {
    output5_data[i] = 23;
  }

  for (int i = 0; i < output6_dims_count; ++i) {
    output6_data[i] = 23;
  }
  for (int i = 0; i < output7_dims_count; ++i) {
    output7_data[i] = 23;
  }

  for (int i = 0; i < output8_dims_count; ++i) {
    output8_data[i] = 23;
  }

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_SPLIT_V, /* version= */ 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteSplitVParams builtin_data;
  builtin_data.num_splits = 8;

  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, 0);
  }
  TfLiteIntArray* inputs_array = IntArrayFromInitializer({3, 0, 1, 2});
  TfLiteIntArray* outputs_array =
      IntArrayFromInitializer({8, 3, 4, 5, 6, 7, 8, 9, 10});
  TfLiteIntArray* temporaries_array = IntArrayFromInitializer({0});

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

  for (int i = 0; i < output4_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output4_data.begin()[i], output4_data[i],
                              1e-5f);
  }
  for (int i = 0; i < output5_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output5_data.begin()[i], output5_data[i],
                              1e-5f);
  }

  for (int i = 0; i < output6_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output6_data.begin()[i], output6_data[i],
                              1e-5f);
  }
  for (int i = 0; i < output7_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output7_data.begin()[i], output7_data[i],
                              1e-5f);
  }

  for (int i = 0; i < output8_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output8_data.begin()[i], output8_data[i],
                              1e-5f);
  }
}

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SPLIT_V_ThreeOutputs) {
  constexpr int output1_dims_count = 3;
  constexpr int output2_dims_count = 3;
  constexpr int output3_dims_count = 6;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  float output3_data[output3_dims_count];

  tflite::testing::TestSplitVThreeOutputsFloat(
      {2, 4, 3},                                // input shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // input values
      {1, 1},                                   // Axis shape
      {0},                                      // Axis value
      {1, 3},                                   // split shape
      {1, 1, 2},                                // split values
      {2, 1, 3},                                // output1 shape
      {1, 2, 3},                                // output1 values
      {2, 1, 3},                                // output2 shape
      {4, 5, 6},                                // output2 values
      {2, 2, 3},                                // output3 shapes
      {7, 8, 9, 10, 11, 12},                    // output3 values
      output1_data, output2_data, output3_data  // output buffers
  );
}

TF_LITE_MICRO_TEST(SPLIT_V_FourDimensionalFloatAxis0) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitVTwoOutputsFloat(
      {4, 2, 2, 2, 2},                                          // Input shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},  // Input values
      {1, 1},                                                   // Axis shape
      {0},                                                      // Axis value
      {1, 2},                           // split_size shape
      {1, 1},                           // split
      {4, 1, 2, 2, 2},                  // Output1 shape
      {1, 2, 3, 4, 5, 6, 7, 8},         // Output1 values
      {4, 1, 2, 2, 2},                  // Output2 shape
      {9, 10, 11, 12, 13, 14, 15, 16},  // Output2 values
      output1_data, output2_data);      // locally allocated output buffers
}

TF_LITE_MICRO_TEST(SPLIT_V_FourDimensionalFloatAxis1) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitVTwoOutputsFloat(
      {4, 2, 2, 2, 2},                                          // Input shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},  // Input values
      {1, 1},                                                   // Axis shape
      {1},                                                      // Axis value
      {1, 2},                        // split_size shape
      {1, 1},                        // split
      {4, 2, 1, 2, 2},               // Output1 shape
      {1, 2, 3, 4, 9, 10, 11, 12},   // Output1 values
      {4, 2, 1, 2, 2},               // Output2 shape
      {5, 6, 7, 8, 13, 14, 15, 16},  // Output2 values
      output1_data, output2_data);   // locally allocated output buffers
}

TF_LITE_MICRO_TEST(SPLIT_VFourDimensionalFloatAxis2) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitVTwoOutputsFloat(
      {4, 2, 2, 2, 2},                                          // Input shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},  // Input values
      {1, 1},                                                   // Axis shape
      {2},                                                      // Axis value
      {1, 2},                        // split_size shape
      {1, 1},                        // split
      {4, 2, 2, 1, 2},               // Output1 shape
      {1, 2, 5, 6, 9, 10, 13, 14},   // Output1 values
      {4, 2, 2, 1, 2},               // Output2 shape
      {3, 4, 7, 8, 11, 12, 15, 16},  // Output2 values
      output1_data, output2_data);   // locally allocated output buffers
}

TF_LITE_MICRO_TEST(SPLIT_V_FourDimensionalFloatAxis3) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitVTwoOutputsFloat(
      {4, 2, 2, 2, 2},                                          // Input shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},  // Input values
      {1, 1},                                                   // Axis shape
      {3},                                                      // Axis value
      {1, 2},                        // split_size shape
      {1, 1},                        // split
      {4, 2, 2, 2, 1},               // Output1 shape
      {1, 3, 5, 7, 9, 11, 13, 15},   // Output1 values
      {4, 2, 2, 2, 1},               // Output2 shape
      {2, 4, 6, 8, 10, 12, 14, 16},  // Output2 values
      output1_data, output2_data);   // locally allocated output buffers
}

TF_LITE_MICRO_TEST(SPLIT_V_FourDimensionalFloatNegativeAxis) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitVTwoOutputsFloat(
      {4, 2, 2, 2, 2},                                          // Input shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},  // Input values
      {1, 1},                                                   // Axis shape
      {-4},                                                     // Axis value
      {1, 2},                           // split_size shape
      {1, 1},                           // split
      {4, 1, 2, 2, 2},                  // Output1 shape
      {1, 2, 3, 4, 5, 6, 7, 8},         // Output1 values
      {4, 1, 2, 2, 2},                  // Output2 shape
      {9, 10, 11, 12, 13, 14, 15, 16},  // Output2 values
      output1_data, output2_data);
}

TF_LITE_MICRO_TEST(SPLIT_V_OneDimensionalFloatAxis0) {
  constexpr int output1_dims_count = 1;
  constexpr int output2_dims_count = 1;
  constexpr int output3_dims_count = 1;
  constexpr int output4_dims_count = 1;
  constexpr int output5_dims_count = 1;
  constexpr int output6_dims_count = 1;
  constexpr int output7_dims_count = 1;
  constexpr int output8_dims_count = 1;

  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  float output3_data[output3_dims_count];
  float output4_data[output4_dims_count];
  float output5_data[output5_dims_count];
  float output6_data[output6_dims_count];
  float output7_data[output7_dims_count];
  float output8_data[output8_dims_count];
  tflite::testing::TestSplitVEightOutputsFloat(
      {1, 8},                      // Input shape
      {1, 2, 3, 4, 5, 6, 7, 8},    // Input values
      {1, 1},                      // Axis shape
      {0},                         // Axis value
      {1, 8},                      // split_size shape
      {1, 1, 1, 1, 1, 1, 1, 1},    // split
      {1, 1},                      // Output1 shape
      {1},                         // Output1 values
      {1, 1},                      // Output2 shape
      {2},                         // Output2 values
      {1, 1},                      // Output3 shape
      {3},                         // Output3 values
      {1, 1},                      // Output4 shape
      {4},                         // Output4 values
      {1, 1},                      // Output5 shape
      {5},                         // Output5 values
      {1, 1},                      // Output6 shape
      {6},                         // Output6 values
      {1, 1},                      // Output7 shape
      {7},                         // Output7 values
      {1, 1},                      // Output8 shape
      {8},                         // Output8 values
      output1_data, output2_data,  // locally allocated output buffers
      output3_data, output4_data, output5_data, output6_data, output7_data,
      output8_data);
}

TF_LITE_MICRO_TEST(SPLIT_V_OneDimensionalFloatTest2) {  
  constexpr int output1_dims_count = 1;
  constexpr int output2_dims_count = 1;
  constexpr int output3_dims_count = 1;
  constexpr int output4_dims_count = 1;
  constexpr int output5_dims_count = 1;
  constexpr int output6_dims_count = 1;
  constexpr int output7_dims_count = 2;
  constexpr int output8_dims_count = 0;

  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  float output3_data[output3_dims_count];
  float output4_data[output4_dims_count];
  float output5_data[output5_dims_count];
  float output6_data[output6_dims_count];
  float output7_data[output7_dims_count];

  tflite::testing::TestSplitVEightOutputsFloat(
      {1, 8},                      // Input shape
      {1, 2, 3, 4, 5, 6, 7, 8},    // Input values
      {1, 1},                      // Axis shape
      {0},                         // Axis value
      {1, 8},                      // split_size shape
      {1, 1, 1, 1, 1, 1, 2, -1},   // split
      {1, 1},                      // Output1 shape
      {1},                         // Output1 values
      {1, 1},                      // Output2 shape
      {2},                         // Output2 values
      {1, 1},                      // Output3 shape
      {3},                         // Output3 values
      {1, 1},                      // Output4 shape
      {4},                         // Output4 values
      {1, 1},                      // Output5 shape
      {5},                         // Output5 values
      {1, 1},                      // Output6 shape
      {6},                         // Output6 values
      {1, 2},                      // Output7 shape
      {7, 8},                      // Output7 values
      {1, 0},                      // Output8 shape
      {},                          // Output8 values
      output1_data, output2_data,  // locally allocated output buffers
      output3_data, output4_data, output5_data, output6_data, output7_data,
      nullptr);
}

TF_LITE_MICRO_TESTS_END
