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
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {

void TestSplitVThreeOutputsFloat(
    const int* input_dims_data, const float* input_data,
    const int* axis_dims_data, const int* axis_data, const int* split_dims_data,
    const int* split_data, const int* output1_dims_data,
    const float* expected_output1_data, const int* output2_dims_data,
    const float* expected_output2_data, const int* output3_dims_data,
    const float* expected_output3_data, float* output1_data,
    float* output2_data, float* output3_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  TfLiteIntArray* split_dims = IntArrayFromInts(split_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInts(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInts(output2_dims_data);
  TfLiteIntArray* output3_dims = IntArrayFromInts(output3_dims_data);

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

      CreateFloatTensor(input_data, input_dims),
      CreateQuantized32Tensor(split_data, split_dims, 1.0),
      CreateQuantized32Tensor(axis_data, axis_dims, 1.0),

      // outputs
      CreateFloatTensor(output1_data, output1_dims),
      CreateFloatTensor(output2_data, output2_dims),
      CreateFloatTensor(output3_data, output3_dims)

  };
  tensors[2].allocation_type = kTfLiteMmapRo;
  tensors[1].allocation_type = kTfLiteMmapRo;

  void* user_data = nullptr;
  TfLiteSplitVParams builtin;
  builtin.num_splits = 3;
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {3, 3, 4, 5};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  TfLiteIntArray* temporaries_array = IntArrayFromInts({0});

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;

  node.user_data = nullptr;
  node.builtin_data = reinterpret_cast<void*>(&builtin);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_SPLIT_V();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, nullptr, micro_test::reporter);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output1_data[i], output1_data[i], 1e-5f);
  }
  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output2_data[i], output2_data[i], 1e-5f);
  }
  for (int i = 0; i < output3_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output3_data[i], output3_data[i], 1e-5f);
  }
}

void TestSplitVTwoOutputsFloat(
    const int* input_dims_data, const float* input_data,
    const int* axis_dims_data, const int* axis_data, const int* split_dims_data,
    const int* size_splits_data, const int* output1_dims_data,
    const float* expected_output1_data, const int* output2_dims_data,
    const float* expected_output2_data, float* output1_data,
    float* output2_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInts(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInts(output2_dims_data);
  TfLiteIntArray* size_splits_dims = IntArrayFromInts(split_dims_data);

  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 2;
  constexpr int axis_size = 1;
  constexpr int split_size = 1;
  constexpr int tensors_size =
      input_size + output_size + axis_size + split_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims),
      CreateQuantized32Tensor(size_splits_data, size_splits_dims, 1.0),
      CreateQuantized32Tensor(axis_data, axis_dims, 1.0),
      CreateFloatTensor(output1_data, output1_dims),
      CreateFloatTensor(output2_data, output2_dims)};

  // Currently only support constant axis tensor.

  tensors[1].allocation_type = kTfLiteMmapRo;
  tensors[2].allocation_type = kTfLiteMmapRo;

  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output1_dims_count; ++i) {
    output1_data[i] = 23;
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    output2_data[i] = 23;
  }

  TfLiteSplitVParams builtin_data;
  builtin_data.num_splits = 2;

  void* user_data = nullptr;

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {2, 3, 4};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  TfLiteIntArray* temporaries_array = IntArrayFromInts({0});

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  // node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_SPLIT_V();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, nullptr, micro_test::reporter);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output1_data[i], output1_data[i], 1e-5f);
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output2_data[i], output2_data[i], 1e-5f);
  }
}
void TestSplitVEightOutputsFloat(

    const int* input_dims_data, const float* input_data,
    const int* axis_dims_data, const int* axis_data, const int* split_dims_data,
    const int* size_splits_data, const int* output1_dims_data,
    const float* expected_output1_data, const int* output2_dims_data,
    const float* expected_output2_data, const int* output3_dims_data,
    const float* expected_output3_data, const int* output4_dims_data,
    const float* expected_output4_data, const int* output5_dims_data,
    const float* expected_output5_data, const int* output6_dims_data,
    const float* expected_output6_data, const int* output7_dims_data,
    const float* expected_output7_data, const int* output8_dims_data,
    const float* expected_output8_data,

    float* output1_data, float* output2_data, float* output3_data,
    float* output4_data, float* output5_data, float* output6_data,
    float* output7_data, float* output8_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInts(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInts(output2_dims_data);
  TfLiteIntArray* output3_dims = IntArrayFromInts(output3_dims_data);
  TfLiteIntArray* output4_dims = IntArrayFromInts(output4_dims_data);
  TfLiteIntArray* output5_dims = IntArrayFromInts(output5_dims_data);
  TfLiteIntArray* output6_dims = IntArrayFromInts(output6_dims_data);
  TfLiteIntArray* output7_dims = IntArrayFromInts(output7_dims_data);
  TfLiteIntArray* output8_dims = IntArrayFromInts(output8_dims_data);

  TfLiteIntArray* size_splits_dims = IntArrayFromInts(split_dims_data);

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

      CreateFloatTensor(input_data, input_dims),
      CreateQuantized32Tensor(size_splits_data, size_splits_dims, 1.0),
      CreateQuantized32Tensor(axis_data, axis_dims, 1.0),
      CreateFloatTensor(output1_data, output1_dims),
      CreateFloatTensor(output2_data, output2_dims),
      CreateFloatTensor(output3_data, output3_dims),
      CreateFloatTensor(output4_data, output4_dims),
      CreateFloatTensor(output5_data, output5_dims),
      CreateFloatTensor(output6_data, output6_dims),
      CreateFloatTensor(output7_data, output7_dims),
      CreateFloatTensor(output8_data, output8_dims)};

  // Currently only support constant axis tensor.
  tensors[1].allocation_type = kTfLiteMmapRo;
  tensors[2].allocation_type = kTfLiteMmapRo;

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

  TfLiteSplitVParams builtin_data;
  builtin_data.num_splits = 8;

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {8, 3, 4, 5, 6, 7, 8, 9, 10};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  TfLiteIntArray* temporaries_array = IntArrayFromInts({0});
  void* user_data = nullptr;
  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;

  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_SPLIT_V();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, nullptr, micro_test::reporter);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output1_data[i], output1_data[i], 1e-5f);
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output2_data[i], output2_data[i], 1e-5f);
  }
  for (int i = 0; i < output3_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output3_data[i], output3_data[i], 1e-5f);
  }

  for (int i = 0; i < output4_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output4_data[i], output4_data[i], 1e-5f);
  }
  for (int i = 0; i < output5_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output5_data[i], output5_data[i], 1e-5f);
  }

  for (int i = 0; i < output6_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output6_data[i], output6_data[i], 1e-5f);
  }
  for (int i = 0; i < output7_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output7_data[i], output7_data[i], 1e-5f);
  }

  for (int i = 0; i < output8_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output8_data[i], output8_data[i], 1e-5f);
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
  int input_shape[] = {2, 4, 3};
  float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int axis_shape[] = {1, 1};
  int axis_values[] = {0};
  int split_shape[] = {1, 3};
  int split_values[] = {1, 1, 2};
  int output1_shape[] = {2, 1, 3};
  float output1_values[] = {1, 2, 3};
  int output2_shape[] = {2, 1, 3};
  float output2_values[] = {4, 5, 6};
  int output3_shape[] = {2, 2, 3};
  float output3_values[] = {7, 8, 9, 10, 11, 12};
  tflite::testing::TestSplitVThreeOutputsFloat(
      input_shape, input_values, axis_shape, axis_values, split_shape,
      split_values, output1_shape, output1_values, output2_shape,
      output2_values, output3_shape, output3_values, output1_data, output2_data,
      output3_data);
}

TF_LITE_MICRO_TEST(SPLIT_V_FourDimensionalFloatAxis0) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];

  int input_shape[] = {4, 2, 2, 2, 2};
  float input_values[] = {1, 2,  3,  4,  5,  6,  7,  8,
                          9, 10, 11, 12, 13, 14, 15, 16};
  int axis_shape[] = {1, 1};
  int axis_value[] = {0};
  int split_size_shape[] = {1, 2};
  int split_data[] = {1, 1};
  int output1_shape[] = {4, 1, 2, 2, 2};
  float output1_values[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int output2_shape[] = {4, 1, 2, 2, 2};
  float output2_values[] = {9, 10, 11, 12, 13, 14, 15, 16};
  tflite::testing::TestSplitVTwoOutputsFloat(
      input_shape, input_values, axis_shape, axis_value, split_size_shape,
      split_data, output1_shape, output1_values, output2_shape, output2_values,
      output1_data, output2_data);
}

TF_LITE_MICRO_TEST(SPLIT_V_FourDimensionalFloatAxis1) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];

  int input_shape[] = {4, 2, 2, 2, 2};
  float input_values[] = {1, 2,  3,  4,  5,  6,  7,  8,
                          9, 10, 11, 12, 13, 14, 15, 16};
  int axis_shape[] = {1, 1};
  int axis_value[] = {1};
  int split_size_shape[] = {1, 2};
  int split_data[] = {1, 1};
  int output1_shape[] = {4, 2, 1, 2, 2};
  float output1_values[] = {1, 2, 3, 4, 9, 10, 11, 12};
  int output2_shape[] = {4, 2, 1, 2, 2};
  float output2_values[] = {5, 6, 7, 8, 13, 14, 15, 16};
  tflite::testing::TestSplitVTwoOutputsFloat(
      input_shape, input_values, axis_shape, axis_value, split_size_shape,
      split_data, output1_shape, output1_values, output2_shape, output2_values,
      output1_data, output2_data);
}

TF_LITE_MICRO_TEST(SPLIT_VFourDimensionalFloatAxis2) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];

  int input_shape[] = {4, 2, 2, 2, 2};
  float input_values[] = {1, 2,  3,  4,  5,  6,  7,  8,
                          9, 10, 11, 12, 13, 14, 15, 16};
  int axis_shape[] = {1, 1};
  int axis_value[] = {2};
  int split_size_shape[] = {1, 2};
  int split_data[] = {1, 1};
  int output1_shape[] = {4, 2, 2, 1, 2};
  float output1_values[] = {1, 2, 5, 6, 9, 10, 13, 14};
  int output2_shape[] = {4, 2, 2, 1, 2};
  float output2_values[] = {3, 4, 7, 8, 11, 12, 15, 16};
  tflite::testing::TestSplitVTwoOutputsFloat(
      input_shape, input_values, axis_shape, axis_value, split_size_shape,
      split_data, output1_shape, output1_values, output2_shape, output2_values,
      output1_data, output2_data);
}

TF_LITE_MICRO_TEST(SPLIT_V_FourDimensionalFloatAxis3) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  int input_shape[] = {4, 2, 2, 2, 2};
  float input_values[] = {1, 2,  3,  4,  5,  6,  7,  8,
                          9, 10, 11, 12, 13, 14, 15, 16};
  int axis_shape[] = {1, 1};
  int axis_value[] = {3};
  int split_size_shape[] = {1, 2};
  int split_data[] = {1, 1};
  int output1_shape[] = {4, 2, 2, 2, 1};
  float output1_values[] = {1, 3, 5, 7, 9, 11, 13, 15};
  int output2_shape[] = {4, 2, 2, 2, 1};
  float output2_values[] = {2, 4, 6, 8, 10, 12, 14, 16};
  tflite::testing::TestSplitVTwoOutputsFloat(
      input_shape, input_values, axis_shape, axis_value, split_size_shape,
      split_data, output1_shape, output1_values, output2_shape, output2_values,
      output1_data, output2_data);
}

TF_LITE_MICRO_TEST(SPLIT_V_FourDimensionalFloatNegativeAxis) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];

  int input_shape[] = {4, 2, 2, 2, 2};
  float input_values[] = {1, 2,  3,  4,  5,  6,  7,  8,
                          9, 10, 11, 12, 13, 14, 15, 16};
  int axis_shape[] = {1, 1};
  int axis_value[] = {-4};
  int split_size_shape[] = {1, 2};
  int split_data[] = {1, 1};
  int output1_shape[] = {4, 1, 2, 2, 2};
  float output1_values[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int output2_shape[] = {4, 1, 2, 2, 2};
  float output2_values[] = {9, 10, 11, 12, 13, 14, 15, 16};
  tflite::testing::TestSplitVTwoOutputsFloat(
      input_shape, input_values, axis_shape, axis_value, split_size_shape,
      split_data, output1_shape, output1_values, output2_shape, output2_values,
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
  int input_shape[] = {1, 8};
  float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int axis_shape[] = {1, 1};
  int axis_value[] = {0};
  int split_size_shape[] = {1, 8};
  int split[] = {1, 1, 1, 1, 1, 1, 1, 1};
  int output1_shape[] = {1, 1};
  float output1_values[] = {1};
  int output2_shape[] = {1, 1};
  float output2_values[] = {2};

  int output3_shape[] = {1, 1};
  float output3_values[] = {3};
  int output4_shape[] = {1, 1};
  float output4_values[] = {4};

  int output5_shape[] = {1, 1};
  float output5_values[] = {5};
  int output6_shape[] = {1, 1};
  float output6_values[] = {6};

  int output7_shape[] = {1, 1};
  float output7_values[] = {7};
  int output8_shape[] = {1, 1};
  float output8_values[] = {8};
  tflite::testing::TestSplitVEightOutputsFloat(
      input_shape, input_values, axis_shape, axis_value, split_size_shape,
      split, output1_shape, output1_values, output2_shape, output2_values,
      output3_shape, output3_values, output4_shape, output4_values,
      output5_shape, output5_values, output6_shape, output6_values,
      output7_shape, output7_values, output8_shape, output8_values,
      output1_data,
      output2_data,  // locally allocated output buffers
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

  int input_shape[] = {1, 8};
  float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int axis_shape[] = {1, 1};
  int axis_value[] = {0};
  int split_size_shape[] = {1, 8};
  int split[] = {1, 1, 1, 1, 1, 1, 2, -1};
  int output1_shape[] = {1, 1};
  float output1_values[] = {1};
  int output2_shape[] = {1, 1};
  float output2_values[] = {2};

  int output3_shape[] = {1, 1};
  float output3_values[] = {3};
  int output4_shape[] = {1, 1};
  float output4_values[] = {4};

  int output5_shape[] = {1, 1};
  float output5_values[] = {5};
  int output6_shape[] = {1, 1};
  float output6_values[] = {6};

  int output7_shape[] = {1, 2};
  float output7_values[] = {7, 8};
  int output8_shape[] = {1, 0};
  float output8_values[1] = {};
  tflite::testing::TestSplitVEightOutputsFloat(
      input_shape, input_values, axis_shape, axis_value, split_size_shape,
      split, output1_shape, output1_values, output2_shape, output2_values,
      output3_shape, output3_values, output4_shape, output4_values,
      output5_shape, output5_values, output6_shape, output6_values,
      output7_shape, output7_values, output8_shape, output8_values,
      output1_data,
      output2_data,  // locally allocated output buffers
      output3_data, output4_data, output5_data, output6_data, output7_data,
      nullptr);
}

TF_LITE_MICRO_TESTS_END
