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
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

void TestSplitTwoOutputsFloat(
    const int* input_dims_data, const float* input_data,
    const int* axis_dims_data, const int32_t* axis_data,
    const int* output1_dims_data, const float* expected_output1_data,
    const int* output2_dims_data, const float* expected_output2_data,
    float* output1_data, float* output2_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInts(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInts(output2_dims_data);
  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 2;
  constexpr int axis_size = 1;
  constexpr int tensors_size = input_size + output_size + axis_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(axis_data, axis_dims), CreateTensor(input_data, input_dims),
      CreateTensor(output1_data, output1_dims),
      CreateTensor(output2_data, output2_dims)};

  // Currently only support constant axis tensor.
  tensors[0].allocation_type = kTfLiteMmapRo;
  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output1_dims_count; ++i) {
    output1_data[i] = 23;
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    output2_data[i] = 23;
  }

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {2, 2, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = tflite::ops::micro::Register_SPLIT();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output1_data[i], output1_data[i], 1e-5f);
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output2_data[i], output2_data[i], 1e-5f);
  }
}

void TestSplitFourOutputsFloat(
    const int* input_dims_data, const float* input_data,
    const int* axis_dims_data, const int32_t* axis_data,
    const int* output1_dims_data, const float* expected_output1_data,
    const int* output2_dims_data, const float* expected_output2_data,
    const int* output3_dims_data, const float* expected_output3_data,
    const int* output4_dims_data, const float* expected_output4_data,
    float* output1_data, float* output2_data, float* output3_data,
    float* output4_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInts(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInts(output2_dims_data);
  TfLiteIntArray* output3_dims = IntArrayFromInts(output3_dims_data);
  TfLiteIntArray* output4_dims = IntArrayFromInts(output4_dims_data);
  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);
  const int output3_dims_count = ElementCount(*output3_dims);
  const int output4_dims_count = ElementCount(*output4_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 4;
  constexpr int axis_size = 1;
  constexpr int tensors_size = input_size + output_size + axis_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(axis_data, axis_dims),
      CreateTensor(input_data, input_dims),
      CreateTensor(output1_data, output1_dims),
      CreateTensor(output2_data, output2_dims),
      CreateTensor(output3_data, output1_dims),
      CreateTensor(output4_data, output1_dims)};

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

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {4, 2, 3, 4, 5};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = tflite::ops::micro::Register_SPLIT();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, nullptr);

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
}

void TestSplitTwoOutputsQuantized(
    const int* input_dims_data, const uint8_t* input_data,
    const int* axis_dims_data, const int32_t* axis_data,
    const int* output1_dims_data, const uint8_t* expected_output1_data,
    const int* output2_dims_data, const uint8_t* expected_output2_data,
    uint8_t* output1_data, uint8_t* output2_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInts(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInts(output2_dims_data);
  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 2;
  constexpr int axis_size = 1;
  constexpr int tensors_size = input_size + output_size + axis_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(axis_data, axis_dims),
      CreateQuantizedTensor(input_data, input_dims, 0, 10),
      CreateQuantizedTensor(output1_data, output1_dims, 0, 10),
      CreateQuantizedTensor(output2_data, output2_dims, 0, 10)};

  // Currently only support constant axis tensor.
  tensors[0].allocation_type = kTfLiteMmapRo;

  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output1_dims_count; ++i) {
    output1_data[i] = 23;
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    output2_data[i] = 23;
  }

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {2, 2, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = tflite::ops::micro::Register_SPLIT();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output1_data[i], output1_data[i]);
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output2_data[i], output2_data[i]);
  }
}

void TestSplitTwoOutputsQuantized32(
    const int* input_dims_data, const int32_t* input_data,
    const int* axis_dims_data, const int32_t* axis_data,
    const int* output1_dims_data, const int32_t* expected_output1_data,
    const int* output2_dims_data, const int32_t* expected_output2_data,
    int32_t* output1_data, int32_t* output2_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInts(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInts(output2_dims_data);
  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 2;
  constexpr int axis_size = 1;
  constexpr int tensors_size = input_size + output_size + axis_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(axis_data, axis_dims), CreateTensor(input_data, input_dims),
      CreateTensor(output1_data, output1_dims),
      CreateTensor(output2_data, output2_dims)};

  // Currently only support constant axis tensor.
  tensors[0].allocation_type = kTfLiteMmapRo;

  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output1_dims_count; ++i) {
    output1_data[i] = 23;
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    output2_data[i] = 23;
  }

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {2, 2, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = tflite::ops::micro::Register_SPLIT();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output1_data[i], output1_data[i]);
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output2_data[i], output2_data[i]);
  }
}

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TwoSplitFourDimensionalAxisZero) {
  const int input_shape[] = {4, 2, 2, 2, 2};
  const float input_data[] = {1, 2,  3,  4,  5,  6,  7,  8,
                              9, 10, 11, 12, 13, 14, 15, 16};
  const int axis_shape[] = {1, 1};
  const int32_t axis_data[] = {0};
  const int output1_shape[] = {4, 1, 2, 2, 2};
  const float golden1[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const int output2_shape[] = {4, 1, 2, 2, 2};
  const float golden2[] = {9, 10, 11, 12, 13, 14, 15, 16};

  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsFloat(
      input_shape, input_data, axis_shape, axis_data, output1_shape, golden1,
      output2_shape, golden2, output1_data, output2_data);
}

TF_LITE_MICRO_TEST(TwoSplitFourDimensionalAxisOne) {
  const int input_shape[] = {4, 2, 2, 2, 2};
  const float input_data[] = {1, 2,  3,  4,  5,  6,  7,  8,
                              9, 10, 11, 12, 13, 14, 15, 16};
  const int axis_shape[] = {1, 1};
  const int32_t axis_data[] = {1};
  const int output1_shape[] = {4, 2, 1, 2, 2};
  const float golden1[] = {1, 2, 3, 4, 9, 10, 11, 12};
  const int output2_shape[] = {4, 2, 1, 2, 2};
  const float golden2[] = {5, 6, 7, 8, 13, 14, 15, 16};

  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsFloat(
      input_shape, input_data, axis_shape, axis_data, output1_shape, golden1,
      output2_shape, golden2, output1_data, output2_data);
}

TF_LITE_MICRO_TEST(TwoSplitFourDimensionalAxisTwo) {
  const int input_shape[] = {4, 2, 2, 2, 2};
  const float input_data[] = {1, 2,  3,  4,  5,  6,  7,  8,
                              9, 10, 11, 12, 13, 14, 15, 16};
  const int axis_shape[] = {1, 1};
  const int32_t axis_data[] = {2};
  const int output1_shape[] = {4, 2, 2, 1, 2};
  const float golden1[] = {1, 2, 5, 6, 9, 10, 13, 14};
  const int output2_shape[] = {4, 2, 2, 1, 2};
  const float golden2[] = {3, 4, 7, 8, 11, 12, 15, 16};

  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsFloat(
      input_shape, input_data, axis_shape, axis_data, output1_shape, golden1,
      output2_shape, golden2, output1_data, output2_data);
}

TF_LITE_MICRO_TEST(TwoSplitFourDimensionalAxisThree) {
  const int input_shape[] = {4, 2, 2, 2, 2};
  const float input_data[] = {1, 2,  3,  4,  5,  6,  7,  8,
                              9, 10, 11, 12, 13, 14, 15, 16};
  const int axis_shape[] = {1, 1};
  const int32_t axis_data[] = {3};
  const int output1_shape[] = {4, 2, 2, 2, 1};
  const float golden1[] = {1, 3, 5, 7, 9, 11, 13, 15};
  const int output2_shape[] = {4, 2, 2, 2, 1};
  const float golden2[] = {2, 4, 6, 8, 10, 12, 14, 16};

  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsFloat(
      input_shape, input_data, axis_shape, axis_data, output1_shape, golden1,
      output2_shape, golden2, output1_data, output2_data);
}

TF_LITE_MICRO_TEST(TwoSplitFourDimensionalNegativeAxis) {
  const int input_shape[] = {4, 2, 2, 2, 2};
  const float input_data[] = {1, 2,  3,  4,  5,  6,  7,  8,
                              9, 10, 11, 12, 13, 14, 15, 16};
  const int axis_shape[] = {1, 1};
  const int32_t axis_data[] = {-4};
  const int output1_shape[] = {4, 1, 2, 2, 2};
  const float golden1[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const int output2_shape[] = {4, 1, 2, 2, 2};
  const float golden2[] = {9, 10, 11, 12, 13, 14, 15, 16};

  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsFloat(
      input_shape, input_data, axis_shape, axis_data, output1_shape, golden1,
      output2_shape, golden2, output1_data, output2_data);
}

TF_LITE_MICRO_TEST(FourSplit) {
  const int input_shape[] = {1, 4};
  const float input_data[] = {1, 2, 3, 4};
  const int axis_shape[] = {1, 1};
  const int32_t axis_data[] = {0};
  const int output1_shape[] = {1, 1};
  const float golden1[] = {1};
  const int output2_shape[] = {1, 1};
  const float golden2[] = {2};
  const int output3_shape[] = {1, 1};
  const float golden3[] = {3};
  const int output4_shape[] = {1, 1};
  const float golden4[] = {4};

  constexpr int output1_dims_count = 1;
  constexpr int output2_dims_count = 1;
  constexpr int output3_dims_count = 1;
  constexpr int output4_dims_count = 1;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  float output3_data[output3_dims_count];
  float output4_data[output4_dims_count];
  tflite::testing::TestSplitFourOutputsFloat(
      input_shape, input_data, axis_shape, axis_data, output1_shape, golden1,
      output2_shape, golden2, output3_shape, golden3, output4_shape, golden4,
      output1_data, output2_data, output3_data, output4_data);
}

TF_LITE_MICRO_TEST(TwoSplitOneDimensional) {
  const int input_shape[] = {1, 2};
  const float input_data[] = {1, 2};
  const int axis_shape[] = {1, 1};
  const int32_t axis_data[] = {0};
  const int output1_shape[] = {1, 1};
  const float golden1[] = {1};
  const int output2_shape[] = {1, 1};
  const float golden2[] = {2};

  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsFloat(
      input_shape, input_data, axis_shape, axis_data, output1_shape, golden1,
      output2_shape, golden2, output1_data, output2_data);
}

TF_LITE_MICRO_TEST(TwoSplitFourDimensionalQuantized) {
  const int input_shape[] = {4, 2, 2, 2, 2};
  const uint8_t input_data[] = {1, 2,  3,  4,  5,  6,  7,  8,
                                9, 10, 11, 12, 13, 14, 15, 16};
  const int axis_shape[] = {1, 1};
  const int32_t axis_data[] = {1};
  const int output1_shape[] = {4, 2, 1, 2, 2};
  const uint8_t golden1[] = {1, 2, 3, 4, 9, 10, 11, 12};
  const int output2_shape[] = {4, 2, 1, 2, 2};
  const uint8_t golden2[] = {5, 6, 7, 8, 13, 14, 15, 16};

  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  uint8_t output1_data[output1_dims_count];
  uint8_t output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsQuantized(
      input_shape, input_data, axis_shape, axis_data, output1_shape, golden1,
      output2_shape, golden2, output1_data, output2_data);
}

TF_LITE_MICRO_TEST(TwoSplitFourDimensionalQuantized32) {
  const int input_shape[] = {4, 2, 2, 2, 2};
  const int32_t input_data[] = {1, 2,  3,  4,  5,  6,  7,  8,
                                9, 10, 11, 12, 13, 14, 15, 16};
  const int axis_shape[] = {1, 1};
  const int32_t axis_data[] = {1};
  const int output1_shape[] = {4, 2, 1, 2, 2};
  const int32_t golden1[] = {1, 2, 3, 4, 9, 10, 11, 12};
  const int output2_shape[] = {4, 2, 1, 2, 2};
  const int32_t golden2[] = {5, 6, 7, 8, 13, 14, 15, 16};

  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  int32_t output1_data[output1_dims_count];
  int32_t output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsQuantized32(
      input_shape, input_data, axis_shape, axis_data, output1_shape, golden1,
      output2_shape, golden2, output1_data, output2_data);
}

TF_LITE_MICRO_TESTS_END
