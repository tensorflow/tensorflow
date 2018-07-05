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
#include "tensorflow/contrib/lite/kernels/internal/tensor_utils.h"
#include <gmock/gmock.h>
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"

namespace tflite {
namespace tensor_utils {

TEST(uKernels, ClipTest) {
  constexpr int kVectorSize = 10;
  constexpr float kAbsLimit = 2.0;
  static float input[kVectorSize] = {0.0,  -0.5, 1.0,  -1.5, 2.0,
                                     -2.5, 3.0,  -3.5, 4.0,  -4.5};
  std::vector<float> output(kVectorSize);
  ClipVector(input, kVectorSize, kAbsLimit, output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear(
                  {0.0, -0.5, 1.0, -1.5, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0})));
}

TEST(uKernels, IsZeroTest) {
  constexpr int kVectorSize = 21;
  static float zeros[kVectorSize] = {0.0};
  EXPECT_TRUE(IsZeroVector(zeros, kVectorSize));

  static float nonzeros[kVectorSize] = {
      1e-6,  1e-7,  1e-8,  1e-9,  1e-10, 1e-11, 1e-12,
      1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19,
      1e-20, 1e-21, 1e-22, 1e-23, 1e-24, 1e-25, 1e-26};
  EXPECT_FALSE(IsZeroVector(nonzeros, kVectorSize));
}

TEST(uKernels, GeneratedIsZeroTest) {
  constexpr int kVectorSize = 39;
  std::vector<float> input(kVectorSize);
  ZeroVector(input.data(), kVectorSize);
  EXPECT_TRUE(IsZeroVector(input.data(), kVectorSize));
}

TEST(uKernels, SymmetricQuantizeFloatsTest) {
  constexpr int kVectorSize = 9;
  static float input[kVectorSize] = {-640, -635.0, -630, 10.0,  2.0,
                                     -5.0, -10.0,  0.0,  1000.0};

  int8 output[kVectorSize];
  float min, max, scaling_factor;
  SymmetricQuantizeFloats(input, kVectorSize, output, &min, &max,
                          &scaling_factor);

  EXPECT_EQ(min, -640);
  EXPECT_EQ(max, 1000);
  // EQ won't work due to fpoint.
  EXPECT_NEAR(scaling_factor, 1000 / 127.0, 1e-6);
  EXPECT_THAT(output,
              testing::ElementsAreArray({-81, -81, -80, 1, 0, -1, -1, 0, 127}));
}

TEST(uKernels, SymmetricQuantizeFloatsAllZerosTest) {
  constexpr int kVectorSize = 9;
  static float input[kVectorSize] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  int8 output[kVectorSize];
  float min, max, scaling_factor;
  SymmetricQuantizeFloats(input, kVectorSize, output, &min, &max,
                          &scaling_factor);

  EXPECT_EQ(min, 0);
  EXPECT_EQ(max, 0);
  EXPECT_EQ(scaling_factor, 1);
  EXPECT_THAT(output, testing::ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(uKernels, SymmetricQuantizeFloatsAllAlmostZeroTest) {
  constexpr int kVectorSize = 9;
  static float input[kVectorSize] = {-1e-5, 3e-5, -7e-6, -9e-5, 1e-6,
                                     4e-5,  9e-6, 2e-4,  0};

  int8 output[kVectorSize];
  float min, max, scaling_factor;
  SymmetricQuantizeFloats(input, kVectorSize, output, &min, &max,
                          &scaling_factor);

  EXPECT_NEAR(min, -9e-05, 1e-6);
  EXPECT_NEAR(max, 0.0002, 1e-6);
  EXPECT_NEAR(scaling_factor, 1.57e-6, 1e-6);
  EXPECT_THAT(output,
              testing::ElementsAreArray({-6, 19, -4, -57, 1, 25, 6, 127, 0}));
}

TEST(uKernels, MatrixBatchVectorMultiplyAccumulateTest) {
  constexpr int kRow = 3;
  constexpr int kCol = 4;
  constexpr int kBatch = 2;
  static float matrix[kRow * kCol] = {1.0,  2.0,  3.0,  4.0,   //
                                      -1.0, -2.0, -3.0, -4.0,  //
                                      1.0,  -2.0, 3.0,  -4.0};
  static float vector[kCol * kBatch] = {1.0, -1.0, 1.0, -1.0,  //
                                        2.0, -2.0, 2.0, -2.0};
  std::vector<float> output(kRow * kBatch);
  std::fill(output.begin(), output.end(), 3.0);
  MatrixBatchVectorMultiplyAccumulate(matrix, kRow, kCol, vector, kBatch,
                                      output.data(), /*result_stride=*/1);
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear({1., 5., 13.,  //
                                                       -1., 7., 23.})));

  std::vector<float> output_with_stride2(kRow * kBatch * 2);
  std::fill(output_with_stride2.begin(), output_with_stride2.end(), 3.0);
  MatrixBatchVectorMultiplyAccumulate(matrix, kRow, kCol, vector, kBatch,
                                      output_with_stride2.data(),
                                      /*result_stride=*/2);
  EXPECT_THAT(output_with_stride2,
              ElementsAreArray(ArrayFloatNear({1., 3., 5., 3., 13., 3.,  //
                                               -1., 3., 7., 3., 23., 3.})));
}

TEST(uKernels, MatrixBatchVectorMultiplyAccumulateSymmetricQuantizedTest) {
  // Note we use 29 columns as this exercises all the neon kernel: the
  // 16-block SIMD code, the 8-block postamble, and the leftover postamble.
  const int a_rows = 4, a_cols = 29;
  const int kWeightsPerUint32 = 4;
  const float a_float_data[] = {
      /* 1st row */
      1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12, 13.13,
      14.14, 15.15, 16.16, 17.17, 18.18, 19.19, 20.2, 21.21, 22.22, 23.23,
      24.24, 25.25, 26.26, 27.27, 28.28, 0,
      /* 2nd row */
      -1.1, -2.2, -3.3, -4.4, -5.5, -6.6, -7.7, -8.8, -9.9, -10.1, -11.11,
      -12.12, -13.13, -14.14, -15.15, -16.16, -17.17, -18.18, -19.19, -20.2,
      -21.21, -22.22, -23.23, -24.24, -25.25, -26.26, -27.27, -28.28, 0,
      /* 3rd row */
      1.1, -2.2, 3.3, -4.4, 5.5, -6.6, 7.7, -8.8, 9.9, -10.1, 11.11, -12.12,
      13.13, -14.14, 15.15, -16.16, 17.17, -18.18, 19.19, -20.2, 21.21, -22.22,
      23.23, -24.24, 25.25, -26.26, 27.27, -28.28, 0,
      /* 4th row */
      -1.1, 2.2, -3.3, 4.4, -5.5, 6.6, -7.7, 8.8, -9.9, 10.1, -11.11, 12.12,
      -13.13, 14.14, -15.15, 16.16, -17.17, 18.18, -19.19, 20.2, -21.21, 22.22,
      -23.23, 24.24, -25.25, 26.26, -27.27, 28.28, 0};

  int8* a_int8_data = reinterpret_cast<int8*>(
      aligned_malloc(a_rows * a_cols, kWeightsPerUint32));
  float a_min, a_max;
  float scaling_factor_a;
  SymmetricQuantizeFloats(a_float_data, a_rows * a_cols, a_int8_data, &a_min,
                          &a_max, &scaling_factor_a);
  const int8 expected_a_int8_data[] = {
      /* 1st row */
      5,
      10,
      15,
      20,
      25,
      30,
      35,
      40,
      44,
      45,
      50,
      54,
      59,
      64,
      68,
      73,
      77,
      82,
      86,
      91,
      95,
      100,
      104,
      109,
      113,
      118,
      122,
      127,
      0,
      /* 2nd row */
      -5,
      -10,
      -15,
      -20,
      -25,
      -30,
      -35,
      -40,
      -44,
      -45,
      -50,
      -54,
      -59,
      -64,
      -68,
      -73,
      -77,
      -82,
      -86,
      -91,
      -95,
      -100,
      -104,
      -109,
      -113,
      -118,
      -122,
      -127,
      0,
      /* 3rd row */
      5,
      -10,
      15,
      -20,
      25,
      -30,
      35,
      -40,
      44,
      -45,
      50,
      -54,
      59,
      -64,
      68,
      -73,
      77,
      -82,
      86,
      -91,
      95,
      -100,
      104,
      -109,
      113,
      -118,
      122,
      -127,
      0,
      /* 4th row */
      -5,
      10,
      -15,
      20,
      -25,
      30,
      -35,
      40,
      -44,
      45,
      -50,
      54,
      -59,
      64,
      -68,
      73,
      -77,
      82,
      -86,
      91,
      -95,
      100,
      -104,
      109,
      -113,
      118,
      -122,
      127,
      0,
  };
  for (int i = 0; i < a_rows * a_cols; ++i) {
    EXPECT_EQ(expected_a_int8_data[i], a_int8_data[i]);
  }

  const int b_rows = 29, b_cols = 1, batches = 2;
  const float b_float_data[] = {
      /* batch 1 */
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      /* batch 2 */
      2.5,
      -2.1,
      3.0,
      -1.3,
      1.3,
      -1.1,
      2.0,
      -1.7,
      1.9,
      -1.5,
      0.5,
      -0.7,
      0.8,
      -0.3,
      2.8,
      -2.8,
      1.1,
      -2.3,
      1.9,
      -1.9,
      2.1,
      -0.5,
      2.4,
      -0.1,
      1.0,
      -2.5,
      0.7,
      -1.9,
      0.2,
  };

  // Quantized values of B:
  int8 b_int8_data[b_rows * b_cols * batches];
  float b_min, b_max;
  float scaling_factor_b[batches];
  SymmetricQuantizeFloats(b_float_data, b_rows * b_cols, b_int8_data, &b_min,
                          &b_max, &scaling_factor_b[0]);
  SymmetricQuantizeFloats(&b_float_data[b_rows * b_cols], b_rows * b_cols,
                          &b_int8_data[b_rows * b_cols], &b_min, &b_max,
                          &scaling_factor_b[1]);

  const int8 expected_b_int8_data[] = {
      /* batch 1 */
      127,
      -127,
      127,
      -127,
      127,
      -127,
      127,
      -127,
      127,
      -127,
      127,
      -127,
      127,
      -127,
      127,
      -127,
      127,
      -127,
      127,
      -127,
      127,
      -127,
      127,
      -127,
      127,
      -127,
      127,
      -127,
      127,
      /* batch 2 */
      106,
      -89,
      127,
      -55,
      55,
      -47,
      85,
      -72,
      80,
      -64,
      21,
      -30,
      34,
      -13,
      119,
      -119,
      47,
      -97,
      80,
      -80,
      89,
      -21,
      102,
      -4,
      42,
      -106,
      30,
      -80,
      8,
  };
  for (int i = 0; i < b_rows * b_cols * batches; ++i) {
    EXPECT_EQ(expected_b_int8_data[i], b_int8_data[i]);
  }

  // Full float operation results in:
  // -13.69, 13.69, 414.11, -414.11
  // -6.325, 6.325, 631.263, -631.263
  float c_float_data[a_rows * b_cols * batches];
  for (int i = 0; i < a_rows * b_cols * batches; ++i) {
    c_float_data[i] = 0.0;
  }

  // Testing product.
  const float scaling_factor_c[2] = {
      scaling_factor_a * scaling_factor_b[0],
      scaling_factor_a * scaling_factor_b[1],
  };
  MatrixBatchVectorMultiplyAccumulate(a_int8_data, a_rows, a_cols, b_int8_data,
                                      scaling_factor_c, batches, c_float_data,
                                      /*result_stride=*/1);

  // Assert we obtain the expected recovered float values.
  const float expected_c_float_data[] = {
      -14.474, 14.474, 414.402, -414.402, -6.92228, 6.92228, 632.042, -632.042,
  };
  for (int i = 0; i < a_rows * b_cols * batches; ++i) {
    EXPECT_NEAR(expected_c_float_data[i], c_float_data[i], 0.001);
  }

  aligned_free(a_int8_data);
}

TEST(uKernels, VectorVectorCwiseProductTest) {
  constexpr int kVectorSize = 10;
  static float input1[kVectorSize] = {0.0,  -0.5, 1.0,  -1.5, 2.0,
                                      -2.5, 3.0,  -3.5, 4.0,  -4.5};
  static float input2[kVectorSize] = {0.1,  -0.1, 0.1,  -0.1, 0.1,
                                      -0.1, 0.1,  -0.1, 0.1,  -0.1};
  std::vector<float> output(kVectorSize);
  VectorVectorCwiseProduct(input1, input2, kVectorSize, output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear(
                  {0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45})));
}

TEST(uKernels, VectorVectorCwiseProductAccumulateTest) {
  constexpr int kVectorSize = 10;
  static float input1[kVectorSize] = {0.0,  -0.5, 1.0,  -1.5, 2.0,
                                      -2.5, 3.0,  -3.5, 4.0,  -4.5};
  static float input2[kVectorSize] = {0.1,  -0.1, 0.1,  -0.1, 0.1,
                                      -0.1, 0.1,  -0.1, 0.1,  -0.1};
  std::vector<float> output(kVectorSize);
  std::fill(output.begin(), output.end(), 1.0);
  VectorVectorCwiseProductAccumulate(input1, input2, kVectorSize,
                                     output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear(
                  {1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45})));
}

TEST(uKernels, VectorBatchVectorAssignTest) {
  constexpr int kVectorSize = 5;
  constexpr int kBatchSize = 3;
  static float input[kVectorSize] = {0.0, -0.5, 1.0, -1.5, 2.0};
  std::vector<float> output(kVectorSize * kBatchSize);
  VectorBatchVectorAssign(input, kVectorSize, kBatchSize, output.data());
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear(
                          {0.0, -0.5, 1.0, -1.5, 2.0, 0.0, -0.5, 1.0, -1.5, 2.0,
                           0.0, -0.5, 1.0, -1.5, 2.0})));
}

TEST(uKernels, ApplySigmoidToVectorTest) {
  constexpr int kVectorSize = 5;
  static float input[kVectorSize] = {0.0, -0.5, 1.0, -1.5, 2.0};
  std::vector<float> output(kVectorSize);
  ApplySigmoidToVector(input, kVectorSize, output.data());
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear(
                          {0.5, 0.377541, 0.731059, 0.182426, 0.880797})));
}

TEST(uKernels, ApplyActivationToVectorTest) {
  constexpr int kVectorSize = 5;
  static float input[kVectorSize] = {0.0, -0.5, 1.0, -1.5, 2.0};
  std::vector<float> output(kVectorSize);
  ApplyActivationToVector(input, kVectorSize, kTfLiteActRelu, output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear({0.0, 0.0, 1.0, 0.0, 2.0})));

  ApplyActivationToVector(input, kVectorSize, kTfLiteActTanh, output.data());
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear(
                          {0.0, -0.462117, 0.761594, -0.905148, 0.964028})));
}

TEST(uKernels, CopyVectorTest) {
  constexpr int kVectorSize = 5;
  static float input[kVectorSize] = {0.0, -0.5, 1.0, -1.5, 2.0};
  std::vector<float> output(kVectorSize);
  CopyVector(input, kVectorSize, output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear({0.0, -0.5, 1.0, -1.5, 2.0})));
}

TEST(uKernels, Sub1VectorTest) {
  constexpr int kVectorSize = 5;
  static float input[kVectorSize] = {0.0, -0.5, 1.0, -1.5, 2.0};
  std::vector<float> output(kVectorSize);
  Sub1Vector(input, kVectorSize, output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear({1.0, 1.5, 0.0, 2.5, -1.0})));
}

TEST(uKernels, ZeroVectorTest) {
  constexpr int kVectorSize = 5;
  std::vector<float> output(kVectorSize);
  ZeroVector(output.data(), kVectorSize);
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear({0.0, 0.0, 0.0, 0.0, 0.0})));
}

TEST(uKernels, BatchVectorBatchVectorDotProductTest) {
  constexpr int kVectorSize = 5;
  constexpr int kBatch = 2;
  static float input1[kVectorSize * kBatch] = {0.0,  -0.5, 1.0,  -1.5, 2.0,
                                               -2.5, 3.0,  -3.5, 4.0,  -4.5};
  static float input2[kVectorSize * kBatch] = {0.1,  -0.1, 0.1,  -0.1, 0.1,
                                               -0.1, 0.1,  -0.1, 0.1,  -0.1};
  std::vector<float> output(kBatch);
  BatchVectorBatchVectorDotProduct(input1, input2, kVectorSize, kBatch,
                                   output.data(), /*result_stride=*/1);
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear({0.5, 1.75})));
}

TEST(uKernels, VectorShiftLeftTest) {
  constexpr int kVectorSize = 5;
  static float input[kVectorSize] = {0.0, -0.5, 1.0, -1.5, 2.0};
  std::vector<float> result(kVectorSize);
  VectorShiftLeft(input, kVectorSize, 3.0);
  result.assign(input, input + kVectorSize);
  EXPECT_THAT(result,
              ElementsAreArray(ArrayFloatNear({-0.5, 1.0, -1.5, 2.0, 3.0})));
}

TEST(uKernels, ReductionSumVectorTest) {
  constexpr int kInputVectorSize = 10;
  constexpr int kOutputVectorSize1 = 5;
  constexpr int kReductionSize1 = 2;
  static float input[kInputVectorSize] = {0.0, -0.5, 1.0, -1.5, 2.0,
                                          0.0, -0.5, 1.0, 1.0,  2.0};
  std::vector<float> result1(kOutputVectorSize1);
  ReductionSumVector(input, result1.data(), kOutputVectorSize1,
                     kReductionSize1);
  EXPECT_THAT(result1,
              ElementsAreArray(ArrayFloatNear({-0.5, -0.5, 2.0, 0.5, 3.0})));

  constexpr int kOutputVectorSize2 = 2;
  constexpr int kReductionSize2 = 5;
  std::vector<float> result2(kOutputVectorSize2);
  ReductionSumVector(input, result2.data(), kOutputVectorSize2,
                     kReductionSize2);
  EXPECT_THAT(result2, ElementsAreArray(ArrayFloatNear({1.0, 3.5})));
}

}  // namespace tensor_utils
}  // namespace tflite
