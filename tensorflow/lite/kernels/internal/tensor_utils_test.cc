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
#include "tensorflow/lite/kernels/internal/tensor_utils.h"

#include <gmock/gmock.h>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/test_util.h"

#ifdef DOTPROD_BENCHMARKS
#include "testing/base/public/benchmark.h"
#endif  // DOTPROD_BENCHMARKS

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

TEST(uKernels, VectorScalarMultiply) {
  constexpr int kVectorSize = 29;
  static int8_t input[kVectorSize];
  for (int i = 0; i < 29; ++i) {
    input[i] = static_cast<int8_t>(i - 14);
  }
  const float scale = 0.1f;
  std::vector<float> output(kVectorSize, 0.0f);
  VectorScalarMultiply(input, kVectorSize, scale, output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear(
                  {-1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5,
                   -0.4, -0.3, -0.2, -0.1, 0,    0.1,  0.2,  0.3,  0.4,  0.5,
                   0.6,  0.7,  0.8,  0.9,  1.0,  1.1,  1.2,  1.3,  1.4})));
}

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

// Test if a float array if full of zero values.
TEST(uKernels, IsZeroFloatTest) {
  // Single NEON vector (= 4 floats)
  {
    const float four_zeros[4] = {0, 0, 0, 0};
    EXPECT_TRUE(IsZeroVector(four_zeros, ARRAY_SIZE(four_zeros)));
  }
  {
    const float four_nonzeros[4] = {1, 2, 3, 4};
    EXPECT_FALSE(IsZeroVector(four_nonzeros, ARRAY_SIZE(four_nonzeros)));
  }
  // Multiple NEON vectors
  {
    const float eight_zeros[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_TRUE(IsZeroVector(eight_zeros, ARRAY_SIZE(eight_zeros)));
  }
  {
    const float eight_nonzeros[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_FALSE(IsZeroVector(eight_nonzeros, ARRAY_SIZE(eight_nonzeros)));
  }
  {
    const float multiple_four_mixed1[8] = {0, 0, 0, 0, 5, 6, 7, 8};
    EXPECT_FALSE(
        IsZeroVector(multiple_four_mixed1, ARRAY_SIZE(multiple_four_mixed1)));
  }
  {
    const float multiple_four_mixed2[8] = {1, 2, 3, 4, 0, 0, 0, 0};
    EXPECT_FALSE(
        IsZeroVector(multiple_four_mixed2, ARRAY_SIZE(multiple_four_mixed2)));
  }
  // less than one NEON vector
  {
    const float three_zeros[3] = {0, 0, 0};
    EXPECT_TRUE(IsZeroVector(three_zeros, ARRAY_SIZE(three_zeros)));
  }
  {
    const float three_nonzeros[3] = {1, 2, 3};
    EXPECT_FALSE(IsZeroVector(three_nonzeros, ARRAY_SIZE(three_nonzeros)));
  }
  {
    const float three_mixed[3] = {1, 0, 3};
    EXPECT_FALSE(IsZeroVector(three_mixed, ARRAY_SIZE(three_mixed)));
  }
  // Postamble after NEON vectors
  {
    const float seven_zeros[7] = {0, 0, 0, 0, 0, 0, 0};
    EXPECT_TRUE(IsZeroVector(seven_zeros, ARRAY_SIZE(seven_zeros)));
  }
  {
    const float seven_nonzeros[7] = {1, 2, 3, 4, 5, 6, 7};
    EXPECT_FALSE(IsZeroVector(seven_nonzeros, ARRAY_SIZE(seven_nonzeros)));
  }
  {
    const float nonzeros_after_zeros[7] = {0, 0, 0, 0, 5, 6, 7};
    EXPECT_FALSE(
        IsZeroVector(nonzeros_after_zeros, ARRAY_SIZE(nonzeros_after_zeros)));
  }
}

// Test if an int8 array if full of zero values.
TEST(uKernels, IsZeroInt8Test) {
  // Single NEON vector (= 16x int8_t)
  {
    const int8_t sixteen_zeros[16] = {0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_TRUE(IsZeroVector(sixteen_zeros, ARRAY_SIZE(sixteen_zeros)));
  }
  {
    const int8_t sixteen_nonzeros[16] = {1, 2,  3,  4,  5,  6,  7,  8,
                                         9, 10, 11, 12, 13, 14, 15, 16};
    EXPECT_FALSE(IsZeroVector(sixteen_nonzeros, ARRAY_SIZE(sixteen_nonzeros)));
  }
  // Multiple NEON vectors
  {
    const int8_t thritytwo_zeros[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_TRUE(IsZeroVector(thritytwo_zeros, ARRAY_SIZE(thritytwo_zeros)));
  }
  {
    const int8_t thritytwo_nonzeros[32] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    EXPECT_FALSE(
        IsZeroVector(thritytwo_nonzeros, ARRAY_SIZE(thritytwo_nonzeros)));
  }
  {
    const int8_t thritytwo_mixed1[32] = {1,  2,  3,  4,  5,  6, 7, 8, 9, 10, 11,
                                         12, 13, 14, 15, 16, 0, 0, 0, 0, 0,  0,
                                         0,  0,  0,  0,  0,  0, 0, 0, 0, 0};
    EXPECT_FALSE(IsZeroVector(thritytwo_mixed1, ARRAY_SIZE(thritytwo_mixed1)));
  }
  {
    const int8_t thritytwo_mixed2[32] = {0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0,
                                         0, 0, 0, 0,  0,  1,  2,  3,  4,  5, 6,
                                         7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    EXPECT_FALSE(IsZeroVector(thritytwo_mixed2, ARRAY_SIZE(thritytwo_mixed2)));
  }
  // less than one NEON vector
  {
    const int8_t fifteen_zeros[15] = {0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0};
    EXPECT_TRUE(IsZeroVector(fifteen_zeros, ARRAY_SIZE(fifteen_zeros)));
  }
  {
    const int8_t fifteen_nonzeros[15] = {1, 2,  3,  4,  5,  6,  7, 8,
                                         9, 10, 11, 12, 13, 14, 15};
    EXPECT_FALSE(IsZeroVector(fifteen_nonzeros, ARRAY_SIZE(fifteen_nonzeros)));
  }
  {
    const int8_t fifteen_mixed[15] = {1, 0, 3,  0, 5,  0, 7, 0,
                                      9, 0, 11, 0, 13, 0, 15};
    EXPECT_FALSE(IsZeroVector(fifteen_mixed, ARRAY_SIZE(fifteen_mixed)));
  }
  // Postamble after NEON vectors
  {
    const int8_t seventeen_zeros[17] = {0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0};
    EXPECT_TRUE(IsZeroVector(seventeen_zeros, ARRAY_SIZE(seventeen_zeros)));
  }
  {
    const int8_t seventeen_nonzeros[17] = {1,  2,  3,  4,  5,  6,  7,  8, 9,
                                           10, 11, 12, 13, 14, 15, 16, 17};
    EXPECT_FALSE(
        IsZeroVector(seventeen_nonzeros, ARRAY_SIZE(seventeen_nonzeros)));
  }
  {
    const int8_t nonzeros_after_zeros[17] = {0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 17};
    EXPECT_FALSE(
        IsZeroVector(nonzeros_after_zeros, ARRAY_SIZE(nonzeros_after_zeros)));
  }
}

#undef ARRAY_SIZE

TEST(uKernels, SymmetricQuantizeFloatsTest) {
  constexpr int kVectorSize = 9;
  static float input[kVectorSize] = {-640, -635.0, -630, 10.0,  2.0,
                                     -5.0, -10.0,  0.0,  1000.0};

  int8_t output[kVectorSize];
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

  int8_t output[kVectorSize];
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

  int8_t output[kVectorSize];
  float min, max, scaling_factor;
  SymmetricQuantizeFloats(input, kVectorSize, output, &min, &max,
                          &scaling_factor);

  EXPECT_NEAR(min, -9e-05, 1e-6);
  EXPECT_NEAR(max, 0.0002, 1e-6);
  EXPECT_NEAR(scaling_factor, 1.57e-6, 1e-6);
  EXPECT_THAT(output,
              testing::ElementsAreArray({-6, 19, -4, -57, 1, 25, 6, 127, 0}));
}

TEST(uKernels, AsymmetricQuantizeFloatsTest) {
  constexpr int kVectorSize = 9;
  static float input[kVectorSize] = {-640, -635.0, -630, 10.0,  2.0,
                                     -5.0, -10.0,  0.0,  1000.0};
  int8_t output[kVectorSize];
  double min = -640.0;
  double max = 1000.0;
  QuantizationParams quantization_params =
      ChooseQuantizationParams<int8_t>(min, max);
  float scale = quantization_params.scale;
  int32_t offset = quantization_params.zero_point;
  float test_scale;
  int32_t test_offset;
  AsymmetricQuantizeFloats(input, kVectorSize, output, &test_scale,
                           &test_offset);
  // EQ won't work due to fpoint.
  EXPECT_NEAR(test_scale, scale, 1e-6);
  EXPECT_EQ(test_offset, offset);
  EXPECT_THAT(output, testing::ElementsAreArray(
                          {-128, -127, -126, -26, -28, -29, -30, -28, 127}));
}

TEST(uKernels, AsymmetricQuantizeFloatsAllZerosTest) {
  constexpr int kVectorSize = 9;
  static float input[kVectorSize] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int8_t output[kVectorSize];
  float test_scale;
  int32_t test_offset;
  AsymmetricQuantizeFloats(input, kVectorSize, output, &test_scale,
                           &test_offset);
  EXPECT_EQ(test_scale, 1);
  EXPECT_EQ(test_offset, 0);
  EXPECT_THAT(output, testing::ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(uKernels, AsymmetricQuantizeFloatsZeroRangeTest) {
  constexpr int kVectorSize = 9;
  static float input[kVectorSize] = {2000, 2000, 2000, 2000, 2000,
                                     2000, 2000, 2000, 2000};
  int8_t output[kVectorSize];
  double min = 0;
  double max = 2000;
  QuantizationParams quantization_params =
      ChooseQuantizationParams<int8_t>(min, max);
  int32_t offset = quantization_params.zero_point;
  float scale = quantization_params.scale;
  float test_scale;
  int32_t test_offset;
  AsymmetricQuantizeFloats(input, kVectorSize, output, &test_scale,
                           &test_offset);
  EXPECT_NEAR(test_scale, scale, 1e-6);
  EXPECT_EQ(test_offset, offset);
  EXPECT_THAT(output, testing::ElementsAreArray(
                          {127, 127, 127, 127, 127, 127, 127, 127, 127}));
}

TEST(uKernels, AsymmetricQuantizeFloatsAllAlmostZeroTest) {
  constexpr int kVectorSize = 9;
  static float input[kVectorSize] = {-1e-5, 3e-5, -7e-6, -9e-5, 1e-6,
                                     4e-5,  9e-6, 2e-4,  0};
  int8_t output[kVectorSize];
  double min = -9e-05;
  double max = 0.0002;
  QuantizationParams quantization_params =
      ChooseQuantizationParams<int8_t>(min, max);
  int32_t offset = quantization_params.zero_point;
  float scale = quantization_params.scale;
  float test_scale;
  int32_t test_offset;
  AsymmetricQuantizeFloats(input, kVectorSize, output, &test_scale,
                           &test_offset);
  EXPECT_NEAR(test_scale, scale, 1e-6);
  EXPECT_EQ(test_offset, offset);
  EXPECT_THAT(output, testing::ElementsAreArray(
                          {-58, -23, -55, -128, -48, -14, -41, 127, -49}));
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
                                      output.data());
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear({1., 5., 13.,  //
                                                       -1., 7., 23.})));
}

// Quantized matmul with 2 * 30 input and 9 * 30 matrix.
TEST(uKernels, QuantMatrixBatchVectorMultiplyAccumulate8x8_16Test) {
  CpuBackendContext context;
  const std::vector<int8_t> input = {
      4,   -41, 5,   -41, 22,  17, -30, 24,  13,  -47, 18, 9,   -11, -30, 16,
      -47, 12,  36,  -20, 27,  -3, 0,   -51, -31, 3,   -8, -38, 43,  23,  12,
      11,  -23, -26, 23,  14,  -9, -44, 22,  21,  -30, 3,  -47, -26, -21, -24,
      -44, 34,  -11, -23, -28, 26, -38, 19,  35,  9,   23, 6,   -42, -25, 28,
  };
  const std::vector<int32_t> input_zeropoint_times_weights = {
      -620, -170, -395, 715, -1220, -1080, 1130, -260, -470,
  };
  const std::vector<int8_t> input_to_gate_weights = {
      -10, -4,  -8,  16,  4,   -16, -1,  11,  1,   2,   -25, 19,  7,   9,   2,
      -24, -2,  10,  -7,  7,   -5,  -2,  3,   4,   3,   -4,  -7,  -11, -13, -18,
      11,  10,  12,  -9,  17,  -15, -5,  20,  -6,  -11, 2,   -6,  -18, 15,  4,
      4,   -9,  -2,  -3,  -9,  -13, 17,  -21, 5,   3,   -12, 0,   -4,  9,   -5,
      10,  -2,  8,   1,   -10, -6,  1,   -9,  10,  11,  -1,  -5,  4,   -7,  -4,
      -4,  4,   12,  -7,  -5,  -9,  -19, 6,   -4,  12,  -17, -22, 0,   9,   -4,
      -5,  5,   -8,  8,   3,   15,  -18, -18, 5,   3,   -12, 5,   -10, 7,   7,
      -9,  17,  2,   -11, -25, 3,   19,  -6,  7,   1,   7,   5,   -3,  11,  3,
      0,   -8,  8,   -2,  -2,  -12, 14,  -5,  7,   8,   16,  20,  -16, -5,  -5,
      1,   -10, -6,  14,  10,  -12, 10,  -6,  5,   0,   3,   8,   -9,  -13, -2,
      4,   4,   -16, -17, -9,  16,  -5,  14,  -9,  -5,  -12, 0,   17,  6,   -1,
      16,  -20, 1,   -11, -1,  -10, -21, 13,  4,   -12, -7,  0,   -14, -6,  3,
      -4,  6,   -18, -3,  -1,  14,  -8,  -6,  -15, 5,   12,  -3,  -10, 4,   6,
      -5,  -20, 0,   3,   -3,  -7,  1,   2,   -10, 7,   -3,  6,   1,   -12, 6,
      4,   -12, 2,   6,   -20, 0,   5,   23,  15,  14,  9,   8,   20,  -2,  9,
      -8,  -8,  -7,  -4,  -8,  -9,  7,   -12, -2,  2,   1,   -14, 31,  4,   -14,
      3,   10,  -18, -17, -1,  18,  1,   12,  0,   7,   -3,  -5,  8,   -9,  18,
      17,  7,   -15, 3,   20,  4,   -8,  16,  6,   -3,  -3,  9,   -4,  -6,  4,
  };
  const int32_t multiplier = 2080364544;
  const int32_t shift = -2;

  std::vector<int32_t> scratch(2 * 9, 0);
  std::vector<int16_t> output = {10, 2, 33, 4, 5,  6,  65, 4,  3,
                                 52, 1, 2,  8, -1, -2, 11, 17, -18};
  MatrixBatchVectorMultiplyAccumulate(
      input.data(), input_zeropoint_times_weights.data(),
      input_to_gate_weights.data(), multiplier, shift,
      /*n_batch=*/2, /*n_input=*/30, /*n_output=*/9, /*output_zp=*/0,
      scratch.data(), output.data(), &context);
  const std::vector<int16_t> expected_output = {
      -210, 331,  153, 139, -570, -657, 258, 515,  -495,
      91,   -243, -73, 603, -744, -269, 169, -748, -174,
  };

  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

TEST(uKernels, HybridMatrixBatchVectorMultiplyAccumulate8x8_16Test) {
  CpuBackendContext context;
  const std::vector<int8_t> input = {
      4,   -41, 5,   -41, 22,  17,  -30, 24,  13,  -47, 18,  9,   -11, -30, 16,
      1,   -47, 12,  36,  -20, 27,  -3,  0,   -51, -31, 3,   -8,  -38, 43,  23,
      12,  1,   11,  -23, -26, 23,  14,  -9,  -44, 22,  21,  -30, 3,   -47, -26,
      -21, -24, 1,   -44, 34,  -11, -23, -28, 26,  -38, 19,  35,  9,   23,  6,
      -42, -25, 28,  1,   4,   -41, 5,   -41, 22,  17,  -30, 24,  13,  -47, 18,
      9,   -11, -30, 16,  1,   -47, 12,  36,  -20, 27,  -3,  0,   -51, -31, 3,
      -8,  -38, 43,  23,  12,  1,   11,  -23, -26, 23,  14,  -9,  -44, 22,  21,
      -30, 3,   -47, -26, -21, -24, 1,   -44, 34,  -11, -23, -28, 26,  -38, 19,
      35,  9,   23,  6,   -42, -25, 28,  1,
  };
  const std::vector<int32_t> input_offsets = {1, 1, 1, 1};

  const std::vector<float> scaling_factors = {
      1.0,
      1.0,
      1.0,
      1.0,
  };

  const std::vector<int8_t> input_to_gate_weights = {
      -10, -4,  -8,  16,  4,  -16, -1,  11,  1,   2,   -25, 19,  7,   9,   2,
      1,   -24, -2,  10,  -7, 7,   -5,  -2,  3,   4,   3,   -4,  -7,  -11, -13,
      -18, 2,   11,  10,  12, -9,  17,  -15, -5,  20,  -6,  -11, 2,   -6,  -18,
      15,  4,   3,   4,   -9, -2,  -3,  -9,  -13, 17,  -21, 5,   3,   -12, 0,
      -4,  9,   -5,  4,   10, -2,  8,   1,   -10, -6,  1,   -9,  10,  11,  -1,
      -5,  4,   -7,  -4,  5,  -4,  4,   12,  -7,  -5,  -9,  -19, 6,   -4,  12,
      -17, -22, 0,   9,   -4, 6,   -5,  5,   -8,  8,   3,   15,  -18, -18, 5,
      3,   -12, 5,   -10, 7,  7,   7,   -9,  17,  2,   -11, -25, 3,   19,  -6,
      7,   1,   7,   5,   -3, 11,  3,   8,   0,   -8,  8,   -2,  -2,  -12, 14,
      -5,  7,   8,   16,  20, -16, -5,  -5,  9,   1,   -10, -6,  14,  10,  -12,
      10,  -6,  5,   0,   3,  8,   -9,  -13, -2,  10,  4,   4,   -16, -17, -9,
      16,  -5,  14,  -9,  -5, -12, 0,   17,  6,   -1,  11,  16,  -20, 1,   -11,
      -1,  -10, -21, 13,  4,  -12, -7,  0,   -14, -6,  3,   12,  -4,  6,   -18,
      -3,  -1,  14,  -8,  -6, -15, 5,   12,  -3,  -10, 4,   6,   13,  -5,  -20,
      0,   3,   -3,  -7,  1,  2,   -10, 7,   -3,  6,   1,   -12, 6,   14,  -5,
      -20, 0,   3,   -3,  -7, 1,   2,   -10, 7,   -3,  6,   1,   -12, 6,   15,
      -5,  -20, 0,   3,   -3, -7,  1,   2,   -10, 7,   -3,  6,   1,   -12, 6,
      16,
  };

  std::vector<int32_t> scratch(5 * 8, 0);
  std::vector<float> output(4 * 8, 0);
  int32_t* row_sums = scratch.data() + 8 * 4;
  bool compute_row_sums = true;
  MatrixBatchVectorMultiplyAccumulate(
      input_to_gate_weights.data(), /*m_rows=*/8, /*m_cols=*/32, input.data(),
      scaling_factors.data(), /*n_batch*/ 4, output.data(), nullptr,
      input_offsets.data(), scratch.data(), row_sums, &compute_row_sums,
      &context);

  const std::vector<float_t> expected_output = {
      -228, 1548,  937, -166, -1164, -1578, -278,  303, 839,  -820,  132,
      1733, -1858, 58,  -425, -587,  -228,  1548,  937, -166, -1164, -1578,
      -278, 303,   839, -820, 132,   1733,  -1858, 58,  -425, -587,
  };

  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
  EXPECT_THAT(compute_row_sums, false);

  std::vector<float> output2(4 * 8, 0);
  MatrixBatchVectorMultiplyAccumulate(
      input_to_gate_weights.data(), /*m_rows=*/8, /*m_cols=*/32, input.data(),
      scaling_factors.data(), /*n_batch*/ 4, output2.data(), nullptr,
      input_offsets.data(), scratch.data(), row_sums, &compute_row_sums,
      &context);

  EXPECT_THAT(output2, testing::ElementsAreArray(expected_output));
}

// Qautnized matmul with 2 * 30 input and 9 * 30 matrix.
TEST(uKernels, QuantMatrixBatchVectorMultiplyAccumulate8x8_8Test) {
  CpuBackendContext context;
  const std::vector<int8_t> input = {
      4,   -41, 5,   -41, 22,  17, -30, 24,  13,  -47, 18, 9,   -11, -30, 16,
      -47, 12,  36,  -20, 27,  -3, 0,   -51, -31, 3,   -8, -38, 43,  23,  12,
      11,  -23, -26, 23,  14,  -9, -44, 22,  21,  -30, 3,  -47, -26, -21, -24,
      -44, 34,  -11, -23, -28, 26, -38, 19,  35,  9,   23, 6,   -42, -25, 28,
  };
  const std::vector<int32_t> input_zeropoint_times_weights = {
      0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  const std::vector<int8_t> input_to_gate_weights = {
      13,  -7,  -20, -22, 8,   -46, 9,   -2,  -18, -42, 40,  28,  -7,  24,  34,
      -7,  -24, -24, 19,  14,  -19, -6,  -2,  -3,  5,   -36, -13, 6,   -27, 36,
      -23, 0,   20,  -37, -23, 9,   17,  -41, 33,  -15, -18, -42, -41, -34, -16,
      -6,  12,  -14, -15, -20, -14, 21,  -3,  -1,  -26, 54,  51,  35,  -14, 9,
      -2,  13,  -6,  39,  34,  -21, 39,  -51, 19,  -44, 52,  0,   -2,  -38, -35,
      -33, 4,   -22, -37, 27,  -23, 3,   -10, 5,   32,  6,   1,   -35, 24,  -19,
      46,  43,  -55, 5,   38,  -14, 32,  -43, -44, -17, -13, -28, 56,  28,  -42,
      4,   10,  -7,  25,  -15, -9,  -25, -14, -15, 6,   -10, -22, 40,  -72, 18,
      -6,  -18, -2,  37,  -13, -10, 11,  -9,  32,  -28, 19,  -2,  4,   -31, 50,
      -15, 23,  -34, -9,  41,  -6,  -34, 17,  2,   24,  -15, 21,  -17, -8,  -20,
      1,   -63, 19,  -40, 12,  -5,  5,   -6,  1,   19,  -9,  -23, 5,   -34, 11,
      26,  21,  54,  34,  -43, -29, 1,   16,  31,  -56, -28, 57,  -15, -23, 37,
      -17, -3,  -6,  29,  18,  77,  17,  -20, -14, -19, 8,   -24, -7,  -45, -3,
      0,   -25, -8,  6,   9,   3,   -15, 51,  4,   -15, -19, -16, -14, -47, -52,
      25,  9,   58,  26,  -9,  -27, 49,  -6,  -21, 21,  18,  12,  -9,  -9,  14,
      31,  -26, -19, -50, 17,  35,  11,  -10, 22,  -16, -43, -2,  26,  55,  -20,
      -7,  21,  33,  -20, 26,  -15, -22, 30,  27,  3,   -34, 26,  12,  -1,  19,
      26,  -25, 10,  30,  30,  -14, -23, -23, -35, -16, 26,  -41, 11,  1,   21,
  };
  const int32_t multiplier = 1347771520;
  const int32_t shift = -7;
  const int32_t output_zp = -11;

  std::vector<int8_t> output = {1, 2, 3, 4, 5,  6,  5,  4,  3,
                                2, 1, 2, 8, -1, -2, 11, 17, 18};
  std::vector<int32_t> scratch(2 * 9, 0);
  MatrixBatchVectorMultiplyAccumulate(
      input.data(), input_zeropoint_times_weights.data(),
      input_to_gate_weights.data(), multiplier, shift,
      /*n_batch=*/2, /*n_input=*/30, /*n_output=*/9, output_zp, scratch.data(),
      output.data(), &context);
  const std::vector<int8_t> expected_output = {
      5,   -9, -2, -30, -5, -11, -22, -18, 18,
      -19, 2,  11, -5,  9,  -2,  10,  -38, -22,
  };

  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Qautnized matmul with 2 * 30 input and 9 * 30 matrix with zero point.
TEST(uKernels, QuantMatrixBatchVectorMultiply8x8_8WithZPTest) {
  const int32_t input_zp = 3;
  const std::vector<int8_t> input = {
      4,   -41, 5,   -41, 22,  17, -30, 24,  13,  -47, 18, 9,   -11, -30, 16,
      -47, 12,  36,  -20, 27,  -3, 0,   -51, -31, 3,   -8, -38, 43,  23,  12,
      11,  -23, -26, 23,  14,  -9, -44, 22,  21,  -30, 3,  -47, -26, -21, -24,
      -44, 34,  -11, -23, -28, 26, -38, 19,  35,  9,   23, 6,   -42, -25, 28,
  };
  const std::vector<int8_t> input_to_gate_weights = {
      13,  -7,  -20, -22, 8,   -46, 9,   -2,  -18, -42, 40,  28,  -7,  24,  34,
      -7,  -24, -24, 19,  14,  -19, -6,  -2,  -3,  5,   -36, -13, 6,   -27, 36,
      -23, 0,   20,  -37, -23, 9,   17,  -41, 33,  -15, -18, -42, -41, -34, -16,
      -6,  12,  -14, -15, -20, -14, 21,  -3,  -1,  -26, 54,  51,  35,  -14, 9,
      -2,  13,  -6,  39,  34,  -21, 39,  -51, 19,  -44, 52,  0,   -2,  -38, -35,
      -33, 4,   -22, -37, 27,  -23, 3,   -10, 5,   32,  6,   1,   -35, 24,  -19,
      46,  43,  -55, 5,   38,  -14, 32,  -43, -44, -17, -13, -28, 56,  28,  -42,
      4,   10,  -7,  25,  -15, -9,  -25, -14, -15, 6,   -10, -22, 40,  -72, 18,
      -6,  -18, -2,  37,  -13, -10, 11,  -9,  32,  -28, 19,  -2,  4,   -31, 50,
      -15, 23,  -34, -9,  41,  -6,  -34, 17,  2,   24,  -15, 21,  -17, -8,  -20,
      1,   -63, 19,  -40, 12,  -5,  5,   -6,  1,   19,  -9,  -23, 5,   -34, 11,
      26,  21,  54,  34,  -43, -29, 1,   16,  31,  -56, -28, 57,  -15, -23, 37,
      -17, -3,  -6,  29,  18,  77,  17,  -20, -14, -19, 8,   -24, -7,  -45, -3,
      0,   -25, -8,  6,   9,   3,   -15, 51,  4,   -15, -19, -16, -14, -47, -52,
      25,  9,   58,  26,  -9,  -27, 49,  -6,  -21, 21,  18,  12,  -9,  -9,  14,
      31,  -26, -19, -50, 17,  35,  11,  -10, 22,  -16, -43, -2,  26,  55,  -20,
      -7,  21,  33,  -20, 26,  -15, -22, 30,  27,  3,   -34, 26,  12,  -1,  19,
      26,  -25, 10,  30,  30,  -14, -23, -23, -35, -16, 26,  -41, 11,  1,   21,
  };
  const int32_t multiplier = 1347771520;
  const int32_t shift = -7;
  const int32_t output_zp = -11;

  std::vector<int8_t> output = {1, 2, 3, 4, 5,  6,  5,  4,  3,
                                2, 1, 2, 8, -1, -2, 11, 17, 18};

  MatrixBatchVectorMultiply(
      input.data(), input_zp, input_to_gate_weights.data(), multiplier, shift,
      /*n_batch=*/2, /*n_input=*/30, /*n_cell=*/9, output.data(), output_zp);
  const std::vector<int8_t> expected_output = {6,   -9,  -4, -32, -10, -17,
                                               -25, -25, 14, -19, 3,   10,
                                               -12, 10,  0,  1,   -57, -41};

  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Qautnized matmul with 2 * 30 input and 9 * 30 matrix with zero point.
TEST(uKernels, QuantMatrixBatchVectorMultiply16x8_8WithZPTest) {
  const std::vector<int16_t> input = {
      400, -41, 5,   -41, 22,  17, -30, 24,  130, -47, 18, 9,   -11, -30, 16,
      -47, 12,  36,  -20, 27,  -3, 0,   -51, -31, 3,   -8, -38, 43,  23,  12,
      11,  -23, -26, 23,  14,  -9, -44, 22,  21,  -30, 3,  -47, -26, -21, -24,
      -44, 34,  -11, -23, -28, 26, -38, 19,  35,  9,   23, 6,   -42, -25, 28,
  };
  const std::vector<int8_t> input_to_gate_weights = {
      13,  -7,  -20, -22, 8,   -46, 9,   -2,  -18, -42, 40,  28,  -7,  24,  34,
      -7,  -24, -24, 19,  14,  -19, -6,  -2,  -3,  5,   -36, -13, 6,   -27, 36,
      -23, 0,   20,  -37, -23, 9,   17,  -41, 33,  -15, -18, -42, -41, -34, -16,
      -6,  12,  -14, -15, -20, -14, 21,  -3,  -1,  -26, 54,  51,  35,  -14, 9,
      -2,  13,  -6,  39,  34,  -21, 39,  -51, 19,  -44, 52,  0,   -2,  -38, -35,
      -33, 4,   -22, -37, 27,  -23, 3,   -10, 5,   32,  6,   1,   -35, 24,  -19,
      46,  43,  -55, 5,   38,  -14, 32,  -43, -44, -17, -13, -28, 56,  28,  -42,
      4,   10,  -7,  25,  -15, -9,  -25, -14, -15, 6,   -10, -22, 40,  -72, 18,
      -6,  -18, -2,  37,  -13, -10, 11,  -9,  32,  -28, 19,  -2,  4,   -31, 50,
      -15, 23,  -34, -9,  41,  -6,  -34, 17,  2,   24,  -15, 21,  -17, -8,  -20,
      1,   -63, 19,  -40, 12,  -5,  5,   -6,  1,   19,  -9,  -23, 5,   -34, 11,
      26,  21,  54,  34,  -43, -29, 1,   16,  31,  -56, -28, 57,  -15, -23, 37,
      -17, -3,  -6,  29,  18,  77,  17,  -20, -14, -19, 8,   -24, -7,  -45, -3,
      0,   -25, -8,  6,   9,   3,   -15, 51,  4,   -15, -19, -16, -14, -47, -52,
      25,  9,   58,  26,  -9,  -27, 49,  -6,  -21, 21,  18,  12,  -9,  -9,  14,
      31,  -26, -19, -50, 17,  35,  11,  -10, 22,  -16, -43, -2,  26,  55,  -20,
      -7,  21,  33,  -20, 26,  -15, -22, 30,  27,  3,   -34, 26,  12,  -1,  19,
      26,  -25, 10,  30,  30,  -14, -23, -23, -35, -16, 26,  -41, 11,  1,   21,
  };

  const std::vector<int32_t> input_zeropoint_times_weights = {
      0, 2, 3, 4, 5, 4, 3, 2, 10,
  };
  const int32_t multiplier = 1347771520;
  const int32_t shift = -8;
  const int32_t output_zp = -11;

  std::vector<int8_t> output = {1, 2, 3, 4, 5,  6,  5,  4,  3,
                                2, 1, 2, 8, -1, -2, 11, 17, 18};

  MatrixBatchVectorMultiply(
      input.data(), input_to_gate_weights.data(), multiplier, shift,
      input_zeropoint_times_weights.data(),
      /*n_batch=*/2, /*n_hidden=*/30, /*n_output=*/9, output_zp, output.data());
  const std::vector<int8_t> expected_output = {4,   -24, -5, 10,  -7,  -13,
                                               -39, 2,   3,  -16, -5,  -1,
                                               -12, -1,  -6, -6,  -33, -25};

  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Quantized matmul with 9 * 30 matrix.
TEST(uKernels, MatrixScalarMultiplyAccumulateTest) {
  std::vector<int32_t> output = {
      -620, -170, -395, 715, -1220, -1080, 1130, -260, -470,
  };
  const std::vector<int8_t> weight = {
      -10, -4,  -8,  16,  4,   -16, -1,  11,  1,   2,   -25, 19,  7,   9,   2,
      -24, -2,  10,  -7,  7,   -5,  -2,  3,   4,   3,   -4,  -7,  -11, -13, -18,
      11,  10,  12,  -9,  17,  -15, -5,  20,  -6,  -11, 2,   -6,  -18, 15,  4,
      4,   -9,  -2,  -3,  -9,  -13, 17,  -21, 5,   3,   -12, 0,   -4,  9,   -5,
      10,  -2,  8,   1,   -10, -6,  1,   -9,  10,  11,  -1,  -5,  4,   -7,  -4,
      -4,  4,   12,  -7,  -5,  -9,  -19, 6,   -4,  12,  -17, -22, 0,   9,   -4,
      -5,  5,   -8,  8,   3,   15,  -18, -18, 5,   3,   -12, 5,   -10, 7,   7,
      -9,  17,  2,   -11, -25, 3,   19,  -6,  7,   1,   7,   5,   -3,  11,  3,
      0,   -8,  8,   -2,  -2,  -12, 14,  -5,  7,   8,   16,  20,  -16, -5,  -5,
      1,   -10, -6,  14,  10,  -12, 10,  -6,  5,   0,   3,   8,   -9,  -13, -2,
      4,   4,   -16, -17, -9,  16,  -5,  14,  -9,  -5,  -12, 0,   17,  6,   -1,
      16,  -20, 1,   -11, -1,  -10, -21, 13,  4,   -12, -7,  0,   -14, -6,  3,
      -4,  6,   -18, -3,  -1,  14,  -8,  -6,  -15, 5,   12,  -3,  -10, 4,   6,
      -5,  -20, 0,   3,   -3,  -7,  1,   2,   -10, 7,   -3,  6,   1,   -12, 6,
      4,   -12, 2,   6,   -20, 0,   5,   23,  15,  14,  9,   8,   20,  -2,  9,
      -8,  -8,  -7,  -4,  -8,  -9,  7,   -12, -2,  2,   1,   -14, 31,  4,   -14,
      3,   10,  -18, -17, -1,  18,  1,   12,  0,   7,   -3,  -5,  8,   -9,  18,
      17,  7,   -15, 3,   20,  4,   -8,  16,  6,   -3,  -3,  9,   -4,  -6,  4,
  };
  MatrixScalarMultiplyAccumulate(weight.data(), 3, 9, 30, output.data());
  const std::vector<int32_t> expected_output = {
      -797, -227, -536, 739, -1187, -1314, 965, -140, -257,
  };

  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Quantized layer norm of n_batch = 2 and n_input = 15.
TEST(uKernels, QuantApplyLayerNormTest) {
  const std::vector<int16_t> input = {
      -310,  596,   34,   -68,  475,  92,  672, -54,  -913, -200,
      -1194, -836,  -620, -237, 991,  533, 721, -736, -8,   -941,
      -372,  -1084, 591,  2557, -779, 175, 582, 956,  -287, 944,
  };
  const std::vector<int16_t> layer_norm_weights = {
      21849, 22882, 20626, 23854, 24779, 26354, 12980, 26231,
      23716, 27271, 24937, 22647, 24715, 22854, 19646,
  };
  const std::vector<int32_t> bias_weight = {
      -14175520, -13805465, -16027609, -13786809, -13321033,
      -14399810, -15055368, -14536623, -14508746, -13784007,
      -15206609, -15125830, -14996304, -14847597, -12814379,
  };
  const int32_t multiplier = 1895840000;
  const int32_t shift = -13;
  const int32_t limit = 1;

  std::vector<int16_t> output(2 * 15, 0);
  ApplyLayerNorm(input.data(), layer_norm_weights.data(), bias_weight.data(),
                 multiplier, shift, limit, 2, 15, output.data());
  const std::vector<int16_t> expected_output = {
      -9407,  5846,   -4802,  -5295,  4822,   -2390,  930,   -5283,
      -20352, -7846,  -26539, -18704, -15829, -8627,  10313, -2522,
      -132,   -16058, -8206,  -19158, -13296, -14407, -1235, 20612,
      -18591, -6738,  -2274,  2602,   -11622, 1565,
  };
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Quantized layer norm of n_batch = 2 and n_input = 15.
TEST(uKernels, QuantApplyLayerNormFloatTest) {
  const std::vector<int16_t> input = {
      -310,  596,   34,   -68,  475,  92,  672, -54,  -913, -200,
      -1194, -836,  -620, -237, 991,  533, 721, -736, -8,   -941,
      -372,  -1084, 591,  2557, -779, 175, 582, 956,  -287, 944,
  };
  const std::vector<int16_t> layer_norm_weights = {
      21849, 22882, 20626, 23854, 24779, 26354, 12980, 26231,
      23716, 27271, 24937, 22647, 24715, 22854, 19646,
  };
  const std::vector<int32_t> bias_weight = {
      -14175520, -13805465, -16027609, -13786809, -13321033,
      -14399810, -15055368, -14536623, -14508746, -13784007,
      -15206609, -15125830, -14996304, -14847597, -12814379,
  };
  const int32_t multiplier = 1895840000;
  const int32_t shift = -13;

  std::vector<int16_t> output(2 * 15, 0);
  ApplyLayerNormFloat(input.data(), layer_norm_weights.data(), multiplier,
                      shift, bias_weight.data(), 2, 15, output.data());
  const std::vector<int16_t> expected_output = {
      -9408,  5844,   -4803,  -5297,  4826,   -2392,  927,   -5286,
      -20353, -7851,  -26534, -18701, -15830, -8623,  10312, -2524,
      -136,   -16053, -8206,  -19160, -13299, -14407, -1233, 20617,
      -18594, -6736,  -2272,  2597,   -11620, 1566};

  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Quantized tanh with Q0.15 input and Q0.15 output.
TEST(uKernels, QuantTanh0Test) {
  const std::vector<int16_t> input = {
      -145, 899, -176, -35,  264, 289,  8,    27,   -37,  -1310,
      -120, 127, -16,  106,  370, -583, -299, 93,   -548, 548,
      653,  -29, -53,  1058, -52, -164, -149, -635, 201,  -1297,
      -145, 899, -176, -35,  264, 289,  8,    27,   -37,  -1310,
      -120, 127, -16,  106,  370, -583, -299, 93,   -548, 548,
      653,  -29, -53,  1058, -52, -164, -149, -635, 201,  -1297,
  };
  std::vector<int16_t> output(4 * 15, 0);
  ApplyTanh(0, input.data(), 4, 15, output.data());
  const std::vector<int16_t> expected_output = {
      -136, 904, -176, -40,  260, 292,  8,    28,   -44,  -1304,
      -120, 120, -24,  112,  376, -576, -308, 88,   -544, 544,
      652,  -32, -60,  1056, -56, -156, -144, -636, 192,  -1300,
      -136, 904, -176, -40,  260, 292,  8,    28,   -44,  -1304,
      -120, 120, -24,  112,  376, -576, -308, 88,   -544, 544,
      652,  -32, -60,  1056, -56, -156, -144, -636, 192,  -1300,
  };
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Quantized tanh with Q3.12 input and Q0.15 output.
TEST(uKernels, QuantTanh3Test) {
  const std::vector<int16_t> input = {
      -145, 899, -176, -35,  264, 289,  8,    27,   -37,  -1310,
      -120, 127, -16,  106,  370, -583, -299, 93,   -548, 548,
      653,  -29, -53,  1058, -52, -164, -149, -635, 201,  -1297,
      -145, 899, -176, -35,  264, 289,  8,    27,   -37,  -1310,
      -120, 127, -16,  106,  370, -583, -299, 93,   -548, 548,
      653,  -29, -53,  1058, -52, -164, -149, -635, 201,  -1297,
  };
  std::vector<int16_t> output(4 * 15, 0);
  ApplyTanh(3, input.data(), 4, 15, output.data());
  const std::vector<int16_t> expected_output = {
      -1156, 7076, -1412, -276, 2104, 2308,  64,    220,   -288,  -10132,
      -964,  1016, -120,  844,  2944, -4640, -2392, 736,   -4352, 4352,
      5180,  -232, -428,  8276, -412, -1308, -1196, -5044, 1612,  -10044,
      -1156, 7076, -1412, -276, 2104, 2308,  64,    220,   -288,  -10132,
      -964,  1016, -120,  844,  2944, -4640, -2392, 736,   -4352, 4352,
      5180,  -232, -428,  8276, -412, -1308, -1196, -5044, 1612,  -10044,
  };
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Quantized tanh with float calculation.
TEST(uKernels, QuantTanhFloatTest) {
  const std::vector<int16_t> input = {
      -1,   0,   1,    -35,  264, 289,  8,    27,   -37,  -1310,
      -120, 127, -16,  106,  370, -583, -299, 93,   -548, 548,
      653,  -29, -53,  1058, -52, -164, -149, -635, 201,  -1297,
      -145, 899, -176, -35,  264, 289,  8,    27,   -37,  -1310,
      -120, 127, -16,  106,  370, -583, -299, 93,   -548, 548,
      653,  -29, -53,  1058, -52, -164, -149, -635, 201,  -1297,
  };
  std::vector<int16_t> output(4 * 15, 0);
  ApplyTanhFloat(input.data(), 4, 15, -12, output.data());
  const std::vector<int16_t> expected_output = {
      -8,    0,    8,     -279, 2109, 2308,  63,    215,   -295,  -10136,
      -959,  1015, -127,  847,  2951, -4632, -2387, 743,   -4358, 4358,
      5180,  -231, -423,  8280, -415, -1311, -1191, -5039, 1606,  -10042,
      -1159, 7078, -1407, -279, 2109, 2308,  63,    215,   -295,  -10136,
      -959,  1015, -127,  847,  2951, -4632, -2387, 743,   -4358, 4358,
      5180,  -231, -423,  8280, -415, -1311, -1191, -5039, 1606,  -10042};

  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Quantized tanh with Q4.11 input and Q0.15 output.
TEST(uKernels, QuantTanh4Test) {
  const std::vector<int16_t> input = {
      -5,  163, -31, -5,  54, 90, 1,  2,  -4, -42, -8,  29,  0,   47, 150,
      -26, -36, 9,   -73, 25, 14, -2, -1, 29, -10, -12, -18, -29, 51, -92,
      -5,  163, -31, -5,  54, 90, 1,  2,  -4, -42, -8,  29,  0,   47, 150,
      -26, -36, 9,   -73, 25, 14, -2, -1, 29, -10, -12, -18, -29, 51, -92,
  };
  std::vector<int16_t> output(4 * 15, 0);
  ApplyTanh(4, input.data(), 4, 15, output.data());
  const std::vector<int16_t> expected_output = {
      -76,  2596, -496, -76, 856,  1436, 24,   36,   -64,   -672,
      -120, 456,  0,    752, 2400, -412, -576, 148,  -1168, 400,
      216,  -36,  -24,  456, -164, -192, -292, -456, 820,   -1476,
      -76,  2596, -496, -76, 856,  1436, 24,   36,   -64,   -672,
      -120, 456,  0,    752, 2400, -412, -576, 148,  -1168, 400,
      216,  -36,  -24,  456, -164, -192, -292, -456, 820,   -1476,
  };
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Quantized sigmoid with Q3.12 input and Q0.15 output.
TEST(uKernels, QuantSigmoidTest) {
  const std::vector<int16_t> input = {
      -10500, 1398,   -6963,  -7404,  485,    -5401,  -1757,  -7668,  -19248,
      -9692,  -24249, -17923, -15840, -10026, 5249,   -89,    1787,   -16178,
      -6691,  -19524, -13439, -24048, -1123,  32767,  -17267, -3378,  823,
      11482,  -11139, 7508,   -10500, 1398,   -6963,  -7404,  485,    -5401,
      -1757,  -7668,  -19248, -9692,  -24249, -17923, -15840, -10026, 5249,
      -89,    1787,   -16178, -6691,  -19524, -13439, -24048, -1123,  32767,
      -17267, -3378,  823,    11482,  -11139, 7508,
  };
  std::vector<int16_t> output(4 * 15, 0);
  ApplySigmoid(input.data(), 4, 15, output.data());
  const std::vector<int16_t> expected_output = {
      2339, 19152, 5063,  4617,  17350, 6917,  12921, 4371,  299,  2813,
      89,   409,   673,   2605,  25646, 16207, 19904, 615,   5353, 273,
      1187, 91,    14153, 32756, 475,   9983,  18026, 30898, 2023, 28246,
      2339, 19152, 5063,  4617,  17350, 6917,  12921, 4371,  299,  2813,
      89,   409,   673,   2605,  25646, 16207, 19904, 615,   5353, 273,
      1187, 91,    14153, 32756, 475,   9983,  18026, 30898, 2023, 28246,
  };
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Quantized sigmoid with Q3.12 input and Q0.15 output.
TEST(uKernels, QuantSigmoidFloatTest) {
  const std::vector<int16_t> input = {
      -10500, 1398,   -6963,  -7404,  485,    -5401,  -1757,  -7668,  -19248,
      -9692,  -24249, -17923, -15840, -10026, 5249,   -89,    1787,   -16178,
      -6691,  -19524, -13439, -24048, -1123,  32767,  -17267, -3378,  823,
      11482,  -11139, 7508,   -10500, 1398,   -6963,  -7404,  485,    -5401,
      -1757,  -7668,  -19248, -9692,  -24249, -17923, -15840, -10026, 5249,
      -89,    1787,   -16178, -6691,  -19524, -13439, -24048, -1123,  32767,
      -17267, -3378,  823,    11482,  -11139, 7508,
  };
  std::vector<int16_t> output(4 * 15, 0);
  ApplySigmoidFloat(input.data(), 4, 15, output.data());
  const std::vector<int16_t> expected_output = {
      2343, 19153, 5061,  4617,  17352, 6915,  12922, 4368,  295,  2811,
      87,   407,   671,   2608,  25647, 16206, 19902, 619,   5352, 276,
      1187, 92,    14151, 32757, 476,   9986,  18024, 30895, 2026, 28249,
      2343, 19153, 5061,  4617,  17352, 6915,  12922, 4368,  295,  2811,
      87,   407,   671,   2608,  25647, 16206, 19902, 619,   5352, 276,
      1187, 92,    14151, 32757, 476,   9986,  18024, 30895, 2026, 28249};

  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Quantized Multiply with 16bit output and 15 bit shift.
TEST(uKernels, QuantMul16bitOut15ShiftTest) {
  const std::vector<int16_t> input1 = {
      2491, 32767, -32768, 32767, -32768, 32767, 32767, -32768, -32768, 2157,
      4545, 14835, 1285,   29498, 26788,  2907,  7877,  6331,   8775,   3001,
      1399, 4683,  1437,   1853,  12163,  4927,  7977,  3001,   16612,  4791,
  };
  const std::vector<int16_t> input2 = {
      -1156, 32767, -32768, -32768, 32767, 2308,  64,    220,   -288,  -10132,
      -964,  1016,  -120,   844,    2944,  -4640, -2392, 736,   -4352, 4352,
      5180,  -232,  -428,   8276,   -412,  -1308, -1196, -5044, 1612,  -10044,
  };
  std::vector<int16_t> output(2 * 15, 0);
  CwiseMul(input1.data(), input2.data(), 2, 15, 15, output.data());
  const std::vector<int16_t> expected_output = {
      -88,  32766, -32768, -32767, -32767, 2308, 64,   -220, 288,   -667,
      -134, 460,   -5,     760,    2407,   -412, -575, 142,  -1165, 399,
      221,  -33,   -19,    468,    -153,   -197, -291, -462, 817,   -1469,
  };
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Quantized Multiply with 16bit output and 19 bit shift.
TEST(uKernels, QuantMul16bitOut19ShiftTest) {
  const std::vector<int16_t> input1 = {
      2491, 32767, -32768, 32767, -32768, 32767, 32767, -32768, -32768, 2157,
      4545, 14835, 1285,   29498, 26788,  2907,  7877,  6331,   8775,   3001,
      1399, 4683,  1437,   1853,  12163,  4927,  7977,  3001,   16612,  4791,
  };
  const std::vector<int16_t> input2 = {
      -1156, 32767, -32768, -32768, 32767, 2308,  64,    220,   -288,  -10132,
      -964,  1016,  -120,   844,    2944,  -4640, -2392, 736,   -4352, 4352,
      5180,  -232,  -428,   8276,   -412,  -1308, -1196, -5044, 1612,  -10044,
  };
  std::vector<int16_t> output(2 * 15, 0);
  CwiseMul(input1.data(), input2.data(), 2, 15, 19, output.data());
  const std::vector<int16_t> expected_output = {
      -5, 2048, 2048, -2048, -2048, 144, 4,   -14, 18,  -42,
      -8, 29,   0,    47,    150,   -26, -36, 9,   -73, 25,
      14, -2,   -1,   29,    -10,   -12, -18, -29, 51,  -92,
  };
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Quantized Multiply with arbitrary scale.
TEST(uKernels, QuantMul8bitArbitrarySclaeTest) {
  // scale = 0.000028.
  int multiplier = 1970324837;
  int shift = -15;

  const std::vector<int16_t> input1 = {
      2491, 32767, -32768, 32767, -32768, 32767, 32767, -32768, -32768, 2157,
      4545, 14835, 1285,   29498, 26788,  2907,  7877,  6331,   8775,   3001,
      1399, 4683,  1437,   1853,  12163,  4927,  7977,  3001,   16612,  4791,
  };
  const std::vector<int16_t> input2 = {
      -1156, 32767, -32768, -32768, 32767, 2308,  64,    220,   -288,  -10132,
      -964,  1016,  -120,   844,    2944,  -4640, -2392, 736,   -4352, 4352,
      5180,  -232,  -428,   8276,   -412,  -1308, -1196, -5044, 1612,  -10044,
  };
  std::vector<int8_t> output(2 * 15, 0);
  CwiseMul(input1.data(), input2.data(), multiplier, shift, 2, 15, 3,
           output.data());
  const std::vector<int8_t> expected_output = {
      -84,  127, 127, -128, -128, 127,  56,   -128, 127,  -128,
      -126, 127, -7,  127,  127,  -128, -128, 127,  -128, 127,
      127,  -33, -20, 127,  -128, -128, -128, -128, 127,  -128,
  };
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Quantized element wise Add with saturation.
TEST(uKernels, QuantAddTest) {
  const std::vector<int16_t> input1 = {
      2491,   32767, -32768, 32767, -32768, 32767, 32767, -32768, -32768, 20000,
      -20000, 14835, 1285,   29498, 26788,  2907,  7877,  6331,   8775,   3001,
      1399,   4683,  1437,   1853,  12163,  4927,  7977,  3001,   16612,  4791,
  };
  const std::vector<int16_t> input2 = {
      -1156,  32767, -32768, -32768, 32767, 2308,  64,    220,   -288,  20000,
      -20000, 1016,  -120,   844,    2944,  -4640, -2392, 736,   -4352, 4352,
      5180,   -232,  -428,   8276,   -412,  -1308, -1196, -5044, 1612,  -10044,
  };
  std::vector<int16_t> output(2 * 15, 0);
  CwiseAdd(input1.data(), input2.data(), 2, 15, output.data());
  const std::vector<int16_t> expected_output = {
      1335,   32767, -32768, -1,    -1,    32767, 32767, -32548, -32768, 32767,
      -32768, 15851, 1165,   30342, 29732, -1733, 5485,  7067,   4423,   7353,
      6579,   4451,  1009,   10129, 11751, 3619,  6781,  -2043,  18224,  -5253,
  };
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

// Quantized clipping for 16 bit.
TEST(uKernels, QuantClip16Test) {
  std::vector<int16_t> input = {
      -10500, 1,     -2,     -7404,  200,    -5401,  -1757, -7668,
      -19248, -9692, -24249, -17923, -15840, -10026, 5249,  -89,
      1787,   -200,  -6691,  -19524, -13439, -24048, -1123, 32767,
      -17267, -3378, 823,    11482,  -11139, 7508,
  };
  CwiseClipping(input.data(), 300, 2, 15);
  const std::vector<int16_t> expected_output = {
      -300, 1,    -2,   -300, 200,  -300, -300, -300, -300, -300,
      -300, -300, -300, -300, 300,  -89,  300,  -200, -300, -300,
      -300, -300, -300, 300,  -300, -300, 300,  300,  -300, 300,
  };
  EXPECT_THAT(input, testing::ElementsAreArray(expected_output));
}

// Quantized clipping for 8 bit.
TEST(uKernels, QuantClip8Test) {
  std::vector<int8_t> input = {
      4,   -11, -5, -34, -10, -17, -27, -22, 15,  127, -128, 1,  3, 56, 3,
      -21, 1,   9,  -13, 10,  0,   -1,  -55, -40, 127, -128, 11, 4, 6,  32,
  };
  CwiseClipping(input.data(), 32, 2, 15);
  const std::vector<int8_t> expected_output = {
      4,   -11, -5, -32, -10, -17, -27, -22, 15,  32, -32, 1,  3, 32, 3,
      -21, 1,   9,  -13, 10,  0,   -1,  -32, -32, 32, -32, 11, 4, 6,  32,
  };
  EXPECT_THAT(input, testing::ElementsAreArray(expected_output));
}

struct MatrixVectorData {
  // Contains dense parameters.
  std::vector<int8_t> matrix;

  // Like matrix, but with about half of the parameters set to zero.
  // Use this to create golden output for sparse matrix tests.
  std::vector<int8_t> zeroed_matrix;

  // zeroed_matrix described in sparse form.
  std::vector<int8_t> sparse_matrix;
  std::vector<uint8_t> ledger;

  std::vector<int8_t> vectors;
  std::vector<float> scale_factors;
  std::vector<float> results;

  // Per channel scale data.
  std::vector<float> per_channel_scales;
  std::vector<int32_t> input_offsets;

  int rows;
  int cols;
  int batch;
};

MatrixVectorData SetupMatrixVectorData(int rows, int cols, int batch,
                                       bool negative = false,
                                       bool is_per_channel = false,
                                       bool init_to_one = false) {
  MatrixVectorData data;
  data.rows = rows;
  data.cols = cols;
  data.batch = batch;

  for (int i = 0; i < rows * cols; i++) {
    int sign = 1;
    if ((i % 3) == 0 && negative) sign = -1;
    data.matrix.push_back(sign * (i % 70));
  }
  for (int i = 0; i < cols * batch; i++) {
    int sign = 1;
    if ((i % 5) == 0 && negative) sign = -1;
    data.vectors.push_back(sign * (i % 50));
  }
  data.scale_factors = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8};
  data.results.resize(rows * batch, init_to_one ? 1 : 0);

  data.zeroed_matrix = data.matrix;

  // Make a sparsification ledger.
  for (int i = 0; i < rows; i++) {
    int max_chunks = cols / 16;
    int selected_chunks = (max_chunks / 2);
    bool row_is_odd = (i % 2) > 0;
    bool max_chunks_is_odd = (max_chunks % 2) > 0;

    data.ledger.push_back(selected_chunks);
    if (max_chunks_is_odd && row_is_odd) {
      selected_chunks++;
    }

    // In odd rows, use odd chunk indexes.
    // In even rows, use even chunk indexes.
    for (int j = 0; j < max_chunks; j++) {
      const int chunk_start = i * cols + (j * 16);
      const int chunk_end = i * cols + (j * 16) + 16;
      if ((j % 2) == (i % 2)) {
        // Copy this chunk into the sparse matrix.
        data.ledger.push_back(j);
        for (int k = chunk_start; k < chunk_end; k++) {
          data.sparse_matrix.push_back(data.matrix[k]);
        }
      } else {
        // Zero this part out of zeroed_matrix.
        for (int k = chunk_start; k < chunk_end; k++) {
          data.zeroed_matrix[k] = 0;
        }
      }
    }
  }

  if (is_per_channel) {
    for (int i = 0; i < rows; i++) {
      if (i % 2 == 0) {
        data.per_channel_scales.push_back(0.5);
      } else {
        data.per_channel_scales.push_back(1.0);
      }
    }

    for (int i = 0; i < batch; i++) {
      for (int j = 0; j < cols; j++) {
        data.vectors[i * cols + j] += i;
      }
      data.input_offsets.push_back(i);
    }
  }
  return data;
}

std::vector<float> TestDotprodMatrixBatchVectorMultiply(
    int rows, int cols, int batch, bool negative = false,
    bool init_to_one = false) {
  MatrixVectorData data =
      SetupMatrixVectorData(rows, cols, batch, negative, false, init_to_one);

  // All partial sums in this computation are small enough to fit in the
  // mantissa of a float, and the scale factors are all integers, so we expect
  // an exact result.
  MatrixBatchVectorMultiplyAccumulate(
      data.matrix.data(), rows, cols, data.vectors.data(),
      data.scale_factors.data(), batch, &data.results[0]);
  return data.results;
}

std::vector<float> TestSparseDotprodMatrixBatchVectorMultiply(
    int rows, int cols, int batch, bool negative = false) {
  MatrixVectorData data = SetupMatrixVectorData(rows, cols, batch, negative);
  SparseMatrixBatchVectorMultiplyAccumulate(
      data.sparse_matrix.data(), data.ledger.data(), rows, cols,
      data.vectors.data(), data.scale_factors.data(), batch, &data.results[0]);
  return data.results;
}

std::vector<float> TestPerChannelDotprodMatrixBatchVectorMultiply(
    int rows, int cols, int batch, bool negative = false,
    bool is_per_channel = true) {
  MatrixVectorData data =
      SetupMatrixVectorData(rows, cols, batch, negative, is_per_channel);

  MatrixBatchVectorMultiplyAccumulate(
      data.matrix.data(), rows, cols, data.vectors.data(),
      data.scale_factors.data(), batch, &data.results[0],
      data.per_channel_scales.data(), data.input_offsets.data());
  return data.results;
}

TEST(uKernels, DotprodMatrixBatchVectorMultiplyAccumulateTest) {
  ASSERT_THAT(TestDotprodMatrixBatchVectorMultiply(4, 16, 1),
              testing::ElementsAre(1240, 3160, 5080, 7000));

  ASSERT_THAT(TestDotprodMatrixBatchVectorMultiply(4, 32, 2),
              testing::ElementsAre(10416, 26288, 8490, 23312, 18276, 70756,
                                   37416, 60916));

  ASSERT_THAT(TestDotprodMatrixBatchVectorMultiply(4, 32, 3),
              testing::ElementsAre(10416, 26288, 8490, 23312, 18276, 70756,
                                   37416, 60916, 52080, 142704, 55878, 125712));

  ASSERT_THAT(TestDotprodMatrixBatchVectorMultiply(8, 1024, 3),
              testing::ElementsAreArray(
                  {841094,  853168,  866642,  840286,  860760,  862754,
                   843678,  872552,  1724476, 1769072, 1747588, 1738844,
                   1758240, 1742916, 1761612, 1755808, 2506896, 2564262,
                   2629188, 2515824, 2598390, 2569236, 2537352, 2645118}));

  const bool kNegative = true;
  ASSERT_THAT(TestDotprodMatrixBatchVectorMultiply(4, 64, 1, kNegative),
              testing::ElementsAre(13696, 6904, 7764, 11806));
  ASSERT_THAT(
      TestDotprodMatrixBatchVectorMultiply(4, 32, 2, kNegative),
      testing::ElementsAre(3436, 3522, 1590, 6972, 2516, 20520, 456, 10628));

  // Initialize the results vector with 1s to verify that the code adds
  // to the results vector instead of zero-ing it first.
  const bool kInitToOne = true;
  ASSERT_THAT(
      TestDotprodMatrixBatchVectorMultiply(4, 32, 2, kNegative, kInitToOne),
      testing::ElementsAre(3437, 3523, 1591, 6973, 2517, 20521, 457, 10629));
}

TEST(uKernels, PerChannelDotprodMatrixBatchVectorMultiplyAccumulateTest) {
  ASSERT_THAT(TestPerChannelDotprodMatrixBatchVectorMultiply(4, 16, 1),
              testing::ElementsAre(1240 / 2, 3160, 5080 / 2, 7000));

  ASSERT_THAT(TestPerChannelDotprodMatrixBatchVectorMultiply(4, 32, 2),
              testing::ElementsAre(10416 / 2, 26288, 8490 / 2, 23312, 18276 / 2,
                                   70756, 37416 / 2, 60916));

  ASSERT_THAT(TestPerChannelDotprodMatrixBatchVectorMultiply(4, 32, 3),
              testing::ElementsAre(10416 / 2, 26288, 8490 / 2, 23312, 18276 / 2,
                                   70756, 37416 / 2, 60916, 52080 / 2, 142704,
                                   55878 / 2, 125712));

  ASSERT_THAT(
      TestPerChannelDotprodMatrixBatchVectorMultiply(8, 1024, 3),
      testing::ElementsAreArray(
          {841094 / 2,  853168,  866642 / 2,  840286,  860760 / 2,  862754,
           843678 / 2,  872552,  1724476 / 2, 1769072, 1747588 / 2, 1738844,
           1758240 / 2, 1742916, 1761612 / 2, 1755808, 2506896 / 2, 2564262,
           2629188 / 2, 2515824, 2598390 / 2, 2569236, 2537352 / 2, 2645118}));
}

TEST(uKernels, DotprodMatrixBatchFourVectorMultiplyAccumulateDotprodTest) {
  ASSERT_THAT(TestDotprodMatrixBatchVectorMultiply(2, 16, 4),
              testing::ElementsAreArray(
                  {1240, 3160, 6320, 18352, 15240, 45576, 4200, 16232}));
  ASSERT_THAT(TestDotprodMatrixBatchVectorMultiply(2, 64, 4),
              testing::ElementsAreArray({45794, 38948, 88536, 84252, 157626,
                                         165312, 209864, 246128}));
  ASSERT_THAT(
      TestDotprodMatrixBatchVectorMultiply(2, 64, 8),
      testing::ElementsAreArray({45794, 38948, 88536, 84252, 157626, 165312,
                                 209864, 246128, 219700, 195550, 279684, 278928,
                                 413616, 445662, 374896, 365952}));

  ASSERT_THAT(
      TestDotprodMatrixBatchVectorMultiply(4, 64, 8),
      testing::ElementsAreArray(
          {45794,  38948,  34622,  32816,  88536,  84252,  85008,  90804,
           157626, 165312, 180558, 203364, 209864, 246128, 236472, 208896,
           219700, 195550, 184000, 185050, 279684, 278928, 293292, 322776,
           413616, 445662, 495348, 513674, 374896, 365952, 321168, 296544}));

  ASSERT_THAT(
      TestDotprodMatrixBatchVectorMultiply(16, 1024, 4),
      testing::ElementsAreArray(
          {841094,  853168,  866642,  840286,  860760,  862754,  843678,
           872552,  837586,  851270,  877414,  834188,  863062,  857846,
           841780,  879054,  1724476, 1769072, 1747588, 1738844, 1758240,
           1742916, 1761612, 1755808, 1737684, 1750780, 1747356, 1754152,
           1748348, 1753324, 1743320, 1754316, 2506896, 2564262, 2629188,
           2515824, 2598390, 2569236, 2537352, 2645118, 2508444, 2571480,
           2610576, 2510442, 2618208, 2566584, 2544570, 2614536, 3458904,
           3502688, 3474792, 3505976, 3499360, 3488264, 3485848, 3512832,
           3500616, 3482520, 3489624, 3469008, 3495992, 3524376, 3465680,
           3526264}));

  ASSERT_THAT(
      TestDotprodMatrixBatchVectorMultiply(4, 128, 4),
      testing::ElementsAreArray({87920, 80024, 92288, 103712, 228148, 224820,
                                 233812, 213124, 271284, 271788, 332772, 328236,
                                 419328, 431328, 411968, 417248}));

  ASSERT_THAT(
      TestDotprodMatrixBatchVectorMultiply(4, 128, 8),
      testing::ElementsAreArray(
          {87920,  80024,  92288,  103712, 228148, 224820, 233812, 213124,
           271284, 271788, 332772, 328236, 419328, 431328, 411968, 417248,
           482680, 523840, 560800, 593560, 563940, 609924, 566868, 644772,
           743708, 857780, 818972, 823284, 708384, 695008, 730912, 872096}));

  const bool kNegative = true;
  EXPECT_THAT(TestDotprodMatrixBatchVectorMultiply(1, 16, 1, kNegative),
              testing::ElementsAre(450));
  EXPECT_THAT(TestDotprodMatrixBatchVectorMultiply(2, 64, 8, kNegative),
              testing::ElementsAreArray({13696, 6904, 9952, 12368, 22848, 61632,
                                         40424, 46776, 57630, 38670, 62976,
                                         49824, 39032, 71988, 60128, 148992}));

  std::vector<float> results =
      TestDotprodMatrixBatchVectorMultiply(256, 1024, 8);
  int64_t sum = 0;
  for (int i = 0; i < results.size(); i++) {
    sum += static_cast<int64_t>(results[i]);
  }
  EXPECT_EQ(7980076336, sum);
}

TEST(uKernels,
     PerChannelDotprodMatrixBatchFourVectorMultiplyAccumulateDotprodTest) {
  ASSERT_THAT(
      TestPerChannelDotprodMatrixBatchVectorMultiply(16, 1024, 4),
      testing::ElementsAreArray(
          {841094 / 2,  853168,  866642 / 2,  840286,  860760 / 2,  862754,
           843678 / 2,  872552,  837586 / 2,  851270,  877414 / 2,  834188,
           863062 / 2,  857846,  841780 / 2,  879054,  1724476 / 2, 1769072,
           1747588 / 2, 1738844, 1758240 / 2, 1742916, 1761612 / 2, 1755808,
           1737684 / 2, 1750780, 1747356 / 2, 1754152, 1748348 / 2, 1753324,
           1743320 / 2, 1754316, 2506896 / 2, 2564262, 2629188 / 2, 2515824,
           2598390 / 2, 2569236, 2537352 / 2, 2645118, 2508444 / 2, 2571480,
           2610576 / 2, 2510442, 2618208 / 2, 2566584, 2544570 / 2, 2614536,
           3458904 / 2, 3502688, 3474792 / 2, 3505976, 3499360 / 2, 3488264,
           3485848 / 2, 3512832, 3500616 / 2, 3482520, 3489624 / 2, 3469008,
           3495992 / 2, 3524376, 3465680 / 2, 3526264}));

  ASSERT_THAT(TestPerChannelDotprodMatrixBatchVectorMultiply(4, 128, 4),
              testing::ElementsAreArray(
                  {87920 / 2, 80024, 92288 / 2, 103712, 228148 / 2, 224820,
                   233812 / 2, 213124, 271284 / 2, 271788, 332772 / 2, 328236,
                   419328 / 2, 431328, 411968 / 2, 417248}));

  ASSERT_THAT(TestPerChannelDotprodMatrixBatchVectorMultiply(4, 128, 8),
              testing::ElementsAreArray(
                  {87920 / 2,  80024,  92288 / 2,  103712, 228148 / 2, 224820,
                   233812 / 2, 213124, 271284 / 2, 271788, 332772 / 2, 328236,
                   419328 / 2, 431328, 411968 / 2, 417248, 482680 / 2, 523840,
                   560800 / 2, 593560, 563940 / 2, 609924, 566868 / 2, 644772,
                   743708 / 2, 857780, 818972 / 2, 823284, 708384 / 2, 695008,
                   730912 / 2, 872096}));
}

TEST(uKernels, DotprodSparseMatrixBatchVectorMultiplyAccumulate) {
  EXPECT_THAT(TestSparseDotprodMatrixBatchVectorMultiply(1, 16, 1),
              testing::ElementsAre(0));
  EXPECT_THAT(TestSparseDotprodMatrixBatchVectorMultiply(1, 32, 1),
              testing::ElementsAre(1240));
  EXPECT_THAT(TestSparseDotprodMatrixBatchVectorMultiply(1, 64, 1),
              testing::ElementsAre(26544));
  EXPECT_THAT(TestSparseDotprodMatrixBatchVectorMultiply(1, 64, 2),
              testing::ElementsAre(26544, 24344));
  EXPECT_THAT(TestSparseDotprodMatrixBatchVectorMultiply(4, 64, 4),
              testing::ElementsAreArray(
                  {26544, 15866, 22140, 11408, 24344, 53248, 42704, 39900,
                   48000, 94146, 101892, 81876, 87712, 105160, 148304, 75936}));

  const bool kNegative = true;
  EXPECT_THAT(TestSparseDotprodMatrixBatchVectorMultiply(1, 64, 1, kNegative),
              testing::ElementsAre(8764));
  EXPECT_THAT(TestSparseDotprodMatrixBatchVectorMultiply(2, 64, 2, kNegative),
              testing::ElementsAre(8764, 5196, 7204, 11148));
}

#ifdef __ANDROID__
TEST(uKernels, MatrixBatchVectorMultiplyAccumulateSymmetricQuantizedTest) {
  // Note we use 29 columns as this exercises all the neon kernel: the
  // 16-block SIMD code, the 8-block postamble, and the leftover postamble.
  const int a_rows = 4, a_cols = 29;
  const int kWeightsPerUint32 = 4;
  /* clang-format off */
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

  int8_t* a_int8_data = reinterpret_cast<int8_t*>(
      aligned_malloc(a_rows * a_cols, kWeightsPerUint32));
  float a_min, a_max;
  float scaling_factor_a;
  SymmetricQuantizeFloats(a_float_data, a_rows * a_cols, a_int8_data, &a_min,
                          &a_max, &scaling_factor_a);
  const int8_t expected_a_int8_data[] = {
    /* 1st row */
    5, 10, 15, 20, 25, 30, 35, 40, 44, 45, 50, 54, 59, 64, 68, 73, 77, 82, 86,
    91, 95, 100, 104, 109, 113, 118, 122, 127, 0,
    /* 2nd row */
    -5, -10, -15, -20, -25, -30, -35, -40, -44, -45, -50, -54, -59, -64, -68,
    -73, -77, -82, -86, -91, -95, -100, -104, -109, -113, -118, -122, -127, 0,
    /* 3rd row */
    5, -10, 15, -20, 25, -30, 35, -40, 44, -45, 50, -54, 59, -64, 68, -73, 77,
    -82, 86, -91, 95, -100, 104, -109, 113, -118, 122, -127, 0,
    /* 4th row */
    -5, 10, -15, 20, -25, 30, -35, 40, -44, 45, -50, 54, -59, 64, -68, 73, -77,
    82, -86, 91, -95, 100, -104, 109, -113, 118, -122, 127, 0,
  };
  for (int i = 0; i < a_rows * a_cols; ++i) {
    EXPECT_EQ(expected_a_int8_data[i], a_int8_data[i]);
  }

  const int b_rows = 29, b_cols = 1, batches = 2;
  const float b_float_data[] = {
    /* batch 1 */
    1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    1.0,
    /* batch 2 */
    2.5, -2.1, 3.0, -1.3, 1.3, -1.1, 2.0, -1.7, 1.9, -1.5, 0.5, -0.7, 0.8, -0.3,
    2.8, -2.8, 1.1, -2.3, 1.9, -1.9, 2.1, -0.5, 2.4, -0.1, 1.0, -2.5, 0.7, -1.9,
    0.2,
  };

  // Quantized values of B:
  int8_t b_int8_data[b_rows * b_cols * batches];
  float b_min, b_max;
  float scaling_factor_b[batches];
  SymmetricQuantizeFloats(b_float_data, b_rows * b_cols, b_int8_data, &b_min,
                          &b_max, &scaling_factor_b[0]);
  SymmetricQuantizeFloats(&b_float_data[b_rows * b_cols], b_rows * b_cols,
                          &b_int8_data[b_rows * b_cols], &b_min, &b_max,
                          &scaling_factor_b[1]);

  const int8_t expected_b_int8_data[] = {
    /* batch 1 */
    127, -127, 127, -127, 127, -127, 127, -127, 127, -127, 127, -127, 127, -127,
    127, -127, 127, -127, 127, -127, 127, -127, 127, -127, 127, -127, 127, -127,
    127,
    /* batch 2 */
    106, -89, 127, -55, 55, -47, 85, -72, 80, -64, 21, -30, 34, -13, 119, -119,
    47, -97, 80, -80, 89, -21, 102, -4, 42, -106, 30, -80, 8,
  };
  /* clang-format on */
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
                                      scaling_factor_c, batches, c_float_data);

  // Assert we obtain the expected recovered float values.
  const float expected_c_float_data[] = {
      -14.474, 14.474, 414.402, -414.402, -6.92228, 6.92228, 632.042, -632.042,
  };
  for (int i = 0; i < a_rows * b_cols * batches; ++i) {
    EXPECT_NEAR(expected_c_float_data[i], c_float_data[i], 0.001);
  }

  // Call version of MatrixBatchVectorMultiplyAccumulate that uses
  // CpuBackendGemm.
  std::vector<int32_t> accum_scratch(a_rows * batches);
  std::vector<float> c_float_data_2(a_rows * batches, 0.0);
  CpuBackendContext context;
  MatrixBatchVectorMultiplyAccumulate(
      a_int8_data, a_rows, a_cols, b_int8_data, scaling_factor_c, batches,
      accum_scratch.data(), c_float_data_2.data(), &context);

  // Assert (again) we obtain the expected recovered float values.
  for (int i = 0; i < a_rows * b_cols * batches; ++i) {
    EXPECT_NEAR(expected_c_float_data[i], c_float_data_2[i], 0.001);
  }

  aligned_free(a_int8_data);
}
#endif  // __ANDROID__

TEST(uKernels, SparseMatrixBatchVectorMultiplyAccumulateTest) {
  const int kRow = 4;
  const int kCol = 48;
  const int kBatch = 2;
  /* clang-format off */
  float matrix[kRow * kCol] = {
      /* 1st row */
      1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12, 13.13,
      14.14, 15.15, 16.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 33.33, 34.34, 35.35, 36.36, 37.37, 38.38,
      39.39, 40.40, 41.41, 42.42, 43.43, 44.44, 0, 0, 0, 0,
      /* 2nd row */
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, -17.17, -18.18, -19.19, -20.2, -21.21, -22.22, -23.23, -24.24,
      -25.25, -26.26, -27.27, -28.28, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0,
      /* 3rd row */
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 17.17, -18.18, 19.19, -20.2, 21.21, -22.22, 23.23, -24.24, 25.25,
      -26.26, 27.27, -28.28, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0,
      /* 4th row */
      -1.1, 2.2, -3.3, 4.4, -5.5, 6.6, -7.7, 8.8, -9.9, 10.1, -11.11, 12.12,
      -13.13, 14.14, -15.15, 16.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -33.33, 34.34, -35.35, 36.36, -37.37,
      38.38, -39.39, 40.40, -41.41, 42.42, -43.43, 44.44, 0, 0, 0, 0};

  // BCSR format of the above matrix.
  float matrix_values[] = {
      /* 1st row */
      1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12, 13.13,
      14.14, 15.15, 16.16, 33.33, 34.34, 35.35, 36.36, 37.37, 38.38, 39.39,
      40.40, 41.41, 42.42, 43.43, 44.44, 0, 0, 0, 0,
      /* 2nd row */
      -17.17, -18.18, -19.19, -20.2, -21.21, -22.22, -23.23, -24.24, -25.25,
      -26.26, -27.27, -28.28, 0, 0.0, 0.0, 0.0,
      /* 3rd row */
      17.17, -18.18, 19.19, -20.2, 21.21, -22.22, 23.23, -24.24, 25.25, -26.26,
      27.27, -28.28, 0, 0.0, 0.0, 0.0,
      /* 4th row */
      -1.1, 2.2, -3.3, 4.4, -5.5, 6.6, -7.7, 8.8, -9.9, 10.1, -11.11, 12.12,
      -13.13, 14.14, -15.15, 16.16, -33.33, 34.34, -35.35, 36.36, -37.37, 38.38,
      -39.39, 40.40, -41.41, 42.42, -43.43, 44.44, 0, 0, 0, 0};
  uint8_t ledger[] = {
      2, 0,  2,  // 1st row
      1, 1,      // 2nd row
      1, 1,      // 3rd row
      2, 0,  2   // 4th row
  };

  float vector[kBatch * kCol] = {
    /* 1st batch */
    1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    /* 2nd batch */
    2.5, 0.0, -2.1, 0.0, 3.0, 0.0, -1.3, 0.0, 1.3, 0.0, -1.1, 0.0, 2.0, 0.0,
    -1.7, 0.0, 1.9, 0.0, -1.5, 0.0, 0.5, 0.0, -0.7, 0.0, 0.8, 0.0, -0.3, 0.0,
    2.8, 0.0, -2.8, 0.0, 1.1, -2.3, 1.9, -1.9, 2.1, -0.5, 2.4, -0.1, 1.0, -2.5,
    0.7, -1.9, 0.2, 0.0, 0.1, 0.2,
  };
  /* clang-format on */

  std::vector<float> dense_output(kRow * kBatch, 0.0);
  MatrixBatchVectorMultiplyAccumulate(matrix, kRow, kCol, vector, kBatch,
                                      dense_output.data());

  EXPECT_THAT(dense_output, ElementsAreArray(ArrayFloatNear(
                                {-13.69, 6.06001, 272.7, -608.03, -9.66602,
                                 -10.201, 10.201, -713.897949},
                                1e-4)));

  std::vector<float> sparse_output(kRow * kBatch, 0.0);
  SparseMatrixBatchVectorMultiplyAccumulate(
      matrix_values, ledger, kRow, kCol, vector, kBatch, sparse_output.data());

  EXPECT_THAT(sparse_output,
              ElementsAreArray(ArrayFloatNear(dense_output, 1e-4)));
}

#ifdef __ANDROID__
TEST(uKernels,
     SparseMatrixBatchVectorMultiplyAccumulateSymmetricQuantizedTest) {
  const int kRow = 4;
  const int kCol = 48;
  const int kBatch = 2;
  /* clang-format off */
  const int8_t quantized_matrix[] = {
      /* 1st row */
      3, 6, 9, 13, 16, 19, 22, 25, 28, 29, 32, 35, 38, 40, 43, 46, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 95, 98, 101, 104, 107, 110, 113, 115,
      118, 121, 124, 127, 0, 0, 0, 0,
      /* 2nd row */
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -49, -52, -55, -58, -61,
      -64, -66, -69, -72, -75, -78, -81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      /* 3rd row */
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, -52, 55, -58, 61, -64,
      66, -69, 72, -75, 78, -81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0,
      /* 4th row */
      -3, 6, -9, 13, -16, 19, -22, 25, -28, 29, -32, 35, -38, 40, -43, 46, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -95, 98, -101, 104, -107, 110,
      -113, 115, -118, 121, -124, 127, 0, 0, 0, 0,
  };
  const int8_t quantized_matrix_values[] = {
      /* 1st row */
      3, 6, 9, 13, 16, 19, 22, 25, 28, 29, 32, 35, 38, 40, 43, 46, 95, 98, 101,
      104, 107, 110, 113, 115, 118, 121, 124, 127, 0, 0, 0, 0,
      /* 2nd row */
      -49, -52, -55, -58, -61, -64, -66, -69, -72, -75, -78, -81, 0, 0, 0, 0,
      /* 3rd row */
      49, -52, 55, -58, 61, -64, 66, -69, 72, -75, 78, -81, 0, 0, 0, 0,
      /* 4th row */
      -3, 6, -9, 13, -16, 19, -22, 25, -28, 29, -32, 35, -38, 40, -43, 46, -95,
      98, -101, 104, -107, 110, -113, 115, -118, 121, -124, 127, 0, 0, 0, 0,
  };
  uint8_t ledger[] = {
      2, 0,  2,  // 1st row
      1, 1,      // 2nd row
      1, 1,      // 3rd row
      2, 0,  2   // 4th row
  };

  float matrix_scaling_factor = 0.349921;

  const int8_t quantized_vector[] = {
      /* 1st batch */
      127, -127, 127, -127, 127, -127, 127, -127, 127, -127, 127, -127, 127,
      -127, 127, -127, 127, -127, 127, -127, 127, -127, 127, -127, 127, -127,
      127, -127, 127, -127, 127, -127, 127, -127, 127, -127, 127, -127, 127,
      -127, 127, -127, 127, -127, 127, -127, 127, -127,
      /* 2nd batch */
      106, 0, -89, 0, 127, 0, -55, 0, 55, 0, -47, 0, 85, 0, -72, 0, 80, 0,
      -64, 0, 21, 0, -30, 0, 34, 0, -13, 0, 119, 0, -119, 0, 47, -97, 80, -80,
      89, -21, 102, -4, 42, -106, 30, -80, 8, 1, 2, 3,
  };
  float vector_scaling_factor[2] = {0.00787402, 0.023622};

  /* clang-format on */
  float result_scaling_factor[2] = {
      matrix_scaling_factor * vector_scaling_factor[0],
      matrix_scaling_factor * vector_scaling_factor[1],
  };
  std::vector<float> dense_output(kRow * kBatch, 0.0);
  MatrixBatchVectorMultiplyAccumulate(quantized_matrix, kRow, kCol,
                                      quantized_vector, result_scaling_factor,
                                      kBatch, dense_output.data());

  EXPECT_THAT(dense_output,
              ElementsAreArray(ArrayFloatNear(
                  {-13.646927, 6.298582, 272.938538, -607.813110, -6.637464,
                   -9.381721, 9.381721, -713.845642})));

  std::vector<float> sparse_output(kRow * kBatch, 0.0);
  SparseMatrixBatchVectorMultiplyAccumulate(
      quantized_matrix_values, ledger, kRow, kCol, quantized_vector,
      result_scaling_factor, kBatch, sparse_output.data());

  EXPECT_THAT(sparse_output,
              ElementsAreArray(ArrayFloatNear(
                  {-13.646927, 6.298582, 272.938538, -607.813110, -6.637464,
                   -9.381721, 9.381721, -713.845642})));
}
#endif  // __ANDROID__

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

TEST(uKernels, VectorBatchVectorAddTest) {
  constexpr int kVectorSize = 3;
  constexpr int kBatchSize = 2;
  static float input[kVectorSize] = {0.0, -0.5, 1.0};
  std::vector<float> output = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  VectorBatchVectorAdd(input, kVectorSize, kBatchSize, output.data());
  EXPECT_THAT(output,
              testing::ElementsAreArray({1.0, 1.5, 4.0, 4.0, 4.5, 7.0}));
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

TEST(uKernels, Sub1VectorTest) {
  constexpr int kVectorSize = 5;
  static float input[kVectorSize] = {0.0, -0.5, 1.0, -1.5, 2.0};
  std::vector<float> output(kVectorSize);
  Sub1Vector(input, kVectorSize, output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear({1.0, 1.5, 0.0, 2.5, -1.0})));
}

TEST(uKernels, Sub1VectorInt16Test) {
  constexpr int kVectorSize = 30;
  static int16_t input[kVectorSize] = {
      32760, 300,   1,     2,    3, 4, 5, 6, 300, 1000,
      32767, 32000, 300,   1,    2, 3, 4, 5, 56,  300,
      1000,  32767, 32761, 1300, 1, 2, 3, 4, 5,   6,
  };
  std::vector<int16_t> output(kVectorSize);
  Sub1Vector(input, kVectorSize, output.data());
  EXPECT_THAT(
      output,
      testing::ElementsAreArray({
          7,     32467, 32766, 32765, 32764, 32763, 32762, 32761, 32467, 31767,
          0,     767,   32467, 32766, 32765, 32764, 32763, 32762, 32711, 32467,
          31767, 0,     6,     31467, 32766, 32765, 32764, 32763, 32762, 32761,
      }));
}

TEST(uKernels, VectorBatchVectorCwiseProductAccumulateInteger) {
  constexpr int kVectorSize = 29;
  constexpr int kBatchSize = 4;
  static int16_t vector[kVectorSize] = {-10, 9,  8,  7,  6,  5,  4,  3,  2, 1,
                                        0,   1,  2,  3,  4,  5,  6,  7,  8, 9,
                                        10,  11, 12, 13, 14, 15, 16, 17, 18};
  const std::vector<int16_t> batch_vector = {
      /* batch 0 */
      10, 11, 12, 13, 14, 15, 16, 17, 18, -10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1,
      2, 3, 4, 5, 6, 7, 8, 9,
      /* batch 1 */
      -10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1,
      2, 3, 4, 5, 6, 7, 8, 9,
      /* batch 2 */
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 11, 12,
      13, 14, 15, 16, 17, 18,
      /* batch 3 */
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 11, 12,
      13, 14, 15, 16, 17, 18};
  std::vector<int16_t> batch_output = {
      /* batch 0 */
      -10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1,
      2, 3, 4, 5, 6, 7, 8, 9,
      /* batch 1 */
      2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10, 9, 8, 7, 6, 5,
      4, 3, 2, 1, 10, 11, 12,
      /* batch 2 */
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 11, 12,
      13, 14, 15, 16, 17, 18,
      /* batch 3 */
      10, 11, 12, 13, 14, 15, 16, 17, 18, -10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1,
      13, 14, 15, 16, 17, 18};
  // Test with 0.25 scale, which is decomposed into (1073741824, -1).
  VectorBatchVectorCwiseProductAccumulate(vector, kVectorSize,
                                          batch_vector.data(), kBatchSize,
                                          1073741824, -1, batch_output.data());

  const std::vector<int16_t> expected_output = {
      /* batch 0 */
      -35, 34, 32, 30, 27, 24, 20, 16, 11, -2, 10, 13, 16, 18, 19, 20, 21, 21,
      20, 0, 4, 8, 12, 17, 23, 29, 35, 42, 50,
      /* batch 1 */
      27, 24, 20, 18, 15, 14, 12, 12, 1, 2, 2, 6, 10, 15, 20, 26, 32, 39, 26, 9,
      11, 13, 15, 18, 22, 26, 30, 35, 51,
      /* batch 2 */
      11, 15, 4, 7, 8, 10, 10, 11, 10, 10, 8, 12, -6, 15, 14, 14, 12, 11, 8, 6,
      27, 32, 46, 54, 61, 70, 78, 88, 97,
      /* batch 3 */
      17, 21, 14, 17, 18, 20, 20, 21, 20, 20, 18, -7, 13, 14, 13, 13, 11, 10, 7,
      5, 26, 31, 37, 56, 63, 72, 80, 90, 99};
  EXPECT_THAT(batch_output, testing::ElementsAreArray(expected_output));
}

TEST(uKernels, VectorBatchVectorCwiseProductAccumulateFloat) {
  constexpr int kVectorSize = 29;
  constexpr int kBatchSize = 4;
  static float input[kVectorSize] = {
      1.1f,   2.2f,   3.3f,   4.4f,   5.5f,   6.6f,   7.7f,   8.8f,
      9.9f,   10.10f, 11.11f, 12.12f, 13.13f, 14.14f, 15.15f, 16.16f,
      17.17f, 18.18f, 19.19f, 20.20f, 21.21f, 22.22f, 23.23f, 24.24f,
      25.25f, 26.26f, 27.27f, 28.28f, 0.0f};
  std::vector<float> output = {
      /* batch 0 */
      1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f, 10.10f, 11.11f,
      12.12f, 13.13f, 14.14f, 15.15f, 16.16f, 17.17f, 18.18f, 19.19f, 20.20f,
      21.21f, 22.22f, 23.23f, 24.24f, 25.25f, 26.26f, 27.27f, 28.28f, 0.0f,
      /* batch 1 */
      -1.1f, -2.2f, -3.3f, -4.4f, -5.5f, -6.6f, -7.7f, -8.8f, -9.9f, -10.10f,
      -11.11f, -12.12f, -13.13f, -14.14f, -15.15f, -16.16f, -17.17f, -18.18f,
      -19.19f, -20.20f, -21.21f, -22.22f, -23.23f, -24.24f, -25.25f, -26.26f,
      -27.27f, -28.28f, 0.0f,
      /* batch 2 */
      1.1f, -2.2f, 3.3f, -4.4f, 5.5f, -6.6f, 7.7f, -8.8f, 9.9f, -10.10f, 11.11f,
      -12.12f, 13.13f, -14.14f, 15.15f, -16.16f, 17.17f, -18.18f, 19.19f,
      -20.20f, 21.21f, -22.22f, 23.23f, -24.24f, 25.25f, -26.26f, 27.27f,
      -28.28f, 0.0f,
      /* batch 3 */
      -1.1f, 2.2f, -3.3f, 4.4f, -5.5f, 6.6f, -7.7f, 8.8f, -9.9f, 10.10f,
      -11.11f, 12.12f, -13.13f, 14.14f, -15.15f, 16.16f, -17.17f, 18.18f,
      -19.19f, 20.20f, -21.21f, 22.22f, -23.23f, 24.24f, -25.25f, 26.26f,
      -27.27f, 28.28f, 0.0f};
  VectorBatchVectorCwiseProductAccumulate(input, kVectorSize, output.data(),
                                          kBatchSize, output.data());

  // Expect output = input * output + output.
  const std::vector<float> expected_output = {
      /* batch 0 */
      2.31f, 7.04f, 14.19f, 23.76f, 35.75f, 50.16f, 66.99f, 86.24f, 107.91f,
      112.11f, 134.5421f, 159.0144f, 185.5269f, 214.0796f, 244.6725f, 277.3056f,
      311.9789f, 348.6924f, 387.4461f, 428.24f, 471.0741f, 515.9484f, 562.8629f,
      611.8176f, 662.8125f, 715.8476f, 770.9229f, 828.0384f, 0.0f,
      /* batch 1 */
      -2.31f, -7.04f, -14.19f, -23.76f, -35.75f, -50.16f, -66.99f, -86.24f,
      -107.91f, -112.11f, -134.5421f, -159.0144f, -185.5269f, -214.0796f,
      -244.6725f, -277.3056f, -311.9789f, -348.6924f, -387.4461f, -428.24f,
      -471.0741f, -515.9484f, -562.8629f, -611.8176f, -662.8125f, -715.8476f,
      -770.9229f, -828.0384f, 0.0f,
      /* batch 2 */
      2.31f, -7.04f, 14.19f, -23.76f, 35.75f, -50.16f, 66.99f, -86.24f, 107.91f,
      -112.11f, 134.5421f, -159.0144f, 185.5269f, -214.0796f, 244.6725f,
      -277.3056f, 311.9789f, -348.6924f, 387.4461f, -428.24f, 471.0741f,
      -515.9484f, 562.8629f, -611.8176f, 662.8125f, -715.8476f, 770.9229f,
      -828.0384f, 0.0f,
      /* batch 3 */
      -2.31f, 7.04f, -14.19f, 23.76f, -35.75f, 50.16f, -66.99f, 86.24f,
      -107.91f, 112.11f, -134.5421f, 159.0144f, -185.5269f, 214.0796f,
      -244.6725f, 277.3056f, -311.9789f, 348.6924f, -387.4461f, 428.24f,
      -471.0741f, 515.9484f, -562.8629f, 611.8176f, -662.8125f, 715.8476f,
      -770.9229f, 828.0384f, 0.0f};
  EXPECT_THAT(output, testing::ElementsAreArray(
                          ArrayFloatNear(expected_output, 6.5e-5f)));
}

TEST(uKernels, VectorBatchVectorCwiseProductNoAccumulate) {
  constexpr int kVectorSize = 29;
  constexpr int kBatchSize = 4;
  static float input[kVectorSize] = {
      1.1,   2.2,   3.3,   4.4,   5.5,   6.6,   7.7,   8.8,   9.9,   10.1,
      11.11, 12.12, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19, 20.2,
      21.21, 22.22, 23.23, 24.24, 25.25, 26.26, 27.27, 28.28, 0};
  std::vector<float> output = {
      /* batch 0 */
      1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12, 13.13,
      14.14, 15.15, 16.16, 17.17, 18.18, 19.19, 20.2, 21.21, 22.22, 23.23,
      24.24, 25.25, 26.26, 27.27, 28.28, 0,
      /* batch 1 */
      -1.1, -2.2, -3.3, -4.4, -5.5, -6.6, -7.7, -8.8, -9.9, -10.1, -11.11,
      -12.12, -13.13, -14.14, -15.15, -16.16, -17.17, -18.18, -19.19, -20.2,
      -21.21, -22.22, -23.23, -24.24, -25.25, -26.26, -27.27, -28.28, 0,
      /* batch 2 */
      1.1, -2.2, 3.3, -4.4, 5.5, -6.6, 7.7, -8.8, 9.9, -10.1, 11.11, -12.12,
      13.13, -14.14, 15.15, -16.16, 17.17, -18.18, 19.19, -20.2, 21.21, -22.22,
      23.23, -24.24, 25.25, -26.26, 27.27, -28.28, 0,
      /* batch 3 */
      -1.1, 2.2, -3.3, 4.4, -5.5, 6.6, -7.7, 8.8, -9.9, 10.1, -11.11, 12.12,
      -13.13, 14.14, -15.15, 16.16, -17.17, 18.18, -19.19, 20.2, -21.21, 22.22,
      -23.23, 24.24, -25.25, 26.26, -27.27, 28.28, 0};
  VectorBatchVectorCwiseProduct(input, kVectorSize, output.data(), kBatchSize,
                                output.data());

  // Expect output = input * output + output.
  const std::vector<float> expected_output = {
      /* batch 0 */
      1.210000, 4.840000, 10.889999, 19.360001, 30.250000, 43.559998, 59.289997,
      77.440002, 98.009995, 102.010010, 123.432091, 146.894394, 172.396896,
      199.939606, 229.522491, 261.145599, 294.808899, 330.512421, 368.256134,
      408.040039, 449.864075, 493.728363, 539.632874, 587.577576, 637.562500,
      689.587585, 743.652954, 799.758423, 0.000000,
      /* batch 1 */
      -1.210000, -4.840000, -10.889999, -19.360001, -30.250000, -43.559998,
      -59.289997, -77.440002, -98.009995, -102.010010, -123.432091, -146.894394,
      -172.396896, -199.939606, -229.522491, -261.145599, -294.808899,
      -330.512421, -368.256134, -408.040039, -449.864075, -493.728363,
      -539.632874, -587.577576, -637.562500, -689.587585, -743.652954,
      -799.758423, 0.000000,
      /* batch 2 */
      1.210000, -4.840000, 10.889999, -19.360001, 30.250000, -43.559998,
      59.289997, -77.440002, 98.009995, -102.010010, 123.432091, -146.894394,
      172.396896, -199.939606, 229.522491, -261.145599, 294.808899, -330.512421,
      368.256134, -408.040039, 449.864075, -493.728363, 539.632874, -587.577576,
      637.562500, -689.587585, 743.652954, -799.758423, 0.000000,
      /* batch 3 */
      -1.210000, 4.840000, -10.889999, 19.360001, -30.250000, 43.559998,
      -59.289997, 77.440002, -98.009995, 102.010010, -123.432091, 146.894394,
      -172.396896, 199.939606, -229.522491, 261.145599, -294.808899, 330.512421,
      -368.256134, 408.040039, -449.864075, 493.728363, -539.632874, 587.577576,
      -637.562500, 689.587585, -743.652954, 799.758423, 0.000000};
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
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
                                   output.data());
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear({0.5, 1.75})));
}

TEST(uKernels, BatchVectorBatchVectorDotProductIntegerTest) {
  constexpr int kVectorSize = 5;
  constexpr int kBatch = 2;
  static int16_t input1[kVectorSize * kBatch] = {0,   5,  10,  -15, 20,
                                                 -25, 30, -35, 40,  -45};
  static int16_t input2[kVectorSize * kBatch] = {1,  -1, 1,  -1, 1,
                                                 -1, 1,  -1, 1,  1};
  std::vector<int32_t> output(kBatch);
  BatchVectorBatchVectorDotProduct(input1, input2, kVectorSize, kBatch,
                                   output.data());
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear({40, 85})));
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

TEST(uKernels, ReductionSumVectorIntegerTest) {
  constexpr int kInputVectorSize = 10;
  constexpr int kOutputVectorSize1 = 5;
  constexpr int kReductionSize1 = 2;
  static int32_t input[kInputVectorSize] = {1, 2, 1, 5, -3, 2, 1, 2, 5, 10};
  std::vector<int32_t> result1(kOutputVectorSize1);
  ReductionSumVector(input, result1.data(), kOutputVectorSize1,
                     kReductionSize1);
  EXPECT_THAT(result1, testing::ElementsAreArray({3, 6, -1, 3, 15}));
}

void TwoGateSaturationgAdd(const int8_t* input, int8_t input_zp,
                           const int8_t* recurrent, int8_t recurrent_zp,
                           int32_t input_effective_scale_a,
                           int32_t input_effective_scale_b,
                           int32_t recurrent_effective_scale_a,
                           int32_t recurrent_effective_scale_b, int32_t n_batch,
                           int32_t n_cell, int16_t* output);

TEST(uKernels, TwoGateSaturateAddTest) {
  const std::vector<int8_t> input1 = {1, 2, 3, 4, 55, 66, 77};
  const std::vector<int8_t> input2 = {100, 2, 3, 4, 55, 66, 77};
  const int32_t input1_zp = 10;
  const int32_t input2_zp = -5;
  const int32_t multiplier1 = 1347771520;
  const int32_t shift1 = -7;
  const int32_t multiplier2 = 1047577121;
  const int32_t shift2 = -6;
  std::vector<int16_t> output(7);

  TwoGateSaturationgAdd(input1.data(), input1_zp, input2.data(), input2_zp,
                        multiplier1, shift1, multiplier2, shift2, 1, 7,
                        output.data());

  const std::vector<int16_t> expected_output = {1, 0, 0, 0, 0, 1, 1};
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

namespace {
// Parameterized test: mean, difference, tolerance.
// Input is constructed as [mean-2*diff, mean-diff, mean+diff, mean+2*diff]
class MeanStddevNormalizationTest
    : public testing::TestWithParam<std::tuple<float, float, float>> {};
}  // namespace

TEST_P(MeanStddevNormalizationTest, SeparateBatches) {
  const float mean = std::get<0>(GetParam());
  const float diff = std::get<1>(GetParam());
  const float tolerance = std::get<2>(GetParam());

  constexpr int kVectorSize = 4;
  const float input[kVectorSize] = {mean - 2 * diff, mean - diff, mean + diff,
                                    mean + 2 * diff};
  float output[kVectorSize];
  MeanStddevNormalization(input, output, kVectorSize, 1);
  std::vector<float> expected_output;
  if (diff == 0.0f) {
    expected_output.assign({0.0f, 0.0f, 0.0f, 0.0f});
  } else {
    const float ksqrt16 = std::sqrt(1.6f);
    const float ksqrt04 = std::sqrt(0.4f);
    expected_output.assign({-ksqrt16, -ksqrt04, ksqrt04, ksqrt16});
  }
  EXPECT_THAT(output, testing::ElementsAreArray(
                          ArrayFloatNear(expected_output, tolerance)));
}

INSTANTIATE_TEST_SUITE_P(
    uKernels, MeanStddevNormalizationTest,
    testing::Values(
        std::make_tuple(0.0f, 0.0f, 0.0f),         // zero mean, zero variance
        std::make_tuple(0.0f, 0.01f, 2.53e-5f),    // zero mean, small variance
        std::make_tuple(0.0f, 100.0f, 1.20e-7f),   // zero mean, large variance
        std::make_tuple(0.01f, 0.0f, 0.0f),        // small mean, zero variance
        std::make_tuple(0.01f, 0.01f, 2.53e-5f),   // small mean, small variance
        std::make_tuple(0.01f, 100.0f, 1.20e-7f),  // small mean, large variance
        std::make_tuple(100.0f, 0.0f, 0.0f),       // large mean, zero variance
        std::make_tuple(100.0f, 0.01f, 1.81e-4f),  // large mean, small variance
        std::make_tuple(100.0f, 100.0f, 1.20e-7f)  // large mean, large variance
        ));

TEST(uKernels, MeanStddevNormalizationAllBatches) {
  constexpr int kVectorSize = 4;
  constexpr int kBatchSize = 9;

  // None-zero input.
  static float input[kVectorSize * kBatchSize] = {
      0.0f,     0.0f,    0.0f,    0.0f,     // zero mean, zero variance
      -0.02f,   -0.01f,  0.01f,   0.02f,    // zero mean, small variance
      -200.0f,  -100.0f, 100.0f,  200.0f,   // zero mean, large variance
      0.01f,    0.01f,   0.01f,   0.01f,    // small mean, zero variance
      -0.01f,   0.0f,    0.02f,   0.03f,    // small mean, small variance
      -199.99f, -99.99f, 100.01f, 200.01f,  // small mean, large variance
      100.0f,   100.0f,  100.0f,  100.0f,   // large mean, zero variance
      99.98f,   99.99f,  100.01f, 100.02f,  // large mean, small variance
      -100.0f,  0.0f,    200.0f,  300.0f,   // large mean, large variance
  };
  float output[kVectorSize * kBatchSize];
  MeanStddevNormalization(input, output, kVectorSize, kBatchSize);
  const float ksqrt16 = std::sqrt(1.6f);
  const float ksqrt04 = std::sqrt(0.4f);
  const std::vector<float> expected_output = {
      0.0f,     0.0f,     0.0f,    0.0f,     // zero mean, zero variance
      -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // zero mean, small variance
      -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // zero mean, large variance
      0.0f,     0.0f,     0.0f,    0.0f,     // small mean, zero variance
      -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // small mean, small variance
      -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // small mean, large variance
      0.0f,     0.0f,     0.0f,    0.0f,     // large mean, zero variance
      -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // large mean, small variance
      -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // large mean, large variance
  };
  EXPECT_THAT(output, testing::ElementsAreArray(
                          ArrayFloatNear(expected_output, 1.81e-4f)));
}

}  // namespace tensor_utils
}  // namespace tflite

#ifdef DOTPROD_BENCHMARKS

// Compile with --copt="-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1" and
// --copt="-DDOTPROD_BENCHMARKS"
// Run with --benchmarks=all
void BM_DotprodBatchOneMultiply(benchmark::State& state) {
  const int rows = state.range(0);
  const int cols = state.range(1);
  const int batch = state.range(2);
  const int copies = state.range(3);

  // For some benchmarks we make multiple matrix copies. This allows us to
  // measure the performance differences of being entirely in cache vs.
  // out of cache.
  std::vector<tflite::tensor_utils::MatrixVectorData> datas;
  for (int i = 0; i < copies; i++) {
    datas.push_back(
        tflite::tensor_utils::SetupMatrixVectorData(rows, cols, batch));
  }

  int copy = 0;
  for (auto _ : state) {
    copy = (copy + 1) % datas.size();
    auto& data = datas[copy];
    for (int i = 0; i < batch; i++) {
      tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          data.matrix.data(), data.rows, data.cols,
          data.vectors.data() + (data.cols * i), data.scale_factors.data(), 1,
          &data.results[0], 1);
      testing::DoNotOptimize(data.results[2]);
    }
  }
}
BENCHMARK(BM_DotprodBatchOneMultiply)
    ->Args({16, 16, 1, 1})
    ->Args({16, 16, 4, 1})
    ->Args({32, 32, 1, 1})
    ->Args({32, 32, 4, 1})
    ->Args({64, 64, 1, 1})
    ->Args({64, 64, 4, 1})
    ->Args({128, 128, 1, 1})
    ->Args({128, 128, 4, 1})
    ->Args({992, 992, 1, 1})
    ->Args({992, 992, 8, 1})
    ->Args({1024, 1024, 1, 1})
    ->Args({1024, 1024, 1, 8})
    ->Args({1024, 1024, 4, 1})
    ->Args({1024, 1024, 4, 8})
    ->Args({1024, 1024, 8, 1})
    ->Args({640, 2048, 1, 1})
    ->Args({640, 2048, 4, 1})
    ->Args({640, 2048, 8, 1})
    ->Args({640, 2048, 8, 8})
    ->Args({2048, 2048, 1, 1})
    ->Args({2048, 2048, 1, 8})
    ->Args({2048, 2048, 8, 1});

void BM_DotprodBatchFourMultiply(benchmark::State& state) {
  const int rows = state.range(0);
  const int cols = state.range(1);
  const int batch = state.range(2);
  const int copies = state.range(3);

  // For some benchmarks we make multiple matrix copies. This allows us to
  // measure the performance differences of being entirely in cache vs.
  // out of cache.
  std::vector<tflite::tensor_utils::MatrixVectorData> datas;
  for (int i = 0; i < copies; i++) {
    datas.push_back(
        tflite::tensor_utils::SetupMatrixVectorData(rows, cols, batch));
  }

  int copy = 0;
  for (auto _ : state) {
    copy = (copy + 1) % datas.size();
    auto& data = datas[copy];
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        data.matrix.data(), data.rows, data.cols, data.vectors.data(),
        data.scale_factors.data(), data.batch, &data.results[0], 1);
    testing::DoNotOptimize(data.results[2]);
  }
}
BENCHMARK(BM_DotprodBatchFourMultiply)
    ->Args({16, 16, 4, 1})
    ->Args({32, 32, 4, 1})
    ->Args({64, 64, 4, 1})
    ->Args({64, 256, 64, 1})
    ->Args({64, 256, 256, 1})
    ->Args({64, 256, 1024, 1})
    ->Args({64, 256, 12544, 1})
    ->Args({128, 128, 2, 1})
    ->Args({128, 128, 3, 1})
    ->Args({128, 128, 4, 1})
    ->Args({128, 128, 5, 1})
    ->Args({640, 640, 4, 1})
    ->Args({992, 992, 8, 1})
    ->Args({1024, 1024, 2, 1})
    ->Args({1024, 1024, 3, 1})
    ->Args({1024, 1024, 4, 1})
    ->Args({1024, 1024, 5, 1})
    ->Args({1024, 1024, 8, 1})
    ->Args({1024, 1024, 8, 8})
    ->Args({1024, 1024, 256, 1})
    ->Args({640, 2048, 2, 1})
    ->Args({640, 2048, 3, 1})
    ->Args({640, 2048, 4, 1})
    ->Args({640, 2048, 4, 8})
    ->Args({640, 2048, 8, 1})
    ->Args({2048, 2048, 3, 1})
    ->Args({2048, 2048, 4, 1})
    ->Args({2048, 2048, 4, 8})
    ->Args({2048, 2048, 5, 1})
    ->Args({2048, 2048, 8, 1});

void BM_DotprodSparseMultiply(benchmark::State& state) {
  const int rows = state.range(0);
  const int cols = state.range(1);
  const int batch = state.range(2);

  const int copies = state.range(3);

  // For some benchmarks we make multiple matrix copies. This allows us to
  // measure the performance differences of being entirely in cache vs.
  // out of cache.
  std::vector<tflite::tensor_utils::MatrixVectorData> datas;
  for (int i = 0; i < copies; i++) {
    datas.push_back(
        tflite::tensor_utils::SetupMatrixVectorData(rows, cols, batch));
  }

  int copy = 0;
  for (auto _ : state) {
    copy = (copy + 1) % datas.size();
    auto& data = datas[copy];
    tflite::tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate(
        data.sparse_matrix.data(), data.ledger.data(), data.rows, data.cols,
        data.vectors.data(), data.scale_factors.data(), data.batch,
        &data.results[0]);
    testing::DoNotOptimize(data.results[2]);
  }
}
BENCHMARK(BM_DotprodSparseMultiply)
    ->Args({128, 128, 1, 1})
    ->Args({128, 128, 4, 1})
    ->Args({640, 640, 4, 1})
    ->Args({992, 992, 8, 1})
    ->Args({1024, 1024, 1, 1})
    ->Args({1024, 1024, 4, 1})
    ->Args({1024, 1024, 8, 1})
    ->Args({640, 2048, 1, 1})
    ->Args({640, 2048, 4, 1})
    ->Args({640, 2048, 8, 1})
    ->Args({2048, 2048, 1, 1})
    ->Args({2048, 2048, 8, 1});

#endif  // DOTPROD_BENCHMARKS
