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

  int rows;
  int cols;
  int batch;
};

MatrixVectorData SetupMatrixVectorData(int rows, int cols, int batch,
                                       bool negative = false) {
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
  data.results.resize(rows * batch, 0);

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
  return data;
}

std::vector<float> TestDotprodMatrixBatchVectorMultiply(int rows, int cols,
                                                        int batch,
                                                        bool negative = false) {
  MatrixVectorData data = SetupMatrixVectorData(rows, cols, batch, negative);

  // All partial sums in this computation are small enough to fit in the
  // mantissa of a float, and the scale factors are all integers, so we expect
  // an exact result.
  MatrixBatchVectorMultiplyAccumulate(
      data.matrix.data(), rows, cols, data.vectors.data(),
      data.scale_factors.data(), batch, &data.results[0], 1);
  return data.results;
}

std::vector<float> TestSparseDotprodMatrixBatchVectorMultiply(
    int rows, int cols, int batch, bool negative = false) {
  MatrixVectorData data = SetupMatrixVectorData(rows, cols, batch, negative);
  SparseMatrixBatchVectorMultiplyAccumulate(
      data.sparse_matrix.data(), data.ledger.data(), rows, cols,
      data.vectors.data(), data.scale_factors.data(), batch, &data.results[0],
      1);
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
                                      dense_output.data(), /*result_stride=*/1);

  EXPECT_THAT(dense_output, ElementsAreArray(ArrayFloatNear(
                                {-13.69, 6.06001, 272.7, -608.03, -9.66602,
                                 -10.201, 10.201, -713.897949},
                                1e-4)));

  std::vector<float> sparse_output(kRow * kBatch, 0.0);
  SparseMatrixBatchVectorMultiplyAccumulate(
      matrix_values, ledger, kRow, kCol, vector, kBatch, sparse_output.data(),
      /*result_stride=*/1);

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
                                      kBatch, dense_output.data(),
                                      /*result_stride=*/1);

  EXPECT_THAT(dense_output,
              ElementsAreArray(ArrayFloatNear(
                  {-13.646927, 6.298582, 272.938538, -607.813110, -6.637464,
                   -9.381721, 9.381721, -713.845642})));

  std::vector<float> sparse_output(kRow * kBatch, 0.0);
  SparseMatrixBatchVectorMultiplyAccumulate(
      quantized_matrix_values, ledger, kRow, kCol, quantized_vector,
      result_scaling_factor, kBatch, sparse_output.data(),
      /*result_stride=*/1);

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

TEST(uKernels, VectorBatchVectorCwiseProductAccumulate) {
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
  VectorBatchVectorCwiseProductAccumulate(input, kVectorSize, output.data(),
                                          kBatchSize, output.data());

  // Expect output = input * output + output.
  const std::vector<float> expected_output = {
      /* batch 0 */
      2.310000, 7.040000, 14.190000, 23.760000, 35.750000, 50.159996, 66.989998,
      86.240005, 107.909996, 112.110008, 134.542084, 159.014389, 185.526901,
      214.079605, 244.672485, 277.305603, 311.978912, 348.692413, 387.446136,
      428.240051, 471.074066, 515.948364, 562.862854, 611.817566, 662.812500,
      715.847595, 770.922974, 828.038452, 0.000000,
      /* batch 1 */
      -2.310000, -7.040000, -14.190000, -23.760000, -35.750000, -50.159996,
      -66.989998, -86.240005, -107.909996, -112.110008, -134.542084,
      -159.014389, -185.526901, -214.079605, -244.672485, -277.305603,
      -311.978912, -348.692413, -387.446136, -428.240051, -471.074066,
      -515.948364, -562.862854, -611.817566, -662.812500, -715.847595,
      -770.922974, -828.038452, 0.000000,
      /* batch 2 */
      2.310000, -7.040000, 14.190000, -23.760000, 35.750000, -50.159996,
      66.989998, -86.240005, 107.909996, -112.110008, 134.542084, -159.014389,
      185.526901, -214.079605, 244.672485, -277.305603, 311.978912, -348.692413,
      387.446136, -428.240051, 471.074066, -515.948364, 562.862854, -611.817566,
      662.812500, -715.847595, 770.922974, -828.038452, 0.000000,
      /* batch 3 */
      -2.310000, 7.040000, -14.190000, 23.760000, -35.750000, 50.159996,
      -66.989998, 86.240005, -107.909996, 112.110008, -134.542084, 159.014389,
      -185.526901, 214.079605, -244.672485, 277.305603, -311.978912, 348.692413,
      -387.446136, 428.240051, -471.074066, 515.948364, -562.862854, 611.817566,
      -662.812500, 715.847595, -770.922974, 828.038452, 0.000000};
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
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

TEST(uKernels, MeanStddevNormalizationNoneZeroInput) {
  constexpr int kVectorSize = 4;
  constexpr int kBatchSize = 2;
  constexpr float kNormalizationEpsilon = 1e-8;

  // None-zero input.
  static float input[kVectorSize * kBatchSize] = {
      0.1, 0.2, 0.3, 0.4,  // batch 0
      0.9, 1.0, 1.1, 1.2,  // batch 1
  };
  std::vector<float> output(kVectorSize * kBatchSize);
  MeanStddevNormalization(input, output.data(), kVectorSize, kBatchSize,
                          kNormalizationEpsilon);
  const std::vector<float> expected_output = {
      -1.34164071, -0.447213531, 0.44721365,  1.34164071,  // batch 0
      -1.34163153, -0.447210163, 0.447211236, 1.3416326,   // batch 1
  };
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

TEST(uKernels, MeanStddevNormalizationAllZeroInput) {
  constexpr int kVectorSize = 4;
  constexpr int kBatchSize = 2;
  constexpr float kNormalizationEpsilon = 1e-8;

  // Zero input.
  static float input[kVectorSize * kBatchSize] = {
      0.0, 0.0, 0.0, 0.0,  // batch 0
      0.0, 0.0, 0.0, 0.0,  // batch 1
  };
  std::vector<float> output(kVectorSize * kBatchSize);
  MeanStddevNormalization(input, output.data(), kVectorSize, kBatchSize,
                          kNormalizationEpsilon);
  const std::vector<float> expected_output = {
      0.0, 0.0, 0.0, 0.0,  // batch 0
      0.0, 0.0, 0.0, 0.0,  // batch 1
  };
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

TEST(uKernels, MeanStddevNormalizationMixed) {
  constexpr int kVectorSize = 4;
  constexpr int kBatchSize = 2;
  constexpr float kNormalizationEpsilon = 1e-8;

  // Mix of zero and non-zero input.
  static float input[kVectorSize * kBatchSize] = {
      0.0, 0.0, 0.0, 0.0,  // batch 0
      0.1, 0.2, 0.3, 0.4,  // batch 1
  };
  std::vector<float> output(kVectorSize * kBatchSize);
  MeanStddevNormalization(input, output.data(), kVectorSize, kBatchSize,
                          kNormalizationEpsilon);
  const std::vector<float> expected_output = {
      0.0,         0.0,          0.0,        0.0,         // batch 0
      -1.34164071, -0.447213531, 0.44721365, 1.34164071,  // batch 1
  };
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
}

TEST(uKernels, MeanStddevNormalizationSmallValue) {
  constexpr int kVectorSize = 4;
  constexpr int kBatchSize = 2;
  constexpr float kNormalizationEpsilon = 1e-8;

  // Mix of zero and non-zero input.
  static float input[kVectorSize * kBatchSize] = {
      3e-5, -7e-6, -9e-5, 1e-6,  // batch 0
      4e-5, 9e-6,  2e-4,  0.0,   // batch 1
  };
  std::vector<float> output(kVectorSize * kBatchSize);
  MeanStddevNormalization(input, output.data(), kVectorSize, kBatchSize,
                          kNormalizationEpsilon);
  const std::vector<float> expected_output = {
      1.04231524,   0.212946132,  -1.64753067, 0.392269224,   // batch 0
      -0.275023013, -0.658201098, 1.70267045,  -0.769446373,  // batch 1
  };
  EXPECT_THAT(output, testing::ElementsAreArray(expected_output));
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

  tflite::tensor_utils::MatrixVectorData data =
      tflite::tensor_utils::SetupMatrixVectorData(rows, cols, batch);
  for (auto _ : state) {
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
    ->Args({16, 16, 1})
    ->Args({16, 16, 4})
    ->Args({32, 32, 1})
    ->Args({32, 32, 4})
    ->Args({64, 64, 1})
    ->Args({64, 64, 4})
    ->Args({128, 128, 1})
    ->Args({128, 128, 4})
    ->Args({992, 992, 1})
    ->Args({992, 992, 8})
    ->Args({1024, 1024, 1})
    ->Args({1024, 1024, 4})
    ->Args({1024, 1024, 8})
    ->Args({640, 2048, 1})
    ->Args({640, 2048, 4})
    ->Args({640, 2048, 8})
    ->Args({2048, 2048, 1})
    ->Args({2048, 2048, 8});

void BM_DotprodBatchFourMultiply(benchmark::State& state) {
  const int rows = state.range(0);
  const int cols = state.range(1);
  const int batch = state.range(2);

  tflite::tensor_utils::MatrixVectorData data =
      tflite::tensor_utils::SetupMatrixVectorData(rows, cols, batch);
  for (auto _ : state) {
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        data.matrix.data(), data.rows, data.cols, data.vectors.data(),
        data.scale_factors.data(), data.batch, &data.results[0], 1);
    testing::DoNotOptimize(data.results[2]);
  }
}
BENCHMARK(BM_DotprodBatchFourMultiply)
    ->Args({16, 16, 4})
    ->Args({32, 32, 4})
    ->Args({64, 64, 4})
    ->Args({128, 128, 4})
    ->Args({640, 640, 4})
    ->Args({992, 992, 8})
    ->Args({1024, 1024, 4})
    ->Args({1024, 1024, 8})
    ->Args({640, 2048, 4})
    ->Args({640, 2048, 8})
    ->Args({2048, 2048, 4})
    ->Args({2048, 2048, 8});

void BM_DotprodSparseMultiply(benchmark::State& state) {
  const int rows = state.range(0);
  const int cols = state.range(1);
  const int batch = state.range(2);

  tflite::tensor_utils::MatrixVectorData data =
      tflite::tensor_utils::SetupMatrixVectorData(rows, cols, batch);
  for (auto _ : state) {
    tflite::tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate(
        data.sparse_matrix.data(), data.ledger.data(), data.rows, data.cols,
        data.vectors.data(), data.scale_factors.data(), data.batch,
        &data.results[0], 1);
    testing::DoNotOptimize(data.results[2]);
  }
}
BENCHMARK(BM_DotprodSparseMultiply)
    ->Args({128, 128, 1})
    ->Args({128, 128, 4})
    ->Args({640, 640, 4})
    ->Args({992, 992, 8})
    ->Args({1024, 1024, 1})
    ->Args({1024, 1024, 4})
    ->Args({1024, 1024, 8})
    ->Args({640, 2048, 1})
    ->Args({640, 2048, 4})
    ->Args({640, 2048, 8})
    ->Args({2048, 2048, 1})
    ->Args({2048, 2048, 8});

#endif  // DOTPROD_BENCHMARKS
