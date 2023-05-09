/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/optimized/fully_connected_4bit.h"

namespace tflite {
namespace {

std::mt19937 random_engine(2023);
std::uniform_real_distribution<float> real_dist(0.f, 1.f);
std::uniform_int_distribution<int32_t> int_dist(-7, 7);

struct TestPack {
  TestPack(std::vector<int8_t> src_data, int src_rows, int src_cols, int width,
           int depth)
      : src_data(src_data),
        src_rows(src_rows),
        src_cols(src_cols),
        width(width),
        depth(depth) {
    rows = (src_rows + (width - 1)) & ~(width - 1);
    cols = (src_cols + (depth - 1)) & ~(depth - 1);
  }

  ~TestPack() { free(packed_data); }
  void Prepack() {
    optimized_4bit::Prepack(&packed_data, src_data.data(), rows, cols, src_rows,
                            src_cols, width, depth);
  }

  std::vector<uint8_t> AsVector() {
    int size = rows * cols / 2;
    std::vector<uint8_t> values(size);
    for (int i = 0; i < size; i++) {
      values[i] = packed_data[i];
    }
    return values;
  }

  std::vector<int8_t> src_data;
  uint8_t* packed_data;
  int src_rows;
  int src_cols;
  int rows;
  int cols;
  int width;
  int depth;
};

class RunPackTests
    : public ::testing::TestWithParam<::testing::tuple<int, int>> {};

TEST_P(RunPackTests, RunPackTests) {
  auto params = GetParam();
  int src_rows = std::get<0>(params);
  int src_cols = std::get<1>(params);
  std::vector<int8_t> test_data;
  test_data.reserve(src_rows * src_cols / 2);
  for (int i = 0; i < src_rows; ++i) {
    int stride = optimized_4bit::FilterDepth / 4;
    int strides = src_cols / stride / 2;
    int v = -7;
    int l = 0;
    for (int j = 0; j < strides; j++) {
      for (int k = 0; k < stride; k++) {  // 8
        int lower = static_cast<uint8_t>(v) & UINT8_C(15);
        int upper = static_cast<uint8_t>(v) << 4;
        test_data.push_back(upper | lower);
        l++;
      }
      v++;
    }
    while (l < (src_cols / 2)) {
      int lower = static_cast<uint8_t>(v) & UINT8_C(15);
      test_data.push_back(lower << 4 | lower);
      l++;
    }
  }
  TestPack test(test_data, src_rows, src_cols, optimized_4bit::FilterWidth,
                optimized_4bit::FilterDepth);
  test.Prepack();
  std::vector<uint8_t> result = test.AsVector();
  int outer_rows = test.rows / optimized_4bit::FilterWidth;
  int outer_cols = test.cols / optimized_4bit::FilterDepth;
  int k = 0;
  for (int i = 0; i < outer_rows; ++i) {
    int v = -7;
    for (int j = 0; j < outer_cols; ++j) {
      for (int w = 0; w < optimized_4bit::FilterWidth; w++) {
        int c = 0;
        for (; c < optimized_4bit::FilterDepth / 2; c++) {
          uint8_t res = result[k++];
          uint8_t res0 = res >> 4;
          uint8_t res1 = res & UINT8_C(15);
          int res00 = res0 - 7;
          int res11 = res1 - 7;
          if ((i * optimized_4bit::FilterWidth + w) < src_rows) {
            if ((j * optimized_4bit::FilterDepth / 2 + c) < src_cols / 2) {
              EXPECT_EQ(res00, v % 8);
            }
            if ((j * optimized_4bit::FilterDepth / 2 + c + 16) < src_cols / 2) {
              EXPECT_EQ(res11, v + 1 % 8);
            }
          }
        }
      }
      v += 2;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(RunPackTests, RunPackTests,
                         ::testing::ValuesIn({
                             std::make_tuple(4, 32),
                             std::make_tuple(4, 46),
                             std::make_tuple(5, 64),
                             std::make_tuple(5, 72),
                         }));

struct TestQuantize {
  TestQuantize(std::vector<float> src_data, int src_rows, int src_cols,
               int width, int depth)
      : src_data(src_data),
        src_rows(src_rows),
        src_cols(src_cols),
        width(width),
        depth(depth) {
    rows = (src_rows + (width - 1)) & ~(width - 1);
    cols = (src_cols + (depth - 1)) & ~(depth - 1);
    scaling_factors.assign(rows, 1.0);
    input_offsets.assign(rows, 0);
    output_data.assign(rows * cols, 0);
  }

  void BatchQuantizeFloats4Bit() {
    optimized_4bit::BatchQuantizeFloats4Bit(
        src_data.data(), src_rows, src_cols, output_data.data(),
        scaling_factors.data(), width, depth, input_offsets.data());
  }

  std::vector<float> src_data;
  int src_rows;
  int src_cols;
  int rows;
  int cols;
  int width;
  int depth;
  std::vector<int8_t> output_data;
  std::vector<float> scaling_factors;
  std::vector<int32_t> input_offsets;
};

class RunQuantizeInputTests
    : public ::testing::TestWithParam<::testing::tuple<int, int, int>> {};

TEST_P(RunQuantizeInputTests, RunQuantizeInputsTests) {
  auto params = GetParam();
  int width = std::get<0>(params);
  int src_rows = std::get<1>(params);
  int src_cols = std::get<2>(params);
  std::vector<float> test_data;
  test_data.reserve(src_rows * src_cols);

  float v = -127.0;
  for (int i = 0; i < src_rows; ++i) {
    for (int j = 0; j < src_cols; ++j) {
      test_data.push_back(v / (i + 1));
      v = -v;
    }
  }
  TestQuantize test(test_data, src_rows, src_cols, width,
                    optimized_4bit::FilterDepth);
  test.BatchQuantizeFloats4Bit();
  int8_t* result = test.output_data.data();
  int k = 0;
  int outer_rows = test.rows / width;
  int outer_cols = test.cols / optimized_4bit::FilterDepth;
  for (int i = 0; i < outer_rows; ++i) {
    for (int j = 0; j < outer_cols; ++j) {
      for (int w = 0; w < width; w++) {
        int c = 0;
        v = -127;
        for (; c < optimized_4bit::FilterDepth; c++) {
          int8_t res = result[k++];
          int res0 = static_cast<int>(res);
          if ((i * width + w) < src_rows) {
            if ((j * optimized_4bit::FilterDepth + c) < src_cols) {
              EXPECT_EQ(res0, v);
            }
          }
          v = -v;
        }
      }
      v += 2;
    }
  }
  for (int i = 0; i < test.rows; i++) {
    if (i >= src_rows) {
      continue;
    }
    EXPECT_EQ(test.input_offsets[i], 0);
    EXPECT_NEAR(test.scaling_factors[i], 1.0 / (1 + i), 1e-3);
  }
}

INSTANTIATE_TEST_SUITE_P(RunQuantizeInputTests, RunQuantizeInputTests,
                         ::testing::ValuesIn({
                             std::make_tuple(1, 1, 32),
                             std::make_tuple(1, 3, 46),
                             std::make_tuple(1, 9, 64),
                             std::make_tuple(1, 25, 72),
                             std::make_tuple(2, 2, 32),
                             std::make_tuple(2, 3, 46),
                             std::make_tuple(2, 9, 64),
                             std::make_tuple(2, 25, 72),
                             std::make_tuple(4, 4, 32),
                             std::make_tuple(4, 5, 46),
                             std::make_tuple(4, 9, 64),
                             std::make_tuple(4, 25, 72),
                         }));

struct TestAssignBiasAndComputeOffset {
  TestAssignBiasAndComputeOffset(std::vector<float> output_data,
                                 std::vector<int32_t> input_offsets,
                                 std::vector<float> input_scales,
                                 std::vector<float> filter_scales,
                                 std::vector<float> bias, int output_rows,
                                 int output_cols, bool use_bias)
      : output_data(output_data),
        input_offsets(input_offsets),
        input_scales(input_scales),
        filter_scales(filter_scales),
        bias(bias),
        output_rows(output_rows),
        output_cols(output_cols),
        use_bias(use_bias) {}

  void AssignBiasAndComputeOffsets() {
    optimized_4bit::AssignBiasAndComputeOffsets(
        input_offsets.data(), input_scales.data(), filter_scales.data(),
        use_bias ? bias.data() : nullptr, output_data.data(), output_cols,
        output_rows);
  }
  std::vector<float> output_data;
  std::vector<int32_t> input_offsets;
  std::vector<float> input_scales;
  std::vector<float> filter_scales;
  std::vector<float> bias;
  int output_rows;
  int output_cols;
  bool use_bias;
};

class RunAssignBiasAndOffsetsTests
    : public ::testing::TestWithParam<::testing::tuple<int, int, bool>> {};

TEST_P(RunAssignBiasAndOffsetsTests, RunAssignBiasAndOffsetssTests) {
  auto params = GetParam();
  int output_rows = std::get<0>(params);
  int output_cols = std::get<1>(params);
  bool use_bias = std::get<2>(params);
  std::vector<float> test_data(output_rows * output_cols, 0);
  std::vector<float> test_input_scales(output_rows);
  std::vector<int32_t> test_input_offsets(output_rows);
  std::vector<float> test_filter_scales(output_cols);
  std::vector<float> test_bias(output_cols);

  for (int i = 0; i < output_rows; ++i) {
    test_input_scales[i] = real_dist(random_engine);
    test_input_offsets[i] = int_dist(random_engine);
  }
  for (int i = 0; i < output_cols; ++i) {
    test_filter_scales[i] = real_dist(random_engine);
    test_bias[i] = real_dist(random_engine);
  }

  TestAssignBiasAndComputeOffset test(
      test_data, test_input_offsets, test_input_scales, test_filter_scales,
      test_bias, output_rows, output_cols, use_bias);
  test.AssignBiasAndComputeOffsets();
  float* result = test.output_data.data();
  for (int i = 0; i < output_rows; ++i) {
    for (int j = 0; j < output_cols; ++j) {
      float val = result[i * output_cols + j];
      float expected = use_bias ? test_bias[j] : 0;
      expected +=
          test_input_offsets[i] * test_input_scales[i] * test_filter_scales[j];
      EXPECT_NEAR(val, expected, 1e-3);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(RunAssignBiasAndOffsetsTests,
                         RunAssignBiasAndOffsetsTests,
                         ::testing::ValuesIn({
                             std::make_tuple(1, 8, true),
                             std::make_tuple(4, 17, false),
                             std::make_tuple(4, 17, true),
                             std::make_tuple(11, 33, false),
                             std::make_tuple(11, 33, true),
                         }));

struct TestUnpack {
  TestUnpack(std::vector<int32_t> src_data, std::vector<float> input_scales,
             std::vector<float> filter_scales, int src_rows, int src_cols,
             int output_rows, int output_cols)
      : src_data(src_data),
        input_scales(input_scales),
        filter_scales(filter_scales),
        src_rows(src_rows),
        src_cols(src_cols),
        output_rows(output_rows),
        output_cols(output_cols) {
    output_data.assign(output_rows * output_cols, 0.0);
  }

  template <int Depth, int Width>
  void Unpack() {
    optimized_4bit::Unpack<Depth, Width>(
        output_data.data(), src_data.data(), output_rows, output_cols,
        input_scales.data(), filter_scales.data(), src_rows, src_cols);
  }

  std::vector<int32_t> src_data;
  std::vector<float> input_scales;
  std::vector<float> filter_scales;
  std::vector<float> output_data;
  int src_rows;
  int src_cols;
  int output_rows;
  int output_cols;
};

class RunUnpackTests
    : public ::testing::TestWithParam<::testing::tuple<int, int, int>> {};

TEST_P(RunUnpackTests, RunUnpackTests) {
  auto params = GetParam();
  int src_rows = std::get<0>(params);
  int src_cols = std::get<1>(params);
  // In this case, we only unpack 1 rhs row,
  // so the batch_size and accumulator rows must match.
  int output_rows = src_rows;
  int output_cols = std::get<2>(params);
  std::vector<float> test_input_scales(src_rows);
  std::vector<float> test_filter_scales(src_cols);
  std::vector<int32_t> test_data;
  test_data.reserve(src_rows * src_cols);
  int outer_cols = src_cols / optimized_4bit::FilterWidth;
  int outer_rows = src_rows;
  for (int j = 0; j < outer_cols; ++j) {
    for (int i = 0; i < outer_rows; ++i) {
      for (int k = 0; k < optimized_4bit::FilterWidth; ++k) {
        test_data.push_back(i);
      }
    }
  }
  for (int i = 0; i < src_rows; ++i) {
    test_input_scales[i] = real_dist(random_engine);
  }
  for (int i = 0; i < src_cols; ++i) {
    test_filter_scales[i] = real_dist(random_engine);
  }
  TestUnpack test(test_data, test_input_scales, test_filter_scales, src_rows,
                  src_cols, output_rows, output_cols);
  test.Unpack<4, 1>();
  std::vector<float> result = test.output_data;
  for (int i = 0; i < output_rows; ++i) {
    for (int j = 0; j < output_cols; ++j) {
      float res = result[i * output_cols + j];
      EXPECT_EQ(res, i * test_input_scales[i] * test_filter_scales[j]);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(RunUnpackTests, RunUnpackTests,
                         ::testing::ValuesIn({
                             std::make_tuple(1, 8, 5),
                             std::make_tuple(3, 4, 4),
                         }));

class RunKernelTests
    : public ::testing::TestWithParam<::testing::tuple<int, int, int, int>> {};

TEST_P(RunKernelTests, RunKernelTests) {
  auto params = GetParam();
  int rhs_width = std::get<0>(params);
  int lhs_layout_rows = std::get<1>(params);
  int rhs_layout_rows = std::get<2>(params);
  int lhs_layout_cols = std::get<3>(params);
  int rhs_layout_cols = lhs_layout_cols;

  std::vector<uint8_t> test_lhs(lhs_layout_rows * lhs_layout_cols / 2, 0.0);
  std::vector<int8_t> test_rhs(rhs_layout_rows * rhs_layout_cols, 0.0);
  std::vector<int32_t> test_accum(lhs_layout_rows * rhs_layout_rows, 0.0);

  int lhs_outer_rows = lhs_layout_rows / optimized_4bit::FilterWidth;
  int lhs_outer_cols = lhs_layout_cols / optimized_4bit::FilterDepth;

  // pack lhs
  for (int i = 0; i < lhs_outer_rows; ++i) {
    for (int j = 0; j < lhs_outer_cols; ++j) {
      for (int k = 0; k < optimized_4bit::FilterWidth; ++k) {
        for (int l = 0; l < optimized_4bit::FilterDepth / 2; ++l) {
          uint8_t u = static_cast<uint8_t>(int_dist(random_engine) + 7);
          uint8_t v = static_cast<uint8_t>(int_dist(random_engine) + 7);
          int lower = static_cast<uint8_t>(v) & UINT8_C(15);
          int upper = static_cast<uint8_t>(u) << 4;
          int cluster_index = (i * lhs_outer_cols + j) *
                              optimized_4bit::FilterDepth *
                              optimized_4bit::FilterWidth / 2;
          int index = cluster_index + k * (optimized_4bit::FilterDepth / 2);
          test_lhs[index + l] = (upper | lower);
        }
      }
    }
  }
  int rhs_outer_rows = rhs_layout_rows / rhs_width;
  int rhs_outer_cols = rhs_layout_cols / optimized_4bit::FilterDepth;

  for (int i = 0; i < rhs_outer_rows; ++i) {
    for (int j = 0; j < rhs_outer_cols; ++j) {
      for (int k = 0; k < rhs_width; ++k) {
        for (int l = 0; l < optimized_4bit::FilterDepth; ++l) {
          int8_t u = static_cast<int8_t>(int_dist(random_engine));
          int cluster_index = (i * rhs_outer_cols + j) *
                              optimized_4bit::FilterDepth * rhs_width;
          int index = cluster_index + k * optimized_4bit::FilterDepth;
          test_rhs[index + l] = u;
        }
      }
    }
  }

  int index = 0;
  std::vector<int32_t> expected_accum(lhs_layout_rows * rhs_layout_rows, 0.0);
  int outer_cols = rhs_outer_cols;
  for (int i = 0; i < lhs_outer_rows; ++i) {
    for (int j = 0; j < rhs_outer_rows; ++j) {
      int32_t accum[1][optimized_4bit::FilterWidth];
      for (int k = 0; k < rhs_width; ++k) {
        for (int l = 0; l < optimized_4bit::FilterWidth; ++l) {
          memset(accum, 0,
                 sizeof(int32_t) * rhs_width * optimized_4bit::FilterWidth);
          for (int m = 0; m < outer_cols; ++m) {
            for (int n = 0; n < optimized_4bit::FilterDepth; ++n) {
              int right_index = ((j * outer_cols + m) * rhs_width + k) *
                                optimized_4bit::FilterDepth;
              int8_t rhs = test_rhs[right_index + n];
              int left_index =
                  ((i * outer_cols + m) * optimized_4bit::FilterWidth + l) *
                  (optimized_4bit::FilterDepth / 2);
              uint8_t lhs = 0;
              if (n < optimized_4bit::FilterDepth / 2) {
                int a = n % 16;
                lhs = static_cast<uint8_t>(test_lhs[left_index + a] >> 4);
              } else {
                int a = n % 16;
                lhs = static_cast<uint8_t>(test_lhs[left_index + a] &
                                           UINT8_C(15));
              }
              int accum_index = ((i * rhs_outer_rows + j) * rhs_width + k) *
                                optimized_4bit::FilterWidth;
              expected_accum[accum_index + l] += rhs * lhs;
            }
          }
        }
      }
    }
  }

  index = 0;
  switch (rhs_width) {
#if defined(FC_4BIT_NEON) && defined(__aarch64__)
    case 4:
      optimized_4bit::RunKernel<optimized_4bit::FilterWidth, 4,
                                optimized_4bit::FilterDepth>(
          test_lhs.data(), test_rhs.data(), test_accum.data(), lhs_layout_rows,
          lhs_layout_cols, rhs_layout_rows, rhs_layout_cols, rhs_layout_rows,
          lhs_layout_rows);
      break;
    case 2:
      optimized_4bit::RunKernel<optimized_4bit::FilterWidth, 2,
                                optimized_4bit::FilterDepth>(
          test_lhs.data(), test_rhs.data(), test_accum.data(), lhs_layout_rows,
          lhs_layout_cols, rhs_layout_rows, rhs_layout_cols, rhs_layout_rows,
          lhs_layout_rows);
      break;
#endif
    case 1:
      [[fallthrough]];
    default:
      optimized_4bit::RunKernel<optimized_4bit::FilterWidth, 1,
                                optimized_4bit::FilterDepth>(
          test_lhs.data(), test_rhs.data(), test_accum.data(), lhs_layout_rows,
          lhs_layout_cols, rhs_layout_rows, rhs_layout_cols, rhs_layout_rows,
          lhs_layout_rows);
      break;
  }

  for (int i = 0; i < (rhs_layout_rows * lhs_layout_rows); ++i) {
    int32_t expected_val = expected_accum[i];
    int32_t val = test_accum[i];
    EXPECT_EQ(val, expected_val);
  }
}

INSTANTIATE_TEST_SUITE_P(
    RunKernelTests, RunKernelTests, ::testing::ValuesIn({
      std::make_tuple(1, 4, 1, 32), std::make_tuple(1, 8, 1, 32),
          std::make_tuple(1, 16, 1, 32), std::make_tuple(1, 4, 1, 64),
          std::make_tuple(1, 8, 1, 64), std::make_tuple(1, 16, 1, 64),
          std::make_tuple(1, 4, 5, 64), std::make_tuple(1, 8, 9, 64),
          std::make_tuple(1, 16, 17, 64),
#if defined(FC_4BIT_NEON) && defined(__aarch64__)
          std::make_tuple(2, 8, 2, 32), std::make_tuple(2, 16, 2, 32),
          std::make_tuple(2, 4, 4, 64), std::make_tuple(2, 8, 4, 64),
          std::make_tuple(2, 16, 4, 64), std::make_tuple(2, 4, 4, 64),
          std::make_tuple(2, 8, 8, 64), std::make_tuple(2, 16, 16, 64),
          std::make_tuple(4, 4, 4, 32), std::make_tuple(4, 8, 4, 32),
          std::make_tuple(4, 16, 4, 32), std::make_tuple(4, 4, 8, 64),
          std::make_tuple(4, 8, 8, 64), std::make_tuple(4, 16, 8, 64),
          std::make_tuple(4, 4, 8, 64), std::make_tuple(4, 8, 12, 64),
          std::make_tuple(4, 16, 32, 64),
#endif
    }));
}  // namespace
}  // namespace tflite
