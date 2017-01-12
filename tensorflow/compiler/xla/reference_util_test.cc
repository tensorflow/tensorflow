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

#include "tensorflow/compiler/xla/reference_util.h"

#include <cmath>
#include <memory>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

// Tests linear algebra routines implemented in ReferenceUtil class.
// TODO(b/23829238): Currently missing tests for the convolution routine.
class ReferenceUtilTest : public ::testing::Test {
 protected:
  ReferenceUtilTest() {
    matrix_ = MakeUnique<Array2D<float>>(rows_, cols_);
    // [1.f  2.f  3.f]
    // [4.f  5.f  6.f]
    for (int64 i = 0; i < rows_; ++i) {
      for (int64 j = 0; j < cols_; ++j) {
        (*matrix_)(i, j) = i * cols_ + j + 1;
      }
    }
  }

  const int64 rows_ = 2;
  const int64 cols_ = 3;
  std::unique_ptr<Array2D<float>> matrix_;
};

TEST_F(ReferenceUtilTest, TransposeArray2D) {
  auto result = ReferenceUtil::TransposeArray2D(*matrix_);
  auto result_literal = LiteralUtil::CreateR2FromArray2D(*result);
  LiteralTestUtil::ExpectR2Near<float>({{1.f, 4.f}, {2.f, 5.f}, {3.f, 6.f}},
                                       *result_literal, ErrorSpec(0.0001));
}

TEST_F(ReferenceUtilTest, MatmulArray2D) {
  Array2D<float> rhs({
      {7.f, 8.f}, {9.f, 10.f}, {11.f, 12.f},
  });
  auto result = ReferenceUtil::MatmulArray2D(*matrix_, rhs);
  auto result_literal = LiteralUtil::CreateR2FromArray2D(*result);
  LiteralTestUtil::ExpectR2Near<float>({{58.f, 64.f}, {139.f, 154.f}},
                                       *result_literal, ErrorSpec(0.0001));
}

TEST_F(ReferenceUtilTest, ReduceToColArray2D) {
  auto add = [](float lhs, float rhs) { return lhs + rhs; };
  auto result = ReferenceUtil::ReduceToColArray2D(*matrix_, 0.0f, add);
  auto result_literal = LiteralUtil::CreateR1<float>(*result);
  LiteralTestUtil::ExpectR1Near<float>({6.f, 15.f}, *result_literal,
                                       ErrorSpec(0.0001));
}

TEST_F(ReferenceUtilTest, ReduceToRowArray2D) {
  auto add = [](float lhs, float rhs) { return lhs + rhs; };
  auto result = ReferenceUtil::ReduceToRowArray2D(*matrix_, 0.0f, add);
  auto result_literal = LiteralUtil::CreateR1<float>(*result);
  LiteralTestUtil::ExpectR1Near<float>({5.f, 7.f, 9.f}, *result_literal,
                                       ErrorSpec(0.0001));
}

TEST_F(ReferenceUtilTest, MapArray2D) {
  auto identity = [](float value) { return log(exp(value)); };
  auto result = ReferenceUtil::MapArray2D(*matrix_, identity);
  auto result_literal = LiteralUtil::CreateR2FromArray2D(*result);
  LiteralTestUtil::ExpectR2NearArray2D(*matrix_, *result_literal,
                                       ErrorSpec(0.0001));
}

TEST_F(ReferenceUtilTest, MapWithIndexArray2D) {
  auto add_index = [](float value, int64 row, int64 col) {
    return value + row + col;
  };
  auto result = ReferenceUtil::MapWithIndexArray2D(*matrix_, add_index);
  auto result_literal = LiteralUtil::CreateR2FromArray2D(*result);
  LiteralTestUtil::ExpectR2Near<float>({{1.f, 3.f, 5.f}, {5.f, 7.f, 9.f}},
                                       *result_literal, ErrorSpec(0.0001));
}

TEST_F(ReferenceUtilTest, MapArray4D) {
  auto input = MakeUnique<Array4D<float>>(/*planes=*/2, /*depth=*/3,
                                          /*height=*/4, /*width=*/5);
  input->FillWithMultiples(1.0f);
  auto multiply_by_two = [](float value) { return 2 * value; };
  auto result = ReferenceUtil::MapArray4D(*input, multiply_by_two);
  auto result_literal = LiteralUtil::CreateR4FromArray4D(*result);

  Array4D<float> expected(/*planes=*/2, /*depth=*/3, /*height=*/4, /*width=*/5);
  expected.FillWithMultiples(2.0f);
  LiteralTestUtil::ExpectR4NearArray4D(expected, *result_literal,
                                       ErrorSpec(0.0001));
}

TEST_F(ReferenceUtilTest, MapWithIndexArray4D) {
  auto input = MakeUnique<Array4D<float>>(/*planes=*/2, /*depth=*/3,
                                          /*height=*/4, /*width=*/5);
  input->FillWithMultiples(1.0f);
  auto subtract_index = [](float value, int64 plane, int64 depth, int64 height,
                           int64 width) {
    return value - (3 * 4 * 5 * plane + 4 * 5 * depth + 5 * height + width);
  };
  auto result = ReferenceUtil::MapWithIndexArray4D(*input, subtract_index);
  auto result_literal = LiteralUtil::CreateR4FromArray4D(*result);

  Array4D<float> expected(/*planes=*/2, /*depth=*/3, /*height=*/4, /*width=*/5);
  expected.Fill(0.0f);
  LiteralTestUtil::ExpectR4NearArray4D(expected, *result_literal,
                                       ErrorSpec(0.0001));
}

TEST_F(ReferenceUtilTest, ConvWithSamePadding) {
  Array4D<float> input(1, 1, 4, 4);
  // clang-format off
  input.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  Array4D<float> weights(1, 1, 2, 2);
  // clang-format off
  weights.FillWithYX(Array2D<float>({
    {5, 6},
    {7, 8},
  }));
  // clang-format on
  std::unique_ptr<Array4D<float>> actual =
      ReferenceUtil::ConvArray4D(input, weights, {1, 1}, Padding::kSame);
  Array4D<float> expected(1, 1, 4, 4);
  // clang-format off
  expected.FillWithYX(Array2D<float>({
    {100, 126, 152,  76},
    {204, 230, 256, 124},
    {308, 334, 360, 172},
    {149, 160, 171,  80},
  }));
  // clang-format on

  auto actual_literal = LiteralUtil::CreateR4FromArray4D(*actual);

  LiteralTestUtil::ExpectR4NearArray4D<float>(expected, *actual_literal,
                                              ErrorSpec(0.0001));
}

TEST_F(ReferenceUtilTest, ConvWithValidPadding) {
  Array4D<float> input(1, 1, 4, 4);
  // clang-format off
  input.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  Array4D<float> weights(1, 1, 2, 2);
  // clang-format off
  weights.FillWithYX(Array2D<float>({
    {5, 6},
    {7, 8},
  }));
  // clang-format on
  std::unique_ptr<Array4D<float>> actual =
      ReferenceUtil::ConvArray4D(input, weights, {1, 1}, Padding::kValid);
  Array4D<float> expected(1, 1, 3, 3);
  // clang-format off
  expected.FillWithYX(Array2D<float>({
    {1*5+2*6+5*7+6*8, 126, 152},
    {204, 230, 256},
    {308, 334, 11*5+12*6+15*7+16*8},
  }));
  // clang-format on

  auto actual_literal = LiteralUtil::CreateR4FromArray4D(*actual);

  LiteralTestUtil::ExpectR4NearArray4D<float>(expected, *actual_literal,
                                              ErrorSpec(0.0001));
}

TEST_F(ReferenceUtilTest, ConvGeneralDimensionsWithSamePadding) {
  // clang-format off
  // Input dimensions: [feature=2, height=3, batch=1, width=4]
  Array4D<float> input({
    {{{1, 2, 3, 4}},
     {{5, 6, 7, 8}},
     {{9, 10, 11, 12}}},
    {{{13, 14, 15, 16}},
     {{17, 18, 19, 20}},
     {{21, 22, 23, 24}}}
  });
  // Weight dimensions:
  // [kernel_output_feature=1, height=3, kernel_input_feature=2, width=3]
  Array4D<float> weight({{
    {{1, 2, 3},
     {4, 5, 6}},
    {{7, 8, 9},
     {10, 11, 12}},
    {{13, 14, 15},
     {16, 17, 18}}
  }});
  // clang-format on

  // Set the convolution dimension numbers.
  ConvolutionDimensionNumbers dimension_numbers;
  dimension_numbers.set_batch_dimension(2);
  dimension_numbers.set_feature_dimension(0);
  dimension_numbers.add_spatial_dimensions(1);
  dimension_numbers.add_spatial_dimensions(3);
  dimension_numbers.set_kernel_output_feature_dimension(0);
  dimension_numbers.set_kernel_input_feature_dimension(2);
  dimension_numbers.add_kernel_spatial_dimensions(1);
  dimension_numbers.add_kernel_spatial_dimensions(3);

  std::unique_ptr<Array4D<float>> actual =
      ReferenceUtil::ConvArray4DGeneralDimensions(
          input, weight, {1, 1}, Padding::kSame, dimension_numbers);
  // clang-format off
  // Result dimensions: [feature=1, height=3, batch=1, width=4]
  Array4D<float> expected({{
    {{1110, 1688, 1838, 1226}},
    {{1683, 2514, 2685, 1761}},
    {{878, 1280, 1358, 866}}
  }});
  // clang-format on

  auto actual_literal = LiteralUtil::CreateR4FromArray4D(*actual);

  LiteralTestUtil::ExpectR4NearArray4D<float>(expected, *actual_literal,
                                              ErrorSpec(0.0001));
}

TEST_F(ReferenceUtilTest, ConvGeneralDimensionsWithValidPadding) {
  // clang-format off
  // Input dimensions: [feature=2, height=3, batch=1, width=4]
  Array4D<float> input({
    {{{1, 2, 3, 4}},
     {{5, 6, 7, 8}},
     {{9, 10, 11, 12}}},
    {{{13, 14, 15, 16}},
     {{17, 18, 19, 20}},
     {{21, 22, 23, 24}}}
  });
  // Weight dimensions:
  // [kernel_output_feature=1, width=3, kernel_input_feature=2, height=3]
  Array4D<float> weight({{
    {{1, 7, 13},
     {4, 10, 16}},
    {{2, 8, 14},
     {5, 11, 17}},
    {{3, 9, 15},
     {6, 12, 18}}
  }});
  // clang-format on

  // Set the convolution dimension numbers.
  ConvolutionDimensionNumbers dimension_numbers;
  dimension_numbers.set_batch_dimension(2);
  dimension_numbers.set_feature_dimension(0);
  dimension_numbers.add_spatial_dimensions(1);
  dimension_numbers.add_spatial_dimensions(3);

  dimension_numbers.set_kernel_output_feature_dimension(0);
  dimension_numbers.set_kernel_input_feature_dimension(2);
  dimension_numbers.add_kernel_spatial_dimensions(3);
  dimension_numbers.add_kernel_spatial_dimensions(1);

  std::unique_ptr<Array4D<float>> actual =
      ReferenceUtil::ConvArray4DGeneralDimensions(
          input, weight, {1, 1}, Padding::kValid, dimension_numbers);
  // clang-format off
  // Result dimensions: [feature=1, height=1, batch=1, width=2]
  Array4D<float> expected({{{{2514, 2685}}}});
  // clang-format on

  auto actual_literal = LiteralUtil::CreateR4FromArray4D(*actual);

  LiteralTestUtil::ExpectR4NearArray4D<float>(expected, *actual_literal,
                                              ErrorSpec(0.0001));
}

}  // namespace
}  // namespace xla
