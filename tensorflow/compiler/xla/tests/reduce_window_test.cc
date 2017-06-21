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

// Tests the reduce-window XLA operation.

#include <limits>
#include <memory>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class ReduceWindowTest : public ClientLibraryTestBase {
 public:
  ReduceWindowTest() : builder_(client_, TestName()) {}

  void ReduceWindowAdd(const ComputationDataHandle& input,
                       tensorflow::gtl::ArraySlice<int64> window_dimensions,
                       tensorflow::gtl::ArraySlice<int64> window_strides,
                       Padding padding) {
    builder_.ReduceWindow(input, builder_.ConstantR0<float>(0.0f),
                          CreateScalarAddComputation(F32, &builder_),
                          window_dimensions, window_strides, padding);
  }

  void ReduceWindowMax(const ComputationDataHandle& input,
                       tensorflow::gtl::ArraySlice<int64> window_dimensions,
                       tensorflow::gtl::ArraySlice<int64> window_strides,
                       Padding padding) {
    builder_.ReduceWindow(
        input, builder_.ConstantLiteral(Literal::MinValue(F32)),
        CreateScalarMax(), window_dimensions, window_strides, padding);
  }

  void ReduceWindowMin(const ComputationDataHandle& input,
                       tensorflow::gtl::ArraySlice<int64> window_dimensions,
                       tensorflow::gtl::ArraySlice<int64> window_strides,
                       Padding padding) {
    builder_.ReduceWindow(input,
                          builder_.ConstantLiteral(Literal::MaxValue(F32)),
                          CreateScalarMinComputation(F32, &builder_),
                          window_dimensions, window_strides, padding);
  }

  ComputationBuilder builder_;
};

XLA_TEST_F(ReduceWindowTest, ZeroElementSmall) {
  Array4D<float> input_array(1, 0, 2, 1);

  const auto input = builder_.ConstantR4FromArray4D<float>(input_array);
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, 2, 1}, {1, 1, 1, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(input_array, 0.0f, {1, 1, 2, 1},
                                              {1, 1, 1, 1}, padding);

  ComputeAndCompareR4<float>(&builder_, *res, {}, ErrorSpec(1e-3, 1e-3));
}

TEST_F(ReduceWindowTest, NonSquareSmall) {
  Array4D<float> input_array(1, 2, 2, 1);
  input_array.FillRandom(2.f);

  const auto input = builder_.ConstantR4FromArray4D<float>(input_array);
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, 2, 1}, {1, 1, 1, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(input_array, 0.0f, {1, 1, 2, 1},
                                              {1, 1, 1, 1}, padding);

  ComputeAndCompareR4<float>(&builder_, *res, {}, ErrorSpec(1e-3, 1e-3));
}

TEST_F(ReduceWindowTest, MiddleDimsSmall) {
  Array4D<float> input_array(1, 3, 3, 1);
  input_array.FillRandom(2.f);

  const auto input = builder_.ConstantR4FromArray4D<float>(input_array);
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, 1, 1}, {1, 2, 2, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(input_array, 0.0f, {1, 1, 1, 1},
                                              {1, 2, 2, 1}, padding);

  ComputeAndCompareR4<float>(&builder_, *res, {}, ErrorSpec(1e-3, 1e-3));
}

TEST_F(ReduceWindowTest, Along2ndMinorDim) {
  Array4D<float> input_array(3, 6, 7, 32);
  input_array.FillRandom(2.f);

  // The parameters of this reduction mimic feature norm (e.g. LRN).
  int lrn_diameter = 7;  // diameter = 2*radius + 1 --> must be odd
  const auto input = builder_.ConstantR4FromArray4D<float>(input_array);
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, lrn_diameter, 1}, {1, 1, 1, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {1, 1, lrn_diameter, 1}, {1, 1, 1, 1}, padding);

  ComputeAndCompareR4<float>(&builder_, *res, {}, ErrorSpec(1e-3, 1e-3));
}

TEST_F(ReduceWindowTest, AmongMajor2Dims) {
  Array4D<float> input_array(4, 4, 6, 8);
  input_array.FillWithMinorDimNum();

  int win_len = 3;
  int win_stride = 1;

  Padding padding = Padding::kSame;
  const auto input_data_handle =
      builder_.ConstantR4FromArray4D<float>(input_array);
  // Reduce only along the x and y dimensions, according to the win_len.
  ReduceWindowAdd(input_data_handle, {win_len, win_len, 1, 1},
                  {win_stride, win_stride, 1, 1}, padding);

  auto result = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {win_len, win_len, 1, 1},
      {win_stride, win_stride, 1, 1}, padding);
  ComputeAndCompareR4<float>(&builder_, *result, {}, ErrorSpec(1e-3, 1e-3));
}

TEST_F(ReduceWindowTest, AmongMajor2DimsMediumSize) {
  Array4D<float> input_array(9, 12, 4, 89);
  input_array.FillRandom(2.0f);

  int win_len = 3;
  int win_stride = 2;

  const auto input_data_handle =
      builder_.ConstantR4FromArray4D<float>(input_array);

  Padding padding = Padding::kSame;
  // Reduce only along the x and y dimensions, according to the win_len.
  ReduceWindowAdd(input_data_handle, {win_len, win_len, 1, 1},
                  {win_stride, win_stride, 1, 1}, padding);

  auto result = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {win_len, win_len, 1, 1},
      {win_stride, win_stride, 1, 1}, padding);

  ComputeAndCompareR4<float>(&builder_, *result, {}, ErrorSpec(1e-3, 1e-3));
}

// TODO(b/32173947): Test support for arbitrary-sized padding.
TEST_F(ReduceWindowTest, DISABLED_AmongMajor2DimsMediumSizeLargePadding) {
  Array4D<float> input_array(9, 12, 4, 89);  // simulate Dim0IsMinor layout
  input_array.FillRandom(2.0f);

  int64 rank = 4;
  int win_len = 3;
  int win_stride = 2;

  const auto input_data_handle =
      builder_.ConstantR4FromArray4D<float>(input_array);

  Padding padding = Padding::kSame;
  // Reduce only along the x and y dimensions, according to the win_len.
  // Create padding vector with large padding values in the reduction dims.
  std::vector<std::pair<int64, int64>> low_high_padding;
  low_high_padding.resize(rank, {4, 4});

  builder_.ReduceWindowWithGeneralPadding(
      input_data_handle, builder_.ConstantR0<float>(0.0f),
      CreateScalarAddComputation(F32, &builder_), {win_len, win_len, 1, 1},
      {win_stride, win_stride, 1, 1}, low_high_padding);

  auto result = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {win_len, win_len, 1, 1},
      {win_stride, win_stride, 1, 1}, padding);

  ComputeAndCompareR4<float>(&builder_, *result, {}, ErrorSpec(1e-3, 1e-3));
}

// TODO(b/31809540): Implement minor dim reduction to reduce num of reshapes.
TEST_F(ReduceWindowTest, ReduceR4AmongXYMinorSmall) {
  Array4D<float> input_array(2, 2, 4, 16);

  Array2D<float> yx({{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f,
                      11.f, 12.f, 13.f, 14.f, 15.f},
                     {16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f,
                      25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f},
                     {32.f, 33.f, 34.f, 35.f, 36.f, 37.f, 38.f, 39.f, 40.f,
                      41.f, 42.f, 43.f, 44.f, 45.f, 46.f, 47.f},
                     {48.f, 49.f, 50.f, 51.f, 52.f, 53.f, 54.f, 55.f, 56.f,
                      57.f, 58.f, 59.f, 60.f, 61.f, 62.f, 63.f}});
  input_array.FillWithYX(yx);

  int win_len = 2;
  int win_stride = 2;
  const auto input = builder_.ConstantR4FromArray4D<float>(input_array);
  Padding padding = Padding::kValid;
  ReduceWindowAdd(input, {1, 1, win_len, win_len},
                  {1, 1, win_stride, win_stride}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {1, 1, win_len, win_len},
      {1, 1, win_stride, win_stride}, padding);
  ComputeAndCompareR4<float>(&builder_, *res, {}, ErrorSpec(1e-3, 1e-3));
}

// TODO(b/31809540): Implement minor dim reduction to reduce num of reshapes.
TEST_F(ReduceWindowTest, ReduceR4AmongXYMinorSmallOverlapped) {
  constexpr int64 p = 2;
  constexpr int64 z = 2;
  constexpr int64 y = 4;
  constexpr int64 x = 16;
  Array4D<float> input_array(p, z, y, x);

  Array2D<float> yx({{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f,
                      11.f, 12.f, 13.f, 14.f, 15.f},
                     {16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f,
                      25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f},
                     {32.f, 33.f, 34.f, 35.f, 36.f, 37.f, 38.f, 39.f, 40.f,
                      41.f, 42.f, 43.f, 44.f, 45.f, 46.f, 47.f},
                     {48.f, 49.f, 50.f, 51.f, 52.f, 53.f, 54.f, 55.f, 56.f,
                      57.f, 58.f, 59.f, 60.f, 61.f, 62.f, 63.f}});
  input_array.FillWithYX(yx);

  int win_len = 4;
  int win_stride = 2;
  const auto input = builder_.ConstantR4FromArray4D<float>(input_array);
  ReduceWindowAdd(input, {1, 1, win_len, win_len},
                  {1, 1, win_stride, win_stride}, Padding::kValid);

  // Expected result
  Array2D<float> yx_result({{408.f, 440.f, 472.f, 504.f, 536.f, 568.f, 600.f}});
  Array4D<float> expected(p, z, 1, 7);
  expected.FillWithYX(yx_result);
  ComputeAndCompareR4<float>(&builder_, expected, {}, ErrorSpec(1e-3, 1e-3));
}

TEST_F(ReduceWindowTest, MaxTrivial) {
  const auto input = builder_.ConstantR1<float>({42});
  ReduceWindowMax(input, {1}, {1}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {42}, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceWindowTest, Add3In3) {
  const auto input = builder_.ConstantR1<float>({20, 100, 3});
  ReduceWindowAdd(input, {3}, {1}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {123}, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceWindowTest, Add4In16Stride4) {
  const auto input = builder_.ConstantR1<float>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ReduceWindowAdd(input, {4}, {4}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {10, 26, 42, 58}, {},
                             ErrorSpec(0.0001));
}

TEST_F(ReduceWindowTest, DISABLED_ON_CPU(DISABLED_ON_GPU(Min3In5Stride2))) {
  const auto input = builder_.ConstantR1<float>({10000, 1000, 100, 10, 1});
  ReduceWindowMin(input, {3}, {2}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {100, 1}, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceWindowTest, Max3In3) {
  const auto input = builder_.ConstantR1<float>({20, 100, 3});
  ReduceWindowMax(input, {3}, {1}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {100}, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceWindowTest, Add2In3) {
  const auto input = builder_.ConstantR1<float>({100, 10, 1});
  ReduceWindowAdd(input, {2}, {1}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {110, 11}, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceWindowTest, Add3In5Stride2) {
  const auto input = builder_.ConstantR1<float>({10000, 1000, 100, 10, 1});
  ReduceWindowAdd(input, {3}, {2}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {11100, 111}, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceWindowTest, Max4In16Stride4) {
  const auto input = builder_.ConstantR1<float>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ReduceWindowMax(input, {4}, {4}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {4, 8, 12, 16}, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceWindowTest, Max4In16Stride3) {
  const auto input = builder_.ConstantR1<float>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ReduceWindowMax(input, {4}, {3}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {4, 7, 10, 13, 16}, {},
                             ErrorSpec(0.0001));
}

TEST_F(ReduceWindowTest, Max4In16Stride8) {
  const auto input = builder_.ConstantR1<float>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ReduceWindowMax(input, {4}, {8}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {4, 12}, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceWindowTest, Max3In5Stride2) {
  const auto input = builder_.ConstantR1<float>({10000, 1000, 100, 10, 1});
  ReduceWindowMax(input, {3}, {2}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {10000, 100}, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceWindowTest, Max3In5Stride1) {
  const auto input = builder_.ConstantR1<float>({10000, 1000, 100, 10, 101});
  ReduceWindowMax(input, {3}, {1}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {10000, 1000, 101}, {},
                             ErrorSpec(0.0001));
}

TEST_F(ReduceWindowTest, Add3In4Stride2) {
  const auto input = builder_.ConstantR1<float>({1000, 100, 10, 1});
  ReduceWindowAdd(input, {3}, {2}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {1110}, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ReduceWindowTest, Add2In3SamePad) {
  const auto input = builder_.ConstantR1<float>({100, 10, 1});
  ReduceWindowAdd(input, {2}, {1}, Padding::kSame);
  ComputeAndCompareR1<float>(&builder_, {110, 11, 1}, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ReduceWindowTest, Add3In3SamePad) {
  const auto input = builder_.ConstantR1<float>({100, 10, 1});
  ReduceWindowAdd(input, {3}, {1}, Padding::kSame);
  ComputeAndCompareR1<float>(&builder_, {110, 111, 11}, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ReduceWindowTest, Add3In3Stride3SamePad) {
  const auto input = builder_.ConstantR1<float>({100, 10, 1});
  ReduceWindowAdd(input, {3}, {2}, Padding::kSame);
  ComputeAndCompareR1<float>(&builder_, {110, 11}, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceWindowTest, Add2x2In2x2Overlapped) {
  Array2D<float> input_array({{1.2f, -2.5f, 0.9f, 1.0f},
                              {3.7f, 0.2f, -1.0f, -0.2f},
                              {-0.4f, 2.7f, 1.1f, 2.2f},
                              {0.6f, 1.7f, 1.4f, -0.2f}});
  auto input = builder_.ConstantR2FromArray2D<float>(input_array);
  ReduceWindowAdd(input, {2, 2}, {1, 1}, Padding::kValid);
  Array2D<float> expected(
      {{2.6f, -2.4f, 0.7f}, {6.2f, 3.0f, 2.1f}, {4.6f, 6.9f, 4.5f}});
  ComputeAndCompareR2<float>(&builder_, expected, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceWindowTest, Add2x2In2x2Disjoint) {
  Array2D<float> input_array({{1.2f, -2.5f, 0.9f, 1.0f},
                              {3.7f, 0.2f, -1.0f, -0.2f},
                              {-0.4f, 2.7f, 1.1f, 2.2f},
                              {0.6f, 1.7f, 1.4f, -0.2f}});
  auto input = builder_.ConstantR2FromArray2D<float>(input_array);
  ReduceWindowAdd(input, {2, 2}, {2, 2}, Padding::kValid);
  Array2D<float> expected({
      {2.6f, 0.7f}, {4.6f, 4.5f},
  });
  ComputeAndCompareR2<float>(&builder_, expected, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceWindowTest, Add1x2In2x2Same) {
  Array2D<float> input_array({{1.0f, 2.0f}, {3.0f, 4.0f}});
  auto input = builder_.ConstantR2FromArray2D<float>(input_array);
  ReduceWindowAdd(input, {1, 2}, {1, 1}, Padding::kSame);
  Array2D<float> expected({
      {3.0f, 2.0f}, {7.0f, 4.0f},
  });
  ComputeAndCompareR2<float>(&builder_, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ReduceWindowTest, Add1x1x2In2x1x2) {
  Array3D<float> input_array(2, 1, 2);
  input_array(0, 0, 0) = 1000;
  input_array(0, 0, 1) = 100;
  input_array(1, 0, 0) = 10;
  input_array(1, 0, 1) = 1;
  auto input = builder_.ConstantR3FromArray3D<float>(input_array);

  ReduceWindowAdd(input, {1, 1, 2}, {1, 1, 1}, Padding::kValid);

  Array3D<float> expected(2, 1, 1);
  expected(0, 0, 0) = 1100;
  expected(1, 0, 0) = 11;
  ComputeAndCompareR3<float>(&builder_, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ReduceWindowTest, Add1x1x2In2x1x3Stride1x1x2) {
  Array3D<float> input_array(2, 1, 3);
  input_array(0, 0, 0) = 100;
  input_array(0, 0, 1) = 10;
  input_array(0, 0, 2) = 1;
  input_array(1, 0, 0) = 500;
  input_array(1, 0, 1) = 50;
  input_array(1, 0, 2) = 5;
  auto input = builder_.ConstantR3FromArray3D<float>(input_array);

  ReduceWindowAdd(input, {1, 1, 2}, {1, 1, 2}, Padding::kValid);

  Array3D<float> expected(2, 1, 1);
  expected(0, 0, 0) = 110;
  expected(1, 0, 0) = 550;
  ComputeAndCompareR3<float>(&builder_, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ReduceWindowTest, Add1x1x2In2x1x3SamePad) {
  Array3D<float> input_array(2, 1, 3);
  input_array(0, 0, 0) = 100;
  input_array(0, 0, 1) = 10;
  input_array(0, 0, 2) = 1;
  input_array(1, 0, 0) = 500;
  input_array(1, 0, 1) = 50;
  input_array(1, 0, 2) = 5;
  auto input = builder_.ConstantR3FromArray3D<float>(input_array);

  ReduceWindowAdd(input, {1, 1, 2}, {1, 1, 1}, Padding::kSame);

  Array3D<float> expected(2, 1, 3);
  expected(0, 0, 0) = 110;
  expected(0, 0, 1) = 11;
  expected(0, 0, 2) = 1;
  expected(1, 0, 0) = 550;
  expected(1, 0, 1) = 55;
  expected(1, 0, 2) = 5;
  ComputeAndCompareR3<float>(&builder_, expected, {}, ErrorSpec(0.0001));
}

// Tests a reduction function that is not a simple add/min/max/etc.
XLA_TEST_F(ReduceWindowTest, NonstandardReduceFunction) {
  Array4D<float> input_array(1, 2, 2, 1);
  input_array(0, 0, 0, 0) = 1;
  input_array(0, 0, 1, 0) = 2;
  input_array(0, 1, 0, 0) = 3;
  input_array(0, 1, 1, 0) = 4;

  const auto input = builder_.ConstantR4FromArray4D<float>(input_array);
  Padding padding = Padding::kValid;

  const Shape scalar = ShapeUtil::MakeShape(F32, {});
  auto b = builder_.CreateSubBuilder("unusual");
  auto lhs = b->Parameter(0, scalar, "lhs");
  auto rhs = b->Parameter(1, scalar, "rhs");
  b->Min(b->Add(lhs, rhs), b->ConstantR0<float>(8.0f));
  Computation reduce_fn = b->BuildAndNoteError();

  builder_.ReduceWindow(input, builder_.ConstantR0<float>(3.0f), reduce_fn,
                        /*window_dimensions=*/{1, 1, 2, 1},
                        /*window_strides=*/{1, 1, 1, 1}, padding);

  const auto reduce_func = [](float arg1, float arg2) {
    return std::min<float>(arg1 + arg2, 8.0f);
  };

  auto expected =
      ReferenceUtil::ReduceWindow4DGeneric(input_array, 3.0f, reduce_func,
                                           /*window=*/{1, 1, 2, 1},
                                           /*stride=*/{1, 1, 1, 1}, padding);

  ComputeAndCompareR4<float>(&builder_, *expected, {}, ErrorSpec(1e-3, 1e-3));
}

TEST_F(ReduceWindowTest, R2ReduceWindowNonOverlappingFromBroadcast) {
  Array2D<float> input_array(6, 4, 1.0f);
  ComputationDataHandle input =
      builder_.Broadcast(builder_.ConstantLiteral(Literal::One(F32)), {6, 4});

  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {4, 2}, {3, 3}, padding);

  auto res = ReferenceUtil::ReduceWindow2DAdd(input_array, 0.0f, {4, 2}, {3, 3},
                                              padding);

  ComputeAndCompareR2<float>(&builder_, *res, {}, ErrorSpec(1e-3, 1e-3));
}

enum Reducer { kAdd, kMax };

struct R4ReduceWindowTestData {
  int64 base_bounds[4];
  int64 window_bounds[4];
  int64 strides[4];
  int64 pad_low[4];
  int64 pad_high[4];

  Reducer reducer;
};

string R4ReduceWindowTestDataToString(
    const ::testing::TestParamInfo<R4ReduceWindowTestData>& data) {
  string str = tensorflow::strings::StrCat(
      "base_bounds_",
      tensorflow::str_util::Join(data.param.base_bounds, "x"),  //
      "__window_bounds_",
      tensorflow::str_util::Join(data.param.window_bounds, "x"),            //
      "__strides_", tensorflow::str_util::Join(data.param.strides, "x"),    //
      "__pad_low_", tensorflow::str_util::Join(data.param.pad_low, "x"),    //
      "__pad_high_", tensorflow::str_util::Join(data.param.pad_high, "x"),  //
      (data.param.reducer == kAdd) ? "add" : "max");
  CHECK(data.param.reducer == kAdd || data.param.reducer == kMax);

  // Test names are not allowed to contain the '-' character.
  std::replace(str.begin(), str.end(), '-', 'n');
  return str;
}

class R4ReduceWindowTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<R4ReduceWindowTestData> {};
TEST_P(R4ReduceWindowTest, DoIt) {
  ComputationBuilder b(client_, TestName());
  const auto& param = GetParam();

  const float kInitValue = 0.0f;

  Array4D<float> input(param.base_bounds[0], param.base_bounds[1],
                       param.base_bounds[2], param.base_bounds[3]);
  input.FillIota(1);
  std::unique_ptr<Literal> input_literal = Literal::CreateR4FromArray4D(input);
  TF_ASSIGN_OR_ASSERT_OK(std::unique_ptr<GlobalData> input_arg,
                         client_->TransferToServer(*input_literal));

  std::vector<std::pair<int64, int64>> padding(4);
  for (int i = 0; i < 4; ++i) {
    padding[i] = {param.pad_low[i], param.pad_high[i]};
  }

  auto parameter = b.Parameter(0, input_literal->shape(), "p0");
  auto pad_value = b.ConstantR0<float>(kInitValue);
  CHECK(param.reducer == kAdd || param.reducer == kMax);
  auto computation = param.reducer == kAdd
                         ? CreateScalarAddComputation(F32, &b)
                         : CreateScalarMaxComputation(F32, &b);
  b.ReduceWindowWithGeneralPadding(
      /*operand=*/parameter,
      /*init_value=*/pad_value,
      /*computation=*/computation,
      /*window_dimensions=*/param.window_bounds,
      /*window_strides=*/param.strides,
      /*padding=*/padding);

  CHECK(param.reducer == kAdd || param.reducer == kMax);
  auto reduce_func = param.reducer == kAdd
                         ? +[](float a, float b) { return a + b; }
                         : +[](float a, float b) { return std::max(a, b); };
  std::unique_ptr<Array4D<float>> expected =
      ReferenceUtil::ReduceWindow4DGeneric(
          /*operand=*/input,
          /*init=*/kInitValue,
          /*reduce_func=*/reduce_func,
          /*window=*/param.window_bounds,
          /*stride=*/param.strides,
          /*padding=*/padding);
  ComputeAndCompareR4<float>(&b, *expected, {input_arg.get()});
}

// base_bounds, window_bounds, strides, pad_low, pad_high
const R4ReduceWindowTestData kR4ReduceWindowTestValues[] = {
    // Minimal edge case.
    R4ReduceWindowTestData{/*base_bounds=*/{1, 1, 1, 1},
                           /*window_bounds=*/{1, 1, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*reducer=*/kAdd},

    // Zero base bound edge case.
    R4ReduceWindowTestData{/*base_bounds=*/{1, 0, 1, 1},
                           /*window_bounds=*/{1, 1, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*reducer=*/kAdd},

    // With non-1x1 window.
    R4ReduceWindowTestData{/*base_bounds=*/{4, 6, 17, 140},
                           /*window_bounds=*/{2, 3, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*reducer=*/kAdd},

    // With max instead of add.
    R4ReduceWindowTestData{/*base_bounds=*/{4, 6, 17, 140},
                           /*window_bounds=*/{2, 3, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*reducer=*/kMax},

    // With stride.
    R4ReduceWindowTestData{/*base_bounds=*/{4, 10, 17, 140},
                           /*window_bounds=*/{3, 2, 1, 1},
                           /*strides=*/{2, 4, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*reducer=*/kAdd},

    // With low padding.
    R4ReduceWindowTestData{/*base_bounds=*/{4, 6, 17, 140},
                           /*window_bounds=*/{3, 2, 1, 1},
                           /*strides=*/{2, 2, 1, 1},
                           /*pad_low=*/{3, 2, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*reducer=*/kAdd},

    // With high padding.
    R4ReduceWindowTestData{/*base_bounds=*/{4, 6, 17, 140},
                           /*window_bounds=*/{3, 2, 1, 1},
                           /*strides=*/{2, 2, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{2, 3, 0, 0},
                           /*reducer=*/kAdd},

    // Window touches both sides of the padding simultaneously.
    R4ReduceWindowTestData{/*base_bounds=*/{1, 1, 17, 140},
                           /*window_bounds=*/{3, 3, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{1, 1, 0, 0},
                           /*pad_high=*/{1, 1, 0, 0},
                           /*reducer=*/kAdd},

    // Window is entirely in the padding for some positions.
    R4ReduceWindowTestData{/*base_bounds=*/{1, 1, 17, 140},
                           /*window_bounds=*/{3, 3, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{4, 4, 0, 0},
                           /*pad_high=*/{4, 4, 0, 0},
                           /*reducer=*/kAdd},

    // Zero base bound with padding edge case.
    R4ReduceWindowTestData{/*base_bounds=*/{2, 0, 3, 4},
                           /*window_bounds=*/{1, 1, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 1, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*reducer=*/kAdd},

    // With stride, low padding and high padding.
    R4ReduceWindowTestData{/*base_bounds=*/{4, 3, 17, 140},
                           /*window_bounds=*/{3, 4, 1, 1},
                           /*strides=*/{3, 1, 1, 1},
                           /*pad_low=*/{10, 1, 0, 0},
                           /*pad_high=*/{2, 3, 0, 0},
                           /*reducer=*/kAdd},
};

INSTANTIATE_TEST_CASE_P(R4ReduceWindowTestInstantiation, R4ReduceWindowTest,
                        ::testing::ValuesIn(kR4ReduceWindowTestValues),
                        R4ReduceWindowTestDataToString);

struct R2ReduceWindowTestData {
  int64 base_bounds[2];
  int64 window_bounds[2];
  int64 strides[2];
  int64 layout[2];
  Padding padding;
  Reducer reducer;
} kR2TestCases[] = {
    {/*base_bounds=*/{4, 18}, /*window_bounds=*/{2, 4},
     /*strides=*/{1, 2}, /*layout=*/{0, 1},
     /*padding=*/Padding::kSame, /*reducer*/ Reducer::kAdd},
    {/*base_bounds=*/{2, 5}, /*window_bounds=*/{2, 4},
     /*strides=*/{1, 1}, /*layout=*/{0, 1},
     /*padding=*/Padding::kSame, /*reducer*/ Reducer::kAdd},
    {/*base_bounds=*/{1, 3}, /*window_bounds=*/{2, 3},
     /*strides=*/{1, 1}, /*layout=*/{0, 1},
     /*padding=*/Padding::kSame, /*reducer*/ Reducer::kAdd},
    {/*base_bounds=*/{3, 129}, /*window_bounds=*/{1, 100},
     /*strides=*/{2, 99}, /*layout=*/{0, 1},
     /*padding=*/Padding::kSame, /*reducer*/ Reducer::kAdd},
    {/*base_bounds=*/{6, 152}, /*window_bounds=*/{2, 25},
     /*strides=*/{5, 4}, /*layout=*/{0, 1},
     /*padding=*/Padding::kSame, /*reducer*/ Reducer::kAdd},
    {/*base_bounds=*/{6, 4}, /*window_bounds=*/{4, 2},
     /*strides=*/{3, 3}, /*layout=*/{0, 1},
     /*padding=*/Padding::kSame, /*reducer*/ Reducer::kAdd},
};

string R2ReduceWindowTestDataToString(
    const ::testing::TestParamInfo<R2ReduceWindowTestData>& data) {
  string str = tensorflow::strings::StrCat(
      "base_bounds_",
      tensorflow::str_util::Join(data.param.base_bounds, "x"),  //
      "__window_bounds_",
      tensorflow::str_util::Join(data.param.window_bounds, "x"),              //
      "__strides_", tensorflow::str_util::Join(data.param.strides, "x"),      //
      "__padding_", data.param.padding == Padding::kSame ? "same" : "valid",  //
      "__layout_", data.param.layout[0], "_", data.param.layout[1],           //
      "__reducer_", data.param.reducer == kAdd ? "add" : "max");
  return str;
}

class R2ReduceWindowTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<R2ReduceWindowTestData> {};

TEST_P(R2ReduceWindowTest, Add) {
  ComputationBuilder b(client_, TestName());
  const auto& param = GetParam();
  CHECK(param.reducer == kAdd);

  const float kInitValue = 0.0f;
  Array2D<float> input(param.base_bounds[0], param.base_bounds[1], 1.0f);
  std::unique_ptr<Literal> input_literal =
      Literal::CreateR2FromArray2DWithLayout(
          input, LayoutUtil::MakeLayout(param.layout));
  TF_ASSIGN_OR_ASSERT_OK(std::unique_ptr<GlobalData> input_arg,
                         client_->TransferToServer(*input_literal));
  b.ReduceWindow(/*operand=*/
                 b.Parameter(0, input_literal->shape(), "p0"),
                 /*init_value=*/b.ConstantR0<float>(kInitValue),
                 /*computation=*/CreateScalarAddComputation(F32, &b),
                 /*window_dimensions=*/param.window_bounds,
                 /*window_strides=*/param.strides, /*padding=*/param.padding);

  auto expected = ReferenceUtil::ReduceWindow2DAdd(
      /*operand=*/input, /*init=*/kInitValue, /*window=*/param.window_bounds,
      /*stride=*/param.strides, /*padding=*/param.padding);

  ComputeAndCompareR2<float>(&b, *expected, {input_arg.get()},
                             ErrorSpec(1e-3, 1e-3));
}

INSTANTIATE_TEST_CASE_P(R2ReduceWindowTestInstantiation, R2ReduceWindowTest,
                        ::testing::ValuesIn(kR2TestCases),
                        R2ReduceWindowTestDataToString);

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendDebugOptionsFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
