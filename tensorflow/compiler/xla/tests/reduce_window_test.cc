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

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

#ifdef XLA_BACKEND_SUPPORTS_BFLOAT16
// Tests both F32 and BF16.
static std::array<bool, 2> use_bfloat16_params{false, true};
#else
// Only tests F32.
static std::array<bool, 1> use_bfloat16_params{false};
#endif

class ReduceWindowTestBase : public ClientLibraryTestBase {
 public:
  ErrorSpec DefaultErrorSpec() const {
    if (use_bfloat16()) {
      return ErrorSpec(2e-1, 6e-2);
    } else {
      return ErrorSpec(1e-3, 1e-3);
    }
  }
};

class ReduceWindowTest : public ::testing::WithParamInterface<bool>,
                         public ReduceWindowTestBase {
 public:
  ReduceWindowTest() : builder_(TestName()) { set_use_bfloat16(GetParam()); }

  void ReduceWindowAdd(const XlaOp input,
                       absl::Span<const int64_t> window_dimensions,
                       absl::Span<const int64_t> window_strides,
                       Padding padding) {
    auto init = CreateConstantFromLiteral(LiteralUtil::CreateR0<float>(0.0f),
                                          &builder_);
    ReduceWindow(input, init,
                 CreateScalarAddComputation(FloatType(), &builder_),
                 window_dimensions, window_strides, padding);
  }

  void ReduceWindowMax(const XlaOp input,
                       absl::Span<const int64_t> window_dimensions,
                       absl::Span<const int64_t> window_strides,
                       Padding padding) {
    auto init =
        CreateConstantFromLiteral(LiteralUtil::MinValue(F32), &builder_);
    ReduceWindow(input, init,
                 CreateScalarMaxComputation(FloatType(), &builder_),
                 window_dimensions, window_strides, padding);
  }

  void ReduceWindowMin(const XlaOp input,
                       absl::Span<const int64_t> window_dimensions,
                       absl::Span<const int64_t> window_strides,
                       Padding padding) {
    auto init =
        CreateConstantFromLiteral(LiteralUtil::MaxValue(F32), &builder_);
    ReduceWindow(input, init,
                 CreateScalarMinComputation(FloatType(), &builder_),
                 window_dimensions, window_strides, padding);
  }

  XlaBuilder builder_;
};

XLA_TEST_P(ReduceWindowTest, MismatchedRanksGivesErrorStatus) {
  const auto input = CreateConstantFromLiteral(
      LiteralUtil::CreateR1<float>({1, 1, 1, 1}), &builder_);
  const auto init_value =
      CreateConstantFromLiteral(LiteralUtil::CreateR0<float>(0), &builder_);
  TF_ASSERT_OK(builder_.first_error());
  ReduceWindow(input, init_value,
               CreateScalarAddComputation(FloatType(), &builder_),
               /*window_dimensions=*/{1, 2},
               /*window_strides=*/{1}, Padding::kValid);
  ASSERT_EQ(builder_.first_error().code(), tensorflow::error::INVALID_ARGUMENT)
      << builder_.first_error();
  ASSERT_THAT(builder_.first_error().error_message(),
              ::testing::HasSubstr("Want input dimensions size"));
}

// Regression test for b/68964348.
XLA_TEST_P(ReduceWindowTest, R0ReduceWindow) {
  const auto input =
      CreateConstantFromLiteral(LiteralUtil::CreateR0<float>(42.0), &builder_);
  const auto init =
      CreateConstantFromLiteral(LiteralUtil::CreateR0<float>(0.0), &builder_);
  ReduceWindow(input, init, CreateScalarAddComputation(FloatType(), &builder_),
               /*window_dimensions=*/{},
               /*window_strides=*/{}, Padding::kSame);
  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateR0<float>(42.0), {},
                           ErrorSpec(0.00001));
}

XLA_TEST_P(ReduceWindowTest, Min3In5Stride2) {
  const auto input = CreateConstantFromLiteral(
      LiteralUtil::CreateR1<float>({10000, 1000, 100, 10, 1}), &builder_);
  ReduceWindowMin(input, {3}, {2}, Padding::kValid);
  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateR1<float>({100, 1}),
                           {}, ErrorSpec(0.00001));
}

XLA_TEST_P(ReduceWindowTest, Min3In5Stride2Same) {
  const auto input = CreateConstantFromLiteral(
      LiteralUtil::CreateR1<float>({10000, 1000, 100, 10, 1}), &builder_);
  ReduceWindowMin(input, {3}, {2}, Padding::kSame);
  ComputeAndCompareLiteral(&builder_,
                           LiteralUtil::CreateR1<float>({1000, 10, 1}), {},
                           ErrorSpec(0.00001));
}

XLA_TEST_P(ReduceWindowTest, Min3In5Stride1WithSamePadding) {
  const auto input = CreateConstantFromLiteral(
      LiteralUtil::CreateR1<float>({10000, 1000, 100, 10, 1}), &builder_);
  ReduceWindowMin(input, /*window_dimensions=*/{3}, /*window_strides=*/{1},
                  Padding::kSame);
  ComputeAndCompareLiteral(&builder_,
                           LiteralUtil::CreateR1<float>({1000, 100, 10, 1, 1}),
                           {}, ErrorSpec(0.00001));
}

XLA_TEST_P(ReduceWindowTest, ZeroElementSmall) {
  Array4D<float> input_array(1, 0, 2, 1);
  const auto input = CreateConstantFromArray(input_array, &builder_);
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, 2, 1}, {1, 1, 1, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(input_array, 0.0f, {1, 1, 2, 1},
                                              {1, 1, 1, 1}, padding);

  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateFromArray(*res), {},
                           DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, NonSquareSmall) {
  Array4D<float> input_array(1, 2, 2, 1);
  input_array.FillRandom(2.f, 2.f);
  const auto input = CreateConstantFromArray(input_array, &builder_);

  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, 2, 1}, {1, 1, 1, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(input_array, 0.0f, {1, 1, 2, 1},
                                              {1, 1, 1, 1}, padding);

  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateFromArray(*res), {},
                           DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, MiddleDimsSmall) {
  Array4D<float> input_array(1, 3, 3, 1);
  input_array.FillRandom(2.f, 2.f);
  const auto input = CreateConstantFromArray(input_array, &builder_);
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, 1, 1}, {1, 2, 2, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(input_array, 0.0f, {1, 1, 1, 1},
                                              {1, 2, 2, 1}, padding);

  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateFromArray(*res), {},
                           DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, Along2ndMinorDim) {
  Array4D<float> input_array(3, 6, 7, 32);
  input_array.FillRandom(2.f, 2.f);
  const auto input = CreateConstantFromArray(input_array, &builder_);

  // The parameters of this reduction mimic feature norm (e.g. LRN).
  int lrn_diameter = 7;  // diameter = 2*radius + 1 --> must be odd
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, lrn_diameter, 1}, {1, 1, 1, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {1, 1, lrn_diameter, 1}, {1, 1, 1, 1}, padding);

  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateFromArray(*res), {},
                           DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, AmongMajor2DimsAdd) {
  Array4D<float> input_array(4, 4, 6, 8);
  input_array.FillWithMinorDimNum();
  const auto input_data_handle =
      CreateConstantFromArray(input_array, &builder_);

  int win_len = 3;
  int win_stride = 1;

  Padding padding = Padding::kSame;
  // Reduce only along the x and y dimensions, according to the win_len.
  ReduceWindowAdd(input_data_handle, {win_len, win_len, 1, 1},
                  {win_stride, win_stride, 1, 1}, padding);

  auto result = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {win_len, win_len, 1, 1},
      {win_stride, win_stride, 1, 1}, padding);

  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateFromArray(*result), {},
                           DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, AmongMajor2DimsMax) {
  Array4D<float> input_array(3, 3, 2, 1);
  input_array.FillWithMinorDimNum();
  const auto input_data_handle =
      CreateConstantFromArray(input_array, &builder_);
  int win_len = 2;
  int win_stride = 1;
  Padding padding = Padding::kValid;
  // Reduce only along the x and y dimensions, according to the win_len.
  ReduceWindowMax(input_data_handle, {win_len, win_len, 1, 1},
                  {win_stride, win_stride, 1, 1}, padding);
  ComputeAndCompare(&builder_, {}, DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, AmongMajor2DimsMediumSize) {
  Array4D<float> input_array(9, 12, 4, 89);
  input_array.FillRandom(2.f, 2.f);

  int win_len = 3;
  int win_stride = 2;

  const auto input_data_handle =
      CreateConstantFromArray(input_array, &builder_);

  Padding padding = Padding::kSame;
  // Reduce only along the x and y dimensions, according to the win_len.
  ReduceWindowAdd(input_data_handle, {win_len, win_len, 1, 1},
                  {win_stride, win_stride, 1, 1}, padding);

  auto result = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {win_len, win_len, 1, 1},
      {win_stride, win_stride, 1, 1}, padding);

  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateFromArray(*result), {},
                           DefaultErrorSpec());
}

// Tests the super windowing logic w.r.t handling prime number of windows in a
// major dimension with reduction.
XLA_TEST_P(ReduceWindowTest, PrimeWindowsInReductionDimension) {
  Array4D<float> input_array(15, 15, 4, 128);
  input_array.FillRandom(2.f, 4.f);

  int win_len = 3;
  int win_stride = 2;

  const auto input_data_handle =
      CreateConstantFromArray(input_array, &builder_);

  Padding padding = Padding::kSame;
  // Reduce only along the x and y dimensions, according to the win_len.
  ReduceWindowAdd(input_data_handle, {win_len, win_len, 1, 1},
                  {win_stride, win_stride, 1, 1}, padding);

  auto result = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {win_len, win_len, 1, 1},
      {win_stride, win_stride, 1, 1}, padding);

  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateFromArray(*result), {},
                           DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, ReduceAlongLaneDimension) {
  Array4D<float> input_array(19, 17, 8, 256);
  input_array.FillWithMinorDimNum();

  const auto input_data_handle =
      CreateConstantFromArray(input_array, &builder_);

  Padding padding = Padding::kSame;
  ReduceWindowAdd(input_data_handle, {1, 1, 1, 11}, {1, 1, 1, 1}, padding);

  auto result = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {1, 1, 1, 11}, {1, 1, 1, 1}, padding);

  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateFromArray(*result), {},
                           DefaultErrorSpec());
}

// Tests a reduction function that is not a simple add/min/max/etc.
XLA_TEST_P(ReduceWindowTest, NonstandardReduceFunction) {
  Array4D<float> input_array(1, 2, 2, 1);
  input_array(0, 0, 0, 0) = 1;
  input_array(0, 0, 1, 0) = 2;
  input_array(0, 1, 0, 0) = 3;
  input_array(0, 1, 1, 0) = 4;
  const auto input = CreateConstantFromArray(input_array, &builder_);

  Padding padding = Padding::kValid;
  const Shape scalar = ShapeUtil::MakeShape(FloatType(), {});
  auto b = builder_.CreateSubBuilder("unusual");
  auto lhs = Parameter(b.get(), 0, scalar, "lhs");
  auto rhs = Parameter(b.get(), 1, scalar, "rhs");
  Min(Add(lhs, rhs),
      CreateConstantFromLiteral(LiteralUtil::CreateR0<float>(8.0f), b.get()));
  XlaComputation reduce_fn = b->BuildAndNoteError();

  ReduceWindow(
      input,
      CreateConstantFromLiteral(LiteralUtil::CreateR0<float>(0.0f), &builder_),
      reduce_fn,
      /*window_dimensions=*/{1, 1, 2, 1},
      /*window_strides=*/{1, 1, 1, 1}, padding);

  const auto reduce_func = [](float arg1, float arg2) {
    return std::min<float>(arg1 + arg2, 8.0f);
  };

  auto expected =
      ReferenceUtil::ReduceWindow4DGeneric(input_array, 0.0f, reduce_func,
                                           /*window=*/{1, 1, 2, 1},
                                           /*stride=*/{1, 1, 1, 1}, padding);

  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateFromArray(*expected),
                           {}, DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, R4UnitWindow) {
  Array4D<float> input_array(13, 12, 8, 15);
  input_array.FillRandom(2.f, 2.f);
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input_array, LayoutUtil::MakeLayout({0, 3, 2, 1}));
  XlaOp input;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input_data, CreateParameterAndTransferLiteral(
                           0, input_literal, "parameter", &builder_, &input));

  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, 7, 1}, {1, 4, 1, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(input_array, 0.0f, {1, 1, 7, 1},
                                              {1, 4, 1, 1}, padding);

  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateFromArray(*res),
                           {input_data.get()}, DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, R6AddMultipleStrides) {
  std::vector<int64_t> input_dims(6, 8);
  auto shape = ShapeUtil::MakeShape(F32, input_dims);

  Literal arg_literal(shape);
  arg_literal.PopulateWithValue(1.0f);
  const auto input = CreateConstantFromLiteral(arg_literal, &builder_);

  Padding padding = Padding::kValid;
  ReduceWindowAdd(input, {3, 1, 3, 3, 1, 1}, {1, 1, 1, 1, 1, 1}, padding);

  std::vector<int64_t> output_layout = {1, 5, 3, 2, 0, 4};
  std::vector<int64_t> output_dims = {6, 8, 6, 6, 8, 8};
  Shape result_shape =
      ShapeUtil::MakeShapeWithLayout(F32, output_dims, output_layout);
  Literal expected(result_shape);
  expected.PopulateWithValue(27.0f);
  ComputeAndCompareLiteral(&builder_, expected, {}, DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, R6Add) {
  std::vector<int64_t> input_dims(6, 8);
  auto shape = ShapeUtil::MakeShape(F32, input_dims);

  Literal arg_literal =
      LiteralUtil::CreateFullWithDescendingLayout<float>(input_dims, 1.0f);

  const auto input = CreateConstantFromLiteral(arg_literal, &builder_);

  Padding padding = Padding::kValid;
  ReduceWindowAdd(input, {1, 1, 3, 3, 1, 1}, {1, 1, 1, 1, 1, 1}, padding);

  std::vector<int64_t> output_dims = {8, 8, 6, 6, 8, 8};
  Literal expected =
      LiteralUtil::CreateFullWithDescendingLayout<float>(output_dims, 9.0f);

  ComputeAndCompareLiteral(&builder_, expected, {}, DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, R4SecondMinorStride) {
  Array4D<float> input_array(2, 1, 27, 119);
  input_array.FillRandom(2.0f);
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input_array, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaOp input;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input_data, CreateParameterAndTransferLiteral(
                           0, input_literal, "parameter", &builder_, &input));

  int win_len = 1;
  int stride = 8;
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, win_len, 1}, {1, 1, stride, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {1, 1, win_len, 1}, {1, 1, stride, 1}, padding);

  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateFromArray(*res),
                           {input_data.get()}, DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, R4SecondMinorUnitStride) {
  Array4D<float> input_array(3, 2, 4, 64);
  input_array.FillRandom(2.0f);
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input_array, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaOp input;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input_data, CreateParameterAndTransferLiteral(
                           0, input_literal, "parameter", &builder_, &input));

  int win_len = 3;
  int stride = 1;
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, win_len, 1}, {1, 1, stride, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {1, 1, win_len, 1}, {1, 1, stride, 1}, padding);

  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateFromArray(*res),
                           {input_data.get()}, DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, R4SecondMinorWin) {
  Array4D<float> input_array(1, 3, 12, 200);
  input_array.FillRandom(2.0f);
  Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      input_array, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  XlaOp input;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input_data, CreateParameterAndTransferLiteral(
                           0, input_literal, "parameter", &builder_, &input));

  int win_len = 8;
  int stride = 5;
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, win_len, 1}, {1, 1, stride, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {1, 1, win_len, 1}, {1, 1, stride, 1}, padding);

  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateFromArray(*res),
                           {input_data.get()}, DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, AmongMajor2DimsMultipleMinor) {
  Array4D<float> input_array(6, 4, 10, 130);
  input_array.FillRandom(2.0f);

  int win_len = 3;
  int win_stride = 2;

  Padding padding = Padding::kSame;
  const auto input_data_handle =
      CreateConstantFromArray(input_array, &builder_);
  // Reduce only along the x and y dimensions, according to the win_len.
  ReduceWindowAdd(input_data_handle, {win_len, win_len, 1, 1},
                  {win_stride, win_stride, 1, 1}, padding);

  auto result = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {win_len, win_len, 1, 1},
      {win_stride, win_stride, 1, 1}, padding);
  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateFromArray(*result), {},
                           DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, Add24In1152_NoOverlap) {
  std::vector<float> input_vector(128 * 9, 1);
  const auto input = CreateConstantFromLiteral(
      LiteralUtil::CreateR1<float>(input_vector), &builder_);
  ReduceWindowAdd(input, {32}, {128}, Padding::kValid);
  ComputeAndCompareLiteral(
      &builder_,
      LiteralUtil::CreateR1<float>({32, 32, 32, 32, 32, 32, 32, 32, 32}), {},
      DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, Add128In128Stride128) {
  std::vector<float> input_vector{
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const auto input = CreateConstantFromLiteral(
      LiteralUtil::CreateR1<float>(input_vector), &builder_);
  ReduceWindowAdd(input, {128}, {128}, Padding::kValid);
  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateR1<float>({1088}), {},
                           DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, Add128In128) {
  std::vector<float> input_vector{
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const auto input = CreateConstantFromLiteral(
      LiteralUtil::CreateR1<float>(input_vector), &builder_);
  ReduceWindowAdd(input, {128}, {1}, Padding::kValid);
  ComputeAndCompareLiteral(&builder_, LiteralUtil::CreateR1<float>({1088}), {},
                           DefaultErrorSpec());
}

// Regression test for a bug that appeared in Inception (b/34784899).
XLA_TEST_P(ReduceWindowTest, R2ReduceWindowInceptionFromBroadcast) {
  Array2D<float> input_array(14, 14, 1.0f);
  const auto input = CreateConstantFromArray(input_array, &builder_);
  int win_len = 3;
  int stride = 1;
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {win_len, win_len}, {stride, stride}, padding);
  ComputeAndCompare(&builder_, {}, DefaultErrorSpec());
}

XLA_TEST_P(ReduceWindowTest, R2ReduceWindowNonOverlappingFromBroadcast) {
  Array2D<float> input_array(6, 4, 1.0f);
  XlaOp input = Broadcast(
      CreateConstantFromLiteral(LiteralUtil::One(F32), &builder_), {6, 4});
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {4, 2}, {3, 3}, padding);
  ComputeAndCompare(&builder_, {}, DefaultErrorSpec());
}

INSTANTIATE_TEST_CASE_P(ReduceWindowTestInstance, ReduceWindowTest,
                        ::testing::ValuesIn(use_bfloat16_params));

enum Reducer { kAdd, kMax };

struct R4ReduceWindowTestData {
  int64_t base_bounds[4];
  int64_t window_bounds[4];
  int64_t strides[4];
  int64_t pad_low[4];
  int64_t pad_high[4];
  int64_t layout[4];

  Reducer reducer;
};

std::string R4ReduceWindowTestDataToString(
    const ::testing::TestParamInfo<
        ::testing::tuple<R4ReduceWindowTestData, bool>>& data) {
  const auto& param = ::testing::get<0>(data.param);
  std::string str = absl::StrCat(
      "base_bounds_", absl::StrJoin(param.base_bounds, "x"),        //
      "__window_bounds_", absl::StrJoin(param.window_bounds, "x"),  //
      "__strides_", absl::StrJoin(param.strides, "x"),              //
      "__pad_low_", absl::StrJoin(param.pad_low, "x"),              //
      "__pad_high_", absl::StrJoin(param.pad_high, "x"),            //
      "__layout_", absl::StrJoin(param.layout, "_"),                //
      (param.reducer == kAdd) ? "_add" : "_max");
  CHECK(param.reducer == kAdd || param.reducer == kMax);

  // Test names are not allowed to contain the '-' character.
  std::replace(str.begin(), str.end(), '-', 'n');
  if (::testing::get<1>(data.param)) {
    absl::StrAppend(&str, "_bfloat16");
  }
  return str;
}

class R4ReduceWindowTest : public ReduceWindowTestBase,
                           public ::testing::WithParamInterface<
                               ::testing::tuple<R4ReduceWindowTestData, bool>> {
 protected:
  R4ReduceWindowTest() { set_use_bfloat16(::testing::get<1>(GetParam())); }

  void DoIt() {
    XlaBuilder b(TestName());
    const auto& param = ::testing::get<0>(GetParam());

    const float kInitValue = 0.0f;

    Array4D<float> input(param.base_bounds[0], param.base_bounds[1],
                         param.base_bounds[2], param.base_bounds[3]);
    // Choose a prime iota length so that each window sees a unique set of
    // values. (Technically, the requirement is that the iota length is
    // relatively prime to all of the dimensions involved in the reduce-window.)
    input.FillRepeatedIota(0, 137);
    // Floating point sum reduction requires higher localized precision. We need
    // the following normalization in order to enable testing of kAdd on large
    // windows.
    input.Each([&](absl::Span<const int64_t> /*indices*/, float* value) {
      *value = *value / 10000000000.f;
    });
    Literal input_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
        input, LayoutUtil::MakeLayout(param.layout));
    XlaOp parameter;
    TF_ASSERT_OK_AND_ASSIGN(auto input_arg,
                            CreateParameterAndTransferLiteral(
                                0, input_literal, "p0", &b, &parameter));

    std::vector<std::pair<int64_t, int64_t>> padding(4);
    for (int i = 0; i < 4; ++i) {
      padding[i] = {param.pad_low[i], param.pad_high[i]};
    }

    auto init_value =
        CreateConstantFromLiteral(LiteralUtil::CreateR0(kInitValue), &b);
    CHECK(param.reducer == kAdd || param.reducer == kMax);
    auto reducer = param.reducer;
    auto computation = reducer == kAdd
                           ? CreateScalarAddComputation(FloatType(), &b)
                           : CreateScalarMaxComputation(FloatType(), &b);
    ReduceWindowWithGeneralPadding(
        /*operand=*/parameter,
        /*init_value=*/init_value,
        /*computation=*/computation,
        /*window_dimensions=*/param.window_bounds,
        /*window_strides=*/param.strides,
        /*base_dilations=*/{},
        /*window_dilations=*/{},
        /*padding=*/padding);

    CHECK(reducer == kAdd || reducer == kMax);
    auto reduce_func = reducer == kAdd
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
    Literal expected_literal = LiteralUtil::CreateFromArray(*expected);
    const Shape& expected_shape_with_layout = ShapeUtil::MakeShapeWithLayout(
        input_literal.shape().element_type(),
        expected_literal.shape().dimensions(), param.layout);
    ComputeAndCompareLiteral(&b, expected_literal, {input_arg.get()},
                             DefaultErrorSpec(), &expected_shape_with_layout);
  }
};

XLA_TEST_P(R4ReduceWindowTest, DoIt) { DoIt(); }

// base_bounds, window_bounds, strides, pad_low, pad_high
const R4ReduceWindowTestData kR4ReduceWindowTestValues[] = {
    // Minimal edge case.
    R4ReduceWindowTestData{/*base_bounds=*/{1, 1, 1, 1},
                           /*window_bounds=*/{1, 1, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // Arbitrary padding (not kSame or kValid).
    R4ReduceWindowTestData{/*base_bounds=*/{9, 12, 4, 89},
                           /*window_bounds=*/{3, 3, 1, 1},
                           /*strides=*/{2, 2, 1, 1},
                           /*pad_low=*/{4, 4, 0, 0},
                           /*pad_high=*/{4, 4, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // Zero base bound edge case.
    R4ReduceWindowTestData{/*base_bounds=*/{1, 0, 1, 1},
                           /*window_bounds=*/{1, 1, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // With max instead of add.
    R4ReduceWindowTestData{/*base_bounds=*/{4, 6, 17, 140},
                           /*window_bounds=*/{2, 3, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kMax},

    // With stride.
    R4ReduceWindowTestData{/*base_bounds=*/{4, 10, 17, 140},
                           /*window_bounds=*/{3, 2, 1, 1},
                           /*strides=*/{2, 4, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // With low padding.
    R4ReduceWindowTestData{/*base_bounds=*/{4, 6, 17, 140},
                           /*window_bounds=*/{3, 2, 1, 1},
                           /*strides=*/{2, 2, 1, 1},
                           /*pad_low=*/{3, 2, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // With high padding.
    R4ReduceWindowTestData{/*base_bounds=*/{4, 6, 17, 140},
                           /*window_bounds=*/{3, 2, 1, 1},
                           /*strides=*/{2, 2, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{2, 3, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // With negative padding on both ends.
    R4ReduceWindowTestData{/*base_bounds=*/{10, 10, 17, 140},
                           /*window_bounds=*/{3, 2, 1, 1},
                           /*strides=*/{2, 2, 1, 1},
                           /*pad_low=*/{-3, -2, 0, 0},
                           /*pad_high=*/{-2, -3, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // With negative low padding and positive high padding.
    R4ReduceWindowTestData{/*base_bounds=*/{10, 10, 17, 140},
                           /*window_bounds=*/{3, 2, 1, 1},
                           /*strides=*/{2, 2, 1, 1},
                           /*pad_low=*/{-3, -2, 0, 0},
                           /*pad_high=*/{2, 3, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // With positive low padding and negative high padding.
    R4ReduceWindowTestData{/*base_bounds=*/{10, 10, 17, 140},
                           /*window_bounds=*/{3, 2, 1, 1},
                           /*strides=*/{2, 2, 1, 1},
                           /*pad_low=*/{3, 2, 0, 0},
                           /*pad_high=*/{-2, -3, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // Window touches both sides of the padding simultaneously.
    R4ReduceWindowTestData{/*base_bounds=*/{1, 1, 17, 140},
                           /*window_bounds=*/{3, 3, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{1, 1, 0, 0},
                           /*pad_high=*/{1, 1, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // Window is entirely in the padding for some positions.
    R4ReduceWindowTestData{/*base_bounds=*/{1, 1, 17, 140},
                           /*window_bounds=*/{3, 3, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{4, 4, 0, 0},
                           /*pad_high=*/{4, 4, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // Zero base bound with padding edge case.
    R4ReduceWindowTestData{/*base_bounds=*/{2, 0, 3, 4},
                           /*window_bounds=*/{1, 1, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 1, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // With stride, low padding and high padding.
    R4ReduceWindowTestData{/*base_bounds=*/{4, 3, 17, 140},
                           /*window_bounds=*/{3, 4, 1, 1},
                           /*strides=*/{3, 1, 1, 1},
                           /*pad_low=*/{10, 1, 0, 0},
                           /*pad_high=*/{2, 3, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // With minor dimension == 129.
    R4ReduceWindowTestData{/*base_bounds=*/{3, 2, 7, 129},
                           /*window_bounds=*/{1, 1, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // With minor dims reduction and non-overlapped stride.
    R4ReduceWindowTestData{/*base_bounds=*/{2, 2, 4, 16},
                           /*window_bounds=*/{1, 1, 2, 2},
                           /*strides=*/{1, 1, 2, 2},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // With minor dims reduction and overlapped stride.
    R4ReduceWindowTestData{/*base_bounds=*/{2, 2, 4, 16},
                           /*window_bounds=*/{1, 1, 4, 4},
                           /*strides=*/{1, 1, 2, 2},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{1, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    R4ReduceWindowTestData{/*base_bounds=*/{8, 100, 100, 3},
                           /*window_bounds=*/{1, 64, 64, 1},
                           /*strides=*/{1, 64, 64, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 0, 2, 1},
                           /*reducer=*/kAdd},

    R4ReduceWindowTestData{/*base_bounds=*/{112, 112, 8, 64},
                           /*window_bounds=*/{112, 112, 1, 8},
                           /*strides=*/{112, 112, 1, 8},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kMax},

    R4ReduceWindowTestData{/*base_bounds=*/{4, 6, 17, 140},
                           /*window_bounds=*/{2, 3, 4, 5},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    // With 0321 layout.
    R4ReduceWindowTestData{/*base_bounds=*/{4, 6, 17, 140},
                           /*window_bounds=*/{2, 3, 4, 5},
                           /*strides=*/{1, 2, 3, 4},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{0, 3, 2, 1},
                           /*reducer=*/kAdd},

    // With 0123 layout.
    R4ReduceWindowTestData{/*base_bounds=*/{4, 6, 13, 17},
                           /*window_bounds=*/{2, 3, 7, 9},
                           /*strides=*/{1, 2, 5, 8},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{0, 1, 2, 3},
                           /*reducer=*/kAdd},
};

INSTANTIATE_TEST_CASE_P(
    R4ReduceWindowTestInstantiation, R4ReduceWindowTest,
    ::testing::Combine(::testing::ValuesIn(kR4ReduceWindowTestValues),
                       ::testing::ValuesIn(use_bfloat16_params)),
    R4ReduceWindowTestDataToString);

class R4ReduceWindowLargeTest : public R4ReduceWindowTest {};

XLA_TEST_P(R4ReduceWindowLargeTest, DISABLED_ON_INTERPRETER(DoIt)) { DoIt(); }

// Test cases that are large/slow/failed.
const R4ReduceWindowTestData kR4ReduceWindowLargeTestValues[] = {
    R4ReduceWindowTestData{/*base_bounds=*/{28, 28, 256, 128},
                           /*window_bounds=*/{3, 3, 1, 5},
                           /*strides=*/{1, 1, 1, 5},
                           /*pad_low=*/{1, 1, 0, 0},
                           /*pad_high=*/{1, 1, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kMax},

    R4ReduceWindowTestData{/*base_bounds=*/{112, 112, 64, 128},
                           /*window_bounds=*/{3, 3, 1, 1},
                           /*strides=*/{2, 2, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{1, 1, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    R4ReduceWindowTestData{/*base_bounds=*/{1, 1, 32768 - 3, 2},
                           /*window_bounds=*/{1, 1, 4, 1},
                           /*strides=*/{1, 1, 4, 1},
                           /*pad_low=*/{0, 0, 1, 0},
                           /*pad_high=*/{0, 0, 2, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kMax},

    // Patterns generated by cumsum/cumprod.
    R4ReduceWindowTestData{/*base_bounds=*/{1021, 1, 16, 16},
                           /*window_bounds=*/{1021, 1, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{1020, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    R4ReduceWindowTestData{/*base_bounds=*/{1021, 1, 16, 16},
                           /*window_bounds=*/{1, 1, 1021, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 0, 1020, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    R4ReduceWindowTestData{/*base_bounds=*/{16, 1, 16, 1021},
                           /*window_bounds=*/{1, 1, 1, 1021},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 0, 0, 1020},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    R4ReduceWindowTestData{/*base_bounds=*/{1021, 1, 16, 16},
                           /*window_bounds=*/{1021, 1, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{1021, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    R4ReduceWindowTestData{/*base_bounds=*/{16, 1, 1021, 16},
                           /*window_bounds=*/{1, 1, 1021, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 0, 1021, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},

    R4ReduceWindowTestData{/*base_bounds=*/{16, 1, 16, 1021},
                           /*window_bounds=*/{1, 1, 1, 1021},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 0, 0, 1021},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*layout=*/{3, 2, 1, 0},
                           /*reducer=*/kAdd},
};

INSTANTIATE_TEST_CASE_P(
    R4ReduceWindowLargeTestInstantiation, R4ReduceWindowLargeTest,
    ::testing::Combine(::testing::ValuesIn(kR4ReduceWindowLargeTestValues),
                       ::testing::ValuesIn(use_bfloat16_params)),
    R4ReduceWindowTestDataToString);

struct R3ReduceWindowTestData {
  int64_t base_bounds[3];
  int64_t window_bounds[3];
  int64_t strides[3];
  int64_t layout[3];
  Padding padding;
  Reducer reducer;
} kR3TestCases[] = {
    {/*base_bounds=*/{2, 1, 2}, /*window_bounds=*/{1, 1, 2},
     /*strides=*/{1, 1, 1}, /*layout=*/{2, 1, 0},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{4, 3, 3}, /*window_bounds=*/{2, 2, 2},
     /*strides=*/{2, 2, 2}, /*layout=*/{2, 1, 0},
     /*padding=*/Padding::kSame, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{4, 3, 3}, /*window_bounds=*/{2, 2, 2},
     /*strides=*/{2, 2, 2}, /*layout=*/{2, 1, 0},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{6, 21, 3}, /*window_bounds=*/{2, 3, 2},
     /*strides=*/{1, 2, 2}, /*layout=*/{2, 1, 0},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{10, 21, 129}, /*window_bounds=*/{2, 9, 1},
     /*strides=*/{5, 2, 1}, /*layout=*/{2, 1, 0},
     /*padding=*/Padding::kSame, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{6, 21, 3}, /*window_bounds=*/{2, 3, 2},
     /*strides=*/{1, 2, 2}, /*layout=*/{0, 1, 2},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{6, 21, 3}, /*window_bounds=*/{2, 3, 2},
     /*strides=*/{1, 2, 2}, /*layout=*/{1, 0, 2},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{95, 202, 251}, /*window_bounds=*/{95, 202, 251},
     /*strides=*/{1, 1, 1}, /*layout=*/{2, 1, 0},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kMax},
    {/*base_bounds=*/{999, 57, 3}, /*window_bounds=*/{999, 57, 3},
     /*strides=*/{1, 1, 1}, /*layout=*/{2, 1, 0},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{178, 302, 64}, /*window_bounds=*/{178, 302, 64},
     /*strides=*/{1, 1, 1}, /*layout=*/{2, 1, 0},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kMax},
    {/*base_bounds=*/{63, 261, 257}, /*window_bounds=*/{63, 261, 257},
     /*strides=*/{1, 1, 1}, /*layout=*/{2, 1, 0},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kMax},
    {/*base_bounds=*/{10003, 10, 5}, /*window_bounds=*/{9999, 7, 3},
     /*strides=*/{1, 1, 1}, /*layout=*/{2, 1, 0},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{9999, 1, 1}, /*window_bounds=*/{9999, 1, 1},
     /*strides=*/{1, 1, 1}, /*layout=*/{2, 1, 0},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{10003, 10, 5}, /*window_bounds=*/{9999, 7, 3},
     /*strides=*/{2, 2, 2}, /*layout=*/{2, 1, 0},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},
};

std::string R3ReduceWindowTestDataToString(
    const ::testing::TestParamInfo<
        ::testing::tuple<R3ReduceWindowTestData, bool>>& data) {
  const auto& param = ::testing::get<0>(data.param);
  std::string str = absl::StrCat(
      "base_bounds_", absl::StrJoin(param.base_bounds, "x"), "__window_bounds_",
      absl::StrJoin(param.window_bounds, "x"), "__strides_",
      absl::StrJoin(param.strides, "x"), "__padding_",
      param.padding == Padding::kSame ? "same" : "valid", "__layout_",
      param.layout[0], "_", param.layout[1], "_", param.layout[2], "__reducer_",
      param.reducer == kAdd ? "add" : "max");
  if (::testing::get<1>(data.param)) {
    absl::StrAppend(&str, "_bfloat16");
  }
  return str;
}

class R3ReduceWindowTest : public ReduceWindowTestBase,
                           public ::testing::WithParamInterface<
                               ::testing::tuple<R3ReduceWindowTestData, bool>> {
 protected:
  R3ReduceWindowTest() { set_use_bfloat16(::testing::get<1>(GetParam())); }
};

XLA_TEST_P(R3ReduceWindowTest, DoIt) {
  XlaBuilder b(TestName());
  const auto& param = ::testing::get<0>(GetParam());

  const float kInitValue = 0.0f;
  Array3D<float> input(param.base_bounds[0], param.base_bounds[1],
                       param.base_bounds[2]);
  // Choose a prime iota length so that each window sees a unique set of values.
  // (Technically, the requirement is that the iota length is relatively prime
  // to all of the dimensions involved in the reduce-window.)
  input.FillRepeatedIota(0, 137);
  Literal input_literal = LiteralUtil::CreateR3FromArray3DWithLayout(
      input, LayoutUtil::MakeLayout(param.layout));
  auto reducer = param.reducer;
  if (use_bfloat16()) {
    input_literal = LiteralUtil::ConvertF32ToBF16(input_literal);

    // To avoid numerical issues, force the reducer to be kMax for bf16
    // inputs.
    reducer = kMax;
  }

  XlaOp parameter = Parameter(&b, 0, input_literal.shape(), "input");
  auto init_value =
      CreateConstantFromLiteral(LiteralUtil::CreateR0(kInitValue), &b);

  auto computation = reducer == kAdd
                         ? CreateScalarAddComputation(FloatType(), &b)
                         : CreateScalarMaxComputation(FloatType(), &b);

  ReduceWindow(/*operand=*/parameter,
               /*init_value=*/init_value,
               /*computation=*/computation,
               /*window_dimensions=*/param.window_bounds,
               /*window_strides=*/param.strides, /*padding=*/param.padding);

  ComputeAndCompare(&b, {std::move(input_literal)}, DefaultErrorSpec());
}

INSTANTIATE_TEST_CASE_P(
    R3ReduceWindowTestInstantiation, R3ReduceWindowTest,
    ::testing::Combine(::testing::ValuesIn(kR3TestCases),
                       ::testing::ValuesIn(use_bfloat16_params)),
    R3ReduceWindowTestDataToString);

struct R2ReduceWindowTestData {
  int64_t base_bounds[2];
  int64_t window_bounds[2];
  int64_t strides[2];
  int64_t base_dilation[2];
  int64_t window_dilation[2];
  int64_t pad_low[2];
  int64_t pad_high[2];
  int64_t layout[2];
  Reducer reducer;
} kR2TestCases[] = {
    {/*base_bounds=*/{4, 18}, /*window_bounds=*/{2, 4},
     /*strides=*/{1, 2},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 1}, /*pad_high=*/{1, 1},
     /*layout=*/{0, 1},
     /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{2, 5}, /*window_bounds=*/{2, 4},
     /*strides=*/{1, 1},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 1}, /*pad_high=*/{1, 2},
     /*layout=*/{0, 1},
     /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{1, 3}, /*window_bounds=*/{2, 3},
     /*strides=*/{1, 1},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 1}, /*pad_high=*/{1, 1},
     /*layout=*/{0, 1},
     /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{3, 129}, /*window_bounds=*/{1, 100},
     /*strides=*/{2, 99},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 0}, /*pad_high=*/{35, 35},
     /*layout=*/{0, 1},
     /*reducer=*/Reducer::kAdd},
// TODO(b/74260408): This test last failed on GPU on 2018-03-08, likely due to a
// ptxas bug.
#ifndef XLA_TEST_BACKEND_GPU
    {/*base_bounds=*/{6, 152}, /*window_bounds=*/{2, 25},
     /*strides=*/{5, 4},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 1}, /*pad_high=*/{10, 11},
     /*layout=*/{0, 1},
     /*reducer=*/Reducer::kAdd},
#endif
    {/*base_bounds=*/{6, 4}, /*window_bounds=*/{4, 2},
     /*strides=*/{3, 3},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 1}, /*pad_high=*/{0, 1},
     /*layout=*/{0, 1},
     /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{5, 147}, /*window_bounds=*/{1, 36},
     /*strides=*/{4, 5},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 0}, /*pad_high=*/{17, 17},
     /*layout=*/{1, 0},
     /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{4, 153}, /*window_bounds=*/{2, 93},
     /*strides=*/{1, 1},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 1}, /*pad_high=*/{46, 46},
     /*layout=*/{1, 0},
     /*reducer=*/Reducer::kAdd},
    // Regression test for a bug that appeared in Inception (b/34784899).
    {/*base_bounds=*/{28, 28}, /*window_bounds=*/{3, 3},
     /*strides=*/{1, 1},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{1, 1}, /*pad_high=*/{1, 1},
     /*layout=*/{1, 0},
     /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{4, 4}, /*window_bounds=*/{2, 2},
     /*strides=*/{1, 1},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 0}, /*pad_high=*/{0, 0},
     /*layout=*/{1, 0},
     /*reducer=*/Reducer::kMax},
    {/*base_bounds=*/{4, 4}, /*window_bounds=*/{2, 2},
     /*strides=*/{1, 1},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{2, 2},
     /*pad_low=*/{0, 0}, /*pad_high=*/{0, 0},
     /*layout=*/{1, 0},
     /*reducer=*/Reducer::kMax},
    {/*base_bounds=*/{4, 4}, /*window_bounds=*/{2, 2},
     /*strides=*/{1, 1},
     /*base_dilation=*/{2, 2}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 0}, /*pad_high=*/{0, 0},
     /*layout=*/{1, 0},
     /*reducer=*/Reducer::kMax},
    {/*base_bounds=*/{4, 4}, /*window_bounds=*/{2, 2},
     /*strides=*/{2, 2},
     /*base_dilation=*/{2, 2}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 0}, /*pad_high=*/{0, 0},
     /*layout=*/{1, 0},
     /*reducer=*/Reducer::kMax},
    {/*base_bounds=*/{4, 4}, /*window_bounds=*/{2, 2},
     /*strides=*/{2, 2},
     /*base_dilation=*/{2, 2}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{3, 3}, /*pad_high=*/{3, 3},
     /*layout=*/{1, 0},
     /*reducer=*/Reducer::kMax},
    {/*base_bounds=*/{4, 4}, /*window_bounds=*/{2, 2},
     /*strides=*/{2, 2},
     /*base_dilation=*/{2, 2}, /*window_dilation=*/{2, 2},
     /*pad_low=*/{0, 0}, /*pad_high=*/{0, 0},
     /*layout=*/{1, 0},
     /*reducer=*/Reducer::kMax},
    // Regression test for a bug that appeared in Inception (b/34784899).
    {/*base_bounds=*/{4, 32}, /*window_bounds=*/{2, 2},
     /*strides=*/{2, 2},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 0}, /*pad_high=*/{0, 0},
     /*layout=*/{1, 0},
     /*reducer=*/Reducer::kAdd},
    // Regression test for b/73903312: bf16 lacks precision to store result of
    // very large windows. Testing with a reasonable window larger than 128.
    {/*base_bounds=*/{8, 130}, /*window_bounds=*/{1, 130},
     /*strides=*/{1, 1},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 130}, /*pad_high=*/{0, 0},
     /*layout=*/{1, 0},
     /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{8, 256}, /*window_bounds=*/{1, 4},
     /*strides=*/{1, 64},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 0}, /*pad_high=*/{0, 0},
     /*layout=*/{1, 0}, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{4096, 4096}, /*window_bounds=*/{1, 4},
     /*strides=*/{1, 1024},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 0}, /*pad-high=*/{0, 0},
     /*layout=*/{1, 0}, /*reducer=*/Reducer::kAdd},
    // Regression test for b/72234705: bf16 lacks precision to store incremental
    // results on very large windows. Using smaller window with minor dim 128.
    {/*base_bounds=*/{8, 128}, /*window_bounds=*/{2, 128},
     /*strides=*/{1, 1},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 0}, /*pad-high=*/{0, 0},
     /*layout=*/{1, 0}, /*reducer=*/Reducer::kAdd},
    // With negative padding on both ends.
    {/*base_bounds=*/{4, 18}, /*window_bounds=*/{2, 4},
     /*strides=*/{1, 2},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, -1}, /*pad_high=*/{0, -1},
     /*layout=*/{0, 1},
     /*reducer=*/Reducer::kAdd},
    // With negative low padding and positive high padding.
    {/*base_bounds=*/{4, 18}, /*window_bounds=*/{2, 4},
     /*strides=*/{1, 2},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, -1}, /*pad_high=*/{0, 1},
     /*layout=*/{0, 1},
     /*reducer=*/Reducer::kAdd},
    // With positive low padding and negative high padding.
    {/*base_bounds=*/{4, 18}, /*window_bounds=*/{2, 4},
     /*strides=*/{1, 2},
     /*base_dilation=*/{1, 1}, /*window_dilation=*/{1, 1},
     /*pad_low=*/{0, 1}, /*pad_high=*/{0, -1},
     /*layout=*/{0, 1},
     /*reducer=*/Reducer::kAdd},
};

std::string R2ReduceWindowTestDataToString(
    const ::testing::TestParamInfo<
        ::testing::tuple<R2ReduceWindowTestData, bool>>& data) {
  const auto& param = ::testing::get<0>(data.param);
  std::string str = absl::StrCat(
      "base_bounds_", absl::StrJoin(param.base_bounds, "x"),            //
      "__window_bounds_", absl::StrJoin(param.window_bounds, "x"),      //
      "__strides_", absl::StrJoin(param.strides, "x"),                  //
      "__base_dilation_", absl::StrJoin(param.base_dilation, "x"),      //
      "__window_dilation_", absl::StrJoin(param.window_dilation, "x"),  //
      "__pad_low_", absl::StrJoin(param.pad_low, "x"), "__pad_high_",
      absl::StrJoin(param.pad_high, "x"), "__layout_", param.layout[0], "_",
      param.layout[1],  //
      "__reducer_", param.reducer == kAdd ? "add" : "max");

  // Test names are not allowed to contain the '-' character.
  std::replace(str.begin(), str.end(), '-', 'n');
  if (::testing::get<1>(data.param)) {
    absl::StrAppend(&str, "_bfloat16");
  }
  return str;
}

class R2ReduceWindowTest : public ReduceWindowTestBase,
                           public ::testing::WithParamInterface<
                               ::testing::tuple<R2ReduceWindowTestData, bool>> {
 protected:
  R2ReduceWindowTest() { set_use_bfloat16(::testing::get<1>(GetParam())); }

  void DoIt() {
    XlaBuilder b(TestName());
    const auto& param = ::testing::get<0>(GetParam());

    Array2D<float> input(param.base_bounds[0], param.base_bounds[1], 1.0f);
    if (!::testing::get<1>(GetParam())) {
      // We only do this in F32 mode, to avoid precision issues with BF16.
      input = *MakeLinspaceArray2D(0, 100, param.base_bounds[0],
                                   param.base_bounds[1]);
    }
    Literal input_literal = LiteralUtil::CreateR2FromArray2DWithLayout(
        input, LayoutUtil::MakeLayout(param.layout));

    XlaOp parameter;
    TF_ASSERT_OK(CreateParameterAndTransferLiteral(0, input_literal, "p0", &b,
                                                   &parameter)
                     .status());

    std::vector<std::pair<int64_t, int64_t>> padding(2);
    for (int i = 0; i < 2; ++i) {
      padding[i] = {param.pad_low[i], param.pad_high[i]};
    }
    auto computation = param.reducer == kAdd
                           ? CreateScalarAddComputation(FloatType(), &b)
                           : CreateScalarMaxComputation(FloatType(), &b);
    const float kInitValue = 0.0f;
    auto init_value =
        CreateConstantFromLiteral(LiteralUtil::CreateR0(kInitValue), &b);
    ReduceWindowWithGeneralPadding(
        /*operand=*/parameter,
        /*init_value=*/init_value,
        /*computation=*/computation,
        /*window_dimensions=*/param.window_bounds,
        /*window_strides=*/param.strides,
        /*base_dilations=*/param.base_dilation,
        /*window_dilations=*/param.window_dilation,
        /*padding=*/padding);

    ComputeAndCompare(&b, {MaybeConvertLiteralToBfloat16(input_literal)},
                      DefaultErrorSpec());
  }
};

XLA_TEST_P(R2ReduceWindowTest, DoIt) { DoIt(); }

INSTANTIATE_TEST_CASE_P(
    R2ReduceWindowTestInstantiation, R2ReduceWindowTest,
    ::testing::Combine(::testing::ValuesIn(kR2TestCases),
                       ::testing::ValuesIn(use_bfloat16_params)),
    R2ReduceWindowTestDataToString);

struct R1ReduceWindowTestData {
  int64_t base_bounds[1];
  int64_t window_bounds[1];
  int64_t strides[1];
  int64_t pad_low[1];
  int64_t pad_high[1];
  Reducer reducer;
} kR1TestCases[] = {
    {/*base_bounds=*/{1}, /*window_bounds=*/{1},
     /*strides=*/{1},
     /*pad_low=*/{xla::MakePadding({1}, {1}, {1}, Padding::kValid)[0].first},
     /*pad_high=*/{xla::MakePadding({1}, {1}, {1}, Padding::kValid)[0].second},
     /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{3}, /*window_bounds=*/{3},
     /*strides=*/{1},
     /*pad_low=*/{xla::MakePadding({3}, {3}, {1}, Padding::kValid)[0].first},
     /*pad_high=*/{xla::MakePadding({3}, {3}, {1}, Padding::kValid)[0].second},
     /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{3}, /*window_bounds=*/{2},
     /*strides=*/{1},
     /*pad_low=*/{xla::MakePadding({3}, {2}, {1}, Padding::kValid)[0].first},
     /*pad_high=*/{xla::MakePadding({3}, {2}, {1}, Padding::kValid)[0].second},
     /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{5}, /*window_bounds=*/{1},
     /*strides=*/{1},
     /*pad_low=*/{xla::MakePadding({5}, {1}, {1}, Padding::kValid)[0].first},
     /*pad_high=*/{xla::MakePadding({5}, {1}, {1}, Padding::kValid)[0].second},
     /*reducer=*/Reducer::kMax},

    {/*base_bounds=*/{16}, /*window_bounds=*/{4},
     /*strides=*/{4},
     /*pad_low=*/{xla::MakePadding({16}, {4}, {4}, Padding::kValid)[0].first},
     /*pad_high=*/{xla::MakePadding({16}, {4}, {4}, Padding::kValid)[0].second},
     /*reducer=*/Reducer::kMax},

    {/*base_bounds=*/{16}, /*window_bounds=*/{4},
     /*strides=*/{3},
     /*pad_low=*/{xla::MakePadding({16}, {4}, {3}, Padding::kValid)[0].first},
     /*pad_high=*/{xla::MakePadding({16}, {4}, {3}, Padding::kValid)[0].second},
     /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{128 * 2},
     /*window_bounds=*/{30},
     /*strides=*/{27},
     /*pad_low=*/
     {xla::MakePadding({128 * 2}, {30}, {27}, Padding::kValid)[0].first},
     /*pad_high=*/
     {xla::MakePadding({128 * 2}, {30}, {27}, Padding::kValid)[0].second},
     /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{128 * 17},
     /*window_bounds=*/{7},
     /*strides=*/{64},
     /*pad_low=*/
     {xla::MakePadding({128 * 17}, {7}, {64}, Padding::kValid)[0].first},
     /*pad_high=*/
     {xla::MakePadding({128 * 17}, {7}, {64}, Padding::kValid)[0].second},
     /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{128 * 2},
     /*window_bounds=*/{32},
     /*strides=*/{56},
     /*pad_low=*/
     {xla::MakePadding({128 * 2}, {32}, {56}, Padding::kValid)[0].first},
     /*pad_high=*/
     {xla::MakePadding({128 * 2}, {32}, {56}, Padding::kValid)[0].second},
     /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{3}, /*window_bounds=*/{2},
     /*strides=*/{1},
     /*pad_low=*/{xla::MakePadding({3}, {2}, {1}, Padding::kSame)[0].first},
     /*pad_high=*/{xla::MakePadding({3}, {2}, {1}, Padding::kSame)[0].second},
     /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{5}, /*window_bounds=*/{3},
     /*strides=*/{2},
     /*pad_low=*/{xla::MakePadding({5}, {3}, {2}, Padding::kSame)[0].first},
     /*pad_high=*/{xla::MakePadding({5}, {3}, {2}, Padding::kSame)[0].second},
     /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{16}, /*window_bounds=*/{4},
     /*strides=*/{3},
     /*pad_low=*/{xla::MakePadding({16}, {4}, {3}, Padding::kSame)[0].first},
     /*pad_high=*/{xla::MakePadding({16}, {4}, {3}, Padding::kSame)[0].second},
     /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{5}, /*window_bounds=*/{5},
     /*strides=*/{1},
     /*pad_low=*/{0},
     /*pad_high=*/{5},
     /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{5}, /*window_bounds=*/{5},
     /*strides=*/{1},
     /*pad_low=*/{5},
     /*pad_high=*/{0},
     /*reducer=*/Reducer::kAdd},

    // With negative padding on both ends.
    {/*base_bounds=*/{15}, /*window_bounds=*/{5},
     /*strides=*/{1},
     /*pad_low=*/{-5},
     /*pad_high=*/{-5},
     /*reducer=*/Reducer::kAdd},

    // With negative low padding and positive high padding.
    {/*base_bounds=*/{15}, /*window_bounds=*/{5},
     /*strides=*/{1},
     /*pad_low=*/{-5},
     /*pad_high=*/{5},
     /*reducer=*/Reducer::kAdd},

    // With positive low padding and negative high padding.
    {/*base_bounds=*/{15}, /*window_bounds=*/{5},
     /*strides=*/{1},
     /*pad_low=*/{5},
     /*pad_high=*/{-5},
     /*reducer=*/Reducer::kAdd},

    // The pattern generated by inclusive scan (cumsum/cumprod).
    {/*base_bounds=*/{4096}, /*window_bounds=*/{4096},
     /*strides=*/{1},
     /*pad_low=*/{4095},
     /*pad_high=*/{0},
     /*reducer=*/Reducer::kMax},

    // The pattern generated by exclusive scan (cumsum/cumprod).
    {/*base_bounds=*/{4095}, /*window_bounds=*/{4095},
     /*strides=*/{1},
     /*pad_low=*/{4095},
     /*pad_high=*/{0},
     /*reducer=*/Reducer::kMax},

    // The pattern generated by inclusive reverse scan (cumsum/cumprod).
    {/*base_bounds=*/{4096}, /*window_bounds=*/{4096},
     /*strides=*/{1},
     /*pad_low=*/{0},
     /*pad_high=*/{4095},
     /*reducer=*/Reducer::kMax},

    // The pattern generated by exclusive reverse scan (cumsum/cumprod).
    {/*base_bounds=*/{4095}, /*window_bounds=*/{4095},
     /*strides=*/{1},
     /*pad_low=*/{0},
     /*pad_high=*/{4095},
     /*reducer=*/Reducer::kMax},
};

std::string R1ReduceWindowTestDataToString(
    const ::testing::TestParamInfo<
        ::testing::tuple<R1ReduceWindowTestData, bool>>& data) {
  const auto& param = ::testing::get<0>(data.param);
  std::string str =
      absl::StrCat("base_bounds_", absl::StrJoin(param.base_bounds, "x"),
                   "__window_bounds_", absl::StrJoin(param.window_bounds, "x"),
                   "__strides_", absl::StrJoin(param.strides, "x"),
                   "__pad_low_", absl::StrJoin(param.pad_low, "x"),
                   "__pad_high_", absl::StrJoin(param.pad_high, "x"),
                   "__reducer_", param.reducer == kAdd ? "add" : "max");

  // Test names are not allowed to contain the '-' character.
  std::replace(str.begin(), str.end(), '-', 'n');
  if (::testing::get<1>(data.param)) {
    absl::StrAppend(&str, "_bfloat16");
  }
  return str;
}

class R1ReduceWindowTest : public ReduceWindowTestBase,
                           public ::testing::WithParamInterface<
                               ::testing::tuple<R1ReduceWindowTestData, bool>> {
 protected:
  R1ReduceWindowTest() { set_use_bfloat16(::testing::get<1>(GetParam())); }
};

XLA_TEST_P(R1ReduceWindowTest, DoIt) {
  XlaBuilder b(TestName());
  const auto& param = ::testing::get<0>(GetParam());
  CHECK(param.reducer == kAdd || param.reducer == kMax);

  const float kInitValue = 0.0f;
  std::vector<float> input_vector(param.base_bounds[0]);
  std::iota(std::begin(input_vector), std::end(input_vector), 0);
  Literal input_literal =
      LiteralUtil::CreateR1(absl::Span<const float>(input_vector));
  XlaOp parameter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto input_arg, CreateParameterAndTransferLiteral(0, input_literal, "p0",
                                                        &b, &parameter));

  std::vector<std::pair<int64_t, int64_t>> padding(1);
  padding[0] = {param.pad_low[0], param.pad_high[0]};

  auto computation = param.reducer == kAdd
                         ? CreateScalarAddComputation(FloatType(), &b)
                         : CreateScalarMaxComputation(FloatType(), &b);
  auto init_value =
      CreateConstantFromLiteral(LiteralUtil::CreateR0(kInitValue), &b);
  ReduceWindowWithGeneralPadding(
      /*operand=*/parameter,
      /*init_value=*/init_value,
      /*computation=*/computation,
      /*window_dimensions=*/param.window_bounds,
      /*window_strides=*/param.strides,
      /*base_dilations=*/{},
      /*window_dilations=*/{},
      /*padding=*/padding);

  auto reduce_func = param.reducer == kAdd
                         ? +[](float a, float b) { return a + b; }
                         : +[](float a, float b) { return std::max(a, b); };
  auto expected = ReferenceUtil::ReduceWindow1DGeneric(
      /*operand=*/absl::Span<const float>(input_vector),
      /*init=*/kInitValue,
      /*reduce_func=*/reduce_func,
      /*window=*/param.window_bounds,
      /*stride=*/param.strides,
      /*padding=*/padding);

  ComputeAndCompareLiteral(&b, LiteralUtil::CreateR1<float>(*expected),
                           {input_arg.get()}, DefaultErrorSpec());
}

INSTANTIATE_TEST_CASE_P(
    R1ReduceWindowTestInstantiation, R1ReduceWindowTest,
    ::testing::Combine(::testing::ValuesIn(kR1TestCases),
                       ::testing::ValuesIn(use_bfloat16_params)),
    R1ReduceWindowTestDataToString);

// Test class for text-based test cases. Note that this compares with the
// results on the interpreter backend.
class ReduceWindowTextTest : public HloTestBase {};

XLA_TEST_F(ReduceWindowTextTest, R2General256x384) {
  const std::string hlo_string = R"(
HloModule R2Window
mul {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT mul = f32[] multiply(lhs, rhs)
}
ENTRY R2Window {
  operand = f32[256,384]{1,0} parameter(0)
  constant = f32[] constant(1)
  ROOT reduce-window = f32[256,384]{1,0} reduce-window(operand, constant), window={size=2x3 pad=0_1x1_1}, to_apply=mul
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{0.001}));
}

XLA_TEST_F(ReduceWindowTextTest, R2General256x384Layout01) {
  const std::string hlo_string = R"(
HloModule R2Window
mul {
lhs = f32[] parameter(0)
rhs = f32[] parameter(1)
ROOT mul = f32[] multiply(lhs, rhs)
}
ENTRY R2Window {
operand = f32[256,384]{0,1} parameter(0)
constant = f32[] constant(1)
ROOT reduce-window = f32[256,384]{0,1} reduce-window(operand, constant), window={size=2x3 pad=0_1x1_1}, to_apply=mul
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{0.001}));
}

XLA_TEST_F(ReduceWindowTextTest, R2General2x5) {
  const std::string hlo_string = R"(
HloModule R2Window
mul {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT mul = f32[] multiply(lhs, rhs)
}
ENTRY R2Window {
  operand = f32[2,5]{1,0} parameter(0)
  constant = f32[] constant(1)
  ROOT reduce-window = f32[3,5]{1,0} reduce-window(operand, constant), window={size=2x1 pad=0_2x0_0}, to_apply=mul
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{0.001}));
}

XLA_TEST_F(ReduceWindowTextTest, R2EffectiveScalar) {
  const std::string hlo_string = R"(
HloModule R2Window
mul {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT mul = f32[] multiply(lhs, rhs)
}
ENTRY R2Window {
  operand = f32[1,1]{1,0} parameter(0)
  negate = f32[1,1]{1,0} negate(operand)
  constant = f32[] constant(1)
  ROOT reduce-window = f32[1,1]{1,0} reduce-window(negate, constant), window={size=1x1 pad=0_0x0_0}, to_apply=mul
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{0.001}));
}

XLA_TEST_F(ReduceWindowTextTest, R3EffectiveScalar) {
  const std::string hlo_string = R"(
HloModule R3Window
mul {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT mul = f32[] multiply(lhs, rhs)
}
ENTRY R3Window {
  operand = f32[1,1,1]{2,1,0} parameter(0)
  negate = f32[1,1,1]{2,1,0} negate(operand)
  constant = f32[] constant(1)
  ROOT reduce-window = f32[1,1,1]{2,1,0}
    reduce-window(negate, constant),
    window={size=1x1x1 pad=0_0x0_0x0_0},
    to_apply=mul
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{0.001}));
}

XLA_TEST_F(HloTestBase, ReduceWindowIdentity) {
  const std::string hlo_string = R"(
HloModule ReduceWindowIdentity
identity.pad_to_reduce_window {
  param0 = f32[] parameter(0)
  ROOT param1 = f32[] parameter(1)
}
ENTRY reduce-window-identity {
  operand = f32[1,32,64]{2,1,0} parameter(0)
  constant.4466 = f32[] constant(0)
  ROOT reduce-window = f32[1,33,64]{2,1,0}
    reduce-window(operand, constant.4466),
      window={size=1x1x1 pad=0_0x1_0x0_0},
      to_apply=identity.pad_to_reduce_window
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_string, std::nullopt));
}

XLA_TEST_F(HloTestBase, ReduceWindowIdentityNoPadding) {
  const std::string hlo_string = R"(
HloModule ReduceWindowIdentity
identity.pad_to_reduce_window {
  param0 = f32[] parameter(0)
  ROOT param1 = f32[] parameter(1)
}
ENTRY reduce-window-identity {
  operand = f32[1,32,64]{2,1,0} parameter(0)
  constant.4466 = f32[] constant(0)
  ROOT reduce-window = f32[1,32,64]{2,1,0}
    reduce-window(operand, constant.4466),
      window={size=1x1x1 pad=0_0x0_0x0_0},
      to_apply=identity.pad_to_reduce_window
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_string, std::nullopt));
}

XLA_TEST_F(HloTestBase, ReduceWindowS32) {
  const std::string hlo_string = R"(
HloModule reduce-window

%identity.pad_to_reduce_window (param0: s32[], param1: s32[]) -> s32[] {
  %param0 = s32[] parameter(0)
  ROOT %param1 = s32[] parameter(1)
}

ENTRY %reduce-window (parameter.0: s32[81,8], parameter.1: s32[]) -> s32[82,8] {
  %parameter.0 = s32[81,8]{1,0} parameter(0)
  %parameter.1 = s32[] parameter(1)
  ROOT %reduce-window = s32[82,8]{1,0} reduce-window(s32[81,8]{1,0} %parameter.0, s32[] %parameter.1), window={size=1x1 pad=0_1x0_0}, to_apply=%identity.pad_to_reduce_window
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_string, std::nullopt));
}

XLA_TEST_F(HloTestBase, ReduceWindowS64) {
  const std::string hlo_string = R"(
HloModule reduce-window

%identity.pad_to_reduce_window (param0: s64[], param1: s64[]) -> s64[] {
  %param0 = s64[] parameter(0)
  ROOT %param1 = s64[] parameter(1)
}

ENTRY %reduce-window (parameter.0: s64[81,8], parameter.1: s64[]) -> s64[82,8] {
  %parameter.0 = s64[81,8]{1,0} parameter(0)
  %parameter.1 = s64[] parameter(1)
  ROOT %reduce-window = s64[82,8]{1,0} reduce-window(s64[81,8]{1,0} %parameter.0, s64[] %parameter.1), window={size=1x1 pad=0_1x0_0}, to_apply=%identity.pad_to_reduce_window
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_string, std::nullopt));
}

XLA_TEST_F(HloTestBase, ReduceWindowF16) {
  const std::string hlo_string = R"(
HloModule reduce-window

%identity.pad_to_reduce_window (param0: f16[], param1: f16[]) -> f16[] {
  %param0 = f16[] parameter(0)
  ROOT %param1 = f16[] parameter(1)
}

ENTRY %reduce-window (parameter.0: f16[81,8], parameter.1: f16[]) -> f16[82,8] {
  %parameter.0 = f16[81,8]{1,0} parameter(0)
  %parameter.1 = f16[] parameter(1)
  ROOT %reduce-window = f16[82,8]{1,0} reduce-window(f16[81,8]{1,0} %parameter.0, f16[] %parameter.1), window={size=1x1 pad=0_1x0_0}, to_apply=%identity.pad_to_reduce_window
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_string, std::nullopt));
}

XLA_TEST_F(ReduceWindowTextTest, R4OnlyDilation) {
  const std::string hlo_string = R"(
HloModule R4OnlyDilation
mul {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT mul = f32[] multiply(lhs, rhs)
}
ENTRY R4OnlyDilation {
  operand = f32[2,2,2,2]{3,2,1,0} parameter(0)
  constant = f32[] constant(1)
  ROOT reduce-window = f32[3,3,3,3]{3,2,1,0}
    reduce-window(operand, constant),
    window={size=1x1x1x1 pad=0_0x0_0x0_0x0_0 lhs_dilate=2x2x2x2},
    to_apply=mul
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{0.001}));
}

XLA_TEST_F(HloTestBase, DISABLED_ON_GPU(ReduceWindowVariadicSupport)) {
  const char* const hlo_string = R"(
HloModule module

sum {
  a0 = f32[] parameter(0)
  a1 = f32[] parameter(1) 
  b0 = f32[] parameter(2)
  b1 = f32[] parameter(3)
  add0 = f32[] add(a0, b0)
  add1 = f32[] add(a1, b1)
  ROOT sum2 = (f32[], f32[]) tuple(add0, add1)
}

ENTRY entry {
  constant = f32[4,2]{1,0} constant({{1,1},{1,4},{2,1},{3,1}})
  constant.1 = f32[] constant(0)
  constant.2 = f32[4,2]{1,0} constant({{1,1},{1,4},{2,1},{3,1}})
  constant.3 = f32[] constant(0)
  reduce-window = (f32[2,2]{1,0}, f32[2,2]{1,0}) 
    reduce-window(constant, constant.2, constant.1, constant.3),
    window={size=5x1 stride=3x1 pad=2_2x0_0}, to_apply=sum
  ROOT copy = (f32[2,2]{1,0}, f32[2,2]{1,0}) copy(reduce-window)
})";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-4, 1e-4}));
}

XLA_TEST_F(HloTestBase, DISABLED_ON_GPU(ReduceWindowVariadicSupport2)) {
  const char* const hlo_string = R"(
HloModule module

sum {
  a0 = f32[] parameter(0)
  a1 = s32[] parameter(1) 
  b0 = f32[] parameter(2)
  b1 = s32[] parameter(3)
  add0 = f32[] add(a0, b0)
  add1 = s32[] add(a1, b1)
  ROOT sum2 = (f32[], s32[]) tuple(add0, add1)
}

ENTRY entry {
  constant = f32[4,2]{1,0} constant({{1,1},{1,4},{2,1},{3,1}})
  constant.1 = f32[] constant(0)
  constant.2 = s32[4,2]{1,0} constant({{1,1},{1,4},{2,1},{3,1}})
  constant.3 = s32[] constant(0)
  ROOT reduce-window = (f32[2,2]{1,0}, s32[2,2]{1,0}) 
    reduce-window(constant, constant.2, constant.1, constant.3),
    window={size=5x1 stride=3x1 pad=2_2x0_0}, to_apply=sum
})";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-4, 1e-4}));
}

XLA_TEST_F(HloTestBase, DISABLED_ON_GPU(ReduceWindowVariadicSupport3)) {
  const char* const hlo_string = R"(
HloModule module

sum {
  a0 = f32[] parameter(0)
  a1 = bf16[] parameter(1) 
  b0 = f32[] parameter(2)
  b1 = bf16[] parameter(3)
  add0 = f32[] add(a0, b0)
  add1 = bf16[] add(a1, b1)
  ROOT sum2 = (f32[], bf16[]) tuple(add0, add1)
}

ENTRY entry {
  constant = f32[4,2]{1,0} constant({{1,1},{1,4},{2,1},{3,1}})
  constant.1 = f32[] constant(0)
  constant.2 = bf16[4,2]{1,0} constant({{1,1},{1,4},{2,1},{3,1}})
  constant.3 = bf16[] constant(0)
  ROOT reduce-window = (f32[2,2]{1,0}, bf16[2,2]{1,0}) 
    reduce-window(constant, constant.2, constant.1, constant.3),
    window={size=5x1 stride=3x1 pad=2_2x0_0}, to_apply=sum
})";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-4, 1e-4}));
}

XLA_TEST_F(HloTestBase, DISABLED_ON_GPU(ReduceWindowVariadicSupport4)) {
  const char* const hlo_string = R"(
HloModule module

sum {
  a0 = f32[] parameter(0)
  a1 = bf16[] parameter(1) 
  b0 = f32[] parameter(2)
  b1 = bf16[] parameter(3)
  add0 = f32[] add(a0, b0)
  add1 = bf16[] multiply(a1, b1)
  ROOT sum2 = (f32[], bf16[]) tuple(add0, add1)
}

ENTRY entry {
  constant = f32[4,2]{1,0} constant({{1,1},{1,4},{2,1},{3,1}})
  constant.1 = f32[] constant(0)
  constant.2 = bf16[4,2]{1,0} constant({{1,1},{1,4},{2,1},{3,1}})
  constant.3 = bf16[] constant(1)
  ROOT reduce-window = (f32[2,2]{1,0}, bf16[2,2]{1,0}) 
    reduce-window(constant, constant.2, constant.1, constant.3),
    window={size=5x1 stride=3x1 pad=2_2x0_0}, to_apply=sum
})";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-4, 1e-4}));
}

XLA_TEST_F(HloTestBase, DISABLED_ON_GPU(ReduceWindowS64Support)) {
  const char* const hlo_string = R"(
HloModule jit_dilated_window_sum.10

%primitive_computation_add.4 (parameter.5: s64[], parameter.6: s64[]) -> s64[] {
  %parameter.5 = s64[] parameter(0), metadata={op_type="add" op_name="add"}
  %parameter.6 = s64[] parameter(1), metadata={op_type="add" op_name="add"}
  ROOT %add.7 = s64[] add(s64[] %parameter.5, s64[] %parameter.6), metadata={op_type="add" op_name="add"}
}

ENTRY %jit_dilated_window_sum.10 (parameter.1: s64[8,10,12]) -> (s64[8,10,12]) {
  %constant.2 = pred[] constant(false)
  %parameter.1 = s64[8,10,12]{2,1,0} parameter(0)
  %constant.3 = s64[] constant(0), metadata={op_type="reduce_window_sum" op_name="jit(dilated_window_sum)/reduce_window_sum[ base_dilation=(1, 1, 1)\n                                           padding=((1, 1), (1, 1), (0, 0))\n                                           window_dilation=(1, 2, 2)\n                                           window_dimensions=(3, 2, 1)\n                                           window_strides=(1, 1, 1) ]" source_file="<ipython-input-1-315291761729>" source_line=9}
  %reduce-window.8 = s64[8,10,12]{2,1,0} reduce-window(s64[8,10,12]{2,1,0} %parameter.1, s64[] %constant.3), window={size=3x2x1 pad=1_1x1_1x0_0 rhs_dilate=1x2x2}, to_apply=%primitive_computation_add.4, metadata={op_type="reduce_window_sum" op_name="jit(dilated_window_sum)/reduce_window_sum[ base_dilation=(1, 1, 1)\n                                           padding=((1, 1), (1, 1), (0, 0))\n                                           window_dilation=(1, 2, 2)\n                                           window_dimensions=(3, 2, 1)\n                                           window_strides=(1, 1, 1) ]" source_file="<ipython-input-1-315291761729>" source_line=9}
  ROOT %tuple.9 = (s64[8,10,12]{2,1,0}) tuple(s64[8,10,12]{2,1,0} %reduce-window.8)
})";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-4, 1e-4}));
}

XLA_TEST_F(HloTestBase, DISABLED_ON_GPU(ReduceWindowS64Support2)) {
  const char* const hlo_string = R"(
HloModule SyncTensorsGraph.43

%MulComputation.18 (x.19: s64[], y.20: s64[]) -> s64[] {
  %x.19 = s64[] parameter(0)
  %y.20 = s64[] parameter(1)
  ROOT %multiply.21 = s64[] multiply(s64[] %x.19, s64[] %y.20)
}

%MulComputation.26 (x.27: s64[], y.28: s64[]) -> s64[] {
  %x.27 = s64[] parameter(0)
  %y.28 = s64[] parameter(1)
  ROOT %multiply.29 = s64[] multiply(s64[] %x.27, s64[] %y.28)
}

%MaxComputation.34 (x.35: s64[], y.36: s64[]) -> s64[] {
  %x.35 = s64[] parameter(0)
  %y.36 = s64[] parameter(1)
  ROOT %maximum.37 = s64[] maximum(s64[] %x.35, s64[] %y.36)
}

ENTRY %SyncTensorsGraph.43 (p0.1: f32[], p1.7: pred[3,3]) -> (pred[]) {
  %constant.8 = pred[] constant(false)
  %reshape.9 = pred[1,1]{1,0} reshape(pred[] %constant.8)
  %broadcast.10 = pred[1,1]{1,0} broadcast(pred[1,1]{1,0} %reshape.9), dimensions={0,1}
  %reshape.11 = pred[] reshape(pred[1,1]{1,0} %broadcast.10)
  %broadcast.12 = pred[3,3]{1,0} broadcast(pred[] %reshape.11), dimensions={}
  %p1.7 = pred[3,3]{1,0} parameter(1)
  %reshape.13 = pred[3,3]{1,0} reshape(pred[3,3]{1,0} %p1.7)
  %reshape.14 = pred[3,3]{1,0} reshape(pred[3,3]{1,0} %reshape.13)
  %convert.24 = s64[3,3]{1,0} convert(pred[3,3]{1,0} %reshape.14)
  %constant.25 = s64[] constant(1)
  %reduce-window.30 = s64[3,3]{1,0} reduce-window(s64[3,3]{1,0} %convert.24, s64[] %constant.25), window={size=3x1 pad=2_0x0_0}, to_apply=%MulComputation.26
  %convert.15 = s32[3,3]{1,0} convert(pred[3,3]{1,0} %reshape.14)
  %convert.16 = s64[3,3]{1,0} convert(s32[3,3]{1,0} %convert.15)
  %constant.17 = s64[] constant(1)
  %reduce-window.22 = s64[3,3]{1,0} reduce-window(s64[3,3]{1,0} %convert.16, s64[] %constant.17), window={size=3x1 pad=2_0x0_0}, to_apply=%MulComputation.18
  %constant.2 = s64[] constant(1)
  %reshape.3 = s64[1,1]{1,0} reshape(s64[] %constant.2)
  %broadcast.4 = s64[1,1]{1,0} broadcast(s64[1,1]{1,0} %reshape.3), dimensions={0,1}
  %reshape.5 = s64[] reshape(s64[1,1]{1,0} %broadcast.4)
  %broadcast.6 = s64[3,3]{1,0} broadcast(s64[] %reshape.5), dimensions={}
  %multiply.23 = s64[3,3]{1,0} multiply(s64[3,3]{1,0} %reduce-window.22, s64[3,3]{1,0} %broadcast.6)
  %subtract.31 = s64[3,3]{1,0} subtract(s64[3,3]{1,0} %reduce-window.30, s64[3,3]{1,0} %multiply.23)
  %abs.32 = s64[3,3]{1,0} abs(s64[3,3]{1,0} %subtract.31)
  %constant.33 = s64[] constant(-9223372036854775808)
  %reduce.38 = s64[] reduce(s64[3,3]{1,0} %abs.32, s64[] %constant.33), dimensions={0,1}, to_apply=%MaxComputation.34
  %reshape.39 = s64[] reshape(s64[] %reduce.38)
  %convert.40 = f32[] convert(s64[] %reshape.39)
  %p0.1 = f32[] parameter(0)
  %compare.41 = pred[] compare(f32[] %convert.40, f32[] %p0.1), direction=LE
  ROOT %tuple.42 = (pred[]) tuple(pred[] %compare.41)
})";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-4, 1e-4}));
}

}  // namespace
}  // namespace xla
