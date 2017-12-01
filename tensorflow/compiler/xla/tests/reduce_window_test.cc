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
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
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

TEST_F(ReduceWindowTest, MismatchedRanksGivesErrorStatus) {
  const auto input = builder_.ConstantR1<float>({1, 1, 1, 1});
  const auto init_value = builder_.ConstantR0<float>(0);
  TF_ASSERT_OK(builder_.first_error());
  builder_.ReduceWindow(input, init_value,
                        CreateScalarAddComputation(F32, &builder_),
                        /*window_dimensions=*/{1, 2},
                        /*window_strides=*/{1}, Padding::kValid);
  ASSERT_EQ(builder_.first_error().code(), tensorflow::error::INVALID_ARGUMENT)
      << builder_.first_error();
  ASSERT_THAT(builder_.first_error().error_message(),
              ::testing::HasSubstr("Want input dimensions size"));
}

TEST_F(ReduceWindowTest, Min3In5Stride2) {
  const auto input = builder_.ConstantR1<float>({10000, 1000, 100, 10, 1});
  ReduceWindowMin(input, {3}, {2}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {100, 1}, {}, ErrorSpec(0.0001));
}

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

TEST_F(ReduceWindowTest, R4UnitWindow) {
  Array4D<float> input_array(13, 12, 8, 15);
  input_array.Fill(1.0f);
  std::unique_ptr<Literal> input_literal =
      Literal::CreateR4FromArray4DWithLayout(
          input_array, LayoutUtil::MakeLayout({0, 3, 2, 1}));
  ComputationDataHandle input =
      builder_.Parameter(0, input_literal->shape(), "operand");

  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, 7, 1}, {1, 4, 1, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(input_array, 0.0f, {1, 1, 7, 1},
                                              {1, 4, 1, 1}, padding);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client_->TransferToServer(*input_literal));
  ComputeAndCompareR4<float>(&builder_, *res, {input_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(HloTestBase, R6AddMultipleStrides) {
  auto b = HloComputation::Builder(TestName());

  std::vector<int64> input_dims(6, 8);
  auto shape = ShapeUtil::MakeShape(F32, input_dims);

  std::unique_ptr<Literal> arg_literal = Literal::CreateFromShape(shape);
  auto generator = [&](tensorflow::gtl::ArraySlice<int64> indexes) -> float {
    return 1.0f;
  };
  TF_EXPECT_OK(arg_literal->Populate<float>(generator));

  auto input =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));

  auto module = CreateNewModule();
  auto add_func = module->AddEmbeddedComputation(add_computation.Build());

  WindowDimension trivial_dim;
  trivial_dim.set_size(1);
  trivial_dim.set_stride(1);
  trivial_dim.set_padding_low(0);
  trivial_dim.set_padding_high(0);
  trivial_dim.set_window_dilation(1);
  trivial_dim.set_base_dilation(1);

  WindowDimension active_dim;
  active_dim.set_size(3);
  active_dim.set_stride(1);
  active_dim.set_padding_low(0);
  active_dim.set_padding_high(0);
  active_dim.set_window_dilation(1);
  active_dim.set_base_dilation(1);

  Window window;
  *window.add_dimensions() = active_dim;
  *window.add_dimensions() = trivial_dim;
  *window.add_dimensions() = active_dim;
  *window.add_dimensions() = active_dim;
  *window.add_dimensions() = trivial_dim;
  *window.add_dimensions() = trivial_dim;

  // Non-monotonic output layout with minor dims trivial.
  std::vector<int64> output_layout = {1, 5, 3, 2, 0, 4};
  std::vector<int64> output_dims = {6, 8, 6, 6, 8, 8};
  Shape result_shape =
      ShapeUtil::MakeShapeWithLayout(F32, output_dims, output_layout);
  b.AddInstruction(HloInstruction::CreateReduceWindow(
      result_shape, input, init_value, window, add_func));

  std::unique_ptr<Literal> expected = Literal::CreateFromShape(result_shape);
  auto out_generator =
      [&](tensorflow::gtl::ArraySlice<int64> indexes) -> float {
    return 27.0f;
  };
  TF_EXPECT_OK(expected->Populate<float>(out_generator));

  module->AddEntryComputation(b.Build());
  auto actual = ExecuteAndTransfer(std::move(module), {});

  LiteralTestUtil::ExpectNear(*actual, *expected, ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(HloTestBase, R6Add) {
  auto b = HloComputation::Builder(TestName());

  std::vector<int64> input_dims(6, 8);
  std::unique_ptr<Literal> arg_literal =
      Literal::CreateFullWithMonotonicDim0MajorLayout<float>(input_dims, 1.0f);
  auto input =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));

  auto module = CreateNewModule();
  auto add_func = module->AddEmbeddedComputation(add_computation.Build());

  WindowDimension trivial_dim;
  trivial_dim.set_size(1);
  trivial_dim.set_stride(1);
  trivial_dim.set_padding_low(0);
  trivial_dim.set_padding_high(0);
  trivial_dim.set_window_dilation(1);
  trivial_dim.set_base_dilation(1);

  WindowDimension active_dim;
  active_dim.set_size(3);
  active_dim.set_stride(1);
  active_dim.set_padding_low(0);
  active_dim.set_padding_high(0);
  active_dim.set_window_dilation(1);
  active_dim.set_base_dilation(1);

  Window window;
  *window.add_dimensions() = trivial_dim;
  *window.add_dimensions() = trivial_dim;
  *window.add_dimensions() = active_dim;
  *window.add_dimensions() = active_dim;
  *window.add_dimensions() = trivial_dim;
  *window.add_dimensions() = trivial_dim;

  Shape shape = ShapeUtil::MakeShape(F32, {8, 8, 6, 6, 8, 8});
  b.AddInstruction(HloInstruction::CreateReduceWindow(shape, input, init_value,
                                                      window, add_func));

  std::vector<int64> output_dims = {8, 8, 6, 6, 8, 8};
  std::unique_ptr<Literal> expected =
      Literal::CreateFullWithMonotonicDim0MajorLayout<float>(output_dims, 9.0f);

  module->AddEntryComputation(b.Build());
  auto actual = ExecuteAndTransfer(std::move(module), {});

  LiteralTestUtil::ExpectNear(*actual, *expected, ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(ReduceWindowTest, R4SecondMinorStride) {
  Array4D<float> input_array(2, 1, 27, 119);
  input_array.FillRandom(2.0f);
  std::unique_ptr<Literal> input_literal =
      Literal::CreateR4FromArray4DWithLayout(
          input_array, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  ComputationDataHandle input =
      builder_.Parameter(0, input_literal->shape(), "operand");

  int win_len = 1;
  int stride = 8;
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, win_len, 1}, {1, 1, stride, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {1, 1, win_len, 1}, {1, 1, stride, 1}, padding);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client_->TransferToServer(*input_literal));
  ComputeAndCompareR4<float>(&builder_, *res, {input_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(ReduceWindowTest, R4SecondMinorUnitStride) {
  Array4D<float> input_array(3, 2, 4, 64);
  input_array.FillRandom(2.0f);
  std::unique_ptr<Literal> input_literal =
      Literal::CreateR4FromArray4DWithLayout(
          input_array, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  ComputationDataHandle input =
      builder_.Parameter(0, input_literal->shape(), "operand");

  int win_len = 3;
  int stride = 1;
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, win_len, 1}, {1, 1, stride, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {1, 1, win_len, 1}, {1, 1, stride, 1}, padding);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client_->TransferToServer(*input_literal));
  ComputeAndCompareR4<float>(&builder_, *res, {input_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(ReduceWindowTest, R4SecondMinorWin) {
  Array4D<float> input_array(1, 3, 12, 200);
  input_array.FillRandom(2.0f);
  std::unique_ptr<Literal> input_literal =
      Literal::CreateR4FromArray4DWithLayout(
          input_array, LayoutUtil::MakeLayout({3, 2, 1, 0}));
  ComputationDataHandle input =
      builder_.Parameter(0, input_literal->shape(), "operand");

  int win_len = 8;
  int stride = 5;
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {1, 1, win_len, 1}, {1, 1, stride, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4DAdd(
      input_array, 0.0f, {1, 1, win_len, 1}, {1, 1, stride, 1}, padding);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client_->TransferToServer(*input_literal));
  ComputeAndCompareR4<float>(&builder_, *res, {input_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

TEST_F(ReduceWindowTest, AmongMajor2DimsMultipleMinor) {
  Array4D<float> input_array(6, 4, 10, 130);
  input_array.FillRandom(2.0f);

  int win_len = 3;
  int win_stride = 2;

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

XLA_TEST_F(ReduceWindowTest, Add24In1152_NoOverlap) {
  std::vector<float> input_vector(128 * 9, 1);
  const auto input = builder_.ConstantR1<float>(input_vector);
  ReduceWindowAdd(input, {32}, {128}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {32, 32, 32, 32, 32, 32, 32, 32, 32},
                             {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ReduceWindowTest, Add128In128Stride128) {
  const auto input = builder_.ConstantR1<float>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ReduceWindowAdd(input, {128}, {128}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {1088}, {}, ErrorSpec(0.0001));
}

// Regression test for a bug that appeared in Inception (b/34784899).
TEST_F(ReduceWindowTest, R2ReduceWindowInceptionFromBroadcast) {
  Array2D<float> input_array(14, 14, 1.0f);
  ComputationDataHandle input =
      builder_.Broadcast(builder_.ConstantLiteral(Literal::One(F32)), {14, 14});

  int win_len = 3;
  int stride = 1;
  Padding padding = Padding::kSame;
  ReduceWindowAdd(input, {win_len, win_len}, {stride, stride}, padding);

  auto res = ReferenceUtil::ReduceWindow2DAdd(
      input_array, 0.0f, {win_len, win_len}, {stride, stride}, padding);

  ComputeAndCompareR2<float>(&builder_, *res, {}, ErrorSpec(1e-3, 1e-3));
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
      public ::testing::WithParamInterface<R4ReduceWindowTestData> {
 protected:
  void DoIt() {
    ComputationBuilder b(client_, TestName());
    const auto& param = GetParam();

    const float kInitValue = 0.0f;

    Array4D<float> input(param.base_bounds[0], param.base_bounds[1],
                         param.base_bounds[2], param.base_bounds[3]);
    input.FillIota(1);
    std::unique_ptr<Literal> input_literal =
        Literal::CreateR4FromArray4D(input);
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_arg,
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
    ComputeAndCompareR4<float>(&b, *expected, {input_arg.get()},
                               ErrorSpec(1e-3, 1e-3));
  }
};

TEST_P(R4ReduceWindowTest, DoIt) { DoIt(); }

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

    // With second minor dimension == 9.
    R4ReduceWindowTestData{/*base_bounds=*/{2, 3, 9, 127},
                           /*window_bounds=*/{1, 1, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*reducer=*/kAdd},

    // With minor dimension == 129.
    R4ReduceWindowTestData{/*base_bounds=*/{3, 2, 7, 129},
                           /*window_bounds=*/{1, 1, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*reducer=*/kAdd},

    // With minor dims reduction and non-overlapped stride.
    R4ReduceWindowTestData{/*base_bounds=*/{2, 2, 4, 16},
                           /*window_bounds=*/{1, 1, 2, 2},
                           /*strides=*/{1, 1, 2, 2},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*reducer=*/kAdd},

    // With minor dims reduction and overlapped stride.
    R4ReduceWindowTestData{/*base_bounds=*/{2, 2, 4, 16},
                           /*window_bounds=*/{1, 1, 4, 4},
                           /*strides=*/{1, 1, 2, 2},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{0, 0, 0, 0},
                           /*reducer=*/kAdd},
};

INSTANTIATE_TEST_CASE_P(R4ReduceWindowTestInstantiation, R4ReduceWindowTest,
                        ::testing::ValuesIn(kR4ReduceWindowTestValues),
                        R4ReduceWindowTestDataToString);

class R4ReduceWindowLargeTest : public R4ReduceWindowTest {};

XLA_TEST_P(R4ReduceWindowLargeTest, DoIt) { DoIt(); }

// Test cases that are large/slow/failed.
const R4ReduceWindowTestData kR4ReduceWindowLargeTestValues[] = {
    R4ReduceWindowTestData{/*base_bounds=*/{28, 28, 256, 128},
                           /*window_bounds=*/{3, 3, 1, 1},
                           /*strides=*/{1, 1, 1, 1},
                           /*pad_low=*/{1, 1, 0, 0},
                           /*pad_high=*/{1, 1, 0, 0},
                           /*reducer=*/kMax},

    R4ReduceWindowTestData{/*base_bounds=*/{112, 112, 64, 128},
                           /*window_bounds=*/{3, 3, 1, 1},
                           /*strides=*/{2, 2, 1, 1},
                           /*pad_low=*/{0, 0, 0, 0},
                           /*pad_high=*/{1, 1, 0, 0},
                           /*reducer=*/kAdd},
};

INSTANTIATE_TEST_CASE_P(R4ReduceWindowLargeTestInstantiation,
                        R4ReduceWindowLargeTest,
                        ::testing::ValuesIn(kR4ReduceWindowLargeTestValues),
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
     /*padding=*/Padding::kSame, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{2, 5}, /*window_bounds=*/{2, 4},
     /*strides=*/{1, 1}, /*layout=*/{0, 1},
     /*padding=*/Padding::kSame, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{1, 3}, /*window_bounds=*/{2, 3},
     /*strides=*/{1, 1}, /*layout=*/{0, 1},
     /*padding=*/Padding::kSame, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{3, 129}, /*window_bounds=*/{1, 100},
     /*strides=*/{2, 99}, /*layout=*/{0, 1},
     /*padding=*/Padding::kSame, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{6, 152}, /*window_bounds=*/{2, 25},
     /*strides=*/{5, 4}, /*layout=*/{0, 1},
     /*padding=*/Padding::kSame, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{6, 4}, /*window_bounds=*/{4, 2},
     /*strides=*/{3, 3}, /*layout=*/{0, 1},
     /*padding=*/Padding::kSame, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{5, 147}, /*window_bounds=*/{1, 36},
     /*strides=*/{4, 5}, /*layout=*/{1, 0},
     /*padding=*/Padding::kSame, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{4, 153}, /*window_bounds=*/{2, 93},
     /*strides=*/{1, 1}, /*layout=*/{1, 0},
     /*padding=*/Padding::kSame, /*reducer=*/Reducer::kAdd},
    // Regression test for a bug that appeared in Inception (b/34784899).
    {/*base_bounds=*/{28, 28}, /*window_bounds=*/{3, 3},
     /*strides=*/{1, 1}, /*layout=*/{1, 0},
     /*padding=*/Padding::kSame, /*reducer=*/Reducer::kAdd},
    // Regression test for a bug that appeared in Inception (b/34784899).
    {/*base_bounds=*/{4, 32}, /*window_bounds=*/{2, 2},
     /*strides=*/{2, 2}, /*layout=*/{1, 0},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},
    {/*base_bounds=*/{4, 4}, /*window_bounds=*/{2, 2},
     /*strides=*/{1, 1}, /*layout=*/{1, 0},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_arg,
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

struct R1ReduceWindowTestData {
  int64 base_bounds[1];
  int64 window_bounds[1];
  int64 strides[1];
  Padding padding;
  Reducer reducer;
} kR1TestCases[] = {
    {/*base_bounds=*/{1}, /*window_bounds=*/{1},
     /*strides=*/{1},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{3}, /*window_bounds=*/{3},
     /*strides=*/{1},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{3}, /*window_bounds=*/{2},
     /*strides=*/{1},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{5}, /*window_bounds=*/{1},
     /*strides=*/{1},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kMax},

    {/*base_bounds=*/{16}, /*window_bounds=*/{4},
     /*strides=*/{4},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kMax},

    {/*base_bounds=*/{16}, /*window_bounds=*/{4},
     /*strides=*/{3},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{128 * 2}, /*window_bounds=*/{30},
     /*strides=*/{27},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{128 * 17}, /*window_bounds=*/{7},
     /*strides=*/{64},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{128 * 2}, /*window_bounds=*/{32},
     /*strides=*/{56},
     /*padding=*/Padding::kValid, /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{3}, /*window_bounds=*/{2},
     /*strides=*/{1},
     /*padding=*/Padding::kSame, /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{5}, /*window_bounds=*/{3},
     /*strides=*/{2},
     /*padding=*/Padding::kSame, /*reducer=*/Reducer::kAdd},

    {/*base_bounds=*/{16}, /*window_bounds=*/{4},
     /*strides=*/{3},
     /*padding=*/Padding::kSame, /*reducer=*/Reducer::kAdd},
};

string R1ReduceWindowTestDataToString(
    const ::testing::TestParamInfo<R1ReduceWindowTestData>& data) {
  string str = tensorflow::strings::StrCat(
      "base_bounds_",
      tensorflow::str_util::Join(data.param.base_bounds, "x"),  //
      "__window_bounds_",
      tensorflow::str_util::Join(data.param.window_bounds, "x"),              //
      "__strides_", tensorflow::str_util::Join(data.param.strides, "x"),      //
      "__padding_", data.param.padding == Padding::kSame ? "same" : "valid",  //
      "__reducer_", data.param.reducer == kAdd ? "add" : "max");
  return str;
}

class R1ReduceWindowTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<R1ReduceWindowTestData> {};

TEST_P(R1ReduceWindowTest, DoIt) {
  ComputationBuilder b(client_, TestName());
  const auto& param = GetParam();
  CHECK(param.reducer == kAdd || param.reducer == kMax);

  const float kInitValue = 0.0f;
  std::vector<float> input_vector(param.base_bounds[0]);
  std::iota(std::begin(input_vector), std::end(input_vector), 0);
  std::unique_ptr<Literal> input_literal =
      Literal::CreateR1(tensorflow::gtl::ArraySlice<float>(input_vector));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_arg,
                          client_->TransferToServer(*input_literal));

  auto computation = param.reducer == kAdd
                         ? CreateScalarAddComputation(F32, &b)
                         : CreateScalarMaxComputation(F32, &b);
  b.ReduceWindow(/*operand=*/
                 b.Parameter(0, input_literal->shape(), "p0"),
                 /*init_value=*/b.ConstantR0<float>(kInitValue),
                 /*computation=*/computation,
                 /*window_dimensions=*/param.window_bounds,
                 /*window_strides=*/param.strides, /*padding=*/param.padding);

  auto reduce_func = param.reducer == kAdd
                         ? +[](float a, float b) { return a + b; }
                         : +[](float a, float b) { return std::max(a, b); };
  auto expected = ReferenceUtil::ReduceWindow1DGeneric(
      /*operand=*/tensorflow::gtl::ArraySlice<float>(input_vector),
      /*init=*/kInitValue,
      /*reduce_func=*/reduce_func,
      /*window=*/param.window_bounds,
      /*stride=*/param.strides, /*padding=*/param.padding);

  ComputeAndCompareR1<float>(&b, tensorflow::gtl::ArraySlice<float>(*expected),
                             {input_arg.get()}, ErrorSpec(1e-3, 1e-3));
}

INSTANTIATE_TEST_CASE_P(R1ReduceWindowTestInstantiation, R1ReduceWindowTest,
                        ::testing::ValuesIn(kR1TestCases),
                        R1ReduceWindowTestDataToString);
}  // namespace
}  // namespace xla
