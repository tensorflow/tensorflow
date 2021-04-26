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

#include <memory>
#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

class BroadcastSimpleTest : public ClientLibraryTestBase {
 public:
  XlaOp BuildBinOp(HloOpcode op, const XlaOp lhs, const XlaOp rhs,
                   XlaBuilder* builder) {
    switch (op) {
      case HloOpcode::kMinimum: {
        return Min(lhs, rhs);
      }
      case HloOpcode::kMaximum: {
        return Max(lhs, rhs);
      }
      case HloOpcode::kMultiply: {
        return Mul(lhs, rhs);
      }
      default: {
        // Default to Add
        return Add(lhs, rhs);
      }
    }
  }

  std::unique_ptr<GlobalData> MakeR3Data(absl::Span<const int64> bounds,
                                         absl::Span<const int64> minor_to_major,
                                         Shape* r3_shape,
                                         Array3D<float>* r3_array, float start,
                                         float end, int seed) {
    *r3_shape = ShapeUtil::MakeShapeWithLayout(F32, bounds, minor_to_major);
    r3_array->FillRandom(start, end, seed);
    auto r3_data = LiteralUtil::CreateR3FromArray3D(*r3_array).Relayout(
        LayoutUtil::MakeLayout(minor_to_major));
    std::unique_ptr<GlobalData> r3_global_data =
        client_->TransferToServer(r3_data).ConsumeValueOrDie();
    return r3_global_data;
  }

  std::unique_ptr<GlobalData> MakeR2Data(absl::Span<const int64> bounds,
                                         absl::Span<const int64> minor_to_major,
                                         Shape* r2_shape,
                                         Array2D<float>* r2_array, float start,
                                         float end, int seed) {
    *r2_shape = ShapeUtil::MakeShapeWithLayout(F32, bounds, minor_to_major);
    r2_array->FillRandom(start, end, seed);
    auto r2_data = LiteralUtil::CreateR2FromArray2D(*r2_array).Relayout(
        LayoutUtil::MakeLayout(minor_to_major));
    std::unique_ptr<GlobalData> r2_global_data =
        client_->TransferToServer(r2_data).ConsumeValueOrDie();
    return r2_global_data;
  }

  float ApplyOpToFloats(HloOpcode op, float lhs, float rhs) {
    switch (op) {
      case HloOpcode::kMinimum: {
        return std::min(lhs, rhs);
      }
      case HloOpcode::kMaximum: {
        return std::max(lhs, rhs);
      }
      case HloOpcode::kMultiply: {
        return lhs * rhs;
      }
      case HloOpcode::kAdd: {
        return lhs + rhs;
      }
      default: {
        // Default to Add
        LOG(FATAL);
      }
    }
  }
};

using ::testing::HasSubstr;

XLA_TEST_F(BroadcastSimpleTest, ScalarNoOpBroadcast) {
  XlaBuilder b(TestName());
  Broadcast(ConstantR0<float>(&b, 1.5), {});
  ComputeAndCompareR0<float>(&b, 1.5, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, ScalarTo2D_2x3) {
  XlaBuilder b(TestName());
  Broadcast(ConstantR0<float>(&b, 2.25), {2, 3});
  Array2D<float> expected(2, 3, 2.25);
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, ScalarParamTo2D_2x3) {
  XlaBuilder b(TestName());
  XlaOp src;
  std::unique_ptr<GlobalData> param_data =
      CreateR0Parameter<float>(2.25f, /*parameter_number=*/0, /*name=*/"src",
                               /*builder=*/&b, /*data_handle=*/&src);

  Broadcast(src, {2, 3});
  Array2D<float> expected(2, 3, 2.25);
  ComputeAndCompareR2<float>(&b, expected, {param_data.get()},
                             ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, ScalarTo2D_2x0) {
  XlaBuilder b(TestName());
  Broadcast(ConstantR0<float>(&b, 2.25), {2, 0});
  Array2D<float> expected(2, 0);
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, ScalarTo2D_0x2) {
  XlaBuilder b(TestName());
  Broadcast(ConstantR0<float>(&b, 2.25), {0, 2});
  Array2D<float> expected(0, 2);
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 1DTo2D) {
  XlaBuilder b(TestName());
  Broadcast(ConstantR1<float>(&b, {1, 2, 3}), {2});

  Array2D<float> expected(2, 3);
  expected(0, 0) = 1;
  expected(0, 1) = 2;
  expected(0, 2) = 3;
  expected(1, 0) = 1;
  expected(1, 1) = 2;
  expected(1, 2) = 3;
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 1DTo2D_WithDimsUsual) {
  XlaBuilder b(TestName());
  BroadcastInDim(ConstantR1<float>(&b, {1, 2}), {2, 2}, {1});

  Array2D<float> expected(2, 2);
  expected(0, 0) = 1;
  expected(0, 1) = 2;
  expected(1, 0) = 1;
  expected(1, 1) = 2;

  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 1DTo2D_WithDimsTranspose) {
  XlaBuilder b(TestName());
  BroadcastInDim(ConstantR1<float>(&b, {1, 2}), {2, 2}, {0});

  Array2D<float> expected(2, 2);
  expected(0, 0) = 1;
  expected(0, 1) = 1;
  expected(1, 0) = 2;
  expected(1, 1) = 2;

  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 2DTo3D_WithDims) {
  XlaBuilder b(TestName());
  BroadcastInDim(ConstantR2<float>(&b, {{1.0, 5.0}, {2.0, 6.0}}), {2, 2, 2},
                 {0, 1});

  Array3D<float> expected(2, 2, 2);
  expected(0, 0, 0) = 1.0;
  expected(1, 0, 0) = 2.0;
  expected(0, 0, 1) = 1.0;
  expected(1, 0, 1) = 2.0;
  expected(0, 1, 0) = 5.0;
  expected(1, 1, 0) = 6.0;
  expected(1, 1, 1) = 6.0;
  expected(0, 1, 1) = 5.0;

  ComputeAndCompareR3<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 2DTo3D_WithDimsNotPossibleWithBroadCast) {
  XlaBuilder b(TestName());
  BroadcastInDim(ConstantR2<float>(&b, {{1.0, 5.0}, {2.0, 6.0}}), {2, 2, 2},
                 {0, 2});

  Array3D<float> expected(2, 2, 2);
  expected(0, 0, 0) = 1.0;
  expected(1, 0, 0) = 2.0;
  expected(0, 0, 1) = 5.0;
  expected(1, 0, 1) = 6.0;
  expected(0, 1, 0) = 1.0;
  expected(1, 1, 0) = 2.0;
  expected(1, 1, 1) = 6.0;
  expected(0, 1, 1) = 5.0;

  ComputeAndCompareR3<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 1DTo2D_WithDimsNotPossibleWithBroadCast) {
  XlaBuilder b(TestName());
  BroadcastInDim(ConstantR1<float>(&b, {1, 2}), {3, 2}, {1});

  Array2D<float> expected(3, 2);
  expected(0, 0) = 1;
  expected(0, 1) = 2;
  expected(1, 0) = 1;
  expected(1, 1) = 2;
  expected(2, 0) = 1;
  expected(2, 1) = 2;

  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

// Tests implicit broadcasting of PREDs.
XLA_TEST_F(BroadcastSimpleTest, BooleanAnd2DTo3D_Pred) {
  XlaBuilder b(TestName());

  Array2D<bool> x_vals(2, 1);
  x_vals(0, 0) = true;
  x_vals(1, 0) = false;
  Array3D<bool> y_vals(2, 2, 1);
  y_vals(0, 0, 0) = false;
  y_vals(0, 1, 0) = false;
  y_vals(1, 0, 0) = true;
  y_vals(1, 1, 0) = true;

  XlaOp x, y;
  auto x_data = CreateR2Parameter<bool>(x_vals, 0, "x", &b, &x);
  auto y_data = CreateR3Parameter<bool>(y_vals, 1, "y", &b, &y);
  And(x, y, /*broadcast_dimensions=*/{1, 2});

  Array3D<bool> expected(2, 2, 1);
  expected(0, 0, 0) = false;
  expected(0, 1, 0) = false;
  expected(1, 0, 0) = true;
  expected(1, 1, 0) = false;

  ComputeAndCompareR3<bool>(&b, expected, {x_data.get(), y_data.get()});
}

XLA_TEST_F(BroadcastSimpleTest, ZeroElement_1DTo2D) {
  XlaBuilder b(TestName());
  Broadcast(ConstantR1<float>(&b, {}), {2});

  Array2D<float> expected(2, 0);
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 1DToZeroElement2D) {
  XlaBuilder b(TestName());
  Broadcast(ConstantR1<float>(&b, {1, 2, 3}), {0});

  Array2D<float> expected(0, 3);
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, InDimensionAndDegenerateBroadcasting) {
  // Verify that binary op and degenerate dimension broadcast work together in
  // the same operation.
  //
  // The lhs shape [1, 2] is first broadcast up to [2, 1, 2] using in-dimension
  // broadcasting (broadcast_dimensions {1, 2}), then is added to the rhs shape
  // [2, 3, 1]. Degenerate dimension broadcasting then broadcasts the size one
  // dimensions.
  XlaBuilder b(TestName());

  Add(ConstantR2<float>(&b, {{1.0, 5.0}}),
      ConstantLiteral(&b, LiteralUtil::CreateR3<float>(
                              {{{2.0}, {3.0}, {4.0}}, {{5.0}, {6.0}, {7.0}}})),
      /*broadcast_dimensions=*/{1, 2});

  auto expected =
      LiteralUtil::CreateR3<float>({{{3.0, 7.0}, {4.0, 8.0}, {5.0, 9.0}},
                                    {{6.0, 10.0}, {7.0, 11.0}, {8.0, 12.0}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

struct R3ImplicitBroadcastSpec {
  std::array<int64, 3> output_bounds;
  std::array<int64, 3> minor2major_layout;
  std::array<int64, 3> input_bounds;
  HloOpcode op;
} kR3ImplicitBroadcastTestCases[] = {
    {{{1, 1, 1}}, {{2, 1, 0}}, {{1, 1, 1}}, HloOpcode::kAdd},
    {{{3, 4, 5}}, {{2, 1, 0}}, {{1, 1, 5}}, HloOpcode::kMaximum},
    {{{3, 4, 5}}, {{2, 1, 0}}, {{1, 4, 1}}, HloOpcode::kMinimum},
    {{{3, 4, 5}}, {{2, 1, 0}}, {{3, 1, 1}}, HloOpcode::kMultiply},
    {{{3, 4, 5}}, {{2, 1, 0}}, {{1, 1, 1}}, HloOpcode::kAdd},
    {{{3, 4, 5}}, {{2, 1, 0}}, {{1, 4, 5}}, HloOpcode::kAdd},
    {{{3, 4, 5}}, {{2, 1, 0}}, {{3, 4, 1}}, HloOpcode::kAdd},
    {{{3, 4, 5}}, {{2, 1, 0}}, {{3, 1, 5}}, HloOpcode::kAdd},
    {{{3, 199, 5}}, {{2, 1, 0}}, {{1, 199, 1}}, HloOpcode::kMinimum},
    {{{3, 4, 199}}, {{2, 1, 0}}, {{1, 1, 199}}, HloOpcode::kAdd},
};

class BroadcastR3ImplicitTest
    : public BroadcastSimpleTest,
      public ::testing::WithParamInterface<R3ImplicitBroadcastSpec> {};

XLA_TEST_P(BroadcastR3ImplicitTest, Doit) {
  const R3ImplicitBroadcastSpec& spec = GetParam();
  XlaBuilder builder(TestName());

  Shape r3_shape, r3_implicit_shape;
  Array3D<float> r3_array(spec.output_bounds[0], spec.output_bounds[1],
                          spec.output_bounds[2]);
  Array3D<float> r3_implicit_array(spec.input_bounds[0], spec.input_bounds[1],
                                   spec.input_bounds[2]);

  std::unique_ptr<GlobalData> r3_global_data =
      MakeR3Data(spec.output_bounds, spec.minor2major_layout, &r3_shape,
                 &r3_array, 1.0, 2.5, 56789);
  std::unique_ptr<GlobalData> r3_implicit_global_data =
      MakeR3Data(spec.input_bounds, spec.minor2major_layout, &r3_implicit_shape,
                 &r3_implicit_array, 1.0, 0.2, 56789);

  auto r3_implicit_parameter =
      Parameter(&builder, 0, r3_implicit_shape, "input");
  auto r3_parameter = Parameter(&builder, 1, r3_shape, "input");
  BuildBinOp(spec.op, r3_implicit_parameter, r3_parameter, &builder);

  Array3D<float> expected_array(spec.output_bounds[0], spec.output_bounds[1],
                                spec.output_bounds[2]);
  auto Each = ([&](absl::Span<const int64> indices, float* value) {
    float r3_implicit = r3_implicit_array(indices[0] % spec.input_bounds[0],
                                          indices[1] % spec.input_bounds[1],
                                          indices[2] % spec.input_bounds[2]);
    float r3 = r3_array(indices[0], indices[1], indices[2]);
    *value = ApplyOpToFloats(spec.op, r3_implicit, r3);
  });

  int n1 = expected_array.n1();
  int n2 = expected_array.n2();
  int n3 = expected_array.n3();
  for (int64 i = 0; i < n1; i++) {
    for (int64 j = 0; j < n2; j++) {
      for (int64 k = 0; k < n3; k++) {
        Each({i, j, k}, &expected_array(i, j, k));
      }
    }
  }
  auto expected = LiteralUtil::CreateR3FromArray3D(expected_array);
  ComputeAndCompareLiteral(
      &builder, expected, {r3_implicit_global_data.get(), r3_global_data.get()},
      ErrorSpec(1e-7, 1e-7));
}

INSTANTIATE_TEST_CASE_P(BroadcastR3ImplicitTestInstances,
                        BroadcastR3ImplicitTest,
                        ::testing::ValuesIn(kR3ImplicitBroadcastTestCases));

// r1 and r3's dim0 matches, and r1's dim1 and dim2 have size 1:
XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_1_2) {
  XlaBuilder b(TestName());
  XlaOp r1h;
  XlaOp r3h;

  Array3D<float> r1d = {{{1}}, {{2}}};
  Array3D<float> r3d = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
  auto r1 = CreateR3Parameter(r1d, 1, "r1", &b, &r1h);
  auto r3 = CreateR3Parameter(r3d, 0, "r3", &b, &r3h);

  Add(r3h, r1h);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 3}, {4, 5}}, {{7, 8}, {9, 10}}});

  ComputeAndCompareLiteral(&b, expected, {r3.get(), r1.get()},
                           ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_0_1) {
  XlaBuilder b(TestName());
  auto r1 = ConstantLiteral(&b, LiteralUtil::CreateR3<float>({{{1, 2}}}));
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 4}, {4, 6}}, {{6, 8}, {8, 10}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_0_2) {
  XlaBuilder b(TestName());
  auto r1 = ConstantLiteral(&b, LiteralUtil::CreateR3<float>({{{1}, {2}}}));
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 3}, {5, 6}}, {{6, 7}, {9, 10}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_0) {
  XlaBuilder b(TestName());
  auto r1 =
      ConstantLiteral(&b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}}));
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 4}, {6, 8}}, {{6, 8}, {10, 12}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_1) {
  XlaBuilder b(TestName());
  auto r1 =
      ConstantLiteral(&b, LiteralUtil::CreateR3<float>({{{1, 2}}, {{3, 4}}}));
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 4}, {4, 6}}, {{8, 10}, {10, 12}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_2) {
  XlaBuilder b(TestName());
  auto r1 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1}, {2}}, {{3}, {4}}}));
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 3}, {5, 6}}, {{8, 9}, {11, 12}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_0_1_2) {
  XlaBuilder b(TestName());
  auto r1 = ConstantLiteral(&b, LiteralUtil::CreateR3<float>({{{1}}}));
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 3}, {4, 5}}, {{6, 7}, {8, 9}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

struct R2ImplicitBroadcastSpec {
  std::array<int64, 2> output_bounds;
  std::array<int64, 2> minor2major_layout;
  std::array<int64, 2> input_bounds1;
  std::array<int64, 2> input_bounds2;
  HloOpcode op1;
  HloOpcode op2;
} kR2ImplicitBroadcastTestCases[] = {
    {{{2, 3}}, {{1, 0}}, {{2, 1}}, {{2, 1}}, HloOpcode::kAdd, HloOpcode::kAdd},
    {{{2, 3}}, {{1, 0}}, {{2, 1}}, {{1, 3}}, HloOpcode::kAdd, HloOpcode::kAdd},
    {{{2, 3}},
     {{1, 0}},
     {{2, 1}},
     {{1, 1}},
     HloOpcode::kAdd,
     HloOpcode::kMinimum},
    {{{2, 3}},
     {{1, 0}},
     {{1, 3}},
     {{1, 1}},
     HloOpcode::kAdd,
     HloOpcode::kMinimum},
    {{{2, 3}},
     {{1, 0}},
     {{1, 1}},
     {{1, 1}},
     HloOpcode::kAdd,
     HloOpcode::kMinimum},
    {{{2, 3}}, {{0, 1}}, {{2, 1}}, {{2, 1}}, HloOpcode::kAdd, HloOpcode::kAdd},
    {{{150, 150}},
     {{1, 0}},
     {{150, 1}},
     {{150, 1}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{150, 150}},
     {{1, 0}},
     {{150, 1}},
     {{1, 150}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{150, 150}},
     {{1, 0}},
     {{150, 1}},
     {{1, 1}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{50, 150}},
     {{1, 0}},
     {{50, 1}},
     {{50, 1}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{50, 150}},
     {{1, 0}},
     {{50, 1}},
     {{1, 150}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{50, 150}},
     {{1, 0}},
     {{50, 1}},
     {{1, 1}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{150, 50}},
     {{1, 0}},
     {{150, 1}},
     {{150, 1}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{150, 50}},
     {{1, 0}},
     {{150, 1}},
     {{1, 50}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{150, 50}},
     {{1, 0}},
     {{150, 1}},
     {{1, 1}},
     HloOpcode::kAdd,
     HloOpcode::kAdd}};

class BroadcastR2ImplicitTest
    : public BroadcastSimpleTest,
      public ::testing::WithParamInterface<R2ImplicitBroadcastSpec> {};

// Test r2 op1 r2_implicit_1 op2 r2_implicit_2
// where R2 is a rank-2 operand, and r2_implicit_2 are two
// rank-2 operands with degenerate dimensions:
XLA_TEST_P(BroadcastR2ImplicitTest, Doit) {
  const R2ImplicitBroadcastSpec& spec = GetParam();

  XlaBuilder builder(TestName());

  // Operands with degenerate dimensions require implicit broadcasting:
  Shape r2_shape, r2_implicit_shape1, r2_implicit_shape2;
  Array2D<float> r2_array(spec.output_bounds[0], spec.output_bounds[1]);
  Array2D<float> r2_implicit_array1(spec.input_bounds1[0],
                                    spec.input_bounds1[1]);
  Array2D<float> r2_implicit_array2(spec.input_bounds2[0],
                                    spec.input_bounds2[1]);

  std::unique_ptr<GlobalData> r2_global_data =
      MakeR2Data(spec.output_bounds, spec.minor2major_layout, &r2_shape,
                 &r2_array, 1.0, 2.5, 56789);
  std::unique_ptr<GlobalData> r2_implicit_global_data1 =
      MakeR2Data(spec.input_bounds1, spec.minor2major_layout,
                 &r2_implicit_shape1, &r2_implicit_array1, 1.0, 0.2, 56789);
  std::unique_ptr<GlobalData> r2_implicit_global_data2 =
      MakeR2Data(spec.input_bounds2, spec.minor2major_layout,
                 &r2_implicit_shape2, &r2_implicit_array2, 0.8, 0.4, 56789);

  auto r2_implicit_parameter1 =
      Parameter(&builder, 0, r2_implicit_shape1, "input0");
  auto r2_parameter = Parameter(&builder, 1, r2_shape, "input1");
  auto r2_implicit_parameter2 =
      Parameter(&builder, 2, r2_implicit_shape2, "input2");

  XlaOp op1 =
      BuildBinOp(spec.op1, r2_implicit_parameter1, r2_parameter, &builder);
  BuildBinOp(spec.op2, op1, r2_implicit_parameter2, &builder);

  Array2D<float> expected_array(spec.output_bounds[0], spec.output_bounds[1]);

  expected_array.Each([&](int64 i, int64 j, float* v) {
    float v1 = r2_implicit_array1(i % spec.input_bounds1[0],
                                  j % spec.input_bounds1[1]);
    float v2 = r2_array(i, j);
    float v3 = r2_implicit_array2(i % spec.input_bounds2[0],
                                  j % spec.input_bounds2[1]);
    float tmp = ApplyOpToFloats(spec.op1, v1, v2);
    *v = ApplyOpToFloats(spec.op2, tmp, v3);
  });

  auto expected = LiteralUtil::CreateR2FromArray2D(expected_array);
  ComputeAndCompareLiteral(
      &builder, expected,
      {r2_implicit_global_data1.get(), r2_global_data.get(),
       r2_implicit_global_data2.get()},
      ErrorSpec(1e-6, 1e-6));
}

INSTANTIATE_TEST_CASE_P(BroadcastR2ImplicitTestInstances,
                        BroadcastR2ImplicitTest,
                        ::testing::ValuesIn(kR2ImplicitBroadcastTestCases));

XLA_TEST_F(BroadcastSimpleTest, Add2DTo2DDegenerate_0) {
  XlaBuilder b(TestName());
  auto r1 = ConstantLiteral(&b, LiteralUtil::CreateR2<float>({{1, 2}}));
  auto r2 = ConstantLiteral(&b, LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}}));
  Add(r2, r1);

  auto expected = LiteralUtil::CreateR2<float>({{2, 4}, {4, 6}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add2DTo2DDegenerate_1) {
  XlaBuilder b(TestName());
  auto r1 = ConstantLiteral(&b, LiteralUtil::CreateR2<float>({{1}, {2}}));
  auto r2 = ConstantLiteral(&b, LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}}));
  Add(r2, r1);

  auto expected = LiteralUtil::CreateR2<float>({{2, 3}, {5, 6}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add1DTo3DInDim0) {
  XlaBuilder b(TestName());
  auto r1 = ConstantR1<float>(&b, {10, 20});
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r3, r1, {0});

  auto expected = LiteralUtil::CreateR3<float>(
      {{{11, 12}, {13, 14}}, {{25, 26}, {27, 28}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add1DTo3DInDim1) {
  XlaBuilder b(TestName());
  auto r1 = ConstantR1<float>(&b, {10, 20});
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r1, r3, {1});

  auto expected = LiteralUtil::CreateR3<float>(
      {{{11, 12}, {23, 24}}, {{15, 16}, {27, 28}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add1DTo3DInDim2) {
  XlaBuilder b(TestName());
  auto r1 = ConstantR1<float>(&b, {10, 20});
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r1, r3, {2});

  auto expected = LiteralUtil::CreateR3<float>(
      {{{11, 22}, {13, 24}}, {{15, 26}, {17, 28}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add1DTo3DInDimAll) {
  XlaBuilder b(TestName());
  auto r1_0 = ConstantR1<float>(&b, {1000, 2000});
  auto r1_1 = ConstantR1<float>(&b, {100, 200});
  auto r1_2 = ConstantR1<float>(&b, {10, 20});
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  for (int i = 0; i < 3; ++i) {
    r3 = Add(r1_0, r3, {0});
    r3 = Add(r3, r1_1, {1});
    r3 = Add(r1_2, r3, {2});
  }
  r3 = Mul(r3, ConstantR0<float>(&b, -2));

  auto expected = LiteralUtil::CreateR3<float>(
      {{{-6 * 1110 - 2, -6 * 1120 - 4}, {-6 * 1210 - 6, -6 * 1220 - 8}},
       {{-6 * 2110 - 10, -6 * 2120 - 12}, {-6 * 2210 - 14, -6 * 2220 - 16}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add1DTo3DInDimAllWithScalarBroadcast) {
  XlaBuilder b(TestName());
  auto r1_0 = ConstantR1<float>(&b, {1000, 2000});
  auto r1_1 = ConstantR1<float>(&b, {100, 200});
  auto r1_2 = ConstantR1<float>(&b, {10, 20});
  auto r0 = ConstantR0<float>(&b, 3);
  auto r3 = Broadcast(r0, {2, 2, 2});
  for (int i = 0; i < 3; ++i) {
    r3 = Add(r1_0, r3, {0});
    r3 = Add(r3, r1_1, {1});
    r3 = Add(r1_2, r3, {2});
  }
  r3 = Mul(r3, ConstantR0<float>(&b, -1));

  auto expected = LiteralUtil::CreateR3<float>(
      {{{-3 * 1110 - 3, -3 * 1120 - 3}, {-3 * 1210 - 3, -3 * 1220 - 3}},
       {{-3 * 2110 - 3, -3 * 2120 - 3}, {-3 * 2210 - 3, -3 * 2220 - 3}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, InvalidBinaryAndDegenerateBroadcasting) {
  // Binary dimension broadcasting of the smaller lhs ([2, 2] up to [2, 2, 2])
  // results in a shape incompatible with the lhs [2, 3, 1].
  XlaBuilder b(TestName());

  Add(ConstantR2<float>(&b, {{1.0, 5.0}, {1.0, 5.0}}),
      ConstantLiteral(&b, LiteralUtil::CreateR3<float>(
                              {{{2.0}, {3.0}, {4.0}}, {{5.0}, {6.0}, {7.0}}})),
      /*broadcast_dimensions=*/{1, 2});

  auto result_status = Execute(&b, {});
  EXPECT_FALSE(result_status.ok());
  EXPECT_THAT(result_status.status().error_message(),
              HasSubstr("dimension 0 mismatch"));
}

XLA_TEST_F(BroadcastSimpleTest, InvalidInDimensionBroadcasting) {
  // Test invalid broadcasting with [1, 2] and [2, 3] inputs.
  XlaBuilder b(TestName());

  Add(ConstantR2<float>(&b, {{1.0, 2.0}}),
      ConstantR2<float>(&b, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));

  auto result_status = Execute(&b, {});
  EXPECT_FALSE(result_status.ok());
  EXPECT_THAT(result_status.status().error_message(),
              HasSubstr("op add with incompatible shapes"));
}

XLA_TEST_F(BroadcastSimpleTest, InvalidDegenerateBroadcasting) {
  // Test invalid broadcasting with [1, 2] and [2, 3] inputs.
  XlaBuilder b(TestName());

  Add(ConstantR2<float>(&b, {{1.0, 2.0}}),
      ConstantR2<float>(&b, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));

  auto result_status = Execute(&b, {});
  EXPECT_FALSE(result_status.ok());
  EXPECT_THAT(result_status.status().error_message(),
              HasSubstr("op add with incompatible shapes"));
}

}  // namespace
}  // namespace xla
