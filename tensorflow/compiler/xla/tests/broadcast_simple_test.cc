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
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

using BroadcastSimpleTest = ClientLibraryTestBase;
using ::testing::HasSubstr;

XLA_TEST_F(BroadcastSimpleTest, ScalarNoOpBroadcast) {
  ComputationBuilder b(client_, TestName());
  b.Broadcast(b.ConstantR0<float>(1.5), {});
  ComputeAndCompareR0<float>(&b, 1.5, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, ScalarTo2D_2x3) {
  ComputationBuilder b(client_, TestName());
  b.Broadcast(b.ConstantR0<float>(2.25), {2, 3});
  Array2D<float> expected(2, 3, 2.25);
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, ScalarParamTo2D_2x3) {
  ComputationBuilder b(client_, TestName());
  ComputationDataHandle src;
  std::unique_ptr<GlobalData> param_data =
      CreateR0Parameter<float>(2.25f, /*parameter_number=*/0, /*name=*/"src",
                               /*builder=*/&b, /*data_handle=*/&src);

  b.Broadcast(src, {2, 3});
  Array2D<float> expected(2, 3, 2.25);
  ComputeAndCompareR2<float>(&b, expected, {param_data.get()},
                             ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, ScalarTo2D_2x0) {
  ComputationBuilder b(client_, TestName());
  b.Broadcast(b.ConstantR0<float>(2.25), {2, 0});
  Array2D<float> expected(2, 0);
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, ScalarTo2D_0x2) {
  ComputationBuilder b(client_, TestName());
  b.Broadcast(b.ConstantR0<float>(2.25), {0, 2});
  Array2D<float> expected(0, 2);
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 1DTo2D) {
  ComputationBuilder b(client_, TestName());
  b.Broadcast(b.ConstantR1<float>({1, 2, 3}), {2});

  Array2D<float> expected(2, 3);
  expected(0, 0) = 1;
  expected(0, 1) = 2;
  expected(0, 2) = 3;
  expected(1, 0) = 1;
  expected(1, 1) = 2;
  expected(1, 2) = 3;
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, ZeroElement_1DTo2D) {
  ComputationBuilder b(client_, TestName());
  b.Broadcast(b.ConstantR1<float>({}), {2});

  Array2D<float> expected(2, 0);
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 1DToZeroElement2D) {
  ComputationBuilder b(client_, TestName());
  b.Broadcast(b.ConstantR1<float>({1, 2, 3}), {0});

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
  ComputationBuilder b(client_, TestName());

  b.Add(b.ConstantR2<float>({{1.0, 5.0}}),
        b.ConstantLiteral(*LiteralUtil::CreateR3<float>(
            {{{2.0}, {3.0}, {4.0}}, {{5.0}, {6.0}, {7.0}}})),
        /*broadcast_dimensions=*/{1, 2});

  auto expected =
      LiteralUtil::CreateR3<float>({{{3.0, 7.0}, {4.0, 8.0}, {5.0, 9.0}},
                                    {{6.0, 10.0}, {7.0, 11.0}, {8.0, 12.0}}});

  ComputeAndCompareLiteral(&b, *expected, {}, ErrorSpec(0.0001));
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
  ComputationBuilder builder(client_, TestName());
  const Shape r3_shape = ShapeUtil::MakeShapeWithLayout(
      F32, spec.output_bounds, spec.minor2major_layout);
  Array3D<float> r3_array(spec.output_bounds[0], spec.output_bounds[1],
                          spec.output_bounds[2]);
  r3_array.FillRandom(1.0, 2.5, 56789);
  auto r3_input =
      LiteralUtil::Relayout(*LiteralUtil::CreateR3FromArray3D(r3_array),
                            LayoutUtil::MakeLayout(spec.minor2major_layout));
  std::unique_ptr<GlobalData> r3_global_data =
      client_->TransferToServer(*r3_input).ConsumeValueOrDie();

  const Shape r3_implicit_shape = ShapeUtil::MakeShapeWithLayout(
      F32, spec.input_bounds, spec.minor2major_layout);
  Array3D<float> r3_implicit_array(spec.input_bounds[0], spec.input_bounds[1],
                                   spec.input_bounds[2]);
  r3_implicit_array.FillRandom(1.0, 0.2, 56789);
  auto r3_implicit_input = LiteralUtil::Relayout(
      *LiteralUtil::CreateR3FromArray3D(r3_implicit_array),
      LayoutUtil::MakeLayout(spec.minor2major_layout));
  std::unique_ptr<GlobalData> r3_implicit_global_data =
      client_->TransferToServer(*r3_implicit_input).ConsumeValueOrDie();

  auto r3_implicit_parameter = builder.Parameter(0, r3_implicit_shape, "input");
  auto r3_parameter = builder.Parameter(1, r3_shape, "input");
  ComputationDataHandle op;
  switch (spec.op) {
    case HloOpcode::kMinimum: {
      auto tmp_op = builder.Min(r3_implicit_parameter, r3_parameter);
      op.Swap(&tmp_op);
      break;
    }
    case HloOpcode::kMaximum: {
      auto tmp_op = builder.Max(r3_implicit_parameter, r3_parameter);
      op.Swap(&tmp_op);
      break;
    }
    case HloOpcode::kMultiply: {
      auto tmp_op = builder.Mul(r3_implicit_parameter, r3_parameter);
      op.Swap(&tmp_op);
      break;
    }
    default: {
      // Default to Add
      auto tmp_op = builder.Add(r3_implicit_parameter, r3_parameter);
      op.Swap(&tmp_op);
    }
  }

  Array3D<float> expected_array(spec.output_bounds[0], spec.output_bounds[1],
                                spec.output_bounds[2]);
  auto Each = ([&](tensorflow::gtl::ArraySlice<int64> indices, float* value) {
    float r3_implicit = r3_implicit_array(indices[0] % spec.input_bounds[0],
                                          indices[1] % spec.input_bounds[1],
                                          indices[2] % spec.input_bounds[2]);
    float r3 = r3_array(indices[0], indices[1], indices[2]);
    switch (spec.op) {
      case HloOpcode::kMinimum: {
        *value = std::min(r3_implicit, r3);
        break;
      }
      case HloOpcode::kMaximum: {
        *value = std::max(r3_implicit, r3);
        break;
      }
      case HloOpcode::kMultiply: {
        *value = r3_implicit * r3;
        break;
      }
      default: {
        // Default to Add
        *value = r3_implicit + r3;
        break;
      }
    }
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
      &builder, *expected,
      {r3_implicit_global_data.get(), r3_global_data.get()},
      ErrorSpec(1e-7, 1e-7));
}

INSTANTIATE_TEST_CASE_P(BroadcastR3ImplicitTestInstances,
                        BroadcastR3ImplicitTest,
                        ::testing::ValuesIn(kR3ImplicitBroadcastTestCases));

// r1 and r3's dim0 matches, and r1's dim1 and dim2 have size 1:
XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_1_2) {
  ComputationBuilder b(client_, TestName());
  ComputationDataHandle r1h;
  ComputationDataHandle r3h;

  Array3D<float> r1d = {{{1}}, {{2}}};
  Array3D<float> r3d = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
  auto r1 = CreateR3Parameter(r1d, 1, "r1", &b, &r1h);
  auto r3 = CreateR3Parameter(r3d, 0, "r3", &b, &r3h);

  b.Add(r3h, r1h);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 3}, {4, 5}}, {{7, 8}, {9, 10}}});

  ComputeAndCompareLiteral(&b, *expected, {r3.get(), r1.get()},
                           ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_0_1) {
  ComputationBuilder b(client_, TestName());
  auto r1 = b.ConstantLiteral(*LiteralUtil::CreateR3<float>({{{1, 2}}}));
  auto r3 = b.ConstantLiteral(
      *LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  b.Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 4}, {4, 6}}, {{6, 8}, {8, 10}}});

  ComputeAndCompareLiteral(&b, *expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_0_2) {
  ComputationBuilder b(client_, TestName());
  auto r1 = b.ConstantLiteral(*LiteralUtil::CreateR3<float>({{{1}, {2}}}));
  auto r3 = b.ConstantLiteral(
      *LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  b.Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 3}, {5, 6}}, {{6, 7}, {9, 10}}});

  ComputeAndCompareLiteral(&b, *expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_0) {
  ComputationBuilder b(client_, TestName());
  auto r1 =
      b.ConstantLiteral(*LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}}));
  auto r3 = b.ConstantLiteral(
      *LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  b.Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 4}, {6, 8}}, {{6, 8}, {10, 12}}});

  ComputeAndCompareLiteral(&b, *expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_1) {
  ComputationBuilder b(client_, TestName());
  auto r1 =
      b.ConstantLiteral(*LiteralUtil::CreateR3<float>({{{1, 2}}, {{3, 4}}}));
  auto r3 = b.ConstantLiteral(
      *LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  b.Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 4}, {4, 6}}, {{8, 10}, {10, 12}}});

  ComputeAndCompareLiteral(&b, *expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_2) {
  ComputationBuilder b(client_, TestName());
  auto r1 = b.ConstantLiteral(
      *LiteralUtil::CreateR3<float>({{{1}, {2}}, {{3}, {4}}}));
  auto r3 = b.ConstantLiteral(
      *LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  b.Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 3}, {5, 6}}, {{8, 9}, {11, 12}}});

  ComputeAndCompareLiteral(&b, *expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_0_1_2) {
  ComputationBuilder b(client_, TestName());
  auto r1 = b.ConstantLiteral(*LiteralUtil::CreateR3<float>({{{1}}}));
  auto r3 = b.ConstantLiteral(
      *LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  b.Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 3}, {4, 5}}, {{6, 7}, {8, 9}}});

  ComputeAndCompareLiteral(&b, *expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add2DTo2DDegenerate_0) {
  ComputationBuilder b(client_, TestName());
  auto r1 = b.ConstantLiteral(*LiteralUtil::CreateR2<float>({{1, 2}}));
  auto r2 = b.ConstantLiteral(*LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}}));
  b.Add(r2, r1);

  auto expected = LiteralUtil::CreateR2<float>({{2, 4}, {4, 6}});

  ComputeAndCompareLiteral(&b, *expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add2DTo2DDegenerate_1) {
  ComputationBuilder b(client_, TestName());
  auto r1 = b.ConstantLiteral(*LiteralUtil::CreateR2<float>({{1}, {2}}));
  auto r2 = b.ConstantLiteral(*LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}}));
  b.Add(r2, r1);

  auto expected = LiteralUtil::CreateR2<float>({{2, 3}, {5, 6}});

  ComputeAndCompareLiteral(&b, *expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add1DTo3DInDim0) {
  ComputationBuilder b(client_, TestName());
  auto r1 = b.ConstantR1<float>({10, 20});
  auto r3 = b.ConstantLiteral(
      *LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  b.Add(r3, r1, {0});

  auto expected = LiteralUtil::CreateR3<float>(
      {{{11, 12}, {13, 14}}, {{25, 26}, {27, 28}}});

  ComputeAndCompareLiteral(&b, *expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add1DTo3DInDim1) {
  ComputationBuilder b(client_, TestName());
  auto r1 = b.ConstantR1<float>({10, 20});
  auto r3 = b.ConstantLiteral(
      *LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  b.Add(r1, r3, {1});

  auto expected = LiteralUtil::CreateR3<float>(
      {{{11, 12}, {23, 24}}, {{15, 16}, {27, 28}}});

  ComputeAndCompareLiteral(&b, *expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add1DTo3DInDim2) {
  ComputationBuilder b(client_, TestName());
  auto r1 = b.ConstantR1<float>({10, 20});
  auto r3 = b.ConstantLiteral(
      *LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  b.Add(r1, r3, {2});

  auto expected = LiteralUtil::CreateR3<float>(
      {{{11, 22}, {13, 24}}, {{15, 26}, {17, 28}}});

  ComputeAndCompareLiteral(&b, *expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add1DTo3DInDimAll) {
  ComputationBuilder b(client_, TestName());
  auto r1_0 = b.ConstantR1<float>({1000, 2000});
  auto r1_1 = b.ConstantR1<float>({100, 200});
  auto r1_2 = b.ConstantR1<float>({10, 20});
  auto r3 = b.ConstantLiteral(
      *LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  for (int i = 0; i < 3; ++i) {
    r3 = b.Add(r1_0, r3, {0});
    r3 = b.Add(r3, r1_1, {1});
    r3 = b.Add(r1_2, r3, {2});
  }
  r3 = b.Mul(r3, b.ConstantR0<float>(-2));

  auto expected = LiteralUtil::CreateR3<float>(
      {{{-6 * 1110 - 2, -6 * 1120 - 4}, {-6 * 1210 - 6, -6 * 1220 - 8}},
       {{-6 * 2110 - 10, -6 * 2120 - 12}, {-6 * 2210 - 14, -6 * 2220 - 16}}});

  ComputeAndCompareLiteral(&b, *expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add1DTo3DInDimAllWithScalarBroadcast) {
  ComputationBuilder b(client_, TestName());
  auto r1_0 = b.ConstantR1<float>({1000, 2000});
  auto r1_1 = b.ConstantR1<float>({100, 200});
  auto r1_2 = b.ConstantR1<float>({10, 20});
  auto r0 = b.ConstantR0<float>(3);
  auto r3 = b.Broadcast(r0, {2, 2, 2});
  for (int i = 0; i < 3; ++i) {
    r3 = b.Add(r1_0, r3, {0});
    r3 = b.Add(r3, r1_1, {1});
    r3 = b.Add(r1_2, r3, {2});
  }
  r3 = b.Mul(r3, b.ConstantR0<float>(-1));

  auto expected = LiteralUtil::CreateR3<float>(
      {{{-3 * 1110 - 3, -3 * 1120 - 3}, {-3 * 1210 - 3, -3 * 1220 - 3}},
       {{-3 * 2110 - 3, -3 * 2120 - 3}, {-3 * 2210 - 3, -3 * 2220 - 3}}});

  ComputeAndCompareLiteral(&b, *expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, InvalidBinaryAndDegenerateBroadcasting) {
  // Binary dimension broadcasting of the smaller lhs ([2, 2] up to [2, 2, 2])
  // results in a shape incompatible with the lhs [2, 3, 1].
  ComputationBuilder b(client_, TestName());

  b.Add(b.ConstantR2<float>({{1.0, 5.0}, {1.0, 5.0}}),
        b.ConstantLiteral(*LiteralUtil::CreateR3<float>(
            {{{2.0}, {3.0}, {4.0}}, {{5.0}, {6.0}, {7.0}}})),
        /*broadcast_dimensions=*/{1, 2});

  auto result_status = Execute(&b, {});
  EXPECT_FALSE(result_status.ok());
  EXPECT_THAT(result_status.status().error_message(),
              HasSubstr("broadcast dimension 0 mismatch"));
}

XLA_TEST_F(BroadcastSimpleTest, InvalidInDimensionBroadcasting) {
  // Test invalid broadcasting with [1, 2] and [2, 3] inputs.
  ComputationBuilder b(client_, TestName());

  b.Add(b.ConstantR2<float>({{1.0, 2.0}}),
        b.ConstantR2<float>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));

  auto result_status = Execute(&b, {});
  EXPECT_FALSE(result_status.ok());
  EXPECT_THAT(result_status.status().error_message(),
              HasSubstr("binary op BINOP_ADD with incompatible shapes"));
}

XLA_TEST_F(BroadcastSimpleTest, InvalidDegenerateBroadcasting) {
  // Test invalid broadcasting with [1, 2] and [2, 3] inputs.
  ComputationBuilder b(client_, TestName());

  b.Add(b.ConstantR2<float>({{1.0, 2.0}}),
        b.ConstantR2<float>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));

  auto result_status = Execute(&b, {});
  EXPECT_FALSE(result_status.ok());
  EXPECT_THAT(result_status.status().error_message(),
              HasSubstr("binary op BINOP_ADD with incompatible shapes"));
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
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
