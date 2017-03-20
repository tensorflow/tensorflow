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
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

using BroadcastSimpleTest = ClientLibraryTestBase;

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
  EXPECT_MATCH(result_status.status().error_message(),
               testing::ContainsRegex("broadcast dimension 0 mismatch"));
}

XLA_TEST_F(BroadcastSimpleTest, InvalidInDimensionBroadcasting) {
  // Test invalid broadcasting with [1, 2] and [2, 3] inputs.
  ComputationBuilder b(client_, TestName());

  b.Add(b.ConstantR2<float>({{1.0, 2.0}}),
        b.ConstantR2<float>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));

  auto result_status = Execute(&b, {});
  EXPECT_FALSE(result_status.ok());
  EXPECT_MATCH(
      result_status.status().error_message(),
      testing::ContainsRegex("binary op BINOP_ADD with incompatible shapes"));
}

XLA_TEST_F(BroadcastSimpleTest, InvalidDegenerateBroadcasting) {
  // Test invalid broadcasting with [1, 2] and [2, 3] inputs.
  ComputationBuilder b(client_, TestName());

  b.Add(b.ConstantR2<float>({{1.0, 2.0}}),
        b.ConstantR2<float>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));

  auto result_status = Execute(&b, {});
  EXPECT_FALSE(result_status.ok());
  EXPECT_MATCH(
      result_status.status().error_message(),
      testing::ContainsRegex("binary op BINOP_ADD with incompatible shapes"));
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
