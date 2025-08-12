/* Copyright 2017 The OpenXLA Authors.

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

// Tests that constants in program memory round trip as expected.

#include "xla/hlo/builder/lib/constants.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "xla/tests/xla_test_backend_predicates.h"
#include <gtest/gtest.h>
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/array4d.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "tsl/platform/ml_dtypes.h"

namespace xla {
namespace {

constexpr ErrorSpec kErrorSpec{1e-3, 1e-5};

using ConstantsTest = ClientLibraryTestRunnerMixin<
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>>;

template <typename T>
class ConstantsFloatTest : public ConstantsTest {};

using FloatTypes =
    ::testing::Types<float, half, tsl::float8_e3m4, tsl::float8_e4m3,
                     tsl::float8_e4m3fn, tsl::float8_e4m3b11fnuz,
                     tsl::float8_e4m3fnuz, tsl::float8_e5m2,
                     tsl::float8_e5m2fnuz
#ifndef XLA_TEST_BACKEND_TPU
                     // TODO(b/385004399): Run tests on these types on TPU.
                     ,
                     tsl::float4_e2m1fn, tsl::float8_e8m0fnu
#endif
                     >;

TYPED_TEST_SUITE(ConstantsFloatTest, FloatTypes);

TEST_F(ConstantsTest, ZeroCellF32) {
  XlaBuilder builder(TestName());
  ConstantR1<float>(&builder, {});

  ComputeAndCompareR1<float>(&builder, {}, {}, kErrorSpec);
}

TYPED_TEST(ConstantsFloatTest, OneCellFloat) {
  std::vector<TypeParam> constant = {TypeParam{2.0}};

  XlaBuilder builder(ConstantsTest::TestName());
  ConstantR1<TypeParam>(&builder, constant);

  ConstantsTest::ComputeAndCompareR1<TypeParam>(&builder, constant, {},
                                                kErrorSpec);
}

TEST_F(ConstantsTest, OneCellS32) {
  std::vector<int32_t> constant = {2};

  XlaBuilder builder(TestName());
  ConstantR1<int32_t>(&builder, constant);

  ComputeAndCompareR1<int32_t>(&builder, constant, {});
}

TEST_F(ConstantsTest, OneCellU32) {
  std::vector<uint32_t> constant = {2};

  XlaBuilder builder(TestName());
  ConstantR1<uint32_t>(&builder, constant);

  ComputeAndCompareR1<uint32_t>(&builder, constant, {});
}

TEST_F(ConstantsTest, OneCellU4) {
  if (test::DeviceTypeIsOneOf({test::kCpu, test::kGpu})) {
    GTEST_SKIP();
  }
  std::vector<u4> constant = {u4(2)};

  XlaBuilder builder(TestName());
  auto c = ConstantR1<u4>(&builder, constant);
  // ComputeAndCompareR1 currently does not support U4, so convert to U8
  ConvertElementType(c, U8);

  ComputeAndCompareR1<uint8_t>(&builder, {2}, {});
}

TEST_F(ConstantsTest, OneCellS4) {
  if (test::DeviceTypeIsOneOf({test::kCpu, test::kGpu})) {
    GTEST_SKIP();
  }
  std::vector<s4> constant = {s4(-2)};

  XlaBuilder builder(TestName());
  auto c = ConstantR1<s4>(&builder, constant);
  // ComputeAndCompareR1 currently does not support S4, so convert to S8
  ConvertElementType(c, S8);

  ComputeAndCompareR1<int8_t>(&builder, {-2}, {});
}

TEST_F(ConstantsTest, EightCells) {
  std::vector<float> constant = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

  XlaBuilder builder(TestName());
  ConstantR1<float>(&builder, constant);

  ComputeAndCompareR1<float>(&builder, constant, {}, kErrorSpec);
}

TEST_F(ConstantsTest, SixteenCells) {
  std::vector<float> constant = {0.0, 1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                                 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};

  XlaBuilder builder(TestName());
  ConstantR1<float>(&builder, constant);

  ComputeAndCompareR1<float>(&builder, constant, {}, kErrorSpec);
}

TEST_F(ConstantsTest, Empty_0x2) {
  XlaBuilder builder(TestName());
  ConstantR2FromArray2D<float>(&builder, Array2D<float>(0, 2));

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 2), {}, kErrorSpec);
}

TEST_F(ConstantsTest, Small_2x2) {
  std::unique_ptr<Array2D<float>> constant =
      MakeLinspaceArray2D(100.0, 200.0, 2, 2);

  XlaBuilder builder(TestName());
  ConstantR2FromArray2D<float>(&builder, *constant);

  ComputeAndCompareR2<float>(&builder, *constant, {}, kErrorSpec);
}

TEST_F(ConstantsTest, Empty_3x0x2) {
  XlaBuilder builder(TestName());
  ConstantLiteral(&builder, LiteralUtil::CreateR3FromArray3D<float>(
                                Array3D<float>(3, 0, 2)));

  ComputeAndCompareR3<float>(&builder, Array3D<float>(3, 0, 2), {});
}

TEST_F(ConstantsTest, Small_2x2x2) {
  XlaBuilder builder(TestName());
  Array3D<float> array3d({
      // x0  x1
      {{1.f, 2.f},   // y0
       {3.f, 4.f}},  // y1

      {{5.f, 6.f},   // y0
       {7.f, 8.f}},  // y1
  });
  ConstantLiteral(&builder, LiteralUtil::CreateR3FromArray3D<float>(array3d));

  ComputeAndCompareR3<float>(&builder, array3d, {});
}

TEST_F(ConstantsTest, Small_3x2x1x1) {
  Array4D<float> input_array(3, 2, 1, 1);
  Array2D<float> pz({
      // z0 z1
      {-1.0f, 4.1f},  // p0
      {2.0f, 4.1f},   // p1
      {5.0f, 4.4f},   // p2
  });
  input_array.FillWithPZ(pz);
  Literal input_literal = LiteralUtil::CreateR4FromArray4D(input_array);

  {
    XlaBuilder builder(TestName());
    ConstantLiteral(&builder, input_literal);
    ComputeAndCompareR4<float>(&builder, input_array, {}, kErrorSpec);
  }

  {
    XlaBuilder builder(TestName());
    ConstantR4FromArray4D<float>(&builder, input_array);
    ComputeAndCompareR4<float>(&builder, input_array, {}, kErrorSpec);
  }
}

// TODO(b/29263943): Support tuple constants.
TEST_F(ConstantsTest, DISABLED_TupleConstant) {
  XlaBuilder builder(TestName());
  ConstantLiteral(&builder, LiteralUtil::MakeTupleFromSlices(
                                {LiteralUtil::CreateR2<float>({{1.0}, {2.0}}),
                                 LiteralUtil::CreateR1<float>({2.0, 42})}));

  Literal result = ExecuteAndTransfer(&builder, {}).value();

  LiteralTestUtil::ExpectR2Near<float>({{1.0}, {2.0}},
                                       LiteralSlice(result, {0}), kErrorSpec);
  LiteralTestUtil::ExpectR1Near<float>({2.0, 42.0}, LiteralSlice(result, {1}),
                                       kErrorSpec);
}

TEST_F(ConstantsTest, Token) {
  XlaBuilder builder(TestName());
  ConstantLiteral(&builder, LiteralUtil::CreateToken());
  // TODO(b/80000000): tokens cannot be returned from computations.
  Tuple(&builder, {});
  TF_ASSERT_OK(ExecuteAndTransfer(&builder, {}).status());
}

TEST_F(ConstantsTest, FullLike) {
  XlaBuilder b(TestName());
  auto val1 = Iota(&b, F32, 3);
  auto val2 = FullLike(val1, 10);
  val1 + val2;
  ComputeAndCompareR1<float>(&b, {10, 11, 12}, {}, kErrorSpec);
}

TEST_F(ConstantsTest, IllegalFullLikeOnTuple) {
  XlaBuilder b(TestName());
  auto tuple = Tuple(&b, {Iota(&b, F32, 3), Iota(&b, F32, 1)});
  FullLike(tuple, 10);  // Illegal; can't do FullLike on a tuple.
  EXPECT_FALSE(b.Build().ok());
}

TEST_F(ConstantsTest, FullLikeScalar) {
  XlaBuilder b(TestName());
  auto scalar1 = ConstantR0WithType(&b, F32, 1);
  auto scalar2 = FullLike(scalar1, 2);
  scalar1 - scalar2;
  ComputeAndCompareR0<float>(&b, -1, {}, kErrorSpec);
}

using ConstantsHloTest = HloPjRtTestBase;

// TODO(b/121147351): Fails on GPU. Not clear if this is expected behavior.
TEST_F(ConstantsHloTest, BitcastOfConstant) {
  if (test::DeviceTypeIsOneOf({test::kGpu, test::kTpu})) {
    GTEST_SKIP();
  }
  const char* testcase = R"(
    HloModule module, is_scheduled=true

    func {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT mul = s32[] add(lhs, rhs)
    }

    ENTRY test {
      constant.0 = s32[1]{0} constant({0})
      parameter.0 = s32[] parameter(0)
      constant-as-scalar = s32[] bitcast(constant.0)
      ROOT result = s32[] call(parameter.0, constant-as-scalar), to_apply=func
    }
  )";
  auto module = ParseAndReturnVerifiedModule(testcase).value();
  auto param = LiteralUtil::CreateR0<int32_t>(1);
  auto result = ExecuteNoHloPasses(std::move(module), {&param});
  EXPECT_TRUE(LiteralTestUtil::Equal(param, result));
}

}  // namespace
}  // namespace xla
