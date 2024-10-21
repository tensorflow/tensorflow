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

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class ConvertTest : public ClientLibraryTestBase {
 public:
  explicit ConvertTest(se::Platform* platform = nullptr)
      : ClientLibraryTestBase(platform) {
    mutable_debug_options()->add_xla_disable_hlo_passes("algsimp");
    mutable_debug_options()->add_xla_disable_hlo_passes("inline");
    mutable_debug_options()->add_xla_disable_hlo_passes(
        "simplify-fp-conversions");
    mutable_debug_options()->set_xla_allow_excess_precision(false);
  }
};

template <typename T>
class ConvertTestT : public ConvertTest {
 public:
  using ConvertTest::ConvertTest;
};
using FloatingPointTypeList =
    ::testing::Types<tsl::float8_e5m2, tsl::float8_e4m3, tsl::float8_e4m3fn,
                     tsl::float8_e5m2fnuz, tsl::float8_e4m3fnuz,
                     tsl::float8_e3m4, Eigen::half, bfloat16, float, double>;
TYPED_TEST_SUITE(ConvertTestT, FloatingPointTypeList);

template <typename T>
class ConvertTestF16 : public ConvertTest {
 public:
  using ConvertTest::ConvertTest;
};
using F16TypeList = ::testing::Types<Eigen::half, bfloat16>;
TYPED_TEST_SUITE(ConvertTestF16, F16TypeList);

TEST_F(ConvertTest, ConvertR1S32ToR1S32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {42, 64});
  ConvertElementType(a, S32);

  std::vector<int32_t> expected = {42, 64};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1S32ToR1U32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {42, 64});
  ConvertElementType(a, U32);

  std::vector<uint32_t> expected = {42, 64};
  ComputeAndCompareR1<uint32_t>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1S32ToR1PRED) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {42, 0, -64});
  ConvertElementType(a, PRED);

  std::array<bool, 3> expected = {true, false, true};
  ComputeAndCompareR1<bool>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1U32ToR1U32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint32_t>(&builder, {42, 64});
  ConvertElementType(a, U32);

  std::vector<uint32_t> expected = {42, 64};
  ComputeAndCompareR1<uint32_t>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1U32ToR1S32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint32_t>(&builder, {42, 64});
  ConvertElementType(a, S32);

  std::vector<int32_t> expected = {42, 64};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1U32ToR1PRED) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint32_t>(&builder, {42, 0, 64});
  ConvertElementType(a, PRED);

  std::array<bool, 3> expected = {true, false, true};
  ComputeAndCompareR1<bool>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1F32ToR1F32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {42.0f, 64.0f});
  ConvertElementType(a, F32);

  std::vector<float> expected = {42.0f, 64.0f};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1F32ToR1PRED) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {42.0f, 0.0f, 64.0f});
  ConvertElementType(a, PRED);

  std::array<bool, 3> expected = {true, false, true};
  ComputeAndCompareR1<bool>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1S32ToR1F32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {42, 64});
  ConvertElementType(a, F32);

  std::vector<float> expected = {42.0f, 64.0f};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1PREDToR1S32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {true, false, true});
  ConvertElementType(a, S32);

  std::vector<int32_t> expected = {1, 0, 1};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1PREDToR1U32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {true, false, true});
  ConvertElementType(a, U32);

  std::vector<uint32_t> expected = {1, 0, 1};
  ComputeAndCompareR1<uint32_t>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1PREDToR1F32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {true, false, true});
  ConvertElementType(a, F32);

  std::vector<float> expected = {1., 0., 1.};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(ConvertTest, ConvertR1S0S32ToR1S0F32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {});
  ConvertElementType(a, F32);

  std::vector<float> expected = {};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1F32ToR1S32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {42.6, 64.4});
  ConvertElementType(a, S32);

  std::vector<int32_t> expected = {42, 64};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

XLA_TEST_F(ConvertTest, ConvertR1S64ToR1F32) {
  XlaBuilder builder(TestName());
  std::vector<int64_t> arg{
      -9223371216516022272,
      -2,
      -1,
      -0x7FFFFFFF,
      -0x80000000,
      0,
      1,
      2,
      1073742145,
      1073742656,
      0x7FFFFFFF,
      0x80000000,
      826720496944058148,
      4296062029846194332,
      0x0007FB72E4000000LL,
      0x0007FB72E4000001LL,
      0x0007FB72E6000000LL,
      0x0007FB72E7000000LL,
      0x0007FB72E7FFFFFFLL,
      0x0007FB72E8000000LL,
      0x0007FB72E8000001LL,
      0x0007FB72EA000000LL,
      0x0007FB72EB000000LL,
      0x0007FB72EBFFFFFFLL,
      0x0007FB72EC000000LL,
      0x7FFFFF0000000000LL,
      0x7FFFFF8000000000LL,
      0x7FFFFFFFFFFFFF00,
      static_cast<int64_t>(0xFFFFFFFFFFFFFFFF),
      static_cast<int64_t>(0x0000f234e67e0001LL),
      static_cast<int64_t>(0x8000000000000000),
      static_cast<int64_t>(0x8000000000000000LL),
      static_cast<int64_t>(0x8000000000000001LL),
      static_cast<int64_t>(0x8000008000000000LL),
      static_cast<int64_t>(0x8000010000000000LL),
  };
  Literal arg_literal = LiteralUtil::CreateR1<int64_t>({arg});
  auto arg_param = Parameter(&builder, 0, arg_literal.shape(), "arg_param");
  std::unique_ptr<GlobalData> arg_data =
      client_->TransferToServer(arg_literal).value();

  ConvertElementType(arg_param, F32);

  std::vector<float> expected(arg.size());
  for (int64_t i = 0; i < arg.size(); ++i) {
    expected[i] = static_cast<float>(arg[i]);
  }
  ComputeAndCompareR1<float>(&builder, expected, {arg_data.get()});
}

XLA_TEST_F(ConvertTest, ConvertR1U32ToR1F32) {
  XlaBuilder builder(TestName());
  std::vector<uint32_t> arg{0,          1,          0x1000,     0x7fffffff,
                            0x80000000, 0x80000001, 0x80000002, 0x80000003,
                            0x80000080, 0x80000081, 0x80000082, 0xFFFFFFFF};
  Literal arg_literal = LiteralUtil::CreateR1<uint32_t>({arg});
  auto arg_param = Parameter(&builder, 0, arg_literal.shape(), "arg_param");
  std::unique_ptr<GlobalData> arg_data =
      client_->TransferToServer(arg_literal).value();

  ConvertElementType(arg_param, F32);

  std::vector<float> expected(arg.size());
  for (int64_t i = 0; i < arg.size(); ++i) {
    expected[i] = static_cast<float>(arg[i]);
  }
  ComputeAndCompareR1<float>(&builder, expected, {arg_data.get()});
}

XLA_TEST_F(ConvertTest, ConvertR1F32ToR1U32) {
  XlaBuilder builder(TestName());
  std::vector<float> arg{0.0f,        1.0f,          16777216.0f,
                         16777218.0f, 2147483647.0f, 4294967040.0f};
  Literal arg_literal = LiteralUtil::CreateR1<float>({arg});
  auto arg_param = Parameter(&builder, 0, arg_literal.shape(), "arg_param");
  std::unique_ptr<GlobalData> arg_data =
      client_->TransferToServer(arg_literal).value();

  ConvertElementType(arg_param, U32);

  std::vector<uint32_t> expected(arg.size());
  for (int64_t i = 0; i < arg.size(); ++i) {
    expected[i] = static_cast<uint32_t>(arg[i]);
  }
  ComputeAndCompareR1<uint32_t>(&builder, expected, {arg_data.get()});
}

XLA_TEST_F(ConvertTest, ConvertR1U32ToR1S64) {
  XlaBuilder builder(TestName());
  std::vector<uint32_t> arg{0, 1, 0x1000, 0x7fffffff, 0x80000082, 0xFFFFFFFF};
  Literal arg_literal = LiteralUtil::CreateR1<uint32_t>({arg});
  auto arg_param = Parameter(&builder, 0, arg_literal.shape(), "arg_param");
  std::unique_ptr<GlobalData> arg_data =
      client_->TransferToServer(arg_literal).value();

  ConvertElementType(arg_param, S64);

  std::vector<int64_t> expected(arg.size());
  for (int64_t i = 0; i < arg.size(); ++i) {
    expected[i] = static_cast<int64_t>(arg[i]);
  }
  ComputeAndCompareR1<int64_t>(&builder, expected, {arg_data.get()});
}

XLA_TEST_F(ConvertTest, ConvertR1S32ToR1S64) {
  XlaBuilder builder(TestName());
  std::vector<int32_t> arg{0, 1, 0x1000, -1, -0x1000};
  Literal arg_literal = LiteralUtil::CreateR1<int32_t>({arg});
  auto arg_param = Parameter(&builder, 0, arg_literal.shape(), "arg_param");
  std::unique_ptr<GlobalData> arg_data =
      client_->TransferToServer(arg_literal).value();

  ConvertElementType(arg_param, S64);

  std::vector<int64_t> expected(arg.size());
  for (int64_t i = 0; i < arg.size(); ++i) {
    expected[i] = static_cast<int64_t>(arg[i]);
  }
  ComputeAndCompareR1<int64_t>(&builder, expected, {arg_data.get()});
}

XLA_TEST_F(ConvertTest, ConvertR1F32ToR1S64) {
  XlaBuilder builder(TestName());
  // Test cases from compiler_rt library.
  std::vector<float> arg{0.0f,
                         0.5f,
                         0.99f,
                         1.0f,
                         1.5f,
                         1.99f,
                         2.0f,
                         2.01f,
                         2147483648.f,
                         -0.5f,
                         -0.99f,
                         -1.0f,
                         -1.5f,
                         -1.99f,
                         -2.0f,
                         -2.01f,
                         9223371487098961920.f,
                         9223370937343148032.f,
                         -9223371487098961920.f,
                         -9223370937343148032.f};
  Literal arg_literal = LiteralUtil::CreateR1<float>({arg});
  auto arg_param = Parameter(&builder, 0, arg_literal.shape(), "arg_param");
  std::unique_ptr<GlobalData> arg_data =
      client_->TransferToServer(arg_literal).value();

  ConvertElementType(arg_param, S64);

  std::vector<int64_t> expected(arg.size());
  for (int64_t i = 0; i < arg.size(); ++i) {
    expected[i] = static_cast<int64_t>(arg[i]);
  }
  ComputeAndCompareR1<int64_t>(&builder, expected, {arg_data.get()});
}

XLA_TEST_F(ConvertTest, ConvertR1U8ToR1F32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint8_t>(&builder, {32, 64});
  ConvertElementType(a, F32);

  std::vector<float> expected = {32.0, 64.0};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(ConvertTest, ConvertR1U8ToR1S32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint8_t>(&builder, {32, 64});
  ConvertElementType(a, S32);

  std::vector<int32_t> expected = {32, 64};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

XLA_TEST_F(ConvertTest, ConvertR1U8ToR1U32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint8_t>(&builder, {32, 64});
  ConvertElementType(a, U32);

  std::vector<uint32_t> expected = {32, 64};
  ComputeAndCompareR1<uint32_t>(&builder, expected, {});
}

XLA_TEST_F(ConvertTest, ConvertR1F32ToR1F64) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {32.0f, 64.0f});
  ConvertElementType(a, F64);

  std::vector<double> expected = {32.0, 64.0};
  ComputeAndCompareR1<double>(&builder, expected, {});
}

XLA_TEST_F(ConvertTest, ConvertR1F64ToR1F32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<double>(&builder, {32.0, 64.0});
  ConvertElementType(a, F32);

  std::vector<float> expected = {32.0f, 64.0f};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertS32Extremes) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {std::numeric_limits<int32_t>::min(),
                                          std::numeric_limits<int32_t>::max()});
  ConvertElementType(a, F32);

  std::vector<float> expected = {
      static_cast<float>(std::numeric_limits<int32_t>::min()),
      static_cast<float>(std::numeric_limits<int32_t>::max())};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(ConvertTest, ConvertMapToS32) {
  XlaBuilder builder(TestName());
  auto b = builder.CreateSubBuilder("convert");
  auto param = Parameter(b.get(), 0, ShapeUtil::MakeShape(F32, {}), "in");
  ConvertElementType(param, S32);
  auto a = ConstantR1<float>(&builder, {42.0f, 64.0f});
  Map(&builder, {a}, b->BuildAndNoteError(), {0});

  std::vector<int32_t> expected = {42, 64};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertMapToF32) {
  XlaBuilder builder(TestName());
  auto b = builder.CreateSubBuilder("convert");
  auto param = Parameter(b.get(), 0, ShapeUtil::MakeShape(S32, {}), "in");
  ConvertElementType(param, F32);
  auto a = ConstantR1<int32_t>(&builder, {42, 64});
  Map(&builder, {a}, b->BuildAndNoteError(), {0});

  std::vector<float> expected = {42.0f, 64.0f};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

// Regression test for b/31758660. When ReshapeMover transforms
//   input -> reshape -> convert
// to
//   input -> convert -> reshape
// the new convert should have the same element type as the old convert.
TEST_F(ConvertTest, ConvertReshape) {
  XlaBuilder builder(TestName());
  auto input = ConstantR1<int32_t>(&builder, {42});
  auto reshape = Reshape(input, /*dimensions=*/{0}, /*new_sizes=*/{});
  ConvertElementType(reshape, F32);

  ComputeAndCompareR0<float>(&builder, 42.0f, {}, ErrorSpec(0.0001));
}

std::vector<float> GetInterestingF16ConversionTestCases() {
  float infinity = std::numeric_limits<float>::infinity();
  float half_min_positive_normal = absl::bit_cast<float, uint32_t>(0x38800000);
  float half_max_subnormal = absl::bit_cast<float, uint32_t>(0x387fc000);
  float half_min_positive_subnormal =
      absl::bit_cast<float, uint32_t>(0x33800000);
  float half_max = 65504.0f;

  std::vector<float> test_cases(
      {-infinity, -(half_max * 2 + 1), -half_max, -42.0f, -1.0f,
       -half_min_positive_subnormal, -half_max_subnormal,
       -half_min_positive_normal, -0.0f, 0.0f, half_min_positive_subnormal,
       half_max_subnormal, half_min_positive_normal, 1.0f, 42.0f, half_max,
       (half_max * 2 + 1), infinity});
  return test_cases;
}

XLA_TEST_F(ConvertTest, ConvertR1F16ToR1F32) {
  std::vector<float> test_cases = GetInterestingF16ConversionTestCases();
  std::vector<half> input;
  absl::c_transform(test_cases, std::back_inserter(input),
                    [](float f) { return Eigen::half(f); });
  std::vector<float> expected_output;
  absl::c_transform(input, std::back_inserter(expected_output),
                    [](Eigen::half h) { return static_cast<float>(h); });

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> dot_lhs_handle,
      client_->TransferToServer(LiteralUtil::CreateR1<half>(input)));

  XlaBuilder builder(TestName());
  ConvertElementType(
      Parameter(&builder, 0,
                ShapeUtil::MakeShape(F16, {static_cast<int64_t>(input.size())}),
                "param"),
      F32);

  ComputeAndCompareR1<float>(&builder, expected_output, {dot_lhs_handle.get()});
}

XLA_TEST_F(ConvertTest, ConvertR1F32ToR1F16) {
  std::vector<float> input = GetInterestingF16ConversionTestCases();
  std::vector<half> expected_output;
  absl::c_transform(input, std::back_inserter(expected_output),
                    [](float f) { return Eigen::half(f); });

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> dot_lhs_handle,
      client_->TransferToServer(LiteralUtil::CreateR1<float>(input)));

  XlaBuilder builder(TestName());
  ConvertElementType(
      Parameter(&builder, 0,
                ShapeUtil::MakeShape(F32, {static_cast<int64_t>(input.size())}),
                "param"),
      F16);

  ComputeAndCompareR1<half>(&builder, expected_output, {dot_lhs_handle.get()});
}

XLA_TEST_F(ConvertTest, ConvertC64ToC64) {
  XlaBuilder builder(TestName());
  std::vector<complex64> x = {{42.0f, 64.0f}};
  ConvertElementType(ConstantR1<complex64>(&builder, x), C64);
  ComputeAndCompareR1<complex64>(&builder, x, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConvertTest, ConvertS64S64) {
  XlaBuilder builder(TestName());
  std::vector<int64_t> x = {{-42, 64}};
  ConvertElementType(ConstantR1<int64_t>(&builder, x), S64);
  ComputeAndCompareR1<int64_t>(&builder, x, {});
}

XLA_TEST_F(ConvertTest, ConvertU64U64) {
  XlaBuilder builder(TestName());
  std::vector<uint64_t> x = {{42, 64}};
  ConvertElementType(ConstantR1<uint64_t>(&builder, x), U64);
  ComputeAndCompareR1<uint64_t>(&builder, x, {});
}

XLA_TEST_F(ConvertTest, ConvertU64S64) {
  XlaBuilder builder(TestName());
  std::vector<uint64_t> unsigned_x = {{42, UINT64_MAX}};
  ConvertElementType(ConstantR1<uint64_t>(&builder, unsigned_x), S64);
  std::vector<int64_t> signed_x = {{42, -1}};
  ComputeAndCompareR1<int64_t>(&builder, signed_x, {});
}

XLA_TEST_F(ConvertTest, ConvertS64U64) {
  XlaBuilder builder(TestName());
  std::vector<int64_t> signed_x = {{42, -1, INT64_MIN}};
  ConvertElementType(ConstantR1<int64_t>(&builder, signed_x), U64);
  std::vector<uint64_t> unsigned_x = {{42, UINT64_MAX, IPow<uint64_t>(2, 63)}};
  ComputeAndCompareR1<uint64_t>(&builder, unsigned_x, {});
}

TEST_F(ConvertTest, ConvertR1S4ToR1S8) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<s4>(&builder, {s4(0), s4(1), s4(2), s4(-8)});
  ConvertElementType(a, S8);

  std::vector<int8_t> expected = {0, 1, 2, -8};
  ComputeAndCompareR1<int8_t>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1S4ParameterToR1S8) {
  XlaBuilder builder(TestName());
  Literal arg_literal =
      LiteralUtil::CreateR1<s4>({s4(0), s4(1), s4(2), s4(-8)});
  auto arg_param = Parameter(&builder, 0, arg_literal.shape(), "arg_param");
  std::unique_ptr<GlobalData> arg_data =
      client_->TransferToServer(arg_literal).value();

  ConvertElementType(arg_param, S8);

  std::vector<int8_t> expected = {0, 1, 2, -8};
  ComputeAndCompareR1<int8_t>(&builder, expected, {arg_data.get()});
}

TEST_F(ConvertTest, ConvertR1U4ToR1U8) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<u4>(&builder, {u4(0), u4(1), u4(2), u4(15)});
  ConvertElementType(a, U8);

  std::vector<uint8_t> expected = {0, 1, 2, 15};
  ComputeAndCompareR1<uint8_t>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1U4ParameterToR1U8) {
  XlaBuilder builder(TestName());
  Literal arg_literal =
      LiteralUtil::CreateR1<u4>({u4(0), u4(1), u4(2), u4(15)});
  auto arg_param = Parameter(&builder, 0, arg_literal.shape(), "arg_param");
  std::unique_ptr<GlobalData> arg_data =
      client_->TransferToServer(arg_literal).value();

  ConvertElementType(arg_param, U8);

  std::vector<uint8_t> expected = {0, 1, 2, 15};
  ComputeAndCompareR1<uint8_t>(&builder, expected, {arg_data.get()});
}

TEST_F(ConvertTest, ConvertR1S8ToR1S4) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int8_t>(&builder, {0, 1, 2, -8});
  ConvertElementType(a, S4);

  std::vector<s4> expected = {s4(0), s4(1), s4(2), s4(-8)};
  ComputeAndCompareR1<s4>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1U8ToR1U4) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<uint8_t>(&builder, {0, 1, 2, 15});
  ConvertElementType(a, U4);

  std::vector<u4> expected = {u4(0), u4(1), u4(2), u4(15)};
  ComputeAndCompareR1<u4>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1S8ToR1S4Roundtrip) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int8_t>(&builder, {0, 8, -8, -9, 127, -128});
  auto b = ConvertElementType(a, S4);
  ConvertElementType(b, S8);

  std::vector<int8_t> expected = {0, -8, -8, 7, -1, 0};
  ComputeAndCompareR1<int8_t>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1F32ToR1S4) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {0., 2.5, -2.5});
  ConvertElementType(a, S4);

  std::vector<s4> expected = {s4(0), s4(2), s4(-2)};
  ComputeAndCompareR1<s4>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1S4ToR1F32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<s4>(&builder, {s4(0), s4(1), s4(2), s4(-8)});
  ConvertElementType(a, F32);

  std::vector<float> expected = {0, 1, 2, -8};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(ConvertTest, ConvertBF16F32) {
  XlaBuilder builder(TestName());

  std::vector<bfloat16> all_bfloats(1 << 16);
  for (int i = 0; i < all_bfloats.size(); ++i) {
    all_bfloats[i] =
        Eigen::numext::bit_cast<bfloat16>(static_cast<uint16_t>(i));
  }

  std::vector<uint32_t> expected(all_bfloats.size());
  for (int i = 0; i < expected.size(); ++i) {
    expected[i] = (1U << 16) * i;
  }

  // Exhaustively test all bf16 to f32 conversions.
  xla::XlaOp all_bfloats_bf16 = ConstantR1<bfloat16>(&builder, all_bfloats);
  xla::XlaOp all_bfloats_f32 = ConvertElementType(all_bfloats_bf16, F32);
  BitcastConvertType(all_bfloats_f32, U32);

  TF_ASSERT_OK_AND_ASSIGN(const auto results, ExecuteAndTransfer(&builder, {}));
  for (int i = 0; i < expected.size(); ++i) {
    const auto result = results.Get<uint32_t>({i});
    const auto correct = expected[i];
    if (all_bfloats[i] != 0.0f &&
        all_bfloats[i] < std::numeric_limits<float>::min()) {
      // Subnormals may not be preserved, zero will do.
      const float same_signed_zero =
          Eigen::numext::signbit(all_bfloats[i]) ? -0.0f : 0.0f;
      if (result != correct) {
        EXPECT_EQ(result, absl::bit_cast<uint32_t>(same_signed_zero));
      }
    } else if (Eigen::numext::isnan(all_bfloats[i])) {
      // NaNs may not be preserved, any NaN will do.
      ASSERT_TRUE(std::isnan(absl::bit_cast<float>(correct)));
      EXPECT_TRUE(std::isnan(absl::bit_cast<float>(result)));
    } else {
      EXPECT_EQ(result, correct);
    }
  }
}

XLA_TEST_F(ConvertTest, ConvertF32BF16) {
  XlaBuilder builder(TestName());

  std::vector<float> floats(100);
  std::minstd_rand0 generator;
  for (int i = 0; i < floats.size(); ++i) {
    floats[i] = generator();

    // Ensure the first 10 cases has rounding.
    if (i < 10) {
      auto val = absl::bit_cast<uint32_t>(floats[i]);
      val |= 1 << 15;
      floats[i] = absl::bit_cast<float>(val);
    }
  }
  // Test NaN and -NaN.
  floats.push_back(std::numeric_limits<float>::quiet_NaN());
  floats.push_back(-std::numeric_limits<float>::quiet_NaN());

  // NaNs that have the LSB of the significand set.
  // Just truncating the significand will result in an `inf` BF16 value.
  floats.push_back(absl::bit_cast<float>(0x7F800001));
  floats.push_back(absl::bit_cast<float>(0xFF800001));

  std::vector<bfloat16> expected(floats.size());
  for (int i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<bfloat16>(floats[i]);
  }

  xla::XlaOp lit_f32 = ConstantR1<float>(&builder, floats);
  xla::XlaOp lit_bf16 = ConvertElementType(lit_f32, BF16);
  BitcastConvertType(lit_bf16, U16);

  TF_ASSERT_OK_AND_ASSIGN(const auto results, ExecuteAndTransfer(&builder, {}));
  for (int i = 0; i < expected.size(); ++i) {
    const auto result = results.Get<uint16_t>({i});
    const auto correct = absl::bit_cast<uint16_t>(expected[i]);
    if (floats[i] != 0.0f && floats[i] < std::numeric_limits<float>::min()) {
      // Subnormals may not be preserved, zero will do.
      const bfloat16 same_signed_zero =
          bfloat16(std::signbit(floats[i]) ? -0.0f : 0.0f);
      if (result != correct) {
        EXPECT_EQ(result, absl::bit_cast<uint16_t>(same_signed_zero));
      }
    } else if (std::isnan(floats[i])) {
      // NaNs may not be preserved, any NaN will do.
      ASSERT_TRUE(std::isnan(absl::bit_cast<bfloat16>(correct)));
      EXPECT_TRUE(std::isnan(absl::bit_cast<bfloat16>(result)));
      if (client_->platform()->Name() == "Host") {
        // The sign bits must match.
        EXPECT_EQ(result >> 15, correct >> 15);
      }
    } else {
      EXPECT_EQ(result, correct);
    }
  }
}

XLA_TYPED_TEST(ConvertTestT, ConvertFPToPred) {
  XlaBuilder builder(this->TestName());
  using FP = TypeParam;

  auto a = ConstantR1<FP>(&builder, {FP{0.0}, FP{0.25}, FP{2.0}, FP{-0.0}});
  ConvertElementType(a, PRED);

  std::array<bool, 4> expected = {false, true, true, false};
  this->template ComputeAndCompareR1<bool>(&builder, expected, {});
}

// ----- F8E5M2

XLA_TEST_F(ConvertTest, ConvertF16F8e5m2Roundtrip) {
  // Convert from FP16 to FP8, then back to FP16.
  XlaBuilder builder(TestName());
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();

  struct TestCase {
    float input;
    float expected_roundtrip;
  } test_cases[] = {
      // clang-format off
      {0.0, 0.0},
      {-0.0, 0.0},
      {1.0, 1.0},
      {-1.0, -1.0},
      {nan, nan},
      {inf, inf},
      // clang-format on
      {0x1.2p0, 0x1p0},        // Round-to-even down
      {0x1.6p0, 0x1.8p0},      // Round-to-even up
      {0x1.Cp15, 0x1.Cp15},    // Max value
      {0x1.DFCp15, 0x1.Cp15},  // Largest number that doesn't overflow
      {0x1.Ep15, inf},         // Smallest number that overflows
      {0x1p16, inf},           // Overflow
      {0x1p-14, 0x1p-14},      // Smallest F8 normal
      {0x1.Cp-15, 0x1p-14},    // Smallest number rounding up to normal

      // Denormal tests
      {0x0.8p-14, 0x0.8p-14},    // Denormal without rounding
      {0x0.Ap-14, 0x0.8p-14},    // Round-to-even down
      {0x0.Ep-14, 0x1.0p-14},    // Round-to-even up
      {0x0.98p-14, 0x0.8p-14},   // Round-to-nearest down
      {0x0.A8p-14, 0x0.Cp-14},   // Round-to-nearest up
      {0x0.2p-14, 0},            // Largest number that underflows
      {0x0.204p-14, 0x0.4p-14},  // Smallest number that doesn't underflow
      {0x0.DFCp-14, 0x0.Cp-14},  // Largest number that rounds to denormal
  };

  std::vector<Eigen::half> inputs;
  std::vector<Eigen::half> expected_roundtrip;
  for (auto test_case : test_cases) {
    inputs.push_back(Eigen::half{test_case.input});
    expected_roundtrip.push_back(Eigen::half{test_case.expected_roundtrip});
  }

  auto f8 =
      ConvertElementType(ConstantR1<Eigen::half>(&builder, inputs), F8E5M2);
  ConvertElementType(f8, F16);
  ComputeAndCompareR1<Eigen::half>(&builder, expected_roundtrip, {},
                                   ErrorSpec(0.));
}

XLA_TEST_F(ConvertTest, DISABLED_ON_CPU(ConvertF32F8e5m2Roundtrip)) {
  // Convert from FP32 to FP8, then back to FP32.
  XlaBuilder builder(TestName());
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();

  struct TestCase {
    float input;
    float expected_roundtrip;
  } test_cases[] = {
      // clang-format off
      {0.0, 0.0},
      {-0.0, -0.0},
      {1.0, 1.0},
      {-1.0, -1.0},
      {nan, nan},
      {inf, inf},
      // clang-format on
      {0x1.2p0, 0x1p0},           // Round-to-even down
      {0x1.6p0, 0x1.8p0},         // Round-to-even up
      {0x1.Cp15, 0x1.Cp15},       // Max value
      {0x1.DFFFFEp15, 0x1.Cp15},  // Largest number that doesn't overflow
      {0x1.Ep15, inf},            // Smallest number that overflows
      {0x1p16, inf},              // Overflow
      {0x1p-14, 0x1p-14},         // Smallest F8 normal
      {0x1.Cp-15, 0x1p-14},       // Smallest number rounding up to normal

      // Denormal tests
      {0x1.0p-15, 0x0.8p-14},       // Denormal without rounding
      {0x1.4p-15, 0x0.8p-14},       // Round-to-even down
      {0x1.Cp-15, 0x1.0p-14},       // Round-to-even up
      {0x1.3p-15, 0x0.8p-14},       // Round-to-nearest down
      {0x1.5p-15, 0x0.Cp-14},       // Round-to-nearest up
      {0x1p-17, 0},                 // Largest number that underflows
      {0x1.000002p-17, 0x0.4p-14},  // Smallest number that doesn't underflow
      {0x1.BFFFFEp-15, 0x0.Cp-14},  // Largest number that rounds to denormal
  };

  std::vector<float> inputs;
  std::vector<float> expected_roundtrip;
  for (auto test_case : test_cases) {
    inputs.push_back(test_case.input);
    expected_roundtrip.push_back(test_case.expected_roundtrip);
  }

  auto f8 = ConvertElementType(ConstantR1<float>(&builder, inputs), F8E5M2);
  ConvertElementType(f8, F32);
  ComputeAndCompareR1<float>(&builder, expected_roundtrip, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e5m2RoundtripExhaustive) {
  // Convert from FP8 to supported floating point type, then back to FP8.
  XlaBuilder builder(this->TestName());

  using From = tsl::float8_e5m2;
  std::vector<From> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i)));
  }

  xla::XlaOp all_f8_as_fp =
      ConvertElementType(ConstantR1<From>(&builder, all_f8),
                         primitive_util::NativeToPrimitiveType<TypeParam>());
  ConvertElementType(all_f8_as_fp, F8E5M2);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e5m2RoundtripExhaustive2) {
  // Convert from supported floating point type to FP8.
  XlaBuilder builder(this->TestName());
  if constexpr (std::is_same_v<TypeParam, tsl::float8_e3m4>) {
    // TODO(b/370786669): Enable this test.
    GTEST_SKIP() << "Skipping test for E3M4 as it requires an ml_dtypes "
                    "release with https://github.com/jax-ml/ml_dtypes/pull/205";
  } else {
    std::vector<TypeParam> all_f8;
    for (int i = 0; i < 256; i++) {
      all_f8.push_back(static_cast<TypeParam>(
          Eigen::numext::bit_cast<tsl::float8_e5m2>(static_cast<uint8_t>(i))));
    }

    ConvertElementType(ConstantR1<TypeParam>(&builder, all_f8), F8E5M2);
    this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
  }
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e5m2RoundtripExhaustive3) {
  // Convert from FP8 to supported floating point type.
  XlaBuilder builder(this->TestName());

  using From = tsl::float8_e5m2;
  std::vector<From> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i)));
  }

  ConvertElementType(ConstantR1<From>(&builder, all_f8),
                     primitive_util::NativeToPrimitiveType<TypeParam>());
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestF16, ConvertF8e5m2F16RoundtripExhaustive4) {
  // Convert from (B)F16 to FP8.
  XlaBuilder builder(this->TestName());

  std::vector<TypeParam> inputs;
  for (int i = 0; i < 65536; i++) {
    inputs.push_back(
        Eigen::numext::bit_cast<TypeParam>(static_cast<uint16_t>(i)));
  }

  xla::XlaOp all_f16_to_f8 = ConstantR1<TypeParam>(&builder, inputs);
  ConvertElementType(all_f16_to_f8, F8E5M2);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

// ----- F8E4M3

XLA_TEST_F(ConvertTest, ConvertF16F8e4m3Roundtrip) {
  // Convert from FP16 to FP8, then back to FP16
  XlaBuilder builder(TestName());
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();

  struct TestCase {
    float input;
    float expected_roundtrip;
  } test_cases[] = {
      // clang-format off
      {0.0, 0.0},
      {-0.0, -0.0},
      {1.0, 1.0},
      {-1.0, -1.0},
      {nan, nan},
      {-nan, -nan},
      {inf, inf},
      {-inf, -inf},
      // clang-format on
      {0x1.1p0, 0x1p0},      // Round-to-even down
      {0x1.3p0, 0x1.4p0},    // Round-to-even up
      {0x1.Ep7, 0x1.Ep7},    // Max value
      {0x1.EFCp7, 0x1.Ep7},  // Largest number that doesn't overflow
      {0x1.Fp7, inf},        // Smallest number that overflows
      {0x1p8, inf},          // Overflow
      {0x1p-6, 0x1p-6},      // Smallest F8 normal
      {0x1.Ep-7, 0x1p-6},    // Smallest number rounding up to normal

      // Denormal tests
      {0x0.2p-6, 0x0.2p-6},     // Smallest denormal
      {0x0.Ep-6, 0x0.Ep-6},     // Largest denormal
      {0x0.8p-6, 0x0.8p-6},     // Denormal without rounding
      {0x0.9p-6, 0x0.8p-6},     // Round-to-even down
      {0x0.Fp-6, 0x0.8p-5},     // Round-to-even up
      {0x0.8Fp-6, 0x0.8p-6},    // Round-to-nearest down
      {0x0.91p-6, 0x0.Ap-6},    // Round-to-nearest up
      {0x1p-10, 0},             // Largest number that underflows
      {0x1.004p-10, 0x0.2p-6},  // Smallest number that doesn't underflow
      {0x0.EFCp-6, 0x0.Ep-6},   // Largest number that rounds to denormal
  };

  std::vector<Eigen::half> inputs;
  std::vector<Eigen::half> expected_roundtrip;
  for (auto test_case : test_cases) {
    inputs.push_back(Eigen::half{test_case.input});
    expected_roundtrip.push_back(Eigen::half{test_case.expected_roundtrip});
  }

  auto f8 =
      ConvertElementType(ConstantR1<Eigen::half>(&builder, inputs), F8E4M3);
  ConvertElementType(f8, F16);
  ComputeAndCompareR1<Eigen::half>(&builder, expected_roundtrip, {},
                                   ErrorSpec(0.));
}

XLA_TEST_F(ConvertTest, DISABLED_ON_CPU(ConvertF32F8e4m3Roundtrip)) {
  // Convert from FP32 to FP8, then back to FP32.
  XlaBuilder builder(TestName());
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();

  struct TestCase {
    float input;
    float expected_roundtrip;
  } test_cases[] = {
      // clang-format off
      {0.0, 0.0},
      {-0.0, -0.0},
      {1.0, 1.0},
      {-1.0, -1.0},
      {nan, nan},
      {-nan, -nan},
      {inf, inf},
      {-inf, -inf},
      // clang-format on
      {0x1.1p0, 0x1p0},         // Round-to-even down
      {0x1.3p0, 0x1.4p0},       // Round-to-even up
      {0x1.Ep7, 0x1.Ep7},       // Max value
      {0x1.EFFFFEp7, 0x1.Ep7},  // Largest number that doesn't overflow
      {0x1.Fp7, inf},           // Smallest number that overflows
      {0x1p8, inf},             // Overflow
      {0x1p-6, 0x1p-6},         // Smallest F8 normal
      {0x1.Ep-7, 0x1p-6},       // Smallest number rounding up to normal

      // Denormal tests
      {0x0.2p-6, 0x0.2p-6},        // Smallest denormal
      {0x0.Ep-6, 0x0.Ep-6},        // Largest denormal
      {0x0.8p-6, 0x0.8p-6},        // Denormal without rounding
      {0x0.9p-6, 0x0.8p-6},        // Round-to-even down
      {0x0.Fp-6, 0x0.8p-5},        // Round-to-even up
      {0x0.8Fp-6, 0x0.8p-6},       // Round-to-nearest down
      {0x0.91p-6, 0x0.Ap-6},       // Round-to-nearest up
      {0x1p-10, 0},                // Largest number that underflows
      {0x1.000002p-10, 0x0.2p-6},  // Smallest number that doesn't underflow
      {0x0.EFFFFEp-6, 0x0.Ep-6},   // Largest number that rounds to denormal
  };

  std::vector<float> inputs;
  std::vector<float> expected_roundtrip;
  for (auto test_case : test_cases) {
    inputs.push_back(test_case.input);
    expected_roundtrip.push_back(test_case.expected_roundtrip);
  }

  auto f8 = ConvertElementType(ConstantR1<float>(&builder, inputs), F8E4M3);
  ConvertElementType(f8, F32);
  ComputeAndCompareR1<float>(&builder, expected_roundtrip, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e4m3RoundtripExhaustive) {
  // Convert from FP8 to supported floating point type, then back to FP8.
  XlaBuilder builder(this->TestName());

  using From = tsl::float8_e4m3;
  std::vector<From> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i)));
  }

  xla::XlaOp all_f8_as_fp =
      ConvertElementType(ConstantR1<From>(&builder, all_f8),
                         primitive_util::NativeToPrimitiveType<TypeParam>());
  ConvertElementType(all_f8_as_fp, F8E4M3);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e4m3RoundtripExhaustive2) {
  // Convert from supported floating point type to FP8.
  XlaBuilder builder(this->TestName());

  std::vector<TypeParam> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(static_cast<TypeParam>(
        Eigen::numext::bit_cast<tsl::float8_e4m3>(static_cast<uint8_t>(i))));
  }

  ConvertElementType(ConstantR1<TypeParam>(&builder, all_f8), F8E4M3);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e4m3RoundtripExhaustive3) {
  // Convert from FP8 to supported floating point type.
  XlaBuilder builder(this->TestName());

  using From = tsl::float8_e4m3;
  std::vector<From> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i)));
  }

  ConvertElementType(ConstantR1<From>(&builder, all_f8),
                     primitive_util::NativeToPrimitiveType<TypeParam>());
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestF16, ConvertF8e4m3F16RoundtripExhaustive4) {
  // Convert from (B)F16 to FP8.
  XlaBuilder builder(this->TestName());

  std::vector<TypeParam> inputs;
  for (int i = 0; i < 65536; i++) {
    inputs.push_back(
        Eigen::numext::bit_cast<TypeParam>(static_cast<uint16_t>(i)));
  }

  xla::XlaOp all_f16_to_f8 = ConstantR1<TypeParam>(&builder, inputs);
  ConvertElementType(all_f16_to_f8, F8E4M3);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

// ----- F8E4M3FN

XLA_TEST_F(ConvertTest, ConvertF16F8e4m3fnRoundtrip) {
  // Convert from FP16 to FP8, then back to FP16.
  XlaBuilder builder(TestName());
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();

  struct TestCase {
    float input;
    float expected_roundtrip;
  } test_cases[] = {
      // clang-format off
      {0.0, 0.0},
      {-0.0, -0.0},
      {1.0, 1.0},
      {-1.0, -1.0},
      {inf, nan},
      // clang-format on
      {0x1.1p0, 0x1p0},    // Round-to-even down
      {0x1.3p0, 0x1.4p0},  // Round-to-even up
      {0x1.Cp8, 0x1.Cp8},  // Max value
      {0x1.Dp8, 0x1.Cp8},  // Largest number that doesn't overflow
      {0x1.D04p8, nan},    // Smallest number that overflows
      {0x1p9, nan},        // Overflow
      {0x1p-6, 0x1p-6},    // Smallest F8 normal
      {0x1.Ep-7, 0x1p-6},  // Smallest number rounding up to normal

      // Denormal tests
      {0x1.0p-8, 0x0.4p-6},     // Denormal without rounding
      {0x1.4p-8, 0x0.4p-6},     // Round-to-even down
      {0x1.Cp-8, 0x0.8p-6},     // Round-to-even up
      {0x1.3p-8, 0x0.4p-6},     // Round-to-nearest down
      {0x1.5p-8, 0x0.6p-6},     // Round-to-nearest up
      {0x1p-10, 0},             // Largest number that underflows
      {0x1.004p-10, 0x0.2p-6},  // Smallest number that doesn't underflow
      {0x1.DFCp-7, 0x0.Ep-6},   // Largest number that rounds to denormal
  };

  std::vector<Eigen::half> inputs;
  std::vector<Eigen::half> expected_roundtrip;
  for (auto test_case : test_cases) {
    inputs.push_back(Eigen::half{test_case.input});
    expected_roundtrip.push_back(Eigen::half{test_case.expected_roundtrip});
  }

  auto f8 =
      ConvertElementType(ConstantR1<Eigen::half>(&builder, inputs), F8E4M3FN);
  ConvertElementType(f8, F16);
  ComputeAndCompareR1<Eigen::half>(&builder, expected_roundtrip, {},
                                   ErrorSpec(0.));
}

XLA_TEST_F(ConvertTest, DISABLED_ON_CPU(ConvertF32F8e4m3fnRoundtrip)) {
  // Convert from FP32 to FP8, then back to FP32.
  XlaBuilder builder(TestName());
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();

  struct TestCase {
    float input;
    float expected_roundtrip;
  } test_cases[] = {
      // clang-format off
      {0.0, 0.0},
      {-0.0, -0.0},
      {1.0, 1.0},
      {-1.0, -1.0},
      {inf, nan},
      // clang-format on
      {0x1.1p0, 0x1p0},     // Round-to-even down
      {0x1.3p0, 0x1.4p0},   // Round-to-even up
      {0x1.Cp8, 0x1.Cp8},   // Max value
      {0x1.Dp8, 0x1.Cp8},   // Largest number that doesn't overflow
      {0x1.D00002p8, nan},  // Smallest number that overflows
      {0x1p9, nan},         // Overflow
      {0x1p-6, 0x1p-6},     // Smallest F8 normal
      {0x1.Ep-7, 0x1p-6},   // Smallest number rounding up to normal

      // Denormal tests
      {0x1.0p-8, 0x0.4p-6},        // Denormal without rounding
      {0x1.4p-8, 0x0.4p-6},        // Round-to-even down
      {0x1.Cp-8, 0x0.8p-6},        // Round-to-even up
      {0x1.3p-8, 0x0.4p-6},        // Round-to-nearest down
      {0x1.5p-8, 0x0.6p-6},        // Round-to-nearest up
      {0x1p-10, 0},                // Largest number that underflows
      {0x1.000002p-10, 0x0.2p-6},  // Smallest number that doesn't underflow
      {0x1.DFFFFEp-7, 0x0.Ep-6},   // Largest number that rounds to denormal
  };

  std::vector<float> inputs;
  std::vector<float> expected_roundtrip;
  for (auto test_case : test_cases) {
    inputs.push_back(test_case.input);
    expected_roundtrip.push_back(test_case.expected_roundtrip);
  }

  auto f8 = ConvertElementType(ConstantR1<float>(&builder, inputs), F8E4M3FN);
  ConvertElementType(f8, F32);
  ComputeAndCompareR1<float>(&builder, expected_roundtrip, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e4m3fnRoundtripExhaustive) {
  // Convert from FP8 to supported floating point type, then back to FP8.
  XlaBuilder builder(this->TestName());

  using From = tsl::float8_e4m3fn;
  std::vector<From> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i)));
  }

  xla::XlaOp all_f8_as_fp =
      ConvertElementType(ConstantR1<From>(&builder, all_f8),
                         primitive_util::NativeToPrimitiveType<TypeParam>());
  ConvertElementType(all_f8_as_fp, F8E4M3FN);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e4m3fnRoundtripExhaustive2) {
  // Convert from supported floating point type to FP8.
  XlaBuilder builder(this->TestName());

  std::vector<TypeParam> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(static_cast<TypeParam>(
        Eigen::numext::bit_cast<tsl::float8_e4m3fn>(static_cast<uint8_t>(i))));
  }

  ConvertElementType(ConstantR1<TypeParam>(&builder, all_f8), F8E4M3FN);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e4m3fnRoundtripExhaustive3) {
  // Convert from FP8 to supported floating point type.
  XlaBuilder builder(this->TestName());

  using From = tsl::float8_e4m3fn;
  std::vector<From> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i)));
  }

  ConvertElementType(ConstantR1<From>(&builder, all_f8),
                     primitive_util::NativeToPrimitiveType<TypeParam>());
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestF16, ConvertF8e4m3fnF16RoundtripExhaustive4) {
  // Convert from (B)F16 to FP8.
  XlaBuilder builder(this->TestName());

  std::vector<TypeParam> inputs;
  for (int i = 0; i < 65536; i++) {
    inputs.push_back(
        Eigen::numext::bit_cast<TypeParam>(static_cast<uint16_t>(i)));
  }

  xla::XlaOp all_f16_to_f8 = ConstantR1<TypeParam>(&builder, inputs);
  ConvertElementType(all_f16_to_f8, F8E4M3FN);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

// ----- F8E4M3B11FNUZ

XLA_TEST_F(ConvertTest, ConvertF16F8e4m3b11fnuzRoundtrip) {
  // Convert from FP16 to FP8, then back to FP16.
  XlaBuilder builder(TestName());
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();

  struct TestCase {
    float input;
    float expected_roundtrip;
  } test_cases[] = {
      // clang-format off
      {0.0, 0.0},
      {-0.0, 0.0},
      {1.0, 1.0},
      {-1.0, -1.0},
      {inf, nan},
      // clang-format on
      {0x1.1p0, 0x1p0},      // Round-to-even down
      {0x1.3p0, 0x1.4p0},    // Round-to-even up
      {0x1.Ep4, 0x1.Ep4},    // Max value
      {0x1.EFCp4, 0x1.Ep4},  // Largest number that doesn't overflow
      {0x1.Fp4, nan},        // Smallest number that overflows
      {0x1p5, nan},          // Overflow
      {0x1p-10, 0x1p-10},    // Smallest F8 normal
      {0x1.Ep-11, 0x1p-10},  // Smallest number rounding up to normal

      // Denormal tests
      {0x1.0p-12, 0x0.4p-10},    // Denormal without rounding
      {0x1.4p-12, 0x0.4p-10},    // Round-to-even down
      {0x1.Cp-12, 0x0.8p-10},    // Round-to-even up
      {0x1.3p-12, 0x0.4p-10},    // Round-to-nearest down
      {0x1.5p-12, 0x0.6p-10},    // Round-to-nearest up
      {0x1p-14, 0},              // Largest number that underflows
      {0x1.004p-14, 0x0.2p-10},  // Smallest number that doesn't underflow
      {0x1.DFCp-11, 0x0.Ep-10},  // Largest number that rounds to denormal
  };

  std::vector<Eigen::half> inputs;
  std::vector<Eigen::half> expected_roundtrip;
  for (auto test_case : test_cases) {
    inputs.push_back(Eigen::half{test_case.input});
    expected_roundtrip.push_back(Eigen::half{test_case.expected_roundtrip});
  }

  auto f8 = ConvertElementType(ConstantR1<Eigen::half>(&builder, inputs),
                               F8E4M3B11FNUZ);
  ConvertElementType(f8, F16);
  ComputeAndCompareR1<Eigen::half>(&builder, expected_roundtrip, {},
                                   ErrorSpec(0.));
}

XLA_TEST_F(ConvertTest, DISABLED_ON_CPU(ConvertF32F8e4m3b11fnuzRoundtrip)) {
  // Convert from FP32 to FP8, then back to FP32.
  XlaBuilder builder(TestName());
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();

  struct TestCase {
    float input;
    float expected_roundtrip;
  } test_cases[] = {
      // clang-format off
      {0.0, 0.0},
      {-0.0, 0.0},
      {1.0, 1.0},
      {-1.0, -1.0},
      {inf, nan},
      // clang-format on
      {0x1.1p0, 0x1p0},         // Round-to-even down
      {0x1.3p0, 0x1.4p0},       // Round-to-even up
      {0x1.Ep4, 0x1.Ep4},       // Max value
      {0x1.EFFFFEp4, 0x1.Ep4},  // Largest number that doesn't overflow
      {0x1.Fp4, nan},           // Smallest number that overflows
      {0x1p5, nan},             // Overflow
      {0x1p-10, 0x1p-10},       // Smallest F8 normal
      {0x1.Ep-11, 0x1p-10},     // Smallest number rounding up to normal

      // Denormal tests
      {0x1.0p-12, 0x0.4p-10},       // Denormal without rounding
      {0x1.4p-12, 0x0.4p-10},       // Round-to-even down
      {0x1.Cp-12, 0x0.8p-10},       // Round-to-even up
      {0x1.3p-12, 0x0.4p-10},       // Round-to-nearest down
      {0x1.5p-12, 0x0.6p-10},       // Round-to-nearest up
      {0x1p-14, 0},                 // Largest number that underflows
      {0x1.000002p-14, 0x0.2p-10},  // Smallest number that doesn't underflow
      {0x1.DFFFFEp-11, 0x0.Ep-10},  // Largest number that rounds to denormal
  };

  std::vector<float> inputs;
  std::vector<float> expected_roundtrip;
  for (auto test_case : test_cases) {
    inputs.push_back(test_case.input);
    expected_roundtrip.push_back(test_case.expected_roundtrip);
  }

  auto f8 =
      ConvertElementType(ConstantR1<float>(&builder, inputs), F8E4M3B11FNUZ);
  ConvertElementType(f8, F32);
  ComputeAndCompareR1<float>(&builder, expected_roundtrip, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e4m3b11fnuzRoundtripExhaustive) {
  // Convert from FP8 to supported floating point type, then back to FP8.
  XlaBuilder builder(this->TestName());

  using From = tsl::float8_e4m3b11fnuz;
  std::vector<From> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i)));
  }

  xla::XlaOp all_f8_as_fp =
      ConvertElementType(ConstantR1<From>(&builder, all_f8),
                         primitive_util::NativeToPrimitiveType<TypeParam>());
  ConvertElementType(all_f8_as_fp, F8E4M3B11FNUZ);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e4m3b11fnuzRoundtripExhaustive2) {
  // Convert from supported floating point type to FP8.
  XlaBuilder builder(this->TestName());

  std::vector<TypeParam> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(
        static_cast<TypeParam>(Eigen::numext::bit_cast<tsl::float8_e4m3b11fnuz>(
            static_cast<uint8_t>(i))));
  }

  ConvertElementType(ConstantR1<TypeParam>(&builder, all_f8), F8E4M3B11FNUZ);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e4m3b11fnuzRoundtripExhaustive3) {
  // Convert from FP8 to supported floating point type.
  XlaBuilder builder(this->TestName());

  using From = tsl::float8_e4m3b11fnuz;
  std::vector<From> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i)));
  }

  ConvertElementType(ConstantR1<From>(&builder, all_f8),
                     primitive_util::NativeToPrimitiveType<TypeParam>());
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestF16, ConvertF8e4m3b11fnuzF16RoundtripExhaustive4) {
  // Convert from (B)F16 to FP8.
  XlaBuilder builder(this->TestName());

  std::vector<TypeParam> all_f16;
  for (int i = 0; i < 65536; i++) {
    all_f16.push_back(
        Eigen::numext::bit_cast<TypeParam>(static_cast<uint16_t>(i)));
  }

  ConvertElementType(ConstantR1<TypeParam>(&builder, all_f16), F8E4M3B11FNUZ);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

// ----- F8E5M2FNUZ

XLA_TEST_F(ConvertTest, ConvertF16F8e5m2fnuzRoundtrip) {
  // Convert from FP16 to FP8, then back to FP16.
  XlaBuilder builder(TestName());
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();

  struct TestCase {
    float input;
    float expected_roundtrip;
  } test_cases[] = {
      // clang-format off
      {0.0, 0.0},
      {-0.0, 0.0},
      {1.0, 1.0},
      {-1.0, -1.0},
      {nan, nan},
      {inf, nan},
      // clang-format on
      {0x1.2p0, 0x1p0},        // Round-to-even down
      {0x1.6p0, 0x1.8p0},      // Round-to-even up
      {0x1.Cp15, 0x1.Cp15},    // Max value
      {0x1.DFCp15, 0x1.Cp15},  // Largest number that doesn't overflow
      {0x1.Ep15, nan},         // Smallest number that overflows
      {0x1p16, nan},           // Overflow
      {0x1p-15, 0x1p-15},      // Smallest F8 normal
      {0x1.Cp-16, 0x1p-15},    // Smallest number rounding up to normal

      // Denormal tests
      {0x0.4p-14, 0x0.8p-15},    // Denormal without rounding
      {0x0.5p-14, 0x0.8p-15},    // Round-to-even down
      {0x0.7p-14, 0x1.0p-15},    // Round-to-even up
      {0x0.4Cp-14, 0x0.8p-15},   // Round-to-nearest down
      {0x0.54p-14, 0x0.Cp-15},   // Round-to-nearest up
      {0x0.1p-14, 0},            // Largest number that underflows
      {0x0.104p-14, 0x0.4p-15},  // Smallest number that doesn't underflow
      {0x0.6FCp-14, 0x0.Cp-15},  // Largest number that rounds to denormal
  };

  std::vector<Eigen::half> inputs;
  std::vector<Eigen::half> expected_roundtrip;
  for (auto test_case : test_cases) {
    inputs.push_back(Eigen::half{test_case.input});
    expected_roundtrip.push_back(Eigen::half{test_case.expected_roundtrip});
  }

  auto f8 =
      ConvertElementType(ConstantR1<Eigen::half>(&builder, inputs), F8E5M2FNUZ);
  ConvertElementType(f8, F16);
  ComputeAndCompareR1<Eigen::half>(&builder, expected_roundtrip, {},
                                   ErrorSpec(0.));
}

XLA_TEST_F(ConvertTest, ConvertF32F8e5m2fnuzRoundtrip) {
  // Convert from FP32 to FP8, then back to FP32.
  XlaBuilder builder(TestName());
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();

  struct TestCase {
    float input;
    float expected_roundtrip;
  } test_cases[] = {
      // clang-format off
      {0.0, 0.0},
      {-0.0, 0.0},
      {1.0, 1.0},
      {-1.0, -1.0},
      {nan, nan},
      {inf, nan},
      // clang-format on
      {0x1.2p0, 0x1p0},           // Round-to-even down
      {0x1.6p0, 0x1.8p0},         // Round-to-even up
      {0x1.Cp15, 0x1.Cp15},       // Max value
      {0x1.DFFFFEp15, 0x1.Cp15},  // Largest number that doesn't overflow
      {0x1.Ep15, nan},            // Smallest number that overflows
      {0x1p16, nan},              // Overflow
      {0x1p-15, 0x1p-15},         // Smallest F8 normal
      {0x1.Cp-16, 0x1p-15},       // Smallest number rounding up to normal

      // Denormal tests
      {0x1.0p-16, 0x0.8p-15},       // Denormal without rounding
      {0x1.4p-16, 0x0.8p-15},       // Round-to-even down
      {0x1.Cp-16, 0x1.0p-15},       // Round-to-even up
      {0x1.3p-16, 0x0.8p-15},       // Round-to-nearest down
      {0x1.5p-16, 0x0.Cp-15},       // Round-to-nearest up
      {0x1p-18, 0},                 // Largest number that underflows
      {0x1.000002p-18, 0x0.4p-15},  // Smallest number that doesn't underflow
      {0x1.BFFFFEp-16, 0x0.Cp-15},  // Largest number that rounds to denormal
  };

  std::vector<float> inputs;
  std::vector<float> expected_roundtrip;
  for (auto test_case : test_cases) {
    inputs.push_back(test_case.input);
    expected_roundtrip.push_back(test_case.expected_roundtrip);
  }

  auto f8 = ConvertElementType(ConstantR1<float>(&builder, inputs), F8E5M2FNUZ);
  ConvertElementType(f8, F32);
  ComputeAndCompareR1<float>(&builder, expected_roundtrip, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e5m2fnuzRoundtripExhaustive) {
  // Convert from FP8 to supported floating point type, then back to FP8.
  XlaBuilder builder(this->TestName());

  using From = tsl::float8_e5m2fnuz;
  std::vector<From> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i)));
  }

  xla::XlaOp all_f8_as_fp =
      ConvertElementType(ConstantR1<From>(&builder, all_f8),
                         primitive_util::NativeToPrimitiveType<TypeParam>());
  ConvertElementType(all_f8_as_fp, F8E5M2FNUZ);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e5m2fnuzRoundtripExhaustive2) {
  // Convert from supported floating point type to FP8.
  XlaBuilder builder(this->TestName());

  if constexpr (std::is_same_v<TypeParam, tsl::float8_e3m4>) {
    // TODO(b/370786669): Enable this test.
    GTEST_SKIP() << "Skipping test for E3M4 as it requires an ml_dtypes "
                    "release with https://github.com/jax-ml/ml_dtypes/pull/205";
  } else {
    std::vector<TypeParam> all_f8;
    for (int i = 0; i < 256; i++) {
      all_f8.push_back(
          static_cast<TypeParam>(Eigen::numext::bit_cast<tsl::float8_e5m2fnuz>(
              static_cast<uint8_t>(i))));
    }

    ConvertElementType(ConstantR1<TypeParam>(&builder, all_f8), F8E5M2FNUZ);
    this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
  }
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e5m2fnuzRoundtripExhaustive3) {
  // Convert from FP8 to supported floating point type.
  XlaBuilder builder(this->TestName());

  using From = tsl::float8_e5m2fnuz;
  std::vector<From> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i)));
  }

  ConvertElementType(ConstantR1<From>(&builder, all_f8),
                     primitive_util::NativeToPrimitiveType<TypeParam>());
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestF16, ConvertF8e5m2fnuzF16RoundtripExhaustive4) {
  // Convert from (B)F16 to FP8.
  XlaBuilder builder(this->TestName());

  std::vector<TypeParam> all_f16;
  for (int i = 0; i < 65536; i++) {
    all_f16.push_back(
        Eigen::numext::bit_cast<TypeParam>(static_cast<uint16_t>(i)));
  }

  ConvertElementType(ConstantR1<TypeParam>(&builder, all_f16), F8E5M2FNUZ);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

// ----- F8E4M3FNUZ

XLA_TEST_F(ConvertTest, ConvertF16F8e4m3fnuzRoundtrip) {
  // Convert from FP16 to FP8, then back to FP16.
  XlaBuilder builder(TestName());
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();

  struct TestCase {
    float input;
    float expected_roundtrip;
  } test_cases[] = {
      // clang-format off
      {0.0, 0.0},
      {-0.0, 0.0},
      {1.0, 1.0},
      {-1.0, -1.0},
      {inf, nan},
      // clang-format on
      {0x1.1p0, 0x1p0},      // Round-to-even down
      {0x1.3p0, 0x1.4p0},    // Round-to-even up
      {0x1.Ep7, 0x1.Ep7},    // Max value
      {0x1.EFCp7, 0x1.Ep7},  // Largest number that doesn't overflow
      {0x1.Fp7, nan},        // Smallest number that overflows
      {0x1p8, nan},          // Overflow
      {0x1p-7, 0x1p-7},      // Smallest F8 normal
      {0x1.Ep-8, 0x1p-7},    // Smallest number rounding up to normal

      // Denormal tests
      {0x1.0p-9, 0x0.4p-7},     // Denormal without rounding
      {0x1.4p-9, 0x0.4p-7},     // Round-to-even down
      {0x1.Cp-9, 0x0.8p-7},     // Round-to-even up
      {0x1.3p-9, 0x0.4p-7},     // Round-to-nearest down
      {0x1.5p-9, 0x0.6p-7},     // Round-to-nearest up
      {0x1p-11, 0},             // Largest number that underflows
      {0x1.004p-11, 0x0.2p-7},  // Smallest number that doesn't underflow
      {0x1.DFCp-8, 0x0.Ep-7},   // Largest number that rounds to denormal
  };

  std::vector<Eigen::half> inputs;
  std::vector<Eigen::half> expected_roundtrip;
  for (auto test_case : test_cases) {
    inputs.push_back(Eigen::half{test_case.input});
    expected_roundtrip.push_back(Eigen::half{test_case.expected_roundtrip});
  }

  auto f8 =
      ConvertElementType(ConstantR1<Eigen::half>(&builder, inputs), F8E4M3FNUZ);
  ConvertElementType(f8, F16);
  ComputeAndCompareR1<Eigen::half>(&builder, expected_roundtrip, {},
                                   ErrorSpec(0.));
}

XLA_TEST_F(ConvertTest, ConvertF32F8e4m3fnuzRoundtrip) {
  // Convert from FP32 to FP8, then back to FP32.
  XlaBuilder builder(TestName());
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();

  struct TestCase {
    float input;
    float expected_roundtrip;
  } test_cases[] = {
      // clang-format off
      {0.0, 0.0},
      {-0.0, 0.0},
      {1.0, 1.0},
      {-1.0, -1.0},
      {inf, nan},
      // clang-format on
      {0x1.1p0, 0x1p0},         // Round-to-even down
      {0x1.3p0, 0x1.4p0},       // Round-to-even up
      {0x1.Ep7, 0x1.Ep7},       // Max value
      {0x1.EFFFFEp7, 0x1.Ep7},  // Largest number that doesn't overflow
      {0x1.Fp7, nan},           // Smallest number that overflows
      {0x1p8, nan},             // Overflow
      {0x1p-7, 0x1p-7},         // Smallest F8 normal
      {0x1.Ep-8, 0x1p-7},       // Smallest number rounding up to normal

      // Denormal tests
      {0x1.0p-9, 0x0.4p-7},        // Denormal without rounding
      {0x1.4p-9, 0x0.4p-7},        // Round-to-even down
      {0x1.Cp-9, 0x0.8p-7},        // Round-to-even up
      {0x1.3p-9, 0x0.4p-7},        // Round-to-nearest down
      {0x1.5p-9, 0x0.6p-7},        // Round-to-nearest up
      {0x1p-11, 0},                // Largest number that underflows
      {0x1.000002p-11, 0x0.2p-7},  // Smallest number that doesn't underflow
      {0x1.DFFFFEp-8, 0x0.Ep-7},   // Largest number that rounds to denormal
  };

  std::vector<float> inputs;
  std::vector<float> expected_roundtrip;
  for (auto test_case : test_cases) {
    inputs.push_back(test_case.input);
    expected_roundtrip.push_back(test_case.expected_roundtrip);
  }

  auto f8 = ConvertElementType(ConstantR1<float>(&builder, inputs), F8E4M3FNUZ);
  ConvertElementType(f8, F32);
  ComputeAndCompareR1<float>(&builder, expected_roundtrip, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e4m3fnuzRoundtripExhaustive) {
  // Convert from FP8 to supported floating point type, then back to FP8.
  XlaBuilder builder(this->TestName());

  using From = tsl::float8_e4m3fnuz;
  std::vector<From> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i)));
  }

  xla::XlaOp all_f8_as_fp =
      ConvertElementType(ConstantR1<From>(&builder, all_f8),
                         primitive_util::NativeToPrimitiveType<TypeParam>());
  ConvertElementType(all_f8_as_fp, F8E4M3FNUZ);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e4m3fnuzRoundtripExhaustive2) {
  // Convert from supported floating point type to FP8.
  XlaBuilder builder(this->TestName());

  std::vector<TypeParam> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(
        static_cast<TypeParam>(Eigen::numext::bit_cast<tsl::float8_e4m3fnuz>(
            static_cast<uint8_t>(i))));
  }

  ConvertElementType(ConstantR1<TypeParam>(&builder, all_f8), F8E4M3FNUZ);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e4m3fnuzRoundtripExhaustive3) {
  // Convert from FP8 to supported floating point type.
  XlaBuilder builder(this->TestName());

  using From = tsl::float8_e4m3fnuz;
  std::vector<From> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i)));
  }

  ConvertElementType(ConstantR1<From>(&builder, all_f8),
                     primitive_util::NativeToPrimitiveType<TypeParam>());
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestF16, ConvertF8e4m3fnuzF16RoundtripExhaustive4) {
  // Convert from (B)F16 to FP8.
  XlaBuilder builder(this->TestName());

  std::vector<TypeParam> all_f16;
  for (int i = 0; i < 65536; i++) {
    all_f16.push_back(
        Eigen::numext::bit_cast<TypeParam>(static_cast<uint16_t>(i)));
  }

  ConvertElementType(ConstantR1<TypeParam>(&builder, all_f16), F8E4M3FNUZ);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

// ----- F8E3M4

XLA_TEST_F(ConvertTest, ConvertF16F8e3m4Roundtrip) {
  // Convert from FP16 to FP8, then back to FP16
  XlaBuilder builder(TestName());
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();

  struct TestCase {
    float input;
    float expected_roundtrip;
  } test_cases[] = {
      // clang-format off
      {0.0, 0.0},
      {-0.0, -0.0},
      {1.0, 1.0},
      {-1.0, -1.0},
      {nan, nan},
      {-nan, -nan},
      {inf, inf},
      {-inf, -inf},
      // clang-format on
      {0x1.08p0, 0x1p0},     // Round-to-even down
      {0x1.18p0, 0x1.2p0},   // Round-to-even up
      {0x1.Fp3, 0x1.Fp3},    // Max value
      {0x1.F7Cp3, 0x1.Fp3},  // Largest number that doesn't overflow
      {0x1.F8p3, inf},       // Smallest number that overflows
      {0x1p4, inf},          // Overflow
      {0x1p-2, 0x1p-2},      // Smallest F8 normal
      {0x1.Fp-3, 0x1p-2},    // Smallest number rounding up to normal

      // Denormal tests
      {0x0.1p-2, 0x0.1p-2},    // Smallest denormal
      {0x0.Fp-2, 0x0.Fp-2},    // Largest denormal
      {0x0.8p-2, 0x0.8p-2},    // Denormal without rounding
      {0x0.88p-2, 0x0.8p-2},   // Round-to-even down
      {0x0.F8p-2, 0x0.8p-1},   // Round-to-even up
      {0x0.87p-2, 0x0.8p-2},   // Round-to-nearest down
      {0x0.89p-2, 0x0.9p-2},   // Round-to-nearest up
      {0x1p-7, 0},             // Largest number that underflows
      {0x1.004p-7, 0x0.1p-2},  // Smallest number that doesn't underflow
      {0x0.F7Cp-2, 0x0.Fp-2},  // Largest number that rounds to denormal
  };

  std::vector<Eigen::half> inputs;
  std::vector<Eigen::half> expected_roundtrip;
  for (auto test_case : test_cases) {
    inputs.push_back(Eigen::half{test_case.input});
    expected_roundtrip.push_back(Eigen::half{test_case.expected_roundtrip});
  }

  auto f8 =
      ConvertElementType(ConstantR1<Eigen::half>(&builder, inputs), F8E3M4);
  ConvertElementType(f8, F16);
  ComputeAndCompareR1<Eigen::half>(&builder, expected_roundtrip, {},
                                   ErrorSpec(0.));
}

XLA_TEST_F(ConvertTest, DISABLED_ON_CPU(ConvertF32F8e3m4Roundtrip)) {
  // Convert from FP32 to FP8, then back to FP32.
  XlaBuilder builder(TestName());
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();

  struct TestCase {
    float input;
    float expected_roundtrip;
  } test_cases[] = {
      // clang-format off
      {0.0, 0.0},
      {-0.0, -0.0},
      {1.0, 1.0},
      {-1.0, -1.0},
      {nan, nan},
      {-nan, -nan},
      {inf, inf},
      {-inf, -inf},
      // clang-format on
      {0x1.08p0, 0x1p0},        // Round-to-even down
      {0x1.18p0, 0x1.2p0},      // Round-to-even up
      {0x1.Fp3, 0x1.Fp3},       // Max value
      {0x1.F7FFFEp3, 0x1.Fp3},  // Largest number that doesn't overflow
      {0x1.F8p3, inf},          // Smallest number that overflows
      {0x1p4, inf},             // Overflow
      {0x1p-2, 0x1p-2},         // Smallest F8 normal
      {0x1.Fp-3, 0x1p-2},       // Smallest number rounding up to normal

      // Denormal tests
      {0x0.1p-2, 0x0.1p-2},       // Smallest denormal
      {0x0.Fp-2, 0x0.Fp-2},       // Largest denormal
      {0x0.8p-2, 0x0.8p-2},       // Denormal without rounding
      {0x0.88p-2, 0x0.8p-2},      // Round-to-even down
      {0x0.F8p-2, 0x0.8p-1},      // Round-to-even up
      {0x0.87p-2, 0x0.8p-2},      // Round-to-nearest down
      {0x0.89p-2, 0x0.9p-2},      // Round-to-nearest up
      {0x1p-7, 0},                // Largest number that underflows
      {0x1.000002p-7, 0x0.1p-2},  // Smallest number that doesn't underflow
      {0x0.F7FFFEp-2, 0x0.Fp-2},  // Largest number that rounds to denormal
  };

  std::vector<float> inputs;
  std::vector<float> expected_roundtrip;
  for (auto test_case : test_cases) {
    inputs.push_back(test_case.input);
    expected_roundtrip.push_back(test_case.expected_roundtrip);
  }

  auto f8 = ConvertElementType(ConstantR1<float>(&builder, inputs), F8E3M4);
  ConvertElementType(f8, F32);
  ComputeAndCompareR1<float>(&builder, expected_roundtrip, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e3m4RoundtripExhaustive) {
  // Convert from FP8 to supported floating point type, then back to FP8.
  XlaBuilder builder(this->TestName());

  using From = tsl::float8_e3m4;
  std::vector<From> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i)));
  }

  xla::XlaOp all_f8_as_fp =
      ConvertElementType(ConstantR1<From>(&builder, all_f8),
                         primitive_util::NativeToPrimitiveType<TypeParam>());
  ConvertElementType(all_f8_as_fp, F8E3M4);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e3m4RoundtripExhaustive2) {
  // Convert from supported floating point type to FP8.
  XlaBuilder builder(this->TestName());

  std::vector<TypeParam> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(static_cast<TypeParam>(
        Eigen::numext::bit_cast<tsl::float8_e3m4>(static_cast<uint8_t>(i))));
  }

  ConvertElementType(ConstantR1<TypeParam>(&builder, all_f8), F8E3M4);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestT, ConvertF8e3m4RoundtripExhaustive3) {
  // Convert from FP8 to supported floating point type.
  XlaBuilder builder(this->TestName());

  using From = tsl::float8_e3m4;
  std::vector<From> all_f8;
  for (int i = 0; i < 256; i++) {
    all_f8.push_back(Eigen::numext::bit_cast<From>(static_cast<uint8_t>(i)));
  }

  ConvertElementType(ConstantR1<From>(&builder, all_f8),
                     primitive_util::NativeToPrimitiveType<TypeParam>());
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

XLA_TYPED_TEST(ConvertTestF16, ConvertF8e3m4F16RoundtripExhaustive4) {
  // Convert from (B)F16 to FP8.
  XlaBuilder builder(this->TestName());

  std::vector<TypeParam> inputs;
  for (int i = 0; i < 65536; i++) {
    inputs.push_back(
        Eigen::numext::bit_cast<TypeParam>(static_cast<uint16_t>(i)));
  }

  xla::XlaOp all_f16_to_f8 = ConstantR1<TypeParam>(&builder, inputs);
  ConvertElementType(all_f16_to_f8, F8E3M4);
  this->ComputeAndCompare(&builder, {}, ErrorSpec(0.));
}

}  // namespace
}  // namespace xla
