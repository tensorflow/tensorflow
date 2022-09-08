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

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class ConvertTest : public ClientLibraryTestBase {
 public:
  explicit ConvertTest(se::Platform* platform = nullptr)
      : ClientLibraryTestBase(platform) {
    mutable_debug_options()->add_xla_disable_hlo_passes("algsimp");
    mutable_debug_options()->add_xla_disable_hlo_passes("inline");
  }
};

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
  ComputeAndCompareR1<uint32_t>(&builder, expected, {});
}

}  // namespace
}  // namespace xla
