/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/transforms/dot_algorithm_rewriter.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;
using ::absl_testing::IsOkAndHolds;
using ::testing::_;
using ::testing::AllOf;
using ::testing::Each;
using ::testing::Not;

class RewriteFp8xNDotTest : public HloHardwareIndependentTestBase {
 protected:
  // Runs RewriteFp8xNDot on the single dot of `hlo` and returns the module.
  std::unique_ptr<VerifiedHloModule> Rewrite(
      absl::string_view hlo, PrecisionConfig::Precision precision,
      bool expect_changed = true) {
    absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
        ParseAndReturnVerifiedModule(hlo);
    CHECK_OK(module.status());
    HloInstruction* dot = nullptr;
    for (HloInstruction* instr :
         (*module)->entry_computation()->instructions()) {
      if (instr->opcode() == HloOpcode::kDot) {
        dot = instr;
      }
    }
    CHECK(dot != nullptr);
    EXPECT_THAT(RewriteFp8xNDot(dot, precision), IsOkAndHolds(expect_changed));
    return *std::move(module);
  }

  // Evaluates `module` on `args` and returns the result as F64 values.
  std::vector<double> Evaluate(const HloModule& module,
                               absl::Span<const Literal* const> args) {
    absl::StatusOr<Literal> result = HloEvaluator().Evaluate(module, args);
    CHECK_OK(result.status());
    absl::StatusOr<Literal> f64 = result->Convert(F64);
    CHECK_OK(f64.status());
    absl::Span<const double> data = f64->data<double>();
    return std::vector<double>(data.begin(), data.end());
  }
};

TEST_F(RewriteFp8xNDotTest, IgnoresOtherAlgorithmsAndOperandTypes) {
  for (auto [lhs_type, algorithm] : {std::tuple{"f32", "dot_bf16_bf16_f32"},
                                     {"f16", "dot_bf16_bf16_fp8x3"},
                                     {"f8e4m3fn", "dot_bf16_bf16_fp8x4"}}) {
    std::unique_ptr<VerifiedHloModule> module =
        Rewrite(absl::StrFormat(R"(
HloModule m
ENTRY e {
  a = %s[8,64] parameter(0)
  b = bf16[64,16] parameter(1)
  ROOT d = f32[8,16] dot(a, b), lhs_contracting_dims={1},
    rhs_contracting_dims={0}, algorithm=%s
})",
                                lhs_type, algorithm),
                PrecisionConfig::DEFAULT, /*expect_changed=*/false);
    EXPECT_THAT(module->entry_computation()->root_instruction(), op::Dot());
  }
}

class RewriteFp8xNDotStructureTest
    : public RewriteFp8xNDotTest,
      public ::testing::WithParamInterface<
          std::tuple<PrecisionConfig::Algorithm, PrecisionConfig::Precision>> {
};

TEST_P(RewriteFp8xNDotStructureTest, EmitsF8DotsSummedSmallestFirst) {
  const auto [algorithm, precision] = GetParam();
  const bool is_x4 = algorithm == PrecisionConfig::ALG_DOT_BF16_BF16_FP8X4;
  std::unique_ptr<VerifiedHloModule> module =
      Rewrite(absl::StrFormat(R"(
HloModule m
ENTRY e {
  a = f32[8,64] parameter(0)
  b = bf16[64,16] parameter(1)
  ROOT d = f32[8,16] dot(a, b), lhs_contracting_dims={1},
    rhs_contracting_dims={0}, algorithm=%s
})",
                              AlgorithmToString(algorithm)),
              precision);

  int num_dots = 0;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() != HloOpcode::kDot) {
      continue;
    }
    ++num_dots;
    EXPECT_EQ(instr->shape().element_type(), F32);
    EXPECT_EQ(instr->operand(0)->shape().element_type(), F8E4M3FN);
    EXPECT_EQ(instr->operand(1)->shape().element_type(), F8E4M3FN);
    EXPECT_EQ(instr->precision_config().algorithm(),
              PrecisionConfig::ALG_UNSET);
    EXPECT_THAT(instr->precision_config().operand_precision(), Each(precision));
  }
  EXPECT_EQ(num_dots, is_x4 ? 4 : 3);

  // The high slice quantizes the scaled operand; the low slice quantizes the
  // 16x scaled residual.
  auto high = op::Convert(op::Multiply(Not(op::Subtract()), _));
  auto low = op::Convert(op::Multiply(op::Subtract(), _));
  auto cross_terms =
      op::Multiply(op::Add(op::Dot(low, high), op::Dot(high, low)), _);
  auto p00 = op::Dot(high, high);
  auto descale = [](auto sum) { return op::Multiply(op::Multiply(sum, _), _); };
  if (is_x4) {
    auto p11 = op::Multiply(op::Dot(low, low), _);
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                descale(op::Add(op::Add(p11, cross_terms), p00)));
  } else {
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                descale(op::Add(cross_terms, p00)));
  }
}

INSTANTIATE_TEST_SUITE_P(
    AlgorithmsAndPrecisions, RewriteFp8xNDotStructureTest,
    ::testing::Combine(
        ::testing::Values(PrecisionConfig::ALG_DOT_BF16_BF16_FP8X3,
                          PrecisionConfig::ALG_DOT_BF16_BF16_FP8X4),
        ::testing::Values(PrecisionConfig::DEFAULT, PrecisionConfig::HIGHEST)));

// lhs_batch_dims={1} maps the LHS scale [4,2] to output dims {1,0}, which
// needs the transpose in BroadcastScaleToDotOutput.
TEST_F(RewriteFp8xNDotTest, BatchedDotWithZeroChannelMatchesF64Reference) {
  constexpr absl::string_view kDims =
      "lhs_batch_dims={1}, rhs_batch_dims={0}, lhs_contracting_dims={2}, "
      "rhs_contracting_dims={1}";
  Literal lhs(ShapeUtil::MakeShape(F32, {4, 2, 64}));
  ASSERT_OK(lhs.Populate<float>([](absl::Span<const int64_t> idx) {
    // Row m=0 of batch 1 is all zero, so its amax hits the zero channel clamp.
    if (idx[0] == 0 && idx[1] == 1) {
      return 0.0f;
    }
    const int64_t i = (idx[0] * 2 + idx[1]) * 64 + idx[2];
    return static_cast<float>(std::sin(1.7 * i + 0.3) * (1 + i % 5));
  }));
  Literal rhs(ShapeUtil::MakeShape(BF16, {2, 64, 3}));
  ASSERT_OK(rhs.Populate<bfloat16>([](absl::Span<const int64_t> idx) {
    const int64_t i = (idx[0] * 64 + idx[1]) * 3 + idx[2];
    return bfloat16(static_cast<float>(std::cos(2.3 * i + 0.1)));
  }));
  const Literal* args[] = {&lhs, &rhs};

  ASSERT_OK_AND_ASSIGN(auto ref_module,
                       ParseAndReturnVerifiedModule(absl::StrFormat(R"(
HloModule m
ENTRY e {
  a = f32[4,2,64] parameter(0)
  b = bf16[2,64,3] parameter(1)
  a64 = f64[4,2,64] convert(a)
  b64 = f64[2,64,3] convert(b)
  ROOT d = f64[2,4,3] dot(a64, b64), %s
})",
                                                                    kDims)));
  const std::vector<double> ref = Evaluate(*ref_module, args);
  auto mse = [&](absl::Span<const double> actual) {
    double sum = 0;
    for (int64_t i = 0; i < ref.size(); ++i) {
      sum += (actual[i] - ref[i]) * (actual[i] - ref[i]);
    }
    return sum / ref.size();
  };

  double mse_fp8x[2];
  for (absl::string_view out_type : {"f32", "bf16"}) {
    for (int x : {3, 4}) {
      std::unique_ptr<VerifiedHloModule> module =
          Rewrite(absl::StrFormat(R"(
HloModule m
ENTRY e {
  a = f32[4,2,64] parameter(0)
  b = bf16[2,64,3] parameter(1)
  ROOT d = %s[2,4,3] dot(a, b), %s, algorithm=dot_bf16_bf16_fp8x%d
})",
                                  out_type, kDims, x),
                  PrecisionConfig::DEFAULT);
      if (out_type == "bf16") {
        EXPECT_THAT(module->entry_computation()->root_instruction(),
                    AllOf(op::Shape("bf16[2,4,3]"), op::Convert()));
      }
      EXPECT_TRUE(absl::c_any_of(module->entry_computation()->instructions(),
                                 [](const HloInstruction* instr) {
                                   return instr->opcode() ==
                                          HloOpcode::kTranspose;
                                 }));

      const std::vector<double> out = Evaluate(*module, args);
      for (int64_t i = 0; i < out.size(); ++i) {
        EXPECT_TRUE(std::isfinite(out[i])) << i;
      }
      // Output [batch=1, m=0, :] comes from the all zero LHS row.
      for (int64_t n = 0; n < 3; ++n) {
        EXPECT_EQ(out[n + 1 * 4 * 3], 0.0) << n;
      }
      if (out_type == "f32") {
        mse_fp8x[x - 3] = mse(out);
      }
    }
  }

  ASSERT_OK_AND_ASSIGN(auto bf16_module,
                       ParseAndReturnVerifiedModule(absl::StrFormat(R"(
HloModule m
ENTRY e {
  p = f32[4,2,64] parameter(0)
  b = bf16[2,64,3] parameter(1)
  a = bf16[4,2,64] convert(p)
  ROOT d = f32[2,4,3] dot(a, b), %s
})",
                                                                    kDims)));
  const double mse_bf16 = mse(Evaluate(*bf16_module, args));
  // Both algorithms drop the F32 LHS bits below the two F8 slices, so they beat
  // a BF16 dot only by rounding less. FP8X4 also adds A_low * B_low.
  EXPECT_LT(mse_fp8x[1], mse_fp8x[0]);
  EXPECT_LT(mse_fp8x[0], mse_bf16);
}

}  // namespace
}  // namespace xla
