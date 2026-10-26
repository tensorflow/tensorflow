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

#include "xla/backends/gpu/transforms/dot_algorithm_rewriter.h"

#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/literal_test_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class DotAlgorithmRewriterTest : public HloHardwareIndependentTestBase {};

TEST_F(DotAlgorithmRewriterTest, DefaultToBF16) {
  const char* hlo_text = R"hlo(
    HloModule test

    ENTRY test {
      p0 = f32[32,32] parameter(0)
      p1 = f32[32,32] parameter(1)
      ROOT dot = f32[32,32] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )hlo";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_default_to_alg_dot_bf16_bf16_f32(true);
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_text, config));
  ASSERT_OK_AND_ASSIGN(
      auto pass_result,
      RunHloPass(DotAlgorithmRewriter(
                     stream_executor::CudaComputeCapability::Hopper()),
                 module.get()));
  EXPECT_TRUE(pass_result);

  const char* expected = R"(
      CHECK: %[[P0:.*]] = f32[32,32]{{.*}} parameter(0)
      CHECK: %[[P0_BF16:.*]] = bf16[32,32]{{.*}} convert(%[[P0]])
      CHECK: %[[P1:.*]] = f32[32,32]{{.*}} parameter(1)
      CHECK: %[[P1_BF16:.*]] = bf16[32,32]{{.*}} convert(%[[P1]])
      CHECK: ROOT %dot.1 = f32[32,32]{{.*}} dot(%[[P0_BF16]], %[[P1_BF16]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_f32
  )";

  ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                       RunFileCheck(module->ToString(), expected));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(DotAlgorithmRewriterTest, NoDefaultToBF16) {
  const char* hlo_text = R"hlo(
    HloModule test

    ENTRY test {
      p0 = f32[32,32] parameter(0)
      p1 = f32[32,32] parameter(1)
      ROOT dot = f32[32,32] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )hlo";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_default_to_alg_dot_bf16_bf16_f32(false);
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_text, config));

  ASSERT_OK_AND_ASSIGN(
      bool pass_result,
      RunHloPass(DotAlgorithmRewriter(
                     stream_executor::CudaComputeCapability::Hopper()),
                 module.get()));
  EXPECT_FALSE(pass_result);
}

TEST_F(DotAlgorithmRewriterTest, DefaultToBF16_NonF32Result) {
  const char* hlo_text = R"hlo(
    HloModule test

    ENTRY test {
      p0 = f32[32,32] parameter(0)
      p1 = f32[32,32] parameter(1)
      ROOT dot = f64[32,32] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )hlo";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_default_to_alg_dot_bf16_bf16_f32(true);
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_text, config));
  ASSERT_OK_AND_ASSIGN(
      auto pass_result,
      RunHloPass(DotAlgorithmRewriter(
                     stream_executor::CudaComputeCapability::Hopper()),
                 module.get()));
  EXPECT_FALSE(pass_result);
}

TEST_F(DotAlgorithmRewriterTest, DefaultToBF16_NonF32Operands) {
  const char* hlo_text = R"hlo(
    HloModule test

    ENTRY test {
      p0 = bf16[32,32] parameter(0)
      p1 = bf16[32,32] parameter(1)
      ROOT dot = f32[32,32] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )hlo";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_default_to_alg_dot_bf16_bf16_f32(true);
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_text, config));
  ASSERT_OK_AND_ASSIGN(
      auto pass_result,
      RunHloPass(DotAlgorithmRewriter(
                     stream_executor::CudaComputeCapability::Hopper()),
                 module.get()));
  EXPECT_FALSE(pass_result);
}

TEST_F(DotAlgorithmRewriterTest, DefaultToBF16_HighestPrecision) {
  const char* hlo_text = R"hlo(
    HloModule test

    ENTRY test {
      p0 = f32[32,32] parameter(0)
      p1 = f32[32,32] parameter(1)
      ROOT dot = f32[32,32] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0},
        operand_precision={highest,highest}
    }
  )hlo";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_default_to_alg_dot_bf16_bf16_f32(true);
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_text, config));
  ASSERT_OK_AND_ASSIGN(
      auto pass_result,
      RunHloPass(DotAlgorithmRewriter(
                     stream_executor::CudaComputeCapability::Hopper()),
                 module.get()));
  EXPECT_FALSE(pass_result);
}

TEST_F(DotAlgorithmRewriterTest, SkipOnP100) {
  const char* hlo_text = R"hlo(
    HloModule test

    ENTRY test {
      p0 = f32[32,32] parameter(0)
      p1 = f32[32,32] parameter(1)
      ROOT dot = f32[32,32] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )hlo";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_default_to_alg_dot_bf16_bf16_f32(true);
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_text, config));
  ASSERT_OK_AND_ASSIGN(
      auto pass_result,
      RunHloPass(DotAlgorithmRewriter(
                     stream_executor::CudaComputeCapability::Pascal()),
                 module.get()));
  EXPECT_FALSE(pass_result);
}

TEST_F(DotAlgorithmRewriterTest, SkipOnV100) {
  const char* hlo_text = R"hlo(
    HloModule test

    ENTRY test {
      p0 = f32[32,32] parameter(0)
      p1 = f32[32,32] parameter(1)
      ROOT dot = f32[32,32] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )hlo";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_default_to_alg_dot_bf16_bf16_f32(true);
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_text, config));
  ASSERT_OK_AND_ASSIGN(
      auto pass_result,
      RunHloPass(
          DotAlgorithmRewriter(stream_executor::CudaComputeCapability::Volta()),
          module.get()));
  EXPECT_FALSE(pass_result);
}

TEST_F(DotAlgorithmRewriterTest, SkipOnTuring) {
  const char* hlo_text = R"hlo(
    HloModule test

    ENTRY test {
      p0 = f32[32,32] parameter(0)
      p1 = f32[32,32] parameter(1)
      ROOT dot = f32[32,32] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )hlo";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_default_to_alg_dot_bf16_bf16_f32(true);
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_text, config));
  ASSERT_OK_AND_ASSIGN(
      auto pass_result,
      RunHloPass(
          DotAlgorithmRewriter(stream_executor::CudaComputeCapability{7, 5}),
          module.get()));
  EXPECT_FALSE(pass_result);
}

TEST_F(DotAlgorithmRewriterTest, RunOnA100) {
  const char* hlo_text = R"hlo(
    HloModule test

    ENTRY test {
      p0 = f32[32,32] parameter(0)
      p1 = f32[32,32] parameter(1)
      ROOT dot = f32[32,32] dot(p0, p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )hlo";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_default_to_alg_dot_bf16_bf16_f32(true);
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_text, config));
  ASSERT_OK_AND_ASSIGN(
      auto pass_result,
      RunHloPass(DotAlgorithmRewriter(
                     stream_executor::CudaComputeCapability::Ampere()),
                 module.get()));
  EXPECT_TRUE(pass_result);
}

TEST_F(DotAlgorithmRewriterTest, Fp8xNOnlyRewritesOnlyFp8xNDots) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      p0 = f32[16,32] parameter(0)
      p1 = f32[32,16] parameter(1)
      bf16_dot = f32[16,16] dot(p0, p1), lhs_contracting_dims={1},
        rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_f32
      fp8_dot = f32[16,16] dot(p0, p1), lhs_contracting_dims={1},
        rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_fp8x3
      ROOT add = f32[16,16] add(bf16_dot, fp8_dot)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(RunHloPass(DotAlgorithmRewriter(
                             stream_executor::CudaComputeCapability::Hopper(),
                             /*fp8xn_only=*/true),
                         module.get()),
              absl_testing::IsOkAndHolds(true));
  int f32_dots = 0;
  int f8_dots = 0;
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    if (inst->opcode() != HloOpcode::kDot) {
      continue;
    }
    if (inst->operand(0)->shape().element_type() == F8E4M3FN) {
      EXPECT_EQ(inst->precision_config().algorithm(),
                PrecisionConfig::ALG_UNSET);
      // HIGHEST disables FP8 fast accumulation in cuBLASLt and Triton.
      EXPECT_THAT(inst->precision_config().operand_precision(),
                  ::testing::Each(PrecisionConfig::HIGHEST));
      ++f8_dots;
    } else {
      EXPECT_EQ(inst->name(), "bf16_dot");
      EXPECT_EQ(inst->operand(0)->shape().element_type(), F32);
      EXPECT_EQ(inst->precision_config().algorithm(),
                PrecisionConfig::ALG_DOT_BF16_BF16_F32);
      ++f32_dots;
    }
  }
  EXPECT_EQ(f32_dots, 1);
  EXPECT_EQ(f8_dots, 3);
}

absl::StatusOr<double> ComputeMse(const Literal& actual,
                                  const Literal& expected) {
  ABSL_ASSIGN_OR_RETURN(Literal a, actual.Convert(F64));
  ABSL_ASSIGN_OR_RETURN(Literal e, expected.Convert(F64));
  absl::Span<const double> av = a.data<double>();
  absl::Span<const double> ev = e.data<double>();
  TF_RET_CHECK(av.size() == ev.size());
  double sum = 0;
  for (size_t i = 0; i < av.size(); ++i) {
    const double d = av[i] - ev[i];
    sum += d * d;
  }
  return sum / av.size();
}

class Fp8xNRewriteTest
    : public DotAlgorithmRewriterTest,
      public ::testing::WithParamInterface<PrecisionConfig::Algorithm> {};

TEST_P(Fp8xNRewriteTest, DecomposesIntoF8DotsAndMatchesF32Reference) {
  const PrecisionConfig::Algorithm alg = GetParam();
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      p0 = f32[4,2,64] parameter(0)
      p1 = bf16[2,64,3] parameter(1)
      c1 = $0[2,64,3] convert(p1)
      ROOT dot = f32[2,4,3] dot(p0, c1), lhs_batch_dims={1},
        lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}$1
    })";
  ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(absl::Substitute(
                       kHlo, "bf16", ", algorithm=" + AlgorithmToString(alg))));
  ASSERT_OK_AND_ASSIGN(auto ref, ParseAndReturnVerifiedModule(
                                     absl::Substitute(kHlo, "f32", "")));
  EXPECT_THAT(RunHloPass(DotAlgorithmRewriter(
                             stream_executor::CudaComputeCapability::Hopper()),
                         module.get()),
              absl_testing::IsOkAndHolds(true));
  int f8_dots = 0;
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kDot) {
      EXPECT_EQ(inst->operand(0)->shape().element_type(), F8E4M3FN);
      EXPECT_EQ(inst->operand(1)->shape().element_type(), F8E4M3FN);
      ++f8_dots;
    }
  }
  EXPECT_EQ(f8_dots, alg == PrecisionConfig::ALG_DOT_BF16_BF16_FP8X4 ? 4 : 3);
  // F32 inputs keep more mantissa bits than two F8E4M3FN slices hold, so even
  // FP8X4 is only about as accurate as rounding the F32 operand to BF16.
  std::minstd_rand0 rng(42);
  ASSERT_OK_AND_ASSIGN(
      Literal lhs, LiteralUtil::CreateRandomLiteral<F32>(
                       ShapeUtil::MakeShape(F32, {4, 2, 64}), &rng, 0, 1000.0));
  ASSERT_OK_AND_ASSIGN(
      Literal rhs, LiteralUtil::CreateRandomLiteral<BF16>(
                       ShapeUtil::MakeShape(BF16, {2, 64, 3}), &rng, 0, 1e-3));
  ASSERT_OK_AND_ASSIGN(Literal actual,
                       HloEvaluator().Evaluate(*module, {&lhs, &rhs}));
  ASSERT_OK_AND_ASSIGN(Literal expected,
                       HloEvaluator().Evaluate(*ref, {&lhs, &rhs}));
  ASSERT_OK_AND_ASSIGN(const double mse, ComputeMse(actual, expected));
  if (alg == PrecisionConfig::ALG_DOT_BF16_BF16_FP8X4) {
    EXPECT_LT(mse, 6e-5);
    EXPECT_TRUE(LiteralTestUtil::Near(expected, actual,
                                      ErrorSpec{/*aabs=*/2.5e-2,
                                                /*arel=*/6e-3}));
  } else {
    EXPECT_LT(mse, 1.6e-4);
    EXPECT_TRUE(LiteralTestUtil::Near(expected, actual,
                                      ErrorSpec{/*aabs=*/4e-2,
                                                /*arel=*/1.6e-2}));
  }

  constexpr absl::string_view kBf16Hlo = R"(
    HloModule m
    ENTRY e {
      p0 = bf16[4,2,64] parameter(0)
      p1 = bf16[2,64,3] parameter(1)
      ROOT dot = bf16[2,4,3] dot(p0, p1), lhs_batch_dims={1},
        lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1},
        algorithm=$0
    })";
  ASSERT_OK_AND_ASSIGN(auto bf16_module,
                       ParseAndReturnVerifiedModule(
                           absl::Substitute(kBf16Hlo, AlgorithmToString(alg))));
  EXPECT_THAT(RunHloPass(DotAlgorithmRewriter(
                             stream_executor::CudaComputeCapability::Hopper()),
                         bf16_module.get()),
              absl_testing::IsOkAndHolds(true));
  const HloInstruction* root =
      bf16_module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(root->shape().element_type(), BF16);
}

INSTANTIATE_TEST_SUITE_P(
    Fp8xN, Fp8xNRewriteTest,
    ::testing::Values(PrecisionConfig::ALG_DOT_BF16_BF16_FP8X3,
                      PrecisionConfig::ALG_DOT_BF16_BF16_FP8X4));

// Evaluates the rewritten BF16, FP8X3 and FP8X4 dots on BF16 inputs against an
// exact F64 reference, mirroring MatmulTest.Fp8xNEmulationAccuracy on TPU.
TEST_F(DotAlgorithmRewriterTest, Fp8xNAccuracyAgainstF64Reference) {
  constexpr int kM = 128;
  constexpr int kK = 256;
  constexpr int kN = 128;
  for (auto [scale_a, scale_b] :
       {std::pair{1.0f, 1.0f}, {100.0f, 0.01f}, {0.01f, 100.0f}}) {
    std::minstd_rand0 rng(7);
    ASSERT_OK_AND_ASSIGN(Literal x, LiteralUtil::CreateRandomLiteral<BF16>(
                                        ShapeUtil::MakeShape(BF16, {kM, kK}),
                                        &rng, 0, 0.05 * scale_a));
    ASSERT_OK_AND_ASSIGN(Literal y, LiteralUtil::CreateRandomLiteral<BF16>(
                                        ShapeUtil::MakeShape(BF16, {kK, kN}),
                                        &rng, 0, 0.05 * scale_b));
    ASSERT_OK_AND_ASSIGN(Literal x64, x.Convert(F64));
    ASSERT_OK_AND_ASSIGN(Literal y64, y.Convert(F64));
    ASSERT_OK_AND_ASSIGN(Literal ref,
                         Literal::Make(ShapeUtil::MakeShape(F64, {kM, kN})));
    for (int m = 0; m < kM; ++m) {
      for (int n = 0; n < kN; ++n) {
        double acc = 0;
        for (int k = 0; k < kK; ++k) {
          acc += x64.Get<double>({m, k}) * y64.Get<double>({k, n});
        }
        ref.Set<double>({m, n}, acc);
      }
    }
    auto run = [&](PrecisionConfig::Algorithm alg) -> absl::StatusOr<Literal> {
      ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(absl::StrFormat(
                           R"(
              HloModule m
              ENTRY e {
                p0 = bf16[%1$d,%2$d] parameter(0)
                p1 = bf16[%2$d,%3$d] parameter(1)
                ROOT dot = f32[%1$d,%3$d] dot(p0, p1),
                  lhs_contracting_dims={1}, rhs_contracting_dims={0},
                  algorithm=%4$s
              })",
                           kM, kK, kN, AlgorithmToString(alg))));
      ABSL_ASSIGN_OR_RETURN(
          bool changed,
          RunHloPass(DotAlgorithmRewriter(
                         stream_executor::CudaComputeCapability::Hopper()),
                     module.get()));
      TF_RET_CHECK(changed);
      return HloEvaluator().Evaluate(*module, {&x, &y});
    };
    ASSERT_OK_AND_ASSIGN(Literal bf16,
                         run(PrecisionConfig::ALG_DOT_BF16_BF16_F32));
    ASSERT_OK_AND_ASSIGN(Literal fp8x3,
                         run(PrecisionConfig::ALG_DOT_BF16_BF16_FP8X3));
    ASSERT_OK_AND_ASSIGN(Literal fp8x4,
                         run(PrecisionConfig::ALG_DOT_BF16_BF16_FP8X4));
    ASSERT_OK_AND_ASSIGN(const double bf16_mse, ComputeMse(bf16, ref));
    ASSERT_OK_AND_ASSIGN(const double fp8x3_mse, ComputeMse(fp8x3, ref));
    ASSERT_OK_AND_ASSIGN(const double fp8x4_mse, ComputeMse(fp8x4, ref));
    ASSERT_OK_AND_ASSIGN(Literal fp8x3_f64, fp8x3.Convert(F64));
    ASSERT_OK_AND_ASSIGN(Literal fp8x4_f64, fp8x4.Convert(F64));
    const std::string tag =
        absl::StrFormat("scale_a=%g scale_b=%g", scale_a, scale_b);
    EXPECT_LT(fp8x4_mse, fp8x3_mse) << tag;
    EXPECT_LT(fp8x4_mse, bf16_mse) << tag;
    // Two F8E4M3FN slices hold a BF16 mantissa exactly, so FP8X4 matches the
    // F64 reference up to F32 accumulation error.
    EXPECT_LT(fp8x4_mse, 1e-17) << tag;
    EXPECT_TRUE(LiteralTestUtil::Near(ref, fp8x4_f64,
                                      ErrorSpec{/*aabs=*/5e-8,
                                                /*arel=*/1e-5}))
        << tag;
    // FP8X3 omits A_low * B_low, so it cannot match a BF16 dot.
    EXPECT_GT(fp8x3_mse, 2e-10) << tag;
    EXPECT_LT(fp8x3_mse, 4e-9) << tag;
    EXPECT_TRUE(LiteralTestUtil::Near(ref, fp8x3_f64,
                                      ErrorSpec{/*aabs=*/3e-4,
                                                /*arel=*/1e-2}))
        << tag;
  }
}

}  // namespace
}  // namespace xla::gpu
