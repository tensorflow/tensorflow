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

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "xla/tests/xla_test_backend_predicates.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/types.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;

class FmaTest : public HloInterpreterReferenceMixin<HloTestBase> {
 protected:
  bool BackendSupportsF16Fma() {
    if (test::DeviceTypeIs(test::kCpu)) {
      return false;
    }
    if (test::DeviceTypeIsOneOf({test::kTpu, test::kInterpreter})) {
      return true;
    }
    if (test_runner().HasProperty(HloRunnerPropertyTag::kUsingGpuOneAPI)) {
      return false;
    }
    return true;
  }

  bool BackendSupportsBf16Fma() {
    if (test::DeviceTypeIs(test::kCpu) ||
        test::DeviceIsOneOf({test::kP100, test::kV100})) {
      return false;
    }
    if (test::DeviceTypeIsOneOf({test::kTpu, test::kInterpreter}) ||
        test::DeviceIsOneOf({test::kA100, test::kH100, test::kB200})) {
      return true;
    }
    if (test_runner().HasProperty(HloRunnerPropertyTag::kUsingGpuOneAPI)) {
      return false;
    }
    // For wildcard GPU targets (such as nvgpu_any or amdgpu_any), check
    // whether compilation rejects bf16 fma as unimplemented on the underlying
    // GPU architecture (e.g., T4 or pre-gfx1250 ROCm).
    constexpr char kProbeHlo[] = R"hlo(
      HloModule probe_bf16_fma
      ENTRY main {
        a = bf16[1] parameter(0)
        b = bf16[1] parameter(1)
        c = bf16[1] parameter(2)
        ROOT out = bf16[1] fma(a, b, c)
      }
    )hlo";
    absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
        ParseAndReturnVerifiedModule(kProbeHlo);
    if (!module.ok()) {
      ADD_FAILURE() << module.status();
      return false;
    }
    absl::Status status =
        CreateExecutable(*std::move(module), /*run_hlo_passes=*/true).status();
    if (absl::IsUnimplemented(status)) {
      return false;
    }
    EXPECT_OK(status);
    return status.ok();
  }
};

TEST_F(FmaTest, FmaF32) {
  constexpr char kHlo[] = R"hlo(
    HloModule fma_f32
    ENTRY main {
      a = f32[10] parameter(0)
      b = f32[10] parameter(1)
      c = f32[10] parameter(2)
      ROOT out = f32[10] fma(a, b, c)
    }
  )hlo";

  const float inf = std::numeric_limits<float>::infinity();
  // (1 + 2^-12)^2 - (1 + 2^-11) = 2^-24 with single rounding, but 0.0 with
  // separate multiply and add because (1 + 2^-12)^2 rounds to (1 + 2^-11).
  const float one_plus_2m12 = std::ldexp(1.0f, 0) + std::ldexp(1.0f, -12);
  const float neg_one_plus_2m11 =
      -(std::ldexp(1.0f, 0) + std::ldexp(1.0f, -11));

  Literal a = LiteralUtil::CreateR1<float>({
      one_plus_2m12,
      1.5f,
      1.5f,
      1.75f,
      -1.5f,
      std::ldexp(1.0f, 100),
      std::ldexp(1.0f, 100),
      0.0f,
      -0.0f,
      1.0f,
  });
  Literal b = LiteralUtil::CreateR1<float>({
      one_plus_2m12,
      1.5f,
      1.5f,
      1.75f,
      1.5f,
      std::ldexp(1.0f, 100),
      std::ldexp(1.0f, -100),
      1.0f,
      1.0f,
      2.0f,
  });
  Literal c = LiteralUtil::CreateR1<float>({
      neg_one_plus_2m11,
      -2.25f,
      -2.0f,
      -2.0f,
      2.0f,
      1.0f,
      1.0f,
      -0.0f,
      -0.0f,
      inf,
  });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunAndCompare(std::move(module), {&a, &b, &c},
                            /*error=*/std::nullopt));
}

TEST_F(FmaTest, FmaF32Subnormals) {
  if (test::DeviceTypeIs(test::kCpu) || test::DeviceTypeIs(test::kTpu)) {
    GTEST_SKIP() << "CPU and TPU flush f32 subnormals to zero.";
  }
  constexpr char kHlo[] = R"hlo(
    HloModule fma_f32_subnormals
    ENTRY main {
      a = f32[4] parameter(0)
      b = f32[4] parameter(1)
      c = f32[4] parameter(2)
      ROOT out = f32[4] fma(a, b, c)
    }
  )hlo";

  Literal a = LiteralUtil::CreateR1<float>({
      std::ldexp(1.0f, -126),
      std::ldexp(1.0f, -100),
      std::ldexp(1.0f, -126),
      std::ldexp(1.0f, -149),
  });
  Literal b = LiteralUtil::CreateR1<float>({
      0.5f,
      std::ldexp(1.0f, -40),
      1.0f,
      2.0f,
  });
  Literal c = LiteralUtil::CreateR1<float>({
      0.0f,
      std::ldexp(1.0f, -126),
      -std::ldexp(1.0f, -127),
      0.0f,
  });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunAndCompare(std::move(module), {&a, &b, &c},
                            /*error=*/std::nullopt));
}

TEST_F(FmaTest, FmaF64) {
  if (!test::BackendSupportsFloat64() || test::DeviceTypeIs(test::kTpu)) {
    GTEST_SKIP() << "Backend does not support strict f64 fma.";
  }
  constexpr char kHlo[] = R"hlo(
    HloModule fma_f64
    ENTRY main {
      a = f64[5] parameter(0)
      b = f64[5] parameter(1)
      c = f64[5] parameter(2)
      ROOT out = f64[5] fma(a, b, c)
    }
  )hlo";

  // (1 + 2^-26) * (1 + 2^-27) - (1 + 2^-26 + 2^-27) = 2^-53 with single
  // rounding, but 0.0 with separate multiply and add.
  const double one_plus_2m26 = 1.0 + std::ldexp(1.0, -26);
  const double one_plus_2m27 = 1.0 + std::ldexp(1.0, -27);
  const double neg_sum = -(1.0 + std::ldexp(1.0, -26) + std::ldexp(1.0, -27));

  Literal a = LiteralUtil::CreateR1<double>({
      one_plus_2m26,
      1.5,
      std::ldexp(1.0, 600),
      std::ldexp(1.0, -1000),
      -0.0,
  });
  Literal b = LiteralUtil::CreateR1<double>({
      one_plus_2m27,
      1.5,
      std::ldexp(1.0, 600),
      std::ldexp(1.0, -22),
      1.0,
  });
  Literal c = LiteralUtil::CreateR1<double>({
      neg_sum,
      -2.25,
      -std::numeric_limits<double>::max(),
      0.0,
      -0.0,
  });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunAndCompare(std::move(module), {&a, &b, &c},
                            /*error=*/std::nullopt));
}

TEST_F(FmaTest, FmaF64Subnormals) {
  if (!test::BackendSupportsFloat64() || test::DeviceTypeIs(test::kCpu) ||
      test::DeviceTypeIs(test::kTpu)) {
    GTEST_SKIP() << "Backend does not support strict f64 subnormal fma.";
  }
  constexpr char kHlo[] = R"hlo(
    HloModule fma_f64_subnormals
    ENTRY main {
      a = f64[2] parameter(0)
      b = f64[2] parameter(1)
      c = f64[2] parameter(2)
      ROOT out = f64[2] fma(a, b, c)
    }
  )hlo";

  Literal a = LiteralUtil::CreateR1<double>({
      std::ldexp(1.0, -1022),
      std::ldexp(1.0, -1074),
  });
  Literal b = LiteralUtil::CreateR1<double>({
      0.5,
      2.0,
  });
  Literal c = LiteralUtil::CreateR1<double>({
      0.0,
      0.0,
  });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunAndCompare(std::move(module), {&a, &b, &c},
                            /*error=*/std::nullopt));
}

TEST_F(FmaTest, FmaF16) {
  if (!BackendSupportsF16Fma()) {
    GTEST_SKIP() << "Backend does not support f16 fma.";
  }
  constexpr char kHlo[] = R"hlo(
    HloModule fma_f16
    ENTRY main {
      a = f16[8] parameter(0)
      b = f16[8] parameter(1)
      c = f16[8] parameter(2)
      ROOT out = f16[8] fma(a, b, c)
    }
  )hlo";

  // (1 + 2^-5) * (1 + 2^-6) - (1 + 2^-5 + 2^-6) = 2^-11 (single-rounded).
  const half a0(1.0f + std::ldexp(1.0f, -5));
  const half b0(1.0f + std::ldexp(1.0f, -6));
  const half c0(-(1.0f + std::ldexp(1.0f, -5) + std::ldexp(1.0f, -6)));
  // 256 * 256 - 65504 = 32 (whereas 256 * 256 overflows f16 to +Inf).
  const half a1(256.0f);
  const half b1(256.0f);
  const half c1(-65504.0f);
  // Subnormal f16 result: 2^-14 * 0.5 + 0 = 2^-15.
  const half a2(std::ldexp(1.0f, -14));
  const half b2(0.5f);
  const half c2(0.0f);
  // Double-rounding traps when promoting f16 FMA to f32:
  // 1) 1.5 * (1 + 2^-10) - 2^-24 -> 1.5 + 2^-10 (0x3e01), whereas fma_f32
  //    rounds to 1.5 + 2^-10 + 2^-11 and then to 1.5 + 2^-9 (0x3e02).
  const half a3(1.5f);
  const half b3(1.0f + std::ldexp(1.0f, -10));
  const half c3(-std::ldexp(1.0f, -24));
  // 2) (1.5 + 3*2^-10) * (1 - 2^-10) + 47*2^-24 -> 1.5 + 2^-10 (0x3e01),
  //    whereas fma_f32 double-rounds to 1.5 + 2^-9 (0x3e02).
  const half a4(1.5f + 3.0f * std::ldexp(1.0f, -10));
  const half b4(1.0f - std::ldexp(1.0f, -10));
  const half c4(47.0f * std::ldexp(1.0f, -24));

  Literal a = LiteralUtil::CreateR1<half>({
      a0,
      a1,
      a2,
      a3,
      a4,
      half(1.5f),
      half(0.0f),
      half(-0.0f),
  });
  Literal b = LiteralUtil::CreateR1<half>({
      b0,
      b1,
      b2,
      b3,
      b4,
      half(1.5f),
      half(1.0f),
      half(1.0f),
  });
  Literal c = LiteralUtil::CreateR1<half>({
      c0,
      c1,
      c2,
      c3,
      c4,
      half(-2.25f),
      half(-0.0f),
      half(-0.0f),
  });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunAndCompare(std::move(module), {&a, &b, &c},
                            /*error=*/std::nullopt));
}

TEST_F(FmaTest, FmaBf16) {
  if (!BackendSupportsBf16Fma()) {
    GTEST_SKIP() << "Backend does not support bf16 fma.";
  }
  constexpr char kHlo[] = R"hlo(
    HloModule fma_bf16
    ENTRY main {
      a = bf16[6] parameter(0)
      b = bf16[6] parameter(1)
      c = bf16[6] parameter(2)
      ROOT out = bf16[6] fma(a, b, c)
    }
  )hlo";

  // Double-rounding trap: 1.5 * (1 + 2^-7) - 2^-24.
  // Exact value is (1.5 + 2^-7) + (2^-8 - 2^-24), which is strictly below the
  // midpoint to 1.5 + 2^-6, so single-rounded bf16 is 1.5 + 2^-7 (0x3fc1).
  // Computing f32 FMA first rounds to 1.5 + 2^-7 + 2^-8, which then rounds to
  // 1.5 + 2^-6 (0x3fc2) in bf16.
  const bfloat16 a0(1.5f);
  const bfloat16 b0(1.0f + std::ldexp(1.0f, -7));
  const bfloat16 c0(-std::ldexp(1.0f, -24));

  // Intermediate f32 overflow rescued by addend: 2^64 * 2^64 - 1.75 * 2^127 =
  // 2^125.
  const bfloat16 a1(std::ldexp(1.0f, 64));
  const bfloat16 b1(std::ldexp(1.0f, 64));
  const bfloat16 c1(-std::ldexp(1.75f, 127));

  Literal a = LiteralUtil::CreateR1<bfloat16>({
      a0,
      a1,
      bfloat16(1.5f),
      bfloat16(1.75f),
      bfloat16(0.0f),
      bfloat16(-0.0f),
  });
  Literal b = LiteralUtil::CreateR1<bfloat16>({
      b0,
      b1,
      bfloat16(1.5f),
      bfloat16(1.75f),
      bfloat16(1.0f),
      bfloat16(1.0f),
  });
  Literal c = LiteralUtil::CreateR1<bfloat16>({
      c0,
      c1,
      bfloat16(-2.25f),
      bfloat16(-2.0f),
      bfloat16(-0.0f),
      bfloat16(-0.0f),
  });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunAndCompare(std::move(module), {&a, &b, &c},
                            /*error=*/std::nullopt));
}

TEST_F(FmaTest, FmaBf16SubnormalProduct) {
  if (!BackendSupportsBf16Fma()) {
    GTEST_SKIP() << "Backend does not support bf16 fma.";
  }
  constexpr char kHlo[] = R"hlo(
    HloModule bf16_subnormal_product
    ENTRY main {
      a = bf16[5] parameter(0)
      b = bf16[5] parameter(1)
      c = bf16[5] parameter(2)
      ROOT out = bf16[5] fma(a, b, c)
    }
  )hlo";

  // Case 0: (129 * 2^-70) * (127 * 2^-70) + 0 = (1 - 2^-14) * 2^-126, which is
  // an f32 subnormal (< 2^-126) that rounds up to the smallest normal bf16
  // value 2^-126 (0x0080) under single-rounded bf16 (even when bf16 flushes
  // subnormals to zero), whereas an unscaled FTZ f32 multiply flushes the
  // product to 0.0 before rounding to bf16.
  // Case 1: (135 * 2^-70) * (242 * 2^-71) + 0 = 127.6171875 * 2^-133 in
  // [127.5 * 2^-133, 127.75 * 2^-133), which also rounds up to 2^-126 in bf16
  // (and tests the subnormal-midpoint bias when scaling by 2.0f into the first
  // normal binade [2^-126, 2^-125)).
  // Case 2: (128 * 2^-70) * (255 * 2^-71) + 0 = 127.5 * 2^-133, the exact
  // midpoint between 127 * 2^-133 (odd) and 128 * 2^-133 = 2^-126 (even), which
  // rounds to 2^-126 under roundTiesToEven.
  // Case 3: 2^127 * 0.0 + 1.0 = 1.0 (verifying pre-scaling is disabled when
  // |a| >= 2.0 so 2 * a does not overflow to +Inf and turn Inf * 0 into NaN).
  // Case 4: 2.0 * 2^-126 + 0.0 = 2^-125.
  Literal a = LiteralUtil::CreateR1<bfloat16>({
      bfloat16(129.0f * std::ldexp(1.0f, -70)),
      bfloat16(135.0f * std::ldexp(1.0f, -70)),
      bfloat16(128.0f * std::ldexp(1.0f, -70)),
      bfloat16(std::ldexp(1.0f, 127)),
      bfloat16(2.0f),
  });
  Literal b = LiteralUtil::CreateR1<bfloat16>({
      bfloat16(127.0f * std::ldexp(1.0f, -70)),
      bfloat16(242.0f * std::ldexp(1.0f, -71)),
      bfloat16(255.0f * std::ldexp(1.0f, -71)),
      bfloat16(0.0f),
      bfloat16(std::ldexp(1.0f, -126)),
  });
  Literal c = LiteralUtil::CreateR1<bfloat16>({
      bfloat16(0.0f),
      bfloat16(0.0f),
      bfloat16(0.0f),
      bfloat16(1.0f),
      bfloat16(0.0f),
  });
  Literal expected = LiteralUtil::CreateR1<bfloat16>({
      bfloat16(std::ldexp(1.0f, -126)),
      bfloat16(std::ldexp(1.0f, -126)),
      bfloat16(std::ldexp(1.0f, -126)),
      bfloat16(1.0f),
      bfloat16(std::ldexp(1.0f, -125)),
  });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(Literal result,
                       Execute(std::move(module), {&a, &b, &c}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(FmaTest, FmaScalarF32) {
  constexpr char kHlo[] = R"hlo(
    HloModule fma_scalar_f32
    ENTRY main {
      a = f32[4] parameter(0)
      b = f32[4] parameter(1)
      c = f32[4] parameter(2)
      s_a = f32[] parameter(3)
      s_b = f32[] parameter(4)
      s_c = f32[] parameter(5)
      v_fma = f32[4] fma(a, b, c)
      s_fma = f32[] fma(s_a, s_b, s_c)
      s_bcast = f32[4] broadcast(s_fma), dimensions={}
      ROOT out = f32[4] add(v_fma, s_bcast)
    }
  )hlo";

  const float one_plus_2m12 = 1.0f + std::ldexp(1.0f, -12);
  const float neg_one_plus_2m11 = -(1.0f + std::ldexp(1.0f, -11));
  Literal a = LiteralUtil::CreateR1<float>({1.5f, -2.0f, 0.5f, one_plus_2m12});
  Literal b = LiteralUtil::CreateR1<float>({2.0f, 3.0f, 0.25f, one_plus_2m12});
  Literal c =
      LiteralUtil::CreateR1<float>({0.5f, 6.0f, -0.125f, neg_one_plus_2m11});
  Literal s_a = LiteralUtil::CreateR0<float>(1.5f);
  Literal s_b = LiteralUtil::CreateR0<float>(1.5f);
  Literal s_c = LiteralUtil::CreateR0<float>(-2.25f);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunAndCompare(std::move(module), {&a, &b, &c, &s_a, &s_b, &s_c},
                            /*error=*/std::nullopt));
}

TEST_F(FmaTest, FmaSubnormalFlushingOnTpu) {
  if (!test::DeviceTypeIs(test::kTpu)) {
    GTEST_SKIP() << "Tests TPU-specific FTZ subnormal flushing for f32/bf16.";
  }
  constexpr char kHlo[] = R"hlo(
    HloModule tpu_fma_ftz
    ENTRY main {
      a_f32 = f32[3] parameter(0)
      b_f32 = f32[3] parameter(1)
      c_f32 = f32[3] parameter(2)
      out_f32 = f32[3] fma(a_f32, b_f32, c_f32)
      a_bf16 = bf16[4] parameter(3)
      b_bf16 = bf16[4] parameter(4)
      c_bf16 = bf16[4] parameter(5)
      out_bf16_fma = bf16[4] fma(a_bf16, b_bf16, c_bf16)
      ROOT out = (f32[3], bf16[4]) tuple(out_f32, out_bf16_fma)
    }
  )hlo";

  const float min_sub = absl::bit_cast<float>(0x00000001u);
  const float min_norm = absl::bit_cast<float>(0x00800000u);
  Literal a_f32 = LiteralUtil::CreateR1<float>({min_norm, -min_norm, min_sub});
  Literal b_f32 = LiteralUtil::CreateR1<float>({0.5f, 0.5f, 2.0f});
  Literal c_f32 = LiteralUtil::CreateR1<float>({0.0f, -0.0f, 1.0f});

  Literal a_bf16 = LiteralUtil::CreateR1<bfloat16>({
      bfloat16(std::ldexp(1.0f, -126)),
      bfloat16(-std::ldexp(1.0f, -126)),
      bfloat16(128.0f * std::ldexp(1.0f, -70)),
      bfloat16(131.0f * std::ldexp(1.0f, -70)),
  });
  Literal b_bf16 = LiteralUtil::CreateR1<bfloat16>({
      bfloat16(0.5f),
      bfloat16(0.5f),
      bfloat16(127.0f * std::ldexp(1.0f, -70)),
      bfloat16(249.0f * std::ldexp(1.0f, -71)),
  });
  Literal c_bf16 = LiteralUtil::CreateR1<bfloat16>({
      bfloat16(0.0f),
      bfloat16(-0.0f),
      bfloat16(0.0f),
      bfloat16(0.0f),
  });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(Literal result,
                       Execute(std::move(module), {&a_f32, &b_f32, &c_f32,
                                                   &a_bf16, &b_bf16, &c_bf16}));
  auto outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({0.0f, -0.0f, 1.0f}), outputs[0]));
  EXPECT_EQ(absl::bit_cast<uint32_t>(outputs[0].Get<float>({1})), 0x80000000u);
  Literal expected_bf16 = LiteralUtil::CreateR1<bfloat16>(
      {bfloat16(0.0f), bfloat16(-0.0f), bfloat16(0.0f), bfloat16(0.0f)});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_bf16, outputs[1]));
  EXPECT_EQ(absl::bit_cast<uint16_t>(outputs[1].Get<bfloat16>({1})), 0x8000u);
}

TEST_F(FmaTest, FmaF64UnimplementedOnTpu) {
  if (!test::DeviceTypeIs(test::kTpu)) {
    GTEST_SKIP() << "f64 fma is supported on non-TPU backends.";
  }
  constexpr char kHlo[] = R"hlo(
    HloModule tpu_fma_f64
    ENTRY main {
      a = f64[4] parameter(0)
      b = f64[4] parameter(1)
      c = f64[4] parameter(2)
      ROOT out = f64[4] fma(a, b, c)
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(CreateExecutable(std::move(module), /*run_hlo_passes=*/true),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(FmaTest, FmaLowPrecisionUnimplementedOnCpu) {
  if (!test::DeviceTypeIs(test::kCpu)) {
    GTEST_SKIP() << "Tests CPU-specific rejection of low-precision fma.";
  }
  for (const char* hlo : {
           R"hlo(
             HloModule cpu_fma_f16
             ENTRY main {
               a = f16[4] parameter(0)
               b = f16[4] parameter(1)
               c = f16[4] parameter(2)
               ROOT out = f16[4] fma(a, b, c)
             }
           )hlo",
           R"hlo(
             HloModule cpu_fma_bf16
             ENTRY main {
               a = bf16[4] parameter(0)
               b = bf16[4] parameter(1)
               c = bf16[4] parameter(2)
               ROOT out = bf16[4] fma(a, b, c)
             }
           )hlo",
       }) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo));
    EXPECT_THAT(CreateExecutable(std::move(module), /*run_hlo_passes=*/true),
                StatusIs(absl::StatusCode::kUnimplemented));
  }
}

TEST_F(FmaTest, FmaBf16UnimplementedOnPreAmpereGpu) {
  if (!test::DeviceIsOneOf({test::kP100, test::kV100})) {
    GTEST_SKIP() << "Tests pre-Ampere GPU rejection of bf16 fma.";
  }
  constexpr char kHlo[] = R"hlo(
    HloModule gpu_fma_bf16_unimplemented
    ENTRY main {
      a = bf16[4] parameter(0)
      b = bf16[4] parameter(1)
      c = bf16[4] parameter(2)
      ROOT out = bf16[4] fma(a, b, c)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(CreateExecutable(std::move(module), /*run_hlo_passes=*/true),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(FmaTest, FmaSub16BitFloatUnimplemented) {
  if (test::DeviceTypeIs(test::kInterpreter)) {
    GTEST_SKIP() << "Interpreter evaluates low-precision fma directly.";
  }
  for (const char* hlo : {
           R"hlo(
             HloModule fma_f8e4m3fn
             ENTRY main {
               a = f8e4m3fn[4] parameter(0)
               b = f8e4m3fn[4] parameter(1)
               c = f8e4m3fn[4] parameter(2)
               ROOT out = f8e4m3fn[4] fma(a, b, c)
             }
           )hlo",
           R"hlo(
             HloModule fma_f8e5m2
             ENTRY main {
               a = f8e5m2[4] parameter(0)
               b = f8e5m2[4] parameter(1)
               c = f8e5m2[4] parameter(2)
               ROOT out = f8e5m2[4] fma(a, b, c)
             }
           )hlo",
       }) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo));
    EXPECT_THAT(CreateExecutable(std::move(module), /*run_hlo_passes=*/true),
                StatusIs(absl::StatusCode::kUnimplemented));
  }
}

TEST_F(FmaTest, FmaBf16FusedWithAlu) {
  if (!BackendSupportsBf16Fma()) {
    GTEST_SKIP() << "Backend does not support bf16 fma.";
  }
  constexpr char kHlo[] = R"hlo(
    HloModule fma_bf16_fused
    ENTRY main {
      a = bf16[8] parameter(0)
      b = bf16[8] parameter(1)
      c = bf16[8] parameter(2)
      a2 = bf16[8] add(a, a)
      m = bf16[8] fma(a2, b, c)
      ROOT out = bf16[8] add(m, b)
    }
  )hlo";

  const bfloat16 half_a0(0.75f);
  const bfloat16 b0(1.0f + std::ldexp(1.0f, -7));
  const bfloat16 c0(-std::ldexp(1.0f, -24));
  Literal a = LiteralUtil::CreateR1<bfloat16>({
      half_a0,
      bfloat16(0.75f),
      bfloat16(0.875f),
      bfloat16(-0.75f),
      bfloat16(0.0f),
      bfloat16(-0.0f),
      bfloat16(1.0f),
      bfloat16(2.0f),
  });
  Literal b = LiteralUtil::CreateR1<bfloat16>({
      b0,
      bfloat16(1.5f),
      bfloat16(1.75f),
      bfloat16(1.5f),
      bfloat16(1.0f),
      bfloat16(1.0f),
      bfloat16(0.5f),
      bfloat16(0.25f),
  });
  Literal c = LiteralUtil::CreateR1<bfloat16>({
      c0,
      bfloat16(-2.25f),
      bfloat16(-2.0f),
      bfloat16(2.0f),
      bfloat16(-0.0f),
      bfloat16(-0.0f),
      bfloat16(0.25f),
      bfloat16(-0.5f),
  });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunAndCompare(std::move(module), {&a, &b, &c},
                            /*error=*/std::nullopt));
}

TEST_F(FmaTest, FmaNaN) {
  const float inf = std::numeric_limits<float>::infinity();
  const float qnan = std::numeric_limits<float>::quiet_NaN();

  {
    constexpr char kHloF32[] = R"hlo(
      HloModule fma_nan_f32
      ENTRY main {
        a = f32[3] parameter(0)
        b = f32[3] parameter(1)
        c = f32[3] parameter(2)
        ROOT out = f32[3] fma(a, b, c)
      }
    )hlo";
    Literal a = LiteralUtil::CreateR1<float>({inf, 1.0f, qnan});
    Literal b = LiteralUtil::CreateR1<float>({0.0f, 2.0f, 2.0f});
    Literal c = LiteralUtil::CreateR1<float>({1.0f, qnan, 1.0f});
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(kHloF32));
    // ErrorSpec{0.0, 0.0} matches NaNs regardless of payload or sign bit while
    // requiring exact equality for any non-NaN values.
    EXPECT_TRUE(
        RunAndCompare(std::move(module), {&a, &b, &c}, ErrorSpec{0.0, 0.0}));
  }

  if (BackendSupportsF16Fma()) {
    constexpr char kHloF16[] = R"hlo(
      HloModule fma_nan_f16
      ENTRY main {
        a = f16[3] parameter(0)
        b = f16[3] parameter(1)
        c = f16[3] parameter(2)
        ROOT out = f16[3] fma(a, b, c)
      }
    )hlo";
    Literal a =
        LiteralUtil::CreateR1<half>({half(inf), half(1.0f), half(qnan)});
    Literal b =
        LiteralUtil::CreateR1<half>({half(0.0f), half(2.0f), half(2.0f)});
    Literal c =
        LiteralUtil::CreateR1<half>({half(1.0f), half(qnan), half(1.0f)});
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(kHloF16));
    EXPECT_TRUE(
        RunAndCompare(std::move(module), {&a, &b, &c}, ErrorSpec{0.0, 0.0}));
  }

  if (BackendSupportsBf16Fma()) {
    constexpr char kHloBf16[] = R"hlo(
      HloModule fma_nan_bf16
      ENTRY main {
        a = bf16[3] parameter(0)
        b = bf16[3] parameter(1)
        c = bf16[3] parameter(2)
        ROOT out = bf16[3] fma(a, b, c)
      }
    )hlo";
    Literal a = LiteralUtil::CreateR1<bfloat16>(
        {bfloat16(inf), bfloat16(1.0f), bfloat16(qnan)});
    Literal b = LiteralUtil::CreateR1<bfloat16>(
        {bfloat16(0.0f), bfloat16(2.0f), bfloat16(2.0f)});
    Literal c = LiteralUtil::CreateR1<bfloat16>(
        {bfloat16(1.0f), bfloat16(qnan), bfloat16(1.0f)});
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(kHloBf16));
    EXPECT_TRUE(
        RunAndCompare(std::move(module), {&a, &b, &c}, ErrorSpec{0.0, 0.0}));
  }
}

}  // namespace
}  // namespace xla
