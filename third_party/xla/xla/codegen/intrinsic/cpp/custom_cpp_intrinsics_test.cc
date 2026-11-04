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

#include "xla/codegen/intrinsic/cpp/custom_cpp_intrinsics.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/cpp/custom_cpp_intrinsics_ll.h"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"
#include "xla/codegen/intrinsic/test_matchers.h"

namespace xla::codegen {
namespace {

using ::testing::ContainsRegex;
using ::testing::Not;
using ::xla::codegen::intrinsic::NearUlps;

constexpr int kTrigUlps = 1;

template <typename VecT, typename Fn>
void ExpectVectorNearTrig(Fn fn, const ElementType<VecT>* xs, size_t n,
                          bool is_cos) {
  using Scalar = ElementType<VecT>;
  constexpr size_t kLanes = NumLanes<VecT>;
  for (size_t i = 0; i < n; i += kLanes) {
    VecT v = Splat<VecT>(0);
    for (size_t lane = 0; lane < kLanes; ++lane) {
      SetLane(v, lane, xs[i + lane]);
    }
    VecT out = fn(v);
    for (size_t lane = 0; lane < kLanes; ++lane) {
      Scalar x = xs[i + lane];
      Scalar actual = GetLane(out, lane);
      double ref = is_cos ? std::cos(static_cast<double>(x))
                          : std::sin(static_cast<double>(x));
      Scalar expected = static_cast<Scalar>(ref);
      EXPECT_THAT(actual, NearUlps(expected, kTrigUlps)) << "x=" << x;
      if (x == Scalar{0} && !is_cos) {
        EXPECT_EQ(std::signbit(actual), std::signbit(x));
      }
    }
  }
}

TEST(CustomCppIntrinsicsTest, SinAndCosF32WithinOneUlp) {
  using Limits = std::numeric_limits<float>;
  alignas(64) constexpr float kInputs[16] = {
      0.0f,
      -0.0f,
      Limits::denorm_min(),
      -Limits::denorm_min(),
      0.5f,
      -1.0f,
      1.5707963f,
      3.1415927f,
      -10.0f,
      0x1.0p+17f,
      0x1.5p+25f,
      -0x1.3p+40f,
      0x1.f37c8ap+95f,
      Limits::infinity(),
      -Limits::infinity(),
      Limits::quiet_NaN(),
  };
  ExpectVectorNearTrig<float>(sin_f32, kInputs, 16, /*is_cos=*/false);
  ExpectVectorNearTrig<Vec4f>(sin_v4f32, kInputs, 16, /*is_cos=*/false);
  ExpectVectorNearTrig<Vec8f>(sin_v8f32, kInputs, 16, /*is_cos=*/false);
  ExpectVectorNearTrig<Vec16f>(sin_v16f32, kInputs, 16, /*is_cos=*/false);

  ExpectVectorNearTrig<float>(cos_f32, kInputs, 16, /*is_cos=*/true);
  ExpectVectorNearTrig<Vec4f>(cos_v4f32, kInputs, 16, /*is_cos=*/true);
  ExpectVectorNearTrig<Vec8f>(cos_v8f32, kInputs, 16, /*is_cos=*/true);
  ExpectVectorNearTrig<Vec16f>(cos_v16f32, kInputs, 16, /*is_cos=*/true);
}

// Inputs found by exhaustive search where a float-float implementation
// returned a result 1 ULP away from the correctly rounded value.
TEST(CustomCppIntrinsicsTest, SinAndCosF32CorrectlyRoundedAtKnownHardCases) {
  auto f = [](uint32_t bits) { return absl::bit_cast<float>(bits); };
  EXPECT_EQ(absl::bit_cast<uint32_t>(cos_f32(f(0x3f4cc5a7))), 0x3f326080);
  EXPECT_EQ(absl::bit_cast<uint32_t>(cos_f32(f(0x4731b173))), 0x3f3010ac);
  EXPECT_EQ(absl::bit_cast<uint32_t>(cos_f32(f(0x40140780))), 0xbf2d06a2);
  EXPECT_EQ(absl::bit_cast<uint32_t>(sin_f32(f(0x3f524c80))), 0x3f3b6e7c);
  Vec8f v = {f(0x3f4cc5a7), f(0x4731b173), f(0x40140780), f(0x3f524c80)};
  Vec8f c = cos_v8f32(v);
  Vec8f s = sin_v8f32(v);
  EXPECT_EQ(absl::bit_cast<uint32_t>(c[0]), 0x3f326080);
  EXPECT_EQ(absl::bit_cast<uint32_t>(c[1]), 0x3f3010ac);
  EXPECT_EQ(absl::bit_cast<uint32_t>(c[2]), 0xbf2d06a2);
  EXPECT_EQ(absl::bit_cast<uint32_t>(s[3]), 0x3f3b6e7c);
}

TEST(CustomCppIntrinsicsTest, SinAndCosF64WithinOneUlp) {
  using Limits = std::numeric_limits<double>;
  alignas(64) constexpr double kInputs[8] = {
      0.0,  -0.0,      Limits::denorm_min(),  0.5,
      -2.5, 0x1.0p+30, 0x1.45f306dc9c883p+50, Limits::infinity(),
  };
  ExpectVectorNearTrig<double>(sin_f64, kInputs, 8, /*is_cos=*/false);
  ExpectVectorNearTrig<Vec2d>(sin_v2f64, kInputs, 8, /*is_cos=*/false);
  ExpectVectorNearTrig<Vec4d>(sin_v4f64, kInputs, 8, /*is_cos=*/false);
  ExpectVectorNearTrig<Vec8d>(sin_v8f64, kInputs, 8, /*is_cos=*/false);

  ExpectVectorNearTrig<double>(cos_f64, kInputs, 8, /*is_cos=*/true);
  ExpectVectorNearTrig<Vec2d>(cos_v2f64, kInputs, 8, /*is_cos=*/true);
  ExpectVectorNearTrig<Vec4d>(cos_v4f64, kInputs, 8, /*is_cos=*/true);
  ExpectVectorNearTrig<Vec8d>(cos_v8f64, kInputs, 8, /*is_cos=*/true);

  // Kahan/Muller worst-case near a multiple of pi/2 (~61 bits of cancellation);
  // compared against the exact rounded reference rather than std::cos because
  // older glibc (< 2.28) std::cos is 8 ULPs off at this input.
  constexpr double kWorstCase = 0x1.6ac5b262ca1ffp+849;
  EXPECT_THAT(sin_f64(kWorstCase), NearUlps(1.0, kTrigUlps));
  EXPECT_THAT(cos_f64(kWorstCase), NearUlps(-0x1.14ae72e6ba22fp-61, kTrigUlps));

  // Below the 2^20 Payne-Hanek threshold: the double closest to a multiple of
  // pi/2 (|r| = 2^-60.5), and one with |r| = 2^-52 where rounding n * pi/2 at
  // 2^-104 already makes r off by 2^-52 relative.
  EXPECT_THAT(cos_f64(0x1.6c6cbc45dc8dep+5),
              NearUlps(-0x1.6d61b58c99c43p-61, kTrigUlps));
  EXPECT_THAT(sin_f64(0x1.6c6cbc45dc8dep+5), NearUlps(1.0, kTrigUlps));
  EXPECT_THAT(cos_f64(0x1.93c05c9ed3cbcp+18),
              NearUlps(-0x1.065d73720c4f9p-52, kTrigUlps));
}

TEST(CustomCppIntrinsicsTest, EmbeddedBitcodeHasNoTargetIntrinsics) {
  ASSERT_TRUE(AreCppGenIntrinsicsAvailable());
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kCustomCppIntrinsicsLlIr);
  ASSERT_NE(module, nullptr);
  EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);
  // `llvm.fma` would become a libcall per lane on targets without FMA.
  EXPECT_THAT(ir, Not(ContainsRegex("llvm\\.fma\\.")));
  EXPECT_THAT(ir, ContainsRegex("llvm\\.fmuladd\\.v2f64"));
  EXPECT_THAT(ir, ContainsRegex("llvm\\.fmuladd\\.v4f64"));
  EXPECT_THAT(ir, ContainsRegex("llvm\\.fmuladd\\.v8f64"));
  EXPECT_THAT(ir, ContainsRegex("llvm\\.fmuladd\\.v16f64"));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm\\.x86")));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm\\.aarch64")));
  EXPECT_THAT(ir, Not(ContainsRegex("__udivti3")));
}

}  // namespace
}  // namespace xla::codegen
