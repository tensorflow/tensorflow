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

#include "xla/codegen/intrinsic/cpp/trig.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/cpp/trig_ll.h"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"
#include "xla/codegen/intrinsic/test_matchers.h"

namespace xla::codegen {
namespace {
using ::testing::ContainsRegex;
using ::xla::codegen::intrinsic::NearUlps;

TEST(TrigTest, SinhF32IsCorrect) {
  double max_ulp_error = 0.0;
  float worst_x = 0.0f;
  int64_t max_int_ulp_diff = 0;

  auto check_point = [&](uint32_t bits) {
    float x;
    std::memcpy(&x, &bits, sizeof(x));
    double expected = std::sinh(static_cast<double>(x));
    float actual = sinh_f32(x);

    float expected_f32 = static_cast<float>(expected);
    uint32_t actual_bits, expected_bits;
    std::memcpy(&actual_bits, &actual, sizeof(actual_bits));
    std::memcpy(&expected_bits, &expected_f32, sizeof(expected_bits));
    int64_t int_diff = std::abs(static_cast<int64_t>(actual_bits) -
                                static_cast<int64_t>(expected_bits));
    if (int_diff > max_int_ulp_diff) {
      max_int_ulp_diff = int_diff;
    }

    int exp;
    std::frexp(expected, &exp);
    double ulp = std::ldexp(1.0, std::max(exp - 24, -149));
    double ulp_error = std::abs(static_cast<double>(actual) - expected) / ulp;
    if (ulp_error > max_ulp_error) {
      max_ulp_error = ulp_error;
      worst_x = x;
    }
  };

  // Sweep the full finite positive domain [0.0f, 89.4159851f] with stride 256,
  // plus exhaustive coverage around the worst-case transition point x
  // = 1.04064.
  constexpr uint32_t kMaxFiniteSinhBits = 0x42b2d4fc;
  for (uint32_t bits = 0; bits <= kMaxFiniteSinhBits; bits += 256) {
    check_point(bits);
  }
  constexpr uint32_t kWorstCaseBits = 0x3f8533bf;  // 1.0406416654586792f
  for (uint32_t bits = kWorstCaseBits - 4096; bits <= kWorstCaseBits + 4096;
       ++bits) {
    check_point(bits);
  }

  EXPECT_LE(max_int_ulp_diff, 1);
  EXPECT_LE(max_ulp_error, 0.81) << "worst_x = " << worst_x;
}

TEST(TrigTest, SinhF32EdgeCases) {
  EXPECT_EQ(sinh_f32(0.0f), 0.0f);
  EXPECT_FALSE(std::signbit(sinh_f32(0.0f)));
  EXPECT_EQ(sinh_f32(-0.0f), -0.0f);
  EXPECT_TRUE(std::signbit(sinh_f32(-0.0f)));

  // Overflow threshold: sinh(89.4159851f) < FLT_MAX, sinh(89.4159927f) >
  // FLT_MAX
  float just_below = 89.4159851f;
  EXPECT_FALSE(std::isinf(sinh_f32(just_below)));
  EXPECT_THAT(
      sinh_f32(just_below),
      NearUlps(static_cast<float>(std::sinh(static_cast<double>(just_below))),
               1));
  EXPECT_FALSE(std::isinf(sinh_f32(-just_below)));
  EXPECT_THAT(
      sinh_f32(-just_below),
      NearUlps(static_cast<float>(std::sinh(static_cast<double>(-just_below))),
               1));

  float just_above = 89.4159927f;
  EXPECT_EQ(sinh_f32(just_above), std::numeric_limits<float>::infinity());
  EXPECT_EQ(sinh_f32(-just_above), -std::numeric_limits<float>::infinity());

  EXPECT_EQ(sinh_f32(std::numeric_limits<float>::infinity()),
            std::numeric_limits<float>::infinity());
  EXPECT_EQ(sinh_f32(-std::numeric_limits<float>::infinity()),
            -std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isnan(sinh_f32(std::numeric_limits<float>::quiet_NaN())));
}

TEST(TrigTest, SinhF32VectorWidthsAreCorrect) {
  Vec4f x4 = {-10.0f, -0.5f, 0.5f, 10.0f};
  Vec4f y4 = sinh_v4f32(x4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(
        y4[i],
        NearUlps(static_cast<float>(std::sinh(static_cast<double>(x4[i]))), 1));
  }

  Vec8f x8 = {-50.0f, -10.0f, -1.0f, -0.1f, 0.1f, 1.0f, 10.0f, 50.0f};
  Vec8f y8 = sinh_v8f32(x8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(
        y8[i],
        NearUlps(static_cast<float>(std::sinh(static_cast<double>(x8[i]))), 1));
  }

  Vec16f x16 = {-80.0f, -50.0f, -20.0f, -5.0f, -1.5f, -0.5f, -0.01f, -0.0f,
                0.0f,   0.01f,  0.5f,   1.5f,  5.0f,  20.0f, 50.0f,  80.0f};
  Vec16f y16 = sinh_v16f32(x16);
  for (int i = 0; i < 16; ++i) {
    EXPECT_THAT(
        y16[i],
        NearUlps(static_cast<float>(std::sinh(static_cast<double>(x16[i]))),
                 1));
  }
}

TEST(TrigTest, SinhIsVectorized) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kTrigLlIr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);
  EXPECT_THAT(ir, ContainsRegex("xla.sinh.v16f32"));
  EXPECT_THAT(ir, ContainsRegex("fmul.*<16 x float>"));
}

}  // namespace
}  // namespace xla::codegen
