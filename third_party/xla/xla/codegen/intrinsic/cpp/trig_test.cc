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
#include "absl/base/casts.h"
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

TEST(TrigTest, CoshF32IsCorrect) {
  double max_ulp_error = 0.0;
  float worst_x = 0.0f;
  int64_t max_int_ulp_diff = 0;

  auto check_point = [&](uint32_t bits) {
    float x;
    std::memcpy(&x, &bits, sizeof(x));
    double expected = std::cosh(static_cast<double>(x));
    float actual = cosh_f32(x);

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

  // Sweep the full finite positive domain [0.0f, 89.4159851f] with stride 64,
  // plus exhaustive coverage around the range-reduction transition points.
  constexpr uint32_t kMaxFiniteCoshBits = 0x42b2d4fc;
  for (uint32_t bits = 0; bits <= kMaxFiniteCoshBits; bits += 64) {
    check_point(bits);
  }
  constexpr uint32_t kSmallTransitionBits = 0x3f317218;  // 0.69314718f
  for (uint32_t bits = kSmallTransitionBits - 4096;
       bits <= kSmallTransitionBits + 4096; ++bits) {
    check_point(bits);
  }
  constexpr uint32_t kTransitionBits = 0x3f8533bf;  // 1.0406416654586792f
  for (uint32_t bits = kTransitionBits - 4096; bits <= kTransitionBits + 4096;
       ++bits) {
    check_point(bits);
  }

  EXPECT_LE(max_int_ulp_diff, 1);
  EXPECT_LE(max_ulp_error, 0.89)
      << "max_ulp_error = " << max_ulp_error << " worst_x = " << worst_x;
}

TEST(TrigTest, CoshF32EdgeCases) {
  EXPECT_EQ(cosh_f32(0.0f), 1.0f);
  EXPECT_EQ(cosh_f32(-0.0f), 1.0f);

  // Overflow threshold: cosh(89.4159851f) < FLT_MAX, cosh(89.4159927f) >
  // FLT_MAX
  float just_below = 89.4159851f;
  EXPECT_FALSE(std::isinf(cosh_f32(just_below)));
  EXPECT_THAT(
      cosh_f32(just_below),
      NearUlps(static_cast<float>(std::cosh(static_cast<double>(just_below))),
               1));
  EXPECT_FALSE(std::isinf(cosh_f32(-just_below)));
  EXPECT_THAT(
      cosh_f32(-just_below),
      NearUlps(static_cast<float>(std::cosh(static_cast<double>(-just_below))),
               1));

  float just_above = 89.4159927f;
  EXPECT_EQ(cosh_f32(just_above), std::numeric_limits<float>::infinity());
  EXPECT_EQ(cosh_f32(-just_above), std::numeric_limits<float>::infinity());

  EXPECT_EQ(cosh_f32(std::numeric_limits<float>::infinity()),
            std::numeric_limits<float>::infinity());
  EXPECT_EQ(cosh_f32(-std::numeric_limits<float>::infinity()),
            std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isnan(cosh_f32(std::numeric_limits<float>::quiet_NaN())));
}

TEST(TrigTest, CoshF32VectorWidthsAreCorrect) {
  Vec4f x4 = {-10.0f, -0.5f, 0.5f, 10.0f};
  Vec4f y4 = cosh_v4f32(x4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(
        y4[i],
        NearUlps(static_cast<float>(std::cosh(static_cast<double>(x4[i]))), 1));
  }

  Vec8f x8 = {-50.0f, -10.0f, -1.0f, -0.1f, 0.1f, 1.0f, 10.0f, 50.0f};
  Vec8f y8 = cosh_v8f32(x8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(
        y8[i],
        NearUlps(static_cast<float>(std::cosh(static_cast<double>(x8[i]))), 1));
  }

  Vec16f x16 = {-80.0f, -50.0f, -20.0f, -5.0f, -1.5f, -0.5f, -0.01f, -0.0f,
                0.0f,   0.01f,  0.5f,   1.5f,  5.0f,  20.0f, 50.0f,  80.0f};
  Vec16f y16 = cosh_v16f32(x16);
  for (int i = 0; i < 16; ++i) {
    EXPECT_THAT(
        y16[i],
        NearUlps(static_cast<float>(std::cosh(static_cast<double>(x16[i]))),
                 1));
  }
}

TEST(TrigTest, CoshIsVectorized) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kTrigLlIr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);
  EXPECT_THAT(ir, ContainsRegex("xla.cosh.v16f32"));
  EXPECT_THAT(ir, ContainsRegex("fmul.*<16 x float>"));
}

TEST(TrigTest, AsinF32IsCorrect) {
  double max_ulp_error = 0.0;
  float worst_x = 0.0f;
  int64_t max_int_ulp_diff = 0;

  auto check_point = [&](uint32_t bits) {
    float x = absl::bit_cast<float>(bits);
    double expected = std::asin(static_cast<double>(x));
    float actual = asin_f32(x);

    float expected_f32 = static_cast<float>(expected);
    uint32_t actual_bits = absl::bit_cast<uint32_t>(actual);
    uint32_t expected_bits = absl::bit_cast<uint32_t>(expected_f32);
    int64_t int_diff = std::abs(static_cast<int64_t>(actual_bits) -
                                static_cast<int64_t>(expected_bits));
    if (int_diff > max_int_ulp_diff) {
      max_int_ulp_diff = int_diff;
    }

    if (expected == 0.0) {
      EXPECT_EQ(actual, expected_f32);
      return;
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

  constexpr uint32_t kOneBits = 0x3f800000;
  constexpr uint32_t kHalfBits = 0x3f000000;
  constexpr uint32_t kSignBit = 0x80000000;

  // Sweep [0.0f, 1.0f] and [-1.0f, -0.0f] with stride 256.
  for (uint32_t bits = 0; bits <= kOneBits; bits += 256) {
    check_point(bits);
    check_point(bits | kSignBit);
  }
  // Exhaustive checks around the branch threshold |x| = 0.5f and boundary |x| =
  // 1.0f.
  for (uint32_t bits = kHalfBits - 8192; bits <= kHalfBits + 8192; ++bits) {
    check_point(bits);
    check_point(bits | kSignBit);
  }
  for (uint32_t bits = kOneBits - 65536; bits <= kOneBits; ++bits) {
    check_point(bits);
    check_point(bits | kSignBit);
  }

  EXPECT_LE(max_int_ulp_diff, 1) << "worst_x = " << worst_x;
  EXPECT_LT(max_ulp_error, 1.0) << "worst_x = " << worst_x;
}

TEST(TrigTest, AsinF32EdgeCases) {
  EXPECT_EQ(asin_f32(0.0f), 0.0f);
  EXPECT_FALSE(std::signbit(asin_f32(0.0f)));
  EXPECT_EQ(asin_f32(-0.0f), -0.0f);
  EXPECT_TRUE(std::signbit(asin_f32(-0.0f)));

  float denorm_min = std::numeric_limits<float>::denorm_min();
  EXPECT_EQ(asin_f32(denorm_min), denorm_min);
  EXPECT_EQ(asin_f32(-denorm_min), -denorm_min);

  EXPECT_THAT(asin_f32(1.0f), NearUlps(static_cast<float>(std::asin(1.0)), 1));
  EXPECT_THAT(asin_f32(-1.0f),
              NearUlps(static_cast<float>(std::asin(-1.0)), 1));

  // Out-of-domain inputs return NaN.
  EXPECT_TRUE(std::isnan(asin_f32(std::nextafter(1.0f, 2.0f))));
  EXPECT_TRUE(std::isnan(asin_f32(std::nextafter(-1.0f, -2.0f))));
  EXPECT_TRUE(std::isnan(asin_f32(2.0f)));
  EXPECT_TRUE(std::isnan(asin_f32(-2.0f)));
  EXPECT_TRUE(std::isnan(asin_f32(std::numeric_limits<float>::infinity())));
  EXPECT_TRUE(std::isnan(asin_f32(-std::numeric_limits<float>::infinity())));
  EXPECT_TRUE(std::isnan(asin_f32(std::numeric_limits<float>::quiet_NaN())));
}

TEST(TrigTest, AsinF32VectorWidthsAreCorrect) {
  Vec4f x4 = {-1.0f, -0.5f, 0.5f, 1.0f};
  Vec4f y4 = asin_v4f32(x4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(
        y4[i],
        NearUlps(static_cast<float>(std::asin(static_cast<double>(x4[i]))), 1));
  }

  Vec8f x8 = {-1.0f, -0.9f, -0.5f, -0.1f, 0.1f, 0.5f, 0.9f, 1.0f};
  Vec8f y8 = asin_v8f32(x8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(
        y8[i],
        NearUlps(static_cast<float>(std::asin(static_cast<double>(x8[i]))), 1));
  }

  Vec16f x16 = {-2.0f, -1.0f, -0.99f, -0.75f, -0.5f, -0.25f, -0.01f, -0.0f,
                0.0f,  0.01f, 0.25f,  0.5f,   0.75f, 0.99f,  1.0f,   2.0f};
  Vec16f y16 = asin_v16f32(x16);
  EXPECT_TRUE(std::isnan(y16[0]));
  EXPECT_TRUE(std::isnan(y16[15]));
  for (int i = 1; i < 15; ++i) {
    EXPECT_THAT(
        y16[i],
        NearUlps(static_cast<float>(std::asin(static_cast<double>(x16[i]))),
                 1));
  }
}

TEST(TrigTest, AsinIsVectorized) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kTrigLlIr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);
  EXPECT_THAT(ir, ContainsRegex("xla.asin.v16f32"));
  EXPECT_THAT(ir, ContainsRegex("fmul.*<16 x float>"));
}

TEST(TrigTest, AcosF32IsCorrect) {
  double max_ulp_error = 0.0;
  float worst_x = 0.0f;
  int64_t max_int_ulp_diff = 0;

  auto check_point = [&](uint32_t bits) {
    float x = absl::bit_cast<float>(bits);
    double expected = std::acos(static_cast<double>(x));
    float actual = acos_f32(x);

    float expected_f32 = static_cast<float>(expected);
    uint32_t actual_bits = absl::bit_cast<uint32_t>(actual);
    uint32_t expected_bits = absl::bit_cast<uint32_t>(expected_f32);
    int64_t int_diff = std::abs(static_cast<int64_t>(actual_bits) -
                                static_cast<int64_t>(expected_bits));
    if (int_diff > max_int_ulp_diff) {
      max_int_ulp_diff = int_diff;
    }

    if (expected == 0.0) {
      EXPECT_EQ(actual, expected_f32);
      return;
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

  constexpr uint32_t kOneBits = 0x3f800000;
  constexpr uint32_t kHalfBits = 0x3f000000;
  constexpr uint32_t kSignBit = 0x80000000;

  // Sweep [0.0f, 1.0f] and [-1.0f, -0.0f] with stride 256.
  for (uint32_t bits = 0; bits <= kOneBits; bits += 256) {
    check_point(bits);
    check_point(bits | kSignBit);
  }
  // Exhaustive checks around the branch threshold |x| = 0.5f and boundary |x| =
  // 1.0f.
  for (uint32_t bits = kHalfBits - 8192; bits <= kHalfBits + 8192; ++bits) {
    check_point(bits);
    check_point(bits | kSignBit);
  }
  for (uint32_t bits = kOneBits - 65536; bits <= kOneBits; ++bits) {
    check_point(bits);
    check_point(bits | kSignBit);
  }

  EXPECT_LE(max_int_ulp_diff, 1) << "worst_x = " << worst_x;
  EXPECT_LT(max_ulp_error, 1.0) << "worst_x = " << worst_x;
}

TEST(TrigTest, AcosF32EdgeCases) {
  EXPECT_EQ(acos_f32(1.0f), 0.0f);
  EXPECT_FALSE(std::signbit(acos_f32(1.0f)));
  EXPECT_THAT(acos_f32(-1.0f),
              NearUlps(static_cast<float>(std::acos(-1.0)), 1));
  EXPECT_THAT(acos_f32(0.0f), NearUlps(static_cast<float>(std::acos(0.0)), 1));
  EXPECT_THAT(acos_f32(-0.0f),
              NearUlps(static_cast<float>(std::acos(-0.0)), 1));

  // Out-of-domain inputs return NaN.
  EXPECT_TRUE(std::isnan(acos_f32(std::nextafter(1.0f, 2.0f))));
  EXPECT_TRUE(std::isnan(acos_f32(std::nextafter(-1.0f, -2.0f))));
  EXPECT_TRUE(std::isnan(acos_f32(2.0f)));
  EXPECT_TRUE(std::isnan(acos_f32(-2.0f)));
  EXPECT_TRUE(std::isnan(acos_f32(std::numeric_limits<float>::infinity())));
  EXPECT_TRUE(std::isnan(acos_f32(-std::numeric_limits<float>::infinity())));
  EXPECT_TRUE(std::isnan(acos_f32(std::numeric_limits<float>::quiet_NaN())));
}

TEST(TrigTest, AcosF32VectorWidthsAreCorrect) {
  Vec4f x4 = {-1.0f, -0.5f, 0.5f, 1.0f};
  Vec4f y4 = acos_v4f32(x4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(
        y4[i],
        NearUlps(static_cast<float>(std::acos(static_cast<double>(x4[i]))), 1));
  }

  Vec8f x8 = {-1.0f, -0.9f, -0.5f, -0.1f, 0.1f, 0.5f, 0.9f, 1.0f};
  Vec8f y8 = acos_v8f32(x8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(
        y8[i],
        NearUlps(static_cast<float>(std::acos(static_cast<double>(x8[i]))), 1));
  }

  Vec16f x16 = {-2.0f, -1.0f, -0.99f, -0.75f, -0.5f, -0.25f, -0.01f, -0.0f,
                0.0f,  0.01f, 0.25f,  0.5f,   0.75f, 0.99f,  1.0f,   2.0f};
  Vec16f y16 = acos_v16f32(x16);
  EXPECT_TRUE(std::isnan(y16[0]));
  EXPECT_TRUE(std::isnan(y16[15]));
  for (int i = 1; i < 15; ++i) {
    EXPECT_THAT(
        y16[i],
        NearUlps(static_cast<float>(std::acos(static_cast<double>(x16[i]))),
                 1));
  }
}

TEST(TrigTest, AcosIsVectorized) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kTrigLlIr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);
  EXPECT_THAT(ir, ContainsRegex("xla.acos.v16f32"));
  EXPECT_THAT(ir, ContainsRegex("fmul.*<16 x float>"));
}

}  // namespace
}  // namespace xla::codegen
