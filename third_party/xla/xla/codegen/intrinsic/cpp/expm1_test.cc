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

#include "xla/codegen/intrinsic/cpp/expm1.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/cpp/expm1_32_ll.h"
#include "xla/codegen/intrinsic/cpp/expm1_64_ll.h"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"
#include "xla/types.h"

namespace xla::codegen::intrinsics {
namespace {

using ::testing::AllOf;
using ::testing::ContainsRegex;
using ::testing::Not;

TEST(Expm1Test, BitcodeIsVectorizedAndPortable) {
  for (const std::string& bitcode :
       {llvm_ir::kExpm132LlIr, llvm_ir::kExpm164LlIr}) {
    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module =
        ParseEmbeddedBitcode(context, bitcode);
    std::string ir;
    llvm::raw_string_ostream stream(ir);
    module->print(stream, nullptr);

    EXPECT_THAT(
        ir,
        AllOf(ContainsRegex("define .*@xla.expm1.f32\\(float"),
              ContainsRegex("define .*@xla.expm1.v4f32\\(<4 x float>"),
              ContainsRegex("define .*@xla.expm1.v8f32\\(<8 x float>"),
              ContainsRegex("define .*@xla.expm1.v16f32\\(.*<16 x float>"),
              ContainsRegex("define .*@xla.expm1.f64\\(double"),
              ContainsRegex("define .*@xla.expm1.v4f64\\(<4 x double>"),
              ContainsRegex("define .*@xla.expm1.v8f64\\(.*<8 x double>"),
              ContainsRegex("fmul <4 x float>"),
              ContainsRegex("fmul <4 x double>"), ContainsRegex("llvm.fmuladd"),
              Not(ContainsRegex("llvm\\.fma")),
              Not(ContainsRegex("llvm\\.x86")),
              Not(ContainsRegex("llvm\\.aarch64"))));
  }
}

float ComputeUlpErrorF32(float actual, double expected) {
  if (std::isnan(expected)) {
    return std::isnan(actual) ? 0.0f : std::numeric_limits<float>::infinity();
  }
  float expected_f32 = static_cast<float>(expected);
  if (std::isinf(expected_f32)) {
    return (std::isinf(actual) &&
            std::signbit(actual) == std::signbit(expected_f32))
               ? 0.0f
               : std::numeric_limits<float>::infinity();
  }
  if (expected == 0.0) {
    return (actual == 0.0f && std::signbit(actual) == std::signbit(expected))
               ? 0.0f
               : std::numeric_limits<float>::infinity();
  }
  float direction = (static_cast<double>(actual) < expected)
                        ? -std::numeric_limits<float>::infinity()
                        : std::numeric_limits<float>::infinity();
  float next = std::nextafter(expected_f32, direction);
  double ulp =
      std::abs(static_cast<double>(next) - static_cast<double>(expected_f32));
  return static_cast<float>(std::abs(static_cast<double>(actual) - expected) /
                            ulp);
}

double ComputeUlpErrorF64(double actual, double expected) {
  if (std::isnan(expected)) {
    return std::isnan(actual) ? 0.0 : std::numeric_limits<double>::infinity();
  }
  if (std::isinf(expected)) {
    return (std::isinf(actual) &&
            std::signbit(actual) == std::signbit(expected))
               ? 0.0
               : std::numeric_limits<double>::infinity();
  }
  if (expected == 0.0) {
    return (actual == 0.0 && std::signbit(actual) == std::signbit(expected))
               ? 0.0
               : std::numeric_limits<double>::infinity();
  }
  double direction = (actual < expected)
                         ? -std::numeric_limits<double>::infinity()
                         : std::numeric_limits<double>::infinity();
  double next = std::nextafter(expected, direction);
  double ulp = std::abs(next - expected);
  return std::abs(actual - expected) / ulp;
}

TEST(Expm1Test, ScalarF32AccuracyAndSpecialCases) {
  const float special_inputs[] = {
      0.0f,
      -0.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::signaling_NaN(),
      std::numeric_limits<float>::denorm_min(),
      -std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),
      -std::numeric_limits<float>::min(),
      88.0f,
      88.722839f,
      89.0f,
      -17.32868f,
      -20.0f,
      -100.0f,
      -0.69314718f,  // -ln(2) where expm1(x) = -0.5
      -0.34657359f,  // -0.5 * ln(2)
      0.34657359f,   // +0.5 * ln(2)
      0.69314718f,   // +ln(2)
      1e-4f,
      -1e-4f,
      1e-7f,
      -1e-7f,
  };
  for (float x : special_inputs) {
    float actual = expm1_f32(x);
    double expected = std::expm1(static_cast<double>(x));
    EXPECT_LE(ComputeUlpErrorF32(actual, expected), 1.0f)
        << "x = " << x << ", actual = " << actual
        << ", expected = " << static_cast<float>(expected);
  }

  for (int i = -10000; i <= 10000; ++i) {
    float x = static_cast<float>(i) * 0.01f;
    float actual = expm1_f32(x);
    double expected = std::expm1(static_cast<double>(x));
    EXPECT_LE(ComputeUlpErrorF32(actual, expected), 1.0f)
        << "x = " << x << ", actual = " << actual
        << ", expected = " << static_cast<float>(expected);
  }
}

TEST(Expm1Test, VectorF32MatchesScalar) {
  Vec4f v4 = {0.0f, -0.69314718f, 1.5f, 89.0f};
  Vec4f r4 = expm1_v4f32(v4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_LE(ComputeUlpErrorF32(r4[i], std::expm1(static_cast<double>(v4[i]))),
              1.0f);
  }

  Vec8f v8 = {0.0f,        -0.0f, -0.69314718f,
              0.69314718f, 1e-5f, std::numeric_limits<float>::quiet_NaN(),
              -20.0f,      89.0f};
  Vec8f r8 = expm1_v8f32(v8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_LE(ComputeUlpErrorF32(r8[i], std::expm1(static_cast<double>(v8[i]))),
              1.0f);
  }
}

TEST(Expm1Test, ScalarF64AccuracyAndSpecialCases) {
  const double special_inputs[] = {
      0.0,
      -0.0,
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::signaling_NaN(),
      std::numeric_limits<double>::denorm_min(),
      -std::numeric_limits<double>::denorm_min(),
      std::numeric_limits<double>::min(),
      -std::numeric_limits<double>::min(),
      709.0,
      709.78271289338397,
      710.0,
      -37.42994775023705,
      -40.0,
      -100.0,
      -0.6931471805599453,  // -ln(2)
      -0.3465735902799726,  // -0.5 * ln(2)
      0.3465735902799726,   // +0.5 * ln(2)
      0.6931471805599453,   // +ln(2)
      1e-8,
      -1e-8,
      1e-15,
      -1e-15,
  };
  for (double x : special_inputs) {
    double actual = expm1_f64(x);
    double expected = std::expm1(x);
    EXPECT_LE(ComputeUlpErrorF64(actual, expected), 1.0)
        << "x = " << x << ", actual = " << actual
        << ", expected = " << expected;
  }

  for (int i = -10000; i <= 10000; ++i) {
    double x = static_cast<double>(i) * 0.07;
    double actual = expm1_f64(x);
    double expected = std::expm1(x);
    EXPECT_LE(ComputeUlpErrorF64(actual, expected), 1.0)
        << "x = " << x << ", actual = " << actual
        << ", expected = " << expected;
  }
}

TEST(Expm1Test, VectorF64MatchesScalar) {
  Vec4d v4 = {0.0, -0.6931471805599453, 1.5, 710.0};
  Vec4d r4 = expm1_v4f64(v4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_LE(ComputeUlpErrorF64(r4[i], std::expm1(v4[i])), 1.0);
  }

  Vec8d v8 = {0.0,
              -0.0,
              -0.6931471805599453,
              0.6931471805599453,
              1e-9,
              std::numeric_limits<double>::quiet_NaN(),
              -40.0,
              710.0};
  Vec8d r8 = expm1_v8f64(v8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_LE(ComputeUlpErrorF64(r8[i], std::expm1(v8[i])), 1.0);
  }
}

TEST(Expm1Test, ExhaustiveF16AndBF16ZeroUlpViaF32) {
  for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
    xla::half h = absl::bit_cast<xla::half>(static_cast<uint16_t>(bits));
    float xf = static_cast<float>(h);
    xla::half actual = static_cast<xla::half>(expm1_f32(xf));
    if (std::isnan(xf)) {
      EXPECT_TRUE(std::isnan(static_cast<float>(actual)));
      continue;
    }
    xla::half expected =
        static_cast<xla::half>(std::expm1(static_cast<double>(xf)));
    if (std::isnan(static_cast<float>(expected))) {
      EXPECT_TRUE(std::isnan(static_cast<float>(actual)));
    } else {
      EXPECT_EQ(absl::bit_cast<uint16_t>(actual),
                absl::bit_cast<uint16_t>(expected))
          << "F16 mismatch at x = " << xf;
    }
  }

  for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
    xla::bfloat16 bf =
        absl::bit_cast<xla::bfloat16>(static_cast<uint16_t>(bits));
    float xbf = static_cast<float>(bf);
    xla::bfloat16 actual_bf = static_cast<xla::bfloat16>(expm1_f32(xbf));
    if (std::isnan(xbf)) {
      EXPECT_TRUE(std::isnan(static_cast<float>(actual_bf)));
      continue;
    }
    xla::bfloat16 expected_bf =
        static_cast<xla::bfloat16>(std::expm1(static_cast<double>(xbf)));
    if (std::isnan(static_cast<float>(expected_bf))) {
      EXPECT_TRUE(std::isnan(static_cast<float>(actual_bf)));
    } else {
      EXPECT_EQ(absl::bit_cast<uint16_t>(actual_bf),
                absl::bit_cast<uint16_t>(expected_bf))
          << "BF16 mismatch at x = " << xbf;
    }
  }
}

}  // namespace
}  // namespace xla::codegen::intrinsics
