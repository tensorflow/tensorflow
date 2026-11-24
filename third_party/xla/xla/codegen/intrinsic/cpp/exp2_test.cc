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

#include "xla/codegen/intrinsic/cpp/exp2.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/cpp/exp2_ll.h"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"
#include "xla/codegen/intrinsic/test_matchers.h"

namespace xla::codegen::intrinsics {
namespace {

using ::testing::AllOf;
using ::testing::ContainsRegex;
using ::testing::Not;
using ::xla::codegen::intrinsic::NearUlps;

TEST(Exp2Test, BitcodeIsVectorizedAndPortable) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kExp2LlIr);
  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);

  EXPECT_THAT(
      ir,
      AllOf(ContainsRegex("define .*@xla\\.exp2\\.f32\\(float"),
            ContainsRegex("define .*@xla\\.exp2\\.v4f32\\(<4 x float>"),
            ContainsRegex("define .*@xla\\.exp2\\.v8f32\\(.*<8 x float>"),
            ContainsRegex("define .*@xla\\.exp2\\.v16f32\\(.*<16 x float>"),
            ContainsRegex("define .*@xla\\.exp2\\.f64\\(double"),
            ContainsRegex("define .*@xla\\.exp2\\.v2f64\\(<2 x double>"),
            ContainsRegex("define .*@xla\\.exp2\\.v4f64\\(.*<4 x double>"),
            ContainsRegex("define .*@xla\\.exp2\\.v8f64\\(.*<8 x double>"),
            ContainsRegex("fmul <4 x float>"),
            ContainsRegex("fmul <4 x double>"), ContainsRegex("llvm.fmuladd"),
            Not(ContainsRegex("llvm\\.fma")), Not(ContainsRegex("llvm\\.x86")),
            Not(ContainsRegex("llvm\\.aarch64")),
            Not(ContainsRegex("llvm\\.rint"))));
}

TEST(Exp2Test, ScalarAndVectorF32AccuracyAndSpecialCases) {
  auto ref_exp2 = [](float x) {
    return static_cast<float>(std::exp2(static_cast<double>(x)));
  };
  const float special_inputs[] = {
      0.0f,
      -0.0f,
      1.0f,
      -1.0f,
      0.5f,
      -0.5f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::signaling_NaN(),
      std::numeric_limits<float>::denorm_min(),
      -std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),
      -std::numeric_limits<float>::min(),
      127.0f,
      127.99999f,
      128.0f,
      129.0f,
      -126.0f,
      -149.0f,
      -149.5f,
      -150.0f,
      -151.0f,
      -200.0f,
  };
  for (float x : special_inputs) {
    EXPECT_THAT(exp2_f32(x), NearUlps(ref_exp2(x), 1)) << "x = " << x;
  }
  EXPECT_FALSE(std::signbit(exp2_f32(-std::numeric_limits<float>::infinity())));

  for (int i = -15200; i <= 12900; ++i) {
    float x = static_cast<float>(i) * 0.01f;
    EXPECT_THAT(exp2_f32(x), NearUlps(ref_exp2(x), 1)) << "x = " << x;
  }

  Vec4f v4 = {0.0f, -0.5f, 1.5f, 128.0f};
  Vec4f r4 = exp2_v4f32(v4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(r4[i], NearUlps(ref_exp2(v4[i]), 1));
  }

  Vec8f v8 = {0.0f,    -0.0f, -149.0f,
              -126.5f, 0.25f, std::numeric_limits<float>::quiet_NaN(),
              -151.0f, 128.0f};
  Vec8f r8 = exp2_v8f32(v8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(r8[i], NearUlps(ref_exp2(v8[i]), 1));
  }

  Vec16f v16 = {0.0f,
                -0.0f,
                1.0f,
                -1.0f,
                0.5f,
                -0.5f,
                127.5f,
                128.0f,
                -126.0f,
                -140.0f,
                -149.0f,
                -151.0f,
                1e-6f,
                -1e-6f,
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity()};
  Vec16f r16 = exp2_v16f32(v16);
  for (int i = 0; i < 16; ++i) {
    EXPECT_THAT(r16[i], NearUlps(ref_exp2(v16[i]), 1));
  }
}

TEST(Exp2Test, ScalarAndVectorF64AccuracyAndSpecialCases) {
  auto ref_exp2 = [](double x) { return std::exp2(x); };
  const double special_inputs[] = {
      0.0,
      -0.0,
      1.0,
      -1.0,
      0.5,
      -0.5,
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::signaling_NaN(),
      std::numeric_limits<double>::denorm_min(),
      -std::numeric_limits<double>::denorm_min(),
      std::numeric_limits<double>::min(),
      -std::numeric_limits<double>::min(),
      1023.0,
      1023.9999999999999,
      1024.0,
      1025.0,
      -1022.0,
      -1050.5,
      -1074.0,
      -1075.0,
      -1076.0,
      -2000.0,
  };
  for (double x : special_inputs) {
    EXPECT_THAT(exp2_f64(x), NearUlps(ref_exp2(x), 1)) << "x = " << x;
  }
  EXPECT_FALSE(
      std::signbit(exp2_f64(-std::numeric_limits<double>::infinity())));

  for (int i = -10770; i <= 10250; ++i) {
    double x = static_cast<double>(i) * 0.1;
    EXPECT_THAT(exp2_f64(x), NearUlps(ref_exp2(x), 1)) << "x = " << x;
  }

  Vec2d v2 = {-1074.0, 1023.5};
  Vec2d r2 = exp2_v2f64(v2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_THAT(r2[i], NearUlps(ref_exp2(v2[i]), 1));
  }

  Vec4d v4 = {0.0, -0.5, 1.5, 1024.0};
  Vec4d r4 = exp2_v4f64(v4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(r4[i], NearUlps(ref_exp2(v4[i]), 1));
  }

  Vec8d v8 = {0.0,     -0.0,  -1074.0,
              -1022.5, 0.25,  std::numeric_limits<double>::quiet_NaN(),
              -1076.0, 1024.0};
  Vec8d r8 = exp2_v8f64(v8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(r8[i], NearUlps(ref_exp2(v8[i]), 1));
  }
}

}  // namespace
}  // namespace xla::codegen::intrinsics
