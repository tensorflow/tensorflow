/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/intrinsic/cpp/atan.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/atan_ll.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"
#include "xla/codegen/intrinsic/test_matchers.h"

namespace xla::codegen {
namespace {
using ::testing::AnyOf;
using ::testing::ContainsRegex;
using ::testing::HasSubstr;
using ::testing::Not;
using ::xla::codegen::intrinsic::NearUlps;

std::string GetFunctionIr(const llvm::Module& module, llvm::StringRef name) {
  llvm::Function* f = module.getFunction(name);
  if (f == nullptr) {
    return "";
  }
  std::string ir;
  llvm::raw_string_ostream stream(ir);
  f->print(stream);
  return ir;
}

void ExpectOnlyIntrinsicDeclarations(const llvm::Module& module) {
  for (const llvm::Function& f : module) {
    if (f.isDeclaration()) {
      EXPECT_TRUE(f.isIntrinsic()) << f.getName().str();
    }
    EXPECT_FALSE(f.hasPersonalityFn()) << f.getName().str();
  }
}

constexpr int kAtanF32Ulps = 3;
constexpr int kAtanF64Ulps = 2;

TEST(AtanTest, AtanIsVectorized) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kAtanLlIr);
  ASSERT_NE(module, nullptr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);

  ExpectOnlyIntrinsicDeclarations(*module);
  EXPECT_NE(module->getFunction("xla.atan.f32"), nullptr);
  EXPECT_NE(module->getFunction("xla.atan.f64"), nullptr);
  // gtest's Windows regex engine has no alternation, so use AnyOf.
  struct Width {
    absl::string_view suffix;
    absl::string_view ir_type;
  };
  for (const Width& w :
       {Width{"v4f32", "<4 x float>"}, Width{"v8f32", "<8 x float>"},
        Width{"v16f32", "<16 x float>"}, Width{"v2f64", "<2 x double>"},
        Width{"v4f64", "<4 x double>"}, Width{"v8f64", "<8 x double>"}}) {
    SCOPED_TRACE(w.suffix);
    EXPECT_THAT(GetFunctionIr(*module, absl::StrCat("xla.atan.", w.suffix)),
                AnyOf(HasSubstr(absl::StrCat("fmuladd.", w.suffix)),
                      HasSubstr(absl::StrCat("fmul ", w.ir_type))));
  }
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.atan")));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.x86")));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.aarch64")));
}

TEST(AtanTest, v2f64AtanIsCorrect) {
  Vec2d x = {0.5, -3.0};
  Vec2d y = atan_v2f64(x);
  for (int i = 0; i < 2; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF64Ulps));
  }
}

TEST(AtanTest, FastAtanfIsCorrect) {
  Vec16f x = {1.0f,  2.0f,  -1.0f, 4.0f,   8.0f,   16.0f,  32.0f, 200.0f,
              -2.0f, -4.0f, -8.0f, -16.0f, -32.0f, -64.0f, 0.0f,  0.5f};
  Vec16f y = atan_v16f32(x);
  for (int i = 0; i < 16; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF32Ulps))
        << x[i] << " " << std::atan(x[i]);
  }
}

TEST(AtanTest, FastAtandIsCorrect) {
  Vec4d x = {1.0, 2.0, -1.0, 4.0};
  Vec4d y = atan_v4f64(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF64Ulps));
  }
}

TEST(AtanTest, LinspaceInputsAtanfCorrectness) {
  constexpr float start = -100.0f;
  constexpr float end = 100.0f;
  constexpr int steps = 1000 * 16;
  constexpr float step = (end - start) / steps;

  for (int i = 0; i < steps; i += 16) {
    Vec16f x;
    for (int j = 0; j < 16; ++j) {
      x[j] = start + (i + j) * step;
    }
    Vec16f y = atan_v16f32(x);
    for (int j = 0; j < 16; ++j) {
      EXPECT_THAT(y[j], NearUlps(std::atan(x[j]), kAtanF32Ulps));
    }
  }
}

TEST(AtanTest, v4f32AtanIsCorrect) {
  Vec4f x = {1.0f, 2.0f, -1.0f, 4.0f};
  Vec4f y = atan_v4f32(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF32Ulps));
  }
}

TEST(AtanTest, v8f32AtanIsCorrect) {
  Vec8f x = {1.0f, 2.0f, -1.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f};
  Vec8f y = atan_v8f32(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF32Ulps));
  }
}

TEST(AtanTest, v8f64AtanIsCorrect) {
  Vec8d x = {1.0, 2.0, -1.0, 4.0, 8.0, 16.0, 32.0, 64.0};
  Vec8d y = atan_v8f64(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF64Ulps));
  }
}

TEST(AtanTest, AtanEdgeCases) {
  constexpr float kPiOver2f = 1.57079632679489661923f;
  constexpr double kPiOver2 = 1.57079632679489661923;

  // Float NaN
  float nan_f = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(std::isnan(atan_f32(nan_f)));

  // Float Infinity
  float inf_f = std::numeric_limits<float>::infinity();
  EXPECT_NEAR(atan_f32(inf_f), kPiOver2f, 1e-6f);
  EXPECT_NEAR(atan_f32(-inf_f), -kPiOver2f, 1e-6f);

  // Double NaN
  double nan_d = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(atan_f64(nan_d)));

  // Double Infinity
  double inf_d = std::numeric_limits<double>::infinity();
  EXPECT_NEAR(atan_f64(inf_d), kPiOver2, 1e-14);
  EXPECT_NEAR(atan_f64(-inf_d), -kPiOver2, 1e-14);
}

}  // namespace
}  // namespace xla::codegen
