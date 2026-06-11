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

#include "xla/codegen/intrinsic/cpp/eigen_unary.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_32_ll.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_64_ll.h"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/test_matchers.h"

#if defined(__has_attribute) && __has_attribute(ext_vector_type) && \
    defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

namespace xla::codegen {
namespace {
using ::testing::ContainsRegex;
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

constexpr int kTanhUlps = 5;
constexpr int kAtanF32Ulps = 3;
constexpr int kAtanF64Ulps = 2;
constexpr int kSinF64Ulps = 3;
constexpr int kCosF64Ulps = 3;

#if defined(__x86_64__)
TEST(EigenUnaryTest, FastTanhfIsCorrect) {
  Vec16f x = {1.0f,  2.0f,  -1.0f, 4.0f,   8.0f,   16.0f,  32.0f, 200.0f,
              -2.0f, -4.0f, -8.0f, -16.0f, -32.0f, -64.0f, 0.0f,  0.5f};
  Vec16f y = tanh_v16f32(x);
  for (int i = 0; i < 16; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::tanh(x[i]), kTanhUlps))
        << x[i] << " " << std::tanh(x[i]);
  }
}

TEST(EigenUnaryTest, FastTanhdIsCorrect) {
  Vec4d x = {1.0, 2.0, -1.0, 4.0};
  Vec4d y = tanh_v4f64(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::tanh(x[i]), kTanhUlps));
  }
}

TEST(EigenUnaryTest, LinspaceInputsTanhfCorrectness) {
  constexpr float start = -100.0f;
  constexpr float end = 100.0f;
  constexpr int steps = 1000 * 16;
  constexpr float step = (end - start) / steps;

  for (int i = 0; i < steps; i += 16) {
    Vec16f x;
    for (int j = 0; j < 16; ++j) {
      x[j] = start + (i + j) * step;
    }
    Vec16f y = tanh_v16f32(x);
    for (int j = 0; j < 16; ++j) {
      EXPECT_THAT(y[j], NearUlps(std::tanh(x[j]), kTanhUlps));
    }
  }
}
#endif

TEST(EigenUnaryTest, v4f32TanhIsCorrect) {
  Vec4f x = {1.0f, 2.0f, -1.0f, 4.0f};
  Vec4f y = tanh_v4f32(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::tanh(x[i]), kTanhUlps));
  }
}

#if defined(__x86_64__)
TEST(EigenUnaryTest, v8f32TanhIsCorrect) {
  Vec8f x = {1.0f, 2.0f, -1.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f};
  Vec8f y = tanh_v8f32(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::tanh(x[i]), kTanhUlps));
  }
}

TEST(EigenUnaryTest, v8f64TanhIsCorrect) {
  Vec8d x = {1.0, 2.0, -1.0, 4.0, 8.0, 16.0, 32.0, 64.0};
  Vec8d y = tanh_v8f64(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::tanh(x[i]), kTanhUlps));
  }
}
#endif

TEST(EigenUnaryTest, FastTanhfIsVectorized32) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary32LlIr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);

  bool is_512 = absl::StrContains(ir, "fmul <16 x float>");
  bool is_256 = absl::StrContains(ir, "fmul <8 x float>");
  EXPECT_TRUE(is_512 || is_256);

  std::string v16f32_ir = GetFunctionIr(*module, "xla.unused.tanh.v16f32");
  std::string v8f32_ir = GetFunctionIr(*module, "xla.unused.tanh.v8f32");
  std::string v8f64_ir = GetFunctionIr(*module, "xla.unused.tanh.v8f64");
  std::string v4f64_ir = GetFunctionIr(*module, "xla.unused.tanh.v4f64");

  if (is_512) {
    EXPECT_THAT(v16f32_ir, ContainsRegex("fmul <16 x float>"));
    EXPECT_THAT(v8f64_ir, ContainsRegex("fmul <8 x double>"));
    EXPECT_THAT(v16f32_ir, ContainsRegex("<16 x float>.*f0x326F951E"));
  } else {
    EXPECT_THAT(v8f32_ir, ContainsRegex("fmul <8 x float>"));
    EXPECT_THAT(v4f64_ir, ContainsRegex("fmul <4 x double>"));
    EXPECT_THAT(v8f32_ir, ContainsRegex("<8 x float>.*f0x326F951E"));
    EXPECT_THAT(v16f32_ir, ContainsRegex("fmul <8 x float>"));
    EXPECT_THAT(v8f64_ir, ContainsRegex("fmul <4 x double>"));
  }
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.x86")));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.aarch64")));
  EXPECT_THAT(ir, ContainsRegex("xla.unused.tanh.v16f32"));
  EXPECT_THAT(ir, ContainsRegex("xla.unused.tanh.v8f64"));
}

TEST(EigenUnaryTest, FastTanhfIsVectorized64) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary64LlIr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);

  std::string v16f32_ir = GetFunctionIr(*module, "xla.unused.tanh.v16f32");
  std::string v8f64_ir = GetFunctionIr(*module, "xla.unused.tanh.v8f64");

  EXPECT_THAT(v16f32_ir, ContainsRegex("fmul <16 x float>"));
  EXPECT_THAT(v8f64_ir, ContainsRegex("fmul <8 x double>"));
  EXPECT_THAT(v16f32_ir, ContainsRegex("<16 x float>.*f0x326F951E"));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.x86")));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.aarch64")));
  EXPECT_THAT(ir, ContainsRegex("xla.unused.tanh.v16f32"));
  EXPECT_THAT(ir, ContainsRegex("xla.unused.tanh.v8f64"));
}

TEST(EigenUnaryTest, GetCppGenIrStringSelectsCorrectVectorWidth) {
  intrinsics::IntrinsicOptions options_default;
  EXPECT_EQ(GetCppGenIrString(options_default), llvm_ir::kEigenUnary32LlIr);

  options_default.features = "+avx512f";
  EXPECT_EQ(GetCppGenIrString(options_default), llvm_ir::kEigenUnary64LlIr);
}

#if defined(__x86_64__)
TEST(EigenUnaryTest, FastAtanfIsCorrect) {
  Vec16f x = {1.0f,  2.0f,  -1.0f, 4.0f,   8.0f,   16.0f,  32.0f, 200.0f,
              -2.0f, -4.0f, -8.0f, -16.0f, -32.0f, -64.0f, 0.0f,  0.5f};
  Vec16f y = atan_v16f32(x);
  for (int i = 0; i < 16; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF32Ulps))
        << x[i] << " " << std::atan(x[i]);
  }
}

TEST(EigenUnaryTest, FastAtandIsCorrect) {
  Vec4d x = {1.0, 2.0, -1.0, 4.0};
  Vec4d y = atan_v4f64(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF64Ulps));
  }
}

TEST(EigenUnaryTest, LinspaceInputsAtanfCorrectness) {
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
#endif

TEST(EigenUnaryTest, v4f32AtanIsCorrect) {
  Vec4f x = {1.0f, 2.0f, -1.0f, 4.0f};
  Vec4f y = atan_v4f32(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF32Ulps));
  }
}

#if defined(__x86_64__)
TEST(EigenUnaryTest, v8f32AtanIsCorrect) {
  Vec8f x = {1.0f, 2.0f, -1.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f};
  Vec8f y = atan_v8f32(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF32Ulps));
  }
}

TEST(EigenUnaryTest, v8f64AtanIsCorrect) {
  Vec8d x = {1.0, 2.0, -1.0, 4.0, 8.0, 16.0, 32.0, 64.0};
  Vec8d y = atan_v8f64(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF64Ulps));
  }
}
#endif

TEST(EigenUnaryTest, AtanIsVectorized32) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary32LlIr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);

  bool is_512 = absl::StrContains(ir, "fmul <16 x float>");
  bool is_256 = absl::StrContains(ir, "fmul <8 x float>");
  EXPECT_TRUE(is_512 || is_256);

  std::string v16f32_ir = GetFunctionIr(*module, "xla.atan.v16f32");
  std::string v8f32_ir = GetFunctionIr(*module, "xla.atan.v8f32");
  std::string v8f64_ir = GetFunctionIr(*module, "xla.atan.v8f64");
  std::string v4f64_ir = GetFunctionIr(*module, "xla.atan.v4f64");

  if (is_512) {
    EXPECT_THAT(v16f32_ir, ContainsRegex("fmul <16 x float>"));
    EXPECT_THAT(v8f64_ir, ContainsRegex("fmul <8 x double>"));
    EXPECT_THAT(v16f32_ir, ContainsRegex("<16 x float>.*f0x3DE56E67"));
    EXPECT_THAT(v8f64_ir, ContainsRegex("<8 x double>.*f0x3EFBF668DC1807E8"));
  } else {
    EXPECT_THAT(v8f32_ir, ContainsRegex("fmul <8 x float>"));
    EXPECT_THAT(v4f64_ir, ContainsRegex("fmul <4 x double>"));
    EXPECT_THAT(v8f32_ir, ContainsRegex("<8 x float>.*f0x3DE56E67"));
    EXPECT_THAT(v4f64_ir, ContainsRegex("<4 x double>.*f0x3EFBF668DC1807E8"));
    EXPECT_THAT(v16f32_ir, ContainsRegex("fmul <8 x float>"));
    EXPECT_THAT(v8f64_ir, ContainsRegex("fmul <4 x double>"));
  }
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.x86")));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.aarch64")));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.v16f32"));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.v8f64"));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.f32"));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.f64"));
}

TEST(EigenUnaryTest, AtanIsVectorized64) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary64LlIr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);

  std::string v16f32_ir = GetFunctionIr(*module, "xla.atan.v16f32");
  std::string v8f64_ir = GetFunctionIr(*module, "xla.atan.v8f64");

  EXPECT_THAT(v16f32_ir, ContainsRegex("fmul <16 x float>"));
  EXPECT_THAT(v8f64_ir, ContainsRegex("fmul <8 x double>"));
  EXPECT_THAT(v16f32_ir, ContainsRegex("<16 x float>.*f0x3DE56E67"));
  EXPECT_THAT(v8f64_ir, ContainsRegex("<8 x double>.*f0x3EFBF668DC1807E8"));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.x86")));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.aarch64")));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.v16f32"));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.v8f64"));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.f32"));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.f64"));
}

TEST(EigenUnaryTest, AtanEdgeCases) {
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

TEST(EigenUnaryTest, FastSindIsCorrect) {
  Vec2d x = {1.0, 2.0};
  Vec2d y = sin_v2f64(x);
  for (int i = 0; i < 2; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::sin(x[i]), kSinF64Ulps));
  }
}

#if defined(__x86_64__)
TEST(EigenUnaryTest, v4f64SinIsCorrect) {
  Vec4d x = {1.0, 2.0, -1.0, 4.0};
  Vec4d y = sin_v4f64(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::sin(x[i]), kSinF64Ulps));
  }
}

TEST(EigenUnaryTest, v8f64SinIsCorrect) {
  Vec8d x = {1.0, 2.0, -1.0, 4.0, 8.0, 16.0, 32.0, 64.0};
  Vec8d y = sin_v8f64(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::sin(x[i]), kSinF64Ulps));
  }
}

TEST(EigenUnaryTest, LinspaceInputsSindCorrectnessVec8d) {
  constexpr double start = -100.0;
  constexpr double end = 100.0;
  constexpr int steps = 1000 * 8;
  constexpr double step = (end - start) / steps;

  for (int i = 0; i < steps; i += 8) {
    Vec8d x;
    for (int j = 0; j < 8; ++j) {
      x[j] = start + (i + j) * step;
    }
    Vec8d y = sin_v8f64(x);
    for (int j = 0; j < 8; ++j) {
      EXPECT_THAT(y[j], NearUlps(std::sin(x[j]), kSinF64Ulps));
    }
  }
}
#endif

TEST(EigenUnaryTest, FastCosdIsCorrect) {
  Vec2d x = {1.0, 2.0};
  Vec2d y = cos_v2f64(x);
  for (int i = 0; i < 2; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::cos(x[i]), kCosF64Ulps));
  }
}

#if defined(__x86_64__)
TEST(EigenUnaryTest, v4f64CosIsCorrect) {
  Vec4d x = {1.0, 2.0, -1.0, 4.0};
  Vec4d y = cos_v4f64(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::cos(x[i]), kCosF64Ulps));
  }
}

TEST(EigenUnaryTest, v8f64CosIsCorrect) {
  Vec8d x = {1.0, 2.0, -1.0, 4.0, 8.0, 16.0, 32.0, 64.0};
  Vec8d y = cos_v8f64(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::cos(x[i]), kCosF64Ulps));
  }
}

TEST(EigenUnaryTest, LinspaceInputsCosdCorrectnessVec8d) {
  constexpr double start = -100.0;
  constexpr double end = 100.0;
  constexpr int steps = 1000 * 8;
  constexpr double step = (end - start) / steps;

  for (int i = 0; i < steps; i += 8) {
    Vec8d x;
    for (int j = 0; j < 8; ++j) {
      x[j] = start + (i + j) * step;
    }
    Vec8d y = cos_v8f64(x);
    for (int j = 0; j < 8; ++j) {
      EXPECT_THAT(y[j], NearUlps(std::cos(x[j]), kCosF64Ulps));
    }
  }
}
#endif

TEST(EigenUnaryTest, SinCosEdgeCasesF64) {
  // NaN
  double nan_d = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(sin_f64(nan_d)));
  EXPECT_TRUE(std::isnan(cos_f64(nan_d)));

  // Infinity
  double inf_d = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(std::isnan(sin_f64(inf_d)));
  EXPECT_TRUE(std::isnan(cos_f64(inf_d)));

  // Huge inputs (> 1e14)
  EXPECT_NEAR(sin_f64(1e15), std::sin(1e15), 1e-10);
  EXPECT_NEAR(cos_f64(1e15), std::cos(1e15), 1e-10);
}

TEST(EigenUnaryTest, SinCosIsVectorized32) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary32LlIr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);

  EXPECT_THAT(ir, ContainsRegex("xla.sin.v4f64"));
  EXPECT_THAT(ir, ContainsRegex("xla.cos.v4f64"));

  std::string v4f64_sin_ir = GetFunctionIr(*module, "xla.sin.v4f64");
  std::string v4f64_cos_ir = GetFunctionIr(*module, "xla.cos.v4f64");

  EXPECT_THAT(v4f64_sin_ir, ContainsRegex("fmul <4 x double>"));
  EXPECT_THAT(v4f64_cos_ir, ContainsRegex("fmul <4 x double>"));
  EXPECT_THAT(v4f64_cos_ir, ContainsRegex("<4 x double>.*f0x41933FD174000000"));
}

TEST(EigenUnaryTest, SinCosIsVectorized64) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary64LlIr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);

  EXPECT_THAT(ir, ContainsRegex("xla.sin.v8f64"));
  EXPECT_THAT(ir, ContainsRegex("xla.cos.v8f64"));

  std::string v8f64_sin_ir = GetFunctionIr(*module, "xla.sin.v8f64");
  std::string v8f64_cos_ir = GetFunctionIr(*module, "xla.cos.v8f64");

  EXPECT_THAT(v8f64_sin_ir, ContainsRegex("fmul <8 x double>"));
  EXPECT_THAT(v8f64_cos_ir, ContainsRegex("fmul <8 x double>"));
  EXPECT_THAT(v8f64_cos_ir, ContainsRegex("<8 x double>.*f0x41933FD174000000"));
}

}  // namespace

}  // namespace xla::codegen

#endif  // defined(__has_attribute) && __has_attribute(ext_vector_type) && ...
