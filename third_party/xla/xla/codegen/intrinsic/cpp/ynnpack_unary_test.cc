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

#include "xla/codegen/intrinsic/cpp/ynnpack_unary.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"
#include "xla/codegen/intrinsic/cpp/ynnpack_unary_16_ll.h"
#include "xla/codegen/intrinsic/cpp/ynnpack_unary_32_ll.h"
#include "xla/codegen/intrinsic/cpp/ynnpack_unary_64_ll.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/test_matchers.h"

namespace xla::codegen {
namespace {

using ::testing::ContainsRegex;
using ::testing::IsNan;
using ::testing::Not;
using ::xla::codegen::intrinsic::NearUlps;

constexpr int kLogUlps = 3;
constexpr int kLog1pUlps = 3;

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

constexpr absl::string_view kScalarSymbols[] = {
    "xla.log.f32",
    "xla.log.f64",
    "xla.log1p.f32",
    "xla.log1p.f64",
};

constexpr absl::string_view kVectorSymbols[] = {
    "xla.log.v2f32",   "xla.log.v4f32",   "xla.log.v8f32",    "xla.log.v16f32",
    "xla.log.v2f64",   "xla.log.v4f64",   "xla.log.v8f64",    "xla.log1p.v2f32",
    "xla.log1p.v4f32", "xla.log1p.v8f32", "xla.log1p.v16f32", "xla.log1p.v2f64",
    "xla.log1p.v4f64", "xla.log1p.v8f64",
};

void VerifyBitcode(const std::string& ir_string) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, ir_string);
  ASSERT_NE(module, nullptr);

  std::string full_ir;
  llvm::raw_string_ostream stream(full_ir);
  module->print(stream, nullptr);

  // Does NOT contain target-specific intrinsics llvm.x86 or llvm.aarch64.
  EXPECT_THAT(full_ir, Not(ContainsRegex("llvm\\.x86")));
  EXPECT_THAT(full_ir, Not(ContainsRegex("llvm\\.aarch64")));

  // Contains all 18 symbols.
  for (absl::string_view symbol : kScalarSymbols) {
    EXPECT_NE(module->getFunction(symbol), nullptr)
        << "Missing symbol: " << symbol;
  }
  for (absl::string_view symbol : kVectorSymbols) {
    EXPECT_NE(module->getFunction(symbol), nullptr)
        << "Missing symbol: " << symbol;
  }

  // In vector functions, does not call @fma or scalar
  // llvm.fma.f32/llvm.fma.f64.
  for (absl::string_view symbol : kVectorSymbols) {
    std::string fn_ir = GetFunctionIr(*module, symbol);
    EXPECT_FALSE(fn_ir.empty()) << "Function IR is empty for: " << symbol;
    if (fn_ir.empty()) {
      continue;
    }
    EXPECT_THAT(fn_ir, Not(ContainsRegex("@fma\\b")))
        << "Found @fma in " << symbol;
    EXPECT_THAT(fn_ir, Not(ContainsRegex("llvm\\.fma\\.f(32|64)")))
        << "Found scalar llvm.fma in " << symbol;
  }
}

// === Bitcode Verification ===

TEST(YnnpackUnaryTest, BitcodeVerification16) {
  VerifyBitcode(llvm_ir::kYnnpackUnary16LlIr);
}

TEST(YnnpackUnaryTest, BitcodeVerification32) {
  VerifyBitcode(llvm_ir::kYnnpackUnary32LlIr);
}

TEST(YnnpackUnaryTest, BitcodeVerification64) {
  VerifyBitcode(llvm_ir::kYnnpackUnary64LlIr);
}

TEST(YnnpackUnaryTest, GetYnnpackIrStringSelectsCorrectVectorWidth) {
  intrinsics::IntrinsicOptions options;
  EXPECT_EQ(GetYnnpackIrString(options), llvm_ir::kYnnpackUnary16LlIr);

  options.features = "+neon,+fp-armv8";
  EXPECT_EQ(GetYnnpackIrString(options), llvm_ir::kYnnpackUnary16LlIr);

  options.features = "+sse4.2";
  EXPECT_EQ(GetYnnpackIrString(options), llvm_ir::kYnnpackUnary16LlIr);

  options.features = "+avx,+avx2";
  EXPECT_EQ(GetYnnpackIrString(options), llvm_ir::kYnnpackUnary32LlIr);

  options.features = "+avx2,+avx512f";
  EXPECT_EQ(GetYnnpackIrString(options), llvm_ir::kYnnpackUnary64LlIr);

  options.prefer_vector_width = 256;
  EXPECT_EQ(GetYnnpackIrString(options), llvm_ir::kYnnpackUnary32LlIr);
}

TEST(YnnpackUnaryTest, UseYnnpackIntrinsicsGating) {
  EXPECT_TRUE(UseYnnpackIntrinsics("+fma"));
  EXPECT_TRUE(UseYnnpackIntrinsics("+avx512f"));
  EXPECT_TRUE(UseYnnpackIntrinsics("+neon"));
  EXPECT_TRUE(UseYnnpackIntrinsics("+avx2,+fma"));
  EXPECT_TRUE(UseYnnpackIntrinsics("+avx512f, +neon"));

  EXPECT_FALSE(UseYnnpackIntrinsics(""));
  EXPECT_FALSE(UseYnnpackIntrinsics("+sse4.2"));
  EXPECT_FALSE(UseYnnpackIntrinsics("+avx"));
  EXPECT_FALSE(UseYnnpackIntrinsics("+avx2"));
  EXPECT_FALSE(UseYnnpackIntrinsics("-fma"));
  EXPECT_FALSE(UseYnnpackIntrinsics("+fma_extra"));

  intrinsics::IntrinsicOptions options;
  options.device_type = intrinsics::DeviceType::kIntelCpu;
  options.features = "+fma";
  EXPECT_TRUE(UseYnnpackIntrinsics(options));

  options.device_type = intrinsics::DeviceType::kAmdCpu;
  options.features = "+avx512f";
  EXPECT_TRUE(UseYnnpackIntrinsics(options));

  options.device_type = intrinsics::DeviceType::kArmCpu;
  options.features = "+neon";
  EXPECT_TRUE(UseYnnpackIntrinsics(options));

  options.device_type = intrinsics::DeviceType::kNvidiaGpu;
  options.features = "+fma";
  EXPECT_FALSE(UseYnnpackIntrinsics(options));

  options.device_type = intrinsics::DeviceType::kIntelCpu;
  options.features = "+avx";
  EXPECT_FALSE(UseYnnpackIntrinsics(options));
}

// === Numerics Verification: Log F32 ===

TEST(YnnpackUnaryTest, LogF32Numerics) {
  const std::vector<float> inputs = {
      0.0001f, 0.01f, 0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 10.0f, 100.0f, 1e6f,
  };
  for (float x : inputs) {
    EXPECT_THAT(xla_log_f32(x), NearUlps(std::log(x), kLogUlps))
        << "for x=" << x;
  }

  Vec2f v2 = {0.5f, 2.0f};
  Vec2f out_v2 = xla_log_v2f32(v2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_THAT(out_v2[i], NearUlps(std::log(v2[i]), kLogUlps));
  }

  Vec4f v4 = {0.1f, 0.5f, 1.0f, 2.0f};
  Vec4f out_v4 = xla_log_v4f32(v4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(out_v4[i], NearUlps(std::log(v4[i]), kLogUlps));
  }

  Vec8f v8 = {0.01f, 0.1f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f, 100.0f};
  Vec8f out_v8 = xla_log_v8f32(v8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(out_v8[i], NearUlps(std::log(v8[i]), kLogUlps));
  }

  Vec16f v16 = {0.001f, 0.01f, 0.05f, 0.1f, 0.25f, 0.5f,  0.75f, 1.0f,
                1.5f,   2.0f,  3.0f,  5.0f, 10.0f, 20.0f, 50.0f, 100.0f};
  Vec16f out_v16 = xla_log_v16f32(v16);
  for (int i = 0; i < 16; ++i) {
    EXPECT_THAT(out_v16[i], NearUlps(std::log(v16[i]), kLogUlps));
  }
}

// === Numerics Verification: Log F64 ===

TEST(YnnpackUnaryTest, LogF64Numerics) {
  const std::vector<double> inputs = {
      0.0001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 10.0, 100.0, 1e12,
  };
  for (double x : inputs) {
    EXPECT_THAT(xla_log_f64(x), NearUlps(std::log(x), kLogUlps))
        << "for x=" << x;
  }

  Vec2d v2 = {0.5, 2.0};
  Vec2d out_v2 = xla_log_v2f64(v2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_THAT(out_v2[i], NearUlps(std::log(v2[i]), kLogUlps));
  }

  Vec4d v4 = {0.1, 0.5, 1.0, 2.0};
  Vec4d out_v4 = xla_log_v4f64(v4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(out_v4[i], NearUlps(std::log(v4[i]), kLogUlps));
  }

  Vec8d v8 = {0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0};
  Vec8d out_v8 = xla_log_v8f64(v8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(out_v8[i], NearUlps(std::log(v8[i]), kLogUlps));
  }
}

// === Numerics Verification: Log1p F32 ===

TEST(YnnpackUnaryTest, Log1pF32Numerics) {
  const std::vector<float> inputs = {
      -0.99f, -0.5f, -0.1f, -1e-4f, 1e-4f, 0.1f, 0.5f, 1.0f, 2.0f, 10.0f,
  };
  for (float x : inputs) {
    EXPECT_THAT(xla_log1p_f32(x), NearUlps(std::log1p(x), kLog1pUlps))
        << "for x=" << x;
  }

  Vec2f v2 = {-0.5f, 1.0f};
  Vec2f out_v2 = xla_log1p_v2f32(v2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_THAT(out_v2[i], NearUlps(std::log1p(v2[i]), kLog1pUlps));
  }

  Vec4f v4 = {-0.9f, -0.1f, 0.5f, 2.0f};
  Vec4f out_v4 = xla_log1p_v4f32(v4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(out_v4[i], NearUlps(std::log1p(v4[i]), kLog1pUlps));
  }

  Vec8f v8 = {-0.8f, -0.5f, -0.1f, -1e-3f, 1e-3f, 0.5f, 2.0f, 10.0f};
  Vec8f out_v8 = xla_log1p_v8f32(v8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(out_v8[i], NearUlps(std::log1p(v8[i]), kLog1pUlps));
  }

  Vec16f v16 = {-0.99f, -0.75f, -0.5f, -0.25f, -0.1f, -1e-4f, -1e-6f, 0.0f,
                1e-6f,  1e-4f,  0.1f,  0.25f,  0.5f,  1.0f,   2.0f,   10.0f};
  Vec16f out_v16 = xla_log1p_v16f32(v16);
  for (int i = 0; i < 16; ++i) {
    EXPECT_THAT(out_v16[i], NearUlps(std::log1p(v16[i]), kLog1pUlps));
  }
}

// === Numerics Verification: Log1p F64 ===

TEST(YnnpackUnaryTest, Log1pF64Numerics) {
  const std::vector<double> inputs = {
      -0.99, -0.5, -0.1, -1e-8, 1e-8, 0.1, 0.5, 1.0, 2.0, 10.0,
  };
  for (double x : inputs) {
    EXPECT_THAT(xla_log1p_f64(x), NearUlps(std::log1p(x), kLog1pUlps))
        << "for x=" << x;
  }

  Vec2d v2 = {-0.5, 1.0};
  Vec2d out_v2 = xla_log1p_v2f64(v2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_THAT(out_v2[i], NearUlps(std::log1p(v2[i]), kLog1pUlps));
  }

  Vec4d v4 = {-0.9, -0.1, 0.5, 2.0};
  Vec4d out_v4 = xla_log1p_v4f64(v4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(out_v4[i], NearUlps(std::log1p(v4[i]), kLog1pUlps));
  }

  Vec8d v8 = {-0.8, -0.5, -0.1, -1e-5, 1e-5, 0.5, 2.0, 10.0};
  Vec8d out_v8 = xla_log1p_v8f64(v8);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(out_v8[i], NearUlps(std::log1p(v8[i]), kLog1pUlps));
  }
}

// === Special Values Verification ===

TEST(YnnpackUnaryTest, SpecialValuesF32) {
  // log1p(-0.0f) has negative sign (std::signbit(...) == true) and value -0.0f.
  float neg_zero_f = -0.0f;
  float log1p_neg_zero_f = xla_log1p_f32(neg_zero_f);
  EXPECT_TRUE(std::signbit(log1p_neg_zero_f));
  EXPECT_EQ(log1p_neg_zero_f, -0.0f);

  Vec2f v2_neg_zero = {neg_zero_f, neg_zero_f};
  Vec2f out_v2_neg_zero = xla_log1p_v2f32(v2_neg_zero);
  EXPECT_TRUE(std::signbit(out_v2_neg_zero[0]));
  EXPECT_TRUE(std::signbit(out_v2_neg_zero[1]));
  EXPECT_EQ(out_v2_neg_zero[0], -0.0f);
  EXPECT_EQ(out_v2_neg_zero[1], -0.0f);

  Vec4f v4_neg_zero = {neg_zero_f, neg_zero_f, neg_zero_f, neg_zero_f};
  Vec4f out_v4_neg_zero = xla_log1p_v4f32(v4_neg_zero);
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(std::signbit(out_v4_neg_zero[i]));
    EXPECT_EQ(out_v4_neg_zero[i], -0.0f);
  }

  Vec8f v8_neg_zero = {neg_zero_f, neg_zero_f, neg_zero_f, neg_zero_f,
                       neg_zero_f, neg_zero_f, neg_zero_f, neg_zero_f};
  Vec8f out_v8_neg_zero = xla_log1p_v8f32(v8_neg_zero);
  for (int i = 0; i < 8; ++i) {
    EXPECT_TRUE(std::signbit(out_v8_neg_zero[i]));
    EXPECT_EQ(out_v8_neg_zero[i], -0.0f);
  }

  Vec16f v16_neg_zero = {neg_zero_f, neg_zero_f, neg_zero_f, neg_zero_f,
                         neg_zero_f, neg_zero_f, neg_zero_f, neg_zero_f,
                         neg_zero_f, neg_zero_f, neg_zero_f, neg_zero_f,
                         neg_zero_f, neg_zero_f, neg_zero_f, neg_zero_f};
  Vec16f out_v16_neg_zero = xla_log1p_v16f32(v16_neg_zero);
  for (int i = 0; i < 16; ++i) {
    EXPECT_TRUE(std::signbit(out_v16_neg_zero[i]));
    EXPECT_EQ(out_v16_neg_zero[i], -0.0f);
  }

  // log(0.0f) == -infinity
  float zero_f = 0.0f;
  EXPECT_EQ(xla_log_f32(zero_f), -std::numeric_limits<float>::infinity());
  Vec2f v2_zero = {zero_f, zero_f};
  Vec2f out_v2_zero = xla_log_v2f32(v2_zero);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(out_v2_zero[i], -std::numeric_limits<float>::infinity());
  }
  Vec4f v4_zero = {zero_f, zero_f, zero_f, zero_f};
  Vec4f out_v4_zero = xla_log_v4f32(v4_zero);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(out_v4_zero[i], -std::numeric_limits<float>::infinity());
  }
  Vec8f v8_zero = {zero_f, zero_f, zero_f, zero_f,
                   zero_f, zero_f, zero_f, zero_f};
  Vec8f out_v8_zero = xla_log_v8f32(v8_zero);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(out_v8_zero[i], -std::numeric_limits<float>::infinity());
  }
  Vec16f v16_zero = {zero_f, zero_f, zero_f, zero_f, zero_f, zero_f,
                     zero_f, zero_f, zero_f, zero_f, zero_f, zero_f,
                     zero_f, zero_f, zero_f, zero_f};
  Vec16f out_v16_zero = xla_log_v16f32(v16_zero);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(out_v16_zero[i], -std::numeric_limits<float>::infinity());
  }

  // log1p(-1.0f) == -infinity
  float neg_one_f = -1.0f;
  EXPECT_EQ(xla_log1p_f32(neg_one_f), -std::numeric_limits<float>::infinity());
  Vec2f v2_neg_one = {neg_one_f, neg_one_f};
  Vec2f out_v2_neg_one = xla_log1p_v2f32(v2_neg_one);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(out_v2_neg_one[i], -std::numeric_limits<float>::infinity());
  }
  Vec4f v4_neg_one = {neg_one_f, neg_one_f, neg_one_f, neg_one_f};
  Vec4f out_v4_neg_one = xla_log1p_v4f32(v4_neg_one);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(out_v4_neg_one[i], -std::numeric_limits<float>::infinity());
  }
  Vec8f v8_neg_one = {neg_one_f, neg_one_f, neg_one_f, neg_one_f,
                      neg_one_f, neg_one_f, neg_one_f, neg_one_f};
  Vec8f out_v8_neg_one = xla_log1p_v8f32(v8_neg_one);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(out_v8_neg_one[i], -std::numeric_limits<float>::infinity());
  }
  Vec16f v16_neg_one = {neg_one_f, neg_one_f, neg_one_f, neg_one_f,
                        neg_one_f, neg_one_f, neg_one_f, neg_one_f,
                        neg_one_f, neg_one_f, neg_one_f, neg_one_f,
                        neg_one_f, neg_one_f, neg_one_f, neg_one_f};
  Vec16f out_v16_neg_one = xla_log1p_v16f32(v16_neg_one);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(out_v16_neg_one[i], -std::numeric_limits<float>::infinity());
  }

  // log(-1.0f) is NaN
  EXPECT_THAT(xla_log_f32(neg_one_f), IsNan());
  Vec2f out_v2_log_neg_one = xla_log_v2f32(v2_neg_one);
  for (int i = 0; i < 2; ++i) {
    EXPECT_THAT(out_v2_log_neg_one[i], IsNan());
  }
  Vec4f out_v4_log_neg_one = xla_log_v4f32(v4_neg_one);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(out_v4_log_neg_one[i], IsNan());
  }
  Vec8f out_v8_log_neg_one = xla_log_v8f32(v8_neg_one);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(out_v8_log_neg_one[i], IsNan());
  }
  Vec16f out_v16_log_neg_one = xla_log_v16f32(v16_neg_one);
  for (int i = 0; i < 16; ++i) {
    EXPECT_THAT(out_v16_log_neg_one[i], IsNan());
  }
}

TEST(YnnpackUnaryTest, SpecialValuesF64) {
  // log1p(-0.0) has negative sign and value -0.0.
  double neg_zero_d = -0.0;
  double log1p_neg_zero_d = xla_log1p_f64(neg_zero_d);
  EXPECT_TRUE(std::signbit(log1p_neg_zero_d));
  EXPECT_EQ(log1p_neg_zero_d, -0.0);

  Vec2d v2_neg_zero = {neg_zero_d, neg_zero_d};
  Vec2d out_v2_neg_zero = xla_log1p_v2f64(v2_neg_zero);
  EXPECT_TRUE(std::signbit(out_v2_neg_zero[0]));
  EXPECT_TRUE(std::signbit(out_v2_neg_zero[1]));
  EXPECT_EQ(out_v2_neg_zero[0], -0.0);
  EXPECT_EQ(out_v2_neg_zero[1], -0.0);

  Vec4d v4_neg_zero = {neg_zero_d, neg_zero_d, neg_zero_d, neg_zero_d};
  Vec4d out_v4_neg_zero = xla_log1p_v4f64(v4_neg_zero);
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(std::signbit(out_v4_neg_zero[i]));
    EXPECT_EQ(out_v4_neg_zero[i], -0.0);
  }

  Vec8d v8_neg_zero = {neg_zero_d, neg_zero_d, neg_zero_d, neg_zero_d,
                       neg_zero_d, neg_zero_d, neg_zero_d, neg_zero_d};
  Vec8d out_v8_neg_zero = xla_log1p_v8f64(v8_neg_zero);
  for (int i = 0; i < 8; ++i) {
    EXPECT_TRUE(std::signbit(out_v8_neg_zero[i]));
    EXPECT_EQ(out_v8_neg_zero[i], -0.0);
  }

  // log(0.0) == -infinity
  double zero_d = 0.0;
  EXPECT_EQ(xla_log_f64(zero_d), -std::numeric_limits<double>::infinity());
  Vec2d v2_zero = {zero_d, zero_d};
  Vec2d out_v2_zero = xla_log_v2f64(v2_zero);
  EXPECT_EQ(out_v2_zero[0], -std::numeric_limits<double>::infinity());
  EXPECT_EQ(out_v2_zero[1], -std::numeric_limits<double>::infinity());
  Vec4d v4_zero = {zero_d, zero_d, zero_d, zero_d};
  Vec4d out_v4_zero = xla_log_v4f64(v4_zero);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(out_v4_zero[i], -std::numeric_limits<double>::infinity());
  }
  Vec8d v8_zero = {zero_d, zero_d, zero_d, zero_d,
                   zero_d, zero_d, zero_d, zero_d};
  Vec8d out_v8_zero = xla_log_v8f64(v8_zero);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(out_v8_zero[i], -std::numeric_limits<double>::infinity());
  }

  // log1p(-1.0) == -infinity
  double neg_one_d = -1.0;
  EXPECT_EQ(xla_log1p_f64(neg_one_d), -std::numeric_limits<double>::infinity());
  Vec2d v2_neg_one = {neg_one_d, neg_one_d};
  Vec2d out_v2_neg_one = xla_log1p_v2f64(v2_neg_one);
  EXPECT_EQ(out_v2_neg_one[0], -std::numeric_limits<double>::infinity());
  EXPECT_EQ(out_v2_neg_one[1], -std::numeric_limits<double>::infinity());
  Vec4d v4_neg_one = {neg_one_d, neg_one_d, neg_one_d, neg_one_d};
  Vec4d out_v4_neg_one = xla_log1p_v4f64(v4_neg_one);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(out_v4_neg_one[i], -std::numeric_limits<double>::infinity());
  }
  Vec8d v8_neg_one = {neg_one_d, neg_one_d, neg_one_d, neg_one_d,
                      neg_one_d, neg_one_d, neg_one_d, neg_one_d};
  Vec8d out_v8_neg_one = xla_log1p_v8f64(v8_neg_one);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(out_v8_neg_one[i], -std::numeric_limits<double>::infinity());
  }

  // log(-1.0) is NaN
  EXPECT_THAT(xla_log_f64(neg_one_d), IsNan());
  Vec2d out_v2_log_neg_one = xla_log_v2f64(v2_neg_one);
  EXPECT_THAT(out_v2_log_neg_one[0], IsNan());
  EXPECT_THAT(out_v2_log_neg_one[1], IsNan());
  Vec4d out_v4_log_neg_one = xla_log_v4f64(v4_neg_one);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(out_v4_log_neg_one[i], IsNan());
  }
  Vec8d out_v8_log_neg_one = xla_log_v8f64(v8_neg_one);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(out_v8_log_neg_one[i], IsNan());
  }
}

}  // namespace
}  // namespace xla::codegen
