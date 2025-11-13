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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/codegen/intrinsic/cpp/eigen_unary_ll.h"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"
#include "xla/codegen/intrinsic/test_matchers.h"

namespace xla::codegen {
namespace {
using ::testing::ContainsRegex;
using ::testing::ElementsAre;
using ::testing::Not;
using ::xla::codegen::intrinsic::NearUlps;

constexpr int kTanhUlps = 1;

TEST(EigenUnaryTest, FastTanhfIsCorrect) {
  Vec4f x = {1.0f, 2.0f, -1.0f, 4.0f};
  Vec4f y = tanh_v4f32(x);
  std::vector<float> y_vec({y[0], y[1], y[2], y[3]});
  EXPECT_THAT(y_vec, ElementsAre(NearUlps(std::tanh(x[0]), kTanhUlps),
                                 NearUlps(std::tanh(x[1]), kTanhUlps),
                                 NearUlps(std::tanh(x[2]), kTanhUlps),
                                 NearUlps(std::tanh(x[3]), kTanhUlps)));
}

TEST(EigenUnaryTest, v8f64TanhIsCorrect) {
  Vec8d x = {1.0, 2.0, -1.0, 4.0, 8.0, 16.0, 32.0, 64.0};
  Vec8d y = tanh_v8f64(x);
  std::vector<double> y_vec({y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]});
  EXPECT_THAT(y_vec, ElementsAre(NearUlps(std::tanh(x[0]), kTanhUlps),
                                 NearUlps(std::tanh(x[1]), kTanhUlps),
                                 NearUlps(std::tanh(x[2]), kTanhUlps),
                                 NearUlps(std::tanh(x[3]), kTanhUlps),
                                 NearUlps(std::tanh(x[4]), kTanhUlps),
                                 NearUlps(std::tanh(x[5]), kTanhUlps),
                                 NearUlps(std::tanh(x[6]), kTanhUlps),
                                 NearUlps(std::tanh(x[7]), kTanhUlps)));
}

TEST(EigenUnaryTest, FastTanhfIsVectorized) {
#ifdef __x86_64__
  const std::string avx2 = llvm_ir::kEigenUnaryLlAvx2Ir;
  EXPECT_THAT(avx2, ContainsRegex("fmul <4 x float>"));
  EXPECT_THAT(avx2, ContainsRegex("<4 x float>.*0x3E4DF2A3C0000000"));
  EXPECT_THAT(avx2, ContainsRegex("llvm.x86"));
  EXPECT_THAT(avx2, Not(ContainsRegex("llvm.aarch64")));
  EXPECT_THAT(avx2, Not(ContainsRegex("llvm.fma.v4f32")));
  EXPECT_THAT(avx2, ContainsRegex("xla.tanh.v4f32"));

  const std::string avx512 = llvm_ir::kEigenUnaryLlAvx512Ir;
  EXPECT_THAT(avx512, ContainsRegex("fmul <4 x float>"));
  EXPECT_THAT(avx512, ContainsRegex("<4 x float>.*0x3E4DF2A3C0000000"));
  EXPECT_THAT(avx512, ContainsRegex("llvm.x86"));
  EXPECT_THAT(avx512, ContainsRegex("llvm.fma.v4f32"));
  EXPECT_THAT(avx512, ContainsRegex("<16 x float>"));
  EXPECT_THAT(avx512, ContainsRegex("<8 x double>"));
#endif

#ifdef __aarch64__
  const std::string neon = llvm_ir::kEigenUnaryLlNeonIr;
  EXPECT_THAT(neon, ContainsRegex("fmul <4 x float>"));
  EXPECT_THAT(neon, ContainsRegex("<4 x float>.*0x3E4DF2A3C0000000"));
  EXPECT_THAT(neon, ContainsRegex("llvm.aarch64.neon"));
  EXPECT_THAT(neon, Not(ContainsRegex("llvm.x86")));
#endif
}

}  // namespace
}  // namespace xla::codegen
