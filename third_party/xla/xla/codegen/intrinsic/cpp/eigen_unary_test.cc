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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/codegen/intrinsic/cpp/eigen_unary_ll.h"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"
#include "xla/codegen/intrinsic/test_matchers.h"

namespace xla::codegen {
namespace {
using ::testing::ContainsRegex;
using ::testing::Not;
using ::xla::codegen::intrinsic::NearUlps;

constexpr int kTanhUlps = 4;

TEST(EigenUnaryTest, FastTanhfIsCorrect) {
  Vec16f x = {1.0f,  2.0f,  -1.0f, 4.0f,   8.0f,   16.0f,  32.0f, 64.0f,
              -2.0f, -4.0f, -8.0f, -16.0f, -32.0f, -64.0f, 0.0f,  0.5f};
  Vec16f y = tanh_v16f32(x);
  for (int i = 0; i < 16; ++i) {
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

TEST(EigenUnaryTest, FastTanhfIsVectorized) {
  const std::string ir = llvm_ir::kEigenUnaryLlIr;
  EXPECT_THAT(ir, ContainsRegex("fmul <16 x float>"));
  EXPECT_THAT(ir, ContainsRegex("fmul <8 x double>"));
  EXPECT_THAT(ir, ContainsRegex("<16 x float>.*0x3E4DF2A3C0000000"));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.x86")));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.aarch64")));
  EXPECT_THAT(ir, ContainsRegex("xla.tanh.v16f32"));
  EXPECT_THAT(ir, ContainsRegex("xla.tanh.v8f64"));
}

}  // namespace
}  // namespace xla::codegen
