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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/codegen/intrinsic/cpp/eigen_unary_ll.h"

namespace xla::codegen {
namespace {
using ::testing::ContainsRegex;
using ::testing::Not;

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
