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
#include "absl/strings/str_join.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_ll.h"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"
#include "xla/codegen/intrinsic/test_matchers.h"

namespace xla::codegen {
namespace {
using ::testing::ContainsRegex;
using ::testing::ElementsAre;
using ::testing::Not;
using ::xla::codegen::intrinsic::NearUlps;

constexpr int kTanhUlps = 4;

TEST(EigenUnaryTest, FastTanhfIsCorrect) {
  Vec16f x = {1.0f,  2.0f,  -1.0f, 4.0f,   8.0f,   16.0f,  32.0f, 64.0f,
              -2.0f, -4.0f, -8.0f, -16.0f, -32.0f, -64.0f, 0.0f,  0.5f};
  Vec16f y = tanh_v16f32(x);
  std::vector<float> y_vec({y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                            y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                            y[15]});
  EXPECT_THAT(y_vec, ElementsAre(NearUlps(std::tanh(x[0]), kTanhUlps),
                                 NearUlps(std::tanh(x[1]), kTanhUlps),
                                 NearUlps(std::tanh(x[2]), kTanhUlps),
                                 NearUlps(std::tanh(x[3]), kTanhUlps),
                                 NearUlps(std::tanh(x[4]), kTanhUlps),
                                 NearUlps(std::tanh(x[5]), kTanhUlps),
                                 NearUlps(std::tanh(x[6]), kTanhUlps),
                                 NearUlps(std::tanh(x[7]), kTanhUlps),
                                 NearUlps(std::tanh(x[8]), kTanhUlps),
                                 NearUlps(std::tanh(x[9]), kTanhUlps),
                                 NearUlps(std::tanh(x[10]), kTanhUlps),
                                 NearUlps(std::tanh(x[11]), kTanhUlps),
                                 NearUlps(std::tanh(x[12]), kTanhUlps),
                                 NearUlps(std::tanh(x[13]), kTanhUlps),
                                 NearUlps(std::tanh(x[14]), kTanhUlps),
                                 NearUlps(std::tanh(x[15]), kTanhUlps)));
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
  std::string cpu_features =
      absl::StrJoin(llvm::sys::getHostCPUFeatures().keys(), ",");
  llvm::errs() << "System CPU features: " << cpu_features << "\n";
  llvm::errs() << "System CPU name: " << llvm::sys::getHostCPUName().str()
               << "\n";
  const std::string ir = llvm_ir::kEigenUnaryLlIr;
  EXPECT_THAT(ir, ContainsRegex("fmul <16 x float>"));
  EXPECT_THAT(ir, ContainsRegex("<16 x float>.*0x3E4DF2A3C0000000"));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.x86")));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.aarch64")));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.fma.v16f32")));
  EXPECT_THAT(ir, ContainsRegex("xla.tanh.v16f32"));
}

}  // namespace
}  // namespace xla::codegen
