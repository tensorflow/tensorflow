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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log_streamer.h"
#include "absl/random/random.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_ll.h"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"
#include "xla/codegen/intrinsic/test_matchers.h"

namespace xla::codegen {
namespace {
using ::testing::ContainsRegex;
using ::testing::Not;
using ::xla::codegen::intrinsic::NearUlps;

constexpr int kTanhUlps = 5;
constexpr int kLogUlps = 5;

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

TEST(EigenUnaryTest, FastLogdIsCorrect) {
  Vec4d x = {1.0, 2.0, 4.0, 8.0};
  Vec4d y = log_v4f64(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::log(x[i]), kLogUlps));
  }
}

template <typename VecType, typename T>
VecType RandomVecRange(T start, T end) {
  constexpr int num_elements = sizeof(VecType) / sizeof(T);

  VecType vec;
  thread_local absl::BitGen gen(absl::MakeTaggedSeedSeq(
      "EIGEN_UNARY_TEST_RANDOM_SEED", absl::LogInfoStreamer().stream()));
  for (int i = 0; i < num_elements; ++i) {
    vec[i] = absl::Uniform<T>(gen, start, end);
  }
  return vec;
}

TEST(EigenUnaryTest, RandomInputsTanhfCorrectness) {
  for (int k = 0; k < 1000; ++k) {
    Vec16f x = RandomVecRange<Vec16f, float>(-100.0f, 100.0f);
    Vec16f y = tanh_v16f32(x);
    for (int i = 0; i < 16; ++i) {
      EXPECT_THAT(y[i], NearUlps(std::tanh(x[i]), kTanhUlps));
    }
  }
}

TEST(EigenUnaryTest, RandomInputsLogfCorrectness) {
  for (int k = 0; k < 1000; ++k) {
    Vec16f x = RandomVecRange<Vec16f, float>(0.1f, 100.0f);
    Vec16f y = log_v16f32(x);
    for (int i = 0; i < 16; ++i) {
      EXPECT_THAT(y[i], NearUlps(std::log(x[i]), kLogUlps));
    }
  }
}

TEST(EigenUnaryTest, v4f32TanhIsCorrect) {
  Vec4f x = {1.0f, 2.0f, -1.0f, 4.0f};
  Vec4f y = tanh_v4f32(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::tanh(x[i]), kTanhUlps));
  }
}

TEST(EigenUnaryTest, v4f32LogIsCorrect) {
  Vec4f x = {1.0f, 2.0f, 4.0f, 8.0f};
  Vec4f y = log_v4f32(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::log(x[i]), kLogUlps));
  }
}

TEST(EigenUnaryTest, v8f32TanhIsCorrect) {
  Vec8f x = {1.0f, 2.0f, -1.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f};
  Vec8f y = tanh_v8f32(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::tanh(x[i]), kTanhUlps));
  }
}

TEST(EigenUnaryTest, v8f32LogIsCorrect) {
  Vec8f x = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f};
  Vec8f y = log_v8f32(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::log(x[i]), kLogUlps));
  }
}

TEST(EigenUnaryTest, v8f64TanhIsCorrect) {
  Vec8d x = {1.0, 2.0, -1.0, 4.0, 8.0, 16.0, 32.0, 64.0};
  Vec8d y = tanh_v8f64(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::tanh(x[i]), kTanhUlps));
  }
}

TEST(EigenUnaryTest, v8f64LogIsCorrect) {
  Vec8d x = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0};
  Vec8d y = log_v8f64(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::log(x[i]), kLogUlps));
  }
}

TEST(EigenUnaryTest, FastTanhfIsVectorized) {
  const std::string ir = llvm_ir::kEigenUnaryLlIr;
  EXPECT_THAT(ir, ContainsRegex("fmul <16 x float>"));
  EXPECT_THAT(ir, ContainsRegex("fmul <8 x double>"));
  EXPECT_THAT(ir, ContainsRegex("<16 x float>.*0x3E4DF2A3C0000000"));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.x86")));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.aarch64")));
  EXPECT_THAT(ir, ContainsRegex("xla.unused.tanh.v16f32"));
  EXPECT_THAT(ir, ContainsRegex("xla.unused.tanh.v8f64"));
  EXPECT_THAT(ir, ContainsRegex("xla.unused.log.v16f32"));
  EXPECT_THAT(ir, ContainsRegex("xla.unused.log.v8f64"));
}

}  // namespace
}  // namespace xla::codegen
