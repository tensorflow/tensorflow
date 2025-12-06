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

#include "xla/codegen/intrinsic/tanh.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Verifier.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"
#include "xla/codegen/intrinsic/test_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
using ::xla::codegen::intrinsic::JitRunner;
using ::xla::codegen::intrinsic::NearUlps;

namespace {
JitRunner CreateJitRunnerWithTanh(Type type) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);
  llvm::Function* tanh_func =
      Tanh::CreateDefinition(module.get(), type).value();
  tanh_func->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*tanh_func));
  return JitRunner(std::move(module), std::move(context));
}

TEST(TanhTest, Name) {
  EXPECT_EQ(Tanh::Name(Type::S(F32)), "xla.tanh.f32");
  EXPECT_EQ(Tanh::Name(Type::V(F64, 16)), "xla.tanh.v16f64");
}

// with_fma = true leads to 3 ULPs of error on F32 and 8 on F64.
constexpr int kNumUlpsF32 = 3;
constexpr int kNumUlpsF64 = 3;  // with_fma = false in tanh.cc

TEST(TanhTest, EmitTanhF32) {
  Type type = Type::S(F32);
  JitRunner jit = CreateJitRunnerWithTanh(type);
  float vals[] = {0.0f,
                  -1.0f,
                  -100.0f,
                  100.0f,
                  0.5f,
                  -0.5f,
                  std::numeric_limits<float>::infinity(),
                  std::numeric_limits<float>::quiet_NaN()};
  auto* fn = jit.GetScalarFn<float(float)>(Tanh::Name(type));
  EXPECT_THAT(fn(std::numeric_limits<float>::infinity()),
              NearUlps<float>(1.0, 0));
  for (float val : vals) {
    float actual = fn(val);
    float expected = std::tanh(val);
    EXPECT_THAT(actual, NearUlps<float>(expected, kNumUlpsF32));
  }
}

TEST(TanhTest, EmitTanhF32_Vector4) {
  // The jit runner must outlive the compiled function.
  Type type = Type::V(F32, 4);
  JitRunner jit = CreateJitRunnerWithTanh(type);
  auto fn = jit.GetVectorizedFn<4, float, float>(Tanh::Name(type));
  const size_t kN = 4;
  std::array<float, kN> vals = {-100.0f, 100.0f, 0.5f, -0.5f};
  std::array<float, kN> actuals = fn(vals);

  for (int i = 0; i < kN; ++i) {
    float expected = std::tanh(vals[i]);
    EXPECT_THAT(actuals[i], NearUlps<float>(expected, kNumUlpsF32))
        << "i = " << i;
  }
}

TEST(TanhTest, EmitTanhF64) {
  Type type = Type::S(F64);
  JitRunner jit = CreateJitRunnerWithTanh(type);
  double vals[] = {0.0f,
                   -1.0f,
                   -100.0f,
                   100.0f,
                   0.5f,
                   -0.5f,
                   std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::quiet_NaN()};
  auto* fn = jit.GetScalarFn<double(double)>(Tanh::Name(type));
  auto inf = std::numeric_limits<double>::infinity();
  EXPECT_THAT(fn(inf), NearUlps<double>(1.0, 0));
  EXPECT_THAT(fn(-inf), NearUlps<double>(-1.0, 0));
  EXPECT_THAT(fn(20), NearUlps<double>(1.0, 0));
  EXPECT_THAT(std::tanh(20), NearUlps<double>(1.0, 0));
  for (double val : vals) {
    double actual = fn(val);
    double expected = std::tanh(val);
    EXPECT_THAT(actual, NearUlps<double>(expected, kNumUlpsF64));
  }
}

TEST(TanhTest, EmitTanhF64_Vector4) {
  // The jit runner must outlive the compiled function.
  Type type = Type::V(F64, 4);
  JitRunner jit = CreateJitRunnerWithTanh(type);
  auto fn = jit.GetVectorizedFn<4, double, double>(Tanh::Name(type));
  const size_t kN = 4;
  std::array<double, kN> vals = {-100.0f, 100.0f, 0.5f, -0.5f};
  std::array<double, kN> actuals = fn(vals);

  for (int i = 0; i < kN; ++i) {
    double expected = std::tanh(vals[i]);
    EXPECT_THAT(actuals[i], NearUlps<double>(expected, kNumUlpsF64))
        << "i = " << i;
  }
}

}  // namespace
}  // namespace xla::codegen::intrinsics
