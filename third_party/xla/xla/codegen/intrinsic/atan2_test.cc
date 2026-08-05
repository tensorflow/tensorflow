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

#include "xla/codegen/intrinsic/atan2.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/IR/Function.h"
#include "llvm/IR/Verifier.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"
#include "xla/codegen/intrinsic/test_matchers.h"
#include "xla/xla_data.pb.h"

#if defined(__x86_64__)
// Dummy reference to prevent linker dead-stripping of SLEEF symbols in the
// test. Since symbols are resolved dynamically via OrcJIT from the host
// process, taking the address of one symbol forces the linker to preserve all
// of them.
extern "C" {
void Sleef_atan2f4_u10();
void Sleef_atan2f8_u10();
void Sleef_atan2f16_u10();
void Sleef_atan2d2_u10();
void Sleef_atan2d4_u10();
void Sleef_atan2d8_u10();
}
void prevent_dead_strip() {
  volatile auto fn1 = &Sleef_atan2f4_u10;
  volatile auto fn2 = &Sleef_atan2f8_u10;
  volatile auto fn3 = &Sleef_atan2f16_u10;
  volatile auto fn4 = &Sleef_atan2d2_u10;
  volatile auto fn5 = &Sleef_atan2d4_u10;
  volatile auto fn6 = &Sleef_atan2d8_u10;
  (void)fn1;
  (void)fn2;
  (void)fn3;
  (void)fn4;
  (void)fn5;
  (void)fn6;
}
#endif

namespace xla::codegen::intrinsics {
using ::xla::codegen::intrinsic::JitRunner;
using ::xla::codegen::intrinsic::NearUlps;

namespace {
JitRunner CreateJitRunnerWithAtan2(Type y, Type x) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);
  llvm::Function* atan2_func =
      Atan2::CreateDefinition(module.get(), y, x).value();
  atan2_func->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*atan2_func));
  return JitRunner(std::move(module), std::move(context));
}

TEST(Atan2Test, Name) {
  EXPECT_EQ(Atan2::Name(Type::S(F32), Type::S(F32)), "xla.atan2.f32.f32");
  EXPECT_EQ(Atan2::Name(Type::V(F64, 4), Type::V(F64, 4)),
            "xla.atan2.v4f64.v4f64");
}

TEST(Atan2Test, EmitAtan2F32) {
  Type type = Type::S(F32);
  JitRunner jit = CreateJitRunnerWithAtan2(type, type);
  auto* fn = jit.GetScalarFn<float(float, float)>(Atan2::Name(type, type));

  float ys[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f};
  float xs[] = {1.0f, 0.0f, -1.0f, 0.5f, -0.5f};

  for (float y : ys) {
    for (float x : xs) {
      float actual = fn(y, x);
      float expected = std::atan2(y, x);
      EXPECT_THAT(actual, NearUlps<float>(expected, 10))
          << "Failed for y = " << y << ", x = " << x;
    }
  }
}

#if defined(__x86_64__)
TEST(Atan2Test, EmitAtan2F32_Vector4) {
  Type type = Type::V(F32, 4);
  JitRunner jit = CreateJitRunnerWithAtan2(type, type);
  auto fn =
      jit.GetVectorizedFn<4, float, float, float>(Atan2::Name(type, type));

  std::array<float, 4> ys = {0.0f, 1.0f, -1.0f, 0.5f};
  std::array<float, 4> xs = {1.0f, 0.0f, -1.0f, 0.5f};

  std::array<float, 4> actuals = fn(ys, xs);

  for (int i = 0; i < 4; ++i) {
    float expected = std::atan2(ys[i], xs[i]);
    EXPECT_THAT(actuals[i], NearUlps<float>(expected, 10))
        << "Failed at index " << i;
  }
}
#endif

TEST(Atan2Test, EmitAtan2F64) {
  Type type = Type::S(F64);
  JitRunner jit = CreateJitRunnerWithAtan2(type, type);
  auto* fn = jit.GetScalarFn<double(double, double)>(Atan2::Name(type, type));

  double ys[] = {0.0, 1.0, -1.0, 0.5, -0.5};
  double xs[] = {1.0, 0.0, -1.0, 0.5, -0.5};

  for (double y : ys) {
    for (double x : xs) {
      double actual = fn(y, x);
      double expected = std::atan2(y, x);
      EXPECT_THAT(actual, NearUlps<double>(expected, 10))
          << "Failed for y = " << y << ", x = " << x;
    }
  }
}

#if defined(__x86_64__)
TEST(Atan2Test, EmitAtan2F64_Vector4) {
  Type type = Type::V(F64, 4);
  JitRunner jit = CreateJitRunnerWithAtan2(type, type);
  auto fn =
      jit.GetVectorizedFn<4, double, double, double>(Atan2::Name(type, type));

  std::array<double, 4> ys = {0.0, 1.0, -1.0, 0.5};
  std::array<double, 4> xs = {1.0, 0.0, -1.0, 0.5};

  std::array<double, 4> actuals = fn(ys, xs);

  for (int i = 0; i < 4; ++i) {
    double expected = std::atan2(ys[i], xs[i]);
    EXPECT_THAT(actuals[i], NearUlps<double>(expected, 10))
        << "Failed at index " << i;
  }
}
#endif

}  // namespace
}  // namespace xla::codegen::intrinsics
