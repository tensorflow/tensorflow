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

#include "xla/codegen/math/tanh.h"

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
#include "xla/codegen/math/intrinsic.h"
#include "xla/codegen/math/simple_jit_runner.h"
#include "xla/codegen/math/test_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
using ::xla::codegen::math::JitRunner;
using ::xla::codegen::math::NearUlps;

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

// with_fma = true leads to 3 ULPs of error.
constexpr int kNumUlps = 3;

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
  for (float val : vals) {
    float actual = fn(val);
    float expected = std::tanh(val);
    EXPECT_THAT(actual, NearUlps<float>(expected, kNumUlps));
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
    EXPECT_THAT(actuals[i], NearUlps<float>(expected, kNumUlps)) << "i = " << i;
  }
}

}  // namespace
}  // namespace xla::codegen::intrinsics
