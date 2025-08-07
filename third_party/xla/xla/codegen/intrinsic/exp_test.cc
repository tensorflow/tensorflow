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

#include "xla/codegen/intrinsic/exp.h"

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
#include "xla/codegen/intrinsic/ldexp.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"
#include "xla/codegen/intrinsic/test_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
using ::xla::codegen::intrinsic::JitRunner;
using ::xla::codegen::intrinsic::NearUlps;

namespace {
JitRunner CreateJitRunnerWithExpF64(Type type) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);
  llvm::Function* ldexp_func =
      Ldexp::CreateDefinition(module.get(), type,
                              Type(S32, type.vector_width()))
          .value();
  ldexp_func->setLinkage(llvm::Function::ExternalLinkage);
  llvm::Function* exp_func = Exp::CreateDefinition(module.get(), type).value();
  exp_func->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*exp_func));
  return JitRunner(std::move(module), std::move(context));
}

TEST(ExpTest, SclarIninsic) {
  EXPECT_EQ(Exp::Name(Type::S(F32)), "xla.exp.f32");
  EXPECT_EQ(Exp::Name(Type::S(F64)), "xla.exp.f64");
}

TEST(ExpTest, VectorIninsic) {
  EXPECT_EQ(Exp::Name(Type::V(F32, 4)), "xla.exp.v4f32");
  EXPECT_EQ(Exp::Name(Type::V(F64, 8)), "xla.exp.v8f64");
}

TEST(ExpTest, EmitExpF64) {
  Type type = Type::S(F64);
  JitRunner jit = CreateJitRunnerWithExpF64(type);
  double vals[] = {0,
                   -1,
                   -100,
                   100,
                   708,
                   -706,
                   std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::quiet_NaN()};
  auto* fn = jit.GetScalarFn<double(double)>(Exp::Name(type));
  for (double val : vals) {
    double actual = fn(val);
    double expected = std::exp(val);
    EXPECT_THAT(actual, NearUlps<double>(expected, 1));
  }
}

TEST(ExpTest, EmitExpF64_Vector4) {
  // The jit runner must outlive the compiled function.
  Type type = Type::V(F64, 4);
  JitRunner jit = CreateJitRunnerWithExpF64(type);
  auto fn = jit.GetVectorizedFn<4, double, double>(Exp::Name(type));
  const size_t kN = 4;
  std::array<double, kN> vals = {-100, 100, 708, -706.1};
  std::array<double, kN> actuals = fn(vals);

  for (int i = 0; i < kN; ++i) {
    double expected = std::exp(vals[i]);
    EXPECT_THAT(actuals[i], NearUlps<double>(expected, 1)) << "i = " << i;
  }
}

}  // namespace
}  // namespace xla::codegen::intrinsics
