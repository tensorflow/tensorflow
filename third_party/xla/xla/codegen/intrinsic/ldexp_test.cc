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

#include "xla/codegen/intrinsic/ldexp.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"
#include "xla/codegen/intrinsic/test_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsic {
namespace {

using ::testing::Eq;
using ::xla::codegen::intrinsics::Ldexp;
using ::xla::codegen::intrinsics::Type;

JitRunner CreateJitRunnerWithLdexpF64(Type type) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);
  llvm::Function* ldexp_func =
      Ldexp::CreateDefinition(module.get(), type,
                              Type(S32, type.vector_width()))
          .value();
  ldexp_func->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*ldexp_func));
  return JitRunner(std::move(module), std::move(context));
}

TEST(LdexpTest, SclarIninsic) {
  EXPECT_EQ(Ldexp::Name(Type::S(F64), Type::S(S32)), "xla.ldexp.f64.i32");
}

TEST(LdexpTest, VectorIninsic) {
  EXPECT_EQ(Ldexp::Name(Type::V(F64, 4), Type::V(S32, 4)),
            "xla.ldexp.v4f64.v4i32");
}

TEST(LdexpTest, EmitLdexpF64) {
  Type type = Type::S(F64);
  JitRunner runner = CreateJitRunnerWithLdexpF64(type);
  auto fn =
      runner.GetScalarFn<double(double, int)>(Ldexp::Name(type, Type::S(S32)));

  double test_values[] = {1.0,
                          2.0,
                          0.5,
                          -1.0,
                          -2.0,
                          -0.5,
                          0.0,
                          2342093482.3,
                          std::numeric_limits<double>::min(),
                          std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::infinity(),
                          -std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN()};
  int32_t exponents[] = {0, 1, -1, 10, -10, 50, -50, -700, 700};

  for (double a_val : test_values) {
    for (int32_t exp_val : exponents) {
      double expected = std::ldexp(a_val, exp_val);
      double result = fn(a_val, exp_val);
      if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(result));
      } else {
        EXPECT_THAT(result, NearUlps<double>(expected, 1));
      }
    }
  }
}

TEST(LdexpTest, ClampsExponent) {
  Type type = Type::S(F64);
  JitRunner runner = CreateJitRunnerWithLdexpF64(type);
  auto* run =
      runner.GetScalarFn<double(double, int)>(Ldexp::Name(type, Type::S(S32)));

  EXPECT_THAT(run(2.0, 1e9), Eq(std::numeric_limits<double>::infinity()));
  EXPECT_THAT(run(std::numeric_limits<double>::min(), 2100),
              Eq(std::numeric_limits<double>::infinity()));
  EXPECT_THAT(run(std::numeric_limits<double>::max(), -2099), Eq(0.0));
}

TEST(LdexpTest, EmitLdexpF64_Vector4) {
  Type type = Type::V(F64, 4);
  JitRunner runner = CreateJitRunnerWithLdexpF64(type);
  auto run = runner.GetVectorizedFn<4, double, double, int>(
      Ldexp::Name(type, Type::V(S32, 4)));

  using DoubleArray4 = std::array<double, 4>;
  std::vector<DoubleArray4> test_values = {
      {1.0, 2.0, 0.5, -1.0},
      {-2.0, -0.5, 0.0, std::numeric_limits<double>::infinity()},
      {-std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::quiet_NaN(), 0, -23434}};
  int32_t exponents[] = {0, 1, -1, 10, -10, 50, -50};

  for (const DoubleArray4 input_values : test_values) {
    for (int32_t exp_val : exponents) {
      std::array<int, 4> exp_val_vec = {exp_val, exp_val, exp_val, exp_val};

      DoubleArray4 actual_results = run(input_values, exp_val_vec);
      for (int32_t j = 0; j < actual_results.size(); ++j) {
        double expected = std::ldexp(input_values[j], exp_val_vec[j]);
        if (std::isnan(expected)) {
          EXPECT_TRUE(std::isnan(actual_results[j]));
        } else {
          EXPECT_THAT(actual_results[j], NearUlps<double>(expected, 1));
        }
      }
    }
  }
}
}  // namespace
}  // namespace xla::codegen::intrinsic
