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

#include "xla/codegen/math/ldexp.h"

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
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/math/simple_jit_runner.h"
#include "xla/codegen/math/test_matchers.h"

namespace xla::codegen::math {
namespace {

using ::testing::Eq;

JitRunner CreateJitRunnerWithLdexpF64(
    std::function<llvm::Type*(llvm::LLVMContext&)> make_type) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);
  llvm::Function* ldexp_func =
      CreateLdexpF64(module.get(), make_type(*context));
  EXPECT_FALSE(llvm::verifyFunction(*ldexp_func));
  return JitRunner(std::move(module), std::move(context));
}

TEST(LdexpTest, EmitLdexpF64) {
  JitRunner runner = CreateJitRunnerWithLdexpF64(llvm::Type::getDoubleTy);

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
  int64_t exponents[] = {0, 1, -1, 10, -10, 50, -50, -700, 700};

  for (double a_val : test_values) {
    for (int64_t exp_val : exponents) {
      double expected = std::ldexp(a_val, exp_val);
      llvm::Expected<double> result_or_err =
          runner.RunJitTest<double(double, int64_t), double>("xla.ldexp.1xf64",
                                                             a_val, exp_val);
      if (auto e = result_or_err.takeError()) {
        EXPECT_TRUE(false) << "Error: " << toString(std::move(e));
      }
      double result = result_or_err.get();

      if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(result));
      } else {
        EXPECT_THAT(result, NearUlps<double>(expected, 1));
      }
    }
  }
}

TEST(LdexpTest, ClampsExponent) {
  JitRunner runner = CreateJitRunnerWithLdexpF64(llvm::Type::getDoubleTy);

  auto run = [&runner](double a, int64_t exp) {
    return runner
        .RunJitTest<double(double, int64_t), double>("xla.ldexp.1xf64", a, exp)
        .get();
  };
  EXPECT_THAT(run(2.0, 1e9), Eq(std::numeric_limits<double>::infinity()));
  EXPECT_THAT(run(std::numeric_limits<double>::min(), 2100),
              Eq(std::numeric_limits<double>::infinity()));
  EXPECT_THAT(run(std::numeric_limits<double>::max(), -2099), Eq(0.0));
}

TEST(LdexpTest, EmitLdexpF64_Vector4) {
  JitRunner runner =
      CreateJitRunnerWithLdexpF64([](llvm::LLVMContext& context) {
        return llvm::VectorType::get(llvm::Type::getDoubleTy(context),
                                     llvm::ElementCount::getFixed(4));
      });

  using DoubleArray4 = std::array<double, 4>;
  std::vector<DoubleArray4> test_values = {
      {1.0, 2.0, 0.5, -1.0},
      {-2.0, -0.5, 0.0, std::numeric_limits<double>::infinity()},
      {-std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::quiet_NaN(), 0, -23434}};
  int64_t exponents[] = {0, 1, -1, 10, -10, 50, -50};

  for (const DoubleArray4 input_values : test_values) {
    for (int64_t exp_val : exponents) {
      std::array<int64_t, 4> exp_val_vec = {exp_val, exp_val, exp_val, exp_val};

      llvm::Expected<DoubleArray4> result_or_err =
          runner.RunJitBinaryVectorized<4>("xla.ldexp.4xf64", input_values,
                                           exp_val_vec);
      if (auto e = result_or_err.takeError()) {
        EXPECT_TRUE(false) << "Error: " << toString(std::move(e));
      }

      DoubleArray4 actual_results = result_or_err.get();
      for (int j = 0; j < actual_results.size(); ++j) {
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
}  // namespace xla::codegen::math
