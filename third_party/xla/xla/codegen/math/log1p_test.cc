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

#include "xla/codegen/math/log1p.h"

#include <cmath>
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
#include "xla/codegen/math/simple_jit_runner.h"
#include "xla/codegen/math/test_matchers.h"

namespace xla::codegen::math {
namespace {

TEST(Log1pTest, Log1pFunctionName) {
  EXPECT_EQ(Log1pFunctionName(1, F32), "xla.log1p.f32");
  EXPECT_EQ(Log1pFunctionName(2, F32), "xla.log1p.v2f32");
  EXPECT_EQ(Log1pFunctionName(1, F64), "xla.log1p.f64");
  EXPECT_EQ(Log1pFunctionName(2, F64), "xla.log1p.v2f64");
}

JitRunner CreateJitRunnerWithLog1p(
    std::function<llvm::Type*(llvm::LLVMContext&)> make_type) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);
  llvm::Function* log1p_func = CreateLog1p(module.get(), make_type(*context));
  log1p_func->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*log1p_func));
  return JitRunner(std::move(module), std::move(context));
}

template <typename T>
std::vector<T> GetTestValues() {
  return {1.0,
          2.0,
          0.5,
          -1.0,
          -0.5,
          0.0,
          2342093482.3,
          std::numeric_limits<float>::min(),
          std::numeric_limits<float>::max(),
          std::numeric_limits<float>::infinity(),
          -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::quiet_NaN()};
}

TEST(Log1pTest, F32) {
  JitRunner runner = CreateJitRunnerWithLog1p(llvm::Type::getFloatTy);
  auto fn = runner.GetScalarFn<float(float)>(Log1pFunctionName(1, F32));

  for (float x_val : GetTestValues<float>()) {
    float expected = std::log1pf(x_val);
    float result = fn(x_val);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(result));
    } else {
      EXPECT_THAT(result, NearUlps<float>(expected, 1));
    }
  }
}

TEST(Log1pTest, F64) {
  JitRunner runner = CreateJitRunnerWithLog1p(llvm::Type::getDoubleTy);
  auto fn = runner.GetScalarFn<double(double)>(Log1pFunctionName(1, F64));

  for (double x_val : GetTestValues<double>()) {
    double expected = std::log1p(x_val);
    double result = fn(x_val);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(result));
    } else {
      EXPECT_THAT(result, NearUlps<double>(expected, 1));
    }
  }
}

}  // namespace
}  // namespace xla::codegen::math
