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

#include "xla/codegen/math/exp.h"

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
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/math/ldexp.h"
#include "xla/codegen/math/simple_jit_runner.h"
#include "xla/codegen/math/test_matchers.h"

namespace xla::codegen::math {
namespace {

JitRunner CreateJitRunnerWithExpF64(
    std::function<llvm::Type*(llvm::LLVMContext&)> make_type) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);
  llvm::Function* ldexp_func =
      CreateLdexpF64(module.get(), make_type(*context));
  ldexp_func->setLinkage(llvm::Function::ExternalLinkage);
  llvm::Function* exp_func = CreateExpF64(module.get(), make_type(*context));
  exp_func->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*exp_func));
  return JitRunner(std::move(module), std::move(context));
}

TEST(ExpTest, EmitExpF64) {
  JitRunner runner = CreateJitRunnerWithExpF64(llvm::Type::getDoubleTy);
  double vals[] = {0,
                   -1,
                   -100,
                   100,
                   708,
                   -706,
                   std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::quiet_NaN()};
  for (double val : vals) {
    double actual =
        runner.RunJitTest<double(double), double>(ExpF64FunctionName(1), val)
            .get();
    double expected = std::exp(val);
    EXPECT_THAT(actual, NearUlps<double>(expected, 1));
  }
}

TEST(ExpTest, EmitExpF64_Vector4) {
  JitRunner runner = CreateJitRunnerWithExpF64([](llvm::LLVMContext& context) {
    return llvm::VectorType::get(llvm::Type::getDoubleTy(context),
                                 llvm::ElementCount::getFixed(4));
  });
  const size_t kN = 4;
  std::array<double, kN> vals = {-100, 100, 708, -706.1};
  std::array<double, kN> actuals =
      runner.RunJitUnaryVectorized(ExpF64FunctionName(kN), vals).get();

  for (int i = 0; i < kN; ++i) {
    double expected = std::exp(vals[i]);
    EXPECT_THAT(actuals[i], NearUlps<double>(expected, 1)) << "i = " << i;
  }
}

}  // namespace
}  // namespace xla::codegen::math
