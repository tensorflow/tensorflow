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

#include "xla/codegen/intrinsic/erf.h"

#include <cmath>
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

namespace xla::codegen::intrinsics {
namespace {

using ::xla::codegen::intrinsic::JitRunner;
using ::xla::codegen::intrinsic::NearUlps;

JitRunner CreateJitRunner(Type type) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);
  llvm::Function* erf_func = Erf::CreateDefinition(module.get(), type).value();
  erf_func->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*erf_func));
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
          -9999,
          9999,
          std::numeric_limits<float>::min(),
          std::numeric_limits<float>::max(),
          std::numeric_limits<float>::infinity(),
          -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::quiet_NaN()};
}

TEST(ErfTest, F32) {
  Type type = Type::S(F32);
  JitRunner runner = CreateJitRunner(type);
  auto fn = runner.GetScalarFn<float(float)>(Erf::Name(type));

  for (float x_val : GetTestValues<float>()) {
    float expected = std::erf(x_val);
    float result = fn(x_val);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(result));
    } else {
      EXPECT_THAT(result, NearUlps<float>(expected, 1));
    }
  }
}

}  // namespace
}  // namespace xla::codegen::intrinsics
