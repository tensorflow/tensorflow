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

#include "xla/codegen/intrinsic/expm1.h"

#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"
#include "xla/codegen/intrinsic/test_matchers.h"
#include "xla/codegen/intrinsic/type.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
namespace {

using ::xla::codegen::intrinsic::JitRunner;
using ::xla::codegen::intrinsic::NearUlps;

TEST(Expm1Test, Expm1FunctionName) {
  EXPECT_EQ(Expm1::Name(Type::S(F32)), "xla.expm1.f32");
  EXPECT_EQ(Expm1::Name(Type::V(F32, 4)), "xla.expm1.v4f32");
  EXPECT_EQ(Expm1::Name(Type::S(F64)), "xla.expm1.f64");
  EXPECT_EQ(Expm1::Name(Type::V(F64, 4)), "xla.expm1.v4f64");
}

JitRunner CreateJitRunnerWithExpm1(Type type) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);
  absl::StatusOr<llvm::Function*> expm1_func_or =
      Expm1::CreateDefinition(module.get(), type);
  TF_CHECK_OK(expm1_func_or.status());
  llvm::Function* expm1_func = *expm1_func_or;
  expm1_func->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*expm1_func));
  return JitRunner(std::move(module), std::move(context));
}

TEST(Expm1Test, F32SpecialAndWorstCaseValues) {
  Type type = Type::S(F32);
  JitRunner runner = CreateJitRunnerWithExpm1(type);
  auto fn = runner.GetScalarFn<float(float)>(Expm1::Name(type));

  std::vector<float> test_values = {
      0.0f,
      -0.0f,
      0.0019223245f,  // Former 6-ULP worst case for tanh decomposition
      0.0063515f,     // Former 2-ULP float16 worst case
      -0.0019223245f,
      0.5f,
      -0.5f,
      1.0f,
      -1.0f,
      -18.0f,
      -20.0f,
      88.0f,
      88.5f,
      88.72283935546875f,
      89.0f,
      std::numeric_limits<float>::min(),
      -std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
  };

  for (float x_val : test_values) {
    float expected = static_cast<float>(std::expm1(static_cast<double>(x_val)));
    float result = fn(x_val);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(result));
    } else if (x_val == 0.0f) {
      EXPECT_EQ(std::signbit(result), std::signbit(x_val));
      EXPECT_EQ(result, expected);
    } else {
      EXPECT_THAT(result, NearUlps<float>(expected, 1)) << "x = " << x_val;
    }
  }
}

TEST(Expm1Test, F64SpecialAndWorstCaseValues) {
  Type type = Type::S(F64);
  JitRunner runner = CreateJitRunnerWithExpm1(type);
  auto fn = runner.GetScalarFn<double(double)>(Expm1::Name(type));

  std::vector<double> test_values = {
      0.0,
      -0.0,
      0.0019223245,
      -0.0019223245,
      0.3465815484523773,
      -0.3465815484523773,
      0.5,
      -0.5,
      1.0,
      -1.0,
      -37.0,
      -40.0,
      709.0,
      709.5,
      709.782712893384,
      710.0,
      std::numeric_limits<double>::min(),
      -std::numeric_limits<double>::min(),
      std::numeric_limits<double>::max(),
      -std::numeric_limits<double>::max(),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),
  };

  for (double x_val : test_values) {
    double expected = std::expm1(x_val);
    double result = fn(x_val);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(result));
    } else if (x_val == 0.0) {
      EXPECT_EQ(std::signbit(result), std::signbit(x_val));
      EXPECT_EQ(result, expected);
    } else {
      EXPECT_THAT(result, NearUlps<double>(expected, 1)) << "x = " << x_val;
    }
  }
}

}  // namespace
}  // namespace xla::codegen::intrinsics
