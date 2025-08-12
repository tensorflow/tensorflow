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
#include "xla/codegen/math/rsqrt.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/TargetParser/Host.h"
#include "xla/codegen/math/intrinsic.h"
#include "xla/codegen/math/simple_jit_runner.h"
#include "xla/codegen/math/test_matchers.h"
#include "xla/primitive_util.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::codegen::math {
namespace {

TEST(RsqrtTest, Name) {
  EXPECT_EQ(Intrinsic::Rsqrt::Name(F32), "xla.rsqrt.f32");
  EXPECT_EQ(Intrinsic::Rsqrt::Name(F32, 4), "xla.rsqrt.v4f32");
  EXPECT_EQ(Intrinsic::Rsqrt::Name(F64, 8), "xla.rsqrt.v8f64");
}

void AddOneOverSqrt(llvm::LLVMContext& context, llvm::Module& module,
                    llvm::Type* type) {
  // insert 1 / sqrt(x) function for comparison.
  llvm::Function* one_over_sqrt_func = llvm::Function::Create(
      llvm::FunctionType::get(type, {type}, /*isVarArg=*/false),
      llvm::GlobalValue::ExternalLinkage, "one_over_sqrt", module);
  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(context, "entry", one_over_sqrt_func);
  llvm::Value* x = one_over_sqrt_func->getArg(0);
  llvm::IRBuilder<> builder(entry_bb);
  llvm::Value* one_over_sqrt = builder.CreateFDiv(
      llvm::ConstantFP::get(type, 1.0),
      builder.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, x));
  builder.CreateRet(one_over_sqrt);
}

JitRunner CreateJitRunnerWithRsqrt(PrimitiveType type, size_t vector_width) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);
  llvm::Function* rsqrt_func =
      Intrinsic::Rsqrt::CreateDefinition(module.get(), type, vector_width)
          .value();
  rsqrt_func->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*rsqrt_func));

  AddOneOverSqrt(*context, *module, rsqrt_func->getReturnType());
  return JitRunner(std::move(module), std::move(context));
}

bool isX86() {
  llvm::StringMap<bool> HostFeatures = llvm::sys::getHostCPUFeatures();
  return HostFeatures.lookup("x86");
}

bool hasAVX512Support() {
  llvm::StringMap<bool> HostFeatures = llvm::sys::getHostCPUFeatures();
  return HostFeatures.lookup("avx512f");
}

TEST(RsqrtTest, EmitRsqrtF32) {
  if (isX86()) {
    JitRunner jit = CreateJitRunnerWithRsqrt(F32, 1);
    auto rsqrt = jit.GetScalarFn<float(float)>(Intrinsic::Rsqrt::Name(F32, 1));
    auto one_over_sqrt = jit.GetScalarFn<float(float)>("one_over_sqrt");
    float vals[] = {
        1.0f,
        4.0f,
        0.25f,
        100.0f,
        1e-10f,
        1e10f,
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::infinity(),
        -1.0f,  // Should produce NaN
        0.0f,   // Should produce infinity
        std::numeric_limits<float>::quiet_NaN(),
    };

    for (float val : vals) {
      float actual = rsqrt(val);
      float expected = one_over_sqrt(val);

      if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(actual)) << "val = " << val;
      } else if (std::isinf(expected)) {
        EXPECT_TRUE(std::isinf(actual)) << "val = " << val;
        EXPECT_EQ(expected > 0, actual > 0) << "val = " << val;
      } else {
        EXPECT_THAT(actual, NearUlps<float>(expected, 1)) << "val = " << val;
      }
    }
  }
}

template <size_t kN, PrimitiveType type>
void TestRsqrt_Vectors() {
  JitRunner jit = CreateJitRunnerWithRsqrt(type, kN);
  using NativeType = primitive_util::NativeTypeOf<type>;
  auto rsqrt = jit.GetVectorizedFn<kN, NativeType, NativeType>(
      Intrinsic::Rsqrt::Name(type, kN));
  std::vector<NativeType> val_vec = {1.0f, 0.0f, 0.25f, 100.0f, -1.0f};
  std::array<NativeType, kN> vals;
  for (size_t i = 0; i < kN; ++i) {
    vals[i] = val_vec[i % val_vec.size()];
  }
  std::array<NativeType, kN> actuals = rsqrt(vals);

  for (int i = 0; i < kN; ++i) {
    NativeType expected = 1.0f / std::sqrt(vals[i]);
    EXPECT_THAT(actuals[i], NearUlps<NativeType>(expected, 1))
        << "i = " << i << " val = " << vals[i] << " kN= " << kN;
  }
}

TEST(RsqrtTest, EmitRsqrtF32_Vectors) {
  if (isX86()) {
    TestRsqrt_Vectors<4, F32>();
    TestRsqrt_Vectors<8, F32>();
    if (hasAVX512Support()) {
      TestRsqrt_Vectors<16, F32>();
    }
  }
}

TEST(RsqrtTest, EmitRsqrtF64_Vectors) {
  if (hasAVX512Support()) {
    TestRsqrt_Vectors<2, F64>();
    TestRsqrt_Vectors<4, F64>();
    TestRsqrt_Vectors<8, F64>();
  }
}

TEST(RsqrtTest, EmitRsqrtF32_EdgeCases) {
  if (isX86()) {
    JitRunner jit = CreateJitRunnerWithRsqrt(F32, 1);
    auto rsqrt = jit.GetScalarFn<float(float)>(Intrinsic::Rsqrt::Name(F32, 1));

    float actual_denorm = rsqrt(std::numeric_limits<float>::denorm_min());
    EXPECT_THAT(actual_denorm,
                NearUlps<float>(std::numeric_limits<float>::infinity(), 1));

    float large_val = std::numeric_limits<float>::max();
    float actual_large = rsqrt(large_val);
    float expected_large = 1.0f / std::sqrt(large_val);
    EXPECT_THAT(actual_large, NearUlps<float>(expected_large, 1));

    float small_val = std::numeric_limits<float>::min();
    float actual_small = rsqrt(small_val);
    float expected_small = 1.0f / std::sqrt(small_val);
    EXPECT_THAT(actual_small, NearUlps<float>(expected_small, 1));
  }
}

}  // namespace
}  // namespace xla::codegen::math
