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
#include <complex>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "Eigen/Core"
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
#include "llvm/TargetParser/Host.h"
#include "xla/codegen/math/intrinsic.h"
#include "xla/codegen/math/simple_jit_runner.h"
#include "xla/codegen/math/test_matchers.h"
#include "xla/primitive_util.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
namespace {

using ::xla::codegen::math::JitRunner;
using ::xla::codegen::math::NearUlps;

TEST(RsqrtTest, Name) {
  EXPECT_EQ(Rsqrt::Name(Type::S(F32)), "xla.rsqrt.f32");
  EXPECT_EQ(Rsqrt::Name(Type::V(F32, 4)), "xla.rsqrt.v4f32");
  EXPECT_EQ(Rsqrt::Name(Type::V(F64, 8)), "xla.rsqrt.v8f64");
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

JitRunner CreateJitRunnerWithRsqrt(Type type) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);

  std::unique_ptr<llvm::TargetMachine> target_machine =
      xla::codegen::math::CreateHostTargetMachine();
  llvm::Function* rsqrt_func =
      Rsqrt::CreateDefinition(module.get(), target_machine.get(), type).value();
  rsqrt_func->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*rsqrt_func));

  AddOneOverSqrt(*context, *module, rsqrt_func->getReturnType());
  return JitRunner(std::move(module), std::move(context));
}

bool hasAvx() {
  llvm::StringMap<bool> HostFeatures = llvm::sys::getHostCPUFeatures();
  return HostFeatures.lookup("avx");
}

bool hasAvx512Support() {
  llvm::StringMap<bool> HostFeatures = llvm::sys::getHostCPUFeatures();
  return HostFeatures.lookup("avx512f");
}

TEST(FeaturesTest, HostFeatures) {
  std::cout << "Host features x86:" << hasAvx()
            << ", avx512f:" << hasAvx512Support() << "\n";
}

TEST(RsqrtTest, EmitRsqrtF32) {
  if (hasAvx()) {
    Type type = Type::S(F32);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    auto rsqrt = jit.GetScalarFn<float(float)>(Rsqrt::Name(type));
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

template <size_t kN, PrimitiveType prim_type>
void TestRsqrt_Vectors() {
  Type type = Type::V(prim_type, kN);
  JitRunner jit = CreateJitRunnerWithRsqrt(type);
  using NativeType = primitive_util::NativeTypeOf<prim_type>;
  auto rsqrt =
      jit.GetVectorizedFn<kN, NativeType, NativeType>(Rsqrt::Name(type));
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
  if (hasAvx()) {
    TestRsqrt_Vectors<4, F32>();
    TestRsqrt_Vectors<8, F32>();
    if (hasAvx512Support()) {
      TestRsqrt_Vectors<16, F32>();
    }
  }
}

TEST(RsqrtTest, EmitRsqrtF64_Vectors) {
  if (hasAvx512Support()) {
    TestRsqrt_Vectors<2, F64>();
    TestRsqrt_Vectors<4, F64>();
    TestRsqrt_Vectors<8, F64>();
  }
}

TEST(RsqrtTest, EmitRsqrtF32_EdgeCases) {
  if (hasAvx()) {
    Type type = Type::S(F32);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    auto rsqrt = jit.GetScalarFn<float(float)>(Rsqrt::Name(type));

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

TEST(RsqrtTest, EmitRsqrtF64) {
  if (hasAvx()) {
    Type type = Type::S(F64);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    auto rsqrt = jit.GetScalarFn<double(double)>(Rsqrt::Name(type));
    auto one_over_sqrt = jit.GetScalarFn<double(double)>("one_over_sqrt");

    EXPECT_THAT(rsqrt(1234.0), NearUlps<double>(one_over_sqrt(1234.0), 1));
  }
}

TEST(RsqrtTest, EmitRsqrtF64_EdgeCasesAvxFallback) {
  if (hasAvx() && !hasAvx512Support()) {
    Type type = Type::S(F64);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    auto rsqrt = jit.GetScalarFn<double(double)>(Rsqrt::Name(type));
    EXPECT_THAT(rsqrt(std::numeric_limits<double>::infinity()),
                NearUlps<double>(0.0, 1));
    // NB: The fallback 1/ sqrt(x) doesn't return 0 for max double.
    // EXPECT_THAT(rsqrt(std::numeric_limits<double>::max()),
    //             NearUlps<double>(0.0, 1));
  }
}

TEST(RsqrtTest, EmitRsqrtF64_EdgeCasesHasAvx) {
  if (hasAvx512Support()) {
    Type type = Type::S(F64);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    auto rsqrt = jit.GetScalarFn<double(double)>(Rsqrt::Name(type));
    auto one_over_sqrt = jit.GetScalarFn<double(double)>("one_over_sqrt");
    EXPECT_THAT(rsqrt(std::numeric_limits<double>::infinity()),
                NearUlps<double>(0.0, 1));
    double max = std::numeric_limits<double>::max();
    EXPECT_THAT(rsqrt(max), NearUlps<double>(one_over_sqrt(max), 1));
  }
}

template <size_t kN>
void TestRsqrtF64EdgeCases() {
  if (hasAvx512Support()) {
    Type type = Type::V(F64, kN);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    using NativeType = double;
    auto rsqrt =
        jit.GetVectorizedFn<kN, NativeType, NativeType>(Rsqrt::Name(type));
    auto one_over_sqrt =
        jit.GetVectorizedFn<kN, NativeType, NativeType>("one_over_sqrt");
    std::vector<NativeType> val_vec = {
        std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::max()};
    std::array<NativeType, kN> vals;
    for (size_t i = 0; i < kN; ++i) {
      vals[i] = val_vec[i % val_vec.size()];
    }
    std::array<NativeType, kN> actuals = rsqrt(vals);
    std::array<NativeType, kN> expected = one_over_sqrt(vals);
    for (int i = 0; i < kN; ++i) {
      EXPECT_THAT(actuals[i], NearUlps<NativeType>(expected[i], 1))
          << "i = " << i << " val = " << vals[i] << " kN= " << kN;
    }

    std::array<NativeType, kN> map_to_zero = {
        8.5390423905955551e+307, std::numeric_limits<double>::infinity()};
    std::array<NativeType, kN> map_to_zero_vals;
    for (size_t i = 0; i < kN; ++i) {
      map_to_zero_vals[i] = map_to_zero[i % map_to_zero.size()];
    }
    std::array<NativeType, kN> actual_zero = rsqrt(map_to_zero_vals);
    std::array<NativeType, kN> expected_zero = one_over_sqrt(map_to_zero_vals);
    for (size_t i = 0; i < kN; ++i) {
      EXPECT_THAT(actual_zero[i], NearUlps<NativeType>(expected_zero[i], 1))
          << "i = " << i << " val = " << map_to_zero_vals[i] << " kN= " << kN;
    }
  }
}

TEST(RsqrtTest, EmitRsqrtF64_EdgeCases_Vectors) {
  if (hasAvx512Support()) {
    TestRsqrtF64EdgeCases<2>();
    TestRsqrtF64EdgeCases<4>();
    TestRsqrtF64EdgeCases<8>();
  }
}

TEST(RsqrtComplex, StdSqrt) {
  std::complex<double> x =
      std::complex<double>(8.5390423905955551e+307, 1.6179238213760051e+308);
  std::complex<double> y = std::complex<double>(1, 0) / std::sqrt(x);
  EXPECT_EQ(y, std::complex<double>(0, 0));
}

TEST(RsqrtComplex, EigenSqrt) {
  std::complex<double> x(8.5390423905955551e+307, 1.6179238213760051e+308);
  Eigen::Array<std::complex<double>, 1, 1> x_eigen;
  x_eigen << x;
  Eigen::Array<std::complex<double>, 1, 1> y_eigen = 1.0 / x_eigen.sqrt();
  EXPECT_EQ(y_eigen(0), std::complex<double>(0, 0));
}

}  // namespace
}  // namespace xla::codegen::intrinsics
