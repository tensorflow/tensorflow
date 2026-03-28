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
#include "xla/codegen/intrinsic/rsqrt.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"
#include "absl/log/log.h"
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
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"
#include "xla/codegen/intrinsic/test_matchers.h"
#include "xla/codegen/intrinsic/type.h"
#include "xla/primitive_util.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
namespace {

using ::xla::codegen::intrinsic::JitRunner;
using ::xla::codegen::intrinsic::NearUlps;

TEST(RsqrtTest, Name) {
  EXPECT_EQ(Rsqrt::Name(Type::S(F32)), "xla.rsqrt.f32");
  EXPECT_EQ(Rsqrt::Name(Type::V(F32, 4)), "xla.rsqrt.v4f32");
  EXPECT_EQ(Rsqrt::Name(Type::V(F64, 8)), "xla.rsqrt.v8f64");
}

constexpr int kF32UlpsPrecision = 1;
constexpr int kF64UlpsPrecision = 1;

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

llvm::StringMap<bool> GetHostCPUFeatures() {
  static const absl::NoDestructor<llvm::StringMap<bool>> features(
      llvm::sys::getHostCPUFeatures());
  return *features;
}
bool isAmd() { return GetHostCPUFeatures().lookup("sse4a"); }
JitRunner CreateJitRunnerWithRsqrt(
    Type type, bool disable_platform_dependent_math = false,
    std::optional<DeviceType> override_device_type = std::nullopt,
    std::optional<std::string> features_override = std::nullopt) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);

  std::unique_ptr<llvm::TargetMachine> target_machine =
      xla::codegen::intrinsic::CreateHostTargetMachine();
  DeviceType device_type = override_device_type.value_or(
      isAmd() ? DeviceType::kAmdCpu : DeviceType::kIntelCpu);
  std::string features = features_override.value_or(
      target_machine->getTargetFeatureString().str());
  llvm::Function* rsqrt_func =
      Rsqrt::CreateDefinition(
          module.get(),
          {features, device_type, disable_platform_dependent_math}, type)
          .value();
  rsqrt_func->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*rsqrt_func));

  AddOneOverSqrt(*context, *module, rsqrt_func->getReturnType());
  return JitRunner(std::move(module), std::move(context));
}

bool hasAvx() { return GetHostCPUFeatures().lookup("avx"); }
bool hasAvx512Support() { return GetHostCPUFeatures().lookup("avx512f"); }

TEST(FeaturesTest, HostFeatures) {
  std::cout << "CPU: " << llvm::sys::getHostCPUName().str() << "\n";
  const llvm::StringMap<bool> features = llvm::sys::getHostCPUFeatures();
  std::cout << "Host features x86:" << hasAvx()
            << ", avx512f:" << hasAvx512Support() << ", IsAmd: " << isAmd()
            << "\n";
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
        EXPECT_THAT(actual, NearUlps<float>(expected, kF32UlpsPrecision))
            << "val = " << val;
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

  size_t prec = prim_type == F32 ? kF32UlpsPrecision : kF64UlpsPrecision;
  for (int i = 0; i < kN; ++i) {
    NativeType expected = 1.0f / std::sqrt(vals[i]);
    EXPECT_THAT(actuals[i], NearUlps<NativeType>(expected, prec))
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
                NearUlps<float>(std::numeric_limits<float>::infinity(),
                                kF32UlpsPrecision));

    float large_val = std::numeric_limits<float>::max();
    float actual_large = rsqrt(large_val);
    float expected_large = 1.0f / std::sqrt(large_val);
    EXPECT_THAT(actual_large,
                NearUlps<float>(expected_large, kF32UlpsPrecision));

    float small_val = std::numeric_limits<float>::min();
    float actual_small = rsqrt(small_val);
    float expected_small = 1.0f / std::sqrt(small_val);
    EXPECT_THAT(actual_small,
                NearUlps<float>(expected_small, kF32UlpsPrecision));

    // Negative subnormals should return NaN, not +-inf.
    float neg_denorm = -std::numeric_limits<float>::denorm_min();
    EXPECT_TRUE(std::isnan(rsqrt(neg_denorm)))
        << "rsqrt(" << neg_denorm << ") = " << rsqrt(neg_denorm);
  }
}

TEST(RsqrtTest, EmitRsqrtF64) {
  if (hasAvx()) {
    Type type = Type::S(F64);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    auto rsqrt = jit.GetScalarFn<double(double)>(Rsqrt::Name(type));
    auto one_over_sqrt = jit.GetScalarFn<double(double)>("one_over_sqrt");

    EXPECT_THAT(rsqrt(1234.0),
                NearUlps<double>(one_over_sqrt(1234.0), kF64UlpsPrecision));
  }
}

TEST(RsqrtTest, EmitRsqrtF64_EdgeCasesAvxFallback) {
  if (hasAvx() && !hasAvx512Support()) {
    Type type = Type::S(F64);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    auto rsqrt = jit.GetScalarFn<double(double)>(Rsqrt::Name(type));
    EXPECT_THAT(rsqrt(std::numeric_limits<double>::infinity()),
                NearUlps<double>(0.0, kF64UlpsPrecision));

    // NB: The fallback 1/ sqrt(x) doesn't return 0 for max double.
    // EXPECT_THAT(rsqrt(std::numeric_limits<double>::max()),
    //             NearUlps<double>(0.0, kF64UlpsPrecision));
  }
}

TEST(RsqrtTest, EmitRsqrtF64_EdgeCasesHasAvx) {
  if (hasAvx512Support()) {
    Type type = Type::S(F64);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    auto rsqrt = jit.GetScalarFn<double(double)>(Rsqrt::Name(type));
    auto one_over_sqrt = jit.GetScalarFn<double(double)>("one_over_sqrt");
    EXPECT_THAT(rsqrt(std::numeric_limits<double>::infinity()),
                NearUlps<double>(0.0, kF64UlpsPrecision));
    double max = std::numeric_limits<double>::max();
    EXPECT_THAT(rsqrt(max),
                NearUlps<double>(one_over_sqrt(max), kF64UlpsPrecision));
    double large = 8.5390423905955551e+307;
    EXPECT_THAT(rsqrt(large),
                NearUlps<double>(one_over_sqrt(large), kF64UlpsPrecision));
    double large2 = 6.112156648698989e+307;
    EXPECT_THAT(rsqrt(large2),
                NearUlps<double>(one_over_sqrt(large2), kF64UlpsPrecision));

    // Negative subnormals should return NaN, not +-inf.
    // -5e-324 is the negative of the smallest F64 subnormal.
    double neg_denorm = -5e-324;
    EXPECT_TRUE(std::isnan(rsqrt(neg_denorm)))
        << "rsqrt(" << neg_denorm << ") = " << rsqrt(neg_denorm);
    double neg_denorm_min = -std::numeric_limits<double>::denorm_min();
    EXPECT_TRUE(std::isnan(rsqrt(neg_denorm_min)))
        << "rsqrt(" << neg_denorm_min << ") = " << rsqrt(neg_denorm_min);
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
      EXPECT_THAT(actuals[i],
                  NearUlps<NativeType>(expected[i], kF64UlpsPrecision))
          << "i = " << i << " val = " << vals[i] << " kN= " << kN;
    }

    std::array<NativeType, kN> map_to_tiny = {
        8.5390423905955551e+307, std::numeric_limits<double>::infinity()};
    std::array<NativeType, kN> map_to_tiny_vals;
    for (size_t i = 0; i < kN; ++i) {
      map_to_tiny_vals[i] = map_to_tiny[i % map_to_tiny.size()];
    }
    std::array<NativeType, kN> actual_tiny = rsqrt(map_to_tiny_vals);
    std::array<NativeType, kN> expected_tiny = one_over_sqrt(map_to_tiny_vals);
    for (size_t i = 0; i < kN; ++i) {
      EXPECT_THAT(actual_tiny[i],
                  NearUlps<NativeType>(expected_tiny[i], kF64UlpsPrecision))
          << "i = " << i << " val = " << map_to_tiny_vals[i] << " kN= " << kN;
    }

    // Test a value that is close to the edge of the range where the refinement
    // is not used.
    std::array<NativeType, kN> edge_vals;
    for (size_t i = 0; i < kN; ++i) {
      edge_vals[i] = 4.5e+307;
    }
    std::array<NativeType, kN> actual_edge = rsqrt(edge_vals);
    std::array<NativeType, kN> expected_edge = one_over_sqrt(edge_vals);
    for (size_t i = 0; i < kN; ++i) {
      EXPECT_THAT(actual_edge[i],
                  NearUlps<NativeType>(expected_edge[i], kF64UlpsPrecision))
          << "i = " << i << " val = " << edge_vals[i] << " kN= " << kN;
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

TEST(RsqrtTest, DisablePlatformDependentMath) {
  // Overriding the device type to AMD should make this test fail on Intel
  // CPUs if disable_platform_dependent_math is not implemented.
  // To check that this test works correctly, run it on an intel CPU and set
  // disable_platform_dependent_math to false to see the test fail.
  Type type = Type::S(F32);
  JitRunner jit =
      CreateJitRunnerWithRsqrt(type, /*disable_platform_dependent_math=*/true,
                               /*override_device_type=*/DeviceType::kAmdCpu,
                               /*features_override=*/"+sse +avx2");
  auto rsqrt = jit.GetScalarFn<float(float)>(Rsqrt::Name(type));
  auto one_over_sqrt = jit.GetScalarFn<float(float)>("one_over_sqrt");
  float inf = std::numeric_limits<float>::infinity();
  EXPECT_EQ(rsqrt(inf), one_over_sqrt(inf));
  EXPECT_EQ(rsqrt(1.0), one_over_sqrt(1.0));
  EXPECT_EQ(rsqrt(0.1), one_over_sqrt(0.1));
  EXPECT_EQ(rsqrt(13.), one_over_sqrt(13.));
}

TEST(RsqrtTest, AmdRsqrtF64) {
  if (isAmd()) {
    Type type = Type::S(F64);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    auto rsqrt = jit.GetScalarFn<double(double)>(Rsqrt::Name(type));
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_THAT(rsqrt(inf), NearUlps<double>(0.0, kF64UlpsPrecision));
    EXPECT_THAT(rsqrt(1.0), NearUlps<double>(1.0, kF64UlpsPrecision));
    EXPECT_THAT(rsqrt(13.0),
                NearUlps<double>(1.0 / std::sqrt(13.0), kF64UlpsPrecision));
  }
}

TEST(RsqrtTest, AmdRsqrtF32) {
  if (isAmd()) {
    Type type = Type::S(F32);
    JitRunner jit = CreateJitRunnerWithRsqrt(type);
    auto rsqrt = jit.GetScalarFn<float(float)>(Rsqrt::Name(type));
    float inf = std::numeric_limits<float>::infinity();
    EXPECT_THAT(rsqrt(inf), NearUlps<float>(0.0, kF32UlpsPrecision));
    EXPECT_THAT(rsqrt(1.0), NearUlps<float>(1.0, kF32UlpsPrecision));
    EXPECT_THAT(rsqrt(13.0),
                NearUlps<float>(1.0 / std::sqrt(13.0), kF32UlpsPrecision));
  }
}

}  // namespace
}  // namespace xla::codegen::intrinsics
