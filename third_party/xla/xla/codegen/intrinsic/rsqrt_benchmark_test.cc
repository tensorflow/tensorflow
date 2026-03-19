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

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/match.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Target/TargetMachine.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/rsqrt.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"
#include "xla/primitive_util.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsic {

using ::xla::codegen::intrinsics::DeviceType;
using ::xla::codegen::intrinsics::Rsqrt;
using ::xla::codegen::intrinsics::Type;

void CreateOneOverSqrt(llvm::LLVMContext& context, llvm::Module& module,
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
      xla::codegen::intrinsic::CreateHostTargetMachine();
  const auto features = target_machine->getTargetFeatureString().str();
  DeviceType device_type = absl::StrContains(features, "+sse4a")
                               ? DeviceType::kAmdCpu
                               : DeviceType::kIntelCpu;
  llvm::Function* rsqrt_func =
      Rsqrt::CreateDefinition(module.get(), {features, device_type, false},
                              type)
          .value();
  rsqrt_func->setLinkage(llvm::Function::ExternalLinkage);
  CreateOneOverSqrt(*context, *module, Type::TypeToIrType(type, *context));
  return JitRunner(std::move(module), std::move(context));
}

enum RsqrtFunction {
  kRsqrt,
  kOneOverSqrt,
};

template <size_t num_elements, PrimitiveType type, RsqrtFunction function>
static void BM_RsqrtVectorized(benchmark::State& state) {
  using NativeType = typename primitive_util::PrimitiveTypeToNative<type>::type;
  Type intrinsic_type = Type::V(type, num_elements);
  JitRunner jit = CreateJitRunnerWithRsqrt(intrinsic_type);
  std::string function_name =
      (function == kRsqrt) ? Rsqrt::Name(intrinsic_type) : "one_over_sqrt";
  auto rsqrt = jit.GetVectorizedFn<num_elements, NativeType, NativeType>(
      function_name, 100'000);
  std::array<NativeType, num_elements> vec;
  for (size_t i = 0; i < num_elements; ++i) {
    vec[i] = static_cast<NativeType>(100.0 + i * 10.0);
  }
  for (auto s : state) {
    rsqrt(vec);
  }
}

BENCHMARK(BM_RsqrtVectorized<4, F32, kRsqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<4, F32, kOneOverSqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<8, F32, kRsqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<8, F32, kOneOverSqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<2, F64, kRsqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<2, F64, kOneOverSqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<4, F64, kRsqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<4, F64, kOneOverSqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<8, F64, kRsqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<8, F64, kOneOverSqrt>)->MeasureProcessCPUTime();
}  // namespace xla::codegen::intrinsic
