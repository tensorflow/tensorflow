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
#include <memory>
#include <string>
#include <utility>

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
#include "llvm/Support/TypeSize.h"
#include "xla/codegen/math/rsqrt.h"
#include "xla/codegen/math/simple_jit_runner.h"
#include "xla/primitive_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::math {

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

JitRunner CreateJitRunnerWithRsqrt(int num_elements, PrimitiveType type) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);
  llvm::Type* llvm_type = llvm_ir::PrimitiveTypeToIrType(type, *context);
  if (num_elements > 1) {
    llvm_type = llvm::VectorType::get(
        llvm_type, llvm::ElementCount::getFixed(num_elements));
  }
  llvm::Function* rsqrt_func = CreateRsqrtX86(module.get(), llvm_type);
  rsqrt_func->setLinkage(llvm::Function::ExternalLinkage);
  CreateOneOverSqrt(*context, *module, llvm_type);
  return JitRunner(std::move(module), std::move(context));
}

enum RsqrtFunction {
  kRsqrt,
  kOneOverSqrt,
};

template <int num_elements, PrimitiveType type, RsqrtFunction function>
static void BM_RsqrtVectorized(benchmark::State& state) {
  using NativeType = typename primitive_util::PrimitiveTypeToNative<type>::type;
  JitRunner jit = CreateJitRunnerWithRsqrt(num_elements, type);
  std::string function_name = (function == kRsqrt)
                                  ? RsqrtFunctionName(num_elements, type)
                                  : "one_over_sqrt";
  auto rsqrt = jit.GetVectorizedFn<num_elements, NativeType, NativeType>(
      function_name, 100'000);
  std::array<NativeType, num_elements> vec = {1.0, -1.0, 100.0, 1e14};
  for (auto s : state) {
    rsqrt(vec);
  }
}

BENCHMARK(BM_RsqrtVectorized<4, F32, kRsqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<4, F32, kOneOverSqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<8, F32, kRsqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<8, F32, kOneOverSqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<8, F64, kRsqrt>)->MeasureProcessCPUTime();
BENCHMARK(BM_RsqrtVectorized<8, F64, kOneOverSqrt>)->MeasureProcessCPUTime();
}  // namespace xla::codegen::math
