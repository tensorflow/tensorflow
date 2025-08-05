/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/cpu/elemental_ir_emitter.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "xla/codegen/math/exp.h"
#include "xla/codegen/math/intrinsic.h"
#include "xla/codegen/math/rsqrt.h"
#include "xla/codegen/math/tanh.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/cpu/elemental_math_emitter.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::cpu {
using ::xla::codegen::intrinsics::Type;

absl::StatusOr<llvm::Value*> CpuElementalIrEmitter::EmitAtan2(
    PrimitiveType prim_type, llvm::Value* lhs, llvm::Value* rhs,
    absl::string_view) {
  return xla::cpu::EmitAtan2(module(), *b(), prim_type, lhs, rhs);
}

absl::StatusOr<llvm::Value*> CpuElementalIrEmitter::EmitTanh(
    PrimitiveType prim_type, llvm::Value* value) {
  if (prim_type == F32 || prim_type == F16) {
    llvm::Function* tanh =
        xla::codegen::intrinsics::Tanh::GetOrInsertDeclaration(
            module(), Type::TypeFromIrType(value->getType()));
    return b()->CreateCall(tanh, value);
  }
  return xla::cpu::EmitTanh(module(), *b(), prim_type, value);
}

absl::StatusOr<llvm::Value*> CpuElementalIrEmitter::EmitErf(
    PrimitiveType prim_type, llvm::Value* value) {
  return xla::cpu::EmitErf(module(), *b(), prim_type, value);
}

absl::StatusOr<llvm::Value*> CpuElementalIrEmitter::EmitExp(
    PrimitiveType prim_type, llvm::Value* value, absl::string_view name) {
  if (prim_type == F64) {
    llvm::Type* f64 = b()->getDoubleTy();
    llvm::Function* exp_f64 =
        xla::codegen::intrinsics::Exp::GetOrInsertDeclaration(
            module(), Type::TypeFromIrType(f64));
    return b()->CreateCall(exp_f64, value);
  }
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::exp, {value},
                                      {value->getType()}, b(), name);
}

absl::StatusOr<llvm::Value*> CpuElementalIrEmitter::EmitRsqrt(
    PrimitiveType prim_type, llvm::Value* value) {
  if (prim_type == F32 || prim_type == F64) {
    llvm::Function* rsqrt_fn =
        xla::codegen::intrinsics::Rsqrt::GetOrInsertDeclaration(
            module(), Type::S(prim_type));
    return b()->CreateCall(rsqrt_fn, value);
  }
  llvm::CallInst* sqrt = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::sqrt, {value}, {value->getType()}, b());
  return FDiv(llvm::ConstantFP::get(sqrt->getType(), 1.0), sqrt);
}

absl::StatusOr<std::vector<llvm::Value*>>
CpuElementalIrEmitter::EmitThreadLocalCall(
    const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
    absl::string_view name, bool is_reducer) {
  if (thread_local_call_fn_ == nullptr) {
    return absl::InternalError("Thread local call function is not set.");
  }

  return thread_local_call_fn_(callee, parameters, name, is_reducer);
}

}  // namespace xla::cpu
