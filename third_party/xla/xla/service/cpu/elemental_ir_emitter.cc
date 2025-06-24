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
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/cpu/elemental_math_emitter.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::cpu {

absl::StatusOr<llvm::Value*> CpuElementalIrEmitter::EmitAtan2(
    PrimitiveType prim_type, llvm::Value* lhs, llvm::Value* rhs,
    absl::string_view) {
  return xla::cpu::EmitAtan2(module(), *b(), prim_type, lhs, rhs);
}

absl::StatusOr<llvm::Value*> CpuElementalIrEmitter::EmitTanh(
    PrimitiveType prim_type, llvm::Value* value) {
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
    llvm::FunctionType* f64_type = llvm::FunctionType::get(f64, {f64}, false);
    llvm::Function* exp_f64 = llvm::cast<llvm::Function>(
        module()->getOrInsertFunction("xla.exp.f64", f64_type).getCallee());
    return b()->CreateCall(exp_f64, value);
  }
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::exp, {value},
                                      {value->getType()}, b(), name);
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
