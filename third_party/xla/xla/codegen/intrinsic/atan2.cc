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

#include "xla/codegen/intrinsic/atan2.h"

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::codegen::intrinsics {

std::vector<std::vector<Type>> Atan2::SupportedVectorTypes() {
#if defined(__x86_64__)
  return {
      // Float (F32) vector widths
      {Type::S(xla::F32), Type::S(xla::F32)},
      {Type::V(xla::F32, 4), Type::V(xla::F32, 4)},
      {Type::V(xla::F32, 8), Type::V(xla::F32, 8)},
      {Type::V(xla::F32, 16), Type::V(xla::F32, 16)},
      // Double (F64) vector widths
      {Type::S(xla::F64), Type::S(xla::F64)},
      {Type::V(xla::F64, 2), Type::V(xla::F64, 2)},
      {Type::V(xla::F64, 4), Type::V(xla::F64, 4)},
      {Type::V(xla::F64, 8), Type::V(xla::F64, 8)},
  };
#else
  return {
      {Type::S(xla::F32), Type::S(xla::F32)},
      {Type::S(xla::F64), Type::S(xla::F64)},
  };
#endif
}

absl::StatusOr<llvm::Function*> Atan2::CreateDefinition(llvm::Module* module,
                                                        Type y, Type x) {
  auto same_type_status = Type::VerifySameWidthAndElementType(y, x);
  if (!same_type_status.ok()) {
    return same_type_status;
  }
  Type type = y;

#if !defined(__x86_64__)
  if (type.vector_width().value_or(1) > 1) {
    return absl::UnimplementedError(
        "Vector types not supported on non-x86 platforms.");
  }
#endif

  std::string sleef_func_name;

  if (type.element_type() == xla::F32) {
    size_t width = type.vector_width().value_or(1);
    if (width == 1) {
      sleef_func_name = "atan2f";
    } else if (width == 4) {
      sleef_func_name = "Sleef_atan2f4_u10";
    } else if (width == 8) {
      sleef_func_name = "Sleef_atan2f8_u10";
    } else if (width == 16) {
      sleef_func_name = "Sleef_atan2f16_u10";
    }
  } else if (type.element_type() == xla::F64) {
    size_t width = type.vector_width().value_or(1);
    if (width == 1) {
      sleef_func_name = "atan2";
    } else if (width == 2) {
      sleef_func_name = "Sleef_atan2d2_u10";
    } else if (width == 4) {
      sleef_func_name = "Sleef_atan2d4_u10";
    } else if (width == 8) {
      sleef_func_name = "Sleef_atan2d8_u10";
    }
  }

  if (sleef_func_name.empty()) {
    return absl::UnimplementedError(
        "Unsupported type/width for SLEEF atan2 JIT definition.");
  }

  llvm::LLVMContext& ctx = module->getContext();
  llvm::Type* ir_type = Type::TypeToIrType(type, ctx);

  std::string intrinsic_name = Name(y, x);
  llvm::Function* function = llvm::Function::Create(
      llvm::FunctionType::get(ir_type, {ir_type, ir_type}, /*isVarArg=*/false),
      llvm::Function::InternalLinkage, intrinsic_name, module);

  llvm::FunctionCallee sleef_fn =
      module->getOrInsertFunction(sleef_func_name, ir_type, ir_type, ir_type);

  llvm::BasicBlock* bb = llvm::BasicBlock::Create(ctx, "entry", function);
  llvm::IRBuilder<> builder(bb);
  llvm::Value* call =
      builder.CreateCall(sleef_fn, {function->getArg(0), function->getArg(1)});
  builder.CreateRet(call);

  return function;
}

}  // namespace xla::codegen::intrinsics
