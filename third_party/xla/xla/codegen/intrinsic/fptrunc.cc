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

#include "xla/codegen/intrinsic/fptrunc.h"

#include "absl/log/check.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
namespace {
llvm::Function* CreateFunction(llvm::Module* module, Type from, Type to) {
  DCHECK_OK(Type::VerifySameWidth(from, to));
  llvm::LLVMContext& context = module->getContext();
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      to.to_ir_type(context), {from.to_ir_type(context)},
      /*isVarArg=*/false);
  llvm::Function* func = llvm::dyn_cast<llvm::Function>(
      module->getOrInsertFunction(FpTrunc::Name(from, to), function_type)
          .getCallee());
  func->getArg(0)->setName("arg");
  return func;
}
}  // namespace

// Truncates an f32 value (scalar or vector) to bf16 with correct rounding.
static llvm::Function* TruncateF32ToBf16(llvm::Module* module, Type from,
                                         Type to) {
  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> builder(context);
  llvm::Function* func = CreateFunction(module, from, to);

  // Wraps a scalar type into a vector type if we are building a vector
  // intrinsic declaration.
  auto vec = [&](llvm::Type* scalar_type) -> llvm::Type* {
    if (from.vector_width().has_value()) {
      return llvm::VectorType::get(scalar_type, *from.vector_width(), false);
    }
    return scalar_type;
  };

  llvm::Type* i16_type = vec(builder.getInt16Ty());
  llvm::Type* i32_type = vec(builder.getInt32Ty());
  llvm::Type* bf16_type = vec(builder.getBFloatTy());

  llvm::Argument* arg = func->getArg(0);
  llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_bb);

  auto* i32 = builder.CreateBitCast(arg, i32_type);

  // Rounding bias for non-nan values.
  auto* lsb = builder.CreateAnd(builder.CreateLShr(i32, 16),
                                llvm::ConstantInt::get(i32_type, 1));
  auto* rounding_bias =
      builder.CreateAdd(llvm::ConstantInt::get(i32_type, 0x7fff), lsb);

  // For NaNs, we set all of them to quiet NaNs by masking the mantissa
  // so that only the MSB is 1, then simply truncate the original value
  // to retain the sign.
  auto* is_nan = builder.createIsFPClass(arg, llvm::FPClassTest::fcNan);
  auto* nan_mask = llvm::ConstantInt::get(i32_type, 0xFFC00000);
  auto* msb = llvm::ConstantInt::get(i32_type, 0x00400000);
  auto* quiet_nan = builder.CreateOr(builder.CreateAnd(i32, nan_mask), msb);
  auto* i16 = builder.CreateTrunc(
      builder.CreateLShr(
          builder.CreateSelect(is_nan, quiet_nan,
                               builder.CreateAdd(i32, rounding_bias)),
          16),
      i16_type);

  llvm::Value* result = builder.CreateBitCast(i16, bf16_type);
  builder.CreateRet(result);

  return func;
}

static llvm::Function* ExtendF8e5m2ToF16(llvm::Module* module, Type from,
                                         Type to) {
  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> builder(context);
  llvm::Function* func = CreateFunction(module, from, to);
  llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_bb);

  llvm::Value* as_int16 = builder.CreateZExt(
      func->getArg(0), Type(S16, from.vector_width()).to_ir_type(context));
  llvm::Value* shifted = builder.CreateShl(as_int16, 8);
  builder.CreateRet(builder.CreateBitCast(shifted, to.to_ir_type(context)));
  return func;
}

absl::StatusOr<llvm::Function*> FpTrunc::CreateDefinition(llvm::Module* module,
                                                          Type from, Type to) {
  TF_RETURN_IF_ERROR(Type::VerifySameWidth(from, to));

  if (from.element_type() == F32 && to.element_type() == BF16) {
    return TruncateF32ToBf16(module, from, to);
  }
  if (from.element_type() == F8E5M2 && to.element_type() == F16) {
    return ExtendF8e5m2ToF16(module, from, to);
  }

  return Internal("Unsupported fptrunc conversion: from=%s to=%s", from.name(),
                  to.name());
}

}  // namespace xla::codegen::intrinsics
