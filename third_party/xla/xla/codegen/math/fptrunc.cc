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

#include "xla/codegen/math/fptrunc.h"

#include <optional>
#include <string>

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
#include "xla/codegen/math/intrinsic.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
// Truncates an f32 value (scalar or vector) to bf16 with correct rounding.
static llvm::Function* TruncateF32ToBf16(llvm::Module* module, Type from,
                                         Type to) {
  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> builder(context);
  DCHECK_EQ(from.element_type(), F32);
  DCHECK_EQ(to.element_type(), BF16);

  // Wraps a scalar type into a vector type if we are building a vector
  // intrinsic declaration.
  auto vec = [&](llvm::Type* scalar_type) -> llvm::Type* {
    if (from.vector_width()) {
      return llvm::VectorType::get(scalar_type, *from.vector_width(), false);
    }
    return scalar_type;
  };

  llvm::Type* i16_type = vec(builder.getInt16Ty());
  llvm::Type* i32_type = vec(builder.getInt32Ty());
  llvm::Type* f32_type = vec(builder.getFloatTy());
  llvm::Type* bf16_type = vec(builder.getBFloatTy());

  llvm::FunctionType* function_type =
      llvm::FunctionType::get(bf16_type, {f32_type}, false);
  llvm::Function* func = llvm::dyn_cast<llvm::Function>(
      module->getOrInsertFunction(FpTrunc::Name(from, to), function_type)
          .getCallee());

  llvm::Argument* arg = func->getArg(0);
  arg->setName("arg");

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

absl::StatusOr<llvm::Function*> FpTrunc::CreateDefinition(llvm::Module* module,
                                                          Type from, Type to) {
  TF_RETURN_IF_ERROR(Type::VerifySameWidth(from, to));

  if (from.element_type() == F32 && to.element_type() == BF16) {
    return TruncateF32ToBf16(module, from, to);
  }

  return Internal("Unsupported fptrunc conversion: from=%s to=%s", from.name(),
                  to.name());
}

}  // namespace xla::codegen::intrinsics
