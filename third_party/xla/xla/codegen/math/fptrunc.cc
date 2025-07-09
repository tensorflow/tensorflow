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

#include <cstddef>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
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
#include "xla/primitive_util.h"

namespace xla::codegen::math {

std::string FptruncFunctionName(size_t num_elements, PrimitiveType from,
                                PrimitiveType to, bool add_suffix) {
  if (add_suffix) {
    return absl::StrCat("xla.fptrunc.",
                        primitive_util::LowercasePrimitiveTypeName(from),
                        ".to.", primitive_util::LowercasePrimitiveTypeName(to),
                        ".v", num_elements);
  }
  return absl::StrCat("xla.fptrunc.",
                      primitive_util::LowercasePrimitiveTypeName(from), ".to.",
                      primitive_util::LowercasePrimitiveTypeName(to));
}

llvm::Function* CreateFptruncF32ToBf16(llvm::Module* module,
                                       llvm::Type* input_type,
                                       bool add_suffix) {
  CHECK(input_type != nullptr);
  CHECK(input_type->isFloatingPointTy() || input_type->isVectorTy())
      << "Vector type must be a floating point or vector of floating point.";
  CHECK(input_type->getScalarType()->isFloatTy())
      << "Only F32 (float) is supported for xla.fptrunc.";

  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> builder(context);

  int num_elements = 1;
  if (llvm::VectorType* vec_ty = llvm::dyn_cast<llvm::VectorType>(input_type)) {
    num_elements = vec_ty->getElementCount().getKnownMinValue();
  }

  llvm::Type* result_type = llvm::Type::getBFloatTy(context);
  if (num_elements > 1) {
    result_type = llvm::VectorType::get(result_type, num_elements, false);
  }

  llvm::FunctionType* function_type =
      llvm::FunctionType::get(result_type, {input_type}, false);
  llvm::Function* func = llvm::dyn_cast<llvm::Function>(
      module
          ->getOrInsertFunction(
              FptruncFunctionName(num_elements, F32, BF16, add_suffix),
              function_type)
          .getCallee());

  llvm::Argument* input_x_arg = func->getArg(0);
  input_x_arg->setName("input_x");

  llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_bb);

  llvm::Type* i16_type = builder.getInt16Ty();
  if (num_elements > 1) {
    i16_type = llvm::VectorType::get(builder.getInt16Ty(), num_elements, false);
  }

  llvm::Type* i32_type = builder.getInt32Ty();
  if (num_elements > 1) {
    i32_type = llvm::VectorType::get(builder.getInt32Ty(), num_elements, false);
  }

  llvm::Type* bfloat_type = builder.getBFloatTy();
  if (num_elements > 1) {
    bfloat_type =
        llvm::VectorType::get(builder.getBFloatTy(), num_elements, false);
  }

  auto* i32 = builder.CreateBitCast(input_x_arg, i32_type);

  // Rounding bias for non-nan values.
  auto* lsb = builder.CreateAnd(builder.CreateLShr(i32, 16),
                                llvm::ConstantInt::get(i32_type, 1));

  auto* rounding_bias =
      builder.CreateAdd(llvm::ConstantInt::get(i32_type, 0x7fff), lsb);

  // For NaNs, we set all of them to quiet NaNs by masking the mantissa
  // so that only the MSB is 1, then simply truncate the original value
  // to retain the sign.
  auto* is_nan = builder.createIsFPClass(input_x_arg, llvm::FPClassTest::fcNan);
  auto* nan_mask = llvm::ConstantInt::get(i32_type, 0xFFC00000);
  auto* msb = llvm::ConstantInt::get(i32_type, 0x00400000);
  auto* quiet_nan = builder.CreateOr(builder.CreateAnd(i32, nan_mask), msb);
  auto* i16 = builder.CreateTrunc(
      builder.CreateLShr(
          builder.CreateSelect(is_nan, quiet_nan,
                               builder.CreateAdd(i32, rounding_bias)),
          16),
      i16_type);

  llvm::Value* result = builder.CreateBitCast(i16, bfloat_type);

  builder.CreateRet(result);

  return func;
}

}  // namespace xla::codegen::math
