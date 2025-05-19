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

#include "xla/codegen/math/ldexp.h"

#include <cstdint>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"

namespace xla::codegen::math {

namespace {
llvm::Value* IntMax(llvm::IRBuilderBase& builder, llvm::Value* v1,
                    llvm::Value* v2) {
  llvm::Value* cmp = builder.CreateICmpSGT(v1, v2);
  return builder.CreateSelect(cmp, v1, v2);
}

llvm::Value* IntMin(llvm::IRBuilderBase& builder, llvm::Value* v1,
                    llvm::Value* v2) {
  llvm::Value* cmp = builder.CreateICmpSLT(v1, v2);
  return builder.CreateSelect(cmp, v1, v2);
}
}  // namespace

llvm::Function* CreateLdexpF64(llvm::Module* module, llvm::Type* vector_type) {
  // This implementation closely follows Eigen's ldexp implementation:
  // https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/arch/Default/GenericPacketMathFunctions.h#L226
  // One key difference being that the 2nd exponent argument is an integer or
  // vector of integers, not doubles.

  CHECK(vector_type != nullptr);
  CHECK(vector_type->isFloatingPointTy() || vector_type->isVectorTy())
      << "Vector type must be a floating point or vector of floating point.";
  CHECK(vector_type->getScalarType()->isDoubleTy())
      << "Only F64 (double) is supported for ldexp.";

  // Determine scalar or vector width for type creation.
  int num_elements = 1;
  llvm::Type* i64_type = llvm::Type::getInt64Ty(module->getContext());

  if (llvm::VectorType* vec_ty =
          llvm::dyn_cast<llvm::VectorType>(vector_type)) {
    num_elements = vec_ty->getElementCount().getKnownMinValue();
    i64_type = llvm::VectorType::get(i64_type, num_elements, false);
  }

  llvm::FunctionType* ldexp_func_type =
      llvm::FunctionType::get(vector_type, {vector_type, i64_type}, false);
  llvm::Function* ldexp_func = llvm::Function::Create(
      ldexp_func_type, llvm::Function::ExternalLinkage,
      absl::StrCat("xla.ldexp.", num_elements, "xf64"), module);
  llvm::AttributeList attrs = ldexp_func->getAttributes();
  attrs =
      attrs.addFnAttribute(module->getContext(), llvm::Attribute::AlwaysInline);
  ldexp_func->setAttributes(attrs);

  llvm::Argument* a = ldexp_func->getArg(0);
  a->setName("a");
  llvm::Argument* exponent = ldexp_func->getArg(1);
  exponent->setName("exponent");

  // 2. Create a basic block
  llvm::BasicBlock* entry_block =
      llvm::BasicBlock::Create(module->getContext(), "entry", ldexp_func);
  llvm::IRBuilder<> builder = llvm::IRBuilder<>(entry_block);

  auto int_vec = [=](int64_t val) {
    return llvm::ConstantInt::get(i64_type, val);
  };

  // Constants for double (F64) based on IEEE 754 standard.
  static constexpr int kMantissaBits = 52;  // Excludes implicit leading '1'.
  static constexpr int kExponentBits = 11;  // And one left for sign.

  // Exponent bias for IEEE 754 double = 1023.
  llvm::Value* bias_val = int_vec((1LL << (kExponentBits - 1)) - 1);

  llvm::Value* max_exponent = llvm::ConstantInt::get(i64_type, 2099);

  // Clamp the exponent: e = min(max(exponent, -max_exponent), max_exponent).
  llvm::Value* neg_max_exponent = builder.CreateNeg(max_exponent);
  llvm::Value* clamped_exponent = IntMax(builder, exponent, neg_max_exponent);
  clamped_exponent = IntMin(builder, clamped_exponent, max_exponent);

  llvm::Value* two_i64_for_shift = int_vec(2);
  // floor(e/4):
  llvm::Value* b = builder.CreateAShr(clamped_exponent, two_i64_for_shift);

  // Calculate 2^b (first factor 'c') using bit manipulation:
  //    a. Add `b` to the exponent `bias` (integer addition).
  //    b. Perform a logical shift left to position the
  //       new exponent value correctly within the 64-bit integer representing
  //       the floating-point number.
  //    c. Bitcast the resulting integer bit pattern to a double.
  llvm::Value* b_plus_bias = builder.CreateAdd(b, bias_val);
  llvm::Value* mantissa_shift = int_vec(kMantissaBits);
  llvm::Value* c_bits = builder.CreateShl(b_plus_bias, mantissa_shift);
  llvm::Value* c = builder.CreateBitCast(c_bits, vector_type);

  // Calculate `out = a * 2^(3b)` which is `a * c * c * c`.
  llvm::Value* out = builder.CreateFMul(a, c);
  out = builder.CreateFMul(out, c);
  out = builder.CreateFMul(out, c);

  // Calculate the remaining exponent adjustment: `b = e - 3*b`.
  llvm::Value* three_b = builder.CreateMul(int_vec(3), b);
  b = builder.CreateSub(clamped_exponent, three_b);

  // Calculate `2^(e-3b)` (the second scaling factor 'c').
  b_plus_bias = builder.CreateAdd(b, bias_val);
  c_bits = builder.CreateShl(b_plus_bias, mantissa_shift);
  c = builder.CreateBitCast(c_bits, vector_type);
  out = builder.CreateFMul(out, c);
  builder.CreateRet(out);

  return ldexp_func;
}

}  // namespace xla::codegen::math
