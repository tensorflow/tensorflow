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
#include "llvm/ADT/APInt.h"
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
#include "xla/service/llvm_ir/llvm_util.h"
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

static llvm::Function* TruncateF16ToF8e4m3fn(llvm::Module* module, Type from,
                                             Type to) {
  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> b(context);
  llvm::Function* func = CreateFunction(module, from, to);
  llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(context, "entry", func);
  b.SetInsertPoint(entry_bb);
  llvm::Value* f16_value = func->getArg(0);

  using llvm::APInt;
  using llvm::Value;

  llvm::Type* i8_ty = to.to_ir_type(context);
  llvm::Type* i16_ty = Type(U16, from.vector_width()).to_ir_type(context);
  auto i8_const = [&](int val) { return llvm::ConstantInt::get(i8_ty, val); };
  auto i16_const = [&](int val) { return llvm::ConstantInt::get(i16_ty, val); };

  // Cast the input value to an integer for bitwise manipulation. Get the
  // absolute value of the input value.
  //   f16_as_int = bitcast(f16_value, int)
  //   f16_abs_bits = f16_as_int & 0x7FFF
  Value* f16_as_int = b.CreateBitCast(f16_value, i16_ty);
  llvm::Value* f16_abs_bits = b.CreateAnd(f16_as_int, i16_const(0x7FFF));

  // Get the sign.
  //   f8_sign = (f16_as_int & 0x8000) >> 8
  Value* f16_sign = b.CreateAnd(f16_as_int, i16_const(0x8000));
  f16_sign = b.CreateLShr(f16_sign, i16_const(8));
  Value* f8_sign = b.CreateTrunc(f16_sign, i8_ty);

  // Truncate the mantissa to 3 bits. ReducePrecision cannot deal with
  // f8E4M3FN's NaN representations, so don't use ReducePrecision to handle
  // exponent reduction. Denormal values are not handled properly here and are
  // dealt with later in this function.
  absl::StatusOr<Value*> f16_reduced_statusor =
      llvm_ir::EmitReducePrecisionIR(/*src_ty=*/F16, f16_value,
                                     /*dest_exponent_bits=*/5,
                                     /*dest_mantissa_bits=*/3,
                                     /*quiet_nans=*/false, &b);
  CHECK_OK(f16_reduced_statusor.status());  // Crash OK
  Value* f16_reduced = f16_reduced_statusor.value();
  f16_reduced = b.CreateBitCast(f16_reduced, i16_ty);

  // Remove the sign bit.
  //   f16_reduced = f16_reduced & 0x7FFF
  f16_reduced = b.CreateAnd(f16_reduced, i16_const(0x7FFF));

  // Bits of the F16 representation of the smallest F8 normal value.
  constexpr int min_normal_value = 0x2400;

  // Round values smaller than the smallest F8 normal value up to the smallest
  // F8 normal value. The case where we round to a denormal value is handled
  // later.
  //    f16_reduced = max(f16_reduced, min_normal_value)
  f16_reduced =
      b.CreateSelect(b.CreateICmpULT(f16_reduced, i16_const(min_normal_value)),
                     i16_const(min_normal_value), f16_reduced);

  constexpr int exponent_bias_difference = 15 - 7;
  constexpr int f8_exponent_bits = 4;
  constexpr int f16_mantissa_bits = 10;
  constexpr int f8_mantissa_bits = 3;
  constexpr int mantissa_bits_difference = f16_mantissa_bits - f8_mantissa_bits;

  // Adjust the exponent by subtracting the difference in exponent bias.
  //   f16_reduced -= (exponent_bias_difference << f16_mantissa_bits)
  f16_reduced = b.CreateSub(
      f16_reduced, i16_const(exponent_bias_difference << f16_mantissa_bits));

  // Shift to convert to F8.
  //   f8_bits = f16_reduced >> mantissa_bits_difference;
  Value* f8_bits =
      b.CreateLShr(f16_reduced, i16_const(mantissa_bits_difference));
  f8_bits = b.CreateTrunc(f8_bits, i8_ty);

  // Bits of the highest F16 value that gets converted to a finite F8 value.
  // In binary: 0 10111 1101111111
  constexpr int max_finite_value = 0x5F7F;

  // If we're above the maximum F8 value, output NaN.
  //   f8_bits = f16_abs_bits > max_finite_value ? 0x7F : f8_bits
  f8_bits =
      b.CreateSelect(b.CreateICmpUGT(f16_abs_bits, i16_const(max_finite_value)),
                     i8_const(0x7F), f8_bits);

  // Handle F16 values that are halfway between denormal F8 values.
  f8_bits = llvm_ir::HandleHalfwayPointsFxToF8<F16, f8_exponent_bits>(
      f16_abs_bits, f8_bits, from.vector_width(), &b);

  // Set the sign bit.
  //   f8_bits |= f8_sign
  f8_bits = b.CreateOr(f8_bits, f8_sign);
  b.CreateRet(f8_bits);
  return func;
}

static llvm::Function* ExtendF8e4m3fnToF16(llvm::Module* module, Type from,
                                           Type to) {
  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> b(context);
  llvm::Function* func = CreateFunction(module, from, to);
  llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(context, "entry", func);
  b.SetInsertPoint(entry_bb);
  llvm::Value* f8_value = func->getArg(0);

  using llvm::APInt;
  using llvm::Value;

  llvm::Type* i8_type = from.to_ir_type(context);
  llvm::Type* i16_type = Type(U16, to.vector_width()).to_ir_type(context);
  auto i8_const = [i8_type](int val) {
    return llvm::ConstantInt::get(i8_type, val);
  };
  auto i16_const = [i16_type](int val) {
    return llvm::ConstantInt::get(i16_type, val);
  };

  // Cast the input value to an integer for bitwise manipulation. Get the
  // absolute value of the input value.
  //   f8_as_int = bitcast(f16_value, int)
  //   f8_abs_bits = f8_as_int & 0x7F
  Value* f8_as_int = b.CreateBitCast(f8_value, i8_type);
  Value* f8_abs_bits = b.CreateAnd(f8_as_int, i8_const(0x7F));

  // We assume below that the value is neither NaN nor denormal. If it NaN or
  // denormal, the output is set to NaN or zero at the end using Select
  // instructions.

  // Get the sign:
  //   f16_sign = (f8_as_int & 0x80) << 8
  Value* f8_sign = b.CreateAnd(f8_as_int, i8_const(0x80));
  Value* f16_sign = b.CreateZExt(f8_sign, i16_type);
  f16_sign = b.CreateShl(f16_sign, i16_const(8));

  constexpr int exponent_bias_difference = 15 - 7;
  constexpr int f16_mantissa_bits = 10;
  constexpr int f8_mantissa_bits = 3;
  constexpr int mantissa_bits_difference = f16_mantissa_bits - f8_mantissa_bits;
  constexpr int f8_mantissa_mask = (1 << f8_mantissa_bits) - 1;

  // Get the exponent:
  //   f8_exponent = (f8_as_int & 0x78) >> f8_mantissa_bits
  Value* f8_exponent_bits = b.CreateAnd(f8_as_int, i8_const(0x78));
  Value* f8_exponent =
      b.CreateLShr(f8_exponent_bits, i8_const(f8_mantissa_bits));

  // Adjust the exponent by adding the difference in exponent bias:
  //   f16_exponent = (f8_exponent + exponent_bias_difference)
  //                  << f16_mantissa_bits
  Value* f16_exponent =
      b.CreateAdd(f8_exponent, i8_const(exponent_bias_difference));
  f16_exponent = b.CreateZExt(f16_exponent, i16_type);
  f16_exponent = b.CreateShl(f16_exponent, i16_const(f16_mantissa_bits));

  // Get the mantissa:
  //   f16_mantissa = (f8_mantissa & f8_mantissa_mask)
  //                  << mantissa_bits_difference
  Value* f8_mantissa = b.CreateAnd(f8_as_int, i8_const(f8_mantissa_mask));
  Value* f16_mantissa = b.CreateZExt(f8_mantissa, i16_type);
  f16_mantissa = b.CreateShl(f16_mantissa, i16_const(mantissa_bits_difference));

  // Combine the exponent and mantissa:
  //   f16_as_int = f16_exponent | f16_mantissa
  Value* f16_as_int = b.CreateOr(f16_exponent, f16_mantissa);

  // Set output to NaN if input is NaN
  //   f16_as_int = f8_abs_bits == 0x7F ? 0x7E00 : f16_as_int
  Value* is_nan = b.CreateICmpEQ(f8_abs_bits, i8_const(0x7F));
  f16_as_int = b.CreateSelect(is_nan, i16_const(0x7E00), f16_as_int);

  // Map from F8 denormal value to F16 value.
  int f8_denormal_to_f16[8] = {
      0x0000,  // 0
      0x1800,  // 1/8 * 2^-6
      0x1C00,  // 2/8 * 2^-6
      0x1E00,  // 3/8 * 2^-6
      0x2000,  // 4/8 * 2^-6
      0x2100,  // 5/8 * 2^-6
      0x2200,  // 6/8 * 2^-6
      0x2300,  // 7/8 * 2^-6
  };

  // If the F8 value is denormal, use the map above to determine the correct F16
  // value.
  //    if (f8_abs_bits < 8) { f16_as_int = f8_denormal_to_f16[f8_abs_bits]; }
  for (int i = 0; i < 8; i++) {
    Value* is_denormal_value = b.CreateICmpEQ(f8_abs_bits, i8_const(i));
    f16_as_int = b.CreateSelect(is_denormal_value,
                                i16_const(f8_denormal_to_f16[i]), f16_as_int);
  }

  // Set the sign bit.
  //   f16_as_int |= f16_sign
  f16_as_int = b.CreateOr(f16_as_int, f16_sign);
  b.CreateRet(b.CreateBitCast(f16_as_int, to.to_ir_type(context)));
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
  if (from.element_type() == F8E4M3FN && to.element_type() == F16) {
    return ExtendF8e4m3fnToF16(module, from, to);
  }
  if (from.element_type() == F16 && to.element_type() == F8E4M3FN) {
    return TruncateF16ToF8e4m3fn(module, from, to);
  }

  return Internal("Unsupported fptrunc conversion: from=%s to=%s", from.name(),
                  to.name());
}

}  // namespace xla::codegen::intrinsics
