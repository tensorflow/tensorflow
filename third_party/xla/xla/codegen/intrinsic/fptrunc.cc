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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/APFloat.h"
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
#include "xla/codegen/intrinsic/type.h"
#include "xla/primitive_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
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

static llvm::Function* ExtendF8e4m3fnToF16(llvm::Module* module, Type from,
                                           Type to) {
  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> b(context);
  llvm::Function* func = CreateFunction(module, from, to);
  llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(context, "entry", func);
  b.SetInsertPoint(entry_bb);
  llvm::Value* f8_value = func->getArg(0);

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

// For debugging purposes; print floating point values to stdout.
void EmitPrintf(llvm::Module* module, Type ty, llvm::Value* value,
                llvm::IRBuilder<>* b) {
  llvm::FunctionType* printf_type = llvm::FunctionType::get(
      b->getInt32Ty(), {b->getInt8Ty()->getPointerTo()}, true);
  llvm::FunctionCallee printf =
      module->getOrInsertFunction("printf", printf_type);
  std::string format_str = "FpTrunc Printf " + ty.name() + " ";
  if (ty.element_type() == F16) {
    ty = Type(F32, ty.vector_width());
    value = b->CreateFPExt(value, b->getFloatTy());
  }
  for (int i = 0; i < ty.vector_width().value_or(1); ++i) {
    switch (ty.element_type()) {
      case F32:
        format_str += "%f ";
        break;
      case F64:
        format_str += "%F ";
        break;
      default:
        LOG(FATAL) << "Unsupported type: " << ty.name();
    }
  }
  format_str += "\n";

  llvm::Value* format_str_ptr = b->CreateGlobalString(format_str);
  std::vector<llvm::Value*> args = {format_str_ptr};
  if (ty.vector_width().has_value()) {
    for (int i = 0; i < ty.vector_width().value(); ++i) {
      args.push_back(b->CreateExtractElement(value, i));
    }
  } else {
    args.push_back(value);
  }
  b->CreateCall(printf, args);
}

// Converts a floating-point value to an 8-bit floating-point value with
// specified properties.
//
// This function is vector-capable. If `from_type` is a vector type, it will
// generate vector instructions for the conversion.
absl::StatusOr<llvm::Function*> EmitFxxToF8E(llvm::Module* module,
                                             const Type& from, const Type& to) {
  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> b(context);
  if (!from.is_scalar() && !from.is_vector()) {
    return absl::InvalidArgumentError("from_type must be a scalar or vector.");
  }
  TF_RETURN_IF_ERROR(Type::VerifySameWidth(from, to));

  llvm::Function* func = CreateFunction(module, from, to);
  llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(context, "entry", func);
  b.SetInsertPoint(entry_bb);
  llvm::Value* from_value = func->getArg(0);

  const PrimitiveType fx_type = from.element_type();
  if (fx_type != F16 && fx_type != F32 && fx_type != F64) {
    return absl::InvalidArgumentError("from_type must be F16, F32, or F64.");
  }

  const uint64_t fx_width = primitive_util::BitWidth(fx_type);
  const uint64_t fx_bias = primitive_util::ExponentBias(fx_type);
  const uint64_t fx_mantissa_bits =
      primitive_util::SignificandWidth(fx_type) - 1;
  const uint64_t fx_exp_bits = primitive_util::ExponentWidth(fx_type);
  const uint64_t f8_exp_bits = primitive_util::ExponentWidth(to.element_type());
  const bool is_fnuz = !primitive_util::HasInfinity(to.element_type()) &&
                       !primitive_util::HasNegativeZero(to.element_type());
  // HACK: The F8E4M3FNUZ format has a max value of 240, which implies a bias
  // of 8. The value in primitive_util is 7.
  // Verified with ml_dtypes
  int f8_bias = primitive_util::ExponentBias(to.element_type());
  if (to.element_type() == F8E4M3FNUZ) {
    f8_bias = 8;
  }

  const uint64_t f8_mantissa_bits =
      primitive_util::SignificandWidth(to.element_type()) - 1;
  const uint64_t exponent_bias_difference = fx_bias - f8_bias;

  const PrimitiveType ix_primitive_type =
      primitive_util::UnsignedIntegralTypeForBitWidth(fx_width);
  llvm::Type* ix_type =
      Type(ix_primitive_type, from.vector_width()).to_ir_type(context);
  llvm::Type* i8_type = to.to_ir_type(context);

  auto make_const = [&](uint64_t val, llvm::Type* type) -> llvm::Constant* {
    llvm::IntegerType* scalar_ty =
        llvm::IntegerType::get(context, type->getScalarSizeInBits());
    uint64_t mask = (scalar_ty->getBitWidth() == 64)
                        ? ~0ULL
                        : ((1ULL << scalar_ty->getBitWidth()) - 1);
    llvm::Constant* scalar_const =
        llvm::ConstantInt::get(scalar_ty, val & mask);
    if (type->isVectorTy()) {
      auto vec_ty = llvm::cast<llvm::FixedVectorType>(type);
      return llvm::ConstantVector::getSplat(vec_ty->getElementCount(),
                                            scalar_const);
    }
    return scalar_const;
  };

  auto ix_const = [&](uint64_t val) { return make_const(val, ix_type); };
  auto i8_const = [&](uint64_t val) { return make_const(val, i8_type); };

  // If biases and exponent widths are identical (e.g., F16 -> F8E5M2),
  // we can delegate all logic to EmitReducePrecisionIR and do a simple shift.
  if (fx_bias == f8_bias && fx_exp_bits == f8_exp_bits) {
    LOG(INFO) << "Using fast path for " << from.name() << " -> " << to.name();
    TF_ASSIGN_OR_RETURN(
        llvm::Value * reduced_precision,
        llvm_ir::EmitReducePrecisionIR(
            /*src_ty=*/fx_type, from_value,
            /*dest_exponent_bits=*/f8_exp_bits,  // <-- Pass DEST exp bits
            /*dest_mantissa_bits=*/f8_mantissa_bits,
            /*quiet_nans=*/true, &b));

    // Bitcast to integer to perform the shift.
    llvm::Value* as_int = b.CreateBitCast(reduced_precision, ix_type);

    // Shift the 8 relevant bits (1 sign + 5 exp + 2 mantissa for F8E5M2)
    // into position. The shift amount is the difference in mantissa bits.
    const uint64_t mantissa_shift = fx_mantissa_bits - f8_mantissa_bits;
    llvm::Value* shifted = b.CreateLShr(as_int, ix_const(mantissa_shift));

    // Truncate from i16/i32/i64 down to i8.
    llvm::Value* truncated = b.CreateTrunc(shifted, i8_type);

    b.CreateRet(truncated);
    return func;  // We are done, skip the general "slow" logic.
  }

  llvm::Constant* nosign_mask = ix_const((1ULL << (fx_width - 1)) - 1);
  llvm::Constant* sign_mask = ix_const(1ULL << (fx_width - 1));
  llvm::Constant* min_normal_value =
      ix_const((exponent_bias_difference + 1) << fx_mantissa_bits);

  using llvm::Value;
  Value* fx_as_int = b.CreateBitCast(from_value, ix_type);
  Value* fx_abs_bits = b.CreateAnd(fx_as_int, nosign_mask);

  Value* fx_sign = b.CreateAnd(fx_as_int, sign_mask);
  fx_sign = b.CreateLShr(fx_sign, ix_const(fx_width - 8));
  Value* f8_sign = b.CreateTrunc(fx_sign, i8_type);

  // To avoid `ReducePrecision`'s assumptions, we only use it for mantissa
  // rounding, keeping the original exponent bits.
  absl::StatusOr<Value*> fx_reduced_statusor = llvm_ir::EmitReducePrecisionIR(
      /*src_ty=*/fx_type, from_value,
      /*dest_exponent_bits=*/primitive_util::ExponentWidth(fx_type),
      /*dest_mantissa_bits=*/f8_mantissa_bits,
      /*quiet_nans=*/true, &b);
  CHECK_OK(fx_reduced_statusor.status());  // Crash OK
  Value* fx_reduced = b.CreateBitCast(fx_reduced_statusor.value(), ix_type);
  fx_reduced = b.CreateAnd(fx_reduced, nosign_mask);

  // Round small values up to the minimum normal F8 value.
  fx_reduced = b.CreateSelect(b.CreateICmpULT(fx_reduced, min_normal_value),
                              min_normal_value, fx_reduced);

  // Adjust the exponent bias.
  fx_reduced = b.CreateSub(
      fx_reduced, ix_const(exponent_bias_difference << fx_mantissa_bits));
  // Shift mantissa into place.
  Value* f8_bits_shifted =
      b.CreateLShr(fx_reduced, ix_const(fx_mantissa_bits - f8_mantissa_bits));
  Value* f8_bits = b.CreateTrunc(f8_bits_shifted, i8_type);

  // Calculate the threshold for overflow. This is the largest value that maps
  // to a finite F8 value, including a rounding component.
  const bool has_inf = primitive_util::HasInfinity(to.element_type());
  const uint64_t max_finite_f8_exp =
      has_inf ? (1ULL << f8_exp_bits) - 2 : (1ULL << f8_exp_bits) - 1;
  const uint64_t max_finite_f8_man = (1ULL << f8_mantissa_bits) - 1;
  const uint64_t max_finite_fx_exp =
      max_finite_f8_exp + exponent_bias_difference;
  const uint64_t man_shift = fx_mantissa_bits - f8_mantissa_bits;
  const uint64_t max_finite_value_exp = max_finite_fx_exp << fx_mantissa_bits;
  const uint64_t max_finite_value_man = max_finite_f8_man << man_shift;
  const uint64_t rounding_bits = (1ULL << man_shift) - 1;
  const uint64_t max_finite_value =
      max_finite_value_exp | max_finite_value_man | rounding_bits;
  Value* is_overflow = b.CreateICmpUGT(fx_abs_bits, ix_const(max_finite_value));

  // Handle format-specific overflow behavior.
  if (has_inf) {
    // For standard formats, overflow becomes infinity.
    // Also propagate Inf/NaN.
    const uint64_t fx_exp_mask =
        ((1ULL << primitive_util::ExponentWidth(fx_type)) - 1)
        << fx_mantissa_bits;
    Value* is_inf_or_nan_input = b.CreateICmpEQ(
        b.CreateAnd(fx_abs_bits, ix_const(fx_exp_mask)), ix_const(fx_exp_mask));

    const uint64_t fx_mantissa_mask = (1ULL << fx_mantissa_bits) - 1;
    Value* fx_mantissa = b.CreateAnd(fx_abs_bits, ix_const(fx_mantissa_mask));
    Value* is_nan_input = b.CreateAnd(is_inf_or_nan_input,
                                      b.CreateICmpNE(fx_mantissa, ix_const(0)));

    const uint64_t f8_exp_mask = ((1ULL << f8_exp_bits) - 1)
                                 << f8_mantissa_bits;
    const uint64_t f8_qnan_mantissa = 1ULL << (f8_mantissa_bits - 1);
    const uint64_t f8_nan_pattern = f8_exp_mask | f8_qnan_mantissa;
    // If the input is NaN, the output is NaN.
    // If the input is Inf or overflows, the output is Inf.
    // Otherwise, it's the computed finite value.
    Value* finite_or_inf =
        b.CreateSelect(is_overflow, i8_const(f8_exp_mask), f8_bits);
    Value* inf_or_nan = b.CreateSelect(is_nan_input, i8_const(f8_nan_pattern),
                                       i8_const(f8_exp_mask));
    f8_bits = b.CreateSelect(is_inf_or_nan_input, inf_or_nan, finite_or_inf);
  } else {
    // For fn/fnuz formats, overflow and input Inf/NaN become NaN.
    const uint64_t fx_exp_mask =
        ((1ULL << primitive_util::ExponentWidth(fx_type)) - 1)
        << fx_mantissa_bits;
    Value* is_inf_or_nan_input = b.CreateICmpEQ(
        b.CreateAnd(fx_abs_bits, ix_const(fx_exp_mask)), ix_const(fx_exp_mask));

    const uint64_t f8_nan_pattern =
        is_fnuz ? 0x80 : (1ULL << (f8_exp_bits + f8_mantissa_bits)) - 1;

    Value* is_special = b.CreateOr(is_overflow, is_inf_or_nan_input);
    f8_bits = b.CreateSelect(is_special, i8_const(f8_nan_pattern), f8_bits);

    if (is_fnuz) {
      // For FNUZ, the NaN value is used for all special cases, and it is
      // unsigned.
      f8_sign = b.CreateSelect(is_special, i8_const(0), f8_sign);
    }
  }

  // Handle halfway points for denormals.
  f8_bits = llvm_ir::HandleHalfwayPointsFxToF8(
      /*fx_type=*/fx_type, /*f8_exponent_bits=*/f8_exp_bits,
      /*f8_mantissa_bits=*/f8_mantissa_bits, /*f8_bias=*/f8_bias,
      /*fx_abs_bits=*/fx_abs_bits, /*f8_bits=*/f8_bits,
      /*vector_width=*/from.vector_width(), &b);

  // For FNUZ types, -0.0 should become +0.0. Since f8_bits is derived from
  // fx_abs_bits, it will be 0 if the input is +/-0.0. We just need to
  // ensure the sign bit is not set in this case.
  if (is_fnuz) {
    Value* is_zero = b.CreateICmpEQ(fx_abs_bits, ix_const(0));
    f8_sign = b.CreateSelect(is_zero, i8_const(0), f8_sign);
  }

  // Apply the sign bit.
  f8_bits = b.CreateOr(f8_bits, f8_sign);

  b.CreateRet(f8_bits);
  return func;
}

absl::StatusOr<llvm::Function*> FpTrunc::CreateDefinition(llvm::Module* module,
                                                          Type from, Type to) {
  TF_RETURN_IF_ERROR(Type::VerifySameWidth(from, to));

  if (primitive_util::IsF8Type(to.element_type()) &&
      (from.element_type() == F16 || from.element_type() == F32 ||
       from.element_type() == F64)) {
    return EmitFxxToF8E(module, from, to);
  }
  if (from.element_type() == F32 && to.element_type() == BF16) {
    return TruncateF32ToBf16(module, from, to);
  }
  if (from.element_type() == F8E5M2 && to.element_type() == F16) {
    return ExtendF8e5m2ToF16(module, from, to);
  }
  if (from.element_type() == F8E4M3FN && to.element_type() == F16) {
    return ExtendF8e4m3fnToF16(module, from, to);
  }

  return Internal("Unsupported fptrunc conversion: from=%s to=%s", from.name(),
                  to.name());
}

}  // namespace xla::codegen::intrinsics
