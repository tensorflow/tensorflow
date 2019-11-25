/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

using absl::StrCat;
using llvm_ir::IrArray;
using llvm_ir::IrName;
using llvm_ir::SetToFirstInsertPoint;

namespace {

int64 GlobalRandomValue() {
  static auto* mu = new tensorflow::mutex();
  static std::mt19937_64 rng{42};
  tensorflow::mutex_lock l(*mu);
  return rng();
}

StatusOr<llvm::Value*> EmitReducePrecisionIR(PrimitiveType src_ty,
                                             llvm::Value* x,
                                             int64 dest_exponent_bits,
                                             int64 dest_mantissa_bits,
                                             llvm::IRBuilder<>* b) {
  using llvm::APInt;

  if (!primitive_util::IsFloatingPointType(src_ty)) {
    return Unimplemented(
        "ReducePrecision cannot accept non-floating-point type %s.",
        PrimitiveType_Name(src_ty));
  }

  // Integer and float types for casting and constant generation.
  llvm::Type* float_type = x->getType();
  int64 nbits = float_type->getPrimitiveSizeInBits();
  llvm::IntegerType* int_type = b->getIntNTy(nbits);

  // SignificandWidth includes the implicit extra bit.
  int src_mantissa_bits = primitive_util::SignificandWidth(src_ty) - 1;
  int src_exponent_bits = nbits - 1 - src_mantissa_bits;

  // Cast the input value to an integer for bitwise manipulation.
  llvm::Value* x_as_int = b->CreateBitCast(x, int_type);

  if (dest_mantissa_bits < src_mantissa_bits) {
    // Last remaining mantissa bit.
    APInt last_mantissa_bit_mask(nbits, 1);
    last_mantissa_bit_mask <<= src_mantissa_bits - dest_mantissa_bits;

    // Compute rounding bias for round-to-nearest with ties to even.  This is
    // equal to a base value of 0111... plus one bit if the last remaining
    // mantissa bit is 1.
    APInt base_rounding_bias = last_mantissa_bit_mask.lshr(1) - 1;
    llvm::Value* x_last_mantissa_bit = b->CreateLShr(
        b->CreateAnd(x_as_int,
                     llvm::ConstantInt::get(int_type, last_mantissa_bit_mask)),
        (src_mantissa_bits - dest_mantissa_bits));
    llvm::Value* x_rounding_bias =
        b->CreateAdd(x_last_mantissa_bit,
                     llvm::ConstantInt::get(int_type, base_rounding_bias));

    // Add rounding bias, and mask out truncated bits.  Note that the case
    // where adding the rounding bias overflows into the exponent bits is
    // correct; the non-masked mantissa bits will all be zero, and the
    // exponent will be incremented by one.
    APInt truncation_mask = ~(last_mantissa_bit_mask - 1);
    x_as_int = b->CreateAdd(x_as_int, x_rounding_bias);
    x_as_int = b->CreateAnd(x_as_int,
                            llvm::ConstantInt::get(int_type, truncation_mask));
  }

  if (dest_exponent_bits < src_exponent_bits) {
    APInt sign_bit_mask(nbits, 1);
    sign_bit_mask <<= nbits - 1;

    APInt exp_bits_mask(nbits, 1);
    exp_bits_mask = ((exp_bits_mask << src_exponent_bits) - 1)
                    << src_mantissa_bits;

    // An exponent of 2^(n-1)-1 -- that is, 0111... with the zero in the most-
    // significant bit -- is equal to 1.0f for all exponent sizes.  Adding
    // 2^(n-1)-1 to this gives us the highest non-infinite exponent for a bit-
    // size of n, and subtracting 2^(n-1)-1 from this gives us the lowest'
    // exponent (corresponding to 0.0f).
    //
    // Thus, the f32 exponent corresponding to the highest non-infinite
    // exponent for a bit size of n is (2^7-1) + 2^(n-1)-1, and the f32
    // exponent corresponding to the lowest exponent for a bit size of n is
    // (2^7-1) - 2^(n-1)-1.
    //
    // Note that we have already checked that exponents_bits >= 1.
    APInt exponent_bias(nbits, 1);
    exponent_bias = (exponent_bias << (src_exponent_bits - 1)) - 1;

    APInt reduced_exponent_bias(nbits, 1);
    reduced_exponent_bias =
        (reduced_exponent_bias << (dest_exponent_bits - 1)) - 1;

    APInt reduced_max_exponent = exponent_bias + reduced_exponent_bias;
    APInt reduced_min_exponent = exponent_bias - reduced_exponent_bias;

    // Do we overflow or underflow?
    llvm::Value* x_exponent =
        b->CreateAnd(x_as_int, llvm::ConstantInt::get(int_type, exp_bits_mask));
    llvm::Value* x_overflows = b->CreateICmpUGT(
        x_exponent, llvm::ConstantInt::get(
                        int_type, reduced_max_exponent << src_mantissa_bits));
    llvm::Value* x_underflows = b->CreateICmpULE(
        x_exponent, llvm::ConstantInt::get(
                        int_type, reduced_min_exponent << src_mantissa_bits));

    // Compute appropriately-signed values of zero and infinity.
    llvm::Value* x_signed_zero =
        b->CreateAnd(x_as_int, llvm::ConstantInt::get(int_type, sign_bit_mask));
    llvm::Value* x_signed_inf = b->CreateOr(
        x_signed_zero, llvm::ConstantInt::get(int_type, exp_bits_mask));

    // Force to zero or infinity if overflow or underflow.  (Note that this
    // truncates all denormal values to zero, rather than rounding them.)
    x_as_int = b->CreateSelect(x_overflows, x_signed_inf, x_as_int);
    x_as_int = b->CreateSelect(x_underflows, x_signed_zero, x_as_int);
  }

  // Cast the result back to a floating-point type.
  llvm::Value* result = b->CreateBitCast(x_as_int, float_type);

  // Correct result for NaN inputs.
  //
  // The exponent handling will "normalize" NaN values to infinities, which is
  // undesirable (except in the case with no mantissa bits, in which case it
  // is mandatory).  This logic also handles cases where mantissa-rounding
  // causes a NaN's mantissa to overflow into the exponent bits, which would
  // otherwise create an erroneous zero value.
  //
  // If the fast-math flags are set to assume no NaNs, the comparison is likely
  // to be optimized away, so there's no point in even emitting it.
  if (!b->getFastMathFlags().noNaNs()) {
    llvm::Value* x_is_nan = b->CreateFCmpUNO(x, x);

    if (dest_mantissa_bits > 0) {
      result = b->CreateSelect(x_is_nan, x, result);
    } else {
      result = b->CreateSelect(
          x_is_nan, llvm::ConstantFP::getInfinity(float_type), result);
    }
  }
  return result;
}

StatusOr<llvm::Value*> EmitF32ToBF16(llvm::Value* f32_value,
                                     llvm::IRBuilder<>* b) {
  TF_ASSIGN_OR_RETURN(
      auto reduced_precision,
      EmitReducePrecisionIR(
          /*src_ty=*/F32, f32_value,
          /*dest_exponent_bits=*/primitive_util::kBFloat16ExponentBits,
          /*dest_mantissa_bits=*/primitive_util::kBFloat16MantissaBits, b));
  auto as_int32 = b->CreateBitCast(reduced_precision, b->getInt32Ty());
  auto shifted = b->CreateLShr(as_int32, 16);
  auto truncated = b->CreateTrunc(shifted, b->getInt16Ty());
  return b->CreateBitCast(truncated, b->getInt16Ty());
}

llvm::Value* EmitBF16ToF32(llvm::Value* bf16_value, llvm::IRBuilder<>* b) {
  auto as_int16 = b->CreateBitCast(bf16_value, b->getInt16Ty());
  auto as_int32 = b->CreateZExt(as_int16, b->getInt32Ty());
  auto shifted = b->CreateShl(as_int32, 16);
  return b->CreateBitCast(shifted, b->getFloatTy());
}

llvm::Value* EmitIntegralToFloating(llvm::Value* integer_value,
                                    PrimitiveType from_type,
                                    PrimitiveType to_type, llvm::Module* module,
                                    llvm::IRBuilder<>* b) {
  if (primitive_util::IsSignedIntegralType(from_type)) {
    return b->CreateSIToFP(integer_value,
                           llvm_ir::PrimitiveTypeToIrType(to_type, module));
  } else {
    CHECK(primitive_util::IsUnsignedIntegralType(from_type) ||
          from_type == PRED);
    return b->CreateUIToFP(integer_value,
                           llvm_ir::PrimitiveTypeToIrType(to_type, module));
  }
}

}  // namespace

StatusOr<llvm::Value*> ElementalIrEmitter::EmitUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) {
  if (ShapeUtil::ElementIsIntegral(op->operand(0)->shape()) ||
      op->operand(0)->shape().element_type() == PRED) {
    return EmitIntegerUnaryOp(op, operand_value);
  } else if (ShapeUtil::ElementIsComplex(op->operand(0)->shape())) {
    return EmitComplexUnaryOp(op, operand_value);
  } else {
    return EmitFloatUnaryOp(op, operand_value);
  }
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitIntegerUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) {
  switch (op->opcode()) {
    case HloOpcode::kConvert: {
      PrimitiveType from_type = op->operand(0)->shape().element_type();
      PrimitiveType to_type = op->shape().element_type();
      CHECK(primitive_util::IsIntegralType(from_type) || from_type == PRED)
          << from_type;
      if (from_type == to_type) {
        return operand_value;
      }
      if (to_type == PRED) {
        return b_->CreateZExt(
            ICmpNE(operand_value,
                   llvm::ConstantInt::get(operand_value->getType(), 0)),
            llvm_ir::PrimitiveTypeToIrType(PRED, module_));
      }
      if (primitive_util::IsIntegralType(to_type)) {
        return IntCast(operand_value,
                       llvm_ir::PrimitiveTypeToIrType(to_type, module_),
                       primitive_util::IsSignedIntegralType(from_type));
      }
      if (primitive_util::IsFloatingPointType(to_type)) {
        if (to_type == BF16) {
          return EmitF32ToBF16(EmitIntegralToFloating(operand_value, from_type,
                                                      F32, module_, b_),
                               b_);
        }
        return EmitIntegralToFloating(operand_value, from_type, to_type,
                                      module_, b_);
      }
      if (primitive_util::IsComplexType(to_type)) {
        auto to_ir_component_type = llvm_ir::PrimitiveTypeToIrType(
            primitive_util::ComplexComponentType(to_type), module_);
        if (primitive_util::IsSignedIntegralType(from_type)) {
          return EmitComposeComplex(
              op, SIToFP(operand_value, to_ir_component_type), nullptr);
        }
        if (primitive_util::IsUnsignedIntegralType(from_type) ||
            from_type == PRED) {
          return EmitComposeComplex(
              op, UIToFP(operand_value, to_ir_component_type), nullptr);
        }
      }
      return Unimplemented("conversion from primitive type %s to %s",
                           PrimitiveType_Name(from_type),
                           PrimitiveType_Name(to_type));
    }
    case HloOpcode::kBitcastConvert: {
      PrimitiveType from_type = op->operand(0)->shape().element_type();
      PrimitiveType to_type = op->shape().element_type();
      CHECK(primitive_util::IsIntegralType(from_type));
      if (from_type == to_type) {
        return operand_value;
      }
      if (primitive_util::BitWidth(from_type) ==
          primitive_util::BitWidth(to_type)) {
        return BitCast(operand_value,
                       llvm_ir::PrimitiveTypeToIrType(to_type, module_));
      }
      return InvalidArgument(
          "bitcast conversion from primitive type %s to %s with unequal "
          "bit-widths (%u versus %u) ",
          PrimitiveType_Name(from_type), PrimitiveType_Name(to_type),
          primitive_util::BitWidth(from_type),
          primitive_util::BitWidth(to_type));
    }
    case HloOpcode::kAbs: {
      bool is_signed =
          primitive_util::IsSignedIntegralType(op->shape().element_type());
      if (is_signed) {
        auto type =
            llvm_ir::PrimitiveTypeToIrType(op->shape().element_type(), module_);
        auto cmp = ICmpSGE(operand_value, GetZero(type));
        return Select(cmp, operand_value, Neg(operand_value));
      } else {
        return operand_value;
      }
    }
    case HloOpcode::kClz: {
      auto is_zero_undef = b_->getFalse();
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::ctlz,
                                          {operand_value, is_zero_undef},
                                          {operand_value->getType()}, b_);
    }
    case HloOpcode::kSign: {
      CHECK(primitive_util::IsSignedIntegralType(op->shape().element_type()))
          << op->shape().element_type();
      auto type =
          llvm_ir::PrimitiveTypeToIrType(op->shape().element_type(), module_);
      auto cmp = ICmpEQ(operand_value, GetZero(type));
      auto ashr = AShr(operand_value, type->getIntegerBitWidth() - 1);
      return Select(cmp, GetZero(type), Or(ashr, 1));
    }
    case HloOpcode::kNegate:
      return Neg(operand_value);
    case HloOpcode::kNot: {
      auto type = op->shape().element_type();
      if (type == PRED) {
        // It is not sufficient to just call CreateNot() here because a PRED
        // is represented as an i8 and the truth value is stored only in the
        // bottom bit.
        return b_->CreateZExt(Not(Trunc(operand_value, b_->getInt1Ty())),
                              llvm_ir::PrimitiveTypeToIrType(PRED, module_));
      } else if (primitive_util::IsIntegralType(type)) {
        return Not(operand_value);
      }
      return Unimplemented("unary op Not is not defined for type '%d'", type);
    }
    case HloOpcode::kPopulationCount: {
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::ctpop,
                                          {operand_value},
                                          {operand_value->getType()}, b_);
    }
    default:
      return Unimplemented("unary integer op '%s'",
                           HloOpcodeString(op->opcode()));
  }
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitFloatUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) {
  switch (op->opcode()) {
    case HloOpcode::kConvert: {
      PrimitiveType from_type = op->operand(0)->shape().element_type();
      PrimitiveType to_type = op->shape().element_type();
      CHECK(primitive_util::IsFloatingPointType(from_type)) << from_type;
      if (from_type == to_type) {
        return operand_value;
      }
      if (primitive_util::IsComplexType(to_type)) {
        PrimitiveType to_component_type =
            primitive_util::ComplexComponentType(to_type);
        if (from_type == to_component_type) {
          return EmitComposeComplex(op, operand_value, nullptr);
        }
        return EmitComposeComplex(
            op,
            FPCast(operand_value,
                   llvm_ir::PrimitiveTypeToIrType(to_component_type, module_)),
            nullptr);
      }
      if (from_type == BF16) {
        TF_RET_CHECK(to_type != BF16);
        operand_value = EmitBF16ToF32(operand_value, b_);
        from_type = F32;
        if (from_type == to_type) {
          return operand_value;
        }
      }
      if (from_type == F32 && to_type == BF16) {
        return EmitF32ToBF16(operand_value, b_);
      }
      if ((from_type == F16 || from_type == F64) && to_type == BF16) {
        // Cast to F32 first. Other floating point formats are not supported by
        // EmitReducePrecisionIR.
        llvm::Value* f32_value = b_->CreateFPCast(
            operand_value, llvm_ir::PrimitiveTypeToIrType(F32, module_));
        return EmitF32ToBF16(f32_value, b_);
      }
      if (to_type == PRED) {
        return b_->CreateZExt(
            FCmpUNE(operand_value,
                    llvm::ConstantFP::get(operand_value->getType(), 0.0)),
            llvm_ir::PrimitiveTypeToIrType(PRED, module_));
      }
      if (primitive_util::IsFloatingPointType(to_type)) {
        return FPCast(operand_value,
                      llvm_ir::PrimitiveTypeToIrType(to_type, module_));
      }
      if (primitive_util::IsSignedIntegralType(to_type)) {
        return FPToSI(operand_value,
                      llvm_ir::PrimitiveTypeToIrType(to_type, module_));
      }
      if (primitive_util::IsUnsignedIntegralType(to_type)) {
        return FPToUI(operand_value,
                      llvm_ir::PrimitiveTypeToIrType(to_type, module_));
      }
      return Unimplemented("unhandled conversion operation: %s => %s",
                           PrimitiveType_Name(from_type),
                           PrimitiveType_Name(to_type));
    }
    case HloOpcode::kBitcastConvert: {
      PrimitiveType from_type = op->operand(0)->shape().element_type();
      PrimitiveType to_type = op->shape().element_type();
      CHECK(primitive_util::IsFloatingPointType(from_type));
      if (from_type == to_type) {
        return operand_value;
      }
      if (primitive_util::BitWidth(from_type) ==
          primitive_util::BitWidth(to_type)) {
        return BitCast(operand_value,
                       llvm_ir::PrimitiveTypeToIrType(to_type, module_));
      }
      return InvalidArgument(
          "bitcast conversion from primitive type %s to %s with unequal "
          "bit-widths (%u versus %u) ",
          PrimitiveType_Name(from_type), PrimitiveType_Name(to_type),
          primitive_util::BitWidth(from_type),
          primitive_util::BitWidth(to_type));
    }
    case HloOpcode::kExp:
      return EmitExp(op->shape().element_type(), operand_value);
    case HloOpcode::kExpm1:
      return EmitExpm1(op->shape().element_type(), operand_value);
    case HloOpcode::kLog:
      return EmitLog(op->shape().element_type(), operand_value);
    case HloOpcode::kLog1p:
      return EmitLog1p(op->shape().element_type(), operand_value);
    case HloOpcode::kCos:
      return EmitCos(op->shape().element_type(), operand_value);
    case HloOpcode::kSin:
      return EmitSin(op->shape().element_type(), operand_value);
    case HloOpcode::kTanh:
      return EmitTanh(op->shape().element_type(), operand_value);
    case HloOpcode::kSqrt:
      return EmitSqrt(op->shape().element_type(), operand_value);
    case HloOpcode::kRsqrt:
      return EmitRsqrt(op->shape().element_type(), operand_value);
    case HloOpcode::kFloor:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::floor,
                                          {operand_value},
                                          {operand_value->getType()}, b_);
    case HloOpcode::kCeil:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::ceil,
                                          {operand_value},
                                          {operand_value->getType()}, b_);
    case HloOpcode::kAbs:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs,
                                          {operand_value},
                                          {operand_value->getType()}, b_);
    case HloOpcode::kRoundNearestAfz:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::round,
                                          {operand_value},
                                          {operand_value->getType()}, b_);
    case HloOpcode::kSign: {
      auto type = operand_value->getType();
      auto zero = llvm::ConstantFP::get(type, 0.0);
      auto ne0_i1 = FCmpONE(operand_value, zero);
      auto ne0_float = UIToFP(ne0_i1, type);
      llvm::Value* result = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::copysign, {ne0_float, operand_value},
          {operand_value->getType()}, b_);
      auto is_nan = FCmpUNO(operand_value, operand_value);
      result = Select(is_nan, operand_value, result);
      return result;
    }
    case HloOpcode::kIsFinite: {
      // abs(x) o!= inf, this works because the comparison returns false if
      // either operand is NaN.
      auto type = operand_value->getType();
      auto abs_value = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::fabs, {operand_value}, {type}, b_);
      auto infinity = llvm::ConstantFP::getInfinity(type);
      auto not_infinite = FCmpONE(abs_value, infinity);
      return b_->CreateZExt(not_infinite,
                            llvm_ir::PrimitiveTypeToIrType(PRED, module_));
    }
    case HloOpcode::kNegate:
      return FNeg(operand_value);
    case HloOpcode::kReal:
      return operand_value;
    case HloOpcode::kImag:
      return llvm::ConstantFP::get(operand_value->getType(), 0.0);
    default:
      return Unimplemented("unary floating-point op '%s'",
                           HloOpcodeString(op->opcode()));
  }
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) {
  PrimitiveType input_type = op->operand(0)->shape().element_type();
  PrimitiveType component_type =
      primitive_util::IsComplexType(input_type)
          ? primitive_util::ComplexComponentType(input_type)
          : input_type;
  switch (op->opcode()) {
    case HloOpcode::kLog: {
      // log(a+bi) = log(abs(a+bi)) + i*atan2(b,a)
      auto a = EmitExtractReal(operand_value);
      auto b = EmitExtractImag(operand_value);
      TF_ASSIGN_OR_RETURN(llvm::Value * angle, EmitAtan2(component_type, b, a));
      TF_ASSIGN_OR_RETURN(llvm::Value * abs,
                          EmitComplexAbs(component_type, operand_value));
      TF_ASSIGN_OR_RETURN(llvm::Value * log_abs, EmitLog(component_type, abs));
      return EmitComposeComplex(op, log_abs, angle);
    }
    case HloOpcode::kLog1p: {
      // log1p(a+bi) = .5*log((a+1)^2+b^2) + i*atan2(b, a + 1)
      auto a = EmitExtractReal(operand_value);
      auto b = EmitExtractImag(operand_value);
      llvm::Type* llvm_ty = a->getType();
      auto one = llvm::ConstantFP::get(llvm_ty, 1.0);
      auto a_plus_one = FAdd(a, one);
      auto sum_sq = FAdd(FMul(a_plus_one, a_plus_one), FMul(b, b));
      TF_ASSIGN_OR_RETURN(auto log_sum_sq, EmitLog(component_type, sum_sq));
      TF_ASSIGN_OR_RETURN(auto angle, EmitAtan2(component_type, b, a_plus_one));
      auto one_half = llvm::ConstantFP::get(llvm_ty, 0.5);
      return EmitComposeComplex(op, FMul(one_half, log_sum_sq), angle);
    }
    case HloOpcode::kConvert: {
      PrimitiveType from_type = op->operand(0)->shape().element_type();
      TF_RET_CHECK(primitive_util::IsComplexType(from_type));
      PrimitiveType to_type = op->shape().element_type();
      TF_RET_CHECK(primitive_util::IsComplexType(to_type));
      if (from_type == to_type) {
        return operand_value;
      }
      PrimitiveType to_component_type =
          primitive_util::ComplexComponentType(to_type);
      auto to_ir_component_type =
          llvm_ir::PrimitiveTypeToIrType(to_component_type, module_);
      return EmitComposeComplex(
          op, FPCast(EmitExtractReal(operand_value), to_ir_component_type),
          FPCast(EmitExtractImag(operand_value), to_ir_component_type));
    }
    case HloOpcode::kExp: {
      // e^(a+bi) = e^a*(cos(b)+sin(b)i)
      TF_ASSIGN_OR_RETURN(
          auto exp_a, EmitExp(component_type, EmitExtractReal(operand_value)));
      TF_ASSIGN_OR_RETURN(
          auto cos_b, EmitCos(component_type, EmitExtractImag(operand_value)));
      TF_ASSIGN_OR_RETURN(
          auto sin_b, EmitSin(component_type, EmitExtractImag(operand_value)));
      return EmitComposeComplex(op, FMul(exp_a, cos_b), FMul(exp_a, sin_b));
    }
    case HloOpcode::kExpm1: {
      // e^(a+bi)-1 = (e^a*cos(b)-1)+e^a*sin(b)i
      TF_ASSIGN_OR_RETURN(
          auto exp_a, EmitExp(component_type, EmitExtractReal(operand_value)));
      TF_ASSIGN_OR_RETURN(
          auto cos_b, EmitCos(component_type, EmitExtractImag(operand_value)));
      TF_ASSIGN_OR_RETURN(
          auto sin_b, EmitSin(component_type, EmitExtractImag(operand_value)));
      auto one = llvm::ConstantFP::get(exp_a->getType(), 1.0);
      auto real_result = FSub(FMul(exp_a, cos_b), one);
      auto imag_result = FMul(exp_a, sin_b);
      return EmitComposeComplex(op, real_result, imag_result);
    }
    case HloOpcode::kCos: {
      // cos(z) = .5(e^(iz) + e^(-iz))
      // cos(a+bi) = .5(e^(-b+ai) + e^(b-ai))
      // now, e^(x+yi) = e^x*(cos(y)+sin(y)i), so we have
      // cos(a+bi) = .5(e^-b*(cos(a)+sin(a)i) + e^b*(cos(-a)+sin(-a)i))
      // cos(-x) = cos(x) and sin(-x) = -sin(x), so
      // cos(a+bi) = .5(e^-b*(cos(a)+sin(a)i) + e^b*(cos(a)-sin(a)i))
      //           = .5(cos(a)*(e^-b+e^b) + i*sin(a)*(e^-b-e^b))
      auto a = EmitExtractReal(operand_value);
      auto b = EmitExtractImag(operand_value);
      auto type = a->getType();
      TF_ASSIGN_OR_RETURN(auto exp_b, EmitExp(component_type, b));
      auto half_exp_b = FMul(llvm::ConstantFP::get(type, 0.5), exp_b);
      auto half_exp_neg_b = FDiv(llvm::ConstantFP::get(type, 0.5), exp_b);
      TF_ASSIGN_OR_RETURN(auto cos_a, EmitCos(component_type, a));
      TF_ASSIGN_OR_RETURN(auto sin_a, EmitSin(component_type, a));
      return EmitComposeComplex(op,
                                FMul(cos_a, FAdd(half_exp_neg_b, half_exp_b)),
                                FMul(sin_a, FSub(half_exp_neg_b, half_exp_b)));
    }
    case HloOpcode::kSin: {
      // sin(z) = .5i(e^(-iz) - e^(iz))
      // sin(a+bi) = .5i(e^(-i(a+bi)) - e^(i(a+bi)))
      //           = .5i(e^(b-ai) - e^(-b+ai))
      // now, e^(x+yi) = e^x*(cos(y)+sin(y)i), so we have
      // sin(a+bi) = 0.5i(e^b*(cos(-a)+sin(-a)i) - e^-b*(cos(a)+sin(a)i))
      //           = 0.5(e^b*(cos(-a)i-sin(-a)) - e^-b*(cos(a)i-sin(a)))
      // cos(-x) = cos(x) and sin(-x) = -sin(x), so
      //           = 0.5(e^b*(cos(a)i+sin(a)) - e^-b*(cos(a)i-sin(a)))
      //           = 0.5(sin(a)*(e^b+e^-b) + i*cos(a)*(e^b-e^-b)
      auto a = EmitExtractReal(operand_value);
      auto b = EmitExtractImag(operand_value);
      auto type = a->getType();
      TF_ASSIGN_OR_RETURN(auto exp_b, EmitExp(component_type, b));
      auto half_exp_b = FMul(llvm::ConstantFP::get(type, 0.5), exp_b);
      auto half_exp_neg_b = FDiv(llvm::ConstantFP::get(type, 0.5), exp_b);
      TF_ASSIGN_OR_RETURN(auto cos_a, EmitCos(component_type, a));
      TF_ASSIGN_OR_RETURN(auto sin_a, EmitSin(component_type, a));
      return EmitComposeComplex(op,
                                FMul(sin_a, FAdd(half_exp_b, half_exp_neg_b)),
                                FMul(cos_a, FSub(half_exp_b, half_exp_neg_b)));
    }
    case HloOpcode::kTanh: {
      /*
      tanh=(exp(x)-exp(-x)) / (exp(x)+exp(-x))
      e^(a+bi) = e^a*(cos(b)+sin(b)i)
      so tanh=(((cos(b)+sin(b)i)e^a - (cos(-b)+sin(-b)i)e^-a)) /
              (((cos(b)+sin(b)i)e^a + (cos(-b)+sin(-b)i)e^-a))
      cos(b)=cos(-b), sin(-b)=-sin(b)
      so tanh=(((cos(b)+sin(b)i)e^a - (cos(b)-sin(b)i)e^-a)) /
              (((cos(b)+sin(b)i)e^a + (cos(b)-sin(b)i)e^-a))
             =(cos(b)e^a+i*sin(b)e^a + cos(b)(-e^-a)+i*sin(b)e^-a) /
              (cos(b)e^a+i*sin(b)e^a + cos(b)e^-a+i*sin(b)(-e^-a))
             =(cos(b)(e^a-e^-a) + i*sin(b)(e^a+e^-a)) /
              (cos(b)(e^a+e^-a) + i*sin(b)(e^a-e^-a))
      This is a complex division, so we can multiply by denom_conj/denom_conj
             =(cos(b)(e^a-e^-a) + i*sin(b)(e^a+e^-a)) *
              (cos(b)(e^a+e^-a) - i*sin(b)(e^a-e^-a)) /
              ((cos(b)(e^a+e^-a))^2 + (sin(b)(e^a-e^-a))^2)
             =(cos(b)^2(e^(2a)-e^(-2a)) + sin(b)^2(e^(2a)-e^(-2a)) +
               i*(cos(b)sin(b)(e^a+e^-a)^2 - cos(b)sin(b)(e^a-e^-a)^2)) /
              ((cos(b)(e^a+e^-a))^2 + (sin(b)(e^a-e^-a))^2)
             =(e^(2a)-e^(-2a) +
               i*[cos(b)sin(b)(e^(2a)+2+e^(-2a))-cos(b)sin(b)(e^(2a)-2+e^(2a)))]
               / (cos(b)^2*(e^(2a)+2+e^(-2a)) + sin(b)^2*(e^(2a)-2+e^(2a))
             =(e^(2a)-e^(-2a) +
               i*cos(b)sin(b)*[e^(2a)+2+e^(-2a)-e^(2a)+2-e^(-2a)]) /
               ([cos(b)^2 + sin(b)^2][e^(2a)+e^(-2a)])+2*[cos(b)^2 - sin(b)^2])
             =(e^(2a)-e^(-2a) + i*cos(b)sin(b)*4) /
              (e^(2a)+e^(-2a)+2*[cos(b)^2 - sin(b)^2])
             =(e^(2a)-e^(-2a) + i*[sin(2b)/2]*4) /
              (e^(2a)+e^(-2a)+2*[cos(2b)])
             =(e^(2a)-e^(-2a) + i*2*sin(2b)) / (e^(2a) + e^(-2a) + 2*cos(2b))
      */
      llvm::Value* a = EmitExtractReal(operand_value);
      llvm::Value* b = EmitExtractImag(operand_value);

      llvm::Type* type = a->getType();

      llvm::Value* neg_one = llvm::ConstantFP::get(type, -1.F);
      llvm::Value* two_a = FAdd(a, a);
      llvm::Value* neg_2a = FMul(neg_one, two_a);

      // When we are calculating the real numerator, e^(2a)-e^(-2a), for small
      // values of `a`, we will get a ULP of 2^-23 using the exp function. Using
      // expm1 to calculate e^(2a)-e^(-2a) = [e^(2a)-1] - [e^(-2a)-1] allows our
      // ULP to be arbitrarily small. For larger values of `a`, calculating the
      // numerator as Exp(2a)-Exp(-2a) vs Expm1(2a)-Expm1(-2a) return virtually
      // identical results.
      TF_ASSIGN_OR_RETURN(llvm::Value * exp_2a_m1,
                          EmitExpm1(component_type, two_a));
      TF_ASSIGN_OR_RETURN(llvm::Value * exp_neg_2a_m1,
                          EmitExpm1(component_type, neg_2a));
      llvm::Value* real_numerator = FSub(exp_2a_m1, exp_neg_2a_m1);

      // We can use the identity cos(2b)+1 = cos(b)^2-sin(b)^2+cos(b)^2+sin(b)^2
      // = 2cos(b)^2. This gives us the ability to be more precise when the
      // denominator is close to zero.
      TF_ASSIGN_OR_RETURN(llvm::Value * cos_b, EmitCos(component_type, b));
      llvm::Value* four = llvm::ConstantFP::get(type, 4.F);
      llvm::Value* cos_b_sq = FMul(cos_b, cos_b);
      llvm::Value* two_cos_2b_p2 = FMul(cos_b_sq, four);

      // Similarly we can compute sin(2b) with the formula sin(2b) =
      // 2*sin(b)*cos(b).
      TF_ASSIGN_OR_RETURN(llvm::Value * sin_b, EmitSin(component_type, b));
      llvm::Value* imag_numerator = FMul(four, FMul(cos_b, sin_b));

      // Expm1(x) is about x for small values of x, but exp_sum_m2 is about x^2
      // for small value of x. As a result, due to floating point precission
      // issues, x^2 is a better approximation than Expm1(x) + Expm1(x) for
      // small values of x.
      llvm::Value* a_sqr = FMul(a, a);
      llvm::Value* use_approx_cutoff = llvm::ConstantFP::get(type, 1e-8);
      llvm::Value* use_approx = FCmpOLT(a_sqr, use_approx_cutoff);

      llvm::Value* exp_sum_m2 =
          Select(use_approx, a_sqr, FAdd(exp_2a_m1, exp_neg_2a_m1));
      llvm::Value* denom = FAdd(exp_sum_m2, two_cos_2b_p2);

      // As `a` grows toward +inf and -inf, the real numerator will grow towards
      // +inf and -inf respectively, while the denominator will always grow
      // towards +inf. The result is real_numerator/denom = NaN, when it should
      // equal +1 and -1 respectively. Therefore, if our denominator is +inf,
      // we just hardcode the limits for the real numbers.
      llvm::Value* inf = llvm::ConstantFP::getInfinity(type);
      llvm::Value* is_inf = FCmpOEQ(exp_sum_m2, inf);
      llvm::Value* real_limit = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::copysign, {neg_one, a}, {type}, b_);

      llvm::Value* real =
          Select(is_inf, real_limit, FDiv(real_numerator, denom));
      llvm::Value* imag = FDiv(imag_numerator, denom);

      // The complex tanh functions have a few corner cases:
      // 1. (+0, +0) => (+0, +0)        - Handled normally
      // 2. (x, +Inf) => (NaN, NaN)     - See below
      // 3. (x, NaN) => (NaN, NaN)      - See below
      // 4. (+inf, y) => (1, +0)        - Handled normally
      // 5. (+Inf, +Inf) => (1, +/-0)   - See below
      // 6. (+Inf, NaN) => (1, +/-0)    - See below
      // 7. (NaN, +0) => (NaN, +0)      - See below
      // 8. (NaN, y) => (NaN, NaN)      - Handled normally
      // 9. (NaN, NaN) => (NaN, NaN)    - Handled normally
      //
      // For the cases that aren't handled normally:
      // 2/3) Part of the calculation we do is that if exp(a) + exp(-a) = +inf,
      //      then we return (+/-1, +/-0). However, this is only true if we
      //      assume that a is infinity or b is finite. In the event that both a
      //      is finite and b is either +/-Inf or NaN, then our normal
      //      calculation would end up returing (+/-1, NaN), as opposed to (NaN,
      //      NaN).
      // 5/6) We always calculate the imagninary value as sin(2b)/denominator.
      //      When the denominator is infinity, this assures us that the zero is
      //      the correct sign. However if our imaginary input results in
      //      sin(2b) = NaN, we calculate our imaginary result as NaN.
      // 7)   In the event that a is NaN, the denominator will be NaN.
      //      Therefore, the normal calculation gives (NaN, NaN) while we need
      //      (NaN, +0).
      if (!(b_->getFastMathFlags().noNaNs() &&
            b_->getFastMathFlags().noInfs())) {
        llvm::Value* abs_a = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs,
                                                          {a}, {type}, b_);
        llvm::Value* zero = llvm::ConstantFP::get(type, 0.F);
        llvm::Value* nan = llvm::ConstantFP::getNaN(type);

        llvm::Value* a_is_inf = FCmpOEQ(abs_a, inf);
        llvm::Value* b_is_zero = FCmpOEQ(b, zero);

        // imag_numerator = 2sin(2b), so sin(2b) is NaN if and only if
        // imag_numerator is NaN.
        llvm::Value* sin_2b_is_nan =
            b_->CreateFCmpUNO(imag_numerator, imag_numerator);

        llvm::Value* real_is_nan =
            b_->CreateAnd(sin_2b_is_nan, b_->CreateNot(a_is_inf));
        llvm::Value* imag_is_zero =
            b_->CreateOr(b_is_zero, b_->CreateAnd(a_is_inf, sin_2b_is_nan));

        real = Select(real_is_nan, nan, real);
        imag = Select(imag_is_zero, zero, imag);
      }

      return EmitComposeComplex(op, real, imag);
    }
    case HloOpcode::kAbs: {
      return EmitComplexAbs(component_type, operand_value);
    }
    case HloOpcode::kSign: {  // Sign(c) = c / |c|
      TF_ASSIGN_OR_RETURN(auto cplx_abs,
                          EmitComplexAbs(component_type, operand_value));
      auto type = cplx_abs->getType();
      auto zero = llvm::ConstantFP::get(type, 0.0);
      auto oeq = FCmpOEQ(cplx_abs, zero);
      return Select(
          oeq, EmitComposeComplex(op, zero, zero),
          EmitComposeComplex(op, FDiv(EmitExtractReal(operand_value), cplx_abs),
                             FDiv(EmitExtractImag(operand_value), cplx_abs)));
    }
    case HloOpcode::kSqrt: {
      return EmitComplexSqrt(op, component_type, operand_value);
    }
    case HloOpcode::kRsqrt: {
      return EmitComplexRsqrt(op, component_type, operand_value);
    }
    case HloOpcode::kNegate:
      return EmitComposeComplex(op, FNeg(EmitExtractReal(operand_value)),
                                FNeg(EmitExtractImag(operand_value)));
    case HloOpcode::kReal:
      return EmitExtractReal(operand_value);
    case HloOpcode::kImag:
      return EmitExtractImag(operand_value);
    default:
      return Unimplemented("unary complex op '%s'",
                           HloOpcodeString(op->opcode()));
  }
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  PrimitiveType operand_type = op->operand(0)->shape().element_type();
  if (ShapeUtil::ElementIsIntegral(op->operand(0)->shape()) ||
      operand_type == PRED) {
    return EmitIntegerBinaryOp(
        op, lhs_value, rhs_value,
        primitive_util::IsSignedIntegralType(operand_type));
  } else if (primitive_util::IsComplexType(operand_type)) {
    return EmitComplexBinaryOp(op, lhs_value, rhs_value);
  } else {
    return EmitFloatBinaryOp(op, lhs_value, rhs_value);
  }
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitFloatBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  switch (op->opcode()) {
    case HloOpcode::kComplex:
      return EmitComposeComplex(op, lhs_value, rhs_value);
    case HloOpcode::kAdd:
      return FAdd(lhs_value, rhs_value);
    case HloOpcode::kSubtract:
      return FSub(lhs_value, rhs_value);
    case HloOpcode::kMultiply:
      return FMul(lhs_value, rhs_value);
    case HloOpcode::kDivide:
      return FDiv(lhs_value, rhs_value);
    case HloOpcode::kRemainder:
      return FRem(lhs_value, rhs_value);
    // LLVM comparisons can be "unordered" (U) or "ordered" (O) -- ordered
    // comparisons always return false when one of the operands is NaN, whereas
    // unordered comparisons return true.
    //
    // We use ordered comparisons for everything except kNe, where we use an
    // unordered comparison.  This makes x != y equivalent to !(x == y), and
    // matches C++'s semantics.
    case HloOpcode::kCompare: {
      switch (op->comparison_direction()) {
        case ComparisonDirection::kEq:
          return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OEQ, lhs_value,
                                         rhs_value, b_);
        case ComparisonDirection::kNe:
          return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_UNE, lhs_value,
                                         rhs_value, b_);
        case ComparisonDirection::kLt:
          return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OLT, lhs_value,
                                         rhs_value, b_);
        case ComparisonDirection::kGt:
          return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OGT, lhs_value,
                                         rhs_value, b_);
        case ComparisonDirection::kLe:
          return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OLE, lhs_value,
                                         rhs_value, b_);
        case ComparisonDirection::kGe:
          return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OGE, lhs_value,
                                         rhs_value, b_);
      }
    }
    case HloOpcode::kMaximum:
      return EmitFloatMax(lhs_value, rhs_value);
    case HloOpcode::kMinimum:
      return EmitFloatMin(lhs_value, rhs_value);
    case HloOpcode::kPower:
      return EmitPow(op->shape().element_type(), lhs_value, rhs_value);
    case HloOpcode::kAtan2:
      return EmitAtan2(op->shape().element_type(), lhs_value, rhs_value);
    default:
      return Unimplemented("binary floating point op '%s'",
                           HloOpcodeString(op->opcode()));
  }
}

// Using sqrt(a^2 + b^2) can cause overflow errors. Therefore we can use
// sqrt(a^2 + b^2) = sqrt(a^2 * (1 + b^2/a^2))
//                 = |a| * sqrt(1 + (b/a)^2)
// With the assumption that |a| >= |b|.
//
// This method returns the min, max, and sqrt term for this calculation. This is
// done to prevent potential overflow errors that can occur from multiplying the
// max with the sqrt term. (i.e. when calculating the sqrt of the absolute
// value, we can take the sqrt of the max and the sqrt term before multiplying
// them together.) If return_sqrt is false, it returns 1 + (b/a)^2 instead of
// sqrt(1 + (b/a)^2).
StatusOr<std::tuple<llvm::Value*, llvm::Value*, llvm::Value*>>
ElementalIrEmitter::EmitComplexAbsHelper(PrimitiveType prim_type,
                                         llvm::Value* operand_value,
                                         bool return_sqrt) {
  llvm::Value* real = EmitExtractReal(operand_value);
  llvm::Value* imag = EmitExtractImag(operand_value);
  llvm::Value* abs_real = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::fabs, {real}, {real->getType()}, b_);
  llvm::Value* abs_imag = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::fabs, {imag}, {imag->getType()}, b_);
  llvm::Value* max = EmitFloatMax(abs_real, abs_imag);
  llvm::Value* min = EmitFloatMin(abs_real, abs_imag);

  llvm::Value* div = FDiv(min, max);
  llvm::Value* div_sq = FMul(div, div);
  llvm::Value* one = llvm::ConstantFP::get(max->getType(), 1);
  llvm::Value* one_p_div_sq = FAdd(one, div_sq);
  TF_ASSIGN_OR_RETURN(llvm::Value * sqrt, EmitSqrt(prim_type, one_p_div_sq));
  return std::make_tuple(min, max, return_sqrt ? sqrt : one_p_div_sq);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexAbs(
    PrimitiveType prim_type, llvm::Value* operand_value) {
  llvm::Value* min;
  llvm::Value* max;
  llvm::Value* sqrt;
  TF_ASSIGN_OR_RETURN(
      std::tie(min, max, sqrt),
      EmitComplexAbsHelper(prim_type, operand_value, /*return_sqrt=*/true));
  llvm::Value* result = FMul(max, sqrt);
  // When (min, max) are (0, 0), (inf, inf), or (NaN, ...), `result` is NaN.
  // In such cases, we return `min` instead of `result`.
  return Select(FCmpUNO(result, result), min, result);
}

// Calculates ComplexAbs in the same way, except using:
// sqrt(|a| * sqrt(1 + (b/a)^2)) = sqrt(|a|) * pow(1 + (b/a)^2, .25)
StatusOr<llvm::Value*> ElementalIrEmitter::EmitSqrtComplexAbs(
    PrimitiveType prim_type, llvm::Value* operand_value) {
  llvm::Value* min;
  llvm::Value* max;
  llvm::Value* one_p_div_sq;
  TF_ASSIGN_OR_RETURN(
      std::tie(min, max, one_p_div_sq),
      EmitComplexAbsHelper(prim_type, operand_value, /*return_sqrt=*/false));
  TF_ASSIGN_OR_RETURN(llvm::Value * sqrt_max, EmitSqrt(prim_type, max));
  TF_ASSIGN_OR_RETURN(llvm::Value * pow,
                      EmitPow(prim_type, one_p_div_sq,
                              llvm::ConstantFP::get(max->getType(), .25)));
  llvm::Value* result = FMul(sqrt_max, pow);
  // When (min, max) are (0, 0), (inf, inf), or (NaN, ...), `result` is NaN.
  // In such cases, we return `min` instead of `result`.
  return Select(FCmpUNO(result, result), min, result);
}

// Calculates ComplexAbs in the same way, except using:
// rsqrt(|a| * sqrt(1 + (b/a)^2)) = rsqrt(|a|) * rsqrt(sqrt(1 + (b/a)^2))
StatusOr<llvm::Value*> ElementalIrEmitter::EmitRsqrtComplexAbs(
    PrimitiveType prim_type, llvm::Value* operand_value) {
  llvm::Value* min;
  llvm::Value* max;
  llvm::Value* sqrt;
  TF_ASSIGN_OR_RETURN(
      std::tie(min, max, sqrt),
      EmitComplexAbsHelper(prim_type, operand_value, /*return_sqrt=*/true));
  TF_ASSIGN_OR_RETURN(llvm::Value * rsqrt_max, EmitRsqrt(prim_type, max));
  TF_ASSIGN_OR_RETURN(llvm::Value * rsqrt_sqrt, EmitRsqrt(prim_type, sqrt));
  llvm::Value* result = FMul(rsqrt_max, rsqrt_sqrt);
  TF_ASSIGN_OR_RETURN(llvm::Value * rsqrt_min, EmitRsqrt(prim_type, min));
  // When (min, max) are (0, 0), (inf, inf), or (NaN, ...), `result` is NaN.
  // In such cases, we return rsqrt(min) instead of `result`.
  return Select(FCmpUNO(result, result), rsqrt_min, result);
}

// Using our EmitComplexPower formula, but setting c=0.5 and d=0, we get:
//   e^[ln(r)*c - t*d] * [cos(ln(r)*d + t*c) + i*sin(ln(r)*d + t*c)]
// = e^[ln(r)*0.5] * [cos(t*0.5) + i*sin(t*0.5)]
// = r^0.5 * [cos(t/2) + i*sin(t/2)]
// = sqrt(r) * [cos(t/2) + i*sin(t/2)]
// where r = |a+bi| and t = atan2(b,a)
// TODO(bixia): See doc for implementation without atan2.
StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexSqrt(
    const HloInstruction* op, PrimitiveType prim_type,
    llvm::Value* operand_value) {
  llvm::Type* type = static_cast<llvm::StructType*>(operand_value->getType())
                         ->getElementType(0);

  TF_ASSIGN_OR_RETURN(llvm::Value * r,
                      EmitSqrtComplexAbs(prim_type, operand_value));

  llvm::Value* a = EmitExtractReal(operand_value);
  llvm::Value* b = EmitExtractImag(operand_value);
  TF_ASSIGN_OR_RETURN(llvm::Value * t, EmitAtan2(prim_type, b, a));

  llvm::Value* c = llvm::ConstantFP::get(type, 0.5);
  llvm::Value* angle = FMul(t, c);
  TF_ASSIGN_OR_RETURN(llvm::Value * cos, EmitCos(prim_type, angle));
  TF_ASSIGN_OR_RETURN(llvm::Value * sin, EmitSin(prim_type, angle));

  llvm::Value* real_part;
  llvm::Value* imag_part;

  llvm::Value* zero = llvm::ConstantFP::get(type, 0);

  if (!(b_->getFastMathFlags().noNaNs() && b_->getFastMathFlags().noInfs())) {
    llvm::Value* inf = llvm::ConstantFP::getInfinity(type);
    llvm::Value* neg_inf = llvm::ConstantFP::getInfinity(type, true);
    llvm::Value* nan = llvm::ConstantFP::getNaN(type);
    llvm::Value* abs_b = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs,
                                                      {b}, {b->getType()}, b_);

    real_part = Select(Or(FCmpOEQ(abs_b, inf), FCmpOEQ(a, inf)), inf,
                       Select(And(FCmpOEQ(a, neg_inf), FCmpONE(abs_b, inf)),
                              zero, FMul(r, cos)));

    llvm::Value* b_signed_inf = llvm_ir::EmitCallToIntrinsic(
        llvm::Intrinsic::copysign, {inf, b}, {b->getType()}, b_);
    imag_part =
        Select(Or(FCmpOEQ(abs_b, inf), FCmpOEQ(a, neg_inf)), b_signed_inf,
               Select(FCmpUNO(r, r), nan,
                      Select(FCmpOEQ(sin, zero), sin, FMul(r, sin))));
  } else {
    real_part = FMul(r, cos);
    imag_part = Select(FCmpOEQ(sin, zero), sin, FMul(r, sin));
  }

  return Select(FCmpOEQ(r, zero), EmitComposeComplex(op, zero, zero),
                EmitComposeComplex(op, real_part, imag_part));
}

// Similar to Sqrt, we can use our EmitComplexPower formula, but set
// c=-0.5 and d=0. We get:
//   e^[ln(r)*c - t*d] * [cos(ln(r)*d + t*c) + i*sin(ln(r)*d + t*c)]
// = e^[ln(r)*-0.5] * [cos(t*-0.5) + i*sin(t*-0.5)]
// = r^(-0.5) * [cos(-t/2) + i*sin(-t/2)]
// = rsqrt(r) * [cos(-t/2) + i*sin(-t/2)]
// where r = |a+bi| and t = atan2(b,a).
StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexRsqrt(
    const HloInstruction* op, PrimitiveType prim_type,
    llvm::Value* operand_value) {
  llvm::Type* type = static_cast<llvm::StructType*>(operand_value->getType())
                         ->getElementType(0);

  TF_ASSIGN_OR_RETURN(llvm::Value * r,
                      EmitRsqrtComplexAbs(prim_type, operand_value));

  llvm::Value* a = EmitExtractReal(operand_value);
  llvm::Value* b = EmitExtractImag(operand_value);
  TF_ASSIGN_OR_RETURN(llvm::Value * t, EmitAtan2(prim_type, b, a));

  llvm::Value* c = llvm::ConstantFP::get(type, -0.5);
  llvm::Value* angle = FMul(t, c);
  TF_ASSIGN_OR_RETURN(llvm::Value * cos, EmitCos(prim_type, angle));
  TF_ASSIGN_OR_RETURN(llvm::Value * sin, EmitSin(prim_type, angle));

  llvm::Value* real_part = FMul(r, cos);
  llvm::Value* imag_part = FMul(r, sin);

  if (!(b_->getFastMathFlags().noNaNs() && b_->getFastMathFlags().noInfs())) {
    llvm::Value* zero = llvm::ConstantFP::get(type, 0);
    llvm::Value* neg_one = llvm::ConstantFP::get(type, -1);
    llvm::Value* inf = llvm::ConstantFP::getInfinity(type);
    llvm::Value* nan = llvm::ConstantFP::getNaN(type);
    // llvm::Value* neg_inf = llvm::ConstantFP::getInfinity(type, true);
    llvm::Value* a_signed_zero = llvm_ir::EmitCallToIntrinsic(
        llvm::Intrinsic::copysign, {zero, a}, {a->getType()}, b_);
    llvm::Value* b_signed_zero = llvm_ir::EmitCallToIntrinsic(
        llvm::Intrinsic::copysign, {zero, b}, {b->getType()}, b_);
    llvm::Value* neg_b_signed_zero = FMul(b_signed_zero, neg_one);

    llvm::Value* abs_a = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs,
                                                      {a}, {a->getType()}, b_);
    llvm::Value* abs_b = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs,
                                                      {b}, {b->getType()}, b_);

    llvm::Value* is_zero_zero = And(FCmpOEQ(b, zero), FCmpOEQ(a, zero));
    real_part = Select(
        is_zero_zero, inf,
        Select(Or(And(FCmpOEQ(abs_b, inf), FCmpUNO(a, a)), FCmpOEQ(abs_a, inf)),
               a_signed_zero, FMul(r, cos)));
    imag_part = Select(
        is_zero_zero, nan,
        Select(Or(And(FCmpOEQ(abs_b, inf), FCmpUNO(a, a)), FCmpOEQ(abs_a, inf)),
               neg_b_signed_zero, FMul(r, sin)));
  } else {
    llvm::Value* zero = llvm::ConstantFP::get(type, 0);
    llvm::Value* inf = llvm::ConstantFP::getInfinity(type);
    llvm::Value* nan = llvm::ConstantFP::getNaN(type);

    llvm::Value* is_zero_zero = And(FCmpOEQ(b, zero), FCmpOEQ(a, zero));
    real_part = Select(is_zero_zero, inf, FMul(r, cos));
    imag_part = Select(is_zero_zero, nan, FMul(r, sin));
  }

  return EmitComposeComplex(op, real_part, imag_part);
}

// (a+bi)^(c+di) =
//    (a*a+b*b)^(0.5c) * exp(-d*atan2(b,a)) * (cos(q) + i*sin(q)),
//    where q = c*atan2(b,a)+0.5d*ln(a*a+b*b)
StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexPower(
    const HloInstruction* op, llvm::Value* a, llvm::Value* b, llvm::Value* c,
    llvm::Value* d) {
  PrimitiveType component_type =
      primitive_util::ComplexComponentType(op->shape().element_type());
  auto aa_p_bb = FAdd(FMul(a, a), FMul(b, b));
  auto zero = llvm::ConstantFP::get(a->getType(), 0);
  auto one_half = llvm::ConstantFP::get(a->getType(), 0.5);
  auto one = llvm::ConstantFP::get(a->getType(), 1);
  auto half_c = FMul(one_half, c);

  TF_ASSIGN_OR_RETURN(auto aa_p_bb_to_half_c,
                      EmitPow(component_type, aa_p_bb, half_c));

  auto neg_d = FNeg(d);
  TF_ASSIGN_OR_RETURN(auto arg_lhs, EmitAtan2(component_type, b, a));
  auto neg_d_arg_lhs = FMul(neg_d, arg_lhs);
  TF_ASSIGN_OR_RETURN(auto e_to_neg_d_arg_lhs,
                      EmitExp(component_type, neg_d_arg_lhs));
  auto coeff = FMul(aa_p_bb_to_half_c, e_to_neg_d_arg_lhs);
  TF_ASSIGN_OR_RETURN(auto ln_aa_p_bb, EmitLog(component_type, aa_p_bb));
  auto half_d = FMul(one_half, d);
  auto q = FAdd(FMul(c, arg_lhs), FMul(half_d, ln_aa_p_bb));
  TF_ASSIGN_OR_RETURN(auto cos_q, EmitCos(component_type, q));
  TF_ASSIGN_OR_RETURN(auto sin_q, EmitSin(component_type, q));
  // d^c is 0 if d is 0 and c > 0. 0^0 is defined to be 1.0, see
  // Branch Cuts for Complex Elementary Functions or Much Ado About
  // Nothing's Sign Bit, W. Kahan, Section 10.
  return Select(
      And(And(FCmpOEQ(aa_p_bb, zero), FCmpOEQ(d, zero)), FCmpOLE(zero, c)),
      EmitComposeComplex(op, Select(FCmpOEQ(zero, c), one, zero), zero),
      EmitComposeComplex(op, FMul(coeff, cos_q), FMul(coeff, sin_q)));
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  switch (op->opcode()) {
    case HloOpcode::kAdd:
      return EmitComposeComplex(
          op, FAdd(EmitExtractReal(lhs_value), EmitExtractReal(rhs_value)),
          FAdd(EmitExtractImag(lhs_value), EmitExtractImag(rhs_value)));
    case HloOpcode::kSubtract:
      return EmitComposeComplex(
          op, FSub(EmitExtractReal(lhs_value), EmitExtractReal(rhs_value)),
          FSub(EmitExtractImag(lhs_value), EmitExtractImag(rhs_value)));
    case HloOpcode::kMultiply:
      return EmitComposeComplex(
          op,
          FSub(FMul(EmitExtractReal(lhs_value), EmitExtractReal(rhs_value)),
               FMul(EmitExtractImag(lhs_value), EmitExtractImag(rhs_value))),
          FAdd(FMul(EmitExtractReal(lhs_value), EmitExtractImag(rhs_value)),
               FMul(EmitExtractImag(lhs_value), EmitExtractReal(rhs_value))));
    case HloOpcode::kDivide: {
      // Division of complex numbers is implemented here, taking into account
      // over/underflow, NaN and Inf values.
      auto a_r = EmitExtractReal(lhs_value);
      auto a_i = EmitExtractImag(lhs_value);
      auto b_r = EmitExtractReal(rhs_value);
      auto b_i = EmitExtractImag(rhs_value);
      auto type = a_r->getType();

      // Smith's algorithm to divide complex numbers. It is just a bit smarter
      // way to compute the following formula:
      //  (a_r + a_i * i) / (b_r + b_i * i)
      //    = (a_r + a_i * i) (b_r - b_i * i) / ((b_r + b_i * i)(b_r - b_i * i))
      //    = ((a_r * b_r + a_i * b_i) + (a_i * b_r - a_r * b_i) * i) / ||b||^2
      //
      // Depending on whether |b_r| < |b_i| we compute either
      //   b_r_b_i_ratio = b_r / b_i
      //   b_r_b_i_denom = b_i + b_r * b_r_b_i_ratio
      //   c_r = (a_r * b_r_b_i_ratio + a_i ) / b_r_b_i_denom
      //   c_i = (a_i * b_r_b_i_ratio - a_r ) / b_r_b_i_denom
      //
      // or
      //
      //   b_i_b_r_ratio = b_i / b_r
      //   b_i_b_r_denom = b_r + b_i * b_i_b_r_denom
      //   c_r = (a_r + a_i * b_i_b_r_ratio ) / b_i_b_r_denom
      //   c_i = (a_i - a_r * b_i_b_r_ratio ) / b_i_b_r_denom
      //
      // See https://dl.acm.org/citation.cfm?id=368661 for more details.
      auto b_r_b_i_ratio = FDiv(b_r, b_i);
      auto b_r_b_i_denom = FAdd(b_i, FMul(b_r_b_i_ratio, b_r));
      auto b_i_b_r_ratio = FDiv(b_i, b_r);
      auto b_i_b_r_denom = FAdd(b_r, FMul(b_i_b_r_ratio, b_i));

      auto b_r_lt_b_i = FCmpOLT(llvm_ir::EmitCallToIntrinsic(
                                    llvm::Intrinsic::fabs, {b_r}, {type}, b_),
                                llvm_ir::EmitCallToIntrinsic(
                                    llvm::Intrinsic::fabs, {b_i}, {type}, b_));
      auto c_r = Select(
          b_r_lt_b_i, FDiv(FAdd(FMul(b_r_b_i_ratio, a_r), a_i), b_r_b_i_denom),
          FDiv(FAdd(FMul(b_i_b_r_ratio, a_i), a_r), b_i_b_r_denom));
      auto c_i = Select(
          b_r_lt_b_i, FDiv(FSub(FMul(b_r_b_i_ratio, a_i), a_r), b_r_b_i_denom),
          FDiv(FSub(a_i, FMul(b_i_b_r_ratio, a_r)), b_i_b_r_denom));
      auto result = EmitComposeComplex(op, c_r, c_i);

      // Consider corner cases, if the result is (NaN, NaN).
      auto zero = llvm::ConstantFP::get(type, 0.0);
      auto one = llvm::ConstantFP::get(type, 1.0);
      auto inf = llvm::ConstantFP::getInfinity(type);

      // Case 1. Zero denominator.
      auto zero_denominator =
          And(And(FCmpOEQ(b_r, zero), FCmpOEQ(b_i, zero)),
              Or(Neg(FCmpONE(a_r, zero)), Neg(FCmpONE(a_i, zero))));
      auto inf_with_sign_of_c = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::copysign, {inf, a_r}, {type}, b_);
      auto zero_denominator_result = EmitComposeComplex(
          op, FMul(inf_with_sign_of_c, a_r), FMul(inf_with_sign_of_c, a_i));

      // Case 2. Infinite numerator, finite denominator.
      auto b_r_finite = FCmpONE(llvm_ir::EmitCallToIntrinsic(
                                    llvm::Intrinsic::fabs, {b_r}, {type}, b_),
                                inf);
      auto b_i_finite = FCmpONE(llvm_ir::EmitCallToIntrinsic(
                                    llvm::Intrinsic::fabs, {b_i}, {type}, b_),
                                inf);
      auto inf_num_finite_denom = And(Or(FCmpOEQ(a_r, inf), FCmpOEQ(a_i, inf)),
                                      And(b_r_finite, b_i_finite));

      auto a_r_inf_with_sign = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::copysign,
          {Select(FCmpOEQ(a_r, inf), one, zero), a_r}, {type}, b_);
      auto a_i_inf_with_sign = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::copysign,
          {Select(FCmpOEQ(a_i, inf), one, zero), a_i}, {type}, b_);
      auto inf_num_finite_denom_result =
          EmitComposeComplex(op,
                             FMul(inf, FAdd(FMul(a_r_inf_with_sign, b_r),
                                            FMul(a_i_inf_with_sign, b_i))),
                             FMul(inf, FSub(FMul(a_i_inf_with_sign, b_r),
                                            FMul(a_r_inf_with_sign, b_i))));

      // Case 3. Finite numerator, infinite denominator.
      auto a_r_finite = FCmpONE(llvm_ir::EmitCallToIntrinsic(
                                    llvm::Intrinsic::fabs, {a_r}, {type}, b_),
                                inf);
      auto a_i_finite = FCmpONE(llvm_ir::EmitCallToIntrinsic(
                                    llvm::Intrinsic::fabs, {a_i}, {type}, b_),
                                inf);
      auto finite_num_inf_denom = And(Or(FCmpOEQ(b_r, inf), FCmpOEQ(b_i, inf)),
                                      And(a_r_finite, a_i_finite));

      auto b_r_inf_with_sign = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::copysign,
          {Select(FCmpOEQ(b_r, inf), one, zero), b_r}, {type}, b_);
      auto b_i_inf_with_sign = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::copysign,
          {Select(FCmpOEQ(b_i, inf), one, zero), b_i}, {type}, b_);
      auto finite_num_inf_denom_result =
          EmitComposeComplex(op,
                             FMul(zero, FAdd(FMul(a_r, b_r_inf_with_sign),
                                             FMul(a_i, b_i_inf_with_sign))),
                             FMul(zero, FSub(FMul(a_i, b_r_inf_with_sign),
                                             FMul(a_r, b_i_inf_with_sign))));

      auto c_nan = And(FCmpUNO(c_r, zero), FCmpUNO(c_i, zero));
      return Select(
          c_nan,
          Select(zero_denominator, zero_denominator_result,
                 Select(inf_num_finite_denom, inf_num_finite_denom_result,
                        Select(finite_num_inf_denom,
                               finite_num_inf_denom_result, result))),
          result);
    }
    // LLVM comparisons can be "unordered" (U) or "ordered" (O) -- ordered
    // comparisons always return false when one of the operands is NaN, whereas
    // unordered comparisons return true.
    //
    // We use ordered comparisons for everything except kNe, where we use an
    // unordered comparison.  This makes x != y equivalent to !(x == y), and
    // matches C++'s semantics.
    case HloOpcode::kCompare: {
      switch (op->comparison_direction()) {
        case ComparisonDirection::kEq:
          return And(llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OEQ,
                                             EmitExtractReal(lhs_value),
                                             EmitExtractReal(rhs_value), b_),
                     llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OEQ,
                                             EmitExtractImag(lhs_value),
                                             EmitExtractImag(rhs_value), b_));
        case ComparisonDirection::kNe:
          return Or(llvm_ir::EmitComparison(llvm::CmpInst::FCMP_UNE,
                                            EmitExtractReal(lhs_value),
                                            EmitExtractReal(rhs_value), b_),
                    llvm_ir::EmitComparison(llvm::CmpInst::FCMP_UNE,
                                            EmitExtractImag(lhs_value),
                                            EmitExtractImag(rhs_value), b_));
        default:
          return Unimplemented(
              "complex comparison '%s'",
              ComparisonDirectionToString(op->comparison_direction()));
      }
    }
    case HloOpcode::kPower: {
      auto a = EmitExtractReal(lhs_value);
      auto b = EmitExtractImag(lhs_value);
      auto c = EmitExtractReal(rhs_value);
      auto d = EmitExtractImag(rhs_value);
      return EmitComplexPower(op, a, b, c, d);
    }
    default:
      return Unimplemented("binary complex op '%s'",
                           HloOpcodeString(op->opcode()));
  }
}

llvm::Value* ElementalIrEmitter::EmitFloatMax(llvm::Value* lhs_value,
                                              llvm::Value* rhs_value) {
  return llvm_ir::EmitFloatMax(lhs_value, rhs_value, b_);
}

llvm::Value* ElementalIrEmitter::EmitFloatMin(llvm::Value* lhs_value,
                                              llvm::Value* rhs_value) {
  return llvm_ir::EmitFloatMin(lhs_value, rhs_value, b_);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitLog(PrimitiveType prim_type,
                                                   llvm::Value* value) {
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::log, {value},
                                      {value->getType()}, b_);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitLog1p(PrimitiveType prim_type,
                                                     llvm::Value* value) {
  auto x = value;
  auto type = llvm_ir::PrimitiveTypeToIrType(prim_type, module_);
  auto one = llvm::ConstantFP::get(type, 1.0);
  auto negative_half = llvm::ConstantFP::get(type, -0.5);
  // When x is large, the naive evaluation of ln(x + 1) is more
  // accurate than the Taylor series.
  TF_ASSIGN_OR_RETURN(auto for_large_x, EmitLog(prim_type, FAdd(x, one)));
  // The Taylor series for ln(x+1) is x - x^2/2 - x^3/3 + ….
  auto for_small_x = FMul(FAdd(FMul(negative_half, x), one), x);
  const auto kAntilogarithmIsSmallThreshold = 1e-4;
  auto abs_x =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {value}, {type}, b_);
  auto x_is_small = FCmpOLT(
      abs_x, llvm::ConstantFP::get(type, kAntilogarithmIsSmallThreshold));
  return Select(x_is_small, for_small_x, for_large_x);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitSqrt(PrimitiveType,
                                                    llvm::Value* value) {
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::sqrt, {value},
                                      {value->getType()}, b_);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitRsqrt(PrimitiveType prim_type,
                                                     llvm::Value* value) {
  TF_ASSIGN_OR_RETURN(auto sqrt, EmitSqrt(prim_type, value));
  return FDiv(llvm::ConstantFP::get(sqrt->getType(), 1.0), sqrt);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitSin(PrimitiveType prim_type,
                                                   llvm::Value* value) {
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::sin, {value},
                                      {value->getType()}, b_);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitCos(PrimitiveType prim_type,
                                                   llvm::Value* value) {
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::cos, {value},
                                      {value->getType()}, b_);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitExp(PrimitiveType prim_type,
                                                   llvm::Value* value) {
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::exp, {value},
                                      {value->getType()}, b_);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitExpm1(PrimitiveType prim_type,
                                                     llvm::Value* value) {
  auto x = value;
  auto type = llvm_ir::PrimitiveTypeToIrType(prim_type, module_);
  auto one = llvm::ConstantFP::get(type, 1.0);
  auto half = llvm::ConstantFP::get(type, 0.5);
  // When the exponent is large, the naive evaluation of e^(x) - 1 is more
  // accurate than the Taylor series.
  TF_ASSIGN_OR_RETURN(auto exp_x, EmitExp(prim_type, value));
  auto for_large_x = FSub(exp_x, one);
  // The Taylor series for exp(x) is 1 + x + x^2/2 + x^3/6 + ….
  // We want exp(x)-1 which is x + x^2/2 + x^3/6 + ….
  // We use the second degree approximation of exp(x)-1 = x + x^2/2.
  auto x_squared = FMul(x, x);
  auto x_squared_over_two = FMul(x_squared, half);
  auto for_small_x = FAdd(x, x_squared_over_two);
  // At this point, the relative errors due to floating point precision loss of
  // calculating exp(x) - 1 and the polynomial exp(x)-1 = x + x^2/2 are about
  // equal, with a value of approximetely 2^-16.
  const auto kExponentIsSmallThreshold = 0.009;
  auto abs_x =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {value}, {type}, b_);
  auto x_is_small =
      FCmpOLT(abs_x, llvm::ConstantFP::get(type, kExponentIsSmallThreshold));
  return Select(x_is_small, for_small_x, for_large_x);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitPow(PrimitiveType prim_type,
                                                   llvm::Value* lhs,
                                                   llvm::Value* rhs) {
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::pow, {lhs, rhs},
                                      {lhs->getType()}, b_);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitAtan2(PrimitiveType prim_type,
                                                     llvm::Value* lhs,
                                                     llvm::Value* rhs) {
  return Unimplemented("atan2");
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitTanh(PrimitiveType prim_type,
                                                    llvm::Value* value) {
  return Unimplemented("tanh");
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitReducePrecision(
    const HloInstruction* hlo, llvm::Value* x) {
  return EmitReducePrecisionIR(
      /*src_ty=*/hlo->operand(0)->shape().element_type(), x,
      /*dest_exponent_bits=*/hlo->exponent_bits(),
      /*dest_mantissa_bits=*/hlo->mantissa_bits(), b_);
}

static llvm::Value* SaturateShiftIfNecessary(llvm::IRBuilder<>* b,
                                             llvm::Value* lhs, llvm::Value* rhs,
                                             llvm::Value* shift_result,
                                             bool saturate_to_sign_bit) {
  llvm::IntegerType* integer_type =
      llvm::cast<llvm::IntegerType>(lhs->getType());
  unsigned integer_bitsize = integer_type->getBitWidth();
  llvm::ConstantInt* integer_bitsize_constant =
      llvm::ConstantInt::get(integer_type, integer_bitsize);
  llvm::ConstantInt* zero = llvm::ConstantInt::get(integer_type, 0);
  llvm::ConstantInt* minus_one = llvm::ConstantInt::get(integer_type, -1);
  llvm::Value* saturated_value;
  if (saturate_to_sign_bit) {
    saturated_value =
        b->CreateSelect(b->CreateICmpSLT(lhs, zero), minus_one, zero);
  } else {
    saturated_value = zero;
  }
  llvm::Value* shift_amt_in_range =
      b->CreateICmpULT(rhs, integer_bitsize_constant, "shft.chk");
  return b->CreateSelect(shift_amt_in_range, shift_result, saturated_value);
}

llvm::Value* ElementalIrEmitter::GetOne(llvm::Type* type) {
  return llvm::ConstantInt::get(llvm::cast<llvm::IntegerType>(type), 1);
}

llvm::Value* ElementalIrEmitter::GetZero(llvm::Type* type) {
  return llvm::ConstantInt::get(llvm::cast<llvm::IntegerType>(type), 0);
}

llvm::Value* ElementalIrEmitter::GetIntSMin(llvm::Type* type) {
  auto* integer_type = llvm::cast<llvm::IntegerType>(type);
  return llvm::ConstantInt::get(integer_type, llvm::APInt::getSignedMinValue(
                                                  integer_type->getBitWidth()));
}

llvm::Value* ElementalIrEmitter::GetMinusOne(llvm::Type* type) {
  auto* integer_type = llvm::cast<llvm::IntegerType>(type);
  return llvm::ConstantInt::get(
      integer_type, llvm::APInt::getAllOnesValue(integer_type->getBitWidth()));
}

llvm::Value* ElementalIrEmitter::IsZero(llvm::Value* v) {
  return ICmpEQ(v, llvm::ConstantInt::get(v->getType(), 0));
}

llvm::Value* ElementalIrEmitter::IsIntMinDivisionOverflow(llvm::Value* lhs,
                                                          llvm::Value* rhs) {
  return And(ICmpEQ(lhs, GetIntSMin(lhs->getType())),
             ICmpEQ(rhs, GetMinusOne(rhs->getType())));
}

llvm::Value* ElementalIrEmitter::EmitIntegerDivide(llvm::Value* lhs,
                                                   llvm::Value* rhs,
                                                   bool is_signed) {
  // Integer division overflow behavior:
  //
  // X / 0 == -1
  // INT_SMIN /s -1 = INT_SMIN

  if (!is_signed) {
    llvm::Value* udiv_is_unsafe = IsZero(rhs);
    llvm::Value* safe_rhs = Select(udiv_is_unsafe, GetOne(lhs->getType()), rhs);
    llvm::Value* safe_div = UDiv(lhs, safe_rhs);
    return Select(udiv_is_unsafe, GetMinusOne(lhs->getType()), safe_div);
  }

  llvm::Value* has_zero_divisor = IsZero(rhs);
  llvm::Value* has_int_min_overflow = IsIntMinDivisionOverflow(lhs, rhs);
  llvm::Value* sdiv_is_unsafe = Or(has_int_min_overflow, has_zero_divisor);
  llvm::Value* safe_rhs = Select(sdiv_is_unsafe, GetOne(lhs->getType()), rhs);
  llvm::Value* safe_div = SDiv(lhs, safe_rhs);

  return Select(
      has_zero_divisor, GetMinusOne(lhs->getType()),
      Select(has_int_min_overflow, GetIntSMin(lhs->getType()), safe_div));
}

llvm::Value* ElementalIrEmitter::EmitIntegerRemainder(llvm::Value* lhs,
                                                      llvm::Value* rhs,
                                                      bool is_signed) {
  // Integer remainder overflow behavior:
  //
  // X % 0 == X
  // INT_SMIN %s -1 = 0

  if (!is_signed) {
    llvm::Value* urem_is_unsafe = IsZero(rhs);
    llvm::Value* safe_rhs = Select(urem_is_unsafe, GetOne(lhs->getType()), rhs);
    llvm::Value* safe_rem = URem(lhs, safe_rhs);
    return Select(urem_is_unsafe, lhs, safe_rem);
  }

  llvm::Value* has_zero_divisor = IsZero(rhs);
  llvm::Value* has_int_min_overflow = IsIntMinDivisionOverflow(lhs, rhs);
  llvm::Value* srem_is_unsafe = Or(has_int_min_overflow, has_zero_divisor);
  llvm::Value* safe_rhs = Select(srem_is_unsafe, GetOne(lhs->getType()), rhs);
  llvm::Value* safe_rem = SRem(lhs, safe_rhs);

  return Select(
      has_zero_divisor, lhs,
      Select(has_int_min_overflow, GetZero(lhs->getType()), safe_rem));
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitIntegerBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value,
    bool is_signed) {
  switch (op->opcode()) {
    // TODO(jingyue): add the "nsw" attribute for signed types.
    case HloOpcode::kAdd:
      return Add(lhs_value, rhs_value);
    case HloOpcode::kSubtract:
      return Sub(lhs_value, rhs_value);
    case HloOpcode::kMultiply:
      return Mul(lhs_value, rhs_value);
    case HloOpcode::kDivide:
      return EmitIntegerDivide(lhs_value, rhs_value, is_signed);
    case HloOpcode::kRemainder:
      return EmitIntegerRemainder(lhs_value, rhs_value, is_signed);
    case HloOpcode::kCompare: {
      switch (op->comparison_direction()) {
        case ComparisonDirection::kEq:
          return llvm_ir::EmitComparison(llvm::CmpInst::ICMP_EQ, lhs_value,
                                         rhs_value, b_);
        case ComparisonDirection::kNe:
          return llvm_ir::EmitComparison(llvm::CmpInst::ICMP_NE, lhs_value,
                                         rhs_value, b_);
        case ComparisonDirection::kLt:
          return llvm_ir::EmitComparison(
              is_signed ? llvm::CmpInst::ICMP_SLT : llvm::CmpInst::ICMP_ULT,
              lhs_value, rhs_value, b_);
        case ComparisonDirection::kGt:
          return llvm_ir::EmitComparison(
              is_signed ? llvm::CmpInst::ICMP_SGT : llvm::CmpInst::ICMP_UGT,
              lhs_value, rhs_value, b_);
        case ComparisonDirection::kLe:
          return llvm_ir::EmitComparison(
              is_signed ? llvm::CmpInst::ICMP_SLE : llvm::CmpInst::ICMP_ULE,
              lhs_value, rhs_value, b_);
        case ComparisonDirection::kGe:
          return llvm_ir::EmitComparison(
              is_signed ? llvm::CmpInst::ICMP_SGE : llvm::CmpInst::ICMP_UGE,
              lhs_value, rhs_value, b_);
      }
    }
    case HloOpcode::kMinimum:
      return EmitIntegralMin(lhs_value, rhs_value, is_signed);
    case HloOpcode::kMaximum:
      return EmitIntegralMax(lhs_value, rhs_value, is_signed);
    case HloOpcode::kAnd:
      return And(lhs_value, rhs_value);
    case HloOpcode::kOr:
      return Or(lhs_value, rhs_value);
    case HloOpcode::kXor:
      return Xor(lhs_value, rhs_value);

    // Shifting out bits >= the number of bits in the type being shifted
    // produces a poison value in LLVM which is basically "deferred undefined
    // behavior" -- doing something observable with such a value precipitates
    // UB.  We replace the poison value with a constant to avoid this deferred
    // UB.
    case HloOpcode::kShiftRightArithmetic:
      return SaturateShiftIfNecessary(b_, lhs_value, rhs_value,
                                      AShr(lhs_value, rhs_value),
                                      /*saturate_to_sign_bit=*/true);
    case HloOpcode::kShiftLeft:
      return SaturateShiftIfNecessary(b_, lhs_value, rhs_value,
                                      Shl(lhs_value, rhs_value),
                                      /*saturate_to_sign_bit=*/false);
    case HloOpcode::kShiftRightLogical:
      return SaturateShiftIfNecessary(b_, lhs_value, rhs_value,
                                      LShr(lhs_value, rhs_value),
                                      /*saturate_to_sign_bit=*/false);
    default:
      return Unimplemented("binary integer op '%s'",
                           HloOpcodeString(op->opcode()));
  }
}

llvm::Value* ElementalIrEmitter::EmitIntegralMax(llvm::Value* lhs_value,
                                                 llvm::Value* rhs_value,
                                                 bool is_signed) {
  return Select(b_->CreateICmp(is_signed ? llvm::ICmpInst::ICMP_SGE
                                         : llvm::ICmpInst::ICMP_UGE,
                               lhs_value, rhs_value),
                lhs_value, rhs_value);
}

llvm::Value* ElementalIrEmitter::EmitIntegralMin(llvm::Value* lhs_value,
                                                 llvm::Value* rhs_value,
                                                 bool is_signed) {
  return Select(b_->CreateICmp(is_signed ? llvm::ICmpInst::ICMP_SLE
                                         : llvm::ICmpInst::ICMP_ULE,
                               lhs_value, rhs_value),
                lhs_value, rhs_value);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalSelect(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& index) {
  TF_ASSIGN_OR_RETURN(llvm::Value * pred_value,
                      operand_to_generator.at(hlo->operand(0))(index));
  TF_ASSIGN_OR_RETURN(llvm::Value * on_true_value,
                      operand_to_generator.at(hlo->operand(1))(index));
  TF_ASSIGN_OR_RETURN(llvm::Value * on_false_value,
                      operand_to_generator.at(hlo->operand(2))(index));
  return Select(Trunc(pred_value, b_->getInt1Ty()), on_true_value,
                on_false_value);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalClamp(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& index) {
  TF_ASSIGN_OR_RETURN(llvm::Value * min_value,
                      operand_to_generator.at(hlo->operand(0))(index));
  TF_ASSIGN_OR_RETURN(llvm::Value * arg_value,
                      operand_to_generator.at(hlo->operand(1))(index));
  TF_ASSIGN_OR_RETURN(llvm::Value * max_value,
                      operand_to_generator.at(hlo->operand(2))(index));
  PrimitiveType prim_type = hlo->shape().element_type();
  if (primitive_util::IsFloatingPointType(prim_type)) {
    return EmitFloatMin(max_value, EmitFloatMax(min_value, arg_value));
  } else if (primitive_util::IsIntegralType(prim_type)) {
    bool is_signed = primitive_util::IsSignedIntegralType(prim_type);
    return EmitIntegralMin(
        max_value, EmitIntegralMax(min_value, arg_value, is_signed), is_signed);
  } else {
    return Unimplemented("Clamp unimplemented for %s",
                         PrimitiveType_Name(prim_type));
  }
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalConcatenate(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& target_index) {
  const int64 concat_dim = hlo->dimensions(0);
  auto source_index = target_index;

  llvm::BasicBlock* init_block = b_->GetInsertBlock();

  // A terminator should be present iff we're emitting code
  // into the middle (as opposed to the end) of a basic block.
  CHECK_EQ(b_->GetInsertPoint() == init_block->end(),
           init_block->getTerminator() == nullptr);

  llvm::BasicBlock* exit_block;
  if (b_->GetInsertPoint() == init_block->end()) {
    exit_block = llvm_ir::CreateBasicBlock(
        /*insert_before=*/nullptr, IrName(hlo, "merge"), b_);
  } else {
    exit_block =
        init_block->splitBasicBlock(b_->GetInsertPoint(), IrName(hlo, "merge"));
    init_block->getTerminator()->eraseFromParent();
  }

  llvm_ir::SetToFirstInsertPoint(exit_block, b_);
  llvm::PHINode* output =
      PHI(llvm_ir::PrimitiveTypeToIrType(hlo->shape().element_type(), module_),
          hlo->operands().size());
  auto prior_insert_point = b_->GetInsertPoint();

  b_->SetInsertPoint(init_block);

  // Assign a unique id for each *different* operand, and count how often each
  // operand is used. If all operands are different, the usage count will be 1
  // for each operand.
  absl::flat_hash_map<const HloInstruction*, int64> to_unique_operand_id;
  std::vector<int64> operand_usage_count;
  for (const auto* operand : hlo->operands()) {
    if (to_unique_operand_id.contains(operand)) {
      ++operand_usage_count[to_unique_operand_id[operand]];
    } else {
      int64 unique_operand_id = to_unique_operand_id.size();
      to_unique_operand_id[operand] = unique_operand_id;
      operand_usage_count.push_back(1);
    }
  }

  // To avoid that we emit the same operand more than once, we create one basic
  // block for each *different* operand with a PHI node for the different source
  // index inputs.
  std::vector<llvm::BasicBlock*> emit_operand_blocks(
      to_unique_operand_id.size(), nullptr);
  std::vector<llvm::PHINode*> source_index_phis(to_unique_operand_id.size(),
                                                nullptr);
  for (const auto* operand : hlo->operands()) {
    int64 operand_id = to_unique_operand_id[operand];
    if (emit_operand_blocks[operand_id] != nullptr) {
      continue;
    }

    emit_operand_blocks[operand_id] = llvm_ir::CreateBasicBlock(
        exit_block, StrCat("concat_index_from_operand_id", operand_id), b_);
    auto saved_insert_point = b_->GetInsertPoint();
    llvm_ir::SetToFirstInsertPoint(emit_operand_blocks[operand_id], b_);
    source_index_phis[operand_id] =
        PHI(source_index.GetType(), operand_usage_count[operand_id]);
    std::vector<llvm::Value*> operand_multi_index = source_index.multidim();
    operand_multi_index[concat_dim] =
        NSWSub(operand_multi_index[concat_dim], source_index_phis[operand_id]);

    // Create the terminator of the block before calling operand generators,
    // because they require non-degenerate basic blocks.
    b_->SetInsertPoint(llvm::BranchInst::Create(
        exit_block, /*InsertAtEnd=*/emit_operand_blocks[operand_id]));
    llvm_ir::IrArray::Index operand_index(operand_multi_index, operand->shape(),
                                          source_index.GetType());
    TF_ASSIGN_OR_RETURN(llvm::Value * value,
                        operand_to_generator.at(operand)(operand_index));
    output->addIncoming(value, b_->GetInsertBlock());
    b_->SetInsertPoint(init_block, saved_insert_point);
  }

  int64 concat_dim_size = 0;
  for (int64 operand_idx = 0; operand_idx < hlo->operand_count();
       ++operand_idx) {
    const HloInstruction* operand = hlo->operand(operand_idx);
    auto false_block = llvm_ir::CreateBasicBlock(
        exit_block, StrCat("concat_index_not_from_operand", operand_idx), b_);
    int64 operand_id = to_unique_operand_id[operand];
    source_index_phis[operand_id]->addIncoming(
        source_index.GetConstantWithIndexType(concat_dim_size),
        b_->GetInsertBlock());
    concat_dim_size += operand->shape().dimensions(concat_dim);
    CondBr(ICmpULT(source_index[concat_dim],
                   source_index.GetConstantWithIndexType(concat_dim_size)),
           emit_operand_blocks[operand_id], false_block);

    // Subtract the size of the concat dimension of the current operand
    // from the source index.
    b_->SetInsertPoint(false_block);
  }

  Unreachable();
  b_->SetInsertPoint(exit_block, prior_insert_point);
  return output;
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalDynamicSlice(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& index) {
  // Emit IR to read dynamic start indices from hlo->operand(1).
  const HloInstruction* input_hlo = hlo->operand(0);
  const int64 rank = input_hlo->shape().rank();
  // Use the same index type for all tensor accesses in the same kernel.
  llvm::Type* index_type = index.GetType();
  std::vector<llvm::Value*> slice_start_multi_index(rank);
  for (int64 i = 0; i < rank; ++i) {
    auto index_typed_const = [&](uint64 c) -> llvm::Constant* {
      return llvm::ConstantInt::get(index_type, c);
    };
    llvm_ir::IrArray::Index zero_index(index_type);
    TF_ASSIGN_OR_RETURN(
        llvm::Value * start_index_value,
        operand_to_generator.at(hlo->operand(1 + i))(zero_index));

    // Clamp the start index so that the sliced portion fits in the operand:
    // start_index = clamp(start_index, 0, operand_dim_size - output_dim_size)
    start_index_value = SExtOrTrunc(start_index_value, index_type);
    int64 largest_valid_start_index =
        input_hlo->shape().dimensions(i) - hlo->shape().dimensions(i);
    CHECK_GE(largest_valid_start_index, 0);

    bool is_signed = ShapeUtil::ElementIsSigned(hlo->operand(1)->shape());
    start_index_value = EmitIntegralMin(
        index_typed_const(largest_valid_start_index),
        EmitIntegralMax(index_typed_const(0), start_index_value, is_signed),
        is_signed);

    start_index_value->setName(IrName(hlo, StrCat("start_idx", i)));
    slice_start_multi_index[i] = start_index_value;
  }

  std::vector<llvm::Value*> input_multi_index(rank);
  for (int64 i = 0; i < rank; ++i) {
    // Emit IR which computes:
    //   input_index = start_index + offset_index
    input_multi_index[i] = Add(slice_start_multi_index[i], index[i]);
  }
  llvm_ir::IrArray::Index input_index(input_multi_index, input_hlo->shape(),
                                      index_type);
  return operand_to_generator.at(input_hlo)(input_index);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalGather(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& index) {
  const Shape& operand_shape = hlo->operand(0)->shape();
  const Shape& indices_shape = hlo->operand(1)->shape();
  const Shape& output_shape = hlo->shape();

  const GatherDimensionNumbers& dim_numbers = hlo->gather_dimension_numbers();

  const llvm_ir::ElementGenerator& operand_generator =
      operand_to_generator.at(hlo->operand(0));
  const llvm_ir::ElementGenerator& indices_generator =
      operand_to_generator.at(hlo->operand(1));

  llvm::Type* index_type = index.GetType();
  // This is the index into `operand` that holds the element we want to
  // generate.
  std::vector<llvm::Value*> operand_multi_index;

  // First copy in the window indices to operand_index. Also collect a mapping
  // from operand dimension to output window dimension. Elided window dimensions
  // map to -1.
  std::vector<int64> operand_to_output_dim(operand_shape.dimensions_size(), -1);
  for (int64 i = 0, e = operand_shape.dimensions_size(), operand_index_dim = 0;
       i < e; i++) {
    if (absl::c_binary_search(dim_numbers.collapsed_slice_dims(), i)) {
      operand_multi_index.push_back(index.GetConstantWithIndexType(0));
    } else {
      int64 output_window_dim = dim_numbers.offset_dims(operand_index_dim++);
      operand_to_output_dim[i] = output_window_dim;
      operand_multi_index.push_back(index[output_window_dim]);
    }
  }

  // This is the index of the index vector in the start_indices tensor.
  std::vector<llvm::Value*> gather_index_index_components;
  {
    for (int64 i = 0, e = output_shape.dimensions_size(); i < e; i++) {
      if (!absl::c_binary_search(dim_numbers.offset_dims(), i)) {
        gather_index_index_components.push_back(index[i]);
      }
    }

    if (gather_index_index_components.size() !=
        indices_shape.dimensions_size()) {
      gather_index_index_components.insert(
          gather_index_index_components.begin() +
              dim_numbers.index_vector_dim(),
          nullptr);
    }
  }

  auto add_to_operand_index = [&](llvm::Value* index_component, int64 dim) {
    llvm::Value* gather_dim_component_extended =
        SExtOrTrunc(index_component, index_type);
    int64 operand_dim = dim_numbers.start_index_map(dim);
    int64 output_dim = operand_to_output_dim[operand_dim];
    // If 'output_dim' is -1, it means 'operand_dim' is an elided window dim.
    // This means we set the iteration index to 0, so for the purpose of the
    // following calculations we can consider the output dimension size to be 1.
    int64 output_dim_size =
        output_dim == -1 ? 1 : output_shape.dimensions(output_dim);
    int64 largest_valid_start_index =
        operand_shape.dimensions(operand_dim) - output_dim_size;
    CHECK_GE(largest_valid_start_index, 0);

    // Clamp the gather index so that the gather region fits in the operand.
    // gather_dim_component_extended_inbound =
    //     clamp(gather_dim_component_extended, 0, largest_valid_start_index);
    bool is_signed = ShapeUtil::ElementIsSigned(indices_shape);
    auto gather_dim_component_extended_inbound = EmitIntegralMin(
        index.GetConstantWithIndexType(largest_valid_start_index),
        EmitIntegralMax(index.GetConstantWithIndexType(0),
                        gather_dim_component_extended, is_signed),
        is_signed);

    operand_multi_index[operand_dim] =
        Add(operand_multi_index[operand_dim],
            gather_dim_component_extended_inbound);
  };

  if (indices_shape.dimensions_size() == dim_numbers.index_vector_dim()) {
    IrArray::Index gather_index_index(gather_index_index_components,
                                      indices_shape, index_type);
    TF_ASSIGN_OR_RETURN(llvm::Value * gather_dim_component,
                        indices_generator(gather_index_index));
    add_to_operand_index(gather_dim_component, 0);
  } else {
    int64 index_vector_size =
        indices_shape.dimensions(dim_numbers.index_vector_dim());
    for (int64 i = 0; i < index_vector_size; i++) {
      gather_index_index_components[dim_numbers.index_vector_dim()] =
          index.GetConstantWithIndexType(i);
      IrArray::Index gather_index_index(gather_index_index_components,
                                        indices_shape, index_type);
      TF_ASSIGN_OR_RETURN(llvm::Value * gather_dim_component,
                          indices_generator(gather_index_index));
      add_to_operand_index(gather_dim_component, i);
    }
  }
  IrArray::Index operand_index(operand_multi_index, operand_shape, index_type);
  return operand_generator(operand_index);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalDynamicUpdateSlice(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& index) {
  const HloInstruction* input_hlo = hlo->operand(0);
  const HloInstruction* update_hlo = hlo->operand(1);
  const HloInstruction* start_hlo = hlo->operand(2);
  // Calculate slice start/end indices.
  const int64 rank = input_hlo->shape().rank();
  std::vector<llvm::Value*> slice_start_multi_index(rank);
  std::vector<llvm::Value*> slice_limit_multi_index(rank);
  // Slice intersection gathers (ANDs) conditions on all ranks for which
  // 'input' is set to 'update'
  llvm::Value* slice_intersection = b_->getTrue();

  for (int64 i = 0; i < rank; ++i) {
    llvm::Type* index_type = index[0]->getType();
    auto index_typed_const = [&](uint64 c) -> llvm::Constant* {
      return llvm::ConstantInt::get(index_type, c);
    };

    llvm_ir::IrArray::Index zero_index(index_type);
    TF_ASSIGN_OR_RETURN(
        llvm::Value * start_index_value,
        operand_to_generator.at(hlo->operand(2 + i))(zero_index));

    // Clamp the start index so that the update region fits in the operand.
    // start_index = clamp(start_index, 0, input_dim_size - update_dim_size)
    start_index_value = SExtOrTrunc(start_index_value, index_type);
    llvm::Value* update_dim_size =
        index_typed_const(update_hlo->shape().dimensions(i));
    int64 largest_valid_start_index =
        input_hlo->shape().dimensions(i) - update_hlo->shape().dimensions(i);
    CHECK_GE(largest_valid_start_index, 0);

    bool is_signed = ShapeUtil::ElementIsSigned(start_hlo->shape());
    start_index_value = EmitIntegralMin(
        index_typed_const(largest_valid_start_index),
        EmitIntegralMax(index_typed_const(0), start_index_value, is_signed),
        is_signed);

    start_index_value->setName(IrName(hlo, StrCat("start_idx", i)));
    slice_start_multi_index[i] = start_index_value;
    slice_limit_multi_index[i] =
        Add(slice_start_multi_index[i], update_dim_size);

    slice_intersection =
        And(slice_intersection, ICmpSGE(index[i], slice_start_multi_index[i]),
            "slice_intersection");
    slice_intersection =
        And(slice_intersection, ICmpSLT(index[i], slice_limit_multi_index[i]),
            "slice_intersection");
  }

  // Emit:
  // if (slice_intersection) -> return data from 'update'.
  // else                    -> return data from 'input'.
  llvm::Value* ret_value_addr = llvm_ir::EmitAllocaAtFunctionEntry(
      llvm_ir::PrimitiveTypeToIrType(hlo->shape().element_type(), module_),
      "ret_value_addr", b_);
  llvm_ir::LlvmIfData if_data =
      llvm_ir::EmitIfThenElse(slice_intersection, "slice_intersection", b_);

  // Handle true BB (return data from 'update')
  SetToFirstInsertPoint(if_data.true_block, b_);
  // Compute update index for intersection case.
  std::vector<llvm::Value*> update_multi_index(rank);
  for (int64 i = 0; i < rank; ++i) {
    update_multi_index[i] = Sub(index[i], slice_start_multi_index[i]);
  }
  llvm_ir::IrArray::Index update_index(update_multi_index, update_hlo->shape(),
                                       index.GetType());
  TF_ASSIGN_OR_RETURN(llvm::Value * true_value,
                      operand_to_generator.at(update_hlo)(update_index));
  Store(true_value, ret_value_addr);

  // Handle false BB (return data from 'input')
  SetToFirstInsertPoint(if_data.false_block, b_);
  TF_ASSIGN_OR_RETURN(llvm::Value * false_value,
                      operand_to_generator.at(input_hlo)(index));
  Store(false_value, ret_value_addr);

  SetToFirstInsertPoint(if_data.after_block, b_);
  return Load(ret_value_addr);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalPad(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& padded_index) {
  std::vector<llvm::Value*> multi_index = padded_index.multidim();
  llvm::Value* in_bounds = b_->getTrue();
  for (size_t i = 0; i < multi_index.size(); ++i) {
    auto index_typed_const = [=](int64 n) {
      return padded_index.GetConstantWithIndexType(n);
    };
    const auto& pad_dim = hlo->padding_config().dimensions(i);
    multi_index[i] =
        Sub(multi_index[i], index_typed_const(pad_dim.edge_padding_low()));
    in_bounds = And(in_bounds, ICmpSGE(multi_index[i], index_typed_const(0)),
                    "in_bounds");
    in_bounds =
        And(in_bounds,
            ICmpEQ(index_typed_const(0),
                   URem(multi_index[i],
                        index_typed_const(pad_dim.interior_padding() + 1))),
            "in_bounds");
    multi_index[i] =
        SDiv(multi_index[i], index_typed_const(pad_dim.interior_padding() + 1));
    in_bounds =
        And(in_bounds,
            ICmpSLT(multi_index[i],
                    index_typed_const(hlo->operand(0)->shape().dimensions(i))),
            "in_bounds");
  }

  // if (in_bounds) {
  //   ret_value = operand0[index];  // source
  // } else {
  //   ret_value = *operand1;        // padding
  // }
  llvm::Value* ret_value_addr = llvm_ir::EmitAllocaAtFunctionEntry(
      llvm_ir::PrimitiveTypeToIrType(hlo->shape().element_type(), module_),
      "pad_result_addr", b_);
  llvm_ir::LlvmIfData if_data =
      llvm_ir::EmitIfThenElse(in_bounds, "in_bounds", b_);
  SetToFirstInsertPoint(if_data.true_block, b_);
  llvm_ir::IrArray::Index index(multi_index, hlo->operand(0)->shape(),
                                padded_index.GetType());
  TF_ASSIGN_OR_RETURN(llvm::Value * operand_value,
                      operand_to_generator.at(hlo->operand(0))(index));
  Store(operand_value, ret_value_addr);

  SetToFirstInsertPoint(if_data.false_block, b_);
  TF_ASSIGN_OR_RETURN(llvm::Value * padding_value,
                      operand_to_generator.at(hlo->operand(1))(
                          IrArray::Index(index.GetType())));
  Store(padding_value, ret_value_addr);

  SetToFirstInsertPoint(if_data.after_block, b_);
  // Don't create phi(operand_value, padding_value) here, because invoking
  // operand_to_generator may create new basic blocks, making the parent
  // of operand_value or padding_value no longer a predecessor of
  // if_data.after_block.
  return Load(ret_value_addr);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalDot(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& dot_result_index) {
  auto lhs_generator = operand_to_generator.at(hlo->operand(0));
  auto rhs_generator = operand_to_generator.at(hlo->operand(1));

  const DotDimensionNumbers& dim_numbers = hlo->dot_dimension_numbers();
  int64 lhs_contracting_dim = dim_numbers.lhs_contracting_dimensions(0);
  int64 rhs_contracting_dim = dim_numbers.rhs_contracting_dimensions(0);

  int64 contracted_dim_size =
      hlo->operand(0)->shape().dimensions(lhs_contracting_dim);
  int64 lhs_dims = hlo->operand(0)->shape().dimensions_size();
  int64 rhs_dims = hlo->operand(1)->shape().dimensions_size();

  llvm::Type* index_type = dot_result_index.GetType();
  auto index_typed_const = [&](uint64 c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_type, c);
  };

  std::unique_ptr<llvm_ir::ForLoop> inner_loop = llvm_ir::ForLoop::EmitForLoop(
      IrName(hlo, "inner"), index_typed_const(0),
      index_typed_const(contracted_dim_size), index_typed_const(1), b_);

  SetToFirstInsertPoint(inner_loop->GetPreheaderBasicBlock(), b_);
  PrimitiveType primitive_type = hlo->shape().element_type();
  llvm::Type* primitive_type_llvm =
      llvm_ir::PrimitiveTypeToIrType(primitive_type, module_);
  llvm::Value* accumulator_alloca =
      llvm_ir::EmitAllocaAtFunctionEntry(primitive_type_llvm, "dot_acc", b_);
  Store(llvm::Constant::getNullValue(primitive_type_llvm), accumulator_alloca);

  SetToFirstInsertPoint(inner_loop->GetBodyBasicBlock(), b_);

  // This is the inner reduction loop for a dot operation that produces
  // one element in the output.  If the operands to the dot operation have
  // shapes [A,B,C,T] and [D,T,E], the result has a shape [A,B,C,D,E].
  // Given an output index [a,b,c,d,e] in the result, we compute:
  //   sum(lhs[a,b,c,t]*rhs[d,t,e] for t in [0, T))

  std::vector<llvm::Value*> lhs_multi_index, rhs_multi_index;
  for (int64 i = 0; i < lhs_dims - 1; i++) {
    lhs_multi_index.push_back(dot_result_index[i]);
  }
  lhs_multi_index.insert(lhs_multi_index.begin() + lhs_contracting_dim,
                         inner_loop->GetIndVarValue());
  IrArray::Index lhs_index(lhs_multi_index, hlo->operand(0)->shape(),
                           index_type);

  int64 num_batch_dims = dim_numbers.rhs_batch_dimensions_size();
  for (int64 i = 0; i < num_batch_dims; i++) {
    rhs_multi_index.push_back(
        dot_result_index[dim_numbers.rhs_batch_dimensions(i)]);
  }
  for (int64 i = 0; i < rhs_dims - 1 - num_batch_dims; i++) {
    rhs_multi_index.push_back(dot_result_index[lhs_dims - 1 + i]);
  }
  rhs_multi_index.insert(rhs_multi_index.begin() + rhs_contracting_dim,
                         inner_loop->GetIndVarValue());
  IrArray::Index rhs_index(rhs_multi_index, hlo->operand(1)->shape(),
                           index_type);

  llvm::Value* current_accumulator = Load(accumulator_alloca);
  TF_ASSIGN_OR_RETURN(llvm::Value * lhs_value, lhs_generator(lhs_index));
  TF_ASSIGN_OR_RETURN(llvm::Value * rhs_value, rhs_generator(rhs_index));
  llvm::Value* next_accumulator;
  if (primitive_util::IsComplexType(primitive_type)) {
    llvm::Value* product_real =
        FSub(FMul(EmitExtractReal(lhs_value), EmitExtractReal(rhs_value)),
             FMul(EmitExtractImag(lhs_value), EmitExtractImag(rhs_value)));
    llvm::Value* product_imag =
        FAdd(FMul(EmitExtractReal(lhs_value), EmitExtractImag(rhs_value)),
             FMul(EmitExtractImag(lhs_value), EmitExtractReal(rhs_value)));
    next_accumulator = InsertValue(
        current_accumulator,
        FAdd(EmitExtractReal(current_accumulator), product_real), {0});
    next_accumulator = InsertValue(
        next_accumulator,
        FAdd(EmitExtractImag(current_accumulator), product_imag), {1});
  } else if (primitive_util::IsFloatingPointType(primitive_type)) {
    next_accumulator = FAdd(current_accumulator, FMul(lhs_value, rhs_value));
  } else {
    next_accumulator = Add(current_accumulator, Mul(lhs_value, rhs_value));
  }
  Store(next_accumulator, accumulator_alloca);

  SetToFirstInsertPoint(inner_loop->GetExitBasicBlock(), b_);
  return Load(accumulator_alloca);
}

llvm_ir::ElementGenerator ElementalIrEmitter::MakeElementGenerator(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator) {
  switch (hlo->opcode()) {
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kConvert:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kTanh:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        TF_ASSIGN_OR_RETURN(llvm::Value * operand_value,
                            operand_to_generator.at(hlo->operand(0))(index));
        return EmitUnaryOp(hlo, operand_value);
      };
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kAtan2:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSubtract:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        const HloInstruction* lhs = hlo->operand(0);
        const HloInstruction* rhs = hlo->operand(1);
        TF_ASSIGN_OR_RETURN(llvm::Value * lhs_value,
                            operand_to_generator.at(lhs)(index));
        TF_ASSIGN_OR_RETURN(llvm::Value * rhs_value,
                            operand_to_generator.at(rhs)(index));
        return EmitBinaryOp(hlo, lhs_value, rhs_value);
      };
    case HloOpcode::kSelect:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        return EmitElementalSelect(hlo, operand_to_generator, index);
      };
    case HloOpcode::kClamp:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        return EmitElementalClamp(hlo, operand_to_generator, index);
      };
    case HloOpcode::kReducePrecision:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        TF_ASSIGN_OR_RETURN(llvm::Value * operand_value,
                            operand_to_generator.at(hlo->operand(0))(index));
        return EmitReducePrecision(hlo, operand_value);
      };
    case HloOpcode::kConcatenate:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index target_index) -> StatusOr<llvm::Value*> {
        return EmitElementalConcatenate(hlo, operand_to_generator,
                                        target_index);
      };
    case HloOpcode::kReverse:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& target_index) -> StatusOr<llvm::Value*> {
        const HloInstruction* operand = hlo->operand(0);
        std::vector<llvm::Value*> source_multi_index = target_index.multidim();
        for (int64 dim : hlo->dimensions()) {
          source_multi_index[dim] = Sub(target_index.GetConstantWithIndexType(
                                            hlo->shape().dimensions(dim) - 1),
                                        target_index[dim]);
        }
        llvm_ir::IrArray::Index source_index(
            source_multi_index, operand->shape(), target_index.GetType());
        return operand_to_generator.at(operand)(source_index);
      };
    case HloOpcode::kBroadcast:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& target_index) -> StatusOr<llvm::Value*> {
        const HloInstruction* operand = hlo->operand(0);
        // The `dimensions` member of the broadcast instruction maps from
        // input dimensions to output dimensions.
        return operand_to_generator.at(operand)(
            target_index.SourceIndexOfBroadcast(hlo->shape(), operand->shape(),
                                                hlo->dimensions(), b_));
      };
    case HloOpcode::kIota:
      return [this, hlo](
                 const IrArray::Index& target_index) -> StatusOr<llvm::Value*> {
        auto* iota = Cast<HloIotaInstruction>(hlo);
        PrimitiveType element_type = iota->shape().element_type();
        IrArray::Index elem_index =
            iota->shape().rank() > 1
                ? target_index.SourceIndexOfBroadcast(
                      iota->shape(),
                      ShapeUtil::MakeShapeWithDescendingLayout(
                          element_type,
                          {iota->shape().dimensions(iota->iota_dimension())}),
                      {iota->iota_dimension()}, b_)
                : target_index;
        llvm::Value* elem_index_linear = elem_index.linear();
        if (elem_index_linear == nullptr) {
          std::vector<int64> iota_bound = {
              iota->shape().dimensions(iota->iota_dimension())};
          elem_index_linear = elem_index.Linearize(iota_bound, b_);
        }
        Shape component_shape =
            ShapeUtil::ElementIsComplex(iota->shape())
                ? ShapeUtil::ComplexComponentShape(iota->shape())
                : iota->shape();
        PrimitiveType component_element_type = component_shape.element_type();
        llvm::Value* iota_result;
        if (primitive_util::IsIntegralType(component_element_type)) {
          iota_result = b_->CreateIntCast(
              elem_index_linear,
              llvm_ir::PrimitiveTypeToIrType(component_element_type, module_),
              /*isSigned=*/false);
        } else {
          TF_RET_CHECK(
              primitive_util::IsFloatingPointType(component_element_type))
              << component_element_type;
          llvm::Type* float_ir_type;
          if (component_element_type == BF16) {
            float_ir_type = llvm_ir::PrimitiveTypeToIrType(F32, module_);
          } else {
            float_ir_type =
                llvm_ir::PrimitiveTypeToIrType(component_element_type, module_);
          }
          llvm::Value* float_val =
              b_->CreateUIToFP(elem_index_linear, float_ir_type);
          if (component_element_type == BF16) {
            TF_ASSIGN_OR_RETURN(iota_result, EmitF32ToBF16(float_val, b_));
          } else {
            iota_result = float_val;
          }
        }
        if (ShapeUtil::ElementIsComplex(iota->shape())) {
          return EmitComposeComplex(iota, iota_result, nullptr);
        } else {
          return iota_result;
        }
      };
    case HloOpcode::kSlice:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        IrArray::Index sliced_index = index.SourceIndexOfSlice(
            /*operand_shape=*/hlo->operand(0)->shape(),
            /*starts=*/hlo->slice_starts(),
            /*strides=*/hlo->slice_strides(), /*builder=*/b_);
        return operand_to_generator.at(hlo->operand(0))(sliced_index);
      };
    case HloOpcode::kDynamicSlice:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        return EmitElementalDynamicSlice(hlo, operand_to_generator, index);
      };

    case HloOpcode::kGather:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        return EmitElementalGather(hlo, operand_to_generator, index);
      };
    case HloOpcode::kDynamicUpdateSlice:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        return EmitElementalDynamicUpdateSlice(hlo, operand_to_generator,
                                               index);
      };
    case HloOpcode::kBitcast:
      CHECK_EQ(ShapeUtil::ElementsIn(hlo->shape()),
               ShapeUtil::ElementsIn(hlo->operand(0)->shape()));
      return [this, hlo, &operand_to_generator](const IrArray::Index& index) {
        const HloInstruction* operand = hlo->operand(0);
        return operand_to_generator.at(operand)(
            index.SourceIndexOfBitcast(hlo->shape(), operand->shape(), b_));
      };
    case HloOpcode::kReshape:
      CHECK_EQ(ShapeUtil::ElementsIn(hlo->shape()),
               ShapeUtil::ElementsIn(hlo->operand(0)->shape()));
      return [this, hlo, &operand_to_generator](const IrArray::Index& index) {
        const HloInstruction* operand = hlo->operand(0);
        return operand_to_generator.at(operand)(
            index.SourceIndexOfReshape(hlo->shape(), operand->shape(), b_));
      };
    case HloOpcode::kCopy:
      return [hlo, &operand_to_generator](
                 const IrArray::Index& target_index) -> StatusOr<llvm::Value*> {
        IrArray::Index source_index(target_index.multidim(),
                                    hlo->operand(0)->shape(),
                                    target_index.GetType());
        TF_ASSIGN_OR_RETURN(
            llvm::Value * operand_value,
            operand_to_generator.at(hlo->operand(0))(source_index));
        return operand_value;
      };
    case HloOpcode::kTranspose:
      return [this, hlo,
              &operand_to_generator](const IrArray::Index& target_index) {
        return operand_to_generator.at(hlo->operand(0))(
            target_index.SourceIndexOfTranspose(
                hlo->shape(), hlo->operand(0)->shape(), hlo->dimensions(), b_));
      };
    case HloOpcode::kPad:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& padded_index) -> StatusOr<llvm::Value*> {
        return EmitElementalPad(hlo, operand_to_generator, padded_index);
      };

    case HloOpcode::kDot:
      return [this, hlo,
              &operand_to_generator](const IrArray::Index& dot_result_index)
                 -> StatusOr<llvm::Value*> {
        return EmitElementalDot(hlo, operand_to_generator, dot_result_index);
      };
    default:
      return [hlo](const IrArray::Index& index) {
        return Unimplemented("Unhandled opcode for elemental IR emission: %s",
                             HloOpcodeString(hlo->opcode()));
      };
  }
}

llvm::Value* ElementalIrEmitter::EmitExtractReal(llvm::Value* value) {
  return ExtractValue(value, {0});
}

llvm::Value* ElementalIrEmitter::EmitExtractImag(llvm::Value* value) {
  return ExtractValue(value, {1});
}

llvm::Value* ElementalIrEmitter::EmitComposeComplex(const HloInstruction* op,
                                                    llvm::Value* real,
                                                    llvm::Value* imag) {
  auto cplx_type =
      llvm_ir::PrimitiveTypeToIrType(op->shape().element_type(), module_);
  auto complex =
      InsertValue(llvm::ConstantAggregateZero::get(cplx_type), real, {0});
  if (imag != nullptr) {
    complex = InsertValue(complex, imag, {1});
  }
  return complex;
}

}  // namespace xla
