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
using llvm_ir::AsStringRef;
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

llvm::Value* EmitReducePrecisionFloat(llvm::Value* x, int64 exponent_bits,
                                      int64 mantissa_bits,
                                      llvm::IRBuilder<>* b) {
  // Integer and float types for casting and constant generation.
  llvm::Type* float_type = x->getType();
  llvm::IntegerType* int_type = b->getInt32Ty();

  // Cast the input value to an integer for bitwise manipulation.
  llvm::Value* x_as_int = b->CreateBitCast(x, int_type);

  if (mantissa_bits < 23) {
    // Last remaining mantissa bit.
    const uint32_t last_mantissa_bit_mask = 1u << (23 - mantissa_bits);

    // Compute rounding bias for round-to-nearest with ties to even.  This is
    // equal to a base value of 0111... plus one bit if the last remaining
    // mantissa bit is 1.
    const uint32_t base_rounding_bias = (last_mantissa_bit_mask >> 1) - 1;
    llvm::Value* x_last_mantissa_bit = b->CreateLShr(
        b->CreateAnd(x_as_int,
                     llvm::ConstantInt::get(int_type, last_mantissa_bit_mask)),
        (23 - mantissa_bits));
    llvm::Value* x_rounding_bias =
        b->CreateAdd(x_last_mantissa_bit,
                     llvm::ConstantInt::get(int_type, base_rounding_bias));

    // Add rounding bias, and mask out truncated bits.  Note that the case
    // where adding the rounding bias overflows into the exponent bits is
    // correct; the non-masked mantissa bits will all be zero, and the
    // exponent will be incremented by one.
    const uint32_t truncation_mask = ~(last_mantissa_bit_mask - 1);
    x_as_int = b->CreateAdd(x_as_int, x_rounding_bias);
    x_as_int = b->CreateAnd(x_as_int,
                            llvm::ConstantInt::get(int_type, truncation_mask));
  }

  if (exponent_bits < 8) {
    // Masks for f32 values.
    const uint32_t f32_sign_bit_mask = 1u << 31;
    const uint32_t f32_exp_bits_mask = 0xffu << 23;

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
    const uint32_t f32_exponent_bias = (1 << 7) - 1;
    const uint32_t reduced_exponent_bias = (1 << (exponent_bits - 1)) - 1;
    const uint32_t reduced_max_exponent =
        f32_exponent_bias + reduced_exponent_bias;
    const uint32_t reduced_min_exponent =
        f32_exponent_bias - reduced_exponent_bias;

    // Do we overflow or underflow?
    llvm::Value* x_exponent = b->CreateAnd(
        x_as_int, llvm::ConstantInt::get(int_type, f32_exp_bits_mask));
    llvm::Value* x_overflows = b->CreateICmpUGT(
        x_exponent,
        llvm::ConstantInt::get(int_type, reduced_max_exponent << 23));
    llvm::Value* x_underflows = b->CreateICmpULE(
        x_exponent,
        llvm::ConstantInt::get(int_type, reduced_min_exponent << 23));

    // Compute appropriately-signed values of zero and infinity.
    llvm::Value* x_signed_zero = b->CreateAnd(
        x_as_int, llvm::ConstantInt::get(int_type, f32_sign_bit_mask));
    llvm::Value* x_signed_inf = b->CreateOr(
        x_signed_zero, llvm::ConstantInt::get(int_type, f32_exp_bits_mask));

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

    if (mantissa_bits > 0) {
      result = b->CreateSelect(x_is_nan, x, result);
    } else {
      result = b->CreateSelect(
          x_is_nan, llvm::ConstantFP::getInfinity(float_type), result);
    }
  }
  return result;
}

llvm::Value* EmitF32ToBF16(llvm::Value* f32_value, llvm::IRBuilder<>* b) {
  auto reduced_precision = EmitReducePrecisionFloat(
      f32_value,
      /*exponent_bits=*/primitive_util::kBFloat16ExponentBits,
      /*mantissa_bits=*/primitive_util::kBFloat16MantissaBits, b);
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
  if (op->opcode() == HloOpcode::kCopy) {
    return operand_value;
  } else if (ShapeUtil::ElementIsIntegral(op->operand(0)->shape()) ||
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
      // TODO(b/32151903): Ensure consistent sign behavior for -0.0.
      auto type = operand_value->getType();
      auto zero = llvm::ConstantFP::get(type, 0.0);
      auto oeq = FCmpOEQ(operand_value, zero);
      auto olt = FCmpOLT(operand_value, zero);
      return Select(oeq, zero,
                    Select(olt, llvm::ConstantFP::get(type, -1.0),
                           llvm::ConstantFP::get(type, 1.0)));
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
      // log(a+bi) = .5*log(a^2+b^2) + i*atan2(b, a)
      auto a = EmitExtractReal(operand_value);
      auto b = EmitExtractImag(operand_value);
      llvm::Type* llvm_ty = a->getType();
      auto sum_sq = FAdd(FMul(a, a), FMul(b, b));
      TF_ASSIGN_OR_RETURN(auto log_sum_sq, EmitLog(component_type, sum_sq));
      TF_ASSIGN_OR_RETURN(auto angle, EmitAtan2(component_type, b, a));
      auto one_half = llvm::ConstantFP::get(llvm_ty, 0.5);
      return EmitComposeComplex(op, FMul(one_half, log_sum_sq), angle);
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
      */
      auto a = EmitExtractReal(operand_value);
      auto b = EmitExtractImag(operand_value);
      TF_ASSIGN_OR_RETURN(auto exp_a, EmitExp(component_type, a));
      TF_ASSIGN_OR_RETURN(auto cos_b, EmitCos(component_type, b));
      TF_ASSIGN_OR_RETURN(auto sin_b, EmitSin(component_type, b));
      auto exp_neg_a = FDiv(llvm::ConstantFP::get(exp_a->getType(), 1), exp_a);
      auto exp_2a_minus_exp_neg_2a =
          FSub(FMul(exp_a, exp_a), FMul(exp_neg_a, exp_neg_a));
      auto cos_b_sq = FMul(cos_b, cos_b);
      auto sin_b_sq = FMul(sin_b, sin_b);
      auto real_num = FAdd(FMul(cos_b_sq, exp_2a_minus_exp_neg_2a),
                           FMul(sin_b_sq, exp_2a_minus_exp_neg_2a));
      auto cos_b_sin_b = FMul(cos_b, sin_b);
      auto exp_a_plus_exp_neg_a = FAdd(exp_a, exp_neg_a);
      auto exp_a_plus_exp_neg_a_sq =
          FMul(exp_a_plus_exp_neg_a, exp_a_plus_exp_neg_a);
      auto exp_a_minus_exp_neg_a = FSub(exp_a, exp_neg_a);
      auto exp_a_minus_exp_neg_a_sq =
          FMul(exp_a_minus_exp_neg_a, exp_a_minus_exp_neg_a);
      auto imag_num = FMul(
          cos_b_sin_b, FSub(exp_a_plus_exp_neg_a_sq, exp_a_minus_exp_neg_a_sq));
      auto denom = FAdd(FMul(cos_b_sq, exp_a_plus_exp_neg_a_sq),
                        FMul(sin_b_sq, exp_a_minus_exp_neg_a_sq));
      return EmitComposeComplex(op, FDiv(real_num, denom),
                                FDiv(imag_num, denom));
    }
    case HloOpcode::kAbs: {
      auto sum_sq = FAdd(
          FMul(EmitExtractReal(operand_value), EmitExtractReal(operand_value)),
          FMul(EmitExtractImag(operand_value), EmitExtractImag(operand_value)));
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::sqrt, {sum_sq},
                                          {sum_sq->getType()}, b_);
    }
    case HloOpcode::kSign: {  // Sign(c) = c / |c|
      auto sum_sq = FAdd(
          FMul(EmitExtractReal(operand_value), EmitExtractReal(operand_value)),
          FMul(EmitExtractImag(operand_value), EmitExtractImag(operand_value)));
      auto cplx_abs = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::sqrt, {sum_sq}, {sum_sq->getType()}, b_);
      auto type = cplx_abs->getType();
      auto zero = llvm::ConstantFP::get(type, 0.0);
      auto oeq = FCmpOEQ(cplx_abs, zero);
      return Select(
          oeq, EmitComposeComplex(op, zero, zero),
          EmitComposeComplex(op, FDiv(EmitExtractReal(operand_value), cplx_abs),
                             FDiv(EmitExtractImag(operand_value), cplx_abs)));
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
    case HloOpcode::kEq:
      return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OEQ, lhs_value,
                                     rhs_value, b_);
    case HloOpcode::kNe:
      return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_UNE, lhs_value,
                                     rhs_value, b_);
    case HloOpcode::kLt:
      return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OLT, lhs_value,
                                     rhs_value, b_);
    case HloOpcode::kGt:
      return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OGT, lhs_value,
                                     rhs_value, b_);
    case HloOpcode::kLe:
      return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OLE, lhs_value,
                                     rhs_value, b_);
    case HloOpcode::kGe:
      return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OGE, lhs_value,
                                     rhs_value, b_);

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
      // (a+bi) / (c+di) = ((a+bi)(c-di)) / ((c+di)(c-di))
      // = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
      auto rhs_sum_sq =
          FAdd(FMul(EmitExtractReal(rhs_value), EmitExtractReal(rhs_value)),
               FMul(EmitExtractImag(rhs_value), EmitExtractImag(rhs_value)));
      auto type = rhs_sum_sq->getType();
      auto zero = llvm::ConstantFP::get(type, 0.0);
      auto oeq = FCmpOEQ(rhs_sum_sq, zero);
      auto real_inf_or_nan = FDiv(EmitExtractReal(lhs_value), zero);
      auto imag_inf_or_nan = FDiv(EmitExtractImag(lhs_value), zero);
      return Select(
          oeq, EmitComposeComplex(op, real_inf_or_nan, imag_inf_or_nan),
          EmitComposeComplex(op,
                             FDiv(FAdd(FMul(EmitExtractReal(lhs_value),
                                            EmitExtractReal(rhs_value)),
                                       FMul(EmitExtractImag(lhs_value),
                                            EmitExtractImag(rhs_value))),
                                  rhs_sum_sq),
                             FDiv(FSub(FMul(EmitExtractImag(lhs_value),
                                            EmitExtractReal(rhs_value)),
                                       FMul(EmitExtractReal(lhs_value),
                                            EmitExtractImag(rhs_value))),
                                  rhs_sum_sq)));
    }
    // LLVM comparisons can be "unordered" (U) or "ordered" (O) -- ordered
    // comparisons always return false when one of the operands is NaN, whereas
    // unordered comparisons return true.
    //
    // We use ordered comparisons for everything except kNe, where we use an
    // unordered comparison.  This makes x != y equivalent to !(x == y), and
    // matches C++'s semantics.
    case HloOpcode::kEq:
      return And(llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OEQ,
                                         EmitExtractReal(lhs_value),
                                         EmitExtractReal(rhs_value), b_),
                 llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OEQ,
                                         EmitExtractImag(lhs_value),
                                         EmitExtractImag(rhs_value), b_));
    case HloOpcode::kNe:
      return Or(llvm_ir::EmitComparison(llvm::CmpInst::FCMP_UNE,
                                        EmitExtractReal(lhs_value),
                                        EmitExtractReal(rhs_value), b_),
                llvm_ir::EmitComparison(llvm::CmpInst::FCMP_UNE,
                                        EmitExtractImag(lhs_value),
                                        EmitExtractImag(rhs_value), b_));

    case HloOpcode::kPower: {
      // (a+bi)^(c+di) =
      //    (a*a+b*b)^(0.5c) * exp(-d*atan2(b,a)) * (cos(q) + i*sin(q)),
      //    where q = c*atan2(b,a)+0.5d*ln(a*a+b*b)
      PrimitiveType component_type =
          primitive_util::ComplexComponentType(op->shape().element_type());
      auto a = EmitExtractReal(lhs_value);
      auto b = EmitExtractImag(lhs_value);
      auto c = EmitExtractReal(rhs_value);
      auto d = EmitExtractImag(rhs_value);
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
      // 0^c is 0 if d is 0 and c > 0. 0^0 is defined to be 1.0, see
      // Branch Cuts for Complex Elementary Functions or Much Ado About
      // Nothing's Sign Bit, W. Kahan, Section 10.
      return Select(
          And(And(FCmpOEQ(aa_p_bb, zero), FCmpOEQ(d, zero)), FCmpOLE(zero, c)),
          EmitComposeComplex(op, Select(FCmpOEQ(zero, c), one, zero), zero),
          EmitComposeComplex(op, FMul(coeff, cos_q), FMul(coeff, sin_q)));
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

StatusOr<llvm::Value*> ElementalIrEmitter::EmitErfInv(PrimitiveType prim_type,
                                                      llvm::Value* x) {
  if (prim_type != F16 && prim_type != F32 && prim_type != F64) {
    return Unimplemented(
        "Inverse erf is only implemented for element "
        "types F16, F32 and F64.");
  }

  // Upcast half to float.
  if (prim_type == F16) {
    x = b_->CreateFPExt(x, b_->getFloatTy());
  }

  auto get_float = [&](const double f) {
    return llvm::ConstantFP::get(x->getType(), f);
  };
  auto multiply_add = [&](absl::Span<const double> coefficients,
                          llvm::Value* w) {
    llvm::Value* p = get_float(coefficients.front());
    coefficients.remove_prefix(1);
    for (float coefficient : coefficients) {
      p = FAdd(FMul(p, w), get_float(coefficient));
    }
    return p;
  };

  // Approximation for inverse error function from
  //   Giles, M., "Approximating the erfinv function".
  // The approximation has the form (float version):
  //   w = -log((1-x)*(1+x))
  //   if ( w < 5 ) {
  //     w = w - 2.5
  //     p = sum_{i=1}^n lq[i]*w^i
  //   } else {
  //     w = sqrt(w) - 3
  //     p = sum_{i=1}^n gq[i]*w^i
  //   }
  //   return p*x
  llvm::Function* logf_fn = llvm::Intrinsic::getDeclaration(
      module_, llvm::Intrinsic::log, {x->getType()});

  llvm::Value* w = FNeg(Call(
      logf_fn, {FMul(FSub(get_float(1.0f), x), FAdd(get_float(1.0f), x))}));

  llvm::Value* p_addr =
      llvm_ir::EmitAllocaAtFunctionEntry(x->getType(), "p.addr", b_);

  if (prim_type == F16 || prim_type == F32) {
    llvm_ir::LlvmIfData if_data = llvm_ir::EmitIfThenElse(
        FCmpOLT(w, get_float(5.0f)), "w_less_than_five", b_);
    // Handle true BB.
    SetToFirstInsertPoint(if_data.true_block, b_);
    {
      llvm::Value* lw = FSub(w, get_float(2.5f));
      absl::Span<const double> lq{
          2.81022636e-08f,  3.43273939e-07f, -3.5233877e-06f,
          -4.39150654e-06f, 0.00021858087f,  -0.00125372503f,
          -0.00417768164f,  0.246640727f,    1.50140941f};
      llvm::Value* p = multiply_add(lq, lw);
      Store(p, p_addr);
    }

    // Handle false BB.
    SetToFirstInsertPoint(if_data.false_block, b_);
    {
      llvm::Function* sqrtf_fn = llvm::Intrinsic::getDeclaration(
          module_, llvm::Intrinsic::sqrt, {b_->getFloatTy()});

      llvm::Value* gw = FSub(Call(sqrtf_fn, w), get_float(3.0f));
      absl::Span<const double> gq{
          -0.000200214257f, 0.000100950558f, 0.00134934322f,
          -0.00367342844f,  0.00573950773f,  -0.0076224613f,
          0.00943887047f,   1.00167406f,     2.83297682f};
      llvm::Value* p = multiply_add(gq, gw);
      Store(p, p_addr);
    }

    SetToFirstInsertPoint(if_data.after_block, b_);
  } else {
    DCHECK(prim_type == F64);

    llvm_ir::LlvmIfData if_data = llvm_ir::EmitIfThenElse(
        FCmpOLT(w, get_float(6.25)), "w_less_than_6.25", b_);

    SetToFirstInsertPoint(if_data.true_block, b_);
    {
      llvm::Value* lw = FSub(w, get_float(3.125));
      absl::Span<const double> c{
          -3.6444120640178196996e-21, -1.685059138182016589e-19,
          1.2858480715256400167e-18,  1.115787767802518096e-17,
          -1.333171662854620906e-16,  2.0972767875968561637e-17,
          6.6376381343583238325e-15,  -4.0545662729752068639e-14,
          -8.1519341976054721522e-14, 2.6335093153082322977e-12,
          -1.2975133253453532498e-11, -5.4154120542946279317e-11,
          1.051212273321532285e-09,   -4.1126339803469836976e-09,
          -2.9070369957882005086e-08, 4.2347877827932403518e-07,
          -1.3654692000834678645e-06, -1.3882523362786468719e-05,
          0.0001867342080340571352,   -0.00074070253416626697512,
          -0.0060336708714301490533,  0.24015818242558961693,
          1.6536545626831027356};
      llvm::Value* p = multiply_add(c, lw);
      Store(p, p_addr);
    }

    SetToFirstInsertPoint(if_data.false_block, b_);
    llvm_ir::LlvmIfData if_data_second = llvm_ir::EmitIfThenElse(
        FCmpOLT(w, get_float(16.0)), "w_less_than_16", b_);
    SetToFirstInsertPoint(if_data_second.true_block, b_);
    {
      llvm::Function* sqrtf_fn = llvm::Intrinsic::getDeclaration(
          module_, llvm::Intrinsic::sqrt, {b_->getDoubleTy()});

      llvm::Value* gw = FSub(Call(sqrtf_fn, w), get_float(3.25));
      absl::Span<const double> t1{
          2.2137376921775787049e-09,  9.0756561938885390979e-08,
          -2.7517406297064545428e-07, 1.8239629214389227755e-08,
          1.5027403968909827627e-06,  -4.013867526981545969e-06,
          2.9234449089955446044e-06,  1.2475304481671778723e-05,
          -4.7318229009055733981e-05, 6.8284851459573175448e-05,
          2.4031110387097893999e-05,  -0.0003550375203628474796,
          0.00095328937973738049703,  -0.0016882755560235047313,
          0.0024914420961078508066,   -0.0037512085075692412107,
          0.005370914553590063617,    1.0052589676941592334,
          3.0838856104922207635};
      llvm::Value* p = multiply_add(t1, gw);
      Store(p, p_addr);
    }

    SetToFirstInsertPoint(if_data_second.false_block, b_);
    {
      llvm::Function* sqrtf_fn = llvm::Intrinsic::getDeclaration(
          module_, llvm::Intrinsic::sqrt, {b_->getDoubleTy()});

      llvm::Value* gw = FSub(Call(sqrtf_fn, w), get_float(5.0));
      absl::Span<const double> t2{
          -2.7109920616438573243e-11, -2.5556418169965252055e-10,
          1.5076572693500548083e-09,  -3.7894654401267369937e-09,
          7.6157012080783393804e-09,  -1.4960026627149240478e-08,
          2.9147953450901080826e-08,  -6.7711997758452339498e-08,
          2.2900482228026654717e-07,  -9.9298272942317002539e-07,
          4.5260625972231537039e-06,  -1.9681778105531670567e-05,
          7.5995277030017761139e-05,  -0.00021503011930044477347,
          -0.00013871931833623122026, 1.0103004648645343977,
          4.8499064014085844221};
      llvm::Value* p = multiply_add(t2, gw);
      Store(p, p_addr);
    }

    SetToFirstInsertPoint(if_data.after_block, b_);
  }
  llvm::Value* p = Load(p_addr);
  x = FMul(p, x);
  // Trunc back to half if needed.
  if (prim_type == F16) {
    x = b_->CreateFPTrunc(x, b_->getHalfTy());
  }
  return x;
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitErfcInv(PrimitiveType prim_type,
                                                       llvm::Value* value) {
  // Compute erfcinv(value) by calculating erfinv(1.0 - value).
  auto type = llvm_ir::PrimitiveTypeToIrType(prim_type, module_);
  auto one = llvm::ConstantFP::get(type, 1.0);
  return EmitErfInv(prim_type, FSub(one, value));
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
  auto x_squared = FAdd(x, x);
  auto x_squared_over_two = FMul(x_squared, half);
  auto for_small_x = FAdd(x, x_squared_over_two);
  const auto kExponentIsSmallThreshold = 1e-5;
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
  if (hlo->operand(0)->shape().element_type() != F32) {
    return Unimplemented("reduce-precision only implemented for F32");
  }
  return EmitReducePrecisionFloat(x, /*exponent_bits=*/hlo->exponent_bits(),
                                  /*mantissa_bits=*/hlo->mantissa_bits(), b_);
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
    case HloOpcode::kEq:
      return llvm_ir::EmitComparison(llvm::CmpInst::ICMP_EQ, lhs_value,
                                     rhs_value, b_);
    case HloOpcode::kNe:
      return llvm_ir::EmitComparison(llvm::CmpInst::ICMP_NE, lhs_value,
                                     rhs_value, b_);
    case HloOpcode::kLt:
      return llvm_ir::EmitComparison(
          is_signed ? llvm::CmpInst::ICMP_SLT : llvm::CmpInst::ICMP_ULT,
          lhs_value, rhs_value, b_);
    case HloOpcode::kGt:
      return llvm_ir::EmitComparison(
          is_signed ? llvm::CmpInst::ICMP_SGT : llvm::CmpInst::ICMP_UGT,
          lhs_value, rhs_value, b_);
    case HloOpcode::kLe:
      return llvm_ir::EmitComparison(
          is_signed ? llvm::CmpInst::ICMP_SLE : llvm::CmpInst::ICMP_ULE,
          lhs_value, rhs_value, b_);
    case HloOpcode::kGe:
      return llvm_ir::EmitComparison(
          is_signed ? llvm::CmpInst::ICMP_SGE : llvm::CmpInst::ICMP_UGE,
          lhs_value, rhs_value, b_);
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

llvm_ir::IrArray::Index ElementalIrEmitter::ElementwiseSourceIndex(
    const llvm_ir::IrArray::Index& target_index, const HloInstruction& hlo,
    int64 operand_no) {
  CHECK(hlo.IsElementwise())
      << "HLO " << hlo.ToString() << " is not elementwise.";

  const Shape& operand_shape = hlo.operand(operand_no)->shape();
  // If the operand is scalar, the source index is always {}.
  if (ShapeUtil::IsScalar(operand_shape)) {
    return llvm_ir::IrArray::Index(target_index.GetType());
  }

  // If no implicit broadcast is needed for this operand, returns the target
  // index as the source index.
  //
  // `IrArray::Index` may contain a physical linear which we can propagate to
  // our operand only if our layouts match.  "only if" is a bit strong since
  // e.g. we can still forward the linear index if the operand shape is
  // [5,1,1,5]{3,2,1,0} and the HLO shape is[5,1,1,5]{3,1,2,0}, but those cases
  // are probably not worth handling here for now.
  if (ShapeUtil::CompatibleIgnoringElementType(operand_shape, hlo.shape()) &&
      LayoutUtil::Equal(operand_shape.layout(), hlo.shape().layout())) {
    return target_index;
  }

  // If implicit broadcast is needed, the source dimensions that are broadcast
  // have index 0.
  CHECK_EQ(operand_shape.rank(), hlo.shape().rank());
  llvm_ir::IrArray::Index source_index(target_index.GetType());
  for (int64 i = 0; i < hlo.shape().rank(); ++i) {
    if (hlo.shape().dimensions(i) == operand_shape.dimensions(i)) {
      source_index.push_back(target_index[i]);
    } else {
      CHECK_EQ(1, operand_shape.dimensions(i));
      source_index.push_back(target_index.GetConstantWithIndexType(0));
    }
  }
  return source_index;
}

StatusOr<llvm::Value*> ElementalIrEmitter::ConvertValueForDistribution(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& index, llvm::Value* raw_value) {
  TF_ASSIGN_OR_RETURN(llvm::Value * a_or_mean,
                      operand_to_generator.at(hlo->operand(0))(index));
  TF_ASSIGN_OR_RETURN(llvm::Value * b_or_sigma,
                      operand_to_generator.at(hlo->operand(1))(index));
  PrimitiveType elem_prim_ty = hlo->shape().element_type();
  llvm::Type* elem_ir_ty =
      llvm_ir::PrimitiveTypeToIrType(elem_prim_ty, module_);
  llvm::Type* raw_value_ty = raw_value->getType();

  // Convert raw integer to float in range [0, 1) if the element is a float.
  llvm::Value* elem_value = raw_value;
  if (elem_ir_ty->isFloatingPointTy()) {
    unsigned raw_value_size_in_bits = raw_value_ty->getPrimitiveSizeInBits();
    CHECK(raw_value_size_in_bits == 32 || raw_value_size_in_bits == 64);
    // Perform the division using the float type with the same number of bits
    // as the raw value to avoid overflow.
    if (raw_value_size_in_bits == 32) {
      elem_value = UIToFP(elem_value, b_->getFloatTy());
      elem_value = FDiv(elem_value,
                        llvm::ConstantFP::get(b_->getFloatTy(), std::exp2(32)));
    } else {
      elem_value = UIToFP(elem_value, b_->getDoubleTy());
      elem_value = FDiv(
          elem_value, llvm::ConstantFP::get(b_->getDoubleTy(), std::exp2(64)));
    }

    if (elem_ir_ty != elem_value->getType()) {
      elem_value = FPTrunc(elem_value, elem_ir_ty);
    }
  }

  // Convert the value for the requested distribution.
  switch (hlo->random_distribution()) {
    case RNG_UNIFORM: {
      if (elem_ir_ty->isFloatingPointTy()) {
        return FAdd(FMul(FSub(b_or_sigma, a_or_mean), elem_value), a_or_mean);
      } else {
        // To generate a uniform random value in [a, b) from a raw random sample
        // in range [0, 2^N), we let range = b - a and return
        // (a + raw_value % range). If range is not a power of 2, raw values
        // larger than (2^N - 2^N % range) are biased toward results in
        // [a, a + (limit % range)). An unbiased algorithm would need to drop
        // raw values and re-sample, but we don't do this because re-sampling in
        // an efficient way is complex, and it's not clear that users need it.
        // In particular, if one thread in a GPU warp needs to re-sample, we pay
        // the same cost as if the whole warp were to re-sample.  So an
        // efficient re-sampling implementation on GPU would need to do
        // nontrivial work to share entropy between threads in the warp.
        auto range = Sub(b_or_sigma, a_or_mean);
        return Add(a_or_mean, URem(elem_value, range));
      }
    }
    case RNG_NORMAL: {
      TF_ASSIGN_OR_RETURN(
          llvm::Value * r,
          EmitErfcInv(elem_prim_ty, FMul(llvm::ConstantFP::get(elem_ir_ty, 2.0),
                                         elem_value)));
      return FAdd(FMul(r, b_or_sigma), a_or_mean);
    }
    default:
      return InvalidArgument(
          "unhandled distribution %s",
          RandomDistribution_Name(hlo->random_distribution()));
  }
}

namespace {

// Checks that the primitive type is supported by the elemental IR emitter for
// Philox RNG and returns the number of elements in each 128 bit sample of the
// Philox RNG algorithm.
int32 GetNumberOfElementsPerPhiloxRngSample(PrimitiveType elem_prim_ty) {
  // Calculate the number of elements, that is the number of random numbers, in
  // a 128 bit sample.
  switch (elem_prim_ty) {
    case U32:
    case S32:
    case F32:
    // The algorithm uses 32 bits to generate values for F16.
    case F16:
      return 4;
    case U64:
    case S64:
    case F64:
      return 2;
    default:
      // BF16 is converted to F16 by the hlo pass HloElementTypeConverter.
      // Other data types are not supported by XLA random operation.
      LOG(FATAL) << "Unrecognized primitive type for RNG " << elem_prim_ty;
  }
  return 0;
}

// Calculates the four uint32 values for the 128-bit Philox sample.
std::array<llvm::Value*, 4> CalculateSampleValues(
    llvm::Value* sample_idx, llvm::Value* hlo_random_value,
    llvm::Value* global_random_number, llvm::Value* rng_state,
    llvm::IRBuilder<>* b) {
  llvm::Type* index_ty = sample_idx->getType();

  std::array<llvm::Value*, 4> counter_values;

  // Use the sample index to initialize counter[0] and counter[1].
  unsigned index_ty_size_in_bits = index_ty->getPrimitiveSizeInBits();
  CHECK(index_ty_size_in_bits == 32 || index_ty_size_in_bits == 64);
  if (index_ty_size_in_bits == 32) {
    counter_values[0] = sample_idx;
    counter_values[1] = b->getInt32(0);
  } else {
    std::tie(counter_values[0], counter_values[1]) =
        llvm_ir::SplitInt64ToInt32s(b, sample_idx);
  }

  // Xor the global state variable with the global random number seed and use
  // the result to initialize counter[2] and counter[3].
  std::tie(counter_values[2], counter_values[3]) = llvm_ir::SplitInt64ToInt32s(
      b, b->CreateXor(rng_state, global_random_number));

  // The algorithm uses a 64 bit key, which is also interpreted as two uint32
  // values.
  llvm::Value* key_values[2];

  // Use a module random number to initialize the key.
  std::tie(key_values[0], key_values[1]) =
      llvm_ir::SplitInt64ToInt32s(b, hlo_random_value);

  // Prepare the constants used in the Philox RNG Algorithm.
  llvm::Value* philoxW32A = b->getInt32(0x9E3779B9);
  llvm::Value* philoxW32B = b->getInt32(0xBB67AE85);
  llvm::Value* philoxM4xW32A = b->getInt32(0xD2511F53);
  llvm::Value* philoxM4xW32B = b->getInt32(0xCD9E8D57);

  // Compute the 128 bit value for the current sample by repeating the
  // single round computation and key raising computation for ten times.
  for (int round = 0; round < 10; ++round) {
    // A single round of computation of the counter values is as follows:
    //  MultiplyHighLow(kPhiloxM4x32A, counter[0], &lo0, &hi0);
    //  MultiplyHighLow(kPhiloxM4x32B, counter[2], &lo1, &hi1);
    //  counter[0] = hi1 ^ counter[1] ^ key[0];
    //  counter[1] = lo1;
    //  counter[2] = hi0 ^ counter[3] ^ key[1];
    //  counter[3] = lo0;
    llvm::Value* lo0;
    llvm::Value* hi0;
    std::tie(lo0, hi0) =
        llvm_ir::UMulLowHigh32(b, philoxM4xW32A, counter_values[0]);
    llvm::Value* lo1;
    llvm::Value* hi1;
    std::tie(lo1, hi1) =
        llvm_ir::UMulLowHigh32(b, philoxM4xW32B, counter_values[2]);
    counter_values[0] =
        b->CreateXor(hi1, b->CreateXor(counter_values[1], key_values[0]));
    counter_values[1] = lo1;
    counter_values[2] =
        b->CreateXor(hi0, b->CreateXor(counter_values[3], key_values[1]));
    counter_values[3] = lo0;
    key_values[0] = b->CreateAdd(key_values[0], philoxW32A);
    key_values[1] = b->CreateAdd(key_values[1], philoxW32B);
  }

  return counter_values;
}

}  // namespace

// Implements the Philox algorithm to generate random numbers in parallel.
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
//   http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
//
// The paper presents a few variants of the Philox algorithm, we picked the
// 4x32_10 version of the algorithm for the following reasons:
//   . 4x32 uses 32-bit multiplication which is fast on GPUs.
//   . The authors recommend the 10-round variant, and TensorFlow also uses it.
//
// Precondition: the RNG instruction is not fused.
llvm_ir::ElementGenerator ElementalIrEmitter::MakePhiloxRngElementGenerator(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator) {
  VLOG(3) << "Using philox RNG algorithm";
  CHECK(!hlo->IsFused());
  // A random number generated by the per module random number generator.
  // This ensures that each RNG HLO generates a different random sequence.
  llvm::Value* hlo_random_value = b_->getInt64(hlo->GetModule()->RandomNew64());
  // A value specified by the configuration or generated by a global random
  // number generator.
  llvm::Value* global_random_number =
      b_->getInt64(hlo_module_config_.seed() != 0 ? hlo_module_config_.seed()
                                                  : GlobalRandomValue());

  int elems_per_sample =
      GetNumberOfElementsPerPhiloxRngSample(hlo->shape().element_type());

  // Allocate stack storage for the 128 bit sample as four int32.
  llvm::Type* int32_ty = b_->getInt32Ty();
  llvm::Value* sample_address = llvm_ir::EmitAllocaAtFunctionEntryWithCount(
      int32_ty, /*element_count=*/b_->getInt32(4), "sample", b_);

  // Load the global state variable for the Philox RNG algorithm.
  llvm::GlobalVariable* rng_state_ptr =
      llvm_ir::GetOrCreateVariableForPhiloxRngState(module_, b_);
  llvm::Value* rng_state = Load(rng_state_ptr, "rng_state_value");

  // Build and return the elemental IR generator to generate a random value for
  // the element corresponding to the current thread.
  //
  // This elemental IR generator computes one sample with multiple random
  // numbers but only returns one random number. As a result, neighboring
  // threads may calculate the same sample unnecessarily. However, if the
  // kernel containing the RNG hlo is unrolled, LLVM is able to optimize away
  // the duplicated computation of the same sample. In particular, if the unroll
  // factor is a multiplier of elems_per_sample, LLVM is able to completely
  // remove such duplicated computation. If the unroll factor is a non-trivial
  // factor of elems_per_sample, LLVM can only partially remove such duplicated
  // computation.
  return [=](const llvm_ir::IrArray::Index& index) -> StatusOr<llvm::Value*> {
    llvm::Type* index_ty = index.GetType();
    // Calculate the linear element index.
    llvm::Value* elem_idx = index.linear();
    if (elem_idx == nullptr) {
      elem_idx = index.Linearize(AsInt64Slice(hlo->shape().dimensions()), b_);
    }

    // Calculate the index for the 128 bit sample and the offset of the current
    // element within the sample.
    llvm::Value* elems_per_sample_value =
        llvm::ConstantInt::get(index_ty, elems_per_sample);
    llvm::Value* sample_idx = UDiv(elem_idx, elems_per_sample_value);
    llvm::Value* elem_offset = URem(elem_idx, elems_per_sample_value);

    std::array<llvm::Value*, 4> counter_values = CalculateSampleValues(
        sample_idx, hlo_random_value, global_random_number, rng_state, b_);

    // Store the four counter_values into the sample_address alloca so we can
    // load the elem_offset'th one below.
    for (int idx = 0; idx < 4; ++idx) {
      Store(counter_values[idx],
            InBoundsGEP(sample_address, b_->getInt32(idx)));
    }

    llvm::Type* int64_ty = b_->getInt64Ty();
    CHECK(elems_per_sample == 2 || elems_per_sample == 4);
    llvm::Type* raw_value_ty = elems_per_sample == 2 ? int64_ty : int32_ty;
    // Retrieve the raw value for the current element from the current sample.
    llvm::Value* raw_elem_value = Load(
        InBoundsGEP(PointerCast(sample_address, raw_value_ty->getPointerTo()),
                    elem_offset),
        "raw_elem_value");

    return ConvertValueForDistribution(hlo, operand_to_generator, index,
                                       raw_elem_value);
  };
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalSelect(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& index) {
  TF_ASSIGN_OR_RETURN(llvm::Value * pred_value,
                      operand_to_generator.at(hlo->operand(0))(
                          ElementwiseSourceIndex(index, *hlo, 0)));
  TF_ASSIGN_OR_RETURN(llvm::Value * on_true_value,
                      operand_to_generator.at(hlo->operand(1))(
                          ElementwiseSourceIndex(index, *hlo, 1)));
  TF_ASSIGN_OR_RETURN(llvm::Value * on_false_value,
                      operand_to_generator.at(hlo->operand(2))(
                          ElementwiseSourceIndex(index, *hlo, 2)));
  return Select(Trunc(pred_value, b_->getInt1Ty()), on_true_value,
                on_false_value);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitElementalClamp(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator,
    const llvm_ir::IrArray::Index& index) {
  TF_ASSIGN_OR_RETURN(llvm::Value * min_value,
                      operand_to_generator.at(hlo->operand(0))(
                          ElementwiseSourceIndex(index, *hlo, 0)));
  TF_ASSIGN_OR_RETURN(llvm::Value * arg_value,
                      operand_to_generator.at(hlo->operand(1))(
                          ElementwiseSourceIndex(index, *hlo, 1)));
  TF_ASSIGN_OR_RETURN(llvm::Value * max_value,
                      operand_to_generator.at(hlo->operand(2))(
                          ElementwiseSourceIndex(index, *hlo, 2)));
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
    exit_block = init_block->splitBasicBlock(b_->GetInsertPoint(),
                                             AsStringRef(IrName(hlo, "merge")));
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
    auto operand_index = source_index;
    operand_index[concat_dim] = source_index_phis[operand_id];

    // Create the terminator of the block before calling operand generators,
    // because they require non-degenerate basic blocks.
    b_->SetInsertPoint(llvm::BranchInst::Create(
        exit_block, /*InsertAtEnd=*/emit_operand_blocks[operand_id]));
    TF_ASSIGN_OR_RETURN(llvm::Value * value,
                        operand_to_generator.at(operand)(operand_index));
    output->addIncoming(value, b_->GetInsertBlock());
    b_->SetInsertPoint(init_block, saved_insert_point);
  }

  for (int64 operand_idx = 0; operand_idx < hlo->operand_count();
       ++operand_idx) {
    const HloInstruction* operand = hlo->operand(operand_idx);
    auto false_block = llvm_ir::CreateBasicBlock(
        exit_block, StrCat("concat_index_not_from_operand", operand_idx), b_);
    auto concat_dim_size =
        llvm::ConstantInt::get(source_index[concat_dim]->getType(),
                               operand->shape().dimensions(concat_dim));
    int64 operand_id = to_unique_operand_id[operand];
    source_index_phis[operand_id]->addIncoming(source_index[concat_dim],
                                               b_->GetInsertBlock());
    CondBr(ICmpULT(source_index[concat_dim], concat_dim_size),
           emit_operand_blocks[operand_id], false_block);

    // Subtract the size of the concat dimension of the current operand
    // from the source index.
    b_->SetInsertPoint(false_block);
    source_index[concat_dim] = Sub(source_index[concat_dim], concat_dim_size);
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
  llvm_ir::IrArray::Index slice_start_index(index_type, rank);
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

    start_index_value->setName(
        AsStringRef(IrName(hlo, StrCat("start_idx", i))));
    slice_start_index[i] = start_index_value;
  }

  llvm_ir::IrArray::Index input_index(index_type, rank);
  for (int64 i = 0; i < rank; ++i) {
    // Emit IR which computes:
    //   input_index = start_index + offset_index
    input_index[i] = Add(slice_start_index[i], index[i]);
  }
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
  IrArray::Index operand_index(index_type);

  // First copy in the window indices to operand_index. Also collect a mapping
  // from operand dimension to output window dimension. Elided window dimensions
  // map to -1.
  std::vector<int64> operand_to_output_dim(operand_shape.dimensions_size(), -1);
  for (int64 i = 0, e = operand_shape.dimensions_size(), operand_index_dim = 0;
       i < e; i++) {
    if (absl::c_binary_search(dim_numbers.collapsed_slice_dims(), i)) {
      operand_index.push_back(index.GetConstantWithIndexType(0));
    } else {
      int64 output_window_dim = dim_numbers.offset_dims(operand_index_dim++);
      operand_to_output_dim[i] = output_window_dim;
      operand_index.push_back(index[output_window_dim]);
    }
  }

  // This is the index of the index vector in the start_indices tensor.
  IrArray::Index gather_index_index(index_type);
  {
    std::vector<llvm::Value*> gather_index_index_components;
    for (int64 i = 0, e = output_shape.dimensions_size(); i < e; i++) {
      if (!absl::c_binary_search(dim_numbers.offset_dims(), i)) {
        gather_index_index.push_back(index[i]);
      }
    }

    if (gather_index_index.size() != indices_shape.dimensions_size()) {
      gather_index_index.InsertAt(dim_numbers.index_vector_dim(), nullptr);
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

    operand_index[operand_dim] =
        Add(operand_index[operand_dim], gather_dim_component_extended_inbound);
  };

  if (indices_shape.dimensions_size() == dim_numbers.index_vector_dim()) {
    TF_ASSIGN_OR_RETURN(llvm::Value * gather_dim_component,
                        indices_generator(gather_index_index));
    add_to_operand_index(gather_dim_component, 0);
  } else {
    int64 index_vector_size =
        indices_shape.dimensions(dim_numbers.index_vector_dim());
    for (int64 i = 0; i < index_vector_size; i++) {
      gather_index_index[dim_numbers.index_vector_dim()] =
          index.GetConstantWithIndexType(i);
      TF_ASSIGN_OR_RETURN(llvm::Value * gather_dim_component,
                          indices_generator(gather_index_index));
      add_to_operand_index(gather_dim_component, i);
    }
  }
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
  llvm_ir::IrArray::Index slice_start_index(index.GetType(), rank);
  llvm_ir::IrArray::Index slice_limit_index(index.GetType(), rank);
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

    start_index_value->setName(
        AsStringRef(IrName(hlo, StrCat("start_idx", i))));
    slice_start_index[i] = start_index_value;
    slice_limit_index[i] = Add(slice_start_index[i], update_dim_size);

    slice_intersection =
        And(slice_intersection, ICmpSGE(index[i], slice_start_index[i]),
            "slice_intersection");
    slice_intersection =
        And(slice_intersection, ICmpSLT(index[i], slice_limit_index[i]),
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
  llvm_ir::IrArray::Index update_index(index.GetType(), rank);
  for (int64 i = 0; i < rank; ++i) {
    update_index[i] = Sub(index[i], slice_start_index[i]);
  }
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
  auto index = padded_index;
  llvm::Value* in_bounds = b_->getTrue();
  for (size_t i = 0; i < index.size(); ++i) {
    auto index_typed_const = [=](int64 n) {
      return llvm::ConstantInt::get(index[i]->getType(), n);
    };
    const auto& pad_dim = hlo->padding_config().dimensions(i);
    index[i] = Sub(index[i], index_typed_const(pad_dim.edge_padding_low()));
    in_bounds =
        And(in_bounds, ICmpSGE(index[i], index_typed_const(0)), "in_bounds");
    in_bounds = And(
        in_bounds,
        ICmpEQ(
            index_typed_const(0),
            URem(index[i], index_typed_const(pad_dim.interior_padding() + 1))),
        "in_bounds");
    index[i] =
        SDiv(index[i], index_typed_const(pad_dim.interior_padding() + 1));
    in_bounds =
        And(in_bounds,
            ICmpSLT(index[i],
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

  llvm::Type* index_type = dot_result_index[0]->getType();
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

  IrArray::Index lhs_index(index_type), rhs_index(index_type);

  for (int64 i = 0; i < lhs_dims - 1; i++) {
    lhs_index.push_back(dot_result_index[i]);
  }
  lhs_index.InsertAt(lhs_contracting_dim, inner_loop->GetIndVarValue());

  for (int64 i = 0; i < rhs_dims - 1; i++) {
    rhs_index.push_back(dot_result_index[lhs_dims - 1 + i]);
  }
  rhs_index.InsertAt(rhs_contracting_dim, inner_loop->GetIndVarValue());

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
    case HloOpcode::kCopy:
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
    case HloOpcode::kReal:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kTanh:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        TF_ASSIGN_OR_RETURN(llvm::Value * operand_value,
                            operand_to_generator.at(hlo->operand(0))(
                                ElementwiseSourceIndex(index, *hlo, 0)));
        return EmitUnaryOp(hlo, operand_value);
      };
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kAtan2:
    case HloOpcode::kComplex:
    case HloOpcode::kDivide:
    case HloOpcode::kEq:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNe:
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
                            operand_to_generator.at(lhs)(
                                ElementwiseSourceIndex(index, *hlo, 0)));
        TF_ASSIGN_OR_RETURN(llvm::Value * rhs_value,
                            operand_to_generator.at(rhs)(
                                ElementwiseSourceIndex(index, *hlo, 1)));
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
                            operand_to_generator.at(hlo->operand(0))(
                                ElementwiseSourceIndex(index, *hlo, 0)));
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
        auto source_index = target_index;
        for (int64 dim : hlo->dimensions()) {
          source_index[dim] =
              Sub(llvm::ConstantInt::get(target_index[dim]->getType(),
                                         hlo->shape().dimensions(dim) - 1),
                  target_index[dim]);
        }
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
        if (primitive_util::IsIntegralType(component_element_type) ||
            component_element_type == PRED) {
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
            iota_result = EmitF32ToBF16(float_val, b_);
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
            /*shape=*/hlo->shape(), /*starts=*/hlo->slice_starts(),
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
    case HloOpcode::kTranspose:
      return [this, hlo,
              &operand_to_generator](const IrArray::Index& target_index) {
        return operand_to_generator.at(hlo->operand(0))(
            target_index.SourceIndexOfTranspose(
                hlo->shape(), hlo->operand(0)->shape(), hlo->dimensions(), b_));
      };
    case HloOpcode::kRng:
      return MakePhiloxRngElementGenerator(hlo, operand_to_generator);
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
