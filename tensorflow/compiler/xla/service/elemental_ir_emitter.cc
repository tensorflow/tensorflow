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
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "tensorflow/compiler/xla/primitive_util.h"
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
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

using llvm_ir::AsStringRef;
using llvm_ir::IrArray;
using llvm_ir::IrName;
using llvm_ir::SetToFirstInsertPoint;
using tensorflow::strings::StrCat;

StatusOr<llvm::Value*> ElementalIrEmitter::EmitUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) const {
  if (op->opcode() == HloOpcode::kCopy) {
    return operand_value;
  } else if (operand_value->getType()->isIntegerTy()) {
    return EmitIntegerUnaryOp(op, operand_value);
  } else if (ShapeUtil::ElementIsComplex(op->operand(0)->shape())) {
    return EmitComplexUnaryOp(op, operand_value);
  } else {
    return EmitFloatUnaryOp(op, operand_value);
  }
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitIntegerUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) const {
  switch (op->opcode()) {
    case HloOpcode::kConvert: {
      PrimitiveType from_type = op->operand(0)->shape().element_type();
      PrimitiveType to_type = op->shape().element_type();
      CHECK(primitive_util::IsIntegralType(from_type) || from_type == PRED);
      if (from_type == to_type) {
        return operand_value;
      }
      if (primitive_util::IsIntegralType(to_type)) {
        return ir_builder_->CreateIntCast(
            operand_value, llvm_ir::PrimitiveTypeToIrType(to_type, module_),
            primitive_util::IsSignedIntegralType(to_type));
      }
      if (primitive_util::IsFloatingPointType(to_type)) {
        if (primitive_util::IsSignedIntegralType(from_type)) {
          return ir_builder_->CreateSIToFP(
              operand_value, llvm_ir::PrimitiveTypeToIrType(to_type, module_));
        }
        if (primitive_util::IsUnsignedIntegralType(from_type) ||
            from_type == PRED) {
          return ir_builder_->CreateUIToFP(
              operand_value, llvm_ir::PrimitiveTypeToIrType(to_type, module_));
        }
      }
      if (primitive_util::IsComplexType(to_type)) {
        auto to_ir_component_type = llvm_ir::PrimitiveTypeToIrType(
            primitive_util::ComplexComponentType(to_type), module_);
        if (primitive_util::IsSignedIntegralType(from_type)) {
          return EmitComposeComplex(
              op,
              ir_builder_->CreateSIToFP(operand_value, to_ir_component_type),
              nullptr);
        }
        if (primitive_util::IsUnsignedIntegralType(from_type) ||
            from_type == PRED) {
          return EmitComposeComplex(
              op,
              ir_builder_->CreateUIToFP(operand_value, to_ir_component_type),
              nullptr);
        }
      }
      return Unimplemented("conversion from primitive type %s to %s",
                           PrimitiveType_Name(from_type).c_str(),
                           PrimitiveType_Name(to_type).c_str());
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
        return ir_builder_->CreateBitCast(
            operand_value, llvm_ir::PrimitiveTypeToIrType(to_type, module_));
      }
      return InvalidArgument(
          "bitcast conversion from primitive type %s to %s with unequal "
          "bit-widths (%u versus %u) ",
          PrimitiveType_Name(from_type).c_str(),
          PrimitiveType_Name(to_type).c_str(),
          primitive_util::BitWidth(from_type),
          primitive_util::BitWidth(to_type));
    }
    case HloOpcode::kAbs: {
      bool is_signed =
          primitive_util::IsSignedIntegralType(op->shape().element_type());
      if (is_signed) {
        auto type =
            llvm_ir::PrimitiveTypeToIrType(op->shape().element_type(), module_);
        auto zero = llvm::ConstantInt::get(type, 0);
        auto cmp = ir_builder_->CreateICmpSGE(operand_value, zero);
        return ir_builder_->CreateSelect(cmp, operand_value,
                                         ir_builder_->CreateNeg(operand_value));
      } else {
        return operand_value;
      }
    }
    case HloOpcode::kSign: {
      bool is_signed =
          primitive_util::IsSignedIntegralType(op->shape().element_type());
      auto type =
          llvm_ir::PrimitiveTypeToIrType(op->shape().element_type(), module_);
      auto zero = llvm::ConstantInt::get(type, 0);
      auto cmp = ir_builder_->CreateICmpEQ(operand_value, zero);
      if (is_signed) {
        auto ashr = ir_builder_->CreateAShr(operand_value,
                                            type->getIntegerBitWidth() - 1);
        return ir_builder_->CreateSelect(cmp, zero,
                                         ir_builder_->CreateOr(ashr, 1));
      } else {
        return ir_builder_->CreateSelect(cmp, zero,
                                         llvm::ConstantInt::get(type, 1));
      }
    }
    case HloOpcode::kNegate:
      return ir_builder_->CreateNeg(operand_value);
    case HloOpcode::kNot: {
      auto type = op->shape().element_type();
      if (type == PRED) {
        // It is not sufficient to just call CreateNot() here because a PRED
        // is represented as an i8 and the truth value is stored only in the
        // bottom bit.
        return ir_builder_->CreateZExt(
            ir_builder_->CreateNot(ir_builder_->CreateTrunc(
                operand_value, ir_builder_->getInt1Ty())),
            llvm_ir::PrimitiveTypeToIrType(PRED, module_));
      } else if (primitive_util::IsIntegralType(type)) {
        return ir_builder_->CreateNot(operand_value);
      }
      return Unimplemented("unary op Not is not defined for type '%d'", type);
    }
    default:
      return Unimplemented("unary integer op '%s'",
                           HloOpcodeString(op->opcode()).c_str());
  }
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitFloatUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) const {
  switch (op->opcode()) {
    case HloOpcode::kConvert: {
      PrimitiveType from_type = op->operand(0)->shape().element_type();
      PrimitiveType to_type = op->shape().element_type();
      CHECK(primitive_util::IsFloatingPointType(from_type));
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
            ir_builder_->CreateFPCast(
                operand_value,
                llvm_ir::PrimitiveTypeToIrType(to_component_type, module_)),
            nullptr);
      }
      if (primitive_util::IsFloatingPointType(to_type)) {
        return ir_builder_->CreateFPCast(
            operand_value, llvm_ir::PrimitiveTypeToIrType(to_type, module_));
      }
      if (primitive_util::IsSignedIntegralType(to_type)) {
        return ir_builder_->CreateFPToSI(
            operand_value, llvm_ir::PrimitiveTypeToIrType(to_type, module_));
      }
      if (primitive_util::IsUnsignedIntegralType(to_type)) {
        return ir_builder_->CreateFPToUI(
            operand_value, llvm_ir::PrimitiveTypeToIrType(to_type, module_));
      }
      return Unimplemented("unhandled conversion operation: %s => %s",
                           PrimitiveType_Name(from_type).c_str(),
                           PrimitiveType_Name(to_type).c_str());
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
        return ir_builder_->CreateBitCast(
            operand_value, llvm_ir::PrimitiveTypeToIrType(to_type, module_));
      }
      return InvalidArgument(
          "bitcast conversion from primitive type %s to %s with unequal "
          "bit-widths (%u versus %u) ",
          PrimitiveType_Name(from_type).c_str(),
          PrimitiveType_Name(to_type).c_str(),
          primitive_util::BitWidth(from_type),
          primitive_util::BitWidth(to_type));
    }
    case HloOpcode::kExp:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::exp, {operand_value},
                                          {operand_value->getType()},
                                          ir_builder_);
    case HloOpcode::kLog:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::log, {operand_value},
                                          {operand_value->getType()},
                                          ir_builder_);
    case HloOpcode::kCos:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::cos, {operand_value},
                                          {operand_value->getType()},
                                          ir_builder_);
    case HloOpcode::kSin:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::sin, {operand_value},
                                          {operand_value->getType()},
                                          ir_builder_);
    case HloOpcode::kFloor:
      return llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::floor, {operand_value}, {operand_value->getType()},
          ir_builder_);
    case HloOpcode::kCeil:
      return llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::ceil, {operand_value}, {operand_value->getType()},
          ir_builder_);
    case HloOpcode::kAbs:
      return llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::fabs, {operand_value}, {operand_value->getType()},
          ir_builder_);
    case HloOpcode::kRoundNearestAfz:
      return llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::round, {operand_value}, {operand_value->getType()},
          ir_builder_);
    case HloOpcode::kSign: {
      // TODO(b/32151903): Ensure consistent sign behavior for -0.0
      auto type = operand_value->getType();
      auto zero = llvm::ConstantFP::get(type, 0.0);
      auto oeq = ir_builder_->CreateFCmpOEQ(operand_value, zero);
      auto olt = ir_builder_->CreateFCmpOLT(operand_value, zero);
      return ir_builder_->CreateSelect(
          oeq, zero,
          ir_builder_->CreateSelect(olt, llvm::ConstantFP::get(type, -1.0),
                                    llvm::ConstantFP::get(type, 1.0)));
    }
    case HloOpcode::kIsFinite: {
      // (x == x) && abs(x) != inf
      auto type = operand_value->getType();
      auto equal_self =
          ir_builder_->CreateFCmpOEQ(operand_value, operand_value);
      auto abs_value = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::fabs, {operand_value}, {type}, ir_builder_);
      auto infinity = llvm::ConstantFP::getInfinity(type);
      auto not_infinite = ir_builder_->CreateFCmpONE(abs_value, infinity);
      auto result_i1 = ir_builder_->CreateAnd(equal_self, not_infinite);
      return ir_builder_->CreateZExt(
          result_i1, llvm_ir::PrimitiveTypeToIrType(PRED, module_));
    }
    case HloOpcode::kNegate:
      return ir_builder_->CreateFNeg(operand_value);
    default:
      return Unimplemented("unary floating-point op '%s'",
                           HloOpcodeString(op->opcode()).c_str());
  }
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) const {
  switch (op->opcode()) {
    // TODO(b/65209142): Angle/Log require atan2.
    // case HloOpcode::kLog:  // log(a+bi) = .5*log(a^2+b^2) + i*atan2(b, a)
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
          op,
          ir_builder_->CreateFPCast(EmitExtractReal(operand_value),
                                    to_ir_component_type),
          ir_builder_->CreateFPCast(EmitExtractImag(operand_value),
                                    to_ir_component_type));
    }
    case HloOpcode::kExp: {
      // e^(a+bi) = e^a*(cos(b)+sin(b)i)
      auto exp_a = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::exp, {EmitExtractReal(operand_value)},
          {EmitExtractReal(operand_value)->getType()}, ir_builder_);
      auto cos_b = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::cos, {EmitExtractImag(operand_value)},
          {EmitExtractImag(operand_value)->getType()}, ir_builder_);
      auto sin_b = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::sin, {EmitExtractImag(operand_value)},
          {EmitExtractImag(operand_value)->getType()}, ir_builder_);
      return EmitComposeComplex(op, ir_builder_->CreateFMul(exp_a, cos_b),
                                ir_builder_->CreateFMul(exp_a, sin_b));
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
      auto exp_b = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::exp, {b},
                                                {type}, ir_builder_);
      auto half_exp_b =
          ir_builder_->CreateFMul(llvm::ConstantFP::get(type, 0.5), exp_b);
      auto half_exp_neg_b =
          ir_builder_->CreateFDiv(llvm::ConstantFP::get(type, 0.5), exp_b);
      auto cos_a = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::cos, {a},
                                                {type}, ir_builder_);
      auto sin_a = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::sin, {a},
                                                {type}, ir_builder_);
      return EmitComposeComplex(
          op,
          ir_builder_->CreateFMul(
              cos_a, ir_builder_->CreateFAdd(half_exp_neg_b, half_exp_b)),
          ir_builder_->CreateFMul(
              sin_a, ir_builder_->CreateFSub(half_exp_neg_b, half_exp_b)));
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
      auto exp_b = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::exp, {b},
                                                {type}, ir_builder_);
      auto half_exp_b =
          ir_builder_->CreateFMul(llvm::ConstantFP::get(type, 0.5), exp_b);
      auto half_exp_neg_b =
          ir_builder_->CreateFDiv(llvm::ConstantFP::get(type, 0.5), exp_b);
      auto cos_a = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::cos, {a},
                                                {type}, ir_builder_);
      auto sin_a = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::sin, {a},
                                                {type}, ir_builder_);
      return EmitComposeComplex(
          op,
          ir_builder_->CreateFMul(
              sin_a, ir_builder_->CreateFAdd(half_exp_b, half_exp_neg_b)),
          ir_builder_->CreateFMul(
              cos_a, ir_builder_->CreateFSub(half_exp_b, half_exp_neg_b)));
    }
    case HloOpcode::kAbs: {
      auto sum_sq = ir_builder_->CreateFAdd(
          ir_builder_->CreateFMul(EmitExtractReal(operand_value),
                                  EmitExtractReal(operand_value)),
          ir_builder_->CreateFMul(EmitExtractImag(operand_value),
                                  EmitExtractImag(operand_value)));
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::sqrt, {sum_sq},
                                          {sum_sq->getType()}, ir_builder_);
    }
    case HloOpcode::kSign: {  // Sign(c) = c / |c|
      auto sum_sq = ir_builder_->CreateFAdd(
          ir_builder_->CreateFMul(EmitExtractReal(operand_value),
                                  EmitExtractReal(operand_value)),
          ir_builder_->CreateFMul(EmitExtractImag(operand_value),
                                  EmitExtractImag(operand_value)));
      auto cplx_abs = llvm_ir::EmitCallToIntrinsic(
          llvm::Intrinsic::sqrt, {sum_sq}, {sum_sq->getType()}, ir_builder_);
      auto type = cplx_abs->getType();
      auto zero = llvm::ConstantFP::get(type, 0.0);
      auto oeq = ir_builder_->CreateFCmpOEQ(cplx_abs, zero);
      return ir_builder_->CreateSelect(
          oeq, EmitComposeComplex(op, zero, zero),
          EmitComposeComplex(
              op,
              ir_builder_->CreateFDiv(EmitExtractReal(operand_value), cplx_abs),
              ir_builder_->CreateFDiv(EmitExtractImag(operand_value),
                                      cplx_abs)));
    }
    case HloOpcode::kNegate:
      return EmitComposeComplex(
          op, ir_builder_->CreateFNeg(EmitExtractReal(operand_value)),
          ir_builder_->CreateFNeg(EmitExtractImag(operand_value)));
    case HloOpcode::kReal:
      return EmitExtractReal(operand_value);
    case HloOpcode::kImag:
      return EmitExtractImag(operand_value);
    default:
      return Unimplemented("unary complex op '%s'",
                           HloOpcodeString(op->opcode()).c_str());
  }
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value,
    llvm::Value* rhs_value) const {
  PrimitiveType operand_type = op->operand(0)->shape().element_type();
  if (lhs_value->getType()->isIntegerTy()) {
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
    const HloInstruction* op, llvm::Value* lhs_value,
    llvm::Value* rhs_value) const {
  switch (op->opcode()) {
    // case HloOpcode::kAtan2:  // TODO(b/65209142): CPU atan2 support
    case HloOpcode::kComplex:
      return EmitComposeComplex(op, lhs_value, rhs_value);
    case HloOpcode::kAdd:
      return ir_builder_->CreateFAdd(lhs_value, rhs_value);
    case HloOpcode::kSubtract:
      return ir_builder_->CreateFSub(lhs_value, rhs_value);
    case HloOpcode::kMultiply:
      return ir_builder_->CreateFMul(lhs_value, rhs_value);
    case HloOpcode::kDivide:
      return ir_builder_->CreateFDiv(lhs_value, rhs_value);
    case HloOpcode::kRemainder:
      return ir_builder_->CreateFRem(lhs_value, rhs_value);
    // LLVM comparisons can be "unordered" (U) or "ordered" (O) -- ordered
    // comparisons always return false when one of the operands is NaN, whereas
    // unordered comparisons return true.
    //
    // We use ordered comparisons for everything except kNe, where we use an
    // unordered comparison.  This makes x != y equivalent to !(x == y), and
    // matches C++'s semantics.
    case HloOpcode::kEq:
      return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OEQ, lhs_value,
                                     rhs_value, ir_builder_);
    case HloOpcode::kNe:
      return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_UNE, lhs_value,
                                     rhs_value, ir_builder_);
    case HloOpcode::kLt:
      return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OLT, lhs_value,
                                     rhs_value, ir_builder_);
    case HloOpcode::kGt:
      return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OGT, lhs_value,
                                     rhs_value, ir_builder_);
    case HloOpcode::kLe:
      return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OLE, lhs_value,
                                     rhs_value, ir_builder_);
    case HloOpcode::kGe:
      return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OGE, lhs_value,
                                     rhs_value, ir_builder_);

    case HloOpcode::kMaximum:
      return EmitFloatMax(lhs_value, rhs_value);
    case HloOpcode::kMinimum:
      return EmitFloatMin(lhs_value, rhs_value);
    case HloOpcode::kPower:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::pow,
                                          {lhs_value, rhs_value},
                                          {lhs_value->getType()}, ir_builder_);

    default:
      return Unimplemented("binary floating point op '%s'",
                           HloOpcodeString(op->opcode()).c_str());
  }
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitComplexBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value,
    llvm::Value* rhs_value) const {
  switch (op->opcode()) {
    case HloOpcode::kAdd:
      return EmitComposeComplex(
          op,
          ir_builder_->CreateFAdd(EmitExtractReal(lhs_value),
                                  EmitExtractReal(rhs_value)),
          ir_builder_->CreateFAdd(EmitExtractImag(lhs_value),
                                  EmitExtractImag(rhs_value)));
    case HloOpcode::kSubtract:
      return EmitComposeComplex(
          op,
          ir_builder_->CreateFSub(EmitExtractReal(lhs_value),
                                  EmitExtractReal(rhs_value)),
          ir_builder_->CreateFSub(EmitExtractImag(lhs_value),
                                  EmitExtractImag(rhs_value)));
    case HloOpcode::kMultiply:
      return EmitComposeComplex(
          op,
          ir_builder_->CreateFSub(
              ir_builder_->CreateFMul(EmitExtractReal(lhs_value),
                                      EmitExtractReal(rhs_value)),
              ir_builder_->CreateFMul(EmitExtractImag(lhs_value),
                                      EmitExtractImag(rhs_value))),
          ir_builder_->CreateFAdd(
              ir_builder_->CreateFMul(EmitExtractReal(lhs_value),
                                      EmitExtractImag(rhs_value)),
              ir_builder_->CreateFMul(EmitExtractImag(lhs_value),
                                      EmitExtractReal(rhs_value))));
    case HloOpcode::kDivide: {
      // (a+bi) / (c+di) = ((a+bi)(c-di)) / ((c+di)(c-di))
      // = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
      auto rhs_sum_sq = ir_builder_->CreateFAdd(
          ir_builder_->CreateFMul(EmitExtractReal(rhs_value),
                                  EmitExtractReal(rhs_value)),
          ir_builder_->CreateFMul(EmitExtractImag(rhs_value),
                                  EmitExtractImag(rhs_value)));
      auto type = rhs_sum_sq->getType();
      auto zero = llvm::ConstantFP::get(type, 0.0);
      auto oeq = ir_builder_->CreateFCmpOEQ(rhs_sum_sq, zero);
      auto real_inf_or_nan =
          ir_builder_->CreateFDiv(EmitExtractReal(lhs_value), zero);
      auto imag_inf_or_nan =
          ir_builder_->CreateFDiv(EmitExtractImag(lhs_value), zero);
      return ir_builder_->CreateSelect(
          oeq, EmitComposeComplex(op, real_inf_or_nan, imag_inf_or_nan),
          EmitComposeComplex(
              op,
              ir_builder_->CreateFDiv(
                  ir_builder_->CreateFAdd(
                      ir_builder_->CreateFMul(EmitExtractReal(lhs_value),
                                              EmitExtractReal(rhs_value)),
                      ir_builder_->CreateFMul(EmitExtractImag(lhs_value),
                                              EmitExtractImag(rhs_value))),
                  rhs_sum_sq),
              ir_builder_->CreateFDiv(
                  ir_builder_->CreateFSub(
                      ir_builder_->CreateFMul(EmitExtractImag(lhs_value),
                                              EmitExtractReal(rhs_value)),
                      ir_builder_->CreateFMul(EmitExtractReal(lhs_value),
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
      return ir_builder_->CreateAnd(
          llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OEQ,
                                  EmitExtractReal(lhs_value),
                                  EmitExtractReal(rhs_value), ir_builder_),
          llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OEQ,
                                  EmitExtractImag(lhs_value),
                                  EmitExtractImag(rhs_value), ir_builder_));
    case HloOpcode::kNe:
      return ir_builder_->CreateOr(
          llvm_ir::EmitComparison(llvm::CmpInst::FCMP_UNE,
                                  EmitExtractReal(lhs_value),
                                  EmitExtractReal(rhs_value), ir_builder_),
          llvm_ir::EmitComparison(llvm::CmpInst::FCMP_UNE,
                                  EmitExtractImag(lhs_value),
                                  EmitExtractImag(rhs_value), ir_builder_));

    // TODO(b/65209142): requires arg(z) -> requires atan|atan2 intrinsic
    // case HloOpcode::kPower:
    // // (a+bi)^(c+di) = exp(i(c+di)*arg(a+bi)) * (a*a+b*b)^(c/2+di/2)
    default:
      return Unimplemented("binary complex op '%s'",
                           HloOpcodeString(op->opcode()).c_str());
  }
}

llvm::Value* ElementalIrEmitter::EmitFloatMax(llvm::Value* lhs_value,
                                              llvm::Value* rhs_value) const {
  return llvm_ir::EmitFloatMax(lhs_value, rhs_value, ir_builder_);
}

llvm::Value* ElementalIrEmitter::EmitFloatMin(llvm::Value* lhs_value,
                                              llvm::Value* rhs_value) const {
  return llvm_ir::EmitFloatMin(lhs_value, rhs_value, ir_builder_);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitErfInv(PrimitiveType prim_type,
                                                      llvm::Value* x) const {
  if (prim_type != F32) {
    return Unimplemented("inverse erf only implemented for F32 (b/34339814)");
  }
  auto getFloat = [&](const float f) {
    return llvm::ConstantFP::get(ir_builder_->getFloatTy(), f);
  };
  auto multiply_add = [&](tensorflow::gtl::ArraySlice<float> coefficients,
                          llvm::Value* w) {
    llvm::Value* p = getFloat(coefficients.front());
    coefficients.pop_front();
    for (float coefficient : coefficients) {
      p = ir_builder_->CreateFAdd(ir_builder_->CreateFMul(p, w),
                                  getFloat(coefficient));
    }
    return p;
  };

  // Approximation for inverse error function from
  //   Giles, M., "Approximating the erfinv function".
  // The approximation has the form:
  //   w = log((1-x)*(1+x))
  //   if ( w < 5 ) {
  //     w = w - 2.5
  //     p = sum_{i=1}^n lq[i]*w^i
  //   } else {
  //     w = sqrt(w) - 3
  //     p = sum_{i=1}^n gq[i]*w^i
  //   }
  //   return p*x
  llvm::Function* logf_fn = llvm::Intrinsic::getDeclaration(
      module_, llvm::Intrinsic::log, {ir_builder_->getFloatTy()});

  llvm::Value* w = ir_builder_->CreateFNeg(ir_builder_->CreateCall(
      logf_fn,
      {ir_builder_->CreateFMul(ir_builder_->CreateFSub(getFloat(1.0f), x),
                               ir_builder_->CreateFAdd(getFloat(1.0f), x))}));

  llvm::Value* p_addr = llvm_ir::EmitAllocaAtFunctionEntry(
      ir_builder_->getFloatTy(), "p.addr", ir_builder_);

  llvm_ir::LlvmIfData if_data =
      llvm_ir::EmitIfThenElse(ir_builder_->CreateFCmpOLT(w, getFloat(5.0f)),
                              "w_less_than_five", ir_builder_);
  // Handle true BB.
  SetToFirstInsertPoint(if_data.true_block, ir_builder_);
  {
    llvm::Value* lw = ir_builder_->CreateFSub(w, getFloat(2.5f));
    tensorflow::gtl::ArraySlice<float> lq{
        2.81022636e-08f,  3.43273939e-07f, -3.5233877e-06f,
        -4.39150654e-06f, 0.00021858087f,  -0.00125372503f,
        -0.00417768164f,  0.246640727f,    1.50140941f};
    llvm::Value* p = multiply_add(lq, lw);
    ir_builder_->CreateStore(p, p_addr);
  }

  // Handle false BB.
  SetToFirstInsertPoint(if_data.false_block, ir_builder_);
  {
    llvm::Function* sqrtf_fn = llvm::Intrinsic::getDeclaration(
        module_, llvm::Intrinsic::sqrt, {ir_builder_->getFloatTy()});

    llvm::Value* gw = ir_builder_->CreateFSub(
        ir_builder_->CreateCall(sqrtf_fn, {w}), getFloat(3.0f));
    tensorflow::gtl::ArraySlice<float> gq{
        -0.000200214257f, 0.000100950558f, 0.00134934322f,
        -0.00367342844f,  0.00573950773f,  -0.0076224613f,
        0.00943887047f,   1.00167406f,     2.83297682f};
    llvm::Value* p = multiply_add(gq, gw);
    ir_builder_->CreateStore(p, p_addr);
  }

  SetToFirstInsertPoint(if_data.after_block, ir_builder_);
  llvm::Value* p = ir_builder_->CreateLoad(p_addr);
  return ir_builder_->CreateFMul(p, x);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitErfcInv(
    PrimitiveType prim_type, llvm::Value* value) const {
  // Compute erfcinv(value) by calculating erfinv(1.0 - value).
  auto type = llvm_ir::PrimitiveTypeToIrType(prim_type, module_);
  auto one = llvm::ConstantFP::get(type, 1.0);
  return EmitErfInv(prim_type, ir_builder_->CreateFSub(one, value));
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitReducePrecision(
    const HloInstruction* hlo, llvm::Value* x) const {
  if (hlo->operand(0)->shape().element_type() != F32) {
    return Unimplemented("reduce-precision only implemented for F32");
  }

  // Integer and float types for casting and constant generation.
  llvm::Type* float_type = x->getType();
  llvm::IntegerType* int_type = ir_builder_->getInt32Ty();

  // Cast the input value to an integer for bitwise manipulation.
  llvm::Value* x_as_int = ir_builder_->CreateBitCast(x, int_type);

  if (hlo->mantissa_bits() < 23) {
    // Last remaining mantissa bit.
    const uint32_t last_mantissa_bit_mask = 1u << (23 - hlo->mantissa_bits());

    // Compute rounding bias for round-to-nearest with ties to even.  This is
    // equal to a base value of 0111... plus one bit if the last remaining
    // mantissa bit is 1.
    const uint32_t base_rounding_bias = (last_mantissa_bit_mask >> 1) - 1;
    llvm::Value* x_last_mantissa_bit = ir_builder_->CreateLShr(
        ir_builder_->CreateAnd(
            x_as_int, llvm::ConstantInt::get(int_type, last_mantissa_bit_mask)),
        (23 - hlo->mantissa_bits()));
    llvm::Value* x_rounding_bias = ir_builder_->CreateAdd(
        x_last_mantissa_bit,
        llvm::ConstantInt::get(int_type, base_rounding_bias));

    // Add rounding bias, and mask out truncated bits.  Note that the case
    // where adding the rounding bias overflows into the exponent bits is
    // correct; the non-masked mantissa bits will all be zero, and the
    // exponent will be incremented by one.
    const uint32_t truncation_mask = ~(last_mantissa_bit_mask - 1);
    x_as_int = ir_builder_->CreateAdd(x_as_int, x_rounding_bias);
    x_as_int = ir_builder_->CreateAnd(
        x_as_int, llvm::ConstantInt::get(int_type, truncation_mask));
  }

  if (hlo->exponent_bits() < 8) {
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
    const uint32_t reduced_exponent_bias =
        (1 << (hlo->exponent_bits() - 1)) - 1;
    const uint32_t reduced_max_exponent =
        f32_exponent_bias + reduced_exponent_bias;
    const uint32_t reduced_min_exponent =
        f32_exponent_bias - reduced_exponent_bias;

    // Do we overflow or underflow?
    llvm::Value* x_exponent = ir_builder_->CreateAnd(
        x_as_int, llvm::ConstantInt::get(int_type, f32_exp_bits_mask));
    llvm::Value* x_overflows = ir_builder_->CreateICmpUGT(
        x_exponent,
        llvm::ConstantInt::get(int_type, reduced_max_exponent << 23));
    llvm::Value* x_underflows = ir_builder_->CreateICmpULE(
        x_exponent,
        llvm::ConstantInt::get(int_type, reduced_min_exponent << 23));

    // Compute appropriately-signed values of zero and infinity.
    llvm::Value* x_signed_zero = ir_builder_->CreateAnd(
        x_as_int, llvm::ConstantInt::get(int_type, f32_sign_bit_mask));
    llvm::Value* x_signed_inf = ir_builder_->CreateOr(
        x_signed_zero, llvm::ConstantInt::get(int_type, f32_exp_bits_mask));

    // Force to zero or infinity if overflow or underflow.  (Note that this
    // truncates all denormal values to zero, rather than rounding them.)
    x_as_int = ir_builder_->CreateSelect(x_overflows, x_signed_inf, x_as_int);
    x_as_int = ir_builder_->CreateSelect(x_underflows, x_signed_zero, x_as_int);
  }

  // Cast the result back to a floating-point type.
  llvm::Value* result = ir_builder_->CreateBitCast(x_as_int, float_type);

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
  if (!ir_builder_->getFastMathFlags().noNaNs()) {
    llvm::Value* x_is_nan = ir_builder_->CreateFCmpUNO(x, x);

    if (hlo->mantissa_bits() > 0) {
      result = ir_builder_->CreateSelect(x_is_nan, x, result);
    } else {
      result = ir_builder_->CreateSelect(
          x_is_nan, llvm::ConstantFP::getInfinity(float_type), result);
    }
  }
  return result;
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitIntegerBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value,
    bool is_signed) const {
  switch (op->opcode()) {
    // TODO(jingyue): add the "nsw" attribute for signed types.
    case HloOpcode::kAdd:
      return ir_builder_->CreateAdd(lhs_value, rhs_value);
    case HloOpcode::kSubtract:
      return ir_builder_->CreateSub(lhs_value, rhs_value);
    case HloOpcode::kMultiply:
      return ir_builder_->CreateMul(lhs_value, rhs_value);
    case HloOpcode::kDivide:
      return is_signed ? ir_builder_->CreateSDiv(lhs_value, rhs_value)
                       : ir_builder_->CreateUDiv(lhs_value, rhs_value);
    case HloOpcode::kRemainder:
      return is_signed ? ir_builder_->CreateSRem(lhs_value, rhs_value)
                       : ir_builder_->CreateURem(lhs_value, rhs_value);
    case HloOpcode::kEq:
      return llvm_ir::EmitComparison(llvm::CmpInst::ICMP_EQ, lhs_value,
                                     rhs_value, ir_builder_);
    case HloOpcode::kNe:
      return llvm_ir::EmitComparison(llvm::CmpInst::ICMP_NE, lhs_value,
                                     rhs_value, ir_builder_);
    case HloOpcode::kLt:
      return llvm_ir::EmitComparison(
          is_signed ? llvm::CmpInst::ICMP_SLT : llvm::CmpInst::ICMP_ULT,
          lhs_value, rhs_value, ir_builder_);
    case HloOpcode::kGt:
      return llvm_ir::EmitComparison(
          is_signed ? llvm::CmpInst::ICMP_SGT : llvm::CmpInst::ICMP_UGT,
          lhs_value, rhs_value, ir_builder_);
    case HloOpcode::kLe:
      return llvm_ir::EmitComparison(
          is_signed ? llvm::CmpInst::ICMP_SLE : llvm::CmpInst::ICMP_ULE,
          lhs_value, rhs_value, ir_builder_);
    case HloOpcode::kGe:
      return llvm_ir::EmitComparison(
          is_signed ? llvm::CmpInst::ICMP_SGE : llvm::CmpInst::ICMP_UGE,
          lhs_value, rhs_value, ir_builder_);
    case HloOpcode::kMinimum:
      return ir_builder_->CreateSelect(
          ir_builder_->CreateICmp(
              is_signed ? llvm::ICmpInst::ICMP_SLE : llvm::ICmpInst::ICMP_ULE,
              lhs_value, rhs_value),
          lhs_value, rhs_value);
    case HloOpcode::kMaximum:
      return ir_builder_->CreateSelect(
          ir_builder_->CreateICmp(
              is_signed ? llvm::ICmpInst::ICMP_SGE : llvm::ICmpInst::ICMP_UGE,
              lhs_value, rhs_value),
          lhs_value, rhs_value);
    case HloOpcode::kAnd:
      return ir_builder_->CreateAnd(lhs_value, rhs_value);
    case HloOpcode::kOr:
      return ir_builder_->CreateOr(lhs_value, rhs_value);
    case HloOpcode::kShiftLeft:
      return ir_builder_->CreateShl(lhs_value, rhs_value);
    case HloOpcode::kShiftRightArithmetic:
      return ir_builder_->CreateAShr(lhs_value, rhs_value);
    case HloOpcode::kShiftRightLogical:
      return ir_builder_->CreateLShr(lhs_value, rhs_value);
    default:
      return Unimplemented("binary integer op '%s'",
                           HloOpcodeString(op->opcode()).c_str());
  }
}

llvm_ir::IrArray::Index ElementalIrEmitter::ElementwiseSourceIndex(
    const llvm_ir::IrArray::Index& target_index, const HloInstruction& hlo,
    int64 operand_no) const {
  CHECK(hlo.IsElementwise())
      << "HLO " << hlo.ToString() << " is not elementwise.";

  const Shape& operand_shape = hlo.operand(operand_no)->shape();
  // If the operand is scalar, the source index is always {}.
  if (ShapeUtil::IsScalar(operand_shape)) {
    return llvm_ir::IrArray::Index();
  }

  // If no implicit broadcast is needed for this operand, returns the target
  // index as the source index.
  if (ShapeUtil::Compatible(operand_shape, hlo.shape())) {
    return target_index;
  }

  // If implicit broadcast is needed, the source dimensions that are broadcast
  // have index 0.
  CHECK_EQ(ShapeUtil::Rank(operand_shape), ShapeUtil::Rank(hlo.shape()));
  llvm_ir::IrArray::Index source_index;
  for (int64 i = 0; i < ShapeUtil::Rank(hlo.shape()); ++i) {
    if (hlo.shape().dimensions(i) == operand_shape.dimensions(i)) {
      source_index.push_back(target_index[i]);
    } else {
      CHECK_EQ(1, operand_shape.dimensions(i));
      source_index.push_back(ir_builder_->getInt64(0));
    }
  }
  return source_index;
}

llvm_ir::ElementGenerator ElementalIrEmitter::MakeRngElementGenerator(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator)
    const {
  PrimitiveType param_prim_type = hlo->operand(0)->shape().element_type();
  llvm::Type* param_ir_type =
      llvm_ir::PrimitiveTypeToIrType(param_prim_type, module_);

  // Same values as PCG library
  // https://github.com/imneme/pcg-c/blob/master/include/pcg_variants.h
  llvm::Value* multiplier = ir_builder_->getInt(
      llvm::APInt(128, {0x4385DF649FCCF645, 0x2360ED051FC65DA4}));
  llvm::Value* increment = ir_builder_->getInt(
      llvm::APInt(128, {0x14057B7EF767814F, 0x5851F42D4C957F2D}));

  auto random_value = [hlo]() {
    const HloModule* module =
        hlo->IsFused() ? hlo->parent()->FusionInstruction()->parent()->parent()
                       : hlo->parent()->parent();
    return module->RandomNew64();
  };

  // Seed each RNG emitter with a new 64-bit seed from the HloModule. If the
  // compilation order is deterministic (i.e., RandomNew64 invocation order is
  // deterministic), then the order of RNG is deterministic for a given seed and
  // hence tests will be deterministic.
  // If the user provides a global seed instruction then we only use 64-bits of
  // the host's random number generator to seed the 128 bit value with the other
  // 64-bits is due to a user specified global seed instruction.
  // Create a GlobalVariable to maintain state between invocations. There is a
  // bug in NVPTX with GlobalVariable and 128 bit values, so using 2 64-bit
  // values.
  llvm::GlobalVariable* state_ptr0 = new llvm::GlobalVariable(
      /*M=*/*module_,
      /*Ty=*/ir_builder_->getInt64Ty(),
      /*isConstant=*/false,
      /*Linkage=*/llvm::GlobalValue::PrivateLinkage,
      /*Initializer=*/ir_builder_->getInt64(random_value()),
      /*Name=*/"state_ptr0");
  uint64 graph_seed = hlo_module_config_.seed() != 0 ? hlo_module_config_.seed()
                                                     : random_value();
  llvm::GlobalVariable* state_ptr1 = new llvm::GlobalVariable(
      /*M=*/*module_,
      /*Ty=*/ir_builder_->getInt64Ty(),
      /*isConstant=*/false,
      /*Linkage=*/llvm::GlobalValue::PrivateLinkage,
      /*Initializer=*/ir_builder_->getInt64(graph_seed),
      /*Name=*/"state_ptr1");

  // We want each thread to use its own stream, so we modify the increment per
  // thread. We want the increment to remain odd, so we shift the thread id left
  // 1 and add it to the increment.
  increment = ir_builder_->CreateAdd(increment,
                                     ir_builder_->CreateShl(EmitThreadId(), 1));

  // PCG-XSL-RR algorithm
  // http://www.pcg-random.org/pdf/toms-oneill-pcg-family-v1.02.pdf
  //   state = multiplier * state + increment
  //   return uint64_t(state ^ (state >> 64))) >>> (state >> 122)
  // where ">>>" is bitwise rotation
  auto get_next_i64 = [=]() {
    llvm::Value* state0 = ir_builder_->CreateZExtOrTrunc(
        ir_builder_->CreateLoad(state_ptr0, "state0"),
        ir_builder_->getInt128Ty());
    llvm::Value* state1 = ir_builder_->CreateShl(
        ir_builder_->CreateZExtOrTrunc(
            ir_builder_->CreateLoad(state_ptr1, "state1"),
            ir_builder_->getInt128Ty()),
        64);
    llvm::Value* state = ir_builder_->CreateOr(state0, state1);
    llvm::Value* updated = ir_builder_->CreateAdd(
        ir_builder_->CreateMul(state, multiplier), increment);
    ir_builder_->CreateStore(
        ir_builder_->CreateTrunc(updated, ir_builder_->getInt64Ty()),
        state_ptr0);
    ir_builder_->CreateStore(
        ir_builder_->CreateTrunc(ir_builder_->CreateLShr(updated, 64),
                                 ir_builder_->getInt64Ty()),
        state_ptr1);

    return llvm_ir::CreateRor(
        ir_builder_->CreateTrunc(
            ir_builder_->CreateXor(state, ir_builder_->CreateLShr(state, 64)),
            ir_builder_->getInt64Ty()),
        ir_builder_->CreateTrunc(ir_builder_->CreateLShr(state, 122),
                                 ir_builder_->getInt64Ty()),
        ir_builder_);
  };

  auto get_next_uniform_float = [=]() {
    return ir_builder_->CreateFDiv(
        ir_builder_->CreateUIToFP(get_next_i64(), param_ir_type),
        llvm::ConstantFP::get(param_ir_type, 0x1p64));
  };

  return [=](const llvm_ir::IrArray::Index& index) -> StatusOr<llvm::Value*> {
    switch (hlo->random_distribution()) {
      case RNG_UNIFORM: {
        TF_ASSIGN_OR_RETURN(llvm::Value * p,
                            operand_to_generator.at(hlo->operand(0))(index));
        TF_ASSIGN_OR_RETURN(llvm::Value * q,
                            operand_to_generator.at(hlo->operand(1))(index));
        if (primitive_util::IsFloatingPointType(param_prim_type)) {
          return ir_builder_->CreateFAdd(
              ir_builder_->CreateFMul(ir_builder_->CreateFSub(q, p),
                                      get_next_uniform_float()),
              p);
        } else {
          auto r = ir_builder_->CreateSub(q, p);
          auto leading_zeros = llvm_ir::EmitCallToIntrinsic(
              llvm::Intrinsic::ctlz, {r, ir_builder_->getInt1(true)},
              {param_ir_type}, ir_builder_);
          auto in_block = ir_builder_->GetInsertBlock();

          // A terminator should be present iff we're emitting code
          // into the middle (as opposed to the end) of a basic block.
          CHECK_EQ(ir_builder_->GetInsertPoint() == in_block->end(),
                   in_block->getTerminator() == nullptr);

          llvm::BasicBlock* body_block;
          llvm::BasicBlock* out_block;

          if (ir_builder_->GetInsertPoint() == in_block->end()) {
            body_block = llvm_ir::CreateBasicBlock(
                nullptr, IrName(hlo, "rng_body"), ir_builder_);
            out_block = llvm_ir::CreateBasicBlock(
                nullptr, IrName(hlo, "rng_out"), ir_builder_);
            llvm::BranchInst::Create(body_block, in_block);
          } else {
            body_block = in_block->splitBasicBlock(
                ir_builder_->GetInsertPoint(), "rng_body");
            out_block = body_block->splitBasicBlock(
                ir_builder_->GetInsertPoint(), "rng_out");
            body_block->getTerminator()->eraseFromParent();
          }

          SetToFirstInsertPoint(body_block, ir_builder_);
          auto random = ir_builder_->CreateAnd(
              ir_builder_->CreateZExtOrTrunc(get_next_i64(), param_ir_type),
              ir_builder_->CreateLShr(llvm::ConstantInt::get(param_ir_type, ~0),
                                      leading_zeros));
          llvm::BranchInst::Create(out_block, body_block,
                                   ir_builder_->CreateICmpULT(random, r),
                                   body_block);
          SetToFirstInsertPoint(out_block, ir_builder_);
          return ir_builder_->CreateAdd(
              p, ir_builder_->CreateSelect(
                     ir_builder_->CreateICmpEQ(p, q),
                     llvm::ConstantInt::get(param_ir_type, 0), random));
        }
      }
      case RNG_NORMAL: {
        TF_ASSIGN_OR_RETURN(llvm::Value * m,
                            operand_to_generator.at(hlo->operand(0))(index));
        TF_ASSIGN_OR_RETURN(llvm::Value * s,
                            operand_to_generator.at(hlo->operand(1))(index));
        TF_ASSIGN_OR_RETURN(
            llvm::Value * r,
            EmitErfcInv(param_prim_type,
                        ir_builder_->CreateFMul(
                            llvm::ConstantFP::get(param_ir_type, 2.0),
                            get_next_uniform_float())));
        return ir_builder_->CreateFAdd(ir_builder_->CreateFMul(r, s), m);
      }
      case RNG_BERNOULLI: {
        TF_ASSIGN_OR_RETURN(llvm::Value * p,
                            operand_to_generator.at(hlo->operand(0))(index));
        return ir_builder_->CreateZExt(
            ir_builder_->CreateFCmpOLT(get_next_uniform_float(), p),
            llvm_ir::PrimitiveTypeToIrType(hlo->shape().element_type(),
                                           module_));
      }
      default:
        return InvalidArgument(
            "unhandled distribution %s",
            RandomDistribution_Name(hlo->random_distribution()).c_str());
    }
  };
}

llvm_ir::ElementGenerator ElementalIrEmitter::MakeElementGenerator(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator)
    const {
  switch (hlo->opcode()) {
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kCeil:
    case HloOpcode::kConvert:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
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
        TF_ASSIGN_OR_RETURN(llvm::Value * pred_value,
                            operand_to_generator.at(hlo->operand(0))(
                                ElementwiseSourceIndex(index, *hlo, 0)));
        TF_ASSIGN_OR_RETURN(llvm::Value * on_true_value,
                            operand_to_generator.at(hlo->operand(1))(
                                ElementwiseSourceIndex(index, *hlo, 1)));
        TF_ASSIGN_OR_RETURN(llvm::Value * on_false_value,
                            operand_to_generator.at(hlo->operand(2))(
                                ElementwiseSourceIndex(index, *hlo, 2)));
        return ir_builder_->CreateSelect(
            ir_builder_->CreateTrunc(pred_value, ir_builder_->getInt1Ty()),
            on_true_value, on_false_value);
      };
    case HloOpcode::kClamp:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        TF_ASSIGN_OR_RETURN(llvm::Value * min_value,
                            operand_to_generator.at(hlo->operand(0))(
                                ElementwiseSourceIndex(index, *hlo, 0)));
        TF_ASSIGN_OR_RETURN(llvm::Value * arg_value,
                            operand_to_generator.at(hlo->operand(1))(
                                ElementwiseSourceIndex(index, *hlo, 1)));
        TF_ASSIGN_OR_RETURN(llvm::Value * max_value,
                            operand_to_generator.at(hlo->operand(2))(
                                ElementwiseSourceIndex(index, *hlo, 2)));
        return EmitFloatMin(max_value, EmitFloatMax(min_value, arg_value));
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
        const int64 concat_dim = hlo->dimensions(0);
        auto source_index = target_index;

        llvm::BasicBlock* init_block = ir_builder_->GetInsertBlock();

        // A terminator should be present iff we're emitting code
        // into the middle (as opposed to the end) of a basic block.
        CHECK_EQ(ir_builder_->GetInsertPoint() == init_block->end(),
                 init_block->getTerminator() == nullptr);

        llvm::BasicBlock* exit_block;
        if (ir_builder_->GetInsertPoint() == init_block->end()) {
          exit_block = llvm_ir::CreateBasicBlock(
              /*insert_before=*/nullptr, IrName(hlo, "merge"), ir_builder_);
        } else {
          exit_block = init_block->splitBasicBlock(
              ir_builder_->GetInsertPoint(), AsStringRef(IrName(hlo, "merge")));
          init_block->getTerminator()->eraseFromParent();
        }

        llvm_ir::SetToFirstInsertPoint(exit_block, ir_builder_);
        llvm::PHINode* output =
            ir_builder_->CreatePHI(llvm_ir::PrimitiveTypeToIrType(
                                       hlo->shape().element_type(), module_),
                                   hlo->operands().size());
        auto prior_insert_point = ir_builder_->GetInsertPoint();

        ir_builder_->SetInsertPoint(init_block);

        for (int64 operand_idx = 0; operand_idx < hlo->operand_count();
             ++operand_idx) {
          const HloInstruction* operand = hlo->operand(operand_idx);
          auto true_block = llvm_ir::CreateBasicBlock(
              exit_block, StrCat("concat_index_from_operand", operand_idx),
              ir_builder_);
          auto false_block = llvm_ir::CreateBasicBlock(
              exit_block, StrCat("concat_index_not_from_operand", operand_idx),
              ir_builder_);
          auto concat_dim_size =
              llvm::ConstantInt::get(source_index[concat_dim]->getType(),
                                     operand->shape().dimensions(concat_dim));
          ir_builder_->CreateCondBr(
              ir_builder_->CreateICmpULT(source_index[concat_dim],
                                         concat_dim_size),
              true_block, false_block);

          // Create the terminator of the true block before calling operand
          // generators, because they require non-degenerate basic blocks.
          ir_builder_->SetInsertPoint(
              llvm::BranchInst::Create(exit_block, /*InsertAtEnd=*/true_block));
          TF_ASSIGN_OR_RETURN(llvm::Value * value,
                              operand_to_generator.at(operand)(source_index));
          output->addIncoming(value, ir_builder_->GetInsertBlock());

          // Subtract the size of the concat dimension of the current operand
          // from the source index.
          ir_builder_->SetInsertPoint(false_block);
          source_index[concat_dim] =
              ir_builder_->CreateSub(source_index[concat_dim], concat_dim_size);
        }

        ir_builder_->CreateUnreachable();
        ir_builder_->SetInsertPoint(exit_block, prior_insert_point);
        return output;
      };
    case HloOpcode::kReverse:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& target_index) -> StatusOr<llvm::Value*> {
        const HloInstruction* operand = hlo->operand(0);
        auto source_index = target_index;
        for (int64 dim : hlo->dimensions()) {
          source_index[dim] = ir_builder_->CreateSub(
              llvm::ConstantInt::get(target_index[dim]->getType(),
                                     hlo->shape().dimensions(dim) - 1),
              target_index[dim]);
        }
        return operand_to_generator.at(operand)(source_index);
      };
    case HloOpcode::kBroadcast:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& target_index) -> StatusOr<llvm::Value*> {
        // The `dimensions` member of the broadcast instruction maps from
        // input dimensions to output dimensions.
        const HloInstruction* operand = hlo->operand(0);
        int64 rank = ShapeUtil::Rank(operand->shape());
        IrArray::Index source_index(rank);
        for (int64 i = 0; i < rank; ++i) {
          source_index[i] = target_index[hlo->dimensions(i)];
        }
        return operand_to_generator.at(operand)(source_index);
      };
    case HloOpcode::kSlice:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        IrArray::Index sliced_index = index.SourceIndexOfSlice(
            /*shape=*/hlo->shape(), /*starts=*/hlo->slice_starts(),
            /*strides=*/hlo->slice_strides(), /*builder=*/ir_builder_);
        return operand_to_generator.at(hlo->operand(0))(sliced_index);
      };
    case HloOpcode::kDynamicSlice:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        // Emit IR to read dynamic start indices from hlo->operand(1).
        const HloInstruction* input_hlo = hlo->operand(0);
        const int64 rank = ShapeUtil::Rank(input_hlo->shape());
        llvm_ir::IrArray::Index slice_start_index(rank);
        for (int64 i = 0; i < rank; ++i) {
          llvm_ir::IrArray::Index dim_index(1, ir_builder_->getInt64(i));
          TF_ASSIGN_OR_RETURN(
              llvm::Value * start_index_value,
              operand_to_generator.at(hlo->operand(1))(dim_index));
          start_index_value->setName(
              AsStringRef(IrName(hlo, StrCat("start_idx", i))));
          slice_start_index[i] = start_index_value;
        }

        llvm_ir::IrArray::Index input_index(rank);
        for (int64 i = 0; i < rank; ++i) {
          // Emit IR which computes:
          //   input_index = (start_index + offset_index) % dim_size
          // Security note: this is the code that keeps the indices in-bounds.
          llvm::Value* dim_size = llvm::ConstantInt::get(
              index[i]->getType(), input_hlo->shape().dimensions(i));
          llvm::Value* start_index = ir_builder_->CreateZExtOrBitCast(
              slice_start_index[i], index[i]->getType());
          input_index[i] = ir_builder_->CreateURem(
              ir_builder_->CreateAdd(start_index, index[i]), dim_size);
        }
        return operand_to_generator.at(input_hlo)(input_index);
      };
    case HloOpcode::kDynamicUpdateSlice:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        const HloInstruction* input_hlo = hlo->operand(0);
        const HloInstruction* update_hlo = hlo->operand(1);
        const HloInstruction* start_hlo = hlo->operand(2);
        // Calculate slice start/end indices.
        const int64 rank = ShapeUtil::Rank(input_hlo->shape());
        llvm_ir::IrArray::Index slice_start_index(rank);
        llvm_ir::IrArray::Index slice_limit_index(rank);
        // Slice starts at update[index - slice_start_index_adjusted],
        // where adjusted value = slice_start_index when in bounds, and
        // adjusted value = slice_start_index - input_dim, when wrapping.
        llvm_ir::IrArray::Index slice_start_index_adjusted(rank);

        // Slice intersection gathers (ANDs) conditions on all ranks for which
        // 'input' is set to 'update'
        llvm::Value* slice_intersection = ir_builder_->getTrue();

        for (int64 i = 0; i < rank; ++i) {
          // Emit IR to read dynamic start indices from 'start_hlo'.
          llvm_ir::IrArray::Index dim_index(1, ir_builder_->getInt64(i));
          TF_ASSIGN_OR_RETURN(llvm::Value * start_index_value,
                              operand_to_generator.at(start_hlo)(dim_index));
          start_index_value->setName(
              AsStringRef(IrName(hlo, StrCat("start_idx", i))));
          slice_start_index[i] = ir_builder_->CreateZExtOrBitCast(
              start_index_value, index[i]->getType());

          llvm::Value* input_dim_size = llvm::ConstantInt::get(
              index[i]->getType(), input_hlo->shape().dimensions(i));
          llvm::Value* update_dim_size = llvm::ConstantInt::get(
              index[i]->getType(), update_hlo->shape().dimensions(i));

          // Generate code to handle wrapping semantics:
          // slice_start_index[i] = slice_start_index[i] % input_dim_size;
          // slice_limit_index[i] = slice_start_index[i] + update_dim_size.
          // slice_start_index[i] is updated in place and it will now be in
          // range. slice_limit_index[i] may be out of range, and it's being
          // URem-ed below if so.
          slice_start_index[i] =
              ir_builder_->CreateURem(slice_start_index[i], input_dim_size);
          slice_limit_index[i] =
              ir_builder_->CreateAdd(slice_start_index[i], update_dim_size);

          // Test if slice_limit_index[i] is in bounds
          llvm::Value* in_bounds =
              ir_builder_->CreateICmpULE(slice_limit_index[i], input_dim_size);
          llvm_ir::LlvmIfData if_in_bounds =
              llvm_ir::EmitIfThenElse(in_bounds, "in_bounds", ir_builder_);

          // Handle true BB (slice_limit_index[i] <= input_dim_size).
          SetToFirstInsertPoint(if_in_bounds.true_block, ir_builder_);
          // Check that index[i] >= slice_start_index[i] &&
          //            index[i] < slice_limit_index[i]
          llvm::Value* slice_intersection_in_bounds = ir_builder_->CreateAnd(
              slice_intersection,
              ir_builder_->CreateICmpSGE(index[i], slice_start_index[i]),
              "slice_intersection_in");
          slice_intersection_in_bounds = ir_builder_->CreateAnd(
              slice_intersection_in_bounds,
              ir_builder_->CreateICmpSLT(index[i], slice_limit_index[i]),
              "slice_intersection_in");

          // Handle false BB (slice_limit_index[i] > input_dim_size).
          SetToFirstInsertPoint(if_in_bounds.false_block, ir_builder_);
          // Check that index[i] >= slice_start_index[i] ||
          //            index[i] < slice_limit_index[i]%input_dim_size.
          llvm::Value* index_wraps = ir_builder_->CreateICmpSLT(
              index[i],
              ir_builder_->CreateURem(slice_limit_index[i], input_dim_size));
          llvm::Value* slice_intersection_or = ir_builder_->CreateOr(
              ir_builder_->CreateICmpSGE(index[i], slice_start_index[i]),
              index_wraps, "slice_intersection_out");
          llvm::Value* slice_intersection_out_of_bounds =
              ir_builder_->CreateAnd(slice_intersection, slice_intersection_or,
                                     "slice_intersection_out");
          // Create value for slice_start_index_adjusted[i] when out of bounds.
          // If within out-of-bounds if.
          llvm_ir::LlvmIfData if_start_needs_adjustment =
              llvm_ir::EmitIfThenElse(index_wraps, "adjust_start", ir_builder_);
          SetToFirstInsertPoint(if_start_needs_adjustment.true_block,
                                ir_builder_);
          llvm::Value* slice_start_index_adjusted_oob =
              ir_builder_->CreateSub(slice_start_index[i], input_dim_size);
          SetToFirstInsertPoint(if_start_needs_adjustment.after_block,
                                ir_builder_);
          llvm::PHINode* slice_start_index_adjusted_phi =
              ir_builder_->CreatePHI(slice_start_index_adjusted_oob->getType(),
                                     2);
          slice_start_index_adjusted_phi->addIncoming(
              slice_start_index_adjusted_oob,
              if_start_needs_adjustment.true_block);
          slice_start_index_adjusted_phi->addIncoming(
              slice_start_index[i], if_start_needs_adjustment.false_block);
          // End of if within if.

          // After checking in/out of bounds.
          SetToFirstInsertPoint(if_in_bounds.after_block, ir_builder_);
          llvm::PHINode* phi_slice_intersection =
              ir_builder_->CreatePHI(slice_intersection->getType(), 2);
          phi_slice_intersection->addIncoming(slice_intersection_in_bounds,
                                              if_in_bounds.true_block);
          phi_slice_intersection->addIncoming(
              slice_intersection_out_of_bounds,
              if_start_needs_adjustment.after_block);
          slice_intersection = phi_slice_intersection;

          llvm::PHINode* phi_index =
              ir_builder_->CreatePHI(slice_start_index[i]->getType(), 2);
          phi_index->addIncoming(slice_start_index[i], if_in_bounds.true_block);
          phi_index->addIncoming(slice_start_index_adjusted_phi,
                                 if_start_needs_adjustment.after_block);
          slice_start_index_adjusted[i] = phi_index;
        }

        // Emit:
        // if (slice_intersection) -> return data from 'update'.
        // else                    -> return data from 'input'.
        llvm::Value* ret_value_addr = llvm_ir::EmitAllocaAtFunctionEntry(
            llvm_ir::PrimitiveTypeToIrType(hlo->shape().element_type(),
                                           module_),
            "ret_value_addr", ir_builder_);
        llvm_ir::LlvmIfData if_data = llvm_ir::EmitIfThenElse(
            slice_intersection, "slice_intersection", ir_builder_);

        // Handle true BB (return data from 'update')
        SetToFirstInsertPoint(if_data.true_block, ir_builder_);
        // Compute update index for intersection case.
        llvm_ir::IrArray::Index update_index(rank);
        for (int64 i = 0; i < rank; ++i) {
          llvm::Value* update_dim_size = llvm::ConstantInt::get(
              index[i]->getType(), update_hlo->shape().dimensions(i));
          // NOTE: Subtraction will be positive due to bounds checking above.
          update_index[i] = ir_builder_->CreateURem(
              ir_builder_->CreateSub(index[i], slice_start_index_adjusted[i]),
              update_dim_size);
        }
        TF_ASSIGN_OR_RETURN(llvm::Value * true_value,
                            operand_to_generator.at(update_hlo)(update_index));
        ir_builder_->CreateStore(true_value, ret_value_addr);

        // Handle false BB (return data from 'input')
        SetToFirstInsertPoint(if_data.false_block, ir_builder_);
        TF_ASSIGN_OR_RETURN(llvm::Value * false_value,
                            operand_to_generator.at(input_hlo)(index));
        ir_builder_->CreateStore(false_value, ret_value_addr);

        SetToFirstInsertPoint(if_data.after_block, ir_builder_);
        return ir_builder_->CreateLoad(ret_value_addr);
      };
    case HloOpcode::kReshape:
      CHECK_EQ(ShapeUtil::ElementsIn(hlo->shape()),
               ShapeUtil::ElementsIn(hlo->operand(0)->shape()));
      return [this, hlo, &operand_to_generator](const IrArray::Index& index) {
        const HloInstruction* operand = hlo->operand(0);
        return operand_to_generator.at(operand)(index.SourceIndexOfReshape(
            hlo->shape(), operand->shape(), ir_builder_));
      };
    case HloOpcode::kTranspose:
      return [this, hlo,
              &operand_to_generator](const IrArray::Index& target_index) {
        return operand_to_generator.at(hlo->operand(0))(
            target_index.SourceIndexOfTranspose(
                hlo->shape(), hlo->operand(0)->shape(), hlo->dimensions(),
                ir_builder_));
      };
    case HloOpcode::kRng:
      return MakeRngElementGenerator(hlo, operand_to_generator);
    case HloOpcode::kPad:
      return [=, &operand_to_generator](
                 const IrArray::Index& padded_index) -> StatusOr<llvm::Value*> {
        auto index = padded_index;
        llvm::Value* in_bounds = ir_builder_->getTrue();
        for (size_t i = 0; i < index.size(); ++i) {
          auto index_typed_const = [=](int64 n) {
            return llvm::ConstantInt::get(index[i]->getType(), n);
          };
          const auto& pad_dim = hlo->padding_config().dimensions(i);
          index[i] = ir_builder_->CreateSub(
              index[i], index_typed_const(pad_dim.edge_padding_low()));
          in_bounds = ir_builder_->CreateAnd(
              in_bounds,
              ir_builder_->CreateICmpSGE(index[i], index_typed_const(0)),
              "in_bounds");
          in_bounds = ir_builder_->CreateAnd(
              in_bounds,
              ir_builder_->CreateICmpEQ(
                  index_typed_const(0),
                  ir_builder_->CreateURem(
                      index[i],
                      index_typed_const(pad_dim.interior_padding() + 1))),
              "in_bounds");
          index[i] = ir_builder_->CreateSDiv(
              index[i], index_typed_const(pad_dim.interior_padding() + 1));
          in_bounds = ir_builder_->CreateAnd(
              in_bounds,
              ir_builder_->CreateICmpSLT(
                  index[i],
                  index_typed_const(hlo->operand(0)->shape().dimensions(i))),
              "in_bounds");
        }

        // if (in_bounds) {
        //   ret_value = operand0[index];  // source
        // } else {
        //   ret_value = *operand1;        // padding
        // }
        llvm::Value* ret_value_addr = llvm_ir::EmitAllocaAtFunctionEntry(
            llvm_ir::PrimitiveTypeToIrType(hlo->shape().element_type(),
                                           module_),
            "pad_result_addr", ir_builder_);
        llvm_ir::LlvmIfData if_data =
            llvm_ir::EmitIfThenElse(in_bounds, "in_bounds", ir_builder_);
        SetToFirstInsertPoint(if_data.true_block, ir_builder_);
        TF_ASSIGN_OR_RETURN(llvm::Value * operand_value,
                            operand_to_generator.at(hlo->operand(0))(index));
        ir_builder_->CreateStore(operand_value, ret_value_addr);

        SetToFirstInsertPoint(if_data.false_block, ir_builder_);
        TF_ASSIGN_OR_RETURN(llvm::Value * padding_value,
                            operand_to_generator.at(hlo->operand(1))({}));
        ir_builder_->CreateStore(padding_value, ret_value_addr);

        SetToFirstInsertPoint(if_data.after_block, ir_builder_);
        // Don't create phi(operand_value, padding_value) here, because invoking
        // operand_to_generator may create new basic blocks, making the parent
        // of operand_value or padding_value no longer a predecessor of
        // if_data.after_block.
        return ir_builder_->CreateLoad(ret_value_addr);
      };

    case HloOpcode::kDot:
      return [=, &operand_to_generator](const IrArray::Index& dot_result_index)
                 -> StatusOr<llvm::Value*> {
        auto lhs_generator = operand_to_generator.at(hlo->operand(0));
        auto rhs_generator = operand_to_generator.at(hlo->operand(1));
        int64 contracted_dim_size = hlo->operand(0)->shape().dimensions(
            hlo->operand(0)->shape().dimensions_size() - 1);
        int64 lhs_dims = hlo->operand(0)->shape().dimensions_size();
        int64 rhs_dims = hlo->operand(1)->shape().dimensions_size();

        std::unique_ptr<llvm_ir::ForLoop> inner_loop =
            llvm_ir::ForLoop::EmitForLoop(
                IrName(hlo, "inner"), ir_builder_->getInt64(0),
                ir_builder_->getInt64(contracted_dim_size),
                ir_builder_->getInt64(1), ir_builder_);

        SetToFirstInsertPoint(inner_loop->GetPreheaderBasicBlock(),
                              ir_builder_);
        PrimitiveType primitive_type = hlo->shape().element_type();
        llvm::Type* primitive_type_llvm =
            llvm_ir::PrimitiveTypeToIrType(primitive_type, module_);
        llvm::Value* accumulator_alloca = llvm_ir::EmitAllocaAtFunctionEntry(
            primitive_type_llvm, "dot_acc", ir_builder_);
        ir_builder_->CreateStore(
            llvm::Constant::getNullValue(primitive_type_llvm),
            accumulator_alloca);

        SetToFirstInsertPoint(inner_loop->GetBodyBasicBlock(), ir_builder_);

        // This is the inner reduction loop for a dot operation that produces
        // one element in the output.  If the operands to the dot operation have
        // shapes [A,B,C,T] and [D,T,E], the result has a shape [A,B,C,D,E].
        // Given an output index [a,b,c,d,e] in the result, we compute:
        //   sum(lhs[a,b,c,t]*rhs[d,t,e] for t in [0, T))

        IrArray::Index lhs_index, rhs_index;

        for (int64 i = 0; i < lhs_dims - 1; i++) {
          lhs_index.push_back(dot_result_index[i]);
        }
        lhs_index.push_back(inner_loop->GetIndVarValue());

        for (int64 i = 0; i < rhs_dims - 2; i++) {
          rhs_index.push_back(dot_result_index[lhs_dims - 1 + i]);
        }
        rhs_index.push_back(inner_loop->GetIndVarValue());
        rhs_index.push_back(dot_result_index.back());

        llvm::Value* current_accumulator =
            ir_builder_->CreateLoad(accumulator_alloca);
        TF_ASSIGN_OR_RETURN(llvm::Value * lhs_value, lhs_generator(lhs_index));
        TF_ASSIGN_OR_RETURN(llvm::Value * rhs_value, rhs_generator(rhs_index));
        llvm::Value* next_accumulator;
        if (primitive_util::IsComplexType(primitive_type)) {
          llvm::Value* product_real = ir_builder_->CreateFSub(
              ir_builder_->CreateFMul(EmitExtractReal(lhs_value),
                                      EmitExtractReal(rhs_value)),
              ir_builder_->CreateFMul(EmitExtractImag(lhs_value),
                                      EmitExtractImag(rhs_value)));
          llvm::Value* product_imag = ir_builder_->CreateFAdd(
              ir_builder_->CreateFMul(EmitExtractReal(lhs_value),
                                      EmitExtractImag(rhs_value)),
              ir_builder_->CreateFMul(EmitExtractImag(lhs_value),
                                      EmitExtractReal(rhs_value)));
          next_accumulator = ir_builder_->CreateInsertValue(
              current_accumulator,
              ir_builder_->CreateFAdd(EmitExtractReal(current_accumulator),
                                      product_real),
              {0});
          next_accumulator = ir_builder_->CreateInsertValue(
              next_accumulator,
              ir_builder_->CreateFAdd(EmitExtractImag(current_accumulator),
                                      product_imag),
              {1});
        } else if (primitive_util::IsFloatingPointType(primitive_type)) {
          next_accumulator = ir_builder_->CreateFAdd(
              current_accumulator,
              ir_builder_->CreateFMul(lhs_value, rhs_value));
        } else {
          next_accumulator = ir_builder_->CreateAdd(
              current_accumulator,
              ir_builder_->CreateMul(lhs_value, rhs_value));
        }
        ir_builder_->CreateStore(next_accumulator, accumulator_alloca);

        SetToFirstInsertPoint(inner_loop->GetExitBasicBlock(), ir_builder_);
        return ir_builder_->CreateLoad(accumulator_alloca);
      };
    default:
      return [this, hlo, &operand_to_generator](const IrArray::Index& index) {
        return Unimplemented("Unhandled opcode for elemental IR emission: %s",
                             HloOpcodeString(hlo->opcode()).c_str());
      };
  }
}

llvm::Value* ElementalIrEmitter::EmitExtractReal(llvm::Value* value) const {
  return ir_builder_->CreateExtractValue(value, {0});
}

llvm::Value* ElementalIrEmitter::EmitExtractImag(llvm::Value* value) const {
  return ir_builder_->CreateExtractValue(value, {1});
}

llvm::Value* ElementalIrEmitter::EmitComposeComplex(const HloInstruction* op,
                                                    llvm::Value* real,
                                                    llvm::Value* imag) const {
  auto cplx_type =
      llvm_ir::PrimitiveTypeToIrType(op->shape().element_type(), module_);
  auto complex = ir_builder_->CreateInsertValue(
      llvm::ConstantAggregateZero::get(cplx_type), real, {0});
  if (imag != nullptr) {
    complex = ir_builder_->CreateInsertValue(complex, imag, {1});
  }
  return complex;
}

}  // namespace xla
