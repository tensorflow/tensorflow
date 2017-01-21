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
#include "external/llvm/include/llvm/IR/BasicBlock.h"
#include "external/llvm/include/llvm/IR/Instructions.h"
#include "external/llvm/include/llvm/IR/Intrinsics.h"
#include "external/llvm/include/llvm/Transforms/Utils/BasicBlockUtils.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
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

using llvm_ir::IrArray;
using llvm_ir::SetToFirstInsertPoint;

StatusOr<llvm::Value*> ElementalIrEmitter::EmitUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) const {
  if (op->opcode() == HloOpcode::kCopy) {
    return operand_value;
  } else {
    return operand_value->getType()->isIntegerTy()
               ? EmitIntegerUnaryOp(op, operand_value)
               : EmitFloatUnaryOp(op, operand_value);
  }
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitIntegerUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) const {
  switch (op->opcode()) {
    case HloOpcode::kConvert: {
      PrimitiveType from_type = op->operand(0)->shape().element_type();
      PrimitiveType to_type = op->shape().element_type();
      CHECK(primitive_util::IsIntegralType(from_type));
      if (from_type == to_type) {
        return operand_value;
      }
      if (primitive_util::IsIntegralType(to_type)) {
        return ir_builder_->CreateIntCast(
            operand_value, llvm_ir::PrimitiveTypeToIrType(to_type, ir_builder_),
            primitive_util::IsSignedIntegralType(to_type));
      }
      if (primitive_util::IsFloatingPointType(to_type)) {
        if (primitive_util::IsSignedIntegralType(from_type)) {
          return ir_builder_->CreateSIToFP(
              operand_value,
              llvm_ir::PrimitiveTypeToIrType(to_type, ir_builder_));
        }
        if (primitive_util::IsUnsignedIntegralType(from_type)) {
          return ir_builder_->CreateUIToFP(
              operand_value,
              llvm_ir::PrimitiveTypeToIrType(to_type, ir_builder_));
        }
      }
      return Unimplemented("conversion from primitive type %s to %s",
                           PrimitiveType_Name(from_type).c_str(),
                           PrimitiveType_Name(to_type).c_str());
    }
    case HloOpcode::kAbs: {
      bool is_signed =
          primitive_util::IsSignedIntegralType(op->shape().element_type());
      if (is_signed) {
        auto type = llvm_ir::PrimitiveTypeToIrType(op->shape().element_type(),
                                                   ir_builder_);
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
      auto type = llvm_ir::PrimitiveTypeToIrType(op->shape().element_type(),
                                                 ir_builder_);
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
    case HloOpcode::kLogicalNot:
      // It is not sufficient to just call CreateNot() here because a PRED is
      // represented as an i8 and the truth value is stored only in the bottom
      // bit.
      return ir_builder_->CreateZExt(
          ir_builder_->CreateNot(ir_builder_->CreateTrunc(
              operand_value, ir_builder_->getInt1Ty())),
          llvm_ir::PrimitiveTypeToIrType(PRED, ir_builder_));
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
      if (primitive_util::IsFloatingPointType(to_type)) {
        return ir_builder_->CreateFPCast(
            operand_value,
            llvm_ir::PrimitiveTypeToIrType(to_type, ir_builder_));
      }
      if (primitive_util::IsSignedIntegralType(to_type)) {
        return ir_builder_->CreateFPToSI(
            operand_value,
            llvm_ir::PrimitiveTypeToIrType(to_type, ir_builder_));
      }
      if (primitive_util::IsUnsignedIntegralType(to_type)) {
        return ir_builder_->CreateFPToUI(
            operand_value,
            llvm_ir::PrimitiveTypeToIrType(to_type, ir_builder_));
      }
      return Unimplemented("unhandled conversion operation: %s => %s",
                           PrimitiveType_Name(from_type).c_str(),
                           PrimitiveType_Name(to_type).c_str());
    }
    case HloOpcode::kExp:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::exp, {operand_value},
                                          {operand_value->getType()},
                                          ir_builder_);
    case HloOpcode::kLog:
      return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::log, {operand_value},
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
    case HloOpcode::kNegate:
      return ir_builder_->CreateFNeg(operand_value);
    default:
      return Unimplemented("unary floating-point op '%s'",
                           HloOpcodeString(op->opcode()).c_str());
  }
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value,
    llvm::Value* rhs_value) const {
  return lhs_value->getType()->isIntegerTy()
             ? EmitIntegerBinaryOp(op, lhs_value, rhs_value,
                                   primitive_util::IsSignedIntegralType(
                                       op->operand(0)->shape().element_type()))
             : EmitFloatBinaryOp(op, lhs_value, rhs_value);
}

StatusOr<llvm::Value*> ElementalIrEmitter::EmitFloatBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value,
    llvm::Value* rhs_value) const {
  switch (op->opcode()) {
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

    // The 'O' prefix on the LLVM ops means "ordered" compare where comparisons
    // with NAN always return false.
    case HloOpcode::kEq:
      return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_OEQ, lhs_value,
                                     rhs_value, ir_builder_);
    case HloOpcode::kNe:
      return llvm_ir::EmitComparison(llvm::CmpInst::FCMP_ONE, lhs_value,
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

llvm::Value* ElementalIrEmitter::EmitFloatMax(llvm::Value* lhs_value,
                                              llvm::Value* rhs_value) const {
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::maxnum,
                                      {lhs_value, rhs_value},
                                      {lhs_value->getType()}, ir_builder_);
}

llvm::Value* ElementalIrEmitter::EmitFloatMin(llvm::Value* lhs_value,
                                              llvm::Value* rhs_value) const {
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::minnum,
                                      {lhs_value, rhs_value},
                                      {lhs_value->getType()}, ir_builder_);
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
  auto type = llvm_ir::PrimitiveTypeToIrType(prim_type, ir_builder_);
  auto one = llvm::ConstantFP::get(type, 1.0);
  return EmitErfInv(prim_type, ir_builder_->CreateFSub(one, value));
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
    case HloOpcode::kLogicalAnd:
      return ir_builder_->CreateAnd(lhs_value, rhs_value);
    case HloOpcode::kLogicalOr:
      return ir_builder_->CreateOr(lhs_value, rhs_value);
    default:
      return Unimplemented("binary integer op '%s'",
                           HloOpcodeString(op->opcode()).c_str());
  }
}

llvm_ir::IrArray::Index ElementalIrEmitter::ElementwiseSourceIndex(
    const llvm_ir::IrArray::Index& target_index, const HloInstruction& hlo,
    int64 operand_no) const {
  CHECK(hlo.IsElementwise()) << "HLO " << hlo.ToString()
                             << " is not elementwise.";

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
      llvm_ir::PrimitiveTypeToIrType(param_prim_type, ir_builder_);

  // Same values as PCG library
  // https://github.com/imneme/pcg-c/blob/master/include/pcg_variants.h
  llvm::Value* multiplier = ir_builder_->getInt(
      llvm::APInt(128, {0x4385DF649FCCF645, 0x2360ED051FC65DA4}));
  llvm::Value* increment = ir_builder_->getInt(
      llvm::APInt(128, {0x14057B7EF767814F, 0x5851F42D4C957F2D}));

  auto random_value = [hlo]() {
    CHECK(hlo->parent() != nullptr && hlo->parent()->parent() != nullptr);
    const HloModule* module = hlo->parent()->parent();
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
              llvm::Intrinsic::ctlz, {r, ir_builder_->getInt1(1)},
              {param_ir_type}, ir_builder_);
          auto in_block = ir_builder_->GetInsertBlock();
          auto body_block = in_block->splitBasicBlock(
              ir_builder_->GetInsertPoint(), "rng_body");
          SetToFirstInsertPoint(body_block, ir_builder_);
          auto out_block = body_block->splitBasicBlock(
              ir_builder_->GetInsertPoint(), "rng_out");
          SetToFirstInsertPoint(body_block, ir_builder_);
          auto random = ir_builder_->CreateAnd(
              ir_builder_->CreateZExtOrTrunc(get_next_i64(), param_ir_type),
              ir_builder_->CreateLShr(llvm::ConstantInt::get(param_ir_type, ~0),
                                      leading_zeros));
          llvm::ReplaceInstWithInst(
              body_block->getTerminator(),
              llvm::BranchInst::Create(out_block, body_block,
                                       ir_builder_->CreateICmpULE(random, r)));
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
                                           ir_builder_));
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
  // TODO(mfdyck): Make capture lists explicit, lest someone forget to cap
  // `operand_to_generator` by ref and its many copies fill memory and cause
  // much woe and process death.
  switch (hlo->opcode()) {
    case HloOpcode::kAbs:
    case HloOpcode::kCeil:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kExp:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kNegate:
    case HloOpcode::kSign:
    case HloOpcode::kTanh:
    case HloOpcode::kLogicalNot:
      return [=, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        TF_ASSIGN_OR_RETURN(llvm::Value * operand_value,
                            operand_to_generator.at(hlo->operand(0))(
                                ElementwiseSourceIndex(index, *hlo, 0)));
        return EmitUnaryOp(hlo, operand_value);
      };
    case HloOpcode::kAdd:
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
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kLogicalAnd:
    case HloOpcode::kLogicalOr:
      return [=, &operand_to_generator](
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
      return [=, &operand_to_generator](
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
      return [=, &operand_to_generator](
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
    case HloOpcode::kConcatenate:
      return [=, &operand_to_generator](
                 const IrArray::Index target_index) -> StatusOr<llvm::Value*> {
        const int64 concat_dim = hlo->dimensions(0);
        auto source_index = target_index;

        llvm::PHINode* output = ir_builder_->CreatePHI(
            llvm_ir::PrimitiveTypeToIrType(hlo->shape().element_type(),
                                           ir_builder_),
            hlo->operands().size());
        llvm::BasicBlock* init_block = ir_builder_->GetInsertBlock();
        auto prior_insert_point = ir_builder_->GetInsertPoint();
        llvm::BasicBlock* exit_block =
            init_block->splitBasicBlock(output, "concat_merge");

        ir_builder_->SetInsertPoint(init_block);
        init_block->getTerminator()->eraseFromParent();

        for (int64 operand_idx = 0; operand_idx < hlo->operand_count();
             ++operand_idx) {
          const HloInstruction* operand = hlo->operand(operand_idx);
          auto true_block = llvm_ir::CreateBasicBlock(
              exit_block, tensorflow::strings::StrCat(
                              "concat_index_from_operand", operand_idx),
              ir_builder_);
          auto false_block = llvm_ir::CreateBasicBlock(
              exit_block, tensorflow::strings::StrCat(
                              "concat_index_not_from_operand", operand_idx),
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
      return [=, &operand_to_generator](
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
      return [=, &operand_to_generator](
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
      return [=, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        IrArray::Index sliced_index(index.size());
        for (int i = 0; i < index.size(); ++i) {
          sliced_index[i] = ir_builder_->CreateAdd(
              index[i], llvm::ConstantInt::get(index[i]->getType(),
                                               hlo->slice_starts(i)));
        }
        return operand_to_generator.at(hlo->operand(0))(sliced_index);
      };
    case HloOpcode::kDynamicSlice:
      return [=, &operand_to_generator](
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
      return [=, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        const HloInstruction* input_hlo = hlo->operand(0);
        const HloInstruction* update_hlo = hlo->operand(1);
        const HloInstruction* start_hlo = hlo->operand(2);
        // Calculate slice start/end indices.
        const int64 rank = ShapeUtil::Rank(input_hlo->shape());
        llvm_ir::IrArray::Index slice_start_index(rank);
        llvm_ir::IrArray::Index slice_limit_index(rank);
        for (int64 i = 0; i < rank; ++i) {
          // Emit IR to read dynamic start indices from 'start_hlo'.
          llvm_ir::IrArray::Index dim_index(1, ir_builder_->getInt64(i));
          TF_ASSIGN_OR_RETURN(llvm::Value * start_index_value,
                              operand_to_generator.at(start_hlo)(dim_index));
          slice_start_index[i] = ir_builder_->CreateZExtOrBitCast(
              start_index_value, index[i]->getType());
          // Emit IR to compute: slice_limit_index = start_index + update_dim
          // NOTE: Although 'start_indices' is dynamic and could be
          // out-of-range, we do not compute 'slice_limit_index' mod input dim
          // size here, because subsequent array index calculations will be
          // computed mod input dim size for safety.
          llvm::Value* update_dim_size = llvm::ConstantInt::get(
              index[i]->getType(), update_hlo->shape().dimensions(i));
          slice_limit_index[i] =
              ir_builder_->CreateAdd(slice_start_index[i], update_dim_size);
        }

        // Check if 'index' intersects start/end indices.
        llvm::Value* slice_intersection =
            llvm::ConstantInt::get(ir_builder_->getInt1Ty(), 1);

        for (int64 i = 0; i < rank; ++i) {
          // Check that index[i] >= slice_start_index[i].
          slice_intersection = ir_builder_->CreateAnd(
              slice_intersection,
              ir_builder_->CreateICmpSGE(index[i], slice_start_index[i]),
              "slice_intersection");

          // Check that index[i] < slice_limit_index[i].
          slice_intersection = ir_builder_->CreateAnd(
              slice_intersection,
              ir_builder_->CreateICmpSLT(index[i], slice_limit_index[i]),
              "slice_intersection");
        }

        // Emit:
        // if (slice_intersection) -> return data from 'update'.
        // else                    -> return data from 'index'.
        llvm::Value* ret_value_addr = llvm_ir::EmitAllocaAtFunctionEntry(
            llvm_ir::PrimitiveTypeToIrType(hlo->shape().element_type(),
                                           ir_builder_),
            "ret_value_addr", ir_builder_);
        llvm_ir::LlvmIfData if_data = llvm_ir::EmitIfThenElse(
            slice_intersection, "slice_intersection", ir_builder_);

        // Handle true BB.
        SetToFirstInsertPoint(if_data.true_block, ir_builder_);
        // Compute update index for intersection case.
        llvm_ir::IrArray::Index update_index(rank);
        for (int64 i = 0; i < rank; ++i) {
          llvm::Value* update_dim_size = llvm::ConstantInt::get(
              index[i]->getType(), update_hlo->shape().dimensions(i));
          // NOTE: Subtraction will be positive due to bounds checking above.
          update_index[i] = ir_builder_->CreateURem(
              ir_builder_->CreateSub(index[i], slice_start_index[i]),
              update_dim_size);
        }
        TF_ASSIGN_OR_RETURN(llvm::Value * true_value,
                            operand_to_generator.at(update_hlo)(update_index));
        ir_builder_->CreateStore(true_value, ret_value_addr);

        // Handle false BB.
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
      return [=, &operand_to_generator](const IrArray::Index& index) {
        const HloInstruction* operand = hlo->operand(0);
        return operand_to_generator.at(operand)(index.SourceIndexOfReshape(
            hlo->shape(), operand->shape(), ir_builder_));
      };
    case HloOpcode::kTranspose:
      return [=, &operand_to_generator](const IrArray::Index& target_index) {
        return operand_to_generator.at(hlo->operand(0))(
            target_index.SourceIndexOfTranspose(
                hlo->shape(), hlo->operand(0)->shape(), hlo->dimensions(),
                ir_builder_));
      };
    case HloOpcode::kRng:
      return MakeRngElementGenerator(hlo, operand_to_generator);
    default:
      return [=, &operand_to_generator](const IrArray::Index& index) {
        return Unimplemented("%s", HloOpcodeString(hlo->opcode()).c_str());
      };
  }
}

}  // namespace xla
