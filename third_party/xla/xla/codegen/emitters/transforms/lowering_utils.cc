/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/codegen/emitters/transforms/lowering_utils.h"

#include <cstdint>

#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "xla/tsl/platform/logging.h"

namespace xla {
namespace emitters {

void EnsureAMDGPUAllocasUseAS5(mlir::Operation* operation) {
  operation->walk([](mlir::LLVM::AllocaOp alloca) {
    auto ptr_type =
        mlir::cast<mlir::LLVM::LLVMPointerType>(alloca.getResult().getType());
    // Check if address space is 0 (default/generic)
    if (ptr_type.getAddressSpace() == 0) {
      mlir::OpBuilder builder(alloca);
      // Create new alloca in address space 5
      auto new_ptr_type =
          mlir::LLVM::LLVMPointerType::get(builder.getContext(), 5);
      auto new_alloca = builder.create<mlir::LLVM::AllocaOp>(
          alloca.getLoc(), new_ptr_type, alloca.getElemType(),
          alloca.getArraySize(), alloca.getAlignment().value_or(0));
      alloca.replaceAllUsesWith(new_alloca.getResult());
      alloca.erase();
    }
  });
  VLOG(3) << "Ensured AMDGPU allocas use address space 5";
}

namespace {

mlir::Value GetConstant(mlir::ImplicitLocOpBuilder& builder,
                        mlir::Type container_ty, mlir::Type elem_ty,
                        int64_t val) {
  mlir::TypedAttr attr;
  if (auto fp_ty = mlir::dyn_cast<mlir::FloatType>(elem_ty)) {
    attr = builder.getFloatAttr(fp_ty, static_cast<double>(val));
  } else {
    attr = builder.getIntegerAttr(elem_ty, val);
  }
  if (auto shaped_ty = mlir::dyn_cast<mlir::ShapedType>(container_ty)) {
    attr = mlir::DenseElementsAttr::get(shaped_ty.clone(elem_ty), attr);
  }
  return mlir::arith::ConstantOp::create(builder, attr);
}

mlir::Value GetUnsignedConstant(mlir::ImplicitLocOpBuilder& builder,
                                mlir::Type container_ty, mlir::Type elem_ty,
                                uint64_t val) {
  mlir::TypedAttr attr;
  if (auto fp_ty = mlir::dyn_cast<mlir::FloatType>(elem_ty)) {
    attr = builder.getFloatAttr(fp_ty, static_cast<double>(val));
  } else {
    attr = builder.getIntegerAttr(elem_ty, val);
  }
  if (auto shaped_ty = mlir::dyn_cast<mlir::ShapedType>(container_ty)) {
    attr = mlir::DenseElementsAttr::get(shaped_ty.clone(elem_ty), attr);
  }
  return mlir::arith::ConstantOp::create(builder, attr);
}

}  // namespace

mlir::Value EmitFloatToIntConvertWithClamping(
    mlir::ImplicitLocOpBuilder& builder, mlir::Value value,
    mlir::FloatType src_fp_element_ty, mlir::IntegerType dst_element_ty,
    mlir::Type src_ty, mlir::Type dst_ty) {
  mlir::Location loc = builder.getLoc();
  if (dst_element_ty.isInteger(1)) {
    mlir::Value zero = GetConstant(builder, src_ty, src_fp_element_ty, 0);
    return mlir::arith::CmpFOp::create(
        builder, loc, mlir::arith::CmpFPredicate::UNE, value, zero);
  }

  mlir::Type signless_dst_elem_ty = dst_element_ty;
  if (dst_element_ty.isUnsignedInteger()) {
    signless_dst_elem_ty = mlir::IntegerType::get(
        builder.getContext(), dst_element_ty.getIntOrFloatBitWidth(),
        mlir::IntegerType::SignednessSemantics::Signless);
  }

  mlir::Type signless_dst_ty = dst_ty;
  if (dst_element_ty.isUnsignedInteger()) {
    if (auto shaped_ty = mlir::dyn_cast<mlir::ShapedType>(dst_ty)) {
      signless_dst_ty = shaped_ty.clone(signless_dst_elem_ty);
    } else {
      signless_dst_ty = signless_dst_elem_ty;
    }
  }

  mlir::Value result;
  if (dst_element_ty.isUnsignedInteger()) {
    mlir::Value fptoui =
        mlir::arith::FPToUIOp::create(builder, loc, signless_dst_ty, value);
    uint64_t max = llvm::maxUIntN(dst_element_ty.getIntOrFloatBitWidth());

    // value <= 0 ? 0 : ...
    mlir::Value clamped = mlir::arith::SelectOp::create(
        builder, loc,
        mlir::arith::CmpFOp::create(
            builder, loc, mlir::arith::CmpFPredicate::OLE, value,
            GetConstant(builder, src_ty, src_fp_element_ty, 0)),
        GetConstant(builder, src_ty, signless_dst_elem_ty, 0), fptoui);
    // value >= static_cast<float>(UINT_MAX) ? UINT_MAX : ...
    clamped = mlir::arith::SelectOp::create(
        builder, loc,
        mlir::arith::CmpFOp::create(
            builder, loc, mlir::arith::CmpFPredicate::OGE, value,
            GetUnsignedConstant(builder, src_ty, src_fp_element_ty, max)),
        GetUnsignedConstant(builder, src_ty, signless_dst_elem_ty, max),
        clamped);
    // isnan(value) ? 0 : ...
    result = mlir::arith::SelectOp::create(
        builder, loc,
        mlir::arith::CmpFOp::create(
            builder, loc, mlir::arith::CmpFPredicate::UNO, value, value),
        GetConstant(builder, src_ty, signless_dst_elem_ty, 0), clamped);
  } else {
    mlir::Value fptosi =
        mlir::arith::FPToSIOp::create(builder, loc, signless_dst_ty, value);
    int64_t min = llvm::minIntN(dst_element_ty.getIntOrFloatBitWidth());
    int64_t max = llvm::maxIntN(dst_element_ty.getIntOrFloatBitWidth());

    // value <= static_cast<float>(INT_MIN) ? INT_MIN : ...
    mlir::Value clamped = mlir::arith::SelectOp::create(
        builder, loc,
        mlir::arith::CmpFOp::create(
            builder, loc, mlir::arith::CmpFPredicate::OLE, value,
            GetConstant(builder, src_ty, src_fp_element_ty, min)),
        GetConstant(builder, src_ty, signless_dst_elem_ty, min), fptosi);
    // value >= static_cast<float>(INT_MAX) ? INT_MAX : ...
    clamped = mlir::arith::SelectOp::create(
        builder, loc,
        mlir::arith::CmpFOp::create(
            builder, loc, mlir::arith::CmpFPredicate::OGE, value,
            GetConstant(builder, src_ty, src_fp_element_ty, max)),
        GetConstant(builder, src_ty, signless_dst_elem_ty, max), clamped);
    // isnan(value) ? 0 : ...
    result = mlir::arith::SelectOp::create(
        builder, loc,
        mlir::arith::CmpFOp::create(
            builder, loc, mlir::arith::CmpFPredicate::UNO, value, value),
        GetConstant(builder, src_ty, signless_dst_elem_ty, 0), clamped);
  }

  if (dst_element_ty.isUnsignedInteger()) {
    result =
        mlir::UnrealizedConversionCastOp::create(builder, loc, dst_ty, result)
            .getResult(0);
  }
  return result;
}

}  // namespace emitters
}  // namespace xla
