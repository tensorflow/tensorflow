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

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::xtile {

#define GEN_PASS_DEF_STABLEHLOLOWERTOXTILEPASS
#include "xla/codegen/xtile/ir/transforms/passes.h.inc"

namespace {

using ::mlir::FloatType;
using ::mlir::getElementTypeOrSelf;
using ::mlir::ImplicitLocOpBuilder;
using ::mlir::IntegerType;
using ::mlir::Location;
using ::mlir::PatternRewriter;
using ::mlir::ShapedType;
using ::mlir::Type;
using ::mlir::UnrealizedConversionCastOp;
using ::mlir::Value;

Value GetConstant(ImplicitLocOpBuilder& builder, Type src_ty,
                  Type dst_element_ty, int64_t x) {
  if (auto src_shaped_ty = mlir::dyn_cast<ShapedType>(src_ty)) {
    return ::xla::xtile::CreateConst(builder, dst_element_ty, x,
                                     src_shaped_ty.getShape());
  }
  return ::xla::xtile::CreateConst(builder, dst_element_ty, x);
}

class LowerConvertOp
    : public mlir::OpRewritePattern<mlir::stablehlo::ConvertOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ConvertOp op,
      mlir::PatternRewriter& rewriter) const override {
    Value value = op.getOperand();
    Type src_ty = value.getType();
    Type dst_ty = op.getResult().getType();

    auto builder = mlir::ImplicitLocOpBuilder(op.getLoc(), rewriter);

    auto converted_value_or_status =
        LowerConvert(builder, op.getLoc(), value, src_ty, dst_ty);

    if (!converted_value_or_status.ok()) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          absl::StrCat("Type conversion not supported: ",
                       converted_value_or_status.status().message()));
    }

    rewriter.replaceOp(op, converted_value_or_status.value());
    return mlir::success();
  }

  absl::StatusOr<Value> LowerConvert(ImplicitLocOpBuilder& builder,
                                     Location loc, Value value, Type src_ty,
                                     Type dst_ty) const {
    Type src_element_ty = getElementTypeOrSelf(src_ty);
    Type fp32_ty = builder.getF32Type();
    Type dst_element_ty = getElementTypeOrSelf(dst_ty);

    if (auto src_shaped_ty = mlir::dyn_cast<ShapedType>(src_ty)) {
      fp32_ty =
          src_shaped_ty.clone(src_shaped_ty.getShape(), builder.getF32Type());
    }

    // All operations on bf16 are done through f32.
    if (src_element_ty.isBF16()) {
      return ::xla::xtile::Cast(
          builder, mlir::arith::ExtFOp::create(builder, loc, fp32_ty, value),
          dst_element_ty);
    }
    if (dst_element_ty.isBF16()) {
      // S8 -> BF16 is directly supported and doesn't need to go through f32.
      if (!src_element_ty.isInteger(8)) {
        return mlir::arith::TruncFOp::create(
            builder, loc, dst_ty,
            ::xla::xtile::Cast(builder, value, builder.getF32Type()));
      }
    }

    // float => float
    auto src_fp_element_ty = mlir::dyn_cast<mlir::FloatType>(src_element_ty);
    auto dst_fp_element_ty = mlir::dyn_cast<mlir::FloatType>(dst_element_ty);
    if (src_fp_element_ty && dst_fp_element_ty) {
      return LowerFloatToFloatConvert(builder, loc, value, src_fp_element_ty,
                                      dst_fp_element_ty, dst_ty);
    }
    // int => int
    auto src_int_element_ty = mlir::dyn_cast<mlir::IntegerType>(src_element_ty);
    auto dst_int_element_ty = mlir::dyn_cast<mlir::IntegerType>(dst_element_ty);
    if (src_int_element_ty && dst_int_element_ty) {
      return LowerIntegerToIntegerConvert(
          builder, loc, value, src_int_element_ty, dst_int_element_ty, dst_ty);
    }
    // int => float
    if (src_int_element_ty && dst_fp_element_ty) {
      return LowerIntegerToFloatConvert(builder, loc, value, src_int_element_ty,
                                        dst_ty);
    }
    // float => int
    if (src_fp_element_ty && dst_int_element_ty) {
      return LowerFloatToIntConvert(builder, loc, value, src_fp_element_ty,
                                    dst_int_element_ty, src_ty, dst_ty);
    }

    return absl::UnimplementedError(absl::StrCat(
        "Type conversion from ", ::xla::llvm_ir::DumpToString(src_ty), " to ",
        ::xla::llvm_ir::DumpToString(dst_ty), " not supported"));
  }

  Value LowerFloatToFloatConvert(ImplicitLocOpBuilder& builder, Location loc,
                                 Value value, FloatType src_fp_element_ty,
                                 FloatType dst_fp_element_ty,
                                 Type dst_ty) const {
    Type fp16_ty = builder.getF16Type();

    if (auto dst_shaped_ty = mlir::dyn_cast<ShapedType>(dst_ty)) {
      fp16_ty =
          dst_shaped_ty.clone(dst_shaped_ty.getShape(), builder.getF16Type());
    }

    if (src_fp_element_ty.getIntOrFloatBitWidth() == 8 &&
        dst_fp_element_ty.getIntOrFloatBitWidth() == 8) {
      // FP8 <-> FP8 conversion needs to go through FP16
      auto fp16_value =
          mlir::arith::ExtFOp::create(builder, loc, fp16_ty, value);
      return mlir::arith::TruncFOp::create(builder, loc, dst_ty, fp16_value);
    }

    if (src_fp_element_ty.getFPMantissaWidth() >
        dst_fp_element_ty.getFPMantissaWidth()) {
      return mlir::arith::TruncFOp::create(builder, loc, dst_ty, value);
    }
    return mlir::arith::ExtFOp::create(builder, loc, dst_ty, value);
  }

  Value LowerIntegerToIntegerConvert(ImplicitLocOpBuilder& builder,
                                     Location loc, Value value,
                                     IntegerType src_element_ty,
                                     IntegerType dst_element_ty,
                                     Type dst_ty) const {
    if (src_element_ty.getIntOrFloatBitWidth() <
        dst_element_ty.getIntOrFloatBitWidth()) {
      if (src_element_ty.isUnsignedInteger()) {
        Value signless_integer_value =
            UnsignedIntegerToSignlessInteger(builder, value);
        Type signless_dst_integer_type = IntegerType::get(
            builder.getContext(), dst_element_ty.getIntOrFloatBitWidth(),
            IntegerType::SignednessSemantics::Signless);

        if (auto dst_shaped_ty = mlir::dyn_cast<ShapedType>(dst_ty)) {
          signless_dst_integer_type = dst_shaped_ty.clone(
              dst_shaped_ty.getShape(), signless_dst_integer_type);
        }

        auto ext_op = mlir::arith::ExtUIOp::create(
            builder, loc, signless_dst_integer_type, signless_integer_value);

        return UnrealizedConversionCastOp::create(builder, loc, dst_ty,
                                                  ext_op.getResult())
            .getResult(0);
      }

      if (src_element_ty.isInteger(1)) {
        return mlir::arith::ExtUIOp::create(builder, loc, dst_ty, value);
      }

      return mlir::arith::ExtSIOp::create(builder, loc, dst_ty, value);
    }
    // int => bool is always value != 0.
    if (dst_element_ty.isInteger(1)) {
      return mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::ne, value,
          ::xla::xtile::ZerosLike(builder, value));
    }
    return mlir::arith::TruncIOp::create(builder, loc, dst_ty, value);
  }

  Value LowerFloatToIntConvert(ImplicitLocOpBuilder& builder, Location loc,
                               Value value, FloatType src_fp_element_ty,
                               IntegerType dst_element_ty, Type src_ty,
                               Type dst_ty) const {
    if (dst_element_ty.isInteger(1)) {
      return mlir::arith::CmpFOp::create(
          builder, loc, mlir::arith::CmpFPredicate::UNE, value,
          ::xla::xtile::ZerosLike(builder, value));
    }
    // The current logic handles signed integer types only. Additional
    // handling is needed for unsigned integer types.
    Value fptosi = mlir::arith::FPToSIOp::create(builder, loc, dst_ty, value);
    int64_t min = llvm::minIntN(dst_element_ty.getIntOrFloatBitWidth());
    int64_t max = llvm::maxIntN(dst_element_ty.getIntOrFloatBitWidth());

    // value <= static_cast<float>(INT_MIN) ? INT_MIN : ...
    Value clamped = mlir::arith::SelectOp::create(
        builder, loc,
        mlir::arith::CmpFOp::create(
            builder, loc, mlir::arith::CmpFPredicate::OLE, value,
            GetConstant(builder, src_ty, src_fp_element_ty, min)),
        GetConstant(builder, src_ty, dst_element_ty, min), fptosi);
    // value >= static_cast<float>(INT_MAX) ? INT_MAX : ...
    clamped = mlir::arith::SelectOp::create(
        builder, loc,
        mlir::arith::CmpFOp::create(
            builder, loc, mlir::arith::CmpFPredicate::OGE, value,
            GetConstant(builder, src_ty, src_fp_element_ty, max)),
        GetConstant(builder, src_ty, dst_element_ty, max), clamped);
    // isnan(value) ? 0 : ...
    return mlir::arith::SelectOp::create(
        builder, loc,
        mlir::arith::CmpFOp::create(
            builder, loc, mlir::arith::CmpFPredicate::UNO, value, value),
        GetConstant(builder, src_ty, dst_element_ty, 0), clamped);
  }

  Value LowerIntegerToFloatConvert(ImplicitLocOpBuilder& builder, Location loc,
                                   Value value, IntegerType src_element_ty,
                                   Type dst_ty) const {
    if (src_element_ty.isInteger(1)) {
      return mlir::arith::UIToFPOp::create(builder, loc, dst_ty, value);
    }
    if (src_element_ty.isUnsignedInteger()) {
      value = UnsignedIntegerToSignlessInteger(builder, value);
      return mlir::arith::UIToFPOp::create(builder, loc, dst_ty, value);
    }
    return mlir::arith::SIToFPOp::create(builder, loc, dst_ty, value);
  }
};

std::optional<mlir::arith::CmpIPredicate> GetCmpIPredicate(
    mlir::stablehlo::ComparisonDirection direction, bool is_unsigned) {
  switch (direction) {
    case mlir::stablehlo::ComparisonDirection::EQ:
      return mlir::arith::CmpIPredicate::eq;
    case mlir::stablehlo::ComparisonDirection::NE:
      return mlir::arith::CmpIPredicate::ne;
    case mlir::stablehlo::ComparisonDirection::LT:
      return is_unsigned ? mlir::arith::CmpIPredicate::ult
                         : mlir::arith::CmpIPredicate::slt;
    case mlir::stablehlo::ComparisonDirection::GT:
      return is_unsigned ? mlir::arith::CmpIPredicate::ugt
                         : mlir::arith::CmpIPredicate::sgt;
    case mlir::stablehlo::ComparisonDirection::LE:
      return is_unsigned ? mlir::arith::CmpIPredicate::ule
                         : mlir::arith::CmpIPredicate::sle;
    case mlir::stablehlo::ComparisonDirection::GE:
      return is_unsigned ? mlir::arith::CmpIPredicate::uge
                         : mlir::arith::CmpIPredicate::sge;
    default:
      return std::nullopt;
  }
}

std::optional<mlir::arith::CmpFPredicate> GetCmpFPredicate(
    mlir::stablehlo::ComparisonDirection direction) {
  switch (direction) {
    case mlir::stablehlo::ComparisonDirection::EQ:
      return mlir::arith::CmpFPredicate::OEQ;
    case mlir::stablehlo::ComparisonDirection::NE:
      return mlir::arith::CmpFPredicate::UNE;
    case mlir::stablehlo::ComparisonDirection::GE:
      return mlir::arith::CmpFPredicate::OGE;
    case mlir::stablehlo::ComparisonDirection::GT:
      return mlir::arith::CmpFPredicate::OGT;
    case mlir::stablehlo::ComparisonDirection::LE:
      return mlir::arith::CmpFPredicate::OLE;
    case mlir::stablehlo::ComparisonDirection::LT:
      return mlir::arith::CmpFPredicate::OLT;
    default:
      return std::nullopt;
  }
}

class LowerCompareOp
    : public mlir::OpRewritePattern<mlir::stablehlo::CompareOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::CompareOp op,
      mlir::PatternRewriter& rewriter) const override {
    Value compare_result = GetCompareOp(rewriter, op);

    rewriter.replaceOp(op, compare_result);
    return mlir::success();
  }

  Value GetCompareOp(mlir::PatternRewriter& rewriter,
                     mlir::stablehlo::CompareOp op) const {
    const Type element_type = mlir::getElementTypeOrSelf(op.getLhs());
    auto direction = op.getComparisonDirection();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    if (mlir::isa<mlir::IntegerType>(element_type)) {
      if (element_type.isUnsignedInteger()) {
        lhs = ::xla::xtile::UnsignedIntegerToSignlessInteger(rewriter, lhs);
        rhs = ::xla::xtile::UnsignedIntegerToSignlessInteger(rewriter, rhs);
      }

      return mlir::arith::CmpIOp::create(
          rewriter, op.getLoc(),
          GetCmpIPredicate(direction,
                           /*is_unsigned=*/element_type.isUnsignedInteger() ||
                               element_type.isInteger(1))
              .value(),
          lhs, rhs);
    }
    return mlir::arith::CmpFOp::create(
        rewriter, op.getLoc(), GetCmpFPredicate(direction).value(), lhs, rhs);
  }
};

struct StablehloLowerToXtilePass
    : public impl::StablehloLowerToXtilePassBase<StablehloLowerToXtilePass> {
  using StablehloLowerToXtilePassBase::StablehloLowerToXtilePassBase;

  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<LowerConvertOp, LowerCompareOp>(mlir_context);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace xla::xtile
