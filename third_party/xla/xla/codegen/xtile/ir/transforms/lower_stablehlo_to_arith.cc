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

#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/emitters/transforms/lowering_utils.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"  // IWYU pragma: keep
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::xtile {

#define GEN_PASS_DEF_STABLEHLOLOWERTOARITHPASS
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

template <typename StableHloOp, typename FloatArithOp, typename IntArithOp,
          typename UnsignedIntArithOp = IntArithOp>
class LowerStableHloOpToArith : public mlir::OpRewritePattern<StableHloOp> {
 public:
  using mlir::OpRewritePattern<StableHloOp>::OpRewritePattern;

 private:
  bool NeedsPromotion(StableHloOp op, mlir::Type element_type) const {
    if (!element_type.isBF16() && !element_type.isF16()) {
      return false;
    }
    return mlir::isa<mlir::stablehlo::DivOp, mlir::stablehlo::RemOp>(op);
  }

  mlir::LogicalResult matchAndRewrite(
      StableHloOp op, mlir::PatternRewriter& rewriter) const override {
    auto result_type = mlir::getElementTypeOrSelf(op.getResult().getType());
    auto operand_type = mlir::getElementTypeOrSelf(op->getOperand(0).getType());
    if (mlir::isa<mlir::ComplexType>(result_type) ||
        mlir::isa<mlir::ComplexType>(operand_type)) {
      return rewriter.notifyMatchFailure(
          op, "Complex types are legalized by StablehloLegalizeToLinalg");
    }
    if (result_type.isFloat()) {
      if (NeedsPromotion(op, result_type)) {
        mlir::Type f32_type = rewriter.getF32Type();
        if (auto ranked_type = mlir::dyn_cast<mlir::RankedTensorType>(
                op.getResult().getType())) {
          f32_type = mlir::RankedTensorType::get(ranked_type.getShape(),
                                                 rewriter.getF32Type());
        }

        llvm::SmallVector<mlir::Value> promoted_operands;
        for (auto operand : op->getOperands()) {
          if (mlir::getElementTypeOrSelf(operand.getType()).isF32()) {
            promoted_operands.push_back(operand);
          } else {
            promoted_operands.push_back(mlir::arith::ExtFOp::create(
                rewriter, op.getLoc(), f32_type, operand));
          }
        }

        auto promoted_op = FloatArithOp::create(rewriter, op.getLoc(), f32_type,
                                                promoted_operands);
        rewriter.replaceOpWithNewOp<mlir::arith::TruncFOp>(
            op, op.getResult().getType(), promoted_op->getResult(0));
      } else {
        auto new_op =
            FloatArithOp::create(rewriter, op.getLoc(), op->getOperands());
        rewriter.replaceOp(op, new_op);
      }
    } else {
      mlir::Operation* new_op = nullptr;
      bool should_guard_ub =
          mlir::isa<mlir::stablehlo::DivOp, mlir::stablehlo::RemOp>(op);

      if (result_type.isUnsignedInteger()) {
        llvm::SmallVector<mlir::Value> signless_operands;
        signless_operands.reserve(op->getOperands().size());
        for (mlir::Value operand : op->getOperands()) {
          signless_operands.push_back(
              ::xla::xtile::UnsignedIntegerToSignlessInteger(rewriter,
                                                             operand));
        }
        new_op = UnsignedIntArithOp::create(rewriter, op.getLoc(),
                                            signless_operands.front().getType(),
                                            signless_operands);

        rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
            op, op.getResult().getType(), new_op->getResult(0));
      } else {
        new_op = IntArithOp::create(rewriter, op.getLoc(), op->getOperands());
        rewriter.replaceOp(op, new_op);
      }

      // Special case for division with zero.
      if (should_guard_ub) {
        new_op->setAttr("xla.guard_ub", rewriter.getUnitAttr());
      }
    }
    return mlir::success();
  }
};

// Pattern to lower a float-only StableHLO operation to a math dialect
// operation. Some operations (like RoundNearestEvenOp) do not have a
// corresponding integer variant in arith or math, so they use a separate
// pattern.
template <typename StableHloOp, typename MathOp>
class LowerStableHloUnaryOpToMath : public mlir::OpRewritePattern<StableHloOp> {
 public:
  using mlir::OpRewritePattern<StableHloOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      StableHloOp op, mlir::PatternRewriter& rewriter) const override {
    auto result_type = mlir::getElementTypeOrSelf(op.getResult().getType());
    if (!result_type.isFloat()) {
      return rewriter.notifyMatchFailure(op, "expected float type");
    }
    rewriter.replaceOpWithNewOp<MathOp>(op, op->getOperands());
    return mlir::success();
  }
};

// Lowering for stablehlo.negate op to arith.negf for float or arith.subi (0 -
// x) for integer.
struct LowerStableHloNegOpToArith
    : public mlir::OpRewritePattern<mlir::stablehlo::NegOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::NegOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto result_type = mlir::getElementTypeOrSelf(op.getResult().getType());
    if (mlir::isa<mlir::ComplexType>(result_type)) {
      return rewriter.notifyMatchFailure(
          op, "complex types are legalized by StablehloLegalizeToLinalg");
    }
    if (result_type.isFloat()) {
      auto new_op =
          mlir::arith::NegFOp::create(rewriter, op.getLoc(), op.getOperand());
      rewriter.replaceOp(op, new_op);
    } else {
      mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      mlir::Value operand = op.getOperand();
      if (result_type.isUnsignedInteger()) {
        operand =
            ::xla::xtile::UnsignedIntegerToSignlessInteger(rewriter, operand);
      }
      mlir::Value zero = ::xla::xtile::ZerosLike(b, operand);
      mlir::Value sub = mlir::arith::SubIOp::create(b, zero, operand);
      if (result_type.isUnsignedInteger()) {
        auto cast = mlir::UnrealizedConversionCastOp::create(
            b, op.getResult().getType(), sub);
        rewriter.replaceOp(op, cast);
      } else {
        rewriter.replaceOp(op, sub);
      }
    }
    return mlir::success();
  }
};

Value LowerFloatToFloatConvert(ImplicitLocOpBuilder& builder, Location loc,
                               Value value, FloatType src_fp_element_ty,
                               FloatType dst_fp_element_ty, Type dst_ty) {
  Type fp16_ty = builder.getF16Type();

  if (auto dst_shaped_ty = mlir::dyn_cast<ShapedType>(dst_ty)) {
    fp16_ty =
        dst_shaped_ty.clone(dst_shaped_ty.getShape(), builder.getF16Type());
  }

  if (src_fp_element_ty.getIntOrFloatBitWidth() == 8 &&
      dst_fp_element_ty.getIntOrFloatBitWidth() == 8) {
    // FP8 <-> FP8 conversion needs to go through FP16
    auto fp16_value = mlir::arith::ExtFOp::create(builder, loc, fp16_ty, value);
    return mlir::arith::TruncFOp::create(builder, loc, dst_ty, fp16_value);
  }

  if (src_fp_element_ty.getFPMantissaWidth() >
      dst_fp_element_ty.getFPMantissaWidth()) {
    return mlir::arith::TruncFOp::create(builder, loc, dst_ty, value);
  }
  return mlir::arith::ExtFOp::create(builder, loc, dst_ty, value);
}

Value LowerIntegerToIntegerConvert(ImplicitLocOpBuilder& builder, Location loc,
                                   Value value, IntegerType src_element_ty,
                                   IntegerType dst_element_ty, Type dst_ty) {
  bool is_src_unsigned = src_element_ty.isUnsignedInteger();
  if (is_src_unsigned) {
    value = UnsignedIntegerToSignlessInteger(builder, value);
  }

  Type signless_dst_ty = dst_ty;
  if (dst_element_ty.isUnsignedInteger()) {
    Type signless_elem_ty = IntegerType::get(
        builder.getContext(), dst_element_ty.getIntOrFloatBitWidth(),
        IntegerType::SignednessSemantics::Signless);
    if (auto shaped_ty = mlir::dyn_cast<ShapedType>(dst_ty)) {
      signless_dst_ty = shaped_ty.clone(signless_elem_ty);
    } else {
      signless_dst_ty = signless_elem_ty;
    }
  }

  Value result;
  if (src_element_ty.getIntOrFloatBitWidth() <
      dst_element_ty.getIntOrFloatBitWidth()) {
    if (is_src_unsigned || src_element_ty.isInteger(1)) {
      result =
          mlir::arith::ExtUIOp::create(builder, loc, signless_dst_ty, value);
    } else {
      result =
          mlir::arith::ExtSIOp::create(builder, loc, signless_dst_ty, value);
    }
  } else if (dst_element_ty.isInteger(1)) {
    // int => bool is always value != 0.
    result = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::ne, value,
        ::xla::xtile::ZerosLike(builder, value));
  } else {
    result =
        mlir::arith::TruncIOp::create(builder, loc, signless_dst_ty, value);
  }

  if (dst_element_ty.isUnsignedInteger() && !dst_element_ty.isInteger(1)) {
    result = UnrealizedConversionCastOp::create(builder, loc, dst_ty, result)
                 .getResult(0);
  }
  return result;
}

Value LowerFloatToIntegerConvert(ImplicitLocOpBuilder& builder, Location loc,
                                 Value value, FloatType src_fp_element_ty,
                                 IntegerType dst_element_ty, Type src_ty,
                                 Type dst_ty) {
  return ::xla::emitters::EmitFloatToIntConvertWithClamping(
      builder, value, src_fp_element_ty, dst_element_ty, src_ty, dst_ty);
}

Value LowerIntegerToFloatConvert(ImplicitLocOpBuilder& builder, Location loc,
                                 Value value, IntegerType src_element_ty,
                                 Type dst_ty) {
  if (src_element_ty.isInteger(1)) {
    return mlir::arith::UIToFPOp::create(builder, loc, dst_ty, value);
  }
  if (src_element_ty.isUnsignedInteger()) {
    value = UnsignedIntegerToSignlessInteger(builder, value);
    return mlir::arith::UIToFPOp::create(builder, loc, dst_ty, value);
  }
  return mlir::arith::SIToFPOp::create(builder, loc, dst_ty, value);
}

}  // namespace

absl::StatusOr<Value> LowerConvert(ImplicitLocOpBuilder& builder, Location loc,
                                   Value value, Type src_ty, Type dst_ty) {
  Type src_element_ty = getElementTypeOrSelf(src_ty);
  Type fp32_ty = builder.getF32Type();
  Type dst_element_ty = getElementTypeOrSelf(dst_ty);

  if (src_element_ty == dst_element_ty) {
    return value;
  }

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
    return LowerIntegerToIntegerConvert(builder, loc, value, src_int_element_ty,
                                        dst_int_element_ty, dst_ty);
  }
  // int => float
  if (src_int_element_ty && dst_fp_element_ty) {
    return LowerIntegerToFloatConvert(builder, loc, value, src_int_element_ty,
                                      dst_ty);
  }
  // float => int
  if (src_fp_element_ty && dst_int_element_ty) {
    return LowerFloatToIntegerConvert(builder, loc, value, src_fp_element_ty,
                                      dst_int_element_ty, src_ty, dst_ty);
  }
  auto GetRealType = [](Type type, Type element_type) -> Type {
    if (auto shaped_ty = mlir::dyn_cast<ShapedType>(type)) {
      return shaped_ty.clone(element_type);
    }
    return element_type;
  };

  // => complex
  auto dst_complex_element_ty =
      mlir::dyn_cast<mlir::ComplexType>(dst_element_ty);
  if (dst_complex_element_ty) {
    Type real_ty = GetRealType(dst_ty, dst_complex_element_ty.getElementType());
    if (auto src_complex_elem_ty =
            mlir::dyn_cast<mlir::ComplexType>(src_element_ty)) {
      Type real_src_ty =
          GetRealType(src_ty, src_complex_elem_ty.getElementType());
      Value real_input =
          mlir::stablehlo::RealOp::create(builder, loc, real_src_ty, value);
      Value imag_input =
          mlir::stablehlo::ImagOp::create(builder, loc, real_src_ty, value);
      ABSL_ASSIGN_OR_RETURN(Value real_part, LowerConvert(builder, loc, real_input,
                                                     real_src_ty, real_ty));
      ABSL_ASSIGN_OR_RETURN(Value imag_part, LowerConvert(builder, loc, imag_input,
                                                     real_src_ty, real_ty));
      return mlir::stablehlo::ComplexOp::create(builder, loc, dst_ty, real_part,
                                                imag_part)
          .getResult();
    }
    ABSL_ASSIGN_OR_RETURN(Value real_part,
                     LowerConvert(builder, loc, value, src_ty, real_ty));
    Value imag_part = ZerosLike(builder, real_part);
    return mlir::stablehlo::ComplexOp::create(builder, loc, dst_ty, real_part,
                                              imag_part)
        .getResult();
  }
  // complex => non-complex
  auto src_complex_element_ty =
      mlir::dyn_cast<mlir::ComplexType>(src_element_ty);
  if (src_complex_element_ty) {
    Type real_ty = GetRealType(src_ty, src_complex_element_ty.getElementType());
    Value real_part =
        mlir::stablehlo::RealOp::create(builder, loc, real_ty, value);
    return LowerConvert(builder, loc, real_part, real_ty, dst_ty);
  }

  return absl::UnimplementedError(absl::StrCat(
      "Type conversion from ", ::xla::llvm_ir::DumpToString(src_ty), " to ",
      ::xla::llvm_ir::DumpToString(dst_ty), " not supported"));
}

namespace {

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
    const Type element_type = mlir::getElementTypeOrSelf(op.getLhs());
    if (mlir::isa<mlir::ComplexType>(element_type)) {
      return rewriter.notifyMatchFailure(
          op, "complex types are legalized by StablehloLegalizeToLinalg");
    }
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

struct StablehloLowerToArithPass
    : public impl::StablehloLowerToArithPassBase<StablehloLowerToArithPass> {
  using StablehloLowerToArithPassBase::StablehloLowerToArithPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<
        LowerStableHloOpToArith<mlir::stablehlo::AddOp, mlir::arith::AddFOp,
                                mlir::arith::AddIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::SubtractOp,
                                mlir::arith::SubFOp, mlir::arith::SubIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::MulOp, mlir::arith::MulFOp,
                                mlir::arith::MulIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::AndOp, mlir::arith::AndIOp,
                                mlir::arith::AndIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::OrOp, mlir::arith::OrIOp,
                                mlir::arith::OrIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::XorOp, mlir::arith::XOrIOp,
                                mlir::arith::XOrIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::DivOp, mlir::arith::DivFOp,
                                mlir::arith::DivSIOp, mlir::arith::DivUIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::RemOp, mlir::arith::RemFOp,
                                mlir::arith::RemSIOp, mlir::arith::RemUIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::MaxOp, mlir::arith::MaximumFOp,
                                mlir::arith::MaxSIOp, mlir::arith::MaxUIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::MinOp, mlir::arith::MinimumFOp,
                                mlir::arith::MinSIOp, mlir::arith::MinUIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::PowOp, mlir::math::PowFOp,
                                mlir::math::IPowIOp>,
        LowerStableHloOpToArith<mlir::stablehlo::AbsOp, mlir::math::AbsFOp,
                                mlir::math::AbsIOp>,
        LowerStableHloNegOpToArith,
        LowerStableHloUnaryOpToMath<mlir::stablehlo::RoundNearestEvenOp,
                                    mlir::math::RoundEvenOp>,
        LowerConvertOp, LowerCompareOp>(mlir_context);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace xla::xtile
