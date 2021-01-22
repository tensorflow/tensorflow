/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Enable the use of M_* math constants.
// NOTE: this must be first in the file to ensure that if cmath is transitively
// included by any other header it has the define set on first processing.
// https://docs.microsoft.com/en-us/cpp/c-runtime-library/math-constants
#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>
#include <vector>

#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_chlo_to_hlo_op.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir-hlo/utils/broadcast_utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace chlo {
namespace {

struct ConvertConstantLikeOp : public OpConversionPattern<ConstantLikeOp> {
  using OpConversionPattern<ConstantLikeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ConstantLikeOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto result_ty = op.getType().cast<ShapedType>();

    // Unranked uses are not supported.  Consider `transform-unranked-hlo`.
    if (!result_ty.hasRank()) return failure();

    // Lower to MHLO constant if statically shaped.
    if (result_ty.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<mhlo::ConstOp>(
          op, DenseElementsAttr::get(result_ty, op.value()));
      return success();
    }

    // Lower to broadcasted constant.
    ConstantLikeOp::Adaptor transformed(operands);
    auto loc = op.getLoc();
    Type extent_tensor_type = shape::getExtentTensorType(op.getContext());
    Value constant = rewriter.create<mhlo::ConstOp>(loc, op.value());
    Value uncasted_shape = rewriter.create<shape::ShapeOfOp>(
        loc, extent_tensor_type, transformed.operand());
    Type shape_ty =
        RankedTensorType::get({result_ty.getRank()}, rewriter.getIndexType());
    Value shape =
        rewriter.create<tensor::CastOp>(loc, shape_ty, uncasted_shape);
    rewriter.replaceOpWithNewOp<mhlo::DynamicBroadcastInDimOp>(
        op, result_ty, constant, shape, rewriter.getI64TensorAttr({}));
    return success();
  }
};

template <typename FTy>
Value MaterializePolynomialApproximation(ConversionPatternRewriter &rewriter,
                                         Location loc, Value x,
                                         const std::vector<FTy> &coefficients) {
  Value poly = chlo::getConstantLike(rewriter, loc, 0.0, x);
  for (FTy c : coefficients) {
    poly = rewriter.create<mhlo::MulOp>(loc, x.getType(), poly, x);
    poly = rewriter.create<mhlo::AddOp>(
        loc, x.getType(), poly, chlo::getConstantLike(rewriter, loc, c, x));
  }
  return poly;
}

// Precondition is |x| >= 1. Use erf approximation, otherwise.
//
// We rely on multiple polynomial approximations for x >= 1. We pass |x| as an
// argument and derive the final approximation for all |x| >= 1.
// This implementation is based on Cephes.
Value MaterializeErfcApproximationF64ForMagnituteGEOne(
    ConversionPatternRewriter &rewriter, Location loc, Value x) {
  assert(x.getType().cast<ShapedType>().getElementType().isF64() &&
         "expect f64 element type");
  const double kMaxlog = 7.09782712893383996843E2;
  const std::vector<double> kErfcPCoefficients{
      2.46196981473530512524E-10, 5.64189564831068821977E-1,
      7.46321056442269912687E0,   4.86371970985681366614E1,
      1.96520832956077098242E2,   5.26445194995477358631E2,
      9.34528527171957607540E2,   1.02755188689515710272E3,
      5.57535335369399327526E2};
  const std::vector<double> kErfcQCoefficients{
      1.00000000000000000000E0, 1.32281951154744992508E1,
      8.67072140885989742329E1, 3.54937778887819891062E2,
      9.75708501743205489753E2, 1.82390916687909736289E3,
      2.24633760818710981792E3, 1.65666309194161350182E3,
      5.57535340817727675546E2};
  const std::vector<double> kErfcRCoefficients{
      5.64189583547755073984E-1, 1.27536670759978104416E0,
      5.01905042251180477414E0,  6.16021097993053585195E0,
      7.40974269950448939160E0,  2.97886665372100240670E0};
  const std::vector<double> kErfcSCoefficients{
      1.00000000000000000000E0, 2.26052863220117276590E0,
      9.39603524938001434673E0, 1.20489539808096656605E1,
      1.70814450747565897222E1, 9.60896809063285878198E0,
      3.36907645100081516050E0};

  // Let z = -x^2.
  Value x_sq = rewriter.create<mhlo::MulOp>(loc, x, x);
  Value z = rewriter.create<mhlo::NegOp>(loc, x_sq);

  // Materialize polynomial approximation for x in [1, 8) as
  //   erfc(x) = exp(z) P(|x|) / Q(|x|).
  Value exp_z = rewriter.create<mhlo::ExpOp>(loc, z);
  Value abs_x = rewriter.create<mhlo::AbsOp>(loc, x);
  Value poly_p = MaterializePolynomialApproximation(rewriter, loc, abs_x,
                                                    kErfcPCoefficients);
  Value exp_z_mul_poly_p = rewriter.create<mhlo::MulOp>(loc, exp_z, poly_p);
  Value poly_q = MaterializePolynomialApproximation(rewriter, loc, abs_x,
                                                    kErfcQCoefficients);
  Value erfc_approx_1_8 =
      rewriter.create<mhlo::DivOp>(loc, exp_z_mul_poly_p, poly_q);

  // Materialize polynomial approximation for x in >= 8 as
  //   erfc(x) exp(z) R(|x|) / S(|x|).
  Value poly_r = MaterializePolynomialApproximation(rewriter, loc, abs_x,
                                                    kErfcRCoefficients);
  Value exp_z_mul_poly_r = rewriter.create<mhlo::MulOp>(loc, exp_z, poly_r);
  Value poly_s = MaterializePolynomialApproximation(rewriter, loc, abs_x,
                                                    kErfcSCoefficients);
  Value erfc_approx_8_inf =
      rewriter.create<mhlo::DivOp>(loc, exp_z_mul_poly_r, poly_s);

  // Combine polynomial approximations for x >= 1.
  const StringAttr kLT = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::LT));
  Value eight = chlo::getConstantLike(rewriter, loc, 8.0, x);
  Value abs_x_lt_8 = rewriter.create<mhlo::CompareOp>(loc, abs_x, eight, kLT);
  Value erfc_approx = rewriter.create<mhlo::SelectOp>(
      loc, abs_x_lt_8, erfc_approx_1_8, erfc_approx_8_inf);

  // Clamp to prevent overflow and materialize approximation for large x as
  //   erfc(x) = 0.
  Value z_lt_neg_maxlog = rewriter.create<mhlo::CompareOp>(
      loc, z, chlo::getConstantLike(rewriter, loc, -kMaxlog, x), kLT);
  Value zero = chlo::getConstantLike(rewriter, loc, 0.0, x);
  Value erfc_approx_clamped =
      rewriter.create<mhlo::SelectOp>(loc, z_lt_neg_maxlog, zero, erfc_approx);

  // Derive approximation for x <= -1 as
  //   erfc(x) = 2 - erfc(-x).
  // Reuse previously materialized approximations all of which take |x| as their
  // argument.
  Value x_lt_zero = rewriter.create<mhlo::CompareOp>(loc, x, zero, kLT);
  Value two = chlo::getConstantLike(rewriter, loc, 2.0, x);
  Value two_sub_erfc_approx_clamped =
      rewriter.create<mhlo::SubOp>(loc, two, erfc_approx_clamped);
  return rewriter.create<mhlo::SelectOp>(
      loc, x_lt_zero, two_sub_erfc_approx_clamped, erfc_approx_clamped);
}

// Precondition is |x| <= 1. Use erfc approximation, otherwise.
// This implementation is based on Cephes.
Value MaterializeErfApproximationF64ForMagnituteLEOne(
    ConversionPatternRewriter &rewriter, Location loc, Value x) {
  assert(x.getType().cast<ShapedType>().getElementType().isF64() &&
         "expect f64 element type");
  const std::vector<double> kErfTCoefficients{
      9.60497373987051638749E0, 9.00260197203842689217E1,
      2.23200534594684319226E3, 7.00332514112805075473E3,
      5.55923013010394962768E4};
  const std::vector<double> kErfUCoefficients{
      1.00000000000000000000E0, 3.35617141647503099647E1,
      5.21357949780152679795E2, 4.59432382970980127987E3,
      2.26290000613890934246E4, 4.92673942608635921086E4};

  // Materialize polynomial approximation for |x| <= 1 as
  //   erf(x) = x T(x^2) / U(x^2).
  Value x_sq = rewriter.create<mhlo::MulOp>(loc, x, x);
  Value poly_t = MaterializePolynomialApproximation(rewriter, loc, x_sq,
                                                    kErfTCoefficients);
  Value x_mul_poly_t = rewriter.create<mhlo::MulOp>(loc, x, poly_t);
  Value poly_u = MaterializePolynomialApproximation(rewriter, loc, x_sq,
                                                    kErfUCoefficients);
  return rewriter.create<mhlo::DivOp>(loc, x_mul_poly_t, poly_u);
}

// This implementation is based on Cephes.
Value MaterializeErfApproximationF64(ConversionPatternRewriter &rewriter,
                                     Location loc, Value x) {
  assert(x.getType().cast<ShapedType>().getElementType().isF64() &&
         "expect f64 element type");

  // Rely on erf approximation for |x| < 1
  //   erf(x) = erf_approx(x)
  Value erf_approx =
      MaterializeErfApproximationF64ForMagnituteLEOne(rewriter, loc, x);

  // Rely on erfc approximation for |x| >= 1 and materialize erf as
  //   erf(x) = 1 - erfc_approx(x)
  Value one = chlo::getConstantLike(rewriter, loc, 1.0, x);
  Value erfc_approx =
      MaterializeErfcApproximationF64ForMagnituteGEOne(rewriter, loc, x);
  Value erfc_based_approx = rewriter.create<mhlo::SubOp>(loc, one, erfc_approx);

  // Materialize approximation selection based on argument.
  Value abs_x = rewriter.create<mhlo::AbsOp>(loc, x);
  const StringAttr kLT = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::LT));
  Value abs_x_lt_one = rewriter.create<mhlo::CompareOp>(loc, abs_x, one, kLT);
  return rewriter.create<mhlo::SelectOp>(loc, abs_x_lt_one, erf_approx,
                                         erfc_based_approx);
}

Value MaterializeErfcApproximationF64(ConversionPatternRewriter &rewriter,
                                      Location loc, Value x) {
  assert(x.getType().cast<ShapedType>().getElementType().isF64() &&
         "expect f64 element type");

  // Rely on erfc approximation for |x| >= 1
  //   erfc(x) = erfc_approx(x)
  Value erfc_approx =
      MaterializeErfcApproximationF64ForMagnituteGEOne(rewriter, loc, x);

  // Rely on erf approximation for |x| < 1 and materialize erfc as
  //   erfc(x) = 1 - erf_approx(x)
  Value one = chlo::getConstantLike(rewriter, loc, 1.0, x);
  Value erf_approx =
      MaterializeErfApproximationF64ForMagnituteLEOne(rewriter, loc, x);
  Value erf_based_approx = rewriter.create<mhlo::SubOp>(loc, one, erf_approx);

  // Materialize approximation selection based on argument.
  Value abs_x = rewriter.create<mhlo::AbsOp>(loc, x);
  const StringAttr kLT = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::LT));
  Value abs_x_lt_one = rewriter.create<mhlo::CompareOp>(loc, abs_x, one, kLT);
  return rewriter.create<mhlo::SelectOp>(loc, abs_x_lt_one, erf_based_approx,
                                         erfc_approx);
}

// Precondition is |x| >= 1. Use erf approximation, otherwise.
//
// We rely on multiple polynomial approximations for x >= 1. We pass |x| as an
// argument and derive the final approximation for all |x| >= 1.
// This implementation is based on Cephes.
Value MaterializeErfcApproximationF32ForMagnitudeGEOne(
    ConversionPatternRewriter &rewriter, Location loc, Value x) {
  assert(x.getType().cast<ShapedType>().getElementType().isF32() &&
         "expect f32 element type");
  const double kMaxlog = 88.72283905206835;
  const std::vector<float> kErfcPCoefficients{
      +2.326819970068386E-2, -1.387039388740657E-1, +3.687424674597105E-1,
      -5.824733027278666E-1, +6.210004621745983E-1, -4.944515323274145E-1,
      +3.404879937665872E-1, -2.741127028184656E-1, +5.638259427386472E-1,
  };
  const std::vector<float> kErfcRCoefficients{
      -1.047766399936249E+1, +1.297719955372516E+1, -7.495518717768503E+0,
      +2.921019019210786E+0, -1.015265279202700E+0, +4.218463358204948E-1,
      -2.820767439740514E-1, +5.641895067754075E-1,
  };

  // Let z = -x^2.
  Value x_sq = rewriter.create<mhlo::MulOp>(loc, x, x);
  Value z = rewriter.create<mhlo::NegOp>(loc, x_sq);

  // Materialize polynomial approximation for x >= 1 as
  //   erfc(x) = exp(z) 1/x P(1/x^2)   if x in [1, 2)
  //   erfc(x) = exp(z) 1/x R(1/x^2)   if x >= 2
  const StringAttr kLT = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::LT));
  Value abs_x = rewriter.create<mhlo::AbsOp>(loc, x);
  Value one = chlo::getConstantLike(rewriter, loc, 1.0, x);
  Value reciprocal_x_sq = rewriter.create<mhlo::DivOp>(loc, one, x_sq);
  Value exp_z = rewriter.create<mhlo::ExpOp>(loc, z);
  Value one_div_abs_x = rewriter.create<mhlo::DivOp>(loc, one, abs_x);
  Value exp_z_mul_one_div_abs_x =
      rewriter.create<mhlo::MulOp>(loc, exp_z, one_div_abs_x);
  Value two = chlo::getConstantLike(rewriter, loc, 2.0, x);
  Value abs_x_lt_two = rewriter.create<mhlo::CompareOp>(loc, abs_x, two, kLT);
  Value poly_p = MaterializePolynomialApproximation(
      rewriter, loc, reciprocal_x_sq, kErfcPCoefficients);
  Value poly_r = MaterializePolynomialApproximation(
      rewriter, loc, reciprocal_x_sq, kErfcRCoefficients);
  Value poly =
      rewriter.create<mhlo::SelectOp>(loc, abs_x_lt_two, poly_p, poly_r);
  Value erfc_approx =
      rewriter.create<mhlo::MulOp>(loc, exp_z_mul_one_div_abs_x, poly);

  // Clamp to prevent overflow and materialize approximation for large x as
  //   erfc(x) = 0.
  Value z_lt_neq_maxlog = rewriter.create<mhlo::CompareOp>(
      loc, z, chlo::getConstantLike(rewriter, loc, -kMaxlog, x), kLT);
  Value zero = chlo::getConstantLike(rewriter, loc, 0.0, x);
  Value erfc_approx_clamped =
      rewriter.create<mhlo::SelectOp>(loc, z_lt_neq_maxlog, zero, erfc_approx);

  // Derive approximation for x <= -1 as
  //   erfc(x) = 2 - erfc(-x).
  // Reuse previously materialized approximations all of which take |x| as their
  // argument.
  Value x_lt_zero = rewriter.create<mhlo::CompareOp>(loc, x, zero, kLT);
  Value two_sub_erfc_approx =
      rewriter.create<mhlo::SubOp>(loc, two, erfc_approx_clamped);
  return rewriter.create<mhlo::SelectOp>(loc, x_lt_zero, two_sub_erfc_approx,
                                         erfc_approx_clamped);
}

// Precondition is |x| <= 1. Use erfc approximation, otherwise.
// This implementation is based on Cephes.
Value MaterializeErfApproximationF32ForMagnitudeLEOne(
    ConversionPatternRewriter &rewriter, Location loc, Value x) {
  assert(x.getType().cast<ShapedType>().getElementType().isF32() &&
         "expect f32 element type");
  const std::vector<float> kErfTCoefficients{
      +7.853861353153693E-5, -8.010193625184903E-4, +5.188327685732524E-3,
      -2.685381193529856E-2, +1.128358514861418E-1, -3.761262582423300E-1,
      +1.128379165726710E+0,
  };

  // Materialize polynomial approximation for |x| <= 1 as
  //   erf(x) = x T(x^2).
  Value x_sq = rewriter.create<mhlo::MulOp>(loc, x, x);
  Value poly_t = MaterializePolynomialApproximation(rewriter, loc, x_sq,
                                                    kErfTCoefficients);
  return rewriter.create<mhlo::MulOp>(loc, x, poly_t);
}

// This is the same approximation as used in Eigen.
Value MaterializeErfApproximationF32(ConversionPatternRewriter &rewriter,
                                     Location loc, Value operand) {
  assert(operand.getType().cast<ShapedType>().getElementType().isF32() &&
         "expect f32 element type");
  const std::vector<float> kAlpha{
      -2.72614225801306e-10f, 2.77068142495902e-08f,  -2.10102402082508e-06f,
      -5.69250639462346e-05f, -7.34990630326855e-04f, -2.95459980854025e-03f,
      -1.60960333262415e-02f,
  };
  const std::vector<float> kBeta{
      -1.45660718464996e-05f, -2.13374055278905e-04f, -1.68282697438203e-03f,
      -7.37332916720468e-03f, -1.42647390514189e-02f,
  };

  // Clamp argument between -4 and 4.
  Value lb = chlo::getConstantLike(rewriter, loc, -4.0, operand);
  Value ub = chlo::getConstantLike(rewriter, loc, 4.0, operand);
  Value x =
      rewriter.create<mhlo::ClampOp>(loc, operand.getType(), lb, operand, ub);
  Value x_sq = rewriter.create<mhlo::MulOp>(loc, x, x);

  // Materialize polynomial approximation for x in [-4, 4] as
  //   erf(x) = x * Alpha(x^2) / Beta(x^2).
  Value alpha_poly =
      MaterializePolynomialApproximation(rewriter, loc, x_sq, kAlpha);
  Value beta_poly =
      MaterializePolynomialApproximation(rewriter, loc, x_sq, kBeta);
  Value x_mul_alpha_poly = rewriter.create<mhlo::MulOp>(loc, x, alpha_poly);
  return rewriter.create<mhlo::DivOp>(loc, x_mul_alpha_poly, beta_poly);
}

Value MaterializeErfcApproximationF32(ConversionPatternRewriter &rewriter,
                                      Location loc, Value x) {
  assert(x.getType().cast<ShapedType>().getElementType().isF32() &&
         "expect f32 element type");

  // Rely on erfc approximation for |x| >= 1
  //   erfc(x) = erfc_approx(x)
  Value erfc_approx =
      MaterializeErfcApproximationF32ForMagnitudeGEOne(rewriter, loc, x);

  // Rely on erf approximation for |x| < 1 and materialize erfc as
  //   erfc(x) = 1 - erf_approx(x)
  Value one = chlo::getConstantLike(rewriter, loc, 1.0, x);
  Value erf_approx =
      MaterializeErfApproximationF32ForMagnitudeLEOne(rewriter, loc, x);
  Value erf_based_approx = rewriter.create<mhlo::SubOp>(loc, one, erf_approx);

  // Materialize approximation selection based on argument.
  const StringAttr kLT = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::LT));
  Value abs_x = rewriter.create<mhlo::AbsOp>(loc, x);
  Value abs_x_lt_one = rewriter.create<mhlo::CompareOp>(loc, abs_x, one, kLT);
  return rewriter.create<mhlo::SelectOp>(loc, abs_x_lt_one, erf_based_approx,
                                         erfc_approx);
}

struct ConvertErfOp : public OpConversionPattern<ErfOp> {
  using OpConversionPattern<ErfOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ErfOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ErfOp::Adaptor transformed(operands);
    Value x = transformed.operand();
    Type ty = x.getType().cast<ShapedType>().getElementType();

    // For now, we support only f64, f32, and f16.
    if (!ty.isF64() && !ty.isF32() && !ty.isF16()) return failure();

    if (ty.isF64()) {
      rewriter.replaceOp(op, MaterializeErfApproximationF64(rewriter, loc, x));
      return success();
    }

    // Cast argument to f32 tensor if needed.
    assert((ty.isF16() || ty.isF32()) && "expect f16 or f32 at this point");
    if (ty.isF16()) {
      x = rewriter.create<mhlo::ConvertOp>(loc, x, rewriter.getF32Type());
    }

    Value result = MaterializeErfApproximationF32(rewriter, loc, x);

    // Cast back if needed.
    if (ty.isF16()) {
      result = rewriter.create<mhlo::ConvertOp>(loc, result, ty);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertErfcOp : public OpConversionPattern<ErfcOp> {
  using OpConversionPattern<ErfcOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ErfcOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ErfcOp::Adaptor transformed(operands);
    Value x = transformed.operand();
    Type ty = x.getType().cast<ShapedType>().getElementType();

    // For now, we support only f64 and f32.
    if (!ty.isF64() && !ty.isF32()) return failure();

    if (ty.isF64()) {
      rewriter.replaceOp(op, MaterializeErfcApproximationF64(rewriter, loc, x));
      return success();
    }

    rewriter.replaceOp(op, MaterializeErfcApproximationF32(rewriter, loc, x));
    return success();
  }
};

// Converts binary ops that statically are determined to not broadcast directly
// to the corresponding mhlo non-broadcasting op.
template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
struct ConvertTrivialNonBroadcastBinaryOp
    : public OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ChloOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only rewrite for statically determinable non-broadcasting cases.
    typename ChloOpTy::Adaptor transformed(operands);
    auto lhs_type =
        transformed.lhs().getType().template dyn_cast<RankedTensorType>();
    auto rhs_type =
        transformed.rhs().getType().template dyn_cast<RankedTensorType>();
    if (!lhs_type || !rhs_type) return failure();

    // Requires rank broadcast.
    if (lhs_type.getRank() != rhs_type.getRank()) return failure();
    // Any dynamic dimension may require broadcasting and requires more
    // analysis.
    if (!lhs_type.hasStaticShape() || !rhs_type.hasStaticShape())
      return failure();

    for (auto extents : llvm::zip(lhs_type.getShape(), rhs_type.getShape())) {
      auto lhs_extent = std::get<0>(extents);
      auto rhs_extent = std::get<1>(extents);
      if (lhs_extent != rhs_extent) {
        return failure();
      }
    }

    rewriter.replaceOp(
        op, {Adaptor::CreateOp(op, op.getResult().getType(), operands[0],
                               operands[1], rewriter)});
    return success();
  }
};

// Converts a binary op with ranked broadcasting operands to explicitly
// broadcast and invoke the corresponding mhlo non-broadcasting op.
// Note that dynamic broadcasting supported by this pattern is only valid for
// "numpy" broadcasting semantics as defined here:
//   https://docs.scipy.org/doc/numpy/reference/ufuncs.html
// Specifically, this includes the following cases:
//   - Same rank broadcast (operands have the same static rank).
//   - Different-rank broadcast, either without a broadcast_dims attribte or
//     with the broadcast_dims attribute set to map to a prefix padding.
//   - Legal combinations of degenerate (1-dim) implicit broadcasting.
// The restriction on broadcast_dims derives from the definition of the
// `shape.broadcast` op, which only supports prefix-padding.
template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
struct ConvertRankedDynamicBroadcastBinaryOp
    : public OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ChloOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only support ranked operands.
    typename ChloOpTy::Adaptor transformed(operands);
    Value lhs = transformed.lhs();
    Value rhs = transformed.rhs();
    auto lhs_type = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhs_type = rhs.getType().dyn_cast<RankedTensorType>();
    auto result_type =
        op.getResult().getType().template dyn_cast<RankedTensorType>();
    if (!lhs_type || !rhs_type || !result_type) return failure();

    // Check for "numpy"-style rank broadcast.
    auto broadcast_dimensions = op.broadcast_dimensions();
    if (broadcast_dimensions &&
        !hlo::IsLegalNumpyRankedBroadcast(lhs, rhs, *broadcast_dimensions)) {
      // Note: It is unclear whether the general specification of explicit
      // broadcast_dimensions on binary ops is a feature we want to carry
      // forward. While it can technically be implemented for ranked-dynamic,
      // it is incompatible with unranked inputs. If this warning is emitted
      // in real programs, it is an indication that the feature should be
      // implemented versus just falling back on the more standard definition
      // of numpy-like prefix-padding.
      op.emitWarning() << "unsupported non prefix-padded dynamic rank "
                       << "broadcast_dimensions = " << *broadcast_dimensions;
      return failure();
    }

    // Compute result shape.
    auto loc = op.getLoc();

    // Insert a constraint on the shapes being broadcastable and insert all
    // future code into an assuming block reliant on the constraint.
    Value lhs_shape = rewriter.create<shape::ShapeOfOp>(loc, lhs);
    Value rhs_shape = rewriter.create<shape::ShapeOfOp>(loc, rhs);
    auto broadcastable_cstr =
        rewriter.create<shape::CstrBroadcastableOp>(loc, lhs_shape, rhs_shape);
    auto assuming_op = rewriter.create<shape::AssumingOp>(
        loc, ArrayRef<Type>{result_type}, broadcastable_cstr.result());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&assuming_op.doRegion());

    int64_t result_rank = std::max(lhs_type.getRank(), rhs_type.getRank());
    Value result_extents =
        hlo::ComputeBinaryElementwiseBroadcastingResultExtents(
            loc, lhs, rhs, rewriter, /*unsafe_as_extent_tensor=*/true);

    // Note that we unconditionally emit DynamicBroadcastInDim ops and let
    // downstream canonicalizations fold them away if possible. This is
    // because, in the dynamic case, there are many corner cases regarding
    // when it is safe to omit, and some of them require analysis to prove
    // properly.
    auto lhs_broadcast_dimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(result_rank - lhs_type.getRank(), result_rank));
    Value broadcasted_lhs = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc,
        RankedTensorType::get(result_type.getShape(),
                              lhs_type.getElementType()),
        lhs, result_extents,
        rewriter.getI64TensorAttr(lhs_broadcast_dimensions));
    auto rhs_broadcast_dimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(result_rank - rhs_type.getRank(), result_rank));
    Value broadcasted_rhs = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc,
        RankedTensorType::get(result_type.getShape(),
                              rhs_type.getElementType()),
        rhs, result_extents,
        rewriter.getI64TensorAttr(rhs_broadcast_dimensions));

    // And generate the final non-broadcasted binary op.
    Value final_result = Adaptor::CreateOp(op, result_type, broadcasted_lhs,
                                           broadcasted_rhs, rewriter);
    rewriter.create<shape::AssumingYieldOp>(loc, final_result);
    rewriter.replaceOp(op, {assuming_op.getResult(0)});
    return success();
  }
};

#include "generated_chlo_legalize_to_hlo.inc"
}  // namespace

void PopulateLegalizeChloToHloPatterns(MLIRContext *context,
                                       OwningRewritePatternList *patterns) {
  populateWithGenerated(context, *patterns);

  // Instantiate conversion templates for conforming binary elementwise ops
  // that do not have different dtypes between operands and results and do
  // not have special attributes that need to be preserved.
  PopulateForBroadcastingBinaryOp<ConvertTrivialNonBroadcastBinaryOp>(
      context, patterns, 10);
  PopulateForBroadcastingBinaryOp<ConvertRankedDynamicBroadcastBinaryOp>(
      context, patterns, 5);

  // Other patterns.
  patterns->insert<ConvertConstantLikeOp, ConvertErfOp, ConvertErfcOp>(context);
}

}  // namespace chlo
}  // namespace mlir
