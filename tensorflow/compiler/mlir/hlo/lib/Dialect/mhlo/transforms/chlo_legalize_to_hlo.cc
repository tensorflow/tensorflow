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

#include "llvm/ADT/SmallVector.h"
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
#include "mlir/IR/ImplicitLocOpBuilder.h"
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

    // Unranked uses are not supported.  Consider `mhlo-transform-unranked-hlo`.
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
    ConversionPatternRewriter &rewriter, Location loc, ValueRange args) {
  Value x = args.front();
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
    ConversionPatternRewriter &rewriter, Location loc, ValueRange args) {
  Value x = args.front();
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
                                     Location loc, ValueRange args) {
  Value x = args.front();
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
                                      Location loc, ValueRange args) {
  Value x = args.front();
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
    ConversionPatternRewriter &rewriter, Location loc, ValueRange args) {
  Value x = args.front();
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
    ConversionPatternRewriter &rewriter, Location loc, ValueRange args) {
  Value x = args.front();
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
                                     Location loc, ValueRange args) {
  Value x = args.front();
  assert(x.getType().cast<ShapedType>().getElementType().isF32() &&
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
  Value lb = chlo::getConstantLike(rewriter, loc, -4.0, x);
  Value ub = chlo::getConstantLike(rewriter, loc, 4.0, x);
  x = rewriter.create<mhlo::ClampOp>(loc, x.getType(), lb, x, ub);
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
                                      Location loc, ValueRange args) {
  Value x = args.front();
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

Value MaterializeWithUpcast(ConversionPatternRewriter &rewriter, Location loc,
                            ValueRange args, FloatType min_precision_ty,
                            Value callback(ConversionPatternRewriter &,
                                           Location, ValueRange)) {
  auto original_ty =
      getElementTypeOrSelf(args.front().getType()).cast<FloatType>();
  bool needs_upcast = original_ty.getWidth() < min_precision_ty.getWidth();

  // Upcast arguments if necessary.
  llvm::SmallVector<Value, 2> casted_args;
  if (needs_upcast) {
    for (Value a : args) {
      casted_args.push_back(
          rewriter.create<mhlo::ConvertOp>(loc, a, min_precision_ty));
    }
    args = casted_args;
  }

  Value result = callback(rewriter, loc, args);

  // Cast back if necessary.
  if (needs_upcast) {
    result = rewriter.create<mhlo::ConvertOp>(loc, result, original_ty);
  }

  return result;
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

    rewriter.replaceOp(op, MaterializeWithUpcast(
                               rewriter, loc, operands, rewriter.getF32Type(),
                               &MaterializeErfApproximationF32));
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

    // For now, we support only f64, f32, and f16.
    if (!ty.isF64() && !ty.isF32() && !ty.isF16()) return failure();

    if (ty.isF64()) {
      rewriter.replaceOp(op, MaterializeErfcApproximationF64(rewriter, loc, x));
      return success();
    }

    rewriter.replaceOp(op, MaterializeWithUpcast(
                               rewriter, loc, operands, rewriter.getF32Type(),
                               &MaterializeErfcApproximationF32));
    return success();
  }
};

// Coefficients for the Lanczos approximation of the gamma function. The
// coefficients are uniquely determined by the choice of g and n (kLanczosGamma
// and kLanczosCoefficients.size() + 1). The coefficients below correspond to
// [7, 9]. [5, 7], [7, 9], [9, 10], and [607/128.0, 15] were evaluated and
// [7, 9] seemed to be the least sensitive to the quality of the log function.
// In particular, [5, 7] is the only choice where -1.5e-5 <= lgamma(2) <= 1.5e-5
// for a particularly inaccurate log function.
constexpr double kLanczosGamma = 7;  // aka g
constexpr double kBaseLanczosCoeff = 0.99999999999980993227684700473478;
constexpr std::array<double, 8> kLanczosCoefficients = {
    676.520368121885098567009190444019, -1259.13921672240287047156078755283,
    771.3234287776530788486528258894,   -176.61502916214059906584551354,
    12.507343278686904814458936853,     -0.13857109526572011689554707,
    9.984369578019570859563e-6,         1.50563273514931155834e-7};

// Compute the Lgamma function using Lanczos' approximation from "A Precision
// Approximation of the Gamma Function". SIAM Journal on Numerical Analysis
// series B. Vol. 1:
//   lgamma(z + 1) = (log(2) + log(pi)) / 2
//                     + (z + 1/2) * log(t(z))
//                     - t(z) + log(a(z))
//   with   t(z) = z + kLanczosGamma + 1/2
//          a(z) = kBaseLanczosCoeff
//                   + sum(k = 1, n, kLanczosCoefficients[i] / (z + k))
Value MaterializeLgamma(ConversionPatternRewriter &rewriter, Location loc,
                        ValueRange args) {
  // If the input is less than 0.5 use Euler's reflection formula.
  //   gamma(x) = pi / (sin(pi * x) * gamma(1 - x))
  // Let z be
  //   z = -x      if x < 1/2
  //   z = x - 1   otheriwse
  Value x = args.front();
  const StringAttr kLT = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::LT));
  Value half = getConstantLike(rewriter, loc, 0.5, x);
  Value need_to_reflect = rewriter.create<mhlo::CompareOp>(loc, x, half, kLT);
  Value neg_x = rewriter.create<mhlo::NegOp>(loc, x);
  Value one = getConstantLike(rewriter, loc, 1, x);
  Value x_sub_one = rewriter.create<mhlo::SubOp>(loc, x, one);
  Value z =
      rewriter.create<mhlo::SelectOp>(loc, need_to_reflect, neg_x, x_sub_one);

  // Materialize
  //   a(z) = kBaseLanczosCoeff
  //            + sum(k = 1, n, kLanczosCoefficients[i] / (z + k))
  Value a = getConstantLike(rewriter, loc, kBaseLanczosCoeff, x);
  for (int i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
    Value coeff = getConstantLike(rewriter, loc, kLanczosCoefficients[i], x);
    Value one_based_index = getConstantLike(rewriter, loc, i + 1, x);
    Value quotient = rewriter.create<mhlo::DivOp>(
        loc, coeff, rewriter.create<mhlo::AddOp>(loc, z, one_based_index));
    a = rewriter.create<mhlo::AddOp>(loc, a, quotient);
  }

  // To improve accuracy on platforms with less-precise log implementations,
  // compute log(kLanczosGamma + 1/2) at compile time and use log1p on the
  // device.
  // Materialize as
  //   log(t) = log(kLanczosGamma + 1/2 + z)
  //          = log(kLanczosGamma + 1/2) + log1p(z / (kLanczosGamma + 1/2)).
  Value lanczos_plus_half =
      getConstantLike(rewriter, loc, kLanczosGamma + 0.5, x);
  Value t = rewriter.create<mhlo::AddOp>(loc, lanczos_plus_half, z);
  Value log_term =
      getConstantLike(rewriter, loc, std::log(kLanczosGamma + 0.5), x);
  Value log1p_term = rewriter.create<mhlo::Log1pOp>(
      loc, rewriter.create<mhlo::DivOp>(loc, z, lanczos_plus_half));
  Value log_t = rewriter.create<mhlo::AddOp>(loc, log_term, log1p_term);

  // Note that t(z) may be large and we need to be careful not to overflow to
  // infinity in the relevant term
  //   r = (z + 1/2) * log(t(z)) - t(z).
  // Therefore, we compute this as
  //   r = (z + 1/2 - t(z) / log(t(z))) * log(t(z)).
  Value t_div_log_t = rewriter.create<mhlo::DivOp>(loc, t, log_t);
  Value sum = rewriter.create<mhlo::SubOp>(
      loc, rewriter.create<mhlo::AddOp>(loc, z, half), t_div_log_t);
  Value r = rewriter.create<mhlo::MulOp>(loc, sum, log_t);

  // Compute the final result (modulo reflection) as
  //   lgamma(z + 1) = (log(2) + log(pi)) / 2 + r + log(a(z)).
  Value log_a = rewriter.create<mhlo::LogOp>(loc, a);
  Value lgamma = rewriter.create<mhlo::AddOp>(
      loc,
      rewriter.create<mhlo::AddOp>(
          loc,
          getConstantLike(rewriter, loc, (std::log(2) + std::log(M_PI)) / 2, x),
          r),
      log_a);

  // Compute the reflected value for x < 0.5 as
  //   lgamma(x) = log(pi) - lgamma(1-x) - log(abs(sin(pi * x))).
  //
  // The abs is needed because lgamma is the log of the absolute value of the
  // gamma function.
  //
  // We have to be careful when computing the final term above. gamma(x) goes
  // to +/-inf at every integer x < 0, and this is controlled by the sin(pi * x)
  // term. The slope is large, so precision is particularly important.
  //
  // Because abs(sin(pi * x)) has period of 1 we can equivalently use
  // abs(sin(pi * frac(x))) where frac(x) is the fractional part of x. This is
  // more numerically accurate: It doesn't overflow to inf like pi * x would and
  // if x is an integer it evaluates to exactly 0 which is important because we
  // then take the log of this value, and log(0) is inf.
  //
  // We don't have a frac(x) primitive in HLO and computing it is tricky, but
  // because abs(sin(pi * x)) = abs(sin(pi * abs(x))), it's good enough for our
  // purposes to use abs(frac(x)) = abs(x) - floor(abs(x)).
  //
  // Furthermore, pi * abs(frac(x)) loses precision when abs(frac(x)) is close
  // to 1. To remedy this, we can use the fact that sin(pi * x) in the domain
  // [0, 1] is symmetric across the line Y=0.5.
  //

  // Convert values of abs_frac > 0.5 to (1 - abs_frac) to improve precision of
  // pi * abs_frac for values of abs_frac close to 1.
  Value abs = rewriter.create<mhlo::AbsOp>(loc, x);
  Value abs_frac = rewriter.create<mhlo::SubOp>(
      loc, abs, rewriter.create<mhlo::FloorOp>(loc, abs));
  Value reduce_abs_frac =
      rewriter.create<mhlo::CompareOp>(loc, half, abs_frac, kLT);
  abs_frac = rewriter.create<mhlo::SelectOp>(
      loc, reduce_abs_frac, rewriter.create<mhlo::SubOp>(loc, one, abs_frac),
      abs_frac);

  // Materialize reflection.
  Value reflection_denom = rewriter.create<mhlo::LogOp>(
      loc,
      rewriter.create<mhlo::SinOp>(
          loc, rewriter.create<mhlo::MulOp>(
                   loc, getConstantLike(rewriter, loc, M_PI, x), abs_frac)));
  Value lgamma_reflection = rewriter.create<mhlo::SubOp>(
      loc,
      rewriter.create<mhlo::SubOp>(
          loc, getConstantLike(rewriter, loc, std::log(M_PI), x),
          reflection_denom),
      lgamma);

  // Avoid computing -inf - inf, which is nan. If reflection_denom is +/-inf,
  // then it "wins" and the result is +/-inf.
  Value finite_reflection_denom =
      rewriter.create<mhlo::IsFiniteOp>(loc, reflection_denom);
  Value neg_reflection_denom =
      rewriter.create<mhlo::NegOp>(loc, reflection_denom);
  lgamma_reflection = rewriter.create<mhlo::SelectOp>(
      loc, finite_reflection_denom, lgamma_reflection, neg_reflection_denom);

  // Select whether or not to rely on the reflection.
  lgamma = rewriter.create<mhlo::SelectOp>(loc, need_to_reflect,
                                           lgamma_reflection, lgamma);

  // Materialize +/-inf behavior as
  //   lgamma(+/-inf) = +inf.
  Value x_is_inf = rewriter.create<chlo::IsInfOp>(loc, x);
  return rewriter.create<mhlo::SelectOp>(
      loc, x_is_inf,
      chlo::getConstantLikeInfValue(rewriter, loc, x, /*negative=*/false),
      lgamma);
}

// Compute the Digamma function using Lanczos' approximation from "A Precision
// Approximation of the Gamma Function". SIAM Journal on Numerical Analysis
// series B. Vol. 1:
//   digamma(z + 1) = log(t(z)) + a'(z) / a(z) - kLanczosGamma / t(z)
//   with   t(z) = z + kLanczosGamma + 1/2
//          a(z) = kBaseLanczosCoeff
//                   + sum(k = 1, n, kLanczosCoefficients[i] / (z + k))
//          a'(z) = - sum(k = 1, n, kLanczosCoefficients[i] / (z + k) / (z + k))
Value MaterializeDigamma(ConversionPatternRewriter &rewriter, Location loc,
                         ValueRange args) {
  // If the input is less than 0.5 use Euler's reflection formula.
  //   digamma(x) = digamma(1 - x) - pi * cot(pi * x)
  // Let z be
  //   z = -x      if x < 1/2
  //   z = x - 1   otheriwse
  Value x = args.front();
  const StringAttr kLT = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::LT));
  Value half = getConstantLike(rewriter, loc, 0.5, x);
  Value need_to_reflect = rewriter.create<mhlo::CompareOp>(loc, x, half, kLT);
  Value neg_x = rewriter.create<mhlo::NegOp>(loc, x);
  Value one = getConstantLike(rewriter, loc, 1, x);
  Value x_sub_one = rewriter.create<mhlo::SubOp>(loc, x, one);
  Value z =
      rewriter.create<mhlo::SelectOp>(loc, need_to_reflect, neg_x, x_sub_one);

  // Materialize
  //   a(z) = kBaseLanczosCoeff
  //            + sum(k = 1, n, kLanczosCoefficients[i] / (z + k))
  //   a'(z) = - sum(k = 1, n, kLanczosCoefficients[i] / (z + k) / (z + k))
  Value zero = getConstantLike(rewriter, loc, 0.0, x);
  Value a = getConstantLike(rewriter, loc, kBaseLanczosCoeff, x);
  Value a_prime = zero;
  for (int i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
    Value coeff = getConstantLike(rewriter, loc, kLanczosCoefficients[i], x);
    Value one_based_index = getConstantLike(rewriter, loc, i + 1, x);
    Value z_term = rewriter.create<mhlo::AddOp>(loc, z, one_based_index);
    a_prime = rewriter.create<mhlo::SubOp>(
        loc, a_prime,
        rewriter.create<mhlo::DivOp>(
            loc, coeff, rewriter.create<mhlo::MulOp>(loc, z_term, z_term)));
    a = rewriter.create<mhlo::AddOp>(
        loc, a, rewriter.create<mhlo::DivOp>(loc, coeff, z_term));
  }

  // To improve accuracy on platforms with less-precise log implementations,
  // compute log(kLanczosGamma + 1/2) at compile time and use log1p on the
  // device.
  // Materialize as
  //   log(t) = log(kLanczosGamma + 1/2 + z)
  //          = log(kLanczosGamma + 1/2) + log1p(z / (kLanczosGamma + 1/2)).
  Value lanczos_plus_half =
      getConstantLike(rewriter, loc, kLanczosGamma + 0.5, x);
  Value t = rewriter.create<mhlo::AddOp>(loc, lanczos_plus_half, z);
  Value log_term =
      getConstantLike(rewriter, loc, std::log(kLanczosGamma + 0.5), x);
  Value log1p_term = rewriter.create<mhlo::Log1pOp>(
      loc, rewriter.create<mhlo::DivOp>(loc, z, lanczos_plus_half));
  Value log_t = rewriter.create<mhlo::AddOp>(loc, log_term, log1p_term);

  // Materialize the final result (modulo reflection) as
  //   digamma(z + 1) = log(t(z)) + a'(z) / a(z) - kLanczosGamma / t(z).
  Value a_prime_div_a = rewriter.create<mhlo::DivOp>(loc, a_prime, a);
  Value lanczos_gamma_div_t = rewriter.create<mhlo::DivOp>(
      loc, getConstantLike(rewriter, loc, kLanczosGamma, x), t);
  Value digamma = rewriter.create<mhlo::SubOp>(
      loc, rewriter.create<mhlo::AddOp>(loc, log_t, a_prime_div_a),
      lanczos_gamma_div_t);

  // We need to be careful how we compute cot(pi * input) below: For
  // near-integral arguments, pi * input can lose precision.
  //
  // Input is already known to be less than 0.5 (otherwise we don't have to
  // reflect). We shift values smaller than -0.5 into the range [-0.5, 0.5] to
  // increase precision of pi * x and the resulting cotangent.
  Value reduced_x = rewriter.create<mhlo::AddOp>(
      loc, x,
      rewriter.create<mhlo::AbsOp>(
          loc, rewriter.create<mhlo::FloorOp>(
                   loc, rewriter.create<mhlo::AddOp>(
                            loc, x, getConstantLike(rewriter, loc, 0.5, x)))));

  // Materialize reflection for inputs less than 0.5 as
  //   digamma(x) = digamma(1 - x) - pi * cot(pi * x)
  //              = digamma(1 - x) - pi * cos(pi * x) / sin(pi * x)
  Value pi = getConstantLike(rewriter, loc, M_PI, x);
  Value pi_mul_reduced_x = rewriter.create<mhlo::MulOp>(loc, pi, reduced_x);
  Value cos = rewriter.create<mhlo::CosOp>(loc, pi_mul_reduced_x);
  Value sin = rewriter.create<mhlo::SinOp>(loc, pi_mul_reduced_x);
  Value reflection = rewriter.create<mhlo::SubOp>(
      loc, digamma,
      rewriter.create<mhlo::DivOp>(
          loc, rewriter.create<mhlo::MulOp>(loc, pi, cos), sin));

  // Select whether or not to rely on the reflection.
  digamma = rewriter.create<mhlo::SelectOp>(loc, need_to_reflect, reflection,
                                            digamma);

  // Digamma has poles at negative integers and zero; return nan for those.
  const StringAttr kLE = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::LE));
  Value is_le_zero = rewriter.create<mhlo::CompareOp>(loc, x, zero, kLE);
  const StringAttr kEQ = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::EQ));
  Value is_int = rewriter.create<mhlo::CompareOp>(
      loc, x, rewriter.create<mhlo::FloorOp>(loc, x), kEQ);
  Value is_pole = rewriter.create<mhlo::AndOp>(loc, is_le_zero, is_int);
  return rewriter.create<mhlo::SelectOp>(
      loc, is_pole,
      getConstantLike(rewriter, loc, std::numeric_limits<double>::quiet_NaN(),
                      x),
      digamma);
}

Value MaterializeZeta(ConversionPatternRewriter &rewriter, Location loc,
                      ValueRange args) {
  assert(args.size() == 2);
  Value x = args[0];
  Value q = args[1];
  static const std::array<double, 12> kZetaCoeffs{
      -7.1661652561756670113e18,
      1.8152105401943546773e17,
      -4.5979787224074726105e15,
      1.1646782814350067249e14,
      -2.950130727918164224e12,
      7.47242496e10,
      -1.8924375803183791606e9,
      47900160.0,
      -1209600.0,
      30240.0,
      -720.0,
      12.0,
  };

  // For speed we'll always use 9 iterations for the initial series estimate,
  // and a 12 term expansion for the Euler-Maclaurin formula.
  Value a = q;
  Value zero = chlo::getConstantLike(rewriter, loc, 0.0, a);
  Value neg_power = zero;
  Value neg_x = rewriter.create<mhlo::NegOp>(loc, x);
  Value initial_sum = rewriter.create<mhlo::PowOp>(loc, q, neg_x);
  Value one = chlo::getConstantLike(rewriter, loc, 1.0, a);
  for (int i = 0; i < 9; ++i) {
    a = rewriter.create<mhlo::AddOp>(loc, a, one);
    neg_power = rewriter.create<mhlo::PowOp>(loc, a, neg_x);
    initial_sum = rewriter.create<mhlo::AddOp>(loc, initial_sum, neg_power);
  }
  a = rewriter.create<mhlo::AddOp>(loc, a, one);
  neg_power = rewriter.create<mhlo::PowOp>(loc, a, neg_x);
  Value one_like_x = chlo::getConstantLike(rewriter, loc, 1.0, x);
  Value x_minus_one = rewriter.create<mhlo::SubOp>(loc, x, one_like_x);
  Value neg_power_mul_a = rewriter.create<mhlo::MulOp>(loc, neg_power, a);
  Value neg_power_mul_a_div_x_minus_one =
      rewriter.create<mhlo::DivOp>(loc, neg_power_mul_a, x_minus_one);
  Value s = rewriter.create<mhlo::AddOp>(loc, initial_sum,
                                         neg_power_mul_a_div_x_minus_one);
  Value a_inverse_square = rewriter.create<mhlo::DivOp>(
      loc, one, rewriter.create<mhlo::MulOp>(loc, a, a));

  Value horner_sum = zero;
  Value factor = one;
  // Use Horner's rule for this.
  // Note this differs from Cephes which does a 'naive' polynomial evaluation.
  // Using Horner's rule allows to avoid some NaN's and Infs from happening,
  // resulting in more numerically stable code.
  for (int i = 0; i < 11; ++i) {
    Value factor_lhs = rewriter.create<mhlo::SubOp>(
        loc, x, chlo::getConstantLike(rewriter, loc, 22 - 2 * i, x));
    Value factor_rhs = rewriter.create<mhlo::SubOp>(
        loc, x, chlo::getConstantLike(rewriter, loc, 21 - 2 * i, x));
    factor = rewriter.create<mhlo::MulOp>(loc, factor_lhs, factor_rhs);
    horner_sum = rewriter.create<mhlo::MulOp>(
        loc, factor,
        rewriter.create<mhlo::MulOp>(
            loc, a_inverse_square,
            rewriter.create<mhlo::AddOp>(
                loc, horner_sum,
                chlo::getConstantLike(rewriter, loc, 1. / kZetaCoeffs[i], a))));
  }
  Value zero_point_five_like_neg_power =
      chlo::getConstantLike(rewriter, loc, .5, neg_power);
  Value x_div_a = rewriter.create<mhlo::DivOp>(loc, x, a);
  s = rewriter.create<mhlo::AddOp>(
      loc, s,
      rewriter.create<mhlo::MulOp>(
          loc, neg_power,
          rewriter.create<mhlo::AddOp>(
              loc, zero_point_five_like_neg_power,
              rewriter.create<mhlo::MulOp>(
                  loc, x_div_a,
                  rewriter.create<mhlo::AddOp>(
                      loc,
                      chlo::getConstantLike(rewriter, loc, 1. / kZetaCoeffs[11],
                                            a),
                      horner_sum)))));

  // Use the initial zeta sum without the correction term coming
  // from Euler-Maclaurin if it is accurate enough.
  const StringAttr kLT = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::LT));
  Value abs_neg_power = rewriter.create<mhlo::AbsOp>(loc, neg_power);
  Value abs_initial_sum = rewriter.create<mhlo::AbsOp>(loc, initial_sum);
  Value output = rewriter.create<mhlo::SelectOp>(
      loc,
      rewriter.create<mhlo::CompareOp>(
          loc, abs_neg_power,
          rewriter.create<mhlo::MulOp>(
              loc, abs_initial_sum,
              chlo::getConstantLikeSmallestFiniteValue(rewriter, loc, a)),
          kLT),
      initial_sum, s);

  // Function is not defined for x < 1.
  Value nan = chlo::getConstantLike(
      rewriter, loc, std::numeric_limits<double>::quiet_NaN(), x);
  output = rewriter.create<mhlo::SelectOp>(
      loc, rewriter.create<mhlo::CompareOp>(loc, x, one_like_x, kLT), nan,
      output);

  // For q <= 0, x must be an integer.
  const StringAttr kLE = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::LE));
  const StringAttr kNE = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::NE));
  Value q_le_zero = rewriter.create<mhlo::CompareOp>(loc, q, zero, kLE);
  Value x_not_int = rewriter.create<mhlo::CompareOp>(
      loc, x, rewriter.create<mhlo::FloorOp>(loc, x), kNE);
  Value x_domain_error =
      rewriter.create<mhlo::AndOp>(loc, q_le_zero, x_not_int);
  output = rewriter.create<mhlo::SelectOp>(loc, x_domain_error, nan, output);

  // For all integer q <= 0, zeta has a pole. The limit is only defined as
  // +inf if x is and even integer.
  const StringAttr kEQ = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::EQ));
  Value inf = chlo::getConstantLike(rewriter, loc,
                                    std::numeric_limits<double>::infinity(), x);
  Value q_is_int = rewriter.create<mhlo::CompareOp>(
      loc, q, rewriter.create<mhlo::FloorOp>(loc, q), kEQ);
  Value at_pole = rewriter.create<mhlo::AndOp>(loc, q_le_zero, q_is_int);
  Value two = chlo::getConstantLike(rewriter, loc, 2.0, x);
  Value x_is_int = rewriter.create<mhlo::CompareOp>(
      loc, x, rewriter.create<mhlo::FloorOp>(loc, x), kEQ);
  Value x_is_even = rewriter.create<mhlo::CompareOp>(
      loc, rewriter.create<mhlo::RemOp>(loc, x, two), zero, kEQ);
  Value x_is_even_int = rewriter.create<mhlo::AndOp>(loc, x_is_int, x_is_even);
  output = rewriter.create<mhlo::SelectOp>(
      loc, at_pole,
      rewriter.create<mhlo::SelectOp>(loc, x_is_even_int, inf, nan), output);

  // For x = 1, this is the harmonic series and diverges.
  output = rewriter.create<mhlo::SelectOp>(
      loc, rewriter.create<mhlo::CompareOp>(loc, x, one, kEQ), inf, output);

  return output;
}

Value MaterializePolygamma(ConversionPatternRewriter &rewriter, Location loc,
                           ValueRange args) {
  PolygammaOp::Adaptor transformed(args);
  Value n = transformed.n();
  Value x = transformed.x();

  // Handle integer n > 0.
  Value one = getConstantLike(rewriter, loc, 1.0, x);
  Value two = getConstantLike(rewriter, loc, 2.0, x);
  Value sign = rewriter.create<mhlo::SubOp>(
      loc,
      rewriter.create<mhlo::MulOp>(loc, two,
                                   rewriter.create<mhlo::RemOp>(loc, n, two)),
      one);
  Value n_plus_one = rewriter.create<mhlo::AddOp>(loc, n, one);
  Value exp_lgamma_np1 = rewriter.create<mhlo::ExpOp>(
      loc, rewriter.create<chlo::LgammaOp>(loc, n_plus_one));
  Value zeta = rewriter.create<chlo::ZetaOp>(loc, n_plus_one, x);
  Value result = rewriter.create<mhlo::MulOp>(
      loc, rewriter.create<mhlo::MulOp>(loc, sign, exp_lgamma_np1), zeta);

  // Handle n = 0.
  const StringAttr kEQ = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::EQ));
  Value zero = getConstantLike(rewriter, loc, 0.0, x);
  Value n_eq_zero = rewriter.create<mhlo::CompareOp>(loc, n, zero, kEQ);
  result = rewriter.create<mhlo::SelectOp>(
      loc, n_eq_zero, rewriter.create<chlo::DigammaOp>(loc, x), result);

  // Check that n is a natural number. Return nan, otherwise.
  const StringAttr kNE = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::NE));
  Value non_int = rewriter.create<mhlo::CompareOp>(
      loc, n, rewriter.create<mhlo::FloorOp>(loc, n), kNE);
  const StringAttr kLT = rewriter.getStringAttr(
      mhlo::stringifyComparisonDirection(mhlo::ComparisonDirection::LT));
  Value negative = rewriter.create<mhlo::CompareOp>(loc, n, zero, kLT);
  Value non_natural = rewriter.create<mhlo::OrOp>(loc, non_int, negative);
  return rewriter.create<mhlo::SelectOp>(
      loc, non_natural,
      getConstantLike(rewriter, loc, std::numeric_limits<double>::quiet_NaN(),
                      x),
      result);
}

struct ConvertLgammaOp : public OpConversionPattern<LgammaOp> {
  using OpConversionPattern<LgammaOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      LgammaOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    FloatType min_precision_ty = rewriter.getF32Type();
    rewriter.replaceOp(
        op, MaterializeWithUpcast(rewriter, op.getLoc(), operands,
                                  min_precision_ty, &MaterializeLgamma));
    return success();
  }
};

struct ConvertDigammaOp : public OpConversionPattern<DigammaOp> {
  using OpConversionPattern<DigammaOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      DigammaOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    FloatType min_precision_ty = rewriter.getF32Type();
    rewriter.replaceOp(
        op, MaterializeWithUpcast(rewriter, op.getLoc(), operands,
                                  min_precision_ty, &MaterializeDigamma));
    return success();
  }
};

struct ConvertPolygammaOp : public OpConversionPattern<PolygammaOp> {
  using OpConversionPattern<PolygammaOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      PolygammaOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    FloatType min_precision_ty = rewriter.getF32Type();
    rewriter.replaceOp(
        op, MaterializeWithUpcast(rewriter, loc, operands, min_precision_ty,
                                  &MaterializePolygamma));
    return success();
  }
};

struct ConvertZetaOp : public OpConversionPattern<ZetaOp> {
  using OpConversionPattern<ZetaOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ZetaOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    FloatType min_precision_ty = rewriter.getF32Type();
    rewriter.replaceOp(
        op, MaterializeWithUpcast(rewriter, loc, operands, min_precision_ty,
                                  &MaterializeZeta));
    return success();
  }
};

struct ConvertSelectOp : public OpConversionPattern<BroadcastSelectOp> {
  using OpConversionPattern<BroadcastSelectOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      BroadcastSelectOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only support ranked operands.
    typename BroadcastSelectOp::Adaptor transformed(operands);
    Value pred = transformed.pred();
    Value on_true = transformed.on_true();
    Value on_false = transformed.on_false();
    auto pred_type = pred.getType().dyn_cast<RankedTensorType>();
    auto on_true_type = on_true.getType().dyn_cast<RankedTensorType>();
    auto on_false_type = on_false.getType().dyn_cast<RankedTensorType>();
    auto result_type = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!pred_type || !on_true_type || !on_false_type || !result_type) {
      return failure();
    }

    auto loc = op.getLoc();

    Value pred_shape = rewriter.createOrFold<shape::ShapeOfOp>(loc, pred);
    Value on_true_shape = rewriter.createOrFold<shape::ShapeOfOp>(loc, on_true);
    Value on_false_shape =
        rewriter.createOrFold<shape::ShapeOfOp>(loc, on_false);
    int64_t result_rank = std::max(
        {pred_type.getRank(), on_true_type.getRank(), on_false_type.getRank()});

    Value broadcastable_cstr =
        rewriter.createOrFold<shape::CstrBroadcastableOp>(
            loc, ValueRange{pred_shape, on_true_shape, on_false_shape});
    auto assuming_op = rewriter.create<shape::AssumingOp>(
        loc, ArrayRef<Type>{result_type}, broadcastable_cstr);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&assuming_op.doRegion());

    Value result_extents = rewriter.createOrFold<shape::BroadcastOp>(
        loc, shape::getExtentTensorType(op.getContext()),
        ValueRange{pred_shape, on_true_shape, on_false_shape},
        /*error=*/nullptr);
    auto shape_type =
        RankedTensorType::get({result_rank}, rewriter.getIndexType());
    result_extents =
        rewriter.createOrFold<tensor::CastOp>(loc, shape_type, result_extents);

    Value broadcasted_pred = pred;
    // Pred has an implicit broadcast for scalars, so use that when convenient.
    if (pred_type.getRank() > 0) {
      auto pred_broadcast_dimensions = llvm::to_vector<4>(
          llvm::seq<int64_t>(result_rank - pred_type.getRank(), result_rank));
      broadcasted_pred = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
          loc,
          RankedTensorType::get(result_type.getShape(),
                                pred_type.getElementType()),
          pred, result_extents,
          rewriter.getI64TensorAttr(pred_broadcast_dimensions));
    }
    auto on_true_broadcast_dimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(result_rank - on_true_type.getRank(), result_rank));
    Value broadcasted_on_true = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc,
        RankedTensorType::get(result_type.getShape(),
                              on_true_type.getElementType()),
        on_true, result_extents,
        rewriter.getI64TensorAttr(on_true_broadcast_dimensions));
    auto on_false_broadcast_dimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(result_rank - on_false_type.getRank(), result_rank));
    Value broadcasted_on_false = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc,
        RankedTensorType::get(result_type.getShape(),
                              on_false_type.getElementType()),
        on_false, result_extents,
        rewriter.getI64TensorAttr(on_false_broadcast_dimensions));

    // And generate the final non-broadcasted ternary op.
    Value final_result = rewriter.create<mhlo::SelectOp>(
        loc, result_type, broadcasted_pred, broadcasted_on_true,
        broadcasted_on_false);
    rewriter.create<shape::AssumingYieldOp>(loc, final_result);
    rewriter.replaceOp(op, {assuming_op.getResult(0)});
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

void PopulateChloBroadcastingPatterns(MLIRContext *context,
                                      OwningRewritePatternList *patterns) {
  // Instantiate conversion templates for conforming binary elementwise ops
  // that do not have different dtypes between operands and results and do
  // not have special attributes that need to be preserved.
  PopulateForBroadcastingBinaryOp<ConvertTrivialNonBroadcastBinaryOp>(
      context, patterns, 10);
  PopulateForBroadcastingBinaryOp<ConvertRankedDynamicBroadcastBinaryOp>(
      context, patterns, 5);
  patterns->insert<ConvertSelectOp>(context);
}

void PopulateLegalizeChloToHloPatterns(MLIRContext *context,
                                       OwningRewritePatternList *patterns) {
  populateWithGenerated(context, *patterns);
  PopulateChloBroadcastingPatterns(context, patterns);

  // Other patterns.
  // clang-format off
  patterns->insert<ConvertConstantLikeOp,
                   ConvertDigammaOp,
                   ConvertErfOp,
                   ConvertErfcOp,
                   ConvertLgammaOp,
                   ConvertPolygammaOp,
                   ConvertZetaOp>(context);
  // clang-format on
}

}  // namespace chlo
}  // namespace mlir
