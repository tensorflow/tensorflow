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
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_chlo_to_hlo_op.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir-hlo/utils/broadcast_utils.h"
#include "mlir-hlo/utils/hlo_utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
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
      ConstantLikeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getType().cast<ShapedType>();

    // Unranked uses are not supported.
    if (!resultTy.hasRank()) return failure();

    // Lower to MHLO constant if statically shaped.
    if (resultTy.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<mhlo::ConstOp>(
          op, DenseElementsAttr::get(resultTy, op.value()));
      return success();
    }

    // Lower to broadcasted constant.
    auto loc = op.getLoc();
    Value constant = rewriter.create<mhlo::ConstOp>(loc, op.value());
    Value shape = rewriter.create<shape::ShapeOfOp>(loc, adaptor.operand());
    rewriter.replaceOpWithNewOp<mhlo::DynamicBroadcastInDimOp>(
        op, resultTy, constant, shape, rewriter.getI64TensorAttr({}));
    return success();
  }
};

template <typename FTy>
Value materializePolynomialApproximation(ConversionPatternRewriter &rewriter,
                                         Location loc, Value x,
                                         ArrayRef<FTy> coefficients) {
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
Value materializeErfcApproximationF64ForMagnituteGeOne(
    ConversionPatternRewriter &rewriter, Location loc, ValueRange args) {
  Value x = args.front();
  assert(x.getType().cast<ShapedType>().getElementType().isF64() &&
         "expect f64 element type");
  const double kMaxlog = 7.09782712893383996843E2;
  const double kErfcPCoefficients[] = {
      2.46196981473530512524E-10, 5.64189564831068821977E-1,
      7.46321056442269912687E0,   4.86371970985681366614E1,
      1.96520832956077098242E2,   5.26445194995477358631E2,
      9.34528527171957607540E2,   1.02755188689515710272E3,
      5.57535335369399327526E2};
  const double kErfcQCoefficients[] = {
      1.00000000000000000000E0, 1.32281951154744992508E1,
      8.67072140885989742329E1, 3.54937778887819891062E2,
      9.75708501743205489753E2, 1.82390916687909736289E3,
      2.24633760818710981792E3, 1.65666309194161350182E3,
      5.57535340817727675546E2};
  const double kErfcRCoefficients[] = {
      5.64189583547755073984E-1, 1.27536670759978104416E0,
      5.01905042251180477414E0,  6.16021097993053585195E0,
      7.40974269950448939160E0,  2.97886665372100240670E0};
  const double kErfcSCoefficients[] = {
      1.00000000000000000000E0, 2.26052863220117276590E0,
      9.39603524938001434673E0, 1.20489539808096656605E1,
      1.70814450747565897222E1, 9.60896809063285878198E0,
      3.36907645100081516050E0};

  // Let z = -x^2.
  Value xSq = rewriter.create<mhlo::MulOp>(loc, x, x);
  Value z = rewriter.create<mhlo::NegOp>(loc, xSq);

  // Materialize polynomial approximation for x in [1, 8) as
  //   erfc(x) = exp(z) P(|x|) / Q(|x|).
  Value expZ = rewriter.create<mhlo::ExpOp>(loc, z);
  Value absX = rewriter.create<mhlo::AbsOp>(loc, x);
  Value polP = materializePolynomialApproximation(
      rewriter, loc, absX, llvm::makeArrayRef(kErfcPCoefficients));
  Value expZMulPolyP = rewriter.create<mhlo::MulOp>(loc, expZ, polP);
  Value polQ = materializePolynomialApproximation(
      rewriter, loc, absX, llvm::makeArrayRef(kErfcQCoefficients));
  Value erfcApprox18 = rewriter.create<mhlo::DivOp>(loc, expZMulPolyP, polQ);

  // Materialize polynomial approximation for x in >= 8 as
  //   erfc(x) exp(z) R(|x|) / S(|x|).
  Value polR = materializePolynomialApproximation(
      rewriter, loc, absX, llvm::makeArrayRef(kErfcRCoefficients));
  Value expZMulPolyR = rewriter.create<mhlo::MulOp>(loc, expZ, polR);
  Value polS = materializePolynomialApproximation(
      rewriter, loc, absX, llvm::makeArrayRef(kErfcSCoefficients));
  Value erfcApprox8Inf = rewriter.create<mhlo::DivOp>(loc, expZMulPolyR, polS);

  // Combine polynomial approximations for x >= 1.
  Value eight = chlo::getConstantLike(rewriter, loc, 8.0, x);
  Value absXLt8 = rewriter.create<mhlo::CompareOp>(
      loc, absX, eight, mhlo::ComparisonDirection::LT);
  Value erfcApprox = rewriter.create<mhlo::SelectOp>(loc, absXLt8, erfcApprox18,
                                                     erfcApprox8Inf);

  // Clamp to prevent overflow and materialize approximation for large x as
  //   erfc(x) = 0.
  Value zLtNegMaxlog = rewriter.create<mhlo::CompareOp>(
      loc, z, chlo::getConstantLike(rewriter, loc, -kMaxlog, x),
      mhlo::ComparisonDirection::LT);
  Value zero = chlo::getConstantLike(rewriter, loc, 0.0, x);
  Value erfcApproxClamped =
      rewriter.create<mhlo::SelectOp>(loc, zLtNegMaxlog, zero, erfcApprox);

  // Derive approximation for x <= -1 as
  //   erfc(x) = 2 - erfc(-x).
  // Reuse previously materialized approximations all of which take |x| as their
  // argument.
  Value xLtZero = rewriter.create<mhlo::CompareOp>(
      loc, x, zero, mhlo::ComparisonDirection::LT);
  Value two = chlo::getConstantLike(rewriter, loc, 2.0, x);
  Value twoSubErfcApproxClamped =
      rewriter.create<mhlo::SubOp>(loc, two, erfcApproxClamped);
  return rewriter.create<mhlo::SelectOp>(loc, xLtZero, twoSubErfcApproxClamped,
                                         erfcApproxClamped);
}

// Precondition is |x| <= 1. Use erfc approximation, otherwise.
// This implementation is based on Cephes.
Value materializeErfApproximationF64ForMagnituteLeOne(
    ConversionPatternRewriter &rewriter, Location loc, ValueRange args) {
  Value x = args.front();
  assert(x.getType().cast<ShapedType>().getElementType().isF64() &&
         "expect f64 element type");
  const double kErfTCoefficients[] = {
      9.60497373987051638749E0, 9.00260197203842689217E1,
      2.23200534594684319226E3, 7.00332514112805075473E3,
      5.55923013010394962768E4};
  const double kErfUCoefficients[] = {
      1.00000000000000000000E0, 3.35617141647503099647E1,
      5.21357949780152679795E2, 4.59432382970980127987E3,
      2.26290000613890934246E4, 4.92673942608635921086E4};

  // Materialize polynomial approximation for |x| <= 1 as
  //   erf(x) = x T(x^2) / U(x^2).
  Value xSq = rewriter.create<mhlo::MulOp>(loc, x, x);
  Value polyT = materializePolynomialApproximation(
      rewriter, loc, xSq, llvm::makeArrayRef(kErfTCoefficients));
  Value xMulPolyT = rewriter.create<mhlo::MulOp>(loc, x, polyT);
  Value polyU = materializePolynomialApproximation(
      rewriter, loc, xSq, llvm::makeArrayRef(kErfUCoefficients));
  return rewriter.create<mhlo::DivOp>(loc, xMulPolyT, polyU);
}

// This implementation is based on Cephes.
Value materializeErfApproximationF64(ConversionPatternRewriter &rewriter,
                                     Location loc, ValueRange args) {
  Value x = args.front();
  assert(x.getType().cast<ShapedType>().getElementType().isF64() &&
         "expect f64 element type");

  // Rely on erf approximation for |x| < 1
  //   erf(x) = erf_approx(x)
  Value erfApprox =
      materializeErfApproximationF64ForMagnituteLeOne(rewriter, loc, x);

  // Rely on erfc approximation for |x| >= 1 and materialize erf as
  //   erf(x) = 1 - erfc_approx(x)
  Value one = chlo::getConstantLike(rewriter, loc, 1.0, x);
  Value erfcApprox =
      materializeErfcApproximationF64ForMagnituteGeOne(rewriter, loc, x);
  Value erfcBasedApprox = rewriter.create<mhlo::SubOp>(loc, one, erfcApprox);

  // Materialize approximation selection based on argument.
  Value absX = rewriter.create<mhlo::AbsOp>(loc, x);
  Value absXLtOne = rewriter.create<mhlo::CompareOp>(
      loc, absX, one, mhlo::ComparisonDirection::LT);
  return rewriter.create<mhlo::SelectOp>(loc, absXLtOne, erfApprox,
                                         erfcBasedApprox);
}

Value materializeErfcApproximationF64(ConversionPatternRewriter &rewriter,
                                      Location loc, ValueRange args) {
  Value x = args.front();
  assert(x.getType().cast<ShapedType>().getElementType().isF64() &&
         "expect f64 element type");

  // Rely on erfc approximation for |x| >= 1
  //   erfc(x) = erfc_approx(x)
  Value erfcApprox =
      materializeErfcApproximationF64ForMagnituteGeOne(rewriter, loc, x);

  // Rely on erf approximation for |x| < 1 and materialize erfc as
  //   erfc(x) = 1 - erf_approx(x)
  Value one = chlo::getConstantLike(rewriter, loc, 1.0, x);
  Value erfApprox =
      materializeErfApproximationF64ForMagnituteLeOne(rewriter, loc, x);
  Value erfBasedApprox = rewriter.create<mhlo::SubOp>(loc, one, erfApprox);

  // Materialize approximation selection based on argument.
  Value absX = rewriter.create<mhlo::AbsOp>(loc, x);
  Value absXLtOne = rewriter.create<mhlo::CompareOp>(
      loc, absX, one, mhlo::ComparisonDirection::LT);
  return rewriter.create<mhlo::SelectOp>(loc, absXLtOne, erfBasedApprox,
                                         erfcApprox);
}

// Precondition is |x| >= 1. Use erf approximation, otherwise.
//
// We rely on multiple polynomial approximations for x >= 1. We pass |x| as an
// argument and derive the final approximation for all |x| >= 1.
// This implementation is based on Cephes.
Value materializeErfcApproximationF32ForMagnitudeGeOne(
    ConversionPatternRewriter &rewriter, Location loc, ValueRange args) {
  Value x = args.front();
  assert(x.getType().cast<ShapedType>().getElementType().isF32() &&
         "expect f32 element type");
  const double kMaxlog = 88.72283905206835;
  const float kErfcPCoefficients[] = {
      +2.326819970068386E-2, -1.387039388740657E-1, +3.687424674597105E-1,
      -5.824733027278666E-1, +6.210004621745983E-1, -4.944515323274145E-1,
      +3.404879937665872E-1, -2.741127028184656E-1, +5.638259427386472E-1,
  };
  const float kErfcRCoefficients[] = {
      -1.047766399936249E+1, +1.297719955372516E+1, -7.495518717768503E+0,
      +2.921019019210786E+0, -1.015265279202700E+0, +4.218463358204948E-1,
      -2.820767439740514E-1, +5.641895067754075E-1,
  };

  // Let z = -x^2.
  Value xSq = rewriter.create<mhlo::MulOp>(loc, x, x);
  Value z = rewriter.create<mhlo::NegOp>(loc, xSq);

  // Materialize polynomial approximation for x >= 1 as
  //   erfc(x) = exp(z) 1/x P(1/x^2)   if x in [1, 2)
  //   erfc(x) = exp(z) 1/x R(1/x^2)   if x >= 2
  Value absX = rewriter.create<mhlo::AbsOp>(loc, x);
  Value one = chlo::getConstantLike(rewriter, loc, 1.0, x);
  Value reciprocalXSq = rewriter.create<mhlo::DivOp>(loc, one, xSq);
  Value expZ = rewriter.create<mhlo::ExpOp>(loc, z);
  Value oneDivAbsX = rewriter.create<mhlo::DivOp>(loc, one, absX);
  Value expZMulOneDivAbsX = rewriter.create<mhlo::MulOp>(loc, expZ, oneDivAbsX);
  Value two = chlo::getConstantLike(rewriter, loc, 2.0, x);
  Value absXLtTwo = rewriter.create<mhlo::CompareOp>(
      loc, absX, two, mhlo::ComparisonDirection::LT);
  Value polP = materializePolynomialApproximation(
      rewriter, loc, reciprocalXSq, llvm::makeArrayRef(kErfcPCoefficients));
  Value polR = materializePolynomialApproximation(
      rewriter, loc, reciprocalXSq, llvm::makeArrayRef(kErfcRCoefficients));
  Value poly = rewriter.create<mhlo::SelectOp>(loc, absXLtTwo, polP, polR);
  Value erfcApprox = rewriter.create<mhlo::MulOp>(loc, expZMulOneDivAbsX, poly);

  // Clamp to prevent overflow and materialize approximation for large x as
  //   erfc(x) = 0.
  Value zLtNeqMaxlog = rewriter.create<mhlo::CompareOp>(
      loc, z, chlo::getConstantLike(rewriter, loc, -kMaxlog, x),
      mhlo::ComparisonDirection::LT);
  Value zero = chlo::getConstantLike(rewriter, loc, 0.0, x);
  Value erfcApproxClamped =
      rewriter.create<mhlo::SelectOp>(loc, zLtNeqMaxlog, zero, erfcApprox);

  // Derive approximation for x <= -1 as
  //   erfc(x) = 2 - erfc(-x).
  // Reuse previously materialized approximations all of which take |x| as their
  // argument.
  Value xLtZero = rewriter.create<mhlo::CompareOp>(
      loc, x, zero, mhlo::ComparisonDirection::LT);
  Value twoSubErfcApprox =
      rewriter.create<mhlo::SubOp>(loc, two, erfcApproxClamped);
  return rewriter.create<mhlo::SelectOp>(loc, xLtZero, twoSubErfcApprox,
                                         erfcApproxClamped);
}

// Precondition is |x| <= 1. Use erfc approximation, otherwise.
// This implementation is based on Cephes.
Value materializeErfApproximationF32ForMagnitudeLeOne(
    ConversionPatternRewriter &rewriter, Location loc, ValueRange args) {
  Value x = args.front();
  assert(x.getType().cast<ShapedType>().getElementType().isF32() &&
         "expect f32 element type");
  const float kErfTCoefficients[] = {
      +7.853861353153693E-5, -8.010193625184903E-4, +5.188327685732524E-3,
      -2.685381193529856E-2, +1.128358514861418E-1, -3.761262582423300E-1,
      +1.128379165726710E+0,
  };

  // Materialize polynomial approximation for |x| <= 1 as
  //   erf(x) = x T(x^2).
  Value xSq = rewriter.create<mhlo::MulOp>(loc, x, x);
  Value polyT = materializePolynomialApproximation(
      rewriter, loc, xSq, llvm::makeArrayRef(kErfTCoefficients));
  return rewriter.create<mhlo::MulOp>(loc, x, polyT);
}

// This is the same approximation as used in Eigen.
Value materializeErfApproximationF32(ConversionPatternRewriter &rewriter,
                                     Location loc, ValueRange args) {
  Value x = args.front();
  assert(x.getType().cast<ShapedType>().getElementType().isF32() &&
         "expect f32 element type");
  const float kAlpha[] = {
      -2.72614225801306e-10f, 2.77068142495902e-08f,  -2.10102402082508e-06f,
      -5.69250639462346e-05f, -7.34990630326855e-04f, -2.95459980854025e-03f,
      -1.60960333262415e-02f,
  };
  const float kBeta[] = {
      -1.45660718464996e-05f, -2.13374055278905e-04f, -1.68282697438203e-03f,
      -7.37332916720468e-03f, -1.42647390514189e-02f,
  };

  // Clamp argument between -4 and 4.
  Value lb = chlo::getConstantLike(rewriter, loc, -4.0, x);
  Value ub = chlo::getConstantLike(rewriter, loc, 4.0, x);
  x = rewriter.create<mhlo::ClampOp>(loc, x.getType(), lb, x, ub);
  Value xSq = rewriter.create<mhlo::MulOp>(loc, x, x);

  // Materialize polynomial approximation for x in [-4, 4] as
  //   erf(x) = x * Alpha(x^2) / Beta(x^2).
  Value alphaPoly = materializePolynomialApproximation(
      rewriter, loc, xSq, llvm::makeArrayRef(kAlpha));
  Value betaPoly = materializePolynomialApproximation(
      rewriter, loc, xSq, llvm::makeArrayRef(kBeta));
  Value xMulAlphaPoly = rewriter.create<mhlo::MulOp>(loc, x, alphaPoly);
  return rewriter.create<mhlo::DivOp>(loc, xMulAlphaPoly, betaPoly);
}

Value materializeErfcApproximationF32(ConversionPatternRewriter &rewriter,
                                      Location loc, ValueRange args) {
  Value x = args.front();
  assert(x.getType().cast<ShapedType>().getElementType().isF32() &&
         "expect f32 element type");

  // Rely on erfc approximation for |x| >= 1
  //   erfc(x) = erfc_approx(x)
  Value erfcApprox =
      materializeErfcApproximationF32ForMagnitudeGeOne(rewriter, loc, x);

  // Rely on erf approximation for |x| < 1 and materialize erfc as
  //   erfc(x) = 1 - erf_approx(x)
  Value one = chlo::getConstantLike(rewriter, loc, 1.0, x);
  Value erfApprox =
      materializeErfApproximationF32ForMagnitudeLeOne(rewriter, loc, x);
  Value erfBasedApprox = rewriter.create<mhlo::SubOp>(loc, one, erfApprox);

  // Materialize approximation selection based on argument.
  Value absX = rewriter.create<mhlo::AbsOp>(loc, x);
  Value absXLtOne = rewriter.create<mhlo::CompareOp>(
      loc, absX, one, mhlo::ComparisonDirection::LT);
  return rewriter.create<mhlo::SelectOp>(loc, absXLtOne, erfBasedApprox,
                                         erfcApprox);
}

Value materializeWithUpcast(ConversionPatternRewriter &rewriter, Location loc,
                            ValueRange args, FloatType minPrecisionTy,
                            Value callback(ConversionPatternRewriter &,
                                           Location, ValueRange)) {
  auto originalTy = getElementTypeOrSelf(args.front().getType());
  auto floatOriginalTy = originalTy.dyn_cast<FloatType>();
  bool needsUpcast =
      floatOriginalTy && floatOriginalTy.getWidth() < minPrecisionTy.getWidth();

  // Upcast arguments if necessary.
  llvm::SmallVector<Value, 2> castedArgs;
  if (needsUpcast) {
    for (Value a : args) {
      castedArgs.push_back(
          rewriter.create<mhlo::ConvertOp>(loc, a, minPrecisionTy));
    }
    args = castedArgs;
  }

  Value result = callback(rewriter, loc, args);

  // Cast back if necessary.
  if (needsUpcast) {
    result = rewriter.create<mhlo::ConvertOp>(loc, result, originalTy);
  }

  return result;
}

struct ConvertErfOp : public OpConversionPattern<ErfOp> {
  using OpConversionPattern<ErfOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ErfOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value x = adaptor.operand();
    Type ty = x.getType().cast<ShapedType>().getElementType();

    // For now, we support only f64, f32, f16 and bf16.
    if (!ty.isF64() && !ty.isF32() && !ty.isF16() && !ty.isBF16())
      return failure();

    if (ty.isF64()) {
      rewriter.replaceOp(op, materializeErfApproximationF64(rewriter, loc, x));
      return success();
    }

    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, loc, adaptor.getOperands(),
                                  rewriter.getF32Type(),
                                  &materializeErfApproximationF32));
    return success();
  }
};

struct ConvertErfcOp : public OpConversionPattern<ErfcOp> {
  using OpConversionPattern<ErfcOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ErfcOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value x = adaptor.operand();
    Type ty = x.getType().cast<ShapedType>().getElementType();

    // For now, we support only f64, f32, f16 and bf16.
    if (!ty.isF64() && !ty.isF32() && !ty.isF16() && !ty.isBF16())
      return failure();

    if (ty.isF64()) {
      rewriter.replaceOp(op, materializeErfcApproximationF64(rewriter, loc, x));
      return success();
    }

    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, loc, adaptor.getOperands(),
                                  rewriter.getF32Type(),
                                  &materializeErfcApproximationF32));
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
Value materializeLgamma(ConversionPatternRewriter &rewriter, Location loc,
                        ValueRange args) {
  // If the input is less than 0.5 use Euler's reflection formula.
  //   gamma(x) = pi / (sin(pi * x) * gamma(1 - x))
  // Let z be
  //   z = -x      if x < 1/2
  //   z = x - 1   otheriwse
  Value x = args.front();
  Value half = getConstantLike(rewriter, loc, 0.5, x);
  Value needToReflect = rewriter.create<mhlo::CompareOp>(
      loc, x, half, mhlo::ComparisonDirection::LT);
  Value negX = rewriter.create<mhlo::NegOp>(loc, x);
  Value one = getConstantLike(rewriter, loc, 1, x);
  Value xSubOne = rewriter.create<mhlo::SubOp>(loc, x, one);
  Value z = rewriter.create<mhlo::SelectOp>(loc, needToReflect, negX, xSubOne);

  // Materialize
  //   a(z) = kBaseLanczosCoeff
  //            + sum(k = 1, n, kLanczosCoefficients[i] / (z + k))
  Value a = getConstantLike(rewriter, loc, kBaseLanczosCoeff, x);
  for (int i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
    Value coeff = getConstantLike(rewriter, loc, kLanczosCoefficients[i], x);
    Value oneBasedIndex = getConstantLike(rewriter, loc, i + 1, x);
    Value quotient = rewriter.create<mhlo::DivOp>(
        loc, coeff, rewriter.create<mhlo::AddOp>(loc, z, oneBasedIndex));
    a = rewriter.create<mhlo::AddOp>(loc, a, quotient);
  }

  // To improve accuracy on platforms with less-precise log implementations,
  // compute log(kLanczosGamma + 1/2) at compile time and use log1p on the
  // device.
  // Materialize as
  //   log(t) = log(kLanczosGamma + 1/2 + z)
  //          = log(kLanczosGamma + 1/2) + log1p(z / (kLanczosGamma + 1/2)).
  Value lanczosPlusHalf =
      getConstantLike(rewriter, loc, kLanczosGamma + 0.5, x);
  Value t = rewriter.create<mhlo::AddOp>(loc, lanczosPlusHalf, z);
  Value logTerm =
      getConstantLike(rewriter, loc, std::log(kLanczosGamma + 0.5), x);
  Value log1pTerm = rewriter.create<mhlo::Log1pOp>(
      loc, rewriter.create<mhlo::DivOp>(loc, z, lanczosPlusHalf));
  Value logT = rewriter.create<mhlo::AddOp>(loc, logTerm, log1pTerm);

  // Note that t(z) may be large and we need to be careful not to overflow to
  // infinity in the relevant term
  //   r = (z + 1/2) * log(t(z)) - t(z).
  // Therefore, we compute this as
  //   r = (z + 1/2 - t(z) / log(t(z))) * log(t(z)).
  Value tDivLogT = rewriter.create<mhlo::DivOp>(loc, t, logT);
  Value sum = rewriter.create<mhlo::SubOp>(
      loc, rewriter.create<mhlo::AddOp>(loc, z, half), tDivLogT);
  Value r = rewriter.create<mhlo::MulOp>(loc, sum, logT);

  // Compute the final result (modulo reflection) as
  //   lgamma(z + 1) = (log(2) + log(pi)) / 2 + r + log(a(z)).
  Value logA = rewriter.create<mhlo::LogOp>(loc, a);
  Value lgamma = rewriter.create<mhlo::AddOp>(
      loc,
      rewriter.create<mhlo::AddOp>(
          loc,
          getConstantLike(rewriter, loc, (std::log(2) + std::log(M_PI)) / 2, x),
          r),
      logA);

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
  Value absFrac = rewriter.create<mhlo::SubOp>(
      loc, abs, rewriter.create<mhlo::FloorOp>(loc, abs));
  Value reduceAbsFrac = rewriter.create<mhlo::CompareOp>(
      loc, half, absFrac, mhlo::ComparisonDirection::LT);
  absFrac = rewriter.create<mhlo::SelectOp>(
      loc, reduceAbsFrac, rewriter.create<mhlo::SubOp>(loc, one, absFrac),
      absFrac);

  // Materialize reflection.
  Value reflectionDenom = rewriter.create<mhlo::LogOp>(
      loc,
      rewriter.create<mhlo::SinOp>(
          loc, rewriter.create<mhlo::MulOp>(
                   loc, getConstantLike(rewriter, loc, M_PI, x), absFrac)));
  Value lgammaReflection = rewriter.create<mhlo::SubOp>(
      loc,
      rewriter.create<mhlo::SubOp>(
          loc, getConstantLike(rewriter, loc, std::log(M_PI), x),
          reflectionDenom),
      lgamma);

  // Avoid computing -inf - inf, which is nan. If reflection_denom is +/-inf,
  // then it "wins" and the result is +/-inf.
  Value finiteReflectionDenom =
      rewriter.create<mhlo::IsFiniteOp>(loc, reflectionDenom);
  Value negReflectionDenom = rewriter.create<mhlo::NegOp>(loc, reflectionDenom);
  lgammaReflection = rewriter.create<mhlo::SelectOp>(
      loc, finiteReflectionDenom, lgammaReflection, negReflectionDenom);

  // Select whether or not to rely on the reflection.
  lgamma = rewriter.create<mhlo::SelectOp>(loc, needToReflect, lgammaReflection,
                                           lgamma);

  // Materialize +/-inf behavior as
  //   lgamma(+/-inf) = +inf.
  Value xIsInf = rewriter.create<chlo::IsInfOp>(loc, x);
  return rewriter.create<mhlo::SelectOp>(
      loc, xIsInf,
      chlo::getConstantLikeInfValue(rewriter, loc, x, /*negative=*/false),
      lgamma);
}

// Uses `rewriter` to materialize the IR for generating a constant tensor of
// log(1/2) values with the same shape and type as `operand`, and associates the
// generated IR to code location `loc`.
//
// Since we currently only support generating integer constants, we actually
// generate the code for -log(2) (which equals log(1/2)).
// TODO(b/190374484): Remove when mhlo::ConstantLikeOp supports complex types.
Value materializeLogOneHalf(ConversionPatternRewriter &rewriter, Location loc,
                            Value operand) {
  auto resultTy = operand.getType().cast<ShapedType>();

  Value two = rewriter.create<mhlo::ConstOp>(
      loc, hlo::GetScalarOfType(getElementTypeOrSelf(operand.getType()), 2));
  Value shape = rewriter.create<shape::ShapeOfOp>(loc, operand);
  Value twoWithOperandShape = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
      loc, resultTy, two, shape, rewriter.getI64TensorAttr({}));

  Value logTwo = rewriter.create<mhlo::LogOp>(loc, twoWithOperandShape);
  return rewriter.create<mhlo::NegOp>(loc, logTwo);
}

// Express `cosh` as
//   cosh(x) = (e^x + e^-x) / 2
//           = e^(x + log(1/2)) + e^(-x + log(1/2))
//
// The second formulation avoids overflowing when e^x = inf but (e^x)/2 is not.
//
// This incorrectly overflows to inf for two f32 input values, namely
// +/-89.4159851, due to rounding error when computing x +/- log(1/2).  The
// correct answer of 3.40281961e+38 (0x7f7fffec) is very close to max-float, so
// we deem this acceptable.
Value materializeCoshApproximation(ConversionPatternRewriter &rewriter,
                                   Location loc, ValueRange operands) {
  CoshOp::Adaptor transformed(operands);
  Value x = transformed.operand();

  // TODO(b/190374484): Use mhlo::ConstantLikeOp when it supports complex types.
  Value logOneHalf = materializeLogOneHalf(rewriter, loc, x);
  Value expAdd = rewriter.create<mhlo::ExpOp>(
      loc, rewriter.create<mhlo::AddOp>(loc, x, logOneHalf));
  Value expSub = rewriter.create<mhlo::ExpOp>(
      loc, rewriter.create<mhlo::SubOp>(loc, logOneHalf, x));
  return rewriter.create<mhlo::AddOp>(loc, expAdd, expSub);
}

struct ConvertCoshOp : public OpConversionPattern<CoshOp> {
  using OpConversionPattern<CoshOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      CoshOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, op.getLoc(), adaptor.getOperands(),
                                  rewriter.getF32Type(),
                                  &materializeCoshApproximation));
    return success();
  }
};

// Compute the Digamma function using Lanczos' approximation from "A Precision
// Approximation of the Gamma Function". SIAM Journal on Numerical Analysis
// series B. Vol. 1:
//   digamma(z + 1) = log(t(z)) + a'(z) / a(z) - kLanczosGamma / t(z)
//   with   t(z) = z + kLanczosGamma + 1/2
//          a(z) = kBaseLanczosCoeff
//                   + sum(k = 1, n, kLanczosCoefficients[i] / (z + k))
//          a'(z) = - sum(k = 1, n, kLanczosCoefficients[i] / (z + k) / (z + k))
Value materializeDigamma(ConversionPatternRewriter &rewriter, Location loc,
                         ValueRange args) {
  // If the input is less than 0.5 use Euler's reflection formula.
  //   digamma(x) = digamma(1 - x) - pi * cot(pi * x)
  // Let z be
  //   z = -x      if x < 1/2
  //   z = x - 1   otheriwse
  Value x = args.front();
  Value half = getConstantLike(rewriter, loc, 0.5, x);
  Value needToReflect = rewriter.create<mhlo::CompareOp>(
      loc, x, half, mhlo::ComparisonDirection::LT);
  Value negX = rewriter.create<mhlo::NegOp>(loc, x);
  Value one = getConstantLike(rewriter, loc, 1, x);
  Value xSubOne = rewriter.create<mhlo::SubOp>(loc, x, one);
  Value z = rewriter.create<mhlo::SelectOp>(loc, needToReflect, negX, xSubOne);

  // Materialize
  //   a(z) = kBaseLanczosCoeff
  //            + sum(k = 1, n, kLanczosCoefficients[i] / (z + k))
  //   a'(z) = - sum(k = 1, n, kLanczosCoefficients[i] / (z + k) / (z + k))
  Value zero = getConstantLike(rewriter, loc, 0.0, x);
  Value a = getConstantLike(rewriter, loc, kBaseLanczosCoeff, x);
  Value aPrime = zero;
  for (int i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
    Value coeff = getConstantLike(rewriter, loc, kLanczosCoefficients[i], x);
    Value oneBasedIndex = getConstantLike(rewriter, loc, i + 1, x);
    Value zTerm = rewriter.create<mhlo::AddOp>(loc, z, oneBasedIndex);
    aPrime = rewriter.create<mhlo::SubOp>(
        loc, aPrime,
        rewriter.create<mhlo::DivOp>(
            loc, coeff, rewriter.create<mhlo::MulOp>(loc, zTerm, zTerm)));
    a = rewriter.create<mhlo::AddOp>(
        loc, a, rewriter.create<mhlo::DivOp>(loc, coeff, zTerm));
  }

  // To improve accuracy on platforms with less-precise log implementations,
  // compute log(kLanczosGamma + 1/2) at compile time and use log1p on the
  // device.
  // Materialize as
  //   log(t) = log(kLanczosGamma + 1/2 + z)
  //          = log(kLanczosGamma + 1/2) + log1p(z / (kLanczosGamma + 1/2)).
  Value lanczosPlusHalf =
      getConstantLike(rewriter, loc, kLanczosGamma + 0.5, x);
  Value t = rewriter.create<mhlo::AddOp>(loc, lanczosPlusHalf, z);
  Value logTerm =
      getConstantLike(rewriter, loc, std::log(kLanczosGamma + 0.5), x);
  Value log1pTerm = rewriter.create<mhlo::Log1pOp>(
      loc, rewriter.create<mhlo::DivOp>(loc, z, lanczosPlusHalf));
  Value logT = rewriter.create<mhlo::AddOp>(loc, logTerm, log1pTerm);

  // Materialize the final result (modulo reflection) as
  //   digamma(z + 1) = log(t(z)) + a'(z) / a(z) - kLanczosGamma / t(z).
  Value aPrimeDivA = rewriter.create<mhlo::DivOp>(loc, aPrime, a);
  Value lanczosGammaDivT = rewriter.create<mhlo::DivOp>(
      loc, getConstantLike(rewriter, loc, kLanczosGamma, x), t);
  Value digamma = rewriter.create<mhlo::SubOp>(
      loc, rewriter.create<mhlo::AddOp>(loc, logT, aPrimeDivA),
      lanczosGammaDivT);

  // We need to be careful how we compute cot(pi * input) below: For
  // near-integral arguments, pi * input can lose precision.
  //
  // Input is already known to be less than 0.5 (otherwise we don't have to
  // reflect). We shift values smaller than -0.5 into the range [-0.5, 0.5] to
  // increase precision of pi * x and the resulting cotangent.
  Value reducedX = rewriter.create<mhlo::AddOp>(
      loc, x,
      rewriter.create<mhlo::AbsOp>(
          loc, rewriter.create<mhlo::FloorOp>(
                   loc, rewriter.create<mhlo::AddOp>(
                            loc, x, getConstantLike(rewriter, loc, 0.5, x)))));

  // Materialize reflection for inputs less than 0.5 as
  //   digamma(x) = digamma(1 - x) - pi * cot(pi * x)
  //              = digamma(1 - x) - pi * cos(pi * x) / sin(pi * x)
  Value pi = getConstantLike(rewriter, loc, M_PI, x);
  Value piMulReducedX = rewriter.create<mhlo::MulOp>(loc, pi, reducedX);
  Value cos = rewriter.create<mhlo::CosOp>(loc, piMulReducedX);
  Value sin = rewriter.create<mhlo::SinOp>(loc, piMulReducedX);
  Value reflection = rewriter.create<mhlo::SubOp>(
      loc, digamma,
      rewriter.create<mhlo::DivOp>(
          loc, rewriter.create<mhlo::MulOp>(loc, pi, cos), sin));

  // Select whether or not to rely on the reflection.
  digamma =
      rewriter.create<mhlo::SelectOp>(loc, needToReflect, reflection, digamma);

  // Digamma has poles at negative integers and zero; return nan for those.
  Value isLeZero = rewriter.create<mhlo::CompareOp>(
      loc, x, zero, mhlo::ComparisonDirection::LE);
  Value isInt = rewriter.create<mhlo::CompareOp>(
      loc, x, rewriter.create<mhlo::FloorOp>(loc, x),
      mhlo::ComparisonDirection::EQ);
  Value isPole = rewriter.create<mhlo::AndOp>(loc, isLeZero, isInt);
  return rewriter.create<mhlo::SelectOp>(
      loc, isPole,
      getConstantLike(rewriter, loc, std::numeric_limits<double>::quiet_NaN(),
                      x),
      digamma);
}

Value materializeZeta(ConversionPatternRewriter &rewriter, Location loc,
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
  Value negPower = zero;
  Value negX = rewriter.create<mhlo::NegOp>(loc, x);
  Value initialSum = rewriter.create<mhlo::PowOp>(loc, q, negX);
  Value one = chlo::getConstantLike(rewriter, loc, 1.0, a);
  for (int i = 0; i < 9; ++i) {
    a = rewriter.create<mhlo::AddOp>(loc, a, one);
    negPower = rewriter.create<mhlo::PowOp>(loc, a, negX);
    initialSum = rewriter.create<mhlo::AddOp>(loc, initialSum, negPower);
  }
  a = rewriter.create<mhlo::AddOp>(loc, a, one);
  negPower = rewriter.create<mhlo::PowOp>(loc, a, negX);
  Value oneLikeX = chlo::getConstantLike(rewriter, loc, 1.0, x);
  Value xMinusOne = rewriter.create<mhlo::SubOp>(loc, x, oneLikeX);
  Value negPowerMulA = rewriter.create<mhlo::MulOp>(loc, negPower, a);
  Value negPowerMulADivXMinusOne =
      rewriter.create<mhlo::DivOp>(loc, negPowerMulA, xMinusOne);
  Value s =
      rewriter.create<mhlo::AddOp>(loc, initialSum, negPowerMulADivXMinusOne);
  Value aInverseSquare = rewriter.create<mhlo::DivOp>(
      loc, one, rewriter.create<mhlo::MulOp>(loc, a, a));

  Value hornerSum = zero;
  Value factor = one;
  // Use Horner's rule for this.
  // Note this differs from Cephes which does a 'naive' polynomial evaluation.
  // Using Horner's rule allows to avoid some NaN's and Infs from happening,
  // resulting in more numerically stable code.
  for (int i = 0; i < 11; ++i) {
    Value factorLhs = rewriter.create<mhlo::SubOp>(
        loc, x, chlo::getConstantLike(rewriter, loc, 22 - 2 * i, x));
    Value factorRhs = rewriter.create<mhlo::SubOp>(
        loc, x, chlo::getConstantLike(rewriter, loc, 21 - 2 * i, x));
    factor = rewriter.create<mhlo::MulOp>(loc, factorLhs, factorRhs);
    hornerSum = rewriter.create<mhlo::MulOp>(
        loc, factor,
        rewriter.create<mhlo::MulOp>(
            loc, aInverseSquare,
            rewriter.create<mhlo::AddOp>(
                loc, hornerSum,
                chlo::getConstantLike(rewriter, loc, 1. / kZetaCoeffs[i], a))));
  }
  Value zeroPointFiveLikeNegPower =
      chlo::getConstantLike(rewriter, loc, .5, negPower);
  Value xDivA = rewriter.create<mhlo::DivOp>(loc, x, a);
  s = rewriter.create<mhlo::AddOp>(
      loc, s,
      rewriter.create<mhlo::MulOp>(
          loc, negPower,
          rewriter.create<mhlo::AddOp>(
              loc, zeroPointFiveLikeNegPower,
              rewriter.create<mhlo::MulOp>(
                  loc, xDivA,
                  rewriter.create<mhlo::AddOp>(
                      loc,
                      chlo::getConstantLike(rewriter, loc, 1. / kZetaCoeffs[11],
                                            a),
                      hornerSum)))));

  // Use the initial zeta sum without the correction term coming
  // from Euler-Maclaurin if it is accurate enough.
  Value absNegPower = rewriter.create<mhlo::AbsOp>(loc, negPower);
  Value absInitialSum = rewriter.create<mhlo::AbsOp>(loc, initialSum);
  Value output = rewriter.create<mhlo::SelectOp>(
      loc,
      rewriter.create<mhlo::CompareOp>(
          loc, absNegPower,
          rewriter.create<mhlo::MulOp>(
              loc, absInitialSum,
              chlo::getConstantLikeSmallestFiniteValue(rewriter, loc, a)),
          mhlo::ComparisonDirection::LT),
      initialSum, s);

  // Function is not defined for x < 1.
  Value nan = chlo::getConstantLike(
      rewriter, loc, std::numeric_limits<double>::quiet_NaN(), x);
  output = rewriter.create<mhlo::SelectOp>(
      loc,
      rewriter.create<mhlo::CompareOp>(loc, x, oneLikeX,
                                       mhlo::ComparisonDirection::LT),
      nan, output);

  // For q <= 0, x must be an integer.
  Value qLeZero = rewriter.create<mhlo::CompareOp>(
      loc, q, zero, mhlo::ComparisonDirection::LE);
  Value xNotInt = rewriter.create<mhlo::CompareOp>(
      loc, x, rewriter.create<mhlo::FloorOp>(loc, x),
      mhlo::ComparisonDirection::NE);
  Value xDomainError = rewriter.create<mhlo::AndOp>(loc, qLeZero, xNotInt);
  output = rewriter.create<mhlo::SelectOp>(loc, xDomainError, nan, output);

  // For all integer q <= 0, zeta has a pole. The limit is only defined as
  // +inf if x is and even integer.
  Value inf = chlo::getConstantLike(rewriter, loc,
                                    std::numeric_limits<double>::infinity(), x);
  Value qIsInt = rewriter.create<mhlo::CompareOp>(
      loc, q, rewriter.create<mhlo::FloorOp>(loc, q),
      mhlo::ComparisonDirection::EQ);
  Value atPole = rewriter.create<mhlo::AndOp>(loc, qLeZero, qIsInt);
  Value two = chlo::getConstantLike(rewriter, loc, 2.0, x);
  Value xIsInt = rewriter.create<mhlo::CompareOp>(
      loc, x, rewriter.create<mhlo::FloorOp>(loc, x),
      mhlo::ComparisonDirection::EQ);
  Value xIsEven = rewriter.create<mhlo::CompareOp>(
      loc, rewriter.create<mhlo::RemOp>(loc, x, two), zero,
      mhlo::ComparisonDirection::EQ);
  Value xIsEvenInt = rewriter.create<mhlo::AndOp>(loc, xIsInt, xIsEven);
  output = rewriter.create<mhlo::SelectOp>(
      loc, atPole, rewriter.create<mhlo::SelectOp>(loc, xIsEvenInt, inf, nan),
      output);

  // For x = 1, this is the harmonic series and diverges.
  output = rewriter.create<mhlo::SelectOp>(
      loc,
      rewriter.create<mhlo::CompareOp>(loc, x, one,
                                       mhlo::ComparisonDirection::EQ),
      inf, output);

  return output;
}

Value materializePolygamma(ConversionPatternRewriter &rewriter, Location loc,
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
  Value nPlusOne = rewriter.create<mhlo::AddOp>(loc, n, one);
  Value expLgammaNp1 = rewriter.create<mhlo::ExpOp>(
      loc, rewriter.create<chlo::LgammaOp>(loc, nPlusOne));
  Value zeta = rewriter.create<chlo::ZetaOp>(loc, nPlusOne, x);
  Value result = rewriter.create<mhlo::MulOp>(
      loc, rewriter.create<mhlo::MulOp>(loc, sign, expLgammaNp1), zeta);

  // Handle n = 0.
  Value zero = getConstantLike(rewriter, loc, 0.0, x);
  Value nEqZero = rewriter.create<mhlo::CompareOp>(
      loc, n, zero, mhlo::ComparisonDirection::EQ);
  result = rewriter.create<mhlo::SelectOp>(
      loc, nEqZero, rewriter.create<chlo::DigammaOp>(loc, x), result);

  // Check that n is a natural number. Return nan, otherwise.
  Value nonInt = rewriter.create<mhlo::CompareOp>(
      loc, n, rewriter.create<mhlo::FloorOp>(loc, n),
      mhlo::ComparisonDirection::NE);
  Value negative = rewriter.create<mhlo::CompareOp>(
      loc, n, zero, mhlo::ComparisonDirection::LT);
  Value nonNatural = rewriter.create<mhlo::OrOp>(loc, nonInt, negative);
  return rewriter.create<mhlo::SelectOp>(
      loc, nonNatural,
      getConstantLike(rewriter, loc, std::numeric_limits<double>::quiet_NaN(),
                      x),
      result);
}

struct ConvertLgammaOp : public OpConversionPattern<LgammaOp> {
  using OpConversionPattern<LgammaOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      LgammaOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FloatType minPrecisionTy = rewriter.getF32Type();
    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, op.getLoc(), adaptor.getOperands(),
                                  minPrecisionTy, &materializeLgamma));
    return success();
  }
};

struct ConvertDigammaOp : public OpConversionPattern<DigammaOp> {
  using OpConversionPattern<DigammaOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      DigammaOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FloatType minPrecisionTy = rewriter.getF32Type();
    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, op.getLoc(), adaptor.getOperands(),
                                  minPrecisionTy, &materializeDigamma));
    return success();
  }
};

Value materializeNextAfter(ConversionPatternRewriter &rewriter, Location loc,
                           ValueRange operands) {
  NextAfterOp::Adaptor transformed(operands);
  Value x = transformed.x();
  Value y = transformed.y();
  auto resultTy = x.getType().cast<ShapedType>();
  auto bitwidth = resultTy.getElementType().getIntOrFloatBitWidth();
  ImplicitLocOpBuilder b(loc, rewriter);
  auto intTy = resultTy.clone(b.getIntegerType(bitwidth));
  auto xAsInt = b.create<mhlo::BitcastConvertOp>(intTy, x);
  auto yAsInt = b.create<mhlo::BitcastConvertOp>(intTy, y);

  // The result is NaN if either "x" or "y" are NaN.
  auto xIsNan = b.create<mhlo::CompareOp>(x, x, mhlo::ComparisonDirection::NE);
  auto yIsNan = b.create<mhlo::CompareOp>(y, y, mhlo::ComparisonDirection::NE);
  auto nanInput = b.create<mhlo::OrOp>(xIsNan, yIsNan);
  auto resultForNan = getConstantLike(
      rewriter, loc, std::numeric_limits<double>::quiet_NaN(), x);
  auto resultForNanAsInt =
      b.create<mhlo::BitcastConvertOp>(intTy, resultForNan);

  // The sign bit is the MSB.
  const int64_t signBit = int64_t{1} << (bitwidth - 1);
  // Discard the sign bit to make the result non-negative.
  auto signMask = getConstantLike(rewriter, loc, signBit, xAsInt);
  auto negatedSignMask = getConstantLike(rewriter, loc, ~signBit, xAsInt);
  auto xAbs = b.create<mhlo::AndOp>(xAsInt, negatedSignMask);
  auto yAbs = b.create<mhlo::AndOp>(yAsInt, negatedSignMask);

  // When both "x" and "y" are equal, the result is "y".
  auto xAndYAreEqual =
      b.create<mhlo::CompareOp>(x, y, mhlo::ComparisonDirection::EQ);
  auto resultForEqual = yAsInt;

  // When both "x" and "y" are 0, the result is "y". This is a separate case
  // from above because "x" and "y" might have a different sign.
  auto zero = getConstantLike(rewriter, loc, 0, xAsInt);
  auto xIsZero =
      b.create<mhlo::CompareOp>(xAbs, zero, mhlo::ComparisonDirection::EQ);
  auto yIsZero =
      b.create<mhlo::CompareOp>(yAbs, zero, mhlo::ComparisonDirection::EQ);
  auto resultForBothZero = yAsInt;

  auto xSign = b.create<mhlo::AndOp>(xAsInt, signMask);
  auto ySign = b.create<mhlo::AndOp>(yAsInt, signMask);

  // If from == 0 && to != 0, we need to return the smallest subnormal number
  // signed like "to".
  auto one = getConstantLike(rewriter, loc, 1, xAsInt);
  auto resultForXZeroYNonZero = b.create<mhlo::OrOp>(ySign, one);

  // If the sign of "x" and "y" disagree:
  // - we need to make the magnitude of "from" smaller so that it is closer to
  //   zero.
  //
  // Otherwise the signs agree:
  // - "x" with a magnitude larger than "y" means we need to make the magnitude
  //   smaller.
  // - "x" with a magnitude smaller than "y" means we need to make the magnitude
  //   larger.
  auto signsDisagree =
      b.create<mhlo::CompareOp>(xSign, ySign, mhlo::ComparisonDirection::NE);
  auto xMagnitudeLargerThanY =
      b.create<mhlo::CompareOp>(xAbs, yAbs, mhlo::ComparisonDirection::GT);
  auto resultHasSmallerMagnitude =
      b.create<mhlo::OrOp>(xMagnitudeLargerThanY, signsDisagree);
  auto minusOne = getConstantLike(rewriter, loc, -1, xAsInt);
  auto magnitudeAdjustment =
      b.create<mhlo::SelectOp>(resultHasSmallerMagnitude, minusOne, one);
  Value result = b.create<mhlo::AddOp>(xAsInt, magnitudeAdjustment);
  // Handle from == +-0.
  result = b.create<mhlo::SelectOp>(
      xIsZero,
      b.create<mhlo::SelectOp>(yIsZero, resultForBothZero,
                               resultForXZeroYNonZero),
      result);
  // Handle from == to.
  result = b.create<mhlo::SelectOp>(xAndYAreEqual, resultForEqual, result);
  // Handle isnan(x) || isnan(y).
  result = b.create<mhlo::SelectOp>(nanInput, resultForNanAsInt, result);

  // Cast back to the original type.
  return b.create<mhlo::BitcastConvertOp>(resultTy, result);
}

struct ConvertNextAfterOp : public OpConversionPattern<NextAfterOp> {
  using OpConversionPattern<NextAfterOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      NextAfterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(
        op, materializeNextAfter(rewriter, op.getLoc(), adaptor.getOperands()));
    return success();
  }
};

struct ConvertPolygammaOp : public OpConversionPattern<PolygammaOp> {
  using OpConversionPattern<PolygammaOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      PolygammaOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    FloatType minPrecisionTy = rewriter.getF32Type();
    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, loc, adaptor.getOperands(),
                                  minPrecisionTy, &materializePolygamma));
    return success();
  }
};

// Sinh(x) = (e^x - e^-x) / 2
//         = e^(x + log(1/2)) - e^(-x + log(1/2)).
//
// The second formulation avoids overflowing when e^x = inf but (e^x)/2 is not
// inf.
//
// This incorrectly overflows to +/-inf for two f32 input values, namely
// +/-89.4159851, due to rounding error when computing x +/- log(1/2).  The
// correct answer of 3.40281961e+38 (0x7f7fffec) is very close to max-float, so
// we deem this acceptable.
Value materializeSinhApproximationForLargeX(ConversionPatternRewriter &rewriter,
                                            Location loc, ValueRange operands) {
  SinhOp::Adaptor transformed(operands);
  Value x = transformed.operand();

  // TODO(b/190374484): Use mhlo::ConstantLikeOp when it supports complex types.
  Value logOneHalf = materializeLogOneHalf(rewriter, loc, x);
  Value expAdd = rewriter.create<mhlo::ExpOp>(
      loc, rewriter.create<mhlo::AddOp>(loc, x, logOneHalf));
  Value expSub = rewriter.create<mhlo::ExpOp>(
      loc, rewriter.create<mhlo::SubOp>(loc, logOneHalf, x));
  return rewriter.create<mhlo::SubOp>(loc, expAdd, expSub);
}

// Express `sinh` as
//   sinh(x) = (e^x - e^-x) / 2                     if |x| < 1
//           = e^(x + log(1/2)) - e^(-x + log(1/2)) otherwise.
Value materializeSinhApproximation(ConversionPatternRewriter &rewriter,
                                   Location loc, ValueRange operands) {
  Value largeSinhResult =
      materializeSinhApproximationForLargeX(rewriter, loc, operands);

  SinhOp::Adaptor transformed(operands);
  Value x = transformed.operand();

  // For smaller x, we get unwanted cancellations of e^x - e^-x, resulting in
  // 0.
  // Rewrite this to avoid that. We use expm1(x) because that preserves the
  // first order term of the taylor series of e^x.
  // (e^(x) - e^(-x)) / 2. =
  // (e^(x) - 1 + 1 - e^(-x)) / 2.
  // (expm1(x) + (e^(x) - 1) / e^x) / 2.
  // (expm1(x) + expm1(x) / (expm1(x) + 1)) / 2.
  Value expm1 = rewriter.create<mhlo::Expm1Op>(loc, x);
  Value one = getConstantLike(rewriter, loc, 1.0, x);
  Value oneHalf = getConstantLike(rewriter, loc, 0.5, x);
  Value expm1PlusOne = rewriter.create<mhlo::AddOp>(loc, expm1, one);
  Value ratio = rewriter.create<mhlo::DivOp>(loc, expm1, expm1PlusOne);
  Value sum = rewriter.create<mhlo::AddOp>(loc, expm1, ratio);
  Value smallSinhResult = rewriter.create<mhlo::MulOp>(loc, oneHalf, sum);

  Value absX = rewriter.create<mhlo::AbsOp>(loc, x);
  Value absXLtOne = rewriter.create<mhlo::CompareOp>(
      loc, absX, one, mhlo::ComparisonDirection::LT);
  return rewriter.create<mhlo::SelectOp>(loc, absXLtOne, smallSinhResult,
                                         largeSinhResult);
}

struct ConvertSinhOp : public OpConversionPattern<SinhOp> {
  using OpConversionPattern<SinhOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      SinhOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value x = adaptor.operand();
    if (x.getType().cast<ShapedType>().getElementType().isa<ComplexType>()) {
      rewriter.replaceOp(op, materializeSinhApproximationForLargeX(
                                 rewriter, op.getLoc(), adaptor.getOperands()));
      return success();
    }
    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, op.getLoc(), adaptor.getOperands(),
                                  rewriter.getF32Type(),
                                  &materializeSinhApproximation));
    return success();
  }
};

Value materializeTan(ConversionPatternRewriter &rewriter, Location loc,
                     ValueRange operands) {
  TanOp::Adaptor transformed(operands);
  return rewriter.create<mhlo::DivOp>(
      loc, rewriter.create<mhlo::SinOp>(loc, transformed.operand()),
      rewriter.create<mhlo::CosOp>(loc, transformed.operand()));
}

struct ConvertTanOp : public OpConversionPattern<TanOp> {
  using OpConversionPattern<TanOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      TanOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, op.getLoc(), adaptor.getOperands(),
                                  rewriter.getF32Type(), &materializeTan));
    return success();
  }
};

// Converts chlo.top_k to MHLO iota, sort, and slice ops.
//
// chlo.top_k sorts along last dimension of the input tensor and then returns
// the top K components' values and indices. This is translated into a few
// ops in MHLO: first generating an integer sequence for the indices,
// then sort both the original input tensor and the indices togheter, and
// at last slice out the top K components.
//
// For example, for the following IR:
//
// %0:2 = "chlo.top_k"(%input, k=8): tensor<16x16xf32> ->
//                                   (tensor<16x8xf32>, tensor<16x8xi32>)
//
// We will get:
//
// %1 = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<16x16xi32>
// %2 = "mhlo.sort"(%input, %1) ({
// ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>,
//      %arg3: tensor<i32>, %arg4: tensor<i32>):
//   %7 = "mhlo.compare"(%arg1, %arg2) {comparison_direction = "GT"}: ...
//   "mhlo.return"(%7) : (tensor<i1>) -> ()
// }) {dimension = 1 : i64, is_stable = true} : ...
// %3 = "mhlo.get_tuple_element"(%2) {index = 0 : i32} : ...
// %4 = "mhlo.get_tuple_element"(%2) {index = 1 : i32} : ...
// %5 = "mhlo.slice"(%3) {limit_indices = dense<[16, 8]> : tensor<2xi64>,
//                           start_indices dense<0> : tensor<2xi64>,
//                           strides = dense<1> : tensor<2xi64>} :
//                              (tensor<16x16xf32>) -> tensor<16x8xf32>
// %6 = "mhlo.slice"(%4) ...
struct ConvertTopKOp : public OpConversionPattern<TopKOp> {
  using OpConversionPattern<TopKOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      TopKOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const override {
    // The last dimension of the operand's shape should be known so we can have
    // clamped end_indices for slices. This is verified by the op.
    auto operandType = op.operand().getType().cast<RankedTensorType>();
    int64_t operandRank = operandType.getRank();
    int64_t lastDimIndex = operandRank - 1;
    int64_t lastDimSize = operandType.getDimSize(lastDimIndex);
    assert(lastDimSize != ShapedType::kDynamicSize);

    // Create an Iota op for indices.
    auto i32Type = rewriter.getIntegerType(32);
    Type iotaType = RankedTensorType::get(operandType.getShape(), i32Type);
    Value iotaOp = rewriter.create<mhlo::IotaOp>(
        op.getLoc(), iotaType, rewriter.getI64IntegerAttr(lastDimIndex));

    // Create the sort op. It takes two inputs, one for the original input, the
    // other for the indices. Use TOTALORDER comparison type instead of the
    // default comparison if the element type is of type float.
    Type elementType = operandType.getElementType();
    auto sortOp = CreateSortOp(&rewriter, op.getLoc(), {op.operand(), iotaOp},
                               {elementType, i32Type}, lastDimIndex,
                               /*is_stable=*/true,
                               /*direction=*/mhlo::ComparisonDirection::GT);

    // Get the sorted input and index tuple element.
    auto tupleFirstElement = sortOp.getResult(0);
    auto tupleSecondElement = sortOp.getResult(1);

    SmallVector<int64_t, 4> beginIndices(operandRank, 0);
    auto endIndices = llvm::to_vector<4>(operandType.getShape());
    endIndices.back() = std::min(static_cast<int64_t>(op.k()), lastDimSize);
    SmallVector<int64_t, 4> strides(operandRank, 1);

    // Get the slice for the top K elements.
    auto indicesTy = RankedTensorType::get(operandRank, rewriter.getI64Type());
    Value values = rewriter.create<mhlo::SliceOp>(
        op.getLoc(), tupleFirstElement,
        DenseIntElementsAttr::get(indicesTy, beginIndices),
        DenseIntElementsAttr::get(indicesTy, endIndices),
        DenseIntElementsAttr::get(indicesTy, strides));
    Value indices = rewriter.create<mhlo::SliceOp>(
        op.getLoc(), tupleSecondElement,
        DenseIntElementsAttr::get(indicesTy, beginIndices),
        DenseIntElementsAttr::get(indicesTy, endIndices),
        DenseIntElementsAttr::get(indicesTy, strides));

    rewriter.replaceOp(op, {values, indices});
    return success();
  }
};

struct ConvertZetaOp : public OpConversionPattern<ZetaOp> {
  using OpConversionPattern<ZetaOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ZetaOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    FloatType minPrecisionTy = rewriter.getF32Type();
    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, loc, adaptor.getOperands(),
                                  minPrecisionTy, &materializeZeta));
    return success();
  }
};

struct ConvertSelectOp : public OpConversionPattern<BroadcastSelectOp> {
  using OpConversionPattern<BroadcastSelectOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      BroadcastSelectOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Only support ranked operands.
    Value pred = adaptor.pred();
    Value onTrue = adaptor.on_true();
    Value onFalse = adaptor.on_false();
    auto predType = pred.getType().dyn_cast<RankedTensorType>();
    auto onTrueType = onTrue.getType().dyn_cast<RankedTensorType>();
    auto onFalseType = onFalse.getType().dyn_cast<RankedTensorType>();
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!predType || !onTrueType || !onFalseType || !resultType) {
      return failure();
    }

    auto loc = op.getLoc();

    Value predShape = rewriter.createOrFold<shape::ShapeOfOp>(loc, pred);
    Value onTrueShape = rewriter.createOrFold<shape::ShapeOfOp>(loc, onTrue);
    Value onFalseShape = rewriter.createOrFold<shape::ShapeOfOp>(loc, onFalse);
    int64_t resultRank = std::max(
        {predType.getRank(), onTrueType.getRank(), onFalseType.getRank()});

    Value broadcastableCstr = rewriter.createOrFold<shape::CstrBroadcastableOp>(
        loc, ValueRange{predShape, onTrueShape, onFalseShape});
    auto assumingOp = rewriter.create<shape::AssumingOp>(
        loc, ArrayRef<Type>{resultType}, broadcastableCstr);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&assumingOp.getDoRegion());

    Value resultExtents = rewriter.createOrFold<shape::BroadcastOp>(
        loc, shape::getExtentTensorType(op.getContext()),
        ValueRange{predShape, onTrueShape, onFalseShape},
        /*error=*/nullptr);
    auto shapeType =
        RankedTensorType::get({resultRank}, rewriter.getIndexType());
    resultExtents =
        rewriter.createOrFold<tensor::CastOp>(loc, shapeType, resultExtents);

    Value broadcastedPred = pred;
    // Pred has an implicit broadcast for scalars, so use that when convenient.
    if (predType.getRank() > 0) {
      auto predBroadcastDimensions = llvm::to_vector<4>(
          llvm::seq<int64_t>(resultRank - predType.getRank(), resultRank));
      broadcastedPred = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
          loc,
          RankedTensorType::get(resultType.getShape(),
                                predType.getElementType()),
          pred, resultExtents,
          rewriter.getI64TensorAttr(predBroadcastDimensions));
    }
    auto onTrueBroadcastDimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(resultRank - onTrueType.getRank(), resultRank));
    Value broadcastedOnTrue = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc,
        RankedTensorType::get(resultType.getShape(),
                              onTrueType.getElementType()),
        onTrue, resultExtents,
        rewriter.getI64TensorAttr(onTrueBroadcastDimensions));
    auto onFalseBroadcastDimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(resultRank - onFalseType.getRank(), resultRank));
    Value broadcastedOnFalse = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc,
        RankedTensorType::get(resultType.getShape(),
                              onFalseType.getElementType()),
        onFalse, resultExtents,
        rewriter.getI64TensorAttr(onFalseBroadcastDimensions));

    // And generate the final non-broadcasted ternary op.
    Value finalResult =
        rewriter.create<mhlo::SelectOp>(loc, resultType, broadcastedPred,
                                        broadcastedOnTrue, broadcastedOnFalse);
    rewriter.create<shape::AssumingYieldOp>(loc, finalResult);
    rewriter.replaceOp(op, {assumingOp.getResult(0)});
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
      ChloOpTy op, typename ChloOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Only rewrite for statically determinable non-broadcasting cases.
    auto lhsType =
        adaptor.lhs().getType().template dyn_cast<RankedTensorType>();
    auto rhsType =
        adaptor.rhs().getType().template dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType) return failure();

    // Requires rank broadcast.
    if (lhsType.getRank() != rhsType.getRank()) return failure();
    // Any dynamic dimension may require broadcasting and requires more
    // analysis.
    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape())
      return failure();

    for (auto extents : llvm::zip(lhsType.getShape(), rhsType.getShape())) {
      auto lhsExtent = std::get<0>(extents);
      auto rhsExtent = std::get<1>(extents);
      if (lhsExtent != rhsExtent) {
        return failure();
      }
    }

    rewriter.replaceOp(op,
                       {Adaptor::CreateOp(op, op.getResult().getType(),
                                          adaptor.getOperands(), rewriter)});
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
      ChloOpTy op, typename ChloOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Only support ranked operands.
    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();
    auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
    auto resultType =
        op.getResult().getType().template dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType || !resultType) return failure();

    // Check for "numpy"-style rank broadcast.
    auto broadcastDimensions = op.broadcast_dimensions();
    if (broadcastDimensions &&
        !hlo::IsLegalNumpyRankedBroadcast(lhs, rhs, *broadcastDimensions)) {
      // Note: It is unclear whether the general specification of explicit
      // broadcast_dimensions on binary ops is a feature we want to carry
      // forward. While it can technically be implemented for ranked-dynamic,
      // it is incompatible with unranked inputs. If this warning is emitted
      // in real programs, it is an indication that the feature should be
      // implemented versus just falling back on the more standard definition
      // of numpy-like prefix-padding.
      op.emitWarning() << "unsupported non prefix-padded dynamic rank "
                       << "broadcast_dimensions = " << *broadcastDimensions;
      return failure();
    }

    // Compute result shape.
    auto loc = op.getLoc();

    // Insert a constraint on the shapes being broadcastable and insert all
    // future code into an assuming block reliant on the constraint.
    Value lhsShape = rewriter.create<shape::ShapeOfOp>(loc, lhs);
    Value rhsShape = rewriter.create<shape::ShapeOfOp>(loc, rhs);
    auto broadcastableCstr =
        rewriter.create<shape::CstrBroadcastableOp>(loc, lhsShape, rhsShape);
    auto assumingOp = rewriter.create<shape::AssumingOp>(
        loc, ArrayRef<Type>{resultType}, broadcastableCstr.getResult());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&assumingOp.getDoRegion());

    int64_t resultRank = std::max(lhsType.getRank(), rhsType.getRank());
    Value resultExtents =
        hlo::ComputeBinaryElementwiseBroadcastingResultExtents(loc, lhs, rhs,
                                                               rewriter);

    // Note that we unconditionally emit DynamicBroadcastInDim ops and let
    // downstream canonicalizations fold them away if possible. This is
    // because, in the dynamic case, there are many corner cases regarding
    // when it is safe to omit, and some of them require analysis to prove
    // properly.
    auto lhsBroadcastDimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(resultRank - lhsType.getRank(), resultRank));
    Value broadcastedLhs = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc,
        RankedTensorType::get(resultType.getShape(), lhsType.getElementType()),
        lhs, resultExtents, rewriter.getI64TensorAttr(lhsBroadcastDimensions));
    auto rhsBroadcastDimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(resultRank - rhsType.getRank(), resultRank));
    Value broadcastedRhs = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc,
        RankedTensorType::get(resultType.getShape(), rhsType.getElementType()),
        rhs, resultExtents, rewriter.getI64TensorAttr(rhsBroadcastDimensions));

    // And generate the final non-broadcasted binary op.
    Value finalResult = Adaptor::CreateOp(
        op, resultType, {broadcastedLhs, broadcastedRhs}, rewriter);
    rewriter.create<shape::AssumingYieldOp>(loc, finalResult);
    rewriter.replaceOp(op, {assumingOp.getResult(0)});
    return success();
  }
};

class ConvertDynamicReshapeOp
    : public OpRewritePattern<chlo::DynamicReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(chlo::DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tensor = op.operand();
    auto shape = op.output_shape();

    auto shape_ty = shape.getType().cast<ShapedType>();
    auto result_ty = op.getType().cast<ShapedType>();

    Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, tensor);
    Value numEls = rewriter.create<shape::NumElementsOp>(loc, inputShape);
    Value cstr = rewriter.create<mhlo::CstrReshapableOp>(loc, numEls, shape);
    rewriter.replaceOpWithNewOp<shape::AssumingOp>(
        op, cstr, [&](OpBuilder &b, Location l) {
          Value computedShape =
              b.create<mhlo::ComputeReshapeShapeOp>(l, shape_ty, numEls, shape);
          SmallVector<Value> result;
          result.push_back(b.create<mhlo::DynamicReshapeOp>(
              l, result_ty, tensor, computedShape));
          return result;
        });

    return success();
  }
};

#include "generated_chlo_legalize_to_hlo.inc"
}  // namespace

void PopulateChloBroadcastingPatterns(MLIRContext *context,
                                      RewritePatternSet *patterns) {
  // Instantiate conversion templates for conforming binary elementwise ops
  // that do not have different dtypes between operands and results and do
  // not have special attributes that need to be preserved.
  PopulateForBroadcastingBinaryOp<ConvertTrivialNonBroadcastBinaryOp>(
      context, patterns, 10);
  PopulateForBroadcastingBinaryOp<ConvertRankedDynamicBroadcastBinaryOp>(
      context, patterns, 5);
  patterns
      ->add<ConvertConstantLikeOp, ConvertDynamicReshapeOp, ConvertSelectOp>(
          context);
}

void PopulateDecomposeChloPatterns(MLIRContext *context,
                                   RewritePatternSet *patterns) {
  populateWithGenerated(*patterns);

  // Other patterns.
  // clang-format off
  patterns->add<ConvertCoshOp,
                   ConvertDigammaOp,
                   ConvertErfOp,
                   ConvertErfcOp,
                   ConvertLgammaOp,
                   ConvertNextAfterOp,
                   ConvertPolygammaOp,
                   ConvertSinhOp,
                   ConvertTanOp,
                   ConvertTopKOp,
                   ConvertZetaOp>(context);
  // clang-format on
}

}  // namespace chlo
}  // namespace mlir
