/* Copyright 2019 The OpenXLA Authors.

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

#ifndef MLIR_HLO_MHLO_TRANSFORMS_MAP_MHLO_TO_SCALAR_OP_H
#define MLIR_HLO_MHLO_TRANSFORMS_MAP_MHLO_TO_SCALAR_OP_H

#include <optional>
#include <type_traits>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace mhlo {
namespace impl {

// A struct to map MhloBinaryOpTy type to the corresponding floating-point and
// integer scalar operation types.
template <typename MhloBinaryOpTy>
struct MhloToScalarOp {
  using FOp = void;
  using IOp = void;
  using UOp = void;
  using COp = void;
};

template <>
struct MhloToScalarOp<mhlo::AddOp> {
  using FOp = ::mlir::arith::AddFOp;
  using IOp = ::mlir::arith::AddIOp;
  using UOp = ::mlir::arith::AddIOp;
  using COp = ::mlir::complex::AddOp;
};
template <>
struct MhloToScalarOp<mhlo::AndOp> {
  using IOp = ::mlir::arith::AndIOp;
  using UOp = ::mlir::arith::AndIOp;
};
template <>
struct MhloToScalarOp<mhlo::CbrtOp> {
  using FOp = ::mlir::math::CbrtOp;
};
template <>
struct MhloToScalarOp<mhlo::CompareOp> {
  using FOp = ::mlir::arith::CmpFOp;
  using IOp = ::mlir::arith::CmpIOp;
  using UOp = ::mlir::arith::CmpIOp;
};
template <>
struct MhloToScalarOp<mhlo::CeilOp> {
  using FOp = ::mlir::math::CeilOp;
};
template <>
struct MhloToScalarOp<mhlo::ClzOp> {
  using IOp = ::mlir::math::CountLeadingZerosOp;
  using UOp = ::mlir::math::CountLeadingZerosOp;
};
template <>
struct MhloToScalarOp<mhlo::CosineOp> {
  using FOp = ::mlir::math::CosOp;
  using COp = ::mlir::complex::CosOp;
};
template <>
struct MhloToScalarOp<mhlo::ErfOp> {
  using FOp = ::mlir::math::ErfOp;
};
template <>
struct MhloToScalarOp<mhlo::ExpOp> {
  using FOp = ::mlir::math::ExpOp;
  using COp = ::mlir::complex::ExpOp;
};
template <>
struct MhloToScalarOp<mhlo::Expm1Op> {
  using FOp = ::mlir::math::ExpM1Op;
  using COp = ::mlir::complex::Expm1Op;
};
template <>
struct MhloToScalarOp<mhlo::FloorOp> {
  using FOp = ::mlir::math::FloorOp;
};
template <>
struct MhloToScalarOp<mhlo::LogOp> {
  using FOp = ::mlir::math::LogOp;
  using COp = ::mlir::complex::LogOp;
};
template <>
struct MhloToScalarOp<mhlo::Log1pOp> {
  using FOp = ::mlir::math::Log1pOp;
  using COp = ::mlir::complex::Log1pOp;
};
template <>
struct MhloToScalarOp<mhlo::MulOp> {
  using FOp = ::mlir::arith::MulFOp;
  using IOp = ::mlir::arith::MulIOp;
  using UOp = ::mlir::arith::MulIOp;
  using COp = ::mlir::complex::MulOp;
};
template <>
struct MhloToScalarOp<mhlo::OrOp> {
  using IOp = ::mlir::arith::OrIOp;
  using UOp = ::mlir::arith::OrIOp;
};
template <>
struct MhloToScalarOp<mhlo::PopulationCountOp> {
  using IOp = ::mlir::math::CtPopOp;
  using UOp = ::mlir::math::CtPopOp;
};
template <>
struct MhloToScalarOp<mhlo::RsqrtOp> {
  using FOp = ::mlir::math::RsqrtOp;
  using COp = ::mlir::complex::RsqrtOp;
};
template <>
struct MhloToScalarOp<mhlo::RoundNearestEvenOp> {
  using FOp = ::mlir::math::RoundEvenOp;
};
template <>
struct MhloToScalarOp<mhlo::RoundOp> {
  using FOp = ::mlir::math::RoundOp;
};
template <>
struct MhloToScalarOp<mhlo::SubtractOp> {
  using FOp = ::mlir::arith::SubFOp;
  using IOp = ::mlir::arith::SubIOp;
  using UOp = ::mlir::arith::SubIOp;
  using COp = ::mlir::complex::SubOp;
};
template <>
struct MhloToScalarOp<mhlo::SqrtOp> {
  using FOp = ::mlir::math::SqrtOp;
  using COp = ::mlir::complex::SqrtOp;
};
template <>
struct MhloToScalarOp<mhlo::SineOp> {
  using FOp = ::mlir::math::SinOp;
  using COp = ::mlir::complex::SinOp;
};
template <>
struct MhloToScalarOp<mhlo::TanOp> {
  using FOp = ::mlir::math::TanOp;
  using COp = ::mlir::complex::TanOp;
};
template <>
struct MhloToScalarOp<mhlo::Atan2Op> {
  using FOp = ::mlir::math::Atan2Op;
  using COp = ::mlir::complex::Atan2Op;
};
template <>
struct MhloToScalarOp<mhlo::TanhOp> {
  using FOp = ::mlir::math::TanhOp;
  using COp = ::mlir::complex::TanhOp;
};
template <>
struct MhloToScalarOp<mhlo::XorOp> {
  using IOp = ::mlir::arith::XOrIOp;
  using UOp = ::mlir::arith::XOrIOp;
};

// Alias for the map from MHLO binary op type to STD floating-point op type.
template <typename MhloOp>
using ScalarFOp = typename MhloToScalarOp<MhloOp>::FOp;
// Alias for the map from MHLO binary op type to STD signed integer op type.
template <typename MhloOp>
using ScalarIOp = typename MhloToScalarOp<MhloOp>::IOp;
// Alias for the map from MHLO binary op type to STD unsigned integer op type.
template <typename MhloOp>
using ScalarUOp = typename MhloToScalarOp<MhloOp>::UOp;
// Alias for the map from MHLO binary op type to STD complex op type.
template <typename MhloOp>
using ScalarCOp = typename MhloToScalarOp<MhloOp>::COp;

template <typename... Args>
struct MapMhloOpToScalarOpImpl {
  Value operator()(Location /*loc*/, ArrayRef<Type> /*ResultTypes*/,
                   ArrayRef<Type> /*argTypes*/, ValueRange /*args*/,
                   OpBuilder* /*b*/) {
    return nullptr;
  }
};

template <typename StdScalarOp>
struct MapMhloOpToScalarOpImpl<StdScalarOp> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> /*argTypes*/, ValueRange args, OpBuilder* b) {
    return b->template create<StdScalarOp>(loc, resultTypes, args,
                                           std::nullopt);
  }
};

template <typename SupportedType, typename StdScalarOp, typename... Args>
struct MapMhloOpToScalarOpImpl<SupportedType, StdScalarOp, Args...> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> argTypes, ValueRange args, OpBuilder* b) {
    Type elementType = getElementTypeOrSelf(argTypes.front());
    if (SupportedType{}(elementType)) {
      return b->template create<StdScalarOp>(loc, resultTypes, args,
                                             std::nullopt);
    }
    return MapMhloOpToScalarOpImpl<Args...>{}(loc, resultTypes, argTypes, args,
                                              b);
  }
};

template <typename SupportedType, typename... Args>
struct MapMhloOpToScalarOpImpl<SupportedType, void, Args...> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> argTypes, ValueRange args, OpBuilder* b) {
    return MapMhloOpToScalarOpImpl<Args...>{}(loc, resultTypes, argTypes, args,
                                              b);
  }
};

struct IsAnyIntegerType {
  bool operator()(Type t) { return mlir::isa<IntegerType>(t); }
};

struct IsSignedIntegerType {
  bool operator()(Type t) {
    // Pretend that signless is signed. This will change eventually.
    return mlir::isa<IntegerType>(t) && !t.isUnsignedInteger() &&
           !t.isSignlessInteger(1);
  }
};

struct IsUnsignedIntegerType {
  bool operator()(Type t) {
    return t.isUnsignedInteger() || t.isSignlessInteger(1);
  }
};

struct IsFloatType {
  bool operator()(Type t) { return mlir::isa<FloatType>(t); }
};

struct IsComplexType {
  bool operator()(Type t) { return mlir::isa<ComplexType>(t); }
};

template <template <typename T> class MapTy, typename OpTy,
          typename PredTy = llvm::is_detected<MapTy, OpTy>>
struct MapableIf {
  using type = void;
};
template <template <typename T> class MapTy, typename OpTy>
struct MapableIf<MapTy, OpTy, std::true_type> {
  using type = MapTy<OpTy>;
};

// Inserts the computation that corresponds to the body of the loop for lowered
// MHLO unary/binary op. Returns the value for the result.
template <typename MhloOpTy>
inline Value mapMhloOpToStdScalarOp(Location loc, ArrayRef<Type> resultTypes,
                                    ArrayRef<Type> argTypes,
                                    typename MhloOpTy::Adaptor adaptor,
                                    OpBuilder* b) {
  using ScalarIOpOrVoid = typename MapableIf<ScalarIOp, MhloOpTy>::type;
  using ScalarUOpOrVoid = typename MapableIf<ScalarUOp, MhloOpTy>::type;
  using ScalarFOpOrVoid = typename MapableIf<ScalarFOp, MhloOpTy>::type;
  using ScalarCOpOrVoid = typename MapableIf<ScalarCOp, MhloOpTy>::type;
  return MapMhloOpToScalarOpImpl<IsSignedIntegerType, ScalarIOpOrVoid,
                                 IsUnsignedIntegerType, ScalarUOpOrVoid,
                                 IsFloatType, ScalarFOpOrVoid, IsComplexType,
                                 ScalarCOpOrVoid>{}(loc, resultTypes, argTypes,
                                                    adaptor.getOperands(), b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::AbsOp>(Location loc,
                                                 ArrayRef<Type> resultTypes,
                                                 ArrayRef<Type> argTypes,
                                                 mhlo::AbsOp::Adaptor adaptor,
                                                 OpBuilder* b) {
  Type elementType = getElementTypeOrSelf(argTypes.front());
  if (mlir::isa<FloatType>(elementType)) {
    return MapMhloOpToScalarOpImpl<IsFloatType, ::mlir::math::AbsFOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), b);
  }
  if (mlir::isa<ComplexType>(elementType)) {
    return MapMhloOpToScalarOpImpl<IsComplexType, ::mlir::complex::AbsOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), b);
  }
  if (elementType.isSignlessInteger() || elementType.isSignedInteger()) {
    // lmhlo.abs(x, result) ->  result = select((x > 0), x, sub(0, x))
    Value lhs = adaptor.getOperand();
    Value zeroIntval =
        b->create<arith::ConstantOp>(loc, b->getZeroAttr(lhs.getType()));
    auto lhsGtZero = b->create<ScalarIOp<CompareOp>>(
        loc, arith::CmpIPredicate::sge, lhs, zeroIntval);
    auto negVal = b->create<ScalarIOp<mhlo::SubtractOp>>(loc, zeroIntval, lhs);
    return b->create<::mlir::arith::SelectOp>(loc, lhsGtZero, lhs, negVal);
  }
  return nullptr;
}

// Return a constant for v of type t, splat if t is a vector type.
inline Value getConstantOrSplat(OpBuilder* b, Location loc, Type t,
                                Attribute v) {
  if (VectorType vecType = mlir::dyn_cast<VectorType>(t)) {
    v = SplatElementsAttr::get(vecType, v);
  }
  return b->create<arith::ConstantOp>(loc, t, cast<TypedAttr>(v));
}

template <typename PredicateType>
inline std::optional<PredicateType> getCmpPredicate(mhlo::ComparisonDirection,
                                                    bool) {
  return std::nullopt;
}

template <>
inline std::optional<arith::CmpFPredicate>
getCmpPredicate<arith::CmpFPredicate>(
    mhlo::ComparisonDirection comparisonDirection, bool isSigned) {
  assert(isSigned && "cannot have an unsigned float!");
  return llvm::StringSwitch<std::optional<arith::CmpFPredicate>>(
             stringifyComparisonDirection(comparisonDirection))
      .Case("EQ", arith::CmpFPredicate::OEQ)
      .Case("NE", arith::CmpFPredicate::UNE)
      .Case("GE", arith::CmpFPredicate::OGE)
      .Case("GT", arith::CmpFPredicate::OGT)
      .Case("LE", arith::CmpFPredicate::OLE)
      .Case("LT", arith::CmpFPredicate::OLT)
      .Default(std::nullopt);
}

template <>
inline std::optional<arith::CmpIPredicate>
getCmpPredicate<arith::CmpIPredicate>(
    mhlo::ComparisonDirection comparisonDirection, bool isSigned) {
  return llvm::StringSwitch<std::optional<arith::CmpIPredicate>>(
             stringifyComparisonDirection(comparisonDirection))
      .Case("EQ", arith::CmpIPredicate::eq)
      .Case("NE", arith::CmpIPredicate::ne)
      .Case("GE",
            isSigned ? arith::CmpIPredicate::sge : arith::CmpIPredicate::uge)
      .Case("GT",
            isSigned ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::ugt)
      .Case("LE",
            isSigned ? arith::CmpIPredicate::sle : arith::CmpIPredicate::ule)
      .Case("LT",
            isSigned ? arith::CmpIPredicate::slt : arith::CmpIPredicate::ult)
      .Default(std::nullopt);
}

inline Value cmpComplex(Location loc, Value lhs, Value rhs,
                        ComparisonDirection comparisonDirection, OpBuilder* b) {
  auto complexType = mlir::cast<ComplexType>(lhs.getType());
  if (mlir::isa<FloatType>(complexType.getElementType())) {
    if (comparisonDirection == ComparisonDirection::EQ) {
      return b->create<complex::EqualOp>(loc, lhs, rhs);
    }
    if (comparisonDirection == ComparisonDirection::NE) {
      return b->create<complex::NotEqualOp>(loc, lhs, rhs);
    }

    // Perform a lexicographical comparison for the (real, imaginary) pair.
    Type complexFloatTy = complexType.getElementType();

    Value lhsReal = b->create<complex::ReOp>(loc, complexFloatTy, lhs);
    Value rhsReal = b->create<complex::ReOp>(loc, complexFloatTy, rhs);

    Value lhsImag = b->create<complex::ImOp>(loc, complexFloatTy, lhs);
    Value rhsImag = b->create<complex::ImOp>(loc, complexFloatTy, rhs);

    auto predicate = getCmpPredicate<arith::CmpFPredicate>(comparisonDirection,
                                                           /*is_signed=*/true);
    assert(predicate.has_value() && "expected valid comparison direction");

    //   if (lhsReal == rhsReal && lhsImag `predicate` rhsImag ||
    //       lhsReal `predicate` rhsReal)
    Value realsAreEq = b->create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                                lhsReal, rhsReal);
    Value imagsAreOrdered =
        b->create<arith::CmpFOp>(loc, *predicate, lhsImag, rhsImag);
    Value realsAreOrdered =
        b->create<arith::CmpFOp>(loc, *predicate, lhsReal, rhsReal);

    Value orLhs = b->create<arith::AndIOp>(loc, realsAreEq, imagsAreOrdered);
    return b->create<arith::OrIOp>(loc, orLhs, realsAreOrdered);
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::CompareOp>(
    Location loc, ArrayRef<Type> /*resultTypes*/, ArrayRef<Type> argTypes,
    mhlo::CompareOp::Adaptor adaptor, OpBuilder* b) {
  ComparisonDirection comparisonDirection = adaptor.getComparisonDirection();
  const auto& lhs = adaptor.getLhs();
  const auto& rhs = adaptor.getRhs();
  Type elementType = getElementTypeOrSelf(argTypes.front());
  if (mlir::isa<IntegerType>(elementType)) {
    bool isUnsigned = IsUnsignedIntegerType{}(elementType);
    std::optional<arith::CmpIPredicate> predicate =
        getCmpPredicate<arith::CmpIPredicate>(comparisonDirection, !isUnsigned);
    assert(predicate.has_value() && "expected valid comparison direction");
    return b->create<ScalarIOp<mhlo::CompareOp>>(loc, predicate.value(), lhs,
                                                 rhs);
  }
  if (auto floatType = mlir::dyn_cast<FloatType>(elementType)) {
    if (adaptor.getCompareType() &&
        *adaptor.getCompareType() == mhlo::ComparisonType::TOTALORDER) {
      // The semantics of totalorder fp compare are
      // -NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN
      auto intType = b->getIntegerType(floatType.getWidth());
      auto zero =
          b->create<arith::ConstantOp>(loc, intType, b->getZeroAttr(intType));
      auto max = b->create<arith::ConstantOp>(
          loc, intType,
          b->getIntegerAttr(intType,
                            APInt::getSignedMaxValue(floatType.getWidth())));
      // Switch from a floating point value to a integer value in such a way
      // that when using the integer value to compare, we get the same result
      // for normal values, and -NaN is treated as the smallest value, and NaN
      // is treated as the largest value.
      // If f is a float, and
      // x = bit_cast<int32_t>(f);
      // y = x < 0 ? numeric_limits<int32_t>::max() - x : x;
      // then y is ordered as an int32_t such that finite values have the
      // obvious order, -0 is ordered before 0, and -NaN and NaN appear at the
      // beginning and end of the ordering.
      auto toIntegral = [&](Value v) {
        auto x = b->create<arith::BitcastOp>(loc, intType, v);
        auto cmp =
            b->create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, x, zero);
        auto sub = b->create<arith::SubIOp>(loc, max, x);
        return b->create<arith::SelectOp>(loc, cmp, sub, x);
      };
      auto lhsInt = toIntegral(lhs);
      auto rhsInt = toIntegral(rhs);
      auto predicate =
          getCmpPredicate<arith::CmpIPredicate>(comparisonDirection,
                                                /*is_signed=*/true);
      assert(predicate.has_value() && "expected valid comparison direction");
      return b->create<arith::CmpIOp>(loc, *predicate, lhsInt, rhsInt);
    }
    std::optional<arith::CmpFPredicate> predicate =
        getCmpPredicate<arith::CmpFPredicate>(comparisonDirection,
                                              /*is_signed=*/true);
    assert(predicate.has_value() && "expected valid comparison direction");
    return b->create<ScalarFOp<mhlo::CompareOp>>(loc, predicate.value(), lhs,
                                                 rhs);
  }
  if (auto complexType = mlir::dyn_cast<ComplexType>(elementType))
    return cmpComplex(loc, lhs, rhs, comparisonDirection, b);
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ReducePrecisionOp>(
    Location loc, ArrayRef<Type> /*resultTypes*/, ArrayRef<Type> argTypes,
    mhlo::ReducePrecisionOp::Adaptor adaptor, OpBuilder* builder) {
  using llvm::APInt;
  mlir::ImplicitLocOpBuilder b(loc, *builder);

  // Integer and float types for casting and constant generation.
  auto floatType =
      mlir::cast<FloatType>(getElementTypeOrSelf(argTypes.front()));
  int64_t nbits = floatType.getWidth();
  auto intType = mlir::IntegerType::get(loc.getContext(), nbits);

  Value xAsInt = b.create<arith::BitcastOp>(intType, adaptor.getOperand());

  // SignificandWidth includes the implicit extra bit.
  auto srcMantissaBits = floatType.getFPMantissaWidth() - 1;
  int srcExponentBits = nbits - 1 - srcMantissaBits;

  // Clear the sign bit, it does not participate in rounding and we will restore
  // it later.
  APInt signBitMask(nbits, 1);
  signBitMask <<= nbits - 1;

  APInt expBitsMask(nbits, 1);
  expBitsMask = ((expBitsMask << srcExponentBits) - 1) << srcMantissaBits;

  auto createConstant = [&](const APInt& v) {
    return b.create<arith::ConstantIntOp>(v.getZExtValue(), intType)
        .getResult();
  };

  Value xAbsBits =
      b.create<arith::AndIOp>(xAsInt, createConstant(~signBitMask));
  Value xIsNan = b.create<arith::CmpIOp>(arith::CmpIPredicate::ugt, xAbsBits,
                                         createConstant(expBitsMask));

  int destMantissaBits = adaptor.getMantissaBits();
  if (destMantissaBits < static_cast<int>(srcMantissaBits)) {
    // Last remaining mantissa bit.
    APInt lastMantissaBitMask(nbits, 1);
    lastMantissaBitMask <<= srcMantissaBits - destMantissaBits;

    // Compute rounding bias for round-to-nearest with ties to even.  This is
    // equal to a base value of 0111... plus one bit if the last remaining
    // mantissa bit is 1.
    APInt baseRoundingBias = lastMantissaBitMask.lshr(1) - 1;

    Value mantissaDiff = b.create<arith::ConstantIntOp>(
        srcMantissaBits - destMantissaBits, intType);
    Value highestMantissaMaskVal = createConstant(lastMantissaBitMask);
    Value baseRoundingBiasVal = createConstant(baseRoundingBias);
    Value xLastMantissaBit = b.create<arith::ShRUIOp>(
        b.create<arith::AndIOp>(xAsInt, highestMantissaMaskVal), mantissaDiff);
    Value xRoundingBias =
        b.create<arith::AddIOp>(xLastMantissaBit, baseRoundingBiasVal);

    // Add rounding bias, and mask out truncated bits.  Note that the case
    // where adding the rounding bias overflows into the exponent bits is
    // correct; the non-masked mantissa bits will all be zero, and the
    // exponent will be incremented by one.
    APInt truncationMask = ~(lastMantissaBitMask - 1);
    Value xRounded = b.create<arith::AddIOp>(xAsInt, xRoundingBias);
    xAsInt = b.create<arith::AndIOp>(xRounded, createConstant(truncationMask));
  }

  int destExponentBits = adaptor.getExponentBits();
  if (destExponentBits < srcExponentBits) {
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
    APInt exponentBias(nbits, 1);
    exponentBias = (exponentBias << (srcExponentBits - 1)) - 1;

    APInt reducedExponentBias(nbits, 1);
    reducedExponentBias = (reducedExponentBias << (destExponentBits - 1)) - 1;

    APInt reducedMaxExponent = exponentBias + reducedExponentBias;
    APInt reducedMinExponent = exponentBias - reducedExponentBias;

    // Do we overflow or underflow?
    Value xExponent =
        b.create<arith::AndIOp>(xAsInt, createConstant(expBitsMask));
    Value xOverflows = b.create<arith::CmpIOp>(
        arith::CmpIPredicate::ugt, xExponent,
        createConstant(reducedMaxExponent << srcMantissaBits));
    Value xUnderflows = b.create<arith::CmpIOp>(
        arith::CmpIPredicate::ule, xExponent,
        createConstant(reducedMinExponent << srcMantissaBits));

    // Compute appropriately-signed values of zero and infinity.
    Value xSignedZero =
        b.create<arith::AndIOp>(xAsInt, createConstant(signBitMask));
    Value xSignedInf =
        b.create<arith::OrIOp>(xSignedZero, createConstant(expBitsMask));

    // Force to zero or infinity if overflow or underflow.  (Note that this
    // truncates all denormal values to zero, rather than rounding them.)
    xAsInt = b.create<arith::SelectOp>(xOverflows, xSignedInf, xAsInt);
    xAsInt = b.create<arith::SelectOp>(xUnderflows, xSignedZero, xAsInt);
  }

  Value result = b.create<arith::BitcastOp>(floatType, xAsInt);
  return b.create<arith::SelectOp>(xIsNan, adaptor.getOperand(), result);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::CopyOp>(
    Location /*loc*/, ArrayRef<Type> /*ResultTypes*/,
    ArrayRef<Type> /*argTypes*/, mhlo::CopyOp::Adaptor adaptor,
    OpBuilder* /*b*/) {
  return adaptor.getOperands().front();
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ComplexOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::ComplexOp::Adaptor adaptor, OpBuilder* b) {
  return MapMhloOpToScalarOpImpl<complex::CreateOp>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::MaxOp>(Location loc,
                                                 ArrayRef<Type> resultTypes,
                                                 ArrayRef<Type> argTypes,
                                                 mhlo::MaxOp::Adaptor adaptor,
                                                 OpBuilder* b) {
  ValueRange operands = adaptor.getOperands();
  Value lhs = operands.front();
  Type complexTy = lhs.getType();

  if (!mlir::isa<ComplexType>(complexTy))
    return MapMhloOpToScalarOpImpl<IsFloatType, arith::MaximumFOp,
                                   IsSignedIntegerType, arith::MaxSIOp,
                                   IsUnsignedIntegerType, arith::MaxUIOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), b);

  assert(resultTypes.size() == 1 && "MaxOp should return a single result");
  assert(operands.size() == 2 && "MaxOp should take exactly two arguments");

  Value rhs = operands.back();
  // 'max' performs a lexicographical comparison for the (real, imaginary) pair.
  Value cond = cmpComplex(loc, lhs, rhs, ComparisonDirection::GE, b);

  return b->create<arith::SelectOp>(loc, cond, lhs, rhs).getResult();
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::MinOp>(Location loc,
                                                 ArrayRef<Type> resultTypes,
                                                 ArrayRef<Type> argTypes,
                                                 mhlo::MinOp::Adaptor adaptor,
                                                 OpBuilder* b) {
  ValueRange operands = adaptor.getOperands();
  Value lhs = operands.front();
  Type complexTy = lhs.getType();

  if (!mlir::isa<ComplexType>(complexTy))
    return MapMhloOpToScalarOpImpl<IsFloatType, arith::MinimumFOp,
                                   IsSignedIntegerType, arith::MinSIOp,
                                   IsUnsignedIntegerType, arith::MinUIOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), b);

  assert(resultTypes.size() == 1 && "MinOp should return a single result");
  assert(operands.size() == 2 && "MinOp should take exactly two arguments");

  Value rhs = operands.back();
  // 'min' performs a lexicographical comparison for the (real, imaginary) pair.
  Value cond = cmpComplex(loc, lhs, rhs, ComparisonDirection::LE, b);

  return b->create<arith::SelectOp>(loc, cond, lhs, rhs).getResult();
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::RealOp>(Location loc,
                                                  ArrayRef<Type> resultTypes,
                                                  ArrayRef<Type> argTypes,
                                                  mhlo::RealOp::Adaptor adaptor,
                                                  OpBuilder* b) {
  if (!mlir::isa<ComplexType>(adaptor.getOperand().getType()))
    return adaptor.getOperand();
  return MapMhloOpToScalarOpImpl<complex::ReOp>{}(loc, resultTypes, argTypes,
                                                  adaptor.getOperands(), b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ImagOp>(Location loc,
                                                  ArrayRef<Type> resultTypes,
                                                  ArrayRef<Type> argTypes,
                                                  mhlo::ImagOp::Adaptor adaptor,
                                                  OpBuilder* b) {
  if (!mlir::isa<ComplexType>(adaptor.getOperand().getType()))
    return b->create<arith::ConstantOp>(
        loc, b->getZeroAttr(adaptor.getOperand().getType()));
  return MapMhloOpToScalarOpImpl<complex::ImOp>{}(loc, resultTypes, argTypes,
                                                  adaptor.getOperands(), b);
}

// 'target_types' is the unconverted type (signed or unsigned if integer),
// 'ResultTypes' is the converted type (signless if integer).
inline Value mapConvertOpToStdScalarOp(Location loc, ArrayRef<Type> targetTypes,
                                       ArrayRef<Type> resultTypes,
                                       ArrayRef<Type> argTypes, ValueRange args,
                                       OpBuilder* b) {
  assert(targetTypes.size() == 1 && "ConvertOp should return a single result");
  assert(resultTypes.size() == 1 && "ConvertOp should return a single result");
  assert(argTypes.size() == 1 && "ConvertOp should take a single argument");
  assert(args.size() == 1 && "ConvertOp should take a single argument");

  Type sourceType = getElementTypeOrSelf(argTypes.front());
  Type targetType = getElementTypeOrSelf(targetTypes.front());
  Type convertedSourceType = getElementTypeOrSelf(args.front());

  // A boolean value is considered to be unsigned when converting to
  // floating-point. Otherwise, it will become `-1`.
  if (IsUnsignedIntegerType{}(sourceType) &&
      mlir::arith::UIToFPOp::areCastCompatible(convertedSourceType,
                                               targetType)) {
    return b->create<mlir::arith::UIToFPOp>(loc, resultTypes, args,
                                            std::nullopt);
  }
  if (mlir::arith::SIToFPOp::areCastCompatible(sourceType, targetType)) {
    return b->create<mlir::arith::SIToFPOp>(loc, resultTypes, args,
                                            std::nullopt);
  }
  if (mlir::isa<FloatType>(sourceType) && mlir::isa<FloatType>(targetType)) {
    if (sourceType == targetType) {
      return args.front();
    }

    mlir::Value src = args.front();
    auto dst = mlir::cast<FloatType>(targetType);
    if (sourceType.getIntOrFloatBitWidth() == dst.getWidth()) {
      // There are no ops for conversions between floats of equal width, so we
      // go through the next-larger standard type.
      sourceType = dst.getWidth() == 8 ? b->getF16Type() : b->getF32Type();
      src = b->create<mlir::arith::ExtFOp>(loc, sourceType, src).getResult();
    }
    assert(sourceType.getIntOrFloatBitWidth() != dst.getWidth());

    if (sourceType.getIntOrFloatBitWidth() > dst.getWidth()) {
      return b->create<mlir::arith::TruncFOp>(loc, resultTypes, src,
                                              std::nullopt);
    }
    return b->create<mlir::arith::ExtFOp>(loc, resultTypes, src, std::nullopt);
  }
  if (targetType.isInteger(/*width=*/1)) {
    // When casting to bool, we need to compare whether the value is equal to
    // zero.
    if (sourceType.isSignlessInteger() || sourceType.isUnsignedInteger()) {
      Value zeroIntval = b->create<arith::ConstantOp>(
          loc, b->getZeroAttr(args.front().getType()));
      return b->create<mlir::arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                            args.front(), zeroIntval);
    }
    if (mlir::isa<FloatType>(sourceType)) {
      Value zero = b->create<arith::ConstantOp>(
          loc, b->getZeroAttr(args.front().getType()));
      return b->create<mlir::arith::CmpFOp>(loc, arith::CmpFPredicate::UNE,
                                            args.front(), zero);
    }
  }
  if (mlir::isa<IntegerType>(sourceType) &&
      mlir::isa<IntegerType>(targetType)) {
    auto src = mlir::cast<IntegerType>(sourceType);
    auto res = mlir::cast<IntegerType>(targetType);
    if (src.getWidth() > res.getWidth()) {
      return b->create<mlir::arith::TruncIOp>(loc, resultTypes, args,
                                              std::nullopt);
    }
    if (src.getWidth() < res.getWidth()) {
      // Special case boolean values, so they get casted to `1` instead of `-1`.
      if (IsUnsignedIntegerType{}(src)) {
        return b->create<mlir::arith::ExtUIOp>(loc, resultTypes, args,
                                               std::nullopt);
      }
      return b->create<mlir::arith::ExtSIOp>(loc, resultTypes, args,
                                             std::nullopt);
    }
    // No conversion is needed for the same width integers
    return args.front();
  }
  if (targetType.isUnsignedInteger() &&
      mlir::arith::FPToUIOp::areCastCompatible(convertedSourceType,
                                               targetType)) {
    return b->create<mlir::arith::FPToUIOp>(loc, resultTypes, args,
                                            std::nullopt);
  }
  if (mlir::arith::FPToSIOp::areCastCompatible(convertedSourceType,
                                               targetType)) {
    return b->create<mlir::arith::FPToSIOp>(loc, resultTypes, args,
                                            std::nullopt);
  }
  if (mlir::isa<ComplexType>(targetType)) {
    Type targetElementType =
        mlir::cast<ComplexType>(targetType).getElementType();
    assert(!mlir::isa<ComplexType>(targetElementType) &&
           "elements of complex numbers should not be complex");
    Value targetReal;
    Value targetImag;
    if (mlir::isa<ComplexType>(sourceType)) {
      // We are converting from complex type: convert real and imaginary parts
      // separately.
      Type sourceElementType =
          mlir::cast<ComplexType>(sourceType).getElementType();
      assert(!mlir::isa<ComplexType>(sourceElementType) &&
             "elements of complex numbers should not be complex");
      Value sourceReal =
          b->create<mlir::complex::ReOp>(loc, sourceElementType, args.front());
      targetReal =
          mapConvertOpToStdScalarOp(loc, targetElementType, targetElementType,
                                    sourceElementType, sourceReal, b);
      Value sourceImag =
          b->create<mlir::complex::ImOp>(loc, sourceElementType, args.front());
      targetImag =
          mapConvertOpToStdScalarOp(loc, targetElementType, targetElementType,
                                    sourceElementType, sourceImag, b);
    } else {
      // We are converting from real (float, integer, etc.) type, convert the
      // real part and set the imaginary part to 0.
      targetReal = mapConvertOpToStdScalarOp(
          loc, targetElementType, targetElementType, argTypes, args, b);
      targetImag = b->create<mlir::arith::ConstantOp>(
          loc, b->getFloatAttr(targetElementType, 0.0));
    }
    return b->create<mlir::complex::CreateOp>(loc, targetType, targetReal,
                                              targetImag);
  }
  if (auto sourceComplexType = mlir::dyn_cast<ComplexType>(sourceType)) {
    auto sourceElementType = sourceComplexType.getElementType();
    // When converting from complex to a non-complex type, we take just the real
    // part of the complex number.
    Value sourceReal =
        b->create<mlir::complex::ReOp>(loc, sourceElementType, args.front());
    return mapConvertOpToStdScalarOp(loc, targetTypes, resultTypes,
                                     sourceElementType, sourceReal, b);
  }
  return nullptr;
}

/// Lower bitcast operations where the input and resulting type are the same
/// bitwidth, thus implying that the operation is fully defined by parallel
/// loops and scalar operations without any shape dimension changes.
template <>
inline Value mapMhloOpToStdScalarOp<mhlo::BitcastConvertOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::BitcastConvertOp::Adaptor adaptor, OpBuilder* b) {
  Type argType = getElementTypeOrSelf(argTypes.front());
  Type resultType = getElementTypeOrSelf(resultTypes.front());

  if (resultType.getIntOrFloatBitWidth() != argType.getIntOrFloatBitWidth())
    return nullptr;

  return b->create<mlir::arith::BitcastOp>(loc, resultTypes,
                                           adaptor.getOperands());
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::DotOp>(Location loc,
                                                 ArrayRef<Type> resultTypes,
                                                 ArrayRef<Type> argTypes,
                                                 mhlo::DotOp::Adaptor adaptor,
                                                 OpBuilder* b) {
  // Dot Op converter from lhlo to affine only accepts float and integer types.
  const auto& lhs = adaptor.getOperands()[0];
  const auto& rhs = adaptor.getOperands()[1];
  const auto& result = adaptor.getOperands()[2];
  Type elementType = lhs.getType();
  if (mlir::isa<FloatType>(elementType)) {
    Value floatMul =
        MapMhloOpToScalarOpImpl<IsFloatType, ::mlir::arith::MulFOp>{}(
            loc, resultTypes, argTypes, {lhs, rhs}, b);
    return MapMhloOpToScalarOpImpl<IsFloatType, ::mlir::arith::AddFOp>{}(
        loc, resultTypes, argTypes, {floatMul, result}, b);
  }
  if (mlir::isa<IntegerType>(elementType)) {
    Value intMul =
        MapMhloOpToScalarOpImpl<IsAnyIntegerType, ::mlir::arith::MulIOp>{}(
            loc, resultTypes, argTypes, {lhs, rhs}, b);
    return MapMhloOpToScalarOpImpl<IsAnyIntegerType, ::mlir::arith::AddIOp>{}(
        loc, resultTypes, argTypes, {intMul, result}, b);
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::IsFiniteOp>(
    Location loc, ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
    mhlo::IsFiniteOp::Adaptor adaptor, OpBuilder* b) {
  if (mlir::isa<FloatType>(adaptor.getX().getType())) {
    auto posInf = APFloat::getInf(
        mlir::cast<FloatType>(adaptor.getX().getType()).getFloatSemantics());
    auto constPosInf = b->create<arith::ConstantOp>(
        loc, b->getFloatAttr(adaptor.getX().getType(), posInf));
    Value absX = b->create<::mlir::math::AbsFOp>(loc, adaptor.getX());
    return b->create<::mlir::arith::CmpFOp>(loc, arith::CmpFPredicate::ONE,
                                            absX, constPosInf);
  }
  return nullptr;
}

/// Implements the conversion of HLO op to scalar op (to use within region of a
/// linalg.generic op) for compare-select style operations like min/max.
template <typename... Args>
struct CompareSelectOpToStdScalarOp {
  static Value map(Location /*loc*/,
                   ComparisonDirection /*comparison_direction*/,
                   ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
                   ValueRange /*args*/, OpBuilder* /*b*/) {
    return nullptr;
  }
};

/// Specialization which allows converting to a comparison operation in standard
/// dialect with a given predicate based on the element type of the operand.
template <typename SupportedType, typename StdCompareOp, typename Predicate,
          typename... Args>
struct CompareSelectOpToStdScalarOp<SupportedType, StdCompareOp, Predicate,
                                    Args...> {
  static Value map(Location loc, ComparisonDirection comparisonDirection,
                   ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
                   ValueRange args, OpBuilder* b) {
    Type elementType = getElementTypeOrSelf(argTypes.front());
    if (isa<SupportedType>(elementType)) {
      auto predicate = getCmpPredicate<Predicate>(
          comparisonDirection, !elementType.isUnsignedInteger());
      assert(predicate.has_value() && "expected valid comparison direction");
      auto cmp = b->template create<StdCompareOp>(loc, predicate.getValue(),
                                                  args[0], args[1]);
      return b->create<::mlir::arith::SelectOp>(loc, cmp, args[0], args[1]);
    }
    return CompareSelectOpToStdScalarOp<Args...>::map(
        loc, comparisonDirection, resultTypes, argTypes, args, b);
  }
};

inline Value mhloAlwaysPropagateNaN(Value v, ValueRange args, Location loc,
                                    OpBuilder* b) {
  Type elementType = getElementTypeOrSelf(args.front().getType());
  if (auto floatType = mlir::dyn_cast<FloatType>(elementType)) {
    Value isnan = b->create<mlir::arith::CmpFOp>(loc, arith::CmpFPredicate::UNO,
                                                 args[0], args[1]);

    auto nanApfloat = APFloat::getQNaN(floatType.getFloatSemantics());
    Value nan = getConstantOrSplat(b, loc, args[0].getType(),
                                   b->getFloatAttr(floatType, nanApfloat));
    v = b->create<mlir::arith::SelectOp>(loc, isnan, nan, v);
  }
  return v;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ClampOp>(Location loc,
                                                   ArrayRef<Type> resultTypes,
                                                   ArrayRef<Type> argTypes,
                                                   mhlo::ClampOp::Adaptor op,
                                                   OpBuilder* b) {
  // clamp(lb, x, ub) = min(max(lb, x), ub)
  Value maxLbX = mapMhloOpToStdScalarOp<mhlo::MaxOp>(
      loc, resultTypes, argTypes, ValueRange{op.getMin(), op.getOperand()}, b);
  return mapMhloOpToStdScalarOp<mhlo::MinOp>(
      loc, resultTypes, argTypes, ValueRange{maxLbX, op.getMax()}, b);
}

template <typename U, typename S>
inline Value makeSafeIntDiv(ImplicitLocOpBuilder& lb, Type originalType,
                            Value lhs, Value rhs, Value returnedOnZero,
                            Value returnedOnSignedOverflow) {
  Type type = lhs.getType();
  auto elementType = mlir::cast<IntegerType>(getElementTypeOrSelf(type));
  Value zero = lb.create<arith::ConstantOp>(lb.getZeroAttr(type));
  auto makeConstant = [&](const APInt& i) {
    return getConstantOrSplat(&lb, lb.getLoc(), type,
                              lb.getIntegerAttr(elementType, i));
  };
  Value one = makeConstant(APInt(elementType.getWidth(), 1));
  Value rhsIsZero =
      lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq, rhs, zero);

  // For unsigned just set the divisor to 1 when it would be 0.
  if (originalType.isUnsignedInteger()) {
    Value safeRhs = lb.create<arith::SelectOp>(rhsIsZero, one, rhs);
    Value safeDiv = lb.create<U>(lhs, safeRhs);
    return lb.create<arith::SelectOp>(rhsIsZero, returnedOnZero, safeDiv);
  }

  // For signed also check for INT_MIN / -1.
  Value smin = makeConstant(APInt::getSignedMinValue(elementType.getWidth()));
  Value lhsIsSmin =
      lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lhs, smin);
  Value minusOne = makeConstant(APInt::getAllOnes(elementType.getWidth()));
  Value rhsIsMinusOne =
      lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq, rhs, minusOne);
  Value hasIntMinOverflow = lb.create<arith::AndIOp>(lhsIsSmin, rhsIsMinusOne);
  Value rhsIsUnsafe = lb.create<arith::OrIOp>(rhsIsZero, hasIntMinOverflow);
  Value safeRhs = lb.create<arith::SelectOp>(rhsIsUnsafe, one, rhs);
  Value safeDiv = lb.create<S>(lhs, safeRhs);
  Value safeSmin = lb.create<arith::SelectOp>(
      hasIntMinOverflow, returnedOnSignedOverflow, safeDiv);
  return lb.create<arith::SelectOp>(rhsIsZero, returnedOnZero, safeSmin);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::DivOp>(Location loc,
                                                 ArrayRef<Type> resultTypes,
                                                 ArrayRef<Type> argTypes,
                                                 mhlo::DivOp::Adaptor adaptor,
                                                 OpBuilder* b) {
  Type originalType = getElementTypeOrSelf(argTypes.front());
  if (mlir::isa<ComplexType, FloatType>(originalType)) {
    return MapMhloOpToScalarOpImpl<IsFloatType, arith::DivFOp, IsComplexType,
                                   complex::DivOp>{}(loc, resultTypes, argTypes,
                                                     adaptor.getOperands(), b);
  }

  // Integer division overflow behavior:
  //
  // X / 0 == -1
  // INT_SMIN /s -1 = INT_SMIN
  ImplicitLocOpBuilder lb(loc, *b);
  Type type = adaptor.getLhs().getType();
  auto elementType = mlir::cast<IntegerType>(getElementTypeOrSelf(type));
  auto makeConstant = [&](const APInt& i) {
    return getConstantOrSplat(&lb, lb.getLoc(), type,
                              lb.getIntegerAttr(elementType, i));
  };
  Value minusOne = makeConstant(APInt::getAllOnes(elementType.getWidth()));
  Value smin = makeConstant(APInt::getSignedMinValue(elementType.getWidth()));
  return makeSafeIntDiv<arith::DivUIOp, arith::DivSIOp>(
      lb, originalType, adaptor.getLhs(), adaptor.getRhs(),
      /*returnedOnZero=*/minusOne,
      /*returnedOnSignedOverflow=*/smin);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::RemOp>(Location loc,
                                                 ArrayRef<Type> resultTypes,
                                                 ArrayRef<Type> argTypes,
                                                 mhlo::RemOp::Adaptor adaptor,
                                                 OpBuilder* b) {
  Type originalType = getElementTypeOrSelf(argTypes.front());
  if (mlir::isa<ComplexType, FloatType>(originalType)) {
    return MapMhloOpToScalarOpImpl<IsFloatType, arith::RemFOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), b);
  }

  // Integer remainder overflow behavior:
  //
  // X % 0 == X
  // INT_SMIN %s -1 = 0
  ImplicitLocOpBuilder lb(loc, *b);
  Type type = adaptor.getLhs().getType();
  Value zero = lb.create<arith::ConstantOp>(lb.getZeroAttr(type));
  return makeSafeIntDiv<arith::RemUIOp, arith::RemSIOp>(
      lb, originalType, adaptor.getLhs(), adaptor.getRhs(),
      /*returnedOnZero=*/adaptor.getLhs(),
      /*returnedOnSignedOverflow=*/zero);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::NegOp>(Location loc,
                                                 ArrayRef<Type> resultTypes,
                                                 ArrayRef<Type> argTypes,
                                                 mhlo::NegOp::Adaptor adaptor,
                                                 OpBuilder* b) {
  Type elementType = getElementTypeOrSelf(adaptor.getOperand().getType());
  if (mlir::isa<ComplexType, FloatType>(elementType)) {
    return MapMhloOpToScalarOpImpl<IsFloatType, ::mlir::arith::NegFOp,
                                   IsComplexType, ::mlir::complex::NegOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), b);
  }
  if (mlir::isa<IntegerType>(elementType)) {
    // lmhlo.neg(x, result) -> result = sub(0, x)
    Value lhs = adaptor.getOperand();
    Value zeroIntval =
        b->create<arith::ConstantOp>(loc, b->getZeroAttr(lhs.getType()));
    return b->create<ScalarIOp<mhlo::SubtractOp>>(loc, zeroIntval, lhs);
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::NotOp>(Location loc,
                                                 ArrayRef<Type> /*ResultTypes*/,
                                                 ArrayRef<Type> /*argTypes*/,
                                                 mhlo::NotOp::Adaptor adaptor,
                                                 OpBuilder* b) {
  Type elementType = getElementTypeOrSelf(adaptor.getOperand().getType());
  if (auto integerType = mlir::dyn_cast<IntegerType>(elementType)) {
    // lmhlo.not(x) -> x ^ -1
    Value allOnes = getConstantOrSplat(
        b, loc, adaptor.getOperand().getType(),
        b->getIntegerAttr(integerType,
                          APInt::getAllOnes(integerType.getWidth())));
    return b->create<::mlir::arith::XOrIOp>(loc, allOnes, adaptor.getOperand());
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::LogisticOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> /*argTypes*/,
    mhlo::LogisticOp::Adaptor adaptor, OpBuilder* b) {
  // 1.0 / (1.0 + exp(-x))
  Value negX = mapMhloOpToStdScalarOp<mhlo::NegOp>(
      loc, resultTypes, resultTypes, {adaptor.getOperand()}, b);
  Value expNegX = mapMhloOpToStdScalarOp<mhlo::ExpOp>(loc, resultTypes,
                                                      resultTypes, {{negX}}, b);

  Type type = getElementTypeOrSelf(resultTypes[0]);
  Value oneFloat =
      mlir::isa<ComplexType>(type)
          ? b->create<arith::ConstantOp>(loc, b->getF32FloatAttr(1.0))
          : getConstantOrSplat(b, loc, resultTypes[0],
                               FloatAttr::get(type, 1.0f));
  Value one = mapConvertOpToStdScalarOp(loc, resultTypes, resultTypes,
                                        {oneFloat.getType()}, {{oneFloat}}, b);
  Value oneAddExprNegX = mapMhloOpToStdScalarOp<mhlo::AddOp>(
      loc, resultTypes, resultTypes, {{expNegX, one}}, b);
  return mapMhloOpToStdScalarOp<mhlo::DivOp>(loc, resultTypes, resultTypes,
                                             {{one, oneAddExprNegX}}, b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::PowOp>(Location loc,
                                                 ArrayRef<Type> resultTypes,
                                                 ArrayRef<Type> argTypes,
                                                 mhlo::PowOp::Adaptor adaptor,
                                                 OpBuilder* b) {
  auto lb = ImplicitLocOpBuilder(loc, *b);
  // TODO: b/315868720 Consider alternate lowerings of mhlo::PowOp with integer
  // operands. Floating point can use std::powf
  auto resultType = getElementTypeOrSelf(resultTypes.front());
  if (mlir::isa<ComplexType, FloatType>(resultType)) {
    return MapMhloOpToScalarOpImpl<IsFloatType, math::PowFOp, IsComplexType,
                                   complex::PowOp>{}(loc, resultTypes, argTypes,
                                                     adaptor.getOperands(), b);
  }

  // Exponentiation by squaring:
  // https://en.wikipedia.org/wiki/Exponentiation_by_squaring;
  Value negOne =
      lb.create<arith::ConstantOp>(lb.getIntegerAttr(resultType, -1));
  Value zero = lb.create<arith::ConstantOp>(lb.getIntegerAttr(resultType, 0));
  Value one = lb.create<arith::ConstantOp>(lb.getIntegerAttr(resultType, 1));
  Value two = lb.create<arith::ConstantOp>(lb.getIntegerAttr(resultType, 2));
  Value step = lb.create<arith::ConstantIndexOp>(1);
  Value lowerBound = lb.create<arith::ConstantIndexOp>(0);
  // Everything else would overflow for any exponent > 1, as 2^64
  // is the larget possible exponent for a 64-bit integer, and
  // that's 1 << 6.
  Value upperBound = lb.create<arith::ConstantIndexOp>(6);
  auto originalBase = adaptor.getLhs();
  auto originalExponent = adaptor.getRhs();

  Value accum =
      lb.create<scf::ForOp>(
            lowerBound, upperBound, step,
            SmallVector<Value>({one, originalBase, originalExponent}),
            [&](OpBuilder& b, Location, Value /*v*/, ValueRange iters) {
              Value accum = iters[0];
              Value base = iters[1];
              Value exponent = iters[2];

              Value condition = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq,
                  b.create<::mlir::arith::AndIOp>(loc, exponent, one), one);
              Value multiplied =
                  b.create<::mlir::arith::MulIOp>(loc, accum, base);
              accum = b.create<::mlir::arith::SelectOp>(loc, condition,
                                                        multiplied, accum);
              base = b.create<::mlir::arith::MulIOp>(loc, base, base);
              exponent = b.create<::mlir::arith::ShRUIOp>(loc, exponent, one);
              b.create<scf::YieldOp>(
                  loc, SmallVector<Value>({accum, base, exponent}));
            })
          .getResult(0);

  Value rhsIsEven = lb.create<arith::CmpIOp>(
      arith::CmpIPredicate::eq,
      lb.create<arith::RemSIOp>(adaptor.getRhs(), two), zero);
  Value rhsIsNegative = lb.create<arith::CmpIOp>(arith::CmpIPredicate::slt,
                                                 adaptor.getRhs(), zero);
  Value lhsIsOne =
      lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq, adaptor.getLhs(), one);
  Value lhsIsNegOne = lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                               adaptor.getLhs(), negOne);

  // The accum is correct when the rhs is non-negative. When rhs is
  // negative, we return 0 for integer, with the exception of lhs values of 1
  // and -1 which have integer results for negative exponents. Specifically, the
  // calculation is the following:
  //
  // - Return accum if the rhs is not negative.
  // - Return 1 or -1 depending on the parity of rhs when the lhs is -1.
  // - Return 1 if lhs is 1.
  // - Else return 0.
  Value ifLhsIsOne = lb.create<::mlir::arith::SelectOp>(lhsIsOne, one, zero);
  Value ifLhsIsNegOne = lb.create<::mlir::arith::SelectOp>(
      lhsIsNegOne, lb.create<::mlir::arith::SelectOp>(rhsIsEven, one, negOne),
      ifLhsIsOne);
  return lb.create<::mlir::arith::SelectOp>(rhsIsNegative, ifLhsIsNegOne,
                                            accum);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::SelectOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::SelectOp::Adaptor adaptor, OpBuilder* b) {
  return MapMhloOpToScalarOpImpl<::mlir::arith::SelectOp>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::SignOp>(Location loc,
                                                  ArrayRef<Type> resultTypes,
                                                  ArrayRef<Type> /*argTypes*/,
                                                  mhlo::SignOp::Adaptor adaptor,
                                                  OpBuilder* b) {
  Value operand = adaptor.getOperand();
  Type elementType = getElementTypeOrSelf(operand.getType());
  if (auto floatType = mlir::dyn_cast<FloatType>(elementType)) {
    Value zero =
        b->create<arith::ConstantOp>(loc, b->getZeroAttr(operand.getType()));
    Value ne0I1 = b->create<::mlir::arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE, operand, zero);
    Value ne0Float =
        b->create<::mlir::arith::UIToFPOp>(loc, zero.getType(), ne0I1);
    Value copySign = b->create<::mlir::math::CopySignOp>(loc, resultTypes,
                                                         ne0Float, operand);
    auto isNan = b->create<::mlir::arith::CmpFOp>(
        loc, arith::CmpFPredicate::UNO, operand, operand);
    return b->create<::mlir::arith::SelectOp>(loc, isNan, operand, copySign);
  }
  if (auto integerType = mlir::dyn_cast<IntegerType>(elementType)) {
    // sign(x) = x == 0 ? 0 : ((x s>> 31) | 1)
    Value zero =
        b->create<arith::ConstantOp>(loc, b->getZeroAttr(operand.getType()));
    Value bitwidthMinusOne = getConstantOrSplat(
        b, loc, operand.getType(),
        b->getIntegerAttr(integerType, integerType.getWidth() - 1));
    Value one = getConstantOrSplat(b, loc, operand.getType(),
                                   b->getIntegerAttr(integerType, 1));
    Value cmp = b->create<::mlir::arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 operand, zero);
    Value ashr =
        b->create<::mlir::arith::ShRSIOp>(loc, operand, bitwidthMinusOne);
    Value orOp = b->create<::mlir::arith::OrIOp>(loc, ashr, one);
    return b->create<::mlir::arith::SelectOp>(loc, cmp, zero, orOp);
  }
  if (mlir::isa<ComplexType>(elementType)) {
    return b->create<::mlir::complex::SignOp>(loc, elementType, operand);
  }
  return nullptr;
}

/// Construct operations to select the saturated value if the shift amount is
/// greater than the bitwidth of the type.
inline Value selectShiftedOrSaturated(ImplicitLocOpBuilder& lb, Value rhs,
                                      Value shifted, Value saturated,
                                      Type type) {
  Type etype = mlir::isa<ShapedType>(type)
                   ? mlir::cast<ShapedType>(type).getElementType()
                   : type;
  auto bitWidthInt = etype.getIntOrFloatBitWidth();
  Value bitWidth = getConstantOrSplat(&lb, lb.getLoc(), type,
                                      lb.getIntegerAttr(etype, bitWidthInt));
  Value cmp = lb.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ugt,
                                             bitWidth, rhs);
  return lb.create<mlir::arith::SelectOp>(cmp, shifted, saturated);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ShiftLeftOp>(
    Location loc, ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
    mhlo::ShiftLeftOp::Adaptor adaptor, OpBuilder* b) {
  ImplicitLocOpBuilder lb(loc, *b);
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  Type type = lhs.getType();

  // "Saturate" if the shift is greater than the bitwidth of the type
  Value zero = lb.create<arith::ConstantOp>(lb.getZeroAttr(type));
  Value shifted = lb.create<mlir::arith::ShLIOp>(lhs, rhs);

  return selectShiftedOrSaturated(lb, rhs, shifted, zero, type);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ShiftRightLogicalOp>(
    Location loc, ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
    mhlo::ShiftRightLogicalOp::Adaptor adaptor, OpBuilder* b) {
  ImplicitLocOpBuilder lb(loc, *b);
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  Type type = lhs.getType();

  // "Saturate" if the shift is greater than the bitwidth of the type
  Value zero = lb.create<arith::ConstantOp>(b->getZeroAttr(type));
  Value shifted = lb.create<mlir::arith::ShRUIOp>(lhs, rhs);

  return selectShiftedOrSaturated(lb, rhs, shifted, zero, type);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ShiftRightArithmeticOp>(
    Location loc, ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
    mhlo::ShiftRightArithmeticOp::Adaptor adaptor, OpBuilder* b) {
  ImplicitLocOpBuilder lb(loc, *b);
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  Type type = lhs.getType();
  Type etype = mlir::isa<ShapedType>(type)
                   ? mlir::cast<ShapedType>(type).getElementType()
                   : type;
  auto bitWidthInt = etype.getIntOrFloatBitWidth();

  // "Saturate" if the shift is greater than the bitwidth of the type
  Value maxShift = getConstantOrSplat(
      b, loc, type, lb.getIntegerAttr(etype, bitWidthInt - 1));
  Value saturatedShifted = lb.create<mlir::arith::ShRSIOp>(lhs, maxShift);
  Value shifted = lb.create<mlir::arith::ShRSIOp>(lhs, rhs);

  return selectShiftedOrSaturated(lb, rhs, shifted, saturatedShifted, type);
}
}  // namespace impl

struct MhloOpToStdScalarOp {
  // Converts mhlo 'op' to linalg and arith ops.
  template <typename MhloOpTy>
  static Value mapOp(MhloOpTy op, ArrayRef<Type> resultTypes, ValueRange args,
                     OpBuilder* b) {
    auto argTypes = llvm::to_vector(op->getOperandTypes());
    return mapOpWithArgTypes(op, resultTypes, argTypes, args, b);
  }

  // Converts mhlo 'op' to linalg and arith ops. The types of 'args' may already
  // be converted, 'argTypes' are their original types.
  template <typename MhloOpTy>
  static Value mapOpWithArgTypes(MhloOpTy op, ArrayRef<Type> resultTypes,
                                 ArrayRef<Type> argTypes, ValueRange args,
                                 OpBuilder* b) {
    static_assert(!std::is_same<MhloOpTy, mhlo::ConvertOp>::value);
    typename MhloOpTy::Adaptor adaptor(args, op->getAttrDictionary(),
                                       op->getPropertiesStorage(),
                                       op->getRegions());
    return mapOpOfType<MhloOpTy>(op.getLoc(), resultTypes, argTypes, adaptor,
                                 b);
  }
  // Overload for mhlo::ConvertOp.
  static Value mapOpWithArgTypes(mhlo::ConvertOp op, ArrayRef<Type> resultTypes,
                                 ArrayRef<Type> argTypes, ValueRange args,
                                 OpBuilder* b) {
    return impl::mapConvertOpToStdScalarOp(op.getLoc(), op.getType(),
                                           resultTypes, argTypes, args, b);
  }

  // Converts mhlo 'op' to linalg and arith ops.
  template <typename MhloOpTy>
  static Value mapOpOfType(Location loc, ArrayRef<Type> resultTypes,
                           ArrayRef<Type> argTypes,
                           typename MhloOpTy::Adaptor adaptor, OpBuilder* b) {
    return impl::mapMhloOpToStdScalarOp<MhloOpTy>(loc, resultTypes, argTypes,
                                                  adaptor, b);
  }

  static Value mapConvertOpToStdScalarOp(Location loc,
                                         ArrayRef<Type> targetTypes,
                                         ArrayRef<Type> resultTypes,
                                         ArrayRef<Type> argTypes,
                                         ValueRange args, OpBuilder* b) {
    return impl::mapConvertOpToStdScalarOp(loc, targetTypes, resultTypes,
                                           argTypes, args, b);
  }
};

}  // namespace mhlo
}  // namespace mlir

#endif  // MLIR_HLO_MHLO_TRANSFORMS_MAP_MHLO_TO_SCALAR_OP_H
