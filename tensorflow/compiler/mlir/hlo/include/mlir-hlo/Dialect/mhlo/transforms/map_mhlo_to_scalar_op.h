/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_DIALECT_MHLO_TRANSFORMS_MAP_MHLO_TO_SCALAR_OP_H
#define MLIR_HLO_DIALECT_MHLO_TRANSFORMS_MAP_MHLO_TO_SCALAR_OP_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"

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
struct MhloToScalarOp<mhlo::CosOp> {
  using FOp = ::mlir::math::CosOp;
  using COp = ::mlir::complex::CosOp;
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
struct MhloToScalarOp<mhlo::MaxOp> {
  using FOp = ::mlir::arith::MaxFOp;
  using IOp = ::mlir::arith::MaxSIOp;
  using UOp = ::mlir::arith::MaxUIOp;
};
template <>
struct MhloToScalarOp<mhlo::MinOp> {
  using FOp = ::mlir::arith::MinFOp;
  using IOp = ::mlir::arith::MinSIOp;
  using UOp = ::mlir::arith::MinUIOp;
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
struct MhloToScalarOp<mhlo::RoundOp> {
  using FOp = ::mlir::math::RoundOp;
};
template <>
struct MhloToScalarOp<mhlo::SubOp> {
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
struct MhloToScalarOp<mhlo::SinOp> {
  using FOp = ::mlir::math::SinOp;
  using COp = ::mlir::complex::SinOp;
};
template <>
struct MhloToScalarOp<mhlo::ShiftLeftOp> {
  using IOp = ::mlir::arith::ShLIOp;
  using UOp = ::mlir::arith::ShLIOp;
};
template <>
struct MhloToScalarOp<mhlo::ShiftRightArithmeticOp> {
  using IOp = ::mlir::arith::ShRSIOp;
  using UOp = ::mlir::arith::ShRSIOp;
};
template <>
struct MhloToScalarOp<mhlo::ShiftRightLogicalOp> {
  using IOp = ::mlir::arith::ShRUIOp;
  using UOp = ::mlir::arith::ShRUIOp;
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
  Value operator()(Location /*loc*/, ArrayRef<Type> /*result_types*/,
                   ArrayRef<Type> /*arg_types*/, ValueRange /*args*/,
                   OpBuilder* /*b*/) {
    return nullptr;
  }
};

template <typename StdScalarOp>
struct MapMhloOpToScalarOpImpl<StdScalarOp> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> /*arg_types*/, ValueRange args,
                   OpBuilder* b) {
    return b->template create<StdScalarOp>(loc, resultTypes, args, mlir::None);
  }
};

template <typename SupportedType, typename StdScalarOp, typename... Args>
struct MapMhloOpToScalarOpImpl<SupportedType, StdScalarOp, Args...> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> argTypes, ValueRange args, OpBuilder* b) {
    Type elementType = getElementTypeOrSelf(argTypes.front());
    if (SupportedType{}(elementType)) {
      return b->template create<StdScalarOp>(loc, resultTypes, args,
                                             mlir::None);
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
  bool operator()(Type t) { return t.isa<IntegerType>(); }
};

struct IsSignedIntegerType {
  bool operator()(Type t) {
    // Pretend that signless is signed. This will change eventually.
    return t.isa<IntegerType>() && !t.isUnsignedInteger() &&
           !t.isSignlessInteger(1);
  }
};

struct IsUnsignedIntegerType {
  bool operator()(Type t) {
    return t.isUnsignedInteger() || t.isSignlessInteger(1);
  }
};

struct IsFloatType {
  bool operator()(Type t) { return t.isa<FloatType>(); }
};

struct IsComplexType {
  bool operator()(Type t) { return t.isa<ComplexType>(); }
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
                                    ArrayRef<Type> argTypes, ValueRange args,
                                    OpBuilder* b) {
  using ScalarIOpOrVoid = typename MapableIf<ScalarIOp, MhloOpTy>::type;
  using ScalarUOpOrVoid = typename MapableIf<ScalarUOp, MhloOpTy>::type;
  using ScalarFOpOrVoid = typename MapableIf<ScalarFOp, MhloOpTy>::type;
  using ScalarCOpOrVoid = typename MapableIf<ScalarCOp, MhloOpTy>::type;
  return MapMhloOpToScalarOpImpl<IsSignedIntegerType, ScalarIOpOrVoid,
                                 IsUnsignedIntegerType, ScalarUOpOrVoid,
                                 IsFloatType, ScalarFOpOrVoid, IsComplexType,
                                 ScalarCOpOrVoid>{}(loc, resultTypes, argTypes,
                                                    args, b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::AbsOp>(Location loc,
                                                 ArrayRef<Type> resultTypes,
                                                 ArrayRef<Type> argTypes,
                                                 ValueRange args,
                                                 OpBuilder* b) {
  Type elementType = getElementTypeOrSelf(argTypes.front());
  if (elementType.isa<FloatType>()) {
    return MapMhloOpToScalarOpImpl<IsFloatType, ::mlir::math::AbsOp>{}(
        loc, resultTypes, argTypes, args, b);
  }
  if (elementType.isa<ComplexType>()) {
    return MapMhloOpToScalarOpImpl<IsComplexType, ::mlir::complex::AbsOp>{}(
        loc, resultTypes, argTypes, args, b);
  }
  if (elementType.isSignlessInteger() || elementType.isSignedInteger()) {
    // lmhlo.abs(x, result) ->  result = select((x > 0), x, sub(0, x))
    Value lhs = args[0];
    Value zeroIntval =
        b->create<arith::ConstantOp>(loc, b->getZeroAttr(lhs.getType()));
    auto lhsGtZero = b->create<ScalarIOp<CompareOp>>(
        loc, arith::CmpIPredicate::sge, lhs, zeroIntval);
    auto negVal = b->create<ScalarIOp<mhlo::SubOp>>(loc, zeroIntval, lhs);
    return b->create<::mlir::arith::SelectOp>(loc, lhsGtZero, lhs, negVal);
  }
  return nullptr;
}

// Return a constant for v of type t, splat if t is a vector type.
inline Value getConstantOrSplat(OpBuilder* b, Location loc, Type t,
                                Attribute v) {
  if (VectorType vecType = t.dyn_cast<VectorType>()) {
    v = SplatElementsAttr::get(vecType, v);
  }
  return b->create<arith::ConstantOp>(loc, t, v);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::CbrtOp>(Location loc,
                                                  ArrayRef<Type> resultTypes,
                                                  ArrayRef<Type> argTypes,
                                                  ValueRange args,
                                                  OpBuilder* b) {
  mhlo::CbrtOp::Adaptor adaptor(args);
  Type elementType = getElementTypeOrSelf(argTypes.front());
  if (auto floatType = elementType.dyn_cast<FloatType>()) {
    // Convert cbrt(x) to copysign(cbrt(abs(x), 1.0 / 3.0), x).
    // This is to allow cbrt using pow while still handling negative numbers. It
    // should match most cbrt intrinsics.
    Value abs = b->create<mlir::math::AbsOp>(loc, adaptor.operand());
    Value third = b->create<arith::ConstantOp>(
        loc, b->getFloatAttr(floatType, 1.0 / 3.0));
    Value pow = b->create<mlir::math::PowFOp>(loc, resultTypes[0], abs, third);
    return b->create<mlir::math::CopySignOp>(loc, floatType, pow,
                                             adaptor.operand());
  }
  return nullptr;
}

template <typename PredicateType>
inline Optional<PredicateType> getCmpPredicate(mhlo::ComparisonDirection,
                                               bool) {
  return llvm::None;
}

template <>
inline Optional<arith::CmpFPredicate> getCmpPredicate<arith::CmpFPredicate>(
    mhlo::ComparisonDirection comparisonDirection, bool isSigned) {
  assert(isSigned && "cannot have an unsigned float!");
  return llvm::StringSwitch<Optional<arith::CmpFPredicate>>(
             stringifyComparisonDirection(comparisonDirection))
      .Case("EQ", arith::CmpFPredicate::OEQ)
      .Case("NE", arith::CmpFPredicate::UNE)
      .Case("GE", arith::CmpFPredicate::OGE)
      .Case("GT", arith::CmpFPredicate::OGT)
      .Case("LE", arith::CmpFPredicate::OLE)
      .Case("LT", arith::CmpFPredicate::OLT)
      .Default(llvm::None);
}

template <>
inline Optional<arith::CmpIPredicate> getCmpPredicate<arith::CmpIPredicate>(
    mhlo::ComparisonDirection comparisonDirection, bool isSigned) {
  return llvm::StringSwitch<Optional<arith::CmpIPredicate>>(
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
      .Default(llvm::None);
}

inline Value mapCompareOpToStdScalarOp(Location loc,
                                       ComparisonDirection comparisonDirection,
                                       ArrayRef<Type> /*result_types*/,
                                       ArrayRef<Type> argTypes, ValueRange args,
                                       OpBuilder* b) {
  const auto& lhs = args[0];
  const auto& rhs = args[1];
  Type elementType = getElementTypeOrSelf(argTypes.front());
  if (elementType.isa<IntegerType>()) {
    bool isUnsigned = IsUnsignedIntegerType{}(elementType);
    Optional<arith::CmpIPredicate> predicate =
        getCmpPredicate<arith::CmpIPredicate>(comparisonDirection, !isUnsigned);
    assert(predicate.hasValue() && "expected valid comparison direction");
    return b->create<ScalarIOp<mhlo::CompareOp>>(loc, predicate.getValue(), lhs,
                                                 rhs);
  }
  if (elementType.isa<FloatType>()) {
    Optional<arith::CmpFPredicate> predicate =
        getCmpPredicate<arith::CmpFPredicate>(comparisonDirection,
                                              /*is_signed=*/true);
    assert(predicate.hasValue() && "expected valid comparison direction");
    return b->create<ScalarFOp<mhlo::CompareOp>>(loc, predicate.getValue(), lhs,
                                                 rhs);
  }
  if (auto complexType = elementType.dyn_cast<ComplexType>()) {
    if (complexType.getElementType().isa<FloatType>()) {
      if (comparisonDirection == ComparisonDirection::EQ) {
        return b->create<complex::EqualOp>(loc, lhs, rhs);
      }
      if (comparisonDirection == ComparisonDirection::NE) {
        return b->create<complex::NotEqualOp>(loc, lhs, rhs);
      }
    }
  }
  return nullptr;
}

inline Value mapReducePrecisionOpToStdScalarOp(
    Location loc, ArrayRef<Type> argTypes, ValueRange args, OpBuilder* builder,
    int destExponentBits, int destMantissaBits) {
  using llvm::APInt;
  mlir::ImplicitLocOpBuilder b(loc, *builder);

  // Integer and float types for casting and constant generation.
  auto floatType =
      argTypes.front().cast<TensorType>().getElementType().cast<FloatType>();
  int64_t nbits = floatType.getWidth();
  auto intType = mlir::IntegerType::get(loc.getContext(), floatType.getWidth());

  Value xAsInt = b.create<arith::BitcastOp>(intType, args[0]);

  // SignificandWidth includes the implicit extra bit.
  auto srcMantissaBits = floatType.getFPMantissaWidth() - 1;
  int srcExponentBits = nbits - 1 - srcMantissaBits;

  // Clear the sign bit, it does not participate in rounding and we will restore
  // it later.
  APInt signBitMask(nbits, 1);
  signBitMask <<= nbits - 1;

  APInt expBitsMask(nbits, 1);
  expBitsMask = ((expBitsMask << srcExponentBits) - 1) << srcMantissaBits;

  if (destMantissaBits < srcMantissaBits) {
    // Last remaining mantissa bit.
    APInt lastMantissaBitMask(nbits, 1);
    lastMantissaBitMask <<= srcMantissaBits - destMantissaBits;

    // Compute rounding bias for round-to-nearest with ties to even.  This is
    // equal to a base value of 0111... plus one bit if the last remaining
    // mantissa bit is 1.
    APInt baseRoundingBias = lastMantissaBitMask.lshr(1) - 1;

    Value mantissaDiff = b.create<arith::ConstantIntOp>(
        srcMantissaBits - destMantissaBits, intType);
    Value highestMantissaMaskVal = b.create<arith::ConstantIntOp>(
        lastMantissaBitMask.getZExtValue(), intType);
    Value baseRoundingBiasVal = b.create<arith::ConstantIntOp>(
        baseRoundingBias.getZExtValue(), intType);
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
    xRounded = b.create<arith::AndIOp>(
        xRounded,
        b.create<arith::ConstantIntOp>(truncationMask.getZExtValue(), intType)
            .getResult());
    xAsInt = xRounded;
  }

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
    Value xExponent = b.create<arith::AndIOp>(
        xAsInt,
        b.create<arith::ConstantIntOp>(expBitsMask.getZExtValue(), intType)
            .getResult());
    Value xOverflows = b.create<arith::CmpIOp>(
        arith::CmpIPredicate::ugt, xExponent,
        b.create<arith::ConstantIntOp>(
             (reducedMaxExponent << srcMantissaBits).getZExtValue(), intType)
            .getResult());
    Value xUnderflows = b.create<arith::CmpIOp>(
        arith::CmpIPredicate::ule, xExponent,
        b.create<arith::ConstantIntOp>(
             (reducedMinExponent << srcMantissaBits).getZExtValue(), intType)
            .getResult());

    // Compute appropriately-signed values of zero and infinity.
    Value xSignedZero = b.create<arith::AndIOp>(
        xAsInt,
        b.create<arith::ConstantIntOp>(signBitMask.getZExtValue(), intType)
            .getResult());
    Value xSignedInf = b.create<arith::OrIOp>(
        xSignedZero,
        b.create<arith::ConstantIntOp>(expBitsMask.getZExtValue(), intType)
            .getResult());

    // Force to zero or infinity if overflow or underflow.  (Note that this
    // truncates all denormal values to zero, rather than rounding them.)
    xAsInt = b.create<arith::SelectOp>(xOverflows, xSignedInf, xAsInt);
    xAsInt = b.create<arith::SelectOp>(xUnderflows, xSignedZero, xAsInt);
  }

  return b.create<arith::BitcastOp>(floatType, xAsInt);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::CopyOp>(
    Location /*loc*/, ArrayRef<Type> /*result_types*/,
    ArrayRef<Type> /*arg_types*/, ValueRange args, OpBuilder* /*b*/) {
  return args.front();
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ComplexOp>(Location loc,
                                                     ArrayRef<Type> resultTypes,
                                                     ArrayRef<Type> argTypes,
                                                     ValueRange args,
                                                     OpBuilder* b) {
  return MapMhloOpToScalarOpImpl<complex::CreateOp>{}(loc, resultTypes,
                                                      argTypes, args, b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::RealOp>(Location loc,
                                                  ArrayRef<Type> resultTypes,
                                                  ArrayRef<Type> argTypes,
                                                  ValueRange args,
                                                  OpBuilder* b) {
  if (!args[0].getType().isa<ComplexType>()) return args[0];
  return MapMhloOpToScalarOpImpl<complex::ReOp>{}(loc, resultTypes, argTypes,
                                                  args, b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ImagOp>(Location loc,
                                                  ArrayRef<Type> resultTypes,
                                                  ArrayRef<Type> argTypes,
                                                  ValueRange args,
                                                  OpBuilder* b) {
  if (!args[0].getType().isa<ComplexType>())
    return b->create<arith::ConstantOp>(loc, b->getZeroAttr(args[0].getType()));
  return MapMhloOpToScalarOpImpl<complex::ImOp>{}(loc, resultTypes, argTypes,
                                                  args, b);
}

// 'target_types' is the unconverted type (signed or unsigned if integer),
// 'result_types' is the converted type (signless if integer).
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
    return b->create<mlir::arith::UIToFPOp>(loc, resultTypes, args, mlir::None);
  }
  if (mlir::arith::SIToFPOp::areCastCompatible(sourceType, targetType)) {
    return b->create<mlir::arith::SIToFPOp>(loc, resultTypes, args, mlir::None);
  }
  if (sourceType.isa<FloatType>() && targetType.isa<FloatType>()) {
    auto src = sourceType.cast<FloatType>();
    auto res = targetType.cast<FloatType>();
    if (src.getWidth() > res.getWidth()) {
      return b->create<mlir::arith::TruncFOp>(loc, resultTypes, args,
                                              mlir::None);
    }
    if (src.getWidth() < res.getWidth()) {
      return b->create<mlir::arith::ExtFOp>(loc, resultTypes, args, mlir::None);
    }
    // There's no direct conversion between different 16 bit floating point
    // types, so go through 32 bit float.
    if (sourceType != targetType) {
      assert(sourceType.isBF16() || targetType.isBF16());
      Value ext = b->create<arith::ExtFOp>(loc, b->getF32Type(), args);
      return b->create<arith::TruncFOp>(loc, resultTypes, ext);
    }
    // No conversion is needed for identical float types.
    return args.front();
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
    if (sourceType.isa<FloatType>()) {
      Value zero = b->create<arith::ConstantOp>(
          loc, b->getZeroAttr(args.front().getType()));
      return b->create<mlir::arith::CmpFOp>(loc, arith::CmpFPredicate::UNE,
                                            args.front(), zero);
    }
  }
  if (sourceType.isa<IntegerType>() && targetType.isa<IntegerType>()) {
    auto src = sourceType.cast<IntegerType>();
    auto res = targetType.cast<IntegerType>();
    if (src.getWidth() > res.getWidth()) {
      return b->create<mlir::arith::TruncIOp>(loc, resultTypes, args,
                                              mlir::None);
    }
    if (src.getWidth() < res.getWidth()) {
      // Special case boolean values, so they get casted to `1` instead of `-1`.
      if (IsUnsignedIntegerType{}(src)) {
        return b->create<mlir::arith::ExtUIOp>(loc, resultTypes, args,
                                               mlir::None);
      }
      return b->create<mlir::arith::ExtSIOp>(loc, resultTypes, args,
                                             mlir::None);
    }
    // No conversion is needed for the same width integers
    return args.front();
  }
  if (targetType.isUnsignedInteger() &&
      mlir::arith::FPToUIOp::areCastCompatible(convertedSourceType,
                                               targetType)) {
    return b->create<mlir::arith::FPToUIOp>(loc, resultTypes, args, mlir::None);
  }
  if (mlir::arith::FPToSIOp::areCastCompatible(convertedSourceType,
                                               targetType)) {
    return b->create<mlir::arith::FPToSIOp>(loc, resultTypes, args, mlir::None);
  }
  if (targetType.isa<ComplexType>()) {
    Type targetElementType = targetType.cast<ComplexType>().getElementType();
    assert(!targetElementType.isa<ComplexType>() &&
           "elements of complex numbers should not be complex");
    Value targetReal;
    Value targetImag;
    if (sourceType.isa<ComplexType>()) {
      // We are converting from complex type: convert real and imaginary parts
      // separately.
      Type sourceElementType = sourceType.cast<ComplexType>().getElementType();
      assert(!sourceElementType.isa<ComplexType>() &&
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
  if (auto sourceComplexType = sourceType.dyn_cast<ComplexType>()) {
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

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::BitcastConvertOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type>, ValueRange args,
    OpBuilder* b) {
  return b->create<mlir::arith::BitcastOp>(loc, resultTypes, args);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::DotOp>(Location loc,
                                                 ArrayRef<Type> resultTypes,
                                                 ArrayRef<Type> argTypes,
                                                 ValueRange args,
                                                 OpBuilder* b) {
  // Dot Op converter from lhlo to affine only accepts float and integer types.
  const auto& lhs = args[0];
  const auto& rhs = args[1];
  const auto& result = args[2];
  Type elementType = lhs.getType();
  if (elementType.isa<FloatType>()) {
    Value floatMul =
        MapMhloOpToScalarOpImpl<IsFloatType, ::mlir::arith::MulFOp>{}(
            loc, resultTypes, argTypes, {lhs, rhs}, b);
    return MapMhloOpToScalarOpImpl<IsFloatType, ::mlir::arith::AddFOp>{}(
        loc, resultTypes, argTypes, {floatMul, result}, b);
  }
  if (elementType.isa<IntegerType>()) {
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
    Location loc, ArrayRef<Type> /*result_types*/, ArrayRef<Type> /*arg_types*/,
    ValueRange args, OpBuilder* b) {
  if (args[0].getType().isa<FloatType>()) {
    auto posInf = APFloat::getInf(
        args[0].getType().cast<FloatType>().getFloatSemantics());
    auto constPosInf = b->create<arith::ConstantOp>(
        loc, b->getFloatAttr(args[0].getType(), posInf));
    Value absX = b->create<::mlir::math::AbsOp>(loc, args[0]);
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
                   ArrayRef<Type> /*result_types*/,
                   ArrayRef<Type> /*arg_types*/, ValueRange /*args*/,
                   OpBuilder* /*b*/) {
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
    if (elementType.isa<SupportedType>()) {
      auto predicate = getCmpPredicate<Predicate>(
          comparisonDirection, !elementType.isUnsignedInteger());
      assert(predicate.hasValue() && "expected valid comparison direction");
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
  if (auto floatType = elementType.dyn_cast<FloatType>()) {
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
inline Value mapMhloOpToStdScalarOp<mhlo::LogisticOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> /*arg_types*/,
    ValueRange args, OpBuilder* b) {
  auto ty = resultTypes.front().cast<FloatType>();
  Value one = b->create<arith::ConstantOp>(loc, b->getFloatAttr(ty, 1.0));
  Value x = args.front();
  Value negX = b->create<arith::NegFOp>(loc, x);
  Value expNegX = b->create<::mlir::math::ExpOp>(loc, negX);
  Value oneAddExpNegX = b->create<arith::AddFOp>(loc, one, expNegX);
  return b->create<arith::DivFOp>(loc, one, oneAddExpNegX);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ClampOp>(Location loc,
                                                   ArrayRef<Type> resultTypes,
                                                   ArrayRef<Type> argTypes,
                                                   ValueRange args,
                                                   OpBuilder* b) {
  mhlo::ClampOp::Adaptor op(args);
  // clamp(lb, x, ub) = min(max(lb, x), ub)
  Value maxLbX = mapMhloOpToStdScalarOp<mhlo::MaxOp>(
      loc, resultTypes, argTypes, {op.min(), op.operand()}, b);
  return mapMhloOpToStdScalarOp<mhlo::MinOp>(loc, resultTypes, argTypes,
                                             {maxLbX, op.max()}, b);
}

template <typename U, typename S>
inline Value makeSafeIntDiv(ImplicitLocOpBuilder& lb, Type originalType,
                            Value lhs, Value rhs, Value returnedOnZero,
                            Value returnedOnSignedOverflow) {
  Type type = lhs.getType();
  auto elementType = getElementTypeOrSelf(type).cast<IntegerType>();
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
  Value minusOne = makeConstant(APInt::getAllOnesValue(elementType.getWidth()));
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
                                                 ValueRange args,
                                                 OpBuilder* b) {
  Type originalType = getElementTypeOrSelf(argTypes.front());
  if (originalType.isa<ComplexType, FloatType>()) {
    return MapMhloOpToScalarOpImpl<IsFloatType, arith::DivFOp, IsComplexType,
                                   complex::DivOp>{}(loc, resultTypes, argTypes,
                                                     args, b);
  }

  // Integer division overflow behavior:
  //
  // X / 0 == -1
  // INT_SMIN /s -1 = INT_SMIN
  ImplicitLocOpBuilder lb(loc, *b);
  Type type = args.front().getType();
  auto elementType = getElementTypeOrSelf(type).cast<IntegerType>();
  auto makeConstant = [&](const APInt& i) {
    return getConstantOrSplat(&lb, lb.getLoc(), type,
                              lb.getIntegerAttr(elementType, i));
  };
  Value minusOne = makeConstant(APInt::getAllOnesValue(elementType.getWidth()));
  Value smin = makeConstant(APInt::getSignedMinValue(elementType.getWidth()));
  return makeSafeIntDiv<arith::DivUIOp, arith::DivSIOp>(
      lb, originalType, args[0], args[1], /*returnedOnZero=*/minusOne,
      /*returnedOnSignedOverflow=*/smin);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::RemOp>(Location loc,
                                                 ArrayRef<Type> resultTypes,
                                                 ArrayRef<Type> argTypes,
                                                 ValueRange args,
                                                 OpBuilder* b) {
  Type originalType = getElementTypeOrSelf(argTypes.front());
  if (originalType.isa<ComplexType, FloatType>()) {
    return MapMhloOpToScalarOpImpl<IsFloatType, arith::RemFOp>{}(
        loc, resultTypes, argTypes, args, b);
  }

  // Integer remainder overflow behavior:
  //
  // X % 0 == X
  // INT_SMIN %s -1 = 0
  ImplicitLocOpBuilder lb(loc, *b);
  Type type = args.front().getType();
  Value zero = lb.create<arith::ConstantOp>(lb.getZeroAttr(type));
  return makeSafeIntDiv<arith::RemUIOp, arith::RemSIOp>(
      lb, originalType, args[0], args[1], /*returnedOnZero=*/args[0],
      /*returnedOnSignedOverflow=*/zero);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::NegOp>(Location loc,
                                                 ArrayRef<Type> resultTypes,
                                                 ArrayRef<Type> argTypes,
                                                 ValueRange args,
                                                 OpBuilder* b) {
  Type elementType = getElementTypeOrSelf(args.front().getType());
  if (elementType.isa<ComplexType, FloatType>()) {
    return MapMhloOpToScalarOpImpl<IsFloatType, ::mlir::arith::NegFOp,
                                   IsComplexType, ::mlir::complex::NegOp>{}(
        loc, resultTypes, argTypes, args, b);
  }
  if (elementType.isa<IntegerType>()) {
    // lmhlo.neg(x, result) -> result = sub(0, x)
    Value lhs = args[0];
    Value zeroIntval =
        b->create<arith::ConstantOp>(loc, b->getZeroAttr(lhs.getType()));
    return b->create<ScalarIOp<mhlo::SubOp>>(loc, zeroIntval, lhs);
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::NotOp>(
    Location loc, ArrayRef<Type> /*result_types*/, ArrayRef<Type> /*arg_types*/,
    ValueRange args, OpBuilder* b) {
  Type elementType = getElementTypeOrSelf(args.front().getType());
  if (auto integerType = elementType.dyn_cast<IntegerType>()) {
    // lmhlo.not(x) -> x ^ -1
    Value allOnes = getConstantOrSplat(
        b, loc, args[0].getType(),
        b->getIntegerAttr(integerType,
                          APInt::getAllOnesValue(integerType.getWidth())));
    return b->create<::mlir::arith::XOrIOp>(loc, allOnes, args[0]);
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::PowOp>(Location loc,
                                                 ArrayRef<Type> resultTypes,
                                                 ArrayRef<Type> argTypes,
                                                 ValueRange args,
                                                 OpBuilder* b) {
  mhlo::PowOp::Adaptor adaptor(args);
  auto lb = ImplicitLocOpBuilder(loc, *b);
  // Floating point can use std::powf
  auto resultType = resultTypes.front();
  if (resultType.isa<ComplexType, FloatType>()) {
    return MapMhloOpToScalarOpImpl<IsFloatType, math::PowFOp, IsComplexType,
                                   complex::PowOp>{}(loc, resultTypes, argTypes,
                                                     args, b);
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
  auto originalBase = adaptor.lhs();
  auto originalExponent = adaptor.rhs();

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
      arith::CmpIPredicate::eq, lb.create<arith::RemSIOp>(adaptor.rhs(), two),
      zero);
  Value rhsIsNegative =
      lb.create<arith::CmpIOp>(arith::CmpIPredicate::slt, adaptor.rhs(), zero);
  Value lhsIsOne =
      lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq, adaptor.lhs(), one);
  Value lhsIsNegOne =
      lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq, adaptor.lhs(), negOne);

  // The accum is correct when the rhs is non-negative. When rhs is
  // negative, we return 0 for integer, with the exception of lhs values of 1
  // and -1 which have integer results for negative exponents. Specifically, the
  // calulation is the following:
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
inline Value mapMhloOpToStdScalarOp<mhlo::SelectOp>(Location loc,
                                                    ArrayRef<Type> resultTypes,
                                                    ArrayRef<Type> argTypes,
                                                    ValueRange args,
                                                    OpBuilder* b) {
  return MapMhloOpToScalarOpImpl<::mlir::arith::SelectOp>{}(loc, resultTypes,
                                                            argTypes, args, b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::SignOp>(Location loc,
                                                  ArrayRef<Type> resultTypes,
                                                  ArrayRef<Type> /*arg_types*/,
                                                  ValueRange args,
                                                  OpBuilder* b) {
  Type elementType = getElementTypeOrSelf(args.front().getType());
  if (auto floatType = elementType.dyn_cast<FloatType>()) {
    Value zero =
        b->create<arith::ConstantOp>(loc, b->getZeroAttr(args[0].getType()));
    Value ne0I1 = b->create<::mlir::arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE, args[0], zero);
    Value ne0Float =
        b->create<::mlir::arith::UIToFPOp>(loc, zero.getType(), ne0I1);
    Value copySign = b->create<::mlir::math::CopySignOp>(loc, resultTypes,
                                                         ne0Float, args[0]);
    auto isNan = b->create<::mlir::arith::CmpFOp>(
        loc, arith::CmpFPredicate::UNO, args[0], args[0]);
    return b->create<::mlir::arith::SelectOp>(loc, isNan, args[0], copySign);
  }
  if (auto integerType = elementType.dyn_cast<IntegerType>()) {
    // sign(x) = x == 0 ? 0 : ((x s>> 31) | 1)
    Value zero =
        b->create<arith::ConstantOp>(loc, b->getZeroAttr(args[0].getType()));
    Value bitwidthMinusOne = getConstantOrSplat(
        b, loc, args[0].getType(),
        b->getIntegerAttr(integerType, integerType.getWidth() - 1));
    Value one = getConstantOrSplat(b, loc, args[0].getType(),
                                   b->getIntegerAttr(integerType, 1));
    Value cmp = b->create<::mlir::arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 args[0], zero);
    Value ashr =
        b->create<::mlir::arith::ShRSIOp>(loc, args[0], bitwidthMinusOne);
    Value orOp = b->create<::mlir::arith::OrIOp>(loc, ashr, one);
    return b->create<::mlir::arith::SelectOp>(loc, cmp, zero, orOp);
  }
  if (elementType.isa<ComplexType>()) {
    return b->create<::mlir::complex::SignOp>(loc, elementType, args.front());
  }
  return nullptr;
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
  // be converted, 'arg_types' are their original types.
  template <typename MhloOpTy>
  static Value mapOpWithArgTypes(MhloOpTy op, ArrayRef<Type> resultTypes,
                                 ArrayRef<Type> argTypes, ValueRange args,
                                 OpBuilder* b) {
    static_assert(!std::is_same<MhloOpTy, mhlo::ConvertOp>::value);
    return mapOpOfType<MhloOpTy>(op.getLoc(), resultTypes, argTypes, args, b);
  }
  // Overload for mhlo::ReducePrecisionOp.
  static Value mapOpWithArgTypes(mhlo::ReducePrecisionOp op,
                                 ArrayRef<Type> result_types,
                                 ArrayRef<Type> argTypes, ValueRange args,
                                 OpBuilder* b) {
    return impl::mapReducePrecisionOpToStdScalarOp(
        op.getLoc(), argTypes, args, b, op.exponent_bits(), op.mantissa_bits());
  }
  // Overload for mhlo::CompareOp.
  static Value mapOpWithArgTypes(mhlo::CompareOp op, ArrayRef<Type> resultTypes,
                                 ArrayRef<Type> argTypes, ValueRange args,
                                 OpBuilder* b) {
    auto comparisonDirection = op.comparison_direction();
    return impl::mapCompareOpToStdScalarOp(op.getLoc(), comparisonDirection,
                                           resultTypes, argTypes, args, b);
  }
  // Overload for mhlo::ConvertOp.
  static Value mapOpWithArgTypes(mhlo::ConvertOp op, ArrayRef<Type> resultTypes,
                                 ArrayRef<Type> argTypes, ValueRange args,
                                 OpBuilder* b) {
    return impl::mapConvertOpToStdScalarOp(op.getLoc(), op.getType(),
                                           resultTypes, argTypes, args, b);
  }

  // Converts mhlo 'op' (except mhlo::CompareOp) to linalg and arith ops.
  template <typename MhloOpTy>
  static Value mapOpOfType(Location loc, ArrayRef<Type> resultTypes,
                           ArrayRef<Type> argTypes, ValueRange args,
                           OpBuilder* b) {
    static_assert(!std::is_same<MhloOpTy, mhlo::CompareOp>::value, "invalid");
    if (std::is_same<MhloOpTy, mhlo::ConvertOp>::value) {
      // Note: this assumes that the caller is passing result/arg types with
      // appropriate signedness.
      return impl::mapConvertOpToStdScalarOp(loc, resultTypes, resultTypes,
                                             argTypes, args, b);
    }
    return impl::mapMhloOpToStdScalarOp<MhloOpTy>(loc, resultTypes, argTypes,
                                                  args, b);
  }
};

}  // namespace mhlo
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_MHLO_TRANSFORMS_MAP_MHLO_TO_SCALAR_OP_H
