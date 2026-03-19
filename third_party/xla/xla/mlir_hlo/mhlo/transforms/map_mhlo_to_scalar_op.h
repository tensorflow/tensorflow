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

#include <cstdint>
#include <optional>
#include <type_traits>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/transformation_helpers.h"
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
struct MhloToScalarOp<mhlo::AcosOp> {
  using FOp = ::mlir::math::AcosOp;
};

template <>
struct MhloToScalarOp<mhlo::AcoshOp> {
  using FOp = ::mlir::math::AcoshOp;
};
template <>
struct MhloToScalarOp<mhlo::AsinOp> {
  using FOp = ::mlir::math::AsinOp;
};
template <>
struct MhloToScalarOp<mhlo::AsinhOp> {
  using FOp = ::mlir::math::AsinhOp;
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
struct MhloToScalarOp<mhlo::CoshOp> {
  using FOp = ::mlir::math::CoshOp;
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
struct MhloToScalarOp<mhlo::SinhOp> {
  using FOp = ::mlir::math::SinhOp;
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
struct MhloToScalarOp<mhlo::AtanhOp> {
  using FOp = ::mlir::math::AtanhOp;
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
                   ArrayRef<NamedAttribute> /*attributes*/, OpBuilder* /*b*/) {
    return nullptr;
  }
};

template <typename StdScalarOp>
struct MapMhloOpToScalarOpImpl<StdScalarOp> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> /*argTypes*/, ValueRange args,
                   ArrayRef<NamedAttribute> attributes, OpBuilder* b) {
    return StdScalarOp::create(*b, loc, resultTypes, args, attributes);
  }
};

template <typename SupportedType, typename StdScalarOp, typename... Args>
struct MapMhloOpToScalarOpImpl<SupportedType, StdScalarOp, Args...> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> argTypes, ValueRange args,
                   ArrayRef<NamedAttribute> attributes, OpBuilder* b) {
    Type elementType = getElementTypeOrSelf(argTypes.front());
    if (SupportedType{}(elementType)) {
      return StdScalarOp::create(*b, loc, resultTypes, args, attributes);
    }
    return MapMhloOpToScalarOpImpl<Args...>{}(loc, resultTypes, argTypes, args,
                                              attributes, b);
  }
};

template <typename SupportedType, typename... Args>
struct MapMhloOpToScalarOpImpl<SupportedType, void, Args...> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> argTypes, ValueRange args,
                   ArrayRef<NamedAttribute> attributes, OpBuilder* b) {
    return MapMhloOpToScalarOpImpl<Args...>{}(loc, resultTypes, argTypes, args,
                                              attributes, b);
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
                                    ArrayRef<NamedAttribute> attributes,
                                    OpBuilder* b) {
  using ScalarIOpOrVoid = typename MapableIf<ScalarIOp, MhloOpTy>::type;
  using ScalarUOpOrVoid = typename MapableIf<ScalarUOp, MhloOpTy>::type;
  using ScalarFOpOrVoid = typename MapableIf<ScalarFOp, MhloOpTy>::type;
  using ScalarCOpOrVoid = typename MapableIf<ScalarCOp, MhloOpTy>::type;
  return MapMhloOpToScalarOpImpl<IsSignedIntegerType, ScalarIOpOrVoid,
                                 IsUnsignedIntegerType, ScalarUOpOrVoid,
                                 IsFloatType, ScalarFOpOrVoid, IsComplexType,
                                 ScalarCOpOrVoid>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::AbsOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::AbsOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  Type elementType = getElementTypeOrSelf(argTypes.front());
  if (mlir::isa<FloatType>(elementType)) {
    return MapMhloOpToScalarOpImpl<IsFloatType, ::mlir::math::AbsFOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
  }
  if (mlir::isa<ComplexType>(elementType)) {
    return MapMhloOpToScalarOpImpl<IsComplexType, ::mlir::complex::AbsOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
  }
  if (elementType.isSignlessInteger() || elementType.isSignedInteger()) {
    // lmhlo.abs(x, result) ->  result = select((x > 0), x, sub(0, x))
    Value lhs = adaptor.getOperand();
    Value zeroIntval =
        arith::ConstantOp::create(*b, loc, b->getZeroAttr(lhs.getType()));
    auto lhsGtZero = ScalarIOp<CompareOp>::create(
        *b, loc, arith::CmpIPredicate::sge, lhs, zeroIntval);
    auto negVal = ScalarIOp<mhlo::SubtractOp>::create(*b, loc, zeroIntval, lhs);
    return ::mlir::arith::SelectOp::create(*b, loc, lhsGtZero, lhs, negVal);
  }
  return nullptr;
}

// Return a constant for v of type t, splat if t is a vector type.
inline Value getConstantOrSplat(OpBuilder* b, Location loc, Type t,
                                Attribute v) {
  if (ShapedType shapedType = mlir::dyn_cast<ShapedType>(t)) {
    v = SplatElementsAttr::get(shapedType, v);
  }
  return arith::ConstantOp::create(*b, loc, t, cast<TypedAttr>(v));
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
      return complex::EqualOp::create(*b, loc, lhs, rhs);
    }
    if (comparisonDirection == ComparisonDirection::NE) {
      return complex::NotEqualOp::create(*b, loc, lhs, rhs);
    }

    // Perform a lexicographical comparison for the (real, imaginary) pair.
    Type complexFloatTy = complexType.getElementType();

    Value lhsReal = complex::ReOp::create(*b, loc, complexFloatTy, lhs);
    Value rhsReal = complex::ReOp::create(*b, loc, complexFloatTy, rhs);

    Value lhsImag = complex::ImOp::create(*b, loc, complexFloatTy, lhs);
    Value rhsImag = complex::ImOp::create(*b, loc, complexFloatTy, rhs);

    auto predicate = getCmpPredicate<arith::CmpFPredicate>(comparisonDirection,
                                                           /*is_signed=*/true);
    assert(predicate.has_value() && "expected valid comparison direction");

    //   if (lhsReal == rhsReal && lhsImag `predicate` rhsImag ||
    //       lhsReal `predicate` rhsReal)
    Value realsAreEq = arith::CmpFOp::create(*b, loc, arith::CmpFPredicate::OEQ,
                                             lhsReal, rhsReal);
    Value imagsAreOrdered =
        arith::CmpFOp::create(*b, loc, *predicate, lhsImag, rhsImag);
    Value realsAreOrdered =
        arith::CmpFOp::create(*b, loc, *predicate, lhsReal, rhsReal);

    Value orLhs = arith::AndIOp::create(*b, loc, realsAreEq, imagsAreOrdered);
    return arith::OrIOp::create(*b, loc, orLhs, realsAreOrdered);
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::CompareOp>(
    Location loc, ArrayRef<Type> /*resultTypes*/, ArrayRef<Type> argTypes,
    mhlo::CompareOp::Adaptor adaptor, ArrayRef<NamedAttribute> /*attributes*/,
    OpBuilder* b) {
  ComparisonDirection comparisonDirection = adaptor.getComparisonDirection();
  const auto& lhs = adaptor.getLhs();
  const auto& rhs = adaptor.getRhs();
  Type elementType = getElementTypeOrSelf(argTypes.front());
  if (mlir::isa<IntegerType>(elementType)) {
    bool isUnsigned = IsUnsignedIntegerType{}(elementType);
    std::optional<arith::CmpIPredicate> predicate =
        getCmpPredicate<arith::CmpIPredicate>(comparisonDirection, !isUnsigned);
    assert(predicate.has_value() && "expected valid comparison direction");
    return ScalarIOp<mhlo::CompareOp>::create(*b, loc, predicate.value(), lhs,
                                              rhs);
  }
  if (auto floatType = mlir::dyn_cast<FloatType>(elementType)) {
    if (adaptor.getCompareType() &&
        *adaptor.getCompareType() == mhlo::ComparisonType::TOTALORDER) {
      // The semantics of totalorder fp compare are
      // -NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN
      auto intType = b->getIntegerType(floatType.getWidth());
      auto zero =
          arith::ConstantOp::create(*b, loc, intType, b->getZeroAttr(intType));
      auto max = arith::ConstantOp::create(
          *b, loc, intType,
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
        auto x = arith::BitcastOp::create(*b, loc, intType, v);
        auto cmp =
            arith::CmpIOp::create(*b, loc, arith::CmpIPredicate::slt, x, zero);
        auto sub = arith::SubIOp::create(*b, loc, max, x);
        return arith::SelectOp::create(*b, loc, cmp, sub, x);
      };
      auto lhsInt = toIntegral(lhs);
      auto rhsInt = toIntegral(rhs);
      auto predicate =
          getCmpPredicate<arith::CmpIPredicate>(comparisonDirection,
                                                /*is_signed=*/true);
      assert(predicate.has_value() && "expected valid comparison direction");
      return arith::CmpIOp::create(*b, loc, *predicate, lhsInt, rhsInt);
    }
    std::optional<arith::CmpFPredicate> predicate =
        getCmpPredicate<arith::CmpFPredicate>(comparisonDirection,
                                              /*is_signed=*/true);
    assert(predicate.has_value() && "expected valid comparison direction");
    return ScalarFOp<mhlo::CompareOp>::create(*b, loc, predicate.value(), lhs,
                                              rhs);
  }
  if (auto complexType = mlir::dyn_cast<ComplexType>(elementType))
    return cmpComplex(loc, lhs, rhs, comparisonDirection, b);
  return nullptr;
}

static bool HasDefaultMantissaBits(Type type, uint32_t mantissa_bits) {
  if (auto float_ty = mlir::dyn_cast<FloatType>(type)) {
    return float_ty.getFPMantissaWidth() == mantissa_bits;
  }
  return false;
}

static bool HasDefaultExponentBits(Type type, uint32_t exponent_bits) {
  if (auto float_ty = mlir::dyn_cast<FloatType>(type)) {
    return float_ty.getWidth() - float_ty.getFPMantissaWidth() - 1 ==
           exponent_bits;
  }
  return false;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ReducePrecisionOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> /*argTypes*/,
    mhlo::ReducePrecisionOp::Adaptor adaptor,
    ArrayRef<NamedAttribute> /*attributes*/, OpBuilder* builder) {
  // TODO(b/373787166): This should actually be a folder, but JAX is adding
  // no-op ReducePrecision ops to workaround an issue with some simplifications
  // allowed with the xla_allow_excess_precision flag. We would already fold
  // these ops away before they reach HLO. Folding them away at emission time
  // keeps the workaround intact.
  if (HasDefaultExponentBits(resultTypes[0], adaptor.getExponentBits()) &&
      HasDefaultMantissaBits(resultTypes[0], adaptor.getMantissaBits())) {
    return adaptor.getOperand();
  }
  return reducePrecision<arith::BitcastOp>(loc, adaptor.getOperand(),
                                           adaptor.getExponentBits(),
                                           adaptor.getMantissaBits(), builder);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::CopyOp>(
    Location /*loc*/, ArrayRef<Type> /*ResultTypes*/,
    ArrayRef<Type> /*argTypes*/, mhlo::CopyOp::Adaptor adaptor,
    ArrayRef<NamedAttribute> /*attributes*/, OpBuilder* /*b*/) {
  return adaptor.getOperands().front();
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ComplexOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::ComplexOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  return MapMhloOpToScalarOpImpl<complex::CreateOp>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::MaxOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::MaxOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  ValueRange operands = adaptor.getOperands();
  Value lhs = operands.front();
  Type complexTy = lhs.getType();

  if (!mlir::isa<ComplexType>(complexTy))
    return MapMhloOpToScalarOpImpl<IsFloatType, arith::MaximumFOp,
                                   IsSignedIntegerType, arith::MaxSIOp,
                                   IsUnsignedIntegerType, arith::MaxUIOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);

  assert(resultTypes.size() == 1 && "MaxOp should return a single result");
  assert(operands.size() == 2 && "MaxOp should take exactly two arguments");

  Value rhs = operands.back();
  // 'max' performs a lexicographical comparison for the (real, imaginary) pair.
  Value cond = cmpComplex(loc, lhs, rhs, ComparisonDirection::GE, b);

  return arith::SelectOp::create(*b, loc, cond, lhs, rhs).getResult();
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::MinOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::MinOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  ValueRange operands = adaptor.getOperands();
  Value lhs = operands.front();
  Type complexTy = lhs.getType();

  if (!mlir::isa<ComplexType>(complexTy))
    return MapMhloOpToScalarOpImpl<IsFloatType, arith::MinimumFOp,
                                   IsSignedIntegerType, arith::MinSIOp,
                                   IsUnsignedIntegerType, arith::MinUIOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);

  assert(resultTypes.size() == 1 && "MinOp should return a single result");
  assert(operands.size() == 2 && "MinOp should take exactly two arguments");

  Value rhs = operands.back();
  // 'min' performs a lexicographical comparison for the (real, imaginary) pair.
  Value cond = cmpComplex(loc, lhs, rhs, ComparisonDirection::LE, b);

  return arith::SelectOp::create(*b, loc, cond, lhs, rhs).getResult();
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::RealOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::RealOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  if (!mlir::isa<ComplexType>(adaptor.getOperand().getType()))
    return adaptor.getOperand();
  return MapMhloOpToScalarOpImpl<complex::ReOp>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ImagOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::ImagOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  if (!mlir::isa<ComplexType>(adaptor.getOperand().getType()))
    return arith::ConstantOp::create(
        *b, loc, b->getZeroAttr(adaptor.getOperand().getType()));
  return MapMhloOpToScalarOpImpl<complex::ImOp>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
}

// 'target_types' is the unconverted type (signed or unsigned if integer),
// 'ResultTypes' is the converted type (signless if integer).
inline Value mapConvertOpToStdScalarOp(Location loc, ArrayRef<Type> targetTypes,
                                       ArrayRef<Type> resultTypes,
                                       ArrayRef<Type> argTypes, ValueRange args,
                                       ArrayRef<NamedAttribute> attributes,
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
    return mlir::arith::UIToFPOp::create(*b, loc, resultTypes, args,
                                         attributes);
  }
  if (mlir::arith::SIToFPOp::areCastCompatible(sourceType, targetType)) {
    return mlir::arith::SIToFPOp::create(*b, loc, resultTypes, args,
                                         attributes);
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
      src = mlir::arith::ExtFOp::create(*b, loc, sourceType, src).getResult();
    }
    assert(sourceType.getIntOrFloatBitWidth() != dst.getWidth());

    if (sourceType.getIntOrFloatBitWidth() > dst.getWidth()) {
      return mlir::arith::TruncFOp::create(*b, loc, resultTypes, src,
                                           attributes);
    }
    return mlir::arith::ExtFOp::create(*b, loc, resultTypes, src, attributes);
  }
  if (targetType.isInteger(/*width=*/1)) {
    // When casting to bool, we need to compare whether the value is equal to
    // zero.
    if (sourceType.isSignlessInteger() || sourceType.isUnsignedInteger()) {
      Value zeroIntval = arith::ConstantOp::create(
          *b, loc, b->getZeroAttr(args.front().getType()));
      return mlir::arith::CmpIOp::create(*b, loc, arith::CmpIPredicate::ne,
                                         args.front(), zeroIntval);
    }
    if (mlir::isa<FloatType>(sourceType)) {
      Value zero = arith::ConstantOp::create(
          *b, loc, b->getZeroAttr(args.front().getType()));
      return mlir::arith::CmpFOp::create(*b, loc, arith::CmpFPredicate::UNE,
                                         args.front(), zero);
    }
  }
  if (mlir::isa<IntegerType>(sourceType) &&
      mlir::isa<IntegerType>(targetType)) {
    auto src = mlir::cast<IntegerType>(sourceType);
    auto res = mlir::cast<IntegerType>(targetType);
    if (src.getWidth() > res.getWidth()) {
      return mlir::arith::TruncIOp::create(*b, loc, resultTypes, args,
                                           attributes);
    }
    if (src.getWidth() < res.getWidth()) {
      // Special case boolean values, so they get casted to `1` instead of `-1`.
      if (IsUnsignedIntegerType{}(src)) {
        return mlir::arith::ExtUIOp::create(*b, loc, resultTypes, args,
                                            attributes);
      }
      return mlir::arith::ExtSIOp::create(*b, loc, resultTypes, args,
                                          attributes);
    }
    // No conversion is needed for the same width integers
    return args.front();
  }
  if (targetType.isUnsignedInteger() &&
      mlir::arith::FPToUIOp::areCastCompatible(convertedSourceType,
                                               targetType)) {
    return mlir::arith::FPToUIOp::create(*b, loc, resultTypes, args,
                                         attributes);
  }
  if (mlir::arith::FPToSIOp::areCastCompatible(convertedSourceType,
                                               targetType)) {
    return mlir::arith::FPToSIOp::create(*b, loc, resultTypes, args,
                                         attributes);
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
          mlir::complex::ReOp::create(*b, loc, sourceElementType, args.front());
      targetReal = mapConvertOpToStdScalarOp(
          loc, targetElementType, targetElementType, sourceElementType,
          sourceReal, attributes, b);
      Value sourceImag =
          mlir::complex::ImOp::create(*b, loc, sourceElementType, args.front());
      targetImag = mapConvertOpToStdScalarOp(
          loc, targetElementType, targetElementType, sourceElementType,
          sourceImag, attributes, b);
    } else {
      // We are converting from real (float, integer, etc.) type, convert the
      // real part and set the imaginary part to 0.
      targetReal =
          mapConvertOpToStdScalarOp(loc, targetElementType, targetElementType,
                                    argTypes, args, attributes, b);
      targetImag = mlir::arith::ConstantOp::create(
          *b, loc, b->getFloatAttr(targetElementType, 0.0));
    }
    return mlir::complex::CreateOp::create(*b, loc, targetType, targetReal,
                                           targetImag);
  }
  if (auto sourceComplexType = mlir::dyn_cast<ComplexType>(sourceType)) {
    auto sourceElementType = sourceComplexType.getElementType();
    // When converting from complex to a non-complex type, we take just the real
    // part of the complex number.
    Value sourceReal =
        mlir::complex::ReOp::create(*b, loc, sourceElementType, args.front());
    return mapConvertOpToStdScalarOp(loc, targetTypes, resultTypes,
                                     sourceElementType, sourceReal, attributes,
                                     b);
  }
  return nullptr;
}

/// Lower bitcast operations where the input and resulting type are the same
/// bitwidth, thus implying that the operation is fully defined by parallel
/// loops and scalar operations without any shape dimension changes.
template <>
inline Value mapMhloOpToStdScalarOp<mhlo::BitcastConvertOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::BitcastConvertOp::Adaptor adaptor,
    ArrayRef<NamedAttribute> attributes, OpBuilder* b) {
  Type argType = getElementTypeOrSelf(argTypes.front());
  Type resultType = getElementTypeOrSelf(resultTypes.front());

  if (resultType.getIntOrFloatBitWidth() != argType.getIntOrFloatBitWidth())
    return nullptr;

  return mlir::arith::BitcastOp::create(*b, loc, resultTypes,
                                        adaptor.getOperands(), attributes);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::DotOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::DotOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  // Dot Op converter from lhlo to affine only accepts float and integer types.
  const auto& lhs = adaptor.getOperands()[0];
  const auto& rhs = adaptor.getOperands()[1];
  const auto& result = adaptor.getOperands()[2];
  Type elementType = lhs.getType();
  if (mlir::isa<FloatType>(elementType)) {
    Value floatMul =
        MapMhloOpToScalarOpImpl<IsFloatType, ::mlir::arith::MulFOp>{}(
            loc, resultTypes, argTypes, {lhs, rhs}, attributes, b);
    return MapMhloOpToScalarOpImpl<IsFloatType, ::mlir::arith::AddFOp>{}(
        loc, resultTypes, argTypes, {floatMul, result}, attributes, b);
  }
  if (mlir::isa<IntegerType>(elementType)) {
    Value intMul =
        MapMhloOpToScalarOpImpl<IsAnyIntegerType, ::mlir::arith::MulIOp>{}(
            loc, resultTypes, argTypes, {lhs, rhs}, attributes, b);
    return MapMhloOpToScalarOpImpl<IsAnyIntegerType, ::mlir::arith::AddIOp>{}(
        loc, resultTypes, argTypes, {intMul, result}, attributes, b);
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::IsFiniteOp>(
    Location loc, ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
    mhlo::IsFiniteOp::Adaptor adaptor, ArrayRef<NamedAttribute> /*attributes*/,
    OpBuilder* b) {
  if (mlir::isa<FloatType>(adaptor.getX().getType())) {
    auto posInf = APFloat::getInf(
        mlir::cast<FloatType>(adaptor.getX().getType()).getFloatSemantics());
    auto constPosInf = arith::ConstantOp::create(
        *b, loc, b->getFloatAttr(adaptor.getX().getType(), posInf));
    Value absX = ::mlir::math::AbsFOp::create(*b, loc, adaptor.getX());
    return ::mlir::arith::CmpFOp::create(*b, loc, arith::CmpFPredicate::ONE,
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
      auto cmp =
          StdCompareOp::create(*b, loc, predicate.getValue(), args[0], args[1]);
      return ::mlir::arith::SelectOp::create(*b, loc, cmp, args[0], args[1]);
    }
    return CompareSelectOpToStdScalarOp<Args...>::map(
        loc, comparisonDirection, resultTypes, argTypes, args, b);
  }
};

inline Value mhloAlwaysPropagateNaN(Value v, ValueRange args, Location loc,
                                    OpBuilder* b) {
  Type elementType = getElementTypeOrSelf(args.front().getType());
  if (auto floatType = mlir::dyn_cast<FloatType>(elementType)) {
    Value isnan = mlir::arith::CmpFOp::create(
        *b, loc, arith::CmpFPredicate::UNO, args[0], args[1]);

    auto nanApfloat = APFloat::getQNaN(floatType.getFloatSemantics());
    Value nan = getConstantOrSplat(b, loc, args[0].getType(),
                                   b->getFloatAttr(floatType, nanApfloat));
    v = mlir::arith::SelectOp::create(*b, loc, isnan, nan, v);
  }
  return v;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ClampOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::ClampOp::Adaptor op, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  // clamp(lb, x, ub) = min(max(lb, x), ub)
  Value maxLbX = mapMhloOpToStdScalarOp<mhlo::MaxOp>(
      loc, resultTypes, argTypes, ValueRange{op.getMin(), op.getOperand()},
      attributes, b);
  return mapMhloOpToStdScalarOp<mhlo::MinOp>(loc, resultTypes, argTypes,
                                             ValueRange{maxLbX, op.getMax()},
                                             attributes, b);
}

template <typename U, typename S>
inline Value makeSafeIntDiv(ImplicitLocOpBuilder& lb, bool isUnsigned,
                            Value lhs, Value rhs, Value returnedOnZero,
                            Value returnedOnSignedOverflow) {
  Type type = lhs.getType();
  auto elementType = mlir::cast<IntegerType>(getElementTypeOrSelf(type));
  Value zero = arith::ConstantOp::create(lb, lb.getZeroAttr(type));
  auto makeConstant = [&](const APInt& i) {
    return getConstantOrSplat(&lb, lb.getLoc(), type,
                              lb.getIntegerAttr(elementType, i));
  };
  Value one = makeConstant(APInt(elementType.getWidth(), 1));
  Value rhsIsZero =
      arith::CmpIOp::create(lb, arith::CmpIPredicate::eq, rhs, zero);

  // For unsigned just set the divisor to 1 when it would be 0.
  if (isUnsigned) {
    Value safeRhs = arith::SelectOp::create(lb, rhsIsZero, one, rhs);
    Value safeDiv = U::create(lb, lhs, safeRhs);
    return arith::SelectOp::create(lb, rhsIsZero, returnedOnZero, safeDiv);
  }

  // For signed also check for INT_MIN / -1.
  Value smin = makeConstant(APInt::getSignedMinValue(elementType.getWidth()));
  Value lhsIsSmin =
      arith::CmpIOp::create(lb, arith::CmpIPredicate::eq, lhs, smin);
  Value minusOne = makeConstant(APInt::getAllOnes(elementType.getWidth()));
  Value rhsIsMinusOne =
      arith::CmpIOp::create(lb, arith::CmpIPredicate::eq, rhs, minusOne);
  Value hasIntMinOverflow = arith::AndIOp::create(lb, lhsIsSmin, rhsIsMinusOne);
  Value rhsIsUnsafe = arith::OrIOp::create(lb, rhsIsZero, hasIntMinOverflow);
  Value safeRhs = arith::SelectOp::create(lb, rhsIsUnsafe, one, rhs);
  Value safeDiv = S::create(lb, lhs, safeRhs);
  Value safeSmin = arith::SelectOp::create(lb, hasIntMinOverflow,
                                           returnedOnSignedOverflow, safeDiv);
  return arith::SelectOp::create(lb, rhsIsZero, returnedOnZero, safeSmin);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::DivOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::DivOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  Type originalType = getElementTypeOrSelf(argTypes.front());
  if (mlir::isa<ComplexType, FloatType>(originalType)) {
    return MapMhloOpToScalarOpImpl<IsFloatType, arith::DivFOp, IsComplexType,
                                   complex::DivOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
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
      lb, originalType.isUnsignedInteger(), adaptor.getLhs(), adaptor.getRhs(),
      /*returnedOnZero=*/minusOne,
      /*returnedOnSignedOverflow=*/smin);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::RemOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::RemOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  Type originalType = getElementTypeOrSelf(argTypes.front());
  if (mlir::isa<ComplexType, FloatType>(originalType)) {
    return MapMhloOpToScalarOpImpl<IsFloatType, arith::RemFOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
  }

  // Integer remainder overflow behavior:
  //
  // X % 0 == X
  // INT_SMIN %s -1 = 0
  ImplicitLocOpBuilder lb(loc, *b);
  Type type = adaptor.getLhs().getType();
  Value zero = arith::ConstantOp::create(lb, lb.getZeroAttr(type));
  return makeSafeIntDiv<arith::RemUIOp, arith::RemSIOp>(
      lb, originalType.isUnsignedInteger(), adaptor.getLhs(), adaptor.getRhs(),
      /*returnedOnZero=*/adaptor.getLhs(),
      /*returnedOnSignedOverflow=*/zero);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::NegOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::NegOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  Type elementType = getElementTypeOrSelf(adaptor.getOperand().getType());
  if (mlir::isa<ComplexType, FloatType>(elementType)) {
    return MapMhloOpToScalarOpImpl<IsFloatType, ::mlir::arith::NegFOp,
                                   IsComplexType, ::mlir::complex::NegOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
  }
  if (mlir::isa<IntegerType>(elementType)) {
    // lmhlo.neg(x, result) -> result = sub(0, x)
    Value lhs = adaptor.getOperand();
    Value zeroIntval =
        arith::ConstantOp::create(*b, loc, b->getZeroAttr(lhs.getType()));
    return ScalarIOp<mhlo::SubtractOp>::create(*b, loc, zeroIntval, lhs);
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::NotOp>(
    Location loc, ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
    mhlo::NotOp::Adaptor adaptor, ArrayRef<NamedAttribute> /*attributes*/,
    OpBuilder* b) {
  Type elementType = getElementTypeOrSelf(adaptor.getOperand().getType());
  if (auto integerType = mlir::dyn_cast<IntegerType>(elementType)) {
    // lmhlo.not(x) -> x ^ -1
    Value allOnes = getConstantOrSplat(
        b, loc, adaptor.getOperand().getType(),
        b->getIntegerAttr(integerType,
                          APInt::getAllOnes(integerType.getWidth())));
    return ::mlir::arith::XOrIOp::create(*b, loc, allOnes,
                                         adaptor.getOperand());
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::LogisticOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> /*argTypes*/,
    mhlo::LogisticOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  // 1.0 / (1.0 + exp(-x))
  Value negX = mapMhloOpToStdScalarOp<mhlo::NegOp>(
      loc, resultTypes, resultTypes, {adaptor.getOperand()}, attributes, b);
  Value expNegX = mapMhloOpToStdScalarOp<mhlo::ExpOp>(
      loc, resultTypes, resultTypes, {{negX}}, attributes, b);

  Type type = getElementTypeOrSelf(resultTypes[0]);
  Value oneFloat =
      mlir::isa<ComplexType>(type)
          ? arith::ConstantOp::create(*b, loc, b->getF32FloatAttr(1.0))
          : getConstantOrSplat(b, loc, resultTypes[0],
                               FloatAttr::get(type, 1.0f));
  Value one = mapConvertOpToStdScalarOp(loc, resultTypes, resultTypes,
                                        {oneFloat.getType()}, {{oneFloat}},
                                        attributes, b);
  Value oneAddExprNegX = mapMhloOpToStdScalarOp<mhlo::AddOp>(
      loc, resultTypes, resultTypes, {{expNegX, one}}, attributes, b);
  return mapMhloOpToStdScalarOp<mhlo::DivOp>(
      loc, resultTypes, resultTypes, {{one, oneAddExprNegX}}, attributes, b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::PowOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::PowOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  auto lb = ImplicitLocOpBuilder(loc, *b);
  // TODO: b/315868720 Consider alternate lowerings of mhlo::PowOp with integer
  // operands. Floating point can use std::powf
  auto elementType = getElementTypeOrSelf(resultTypes.front());
  if (mlir::isa<ComplexType, FloatType>(elementType)) {
    return MapMhloOpToScalarOpImpl<IsFloatType, math::PowFOp, IsComplexType,
                                   complex::PowOp>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
  }

  // Exponentiation by squaring:
  // https://en.wikipedia.org/wiki/Exponentiation_by_squaring;
  Value negOne = getConstantOrSplat(&lb, loc, resultTypes[0],
                                    lb.getIntegerAttr(elementType, -1));
  Value zero = getConstantOrSplat(&lb, loc, resultTypes[0],
                                  lb.getIntegerAttr(elementType, 0));
  Value one = getConstantOrSplat(&lb, loc, resultTypes[0],
                                 lb.getIntegerAttr(elementType, 1));
  Value two = getConstantOrSplat(&lb, loc, resultTypes[0],
                                 lb.getIntegerAttr(elementType, 2));
  Value step = arith::ConstantIndexOp::create(lb, 1);
  Value lowerBound = arith::ConstantIndexOp::create(lb, 0);
  // Everything else would overflow for any exponent > 1, as 2^64
  // is the larget possible exponent for a 64-bit integer, and
  // that's 1 << 6.
  Value upperBound = arith::ConstantIndexOp::create(lb, 6);
  auto originalBase = adaptor.getLhs();
  auto originalExponent = adaptor.getRhs();

  Value accum =
      scf::ForOp::create(
          lb, lowerBound, upperBound, step,
          SmallVector<Value>({one, originalBase, originalExponent}),
          [&](OpBuilder& b, Location, Value /*v*/, ValueRange iters) {
            Value accum = iters[0];
            Value base = iters[1];
            Value exponent = iters[2];

            Value condition = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::eq,
                ::mlir::arith::AndIOp::create(b, loc, exponent, one), one);
            Value multiplied =
                ::mlir::arith::MulIOp::create(b, loc, accum, base);
            accum = ::mlir::arith::SelectOp::create(b, loc, condition,
                                                    multiplied, accum);
            base = ::mlir::arith::MulIOp::create(b, loc, base, base);
            exponent = ::mlir::arith::ShRUIOp::create(b, loc, exponent, one);
            scf::YieldOp::create(b, loc,
                                 SmallVector<Value>({accum, base, exponent}));
          })
          .getResult(0);

  Value rhsIsEven = arith::CmpIOp::create(
      lb, arith::CmpIPredicate::eq,
      arith::RemSIOp::create(lb, adaptor.getRhs(), two), zero);
  Value rhsIsNegative = arith::CmpIOp::create(lb, arith::CmpIPredicate::slt,
                                              adaptor.getRhs(), zero);
  Value lhsIsOne = arith::CmpIOp::create(lb, arith::CmpIPredicate::eq,
                                         adaptor.getLhs(), one);
  Value lhsIsNegOne = arith::CmpIOp::create(lb, arith::CmpIPredicate::eq,
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
  Value ifLhsIsOne = ::mlir::arith::SelectOp::create(lb, lhsIsOne, one, zero);
  Value ifLhsIsNegOne = ::mlir::arith::SelectOp::create(
      lb, lhsIsNegOne,
      ::mlir::arith::SelectOp::create(lb, rhsIsEven, one, negOne), ifLhsIsOne);
  return ::mlir::arith::SelectOp::create(lb, rhsIsNegative, ifLhsIsNegOne,
                                         accum);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::SelectOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::SelectOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  return MapMhloOpToScalarOpImpl<::mlir::arith::SelectOp>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::SignOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> /*argTypes*/,
    mhlo::SignOp::Adaptor adaptor, ArrayRef<NamedAttribute> /*attributes*/,
    OpBuilder* b) {
  Value operand = adaptor.getOperand();
  Type elementType = getElementTypeOrSelf(operand.getType());
  if (auto floatType = mlir::dyn_cast<FloatType>(elementType)) {
    Value zero =
        arith::ConstantOp::create(*b, loc, b->getZeroAttr(operand.getType()));
    Value ne0I1 = ::mlir::arith::CmpFOp::create(
        *b, loc, arith::CmpFPredicate::ONE, operand, zero);
    Value ne0Float =
        ::mlir::arith::UIToFPOp::create(*b, loc, zero.getType(), ne0I1);
    Value copySign = ::mlir::math::CopySignOp::create(*b, loc, resultTypes,
                                                      ne0Float, operand);
    auto isNan = ::mlir::arith::CmpFOp::create(
        *b, loc, arith::CmpFPredicate::UNO, operand, operand);
    return ::mlir::arith::SelectOp::create(*b, loc, isNan, operand, copySign);
  }
  if (auto integerType = mlir::dyn_cast<IntegerType>(elementType)) {
    // sign(x) = x == 0 ? 0 : ((x s>> 31) | 1)
    Value zero =
        arith::ConstantOp::create(*b, loc, b->getZeroAttr(operand.getType()));
    Value bitwidthMinusOne = getConstantOrSplat(
        b, loc, operand.getType(),
        b->getIntegerAttr(integerType, integerType.getWidth() - 1));
    Value one = getConstantOrSplat(b, loc, operand.getType(),
                                   b->getIntegerAttr(integerType, 1));
    Value cmp = ::mlir::arith::CmpIOp::create(*b, loc, arith::CmpIPredicate::eq,
                                              operand, zero);
    Value ashr =
        ::mlir::arith::ShRSIOp::create(*b, loc, operand, bitwidthMinusOne);
    Value orOp = ::mlir::arith::OrIOp::create(*b, loc, ashr, one);
    return ::mlir::arith::SelectOp::create(*b, loc, cmp, zero, orOp);
  }
  if (mlir::isa<ComplexType>(elementType)) {
    return ::mlir::complex::SignOp::create(*b, loc, elementType, operand);
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
  Value cmp = mlir::arith::CmpIOp::create(lb, mlir::arith::CmpIPredicate::ugt,
                                          bitWidth, rhs);
  return mlir::arith::SelectOp::create(lb, cmp, shifted, saturated);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ShiftLeftOp>(
    Location loc, ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
    mhlo::ShiftLeftOp::Adaptor adaptor, ArrayRef<NamedAttribute> /*attributes*/,
    OpBuilder* b) {
  ImplicitLocOpBuilder lb(loc, *b);
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  Type type = lhs.getType();

  // "Saturate" if the shift is greater than the bitwidth of the type
  Value zero = arith::ConstantOp::create(lb, lb.getZeroAttr(type));
  Value shifted = mlir::arith::ShLIOp::create(lb, lhs, rhs);

  return selectShiftedOrSaturated(lb, rhs, shifted, zero, type);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ShiftRightLogicalOp>(
    Location loc, ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
    mhlo::ShiftRightLogicalOp::Adaptor adaptor,
    ArrayRef<NamedAttribute> /*attributes*/, OpBuilder* b) {
  ImplicitLocOpBuilder lb(loc, *b);
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  Type type = lhs.getType();

  // "Saturate" if the shift is greater than the bitwidth of the type
  Value zero = arith::ConstantOp::create(lb, b->getZeroAttr(type));
  Value shifted = mlir::arith::ShRUIOp::create(lb, lhs, rhs);

  return selectShiftedOrSaturated(lb, rhs, shifted, zero, type);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ShiftRightArithmeticOp>(
    Location loc, ArrayRef<Type> /*ResultTypes*/, ArrayRef<Type> /*argTypes*/,
    mhlo::ShiftRightArithmeticOp::Adaptor adaptor,
    ArrayRef<NamedAttribute> /*attributes*/, OpBuilder* b) {
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
  Value saturatedShifted = mlir::arith::ShRSIOp::create(lb, lhs, maxShift);
  Value shifted = mlir::arith::ShRSIOp::create(lb, lhs, rhs);

  return selectShiftedOrSaturated(lb, rhs, shifted, saturatedShifted, type);
}
}  // namespace impl

struct MhloOpToStdScalarOp {
  // Converts mhlo 'op' to linalg and arith ops.
  template <typename MhloOpTy>
  static Value mapOp(MhloOpTy op, ArrayRef<Type> resultTypes, ValueRange args,
                     ArrayRef<NamedAttribute> attributes, OpBuilder* b) {
    auto argTypes = llvm::to_vector(op->getOperandTypes());
    return mapOpWithArgTypes(op, resultTypes, argTypes, args, attributes, b);
  }

  // Converts mhlo 'op' to linalg and arith ops. The types of 'args' may already
  // be converted, 'argTypes' are their original types.
  template <typename MhloOpTy>
  static Value mapOpWithArgTypes(MhloOpTy op, ArrayRef<Type> resultTypes,
                                 ArrayRef<Type> argTypes, ValueRange args,
                                 ArrayRef<NamedAttribute> attributes,
                                 OpBuilder* b) {
    static_assert(!std::is_same<MhloOpTy, mhlo::ConvertOp>::value);
    typename MhloOpTy::Adaptor adaptor(args, op->getAttrDictionary(),
                                       op->getPropertiesStorage(),
                                       op->getRegions());
    return mapOpOfType<MhloOpTy>(op.getLoc(), resultTypes, argTypes, adaptor,
                                 attributes, b);
  }
  // Overload for mhlo::ConvertOp.
  static Value mapOpWithArgTypes(mhlo::ConvertOp op, ArrayRef<Type> resultTypes,
                                 ArrayRef<Type> argTypes, ValueRange args,
                                 ArrayRef<NamedAttribute> attributes,
                                 OpBuilder* b) {
    return impl::mapConvertOpToStdScalarOp(
        op.getLoc(), op.getType(), resultTypes, argTypes, args, attributes, b);
  }

  // Converts mhlo 'op' to linalg and arith ops.
  template <typename MhloOpTy>
  static Value mapOpOfType(Location loc, ArrayRef<Type> resultTypes,
                           ArrayRef<Type> argTypes,
                           typename MhloOpTy::Adaptor adaptor,
                           ArrayRef<NamedAttribute> attributes, OpBuilder* b) {
    return impl::mapMhloOpToStdScalarOp<MhloOpTy>(loc, resultTypes, argTypes,
                                                  adaptor, attributes, b);
  }

  static Value mapConvertOpToStdScalarOp(
      Location loc, ArrayRef<Type> targetTypes, ArrayRef<Type> resultTypes,
      ArrayRef<Type> argTypes, ValueRange args,
      ArrayRef<NamedAttribute> attributes, OpBuilder* b) {
    return impl::mapConvertOpToStdScalarOp(loc, targetTypes, resultTypes,
                                           argTypes, args, attributes, b);
  }
};

}  // namespace mhlo
}  // namespace mlir

#endif  // MLIR_HLO_MHLO_TRANSFORMS_MAP_MHLO_TO_SCALAR_OP_H
