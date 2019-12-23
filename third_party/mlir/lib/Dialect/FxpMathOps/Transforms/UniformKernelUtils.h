//===- UniformKernelUtils.h - Utilities for lowering uniform math - C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_FXPMATH_UNIFORM_KERNEL_UTILS_H_
#define MLIR_FXPMATH_UNIFORM_KERNEL_UTILS_H_

#include "mlir/Dialect/QuantOps/QuantOps.h"
#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "mlir/Dialect/QuantOps/UniformSupport.h"
#include "mlir/IR/Operation.h"

#include <cmath>

namespace mlir {
namespace fxpmath {
namespace detail {

inline quant::UniformQuantizedType getUniformElementType(Type t) {
  return quant::QuantizedType::getQuantizedElementType(t)
      .dyn_cast_or_null<quant::UniformQuantizedType>();
}

inline bool hasStorageBitWidth(quant::QuantizedType t,
                               ArrayRef<unsigned> checkWidths) {
  unsigned w = t.getStorageType().getIntOrFloatBitWidth();
  for (unsigned checkWidth : checkWidths) {
    if (w == checkWidth)
      return true;
  }
  return false;
}

/// Computes the log2(x), rounded to an integral value. Returns whether 'x' can
/// be considered an exact integral value.
template <typename F> bool integralLog2(F x, int &log2Result) {
  const F xLog2 = std::log(x) * (1.0 / std::log(2.0));
  const F xLog2Rounded = std::round(xLog2);
  const F xLog2Frac = xLog2 - xLog2Rounded;
  log2Result = static_cast<int>(xLog2Rounded);
  // Allow small comparison slop below the level that would make a difference
  // for 2^16 levels.
  return std::abs(xLog2Frac) < 1e-6;
}

/// Helper class for operating on binary operations where all operands
/// and the result are a UniformQuantizedType.
struct UniformBinaryOpInfo {
  UniformBinaryOpInfo(Operation *op, Value lhs, Value rhs,
                      Optional<APFloat> clampMin, Optional<APFloat> clampMax)
      : op(op), lhs(lhs), rhs(rhs), clampMin(clampMin), clampMax(clampMax),
        lhsType(getUniformElementType(lhs->getType())),
        rhsType(getUniformElementType(rhs->getType())),
        resultType(getUniformElementType(*op->result_type_begin())),
        lhsStorageType(quant::QuantizedType::castToStorageType(lhs->getType())),
        rhsStorageType(quant::QuantizedType::castToStorageType(rhs->getType())),
        resultStorageType(
            quant::QuantizedType::castToStorageType(*op->result_type_begin())) {
  }

  /// Returns whether this info is valid (all types defined, etc).
  bool isValid() const {
    return lhsType && rhsType && resultType && lhsStorageType &&
           rhsStorageType && resultStorageType;
  }

  /// Gets the final quantized result type of the result.
  Type getQuantizedResultType() const { return *op->result_type_begin(); }

  /// Returns whether the storage type of all operands is identical.
  bool isSameStorageType() const {
    return lhsType.getStorageType() == rhsType.getStorageType() &&
           lhsType.getStorageType() == resultType.getStorageType();
  }

  /// Returns whether all operands and result are considered fixedpoint power
  /// of two, setting the lhs, rhs, and result log2 scale references.
  bool isFixedPointPOT(int &lhsLog2Scale, int &rhsLog2Scale,
                       int &resultLog2Scale) const {
    if (!lhsType.isFixedPoint() || !rhsType.isFixedPoint() ||
        !resultType.isFixedPoint()) {
      return false;
    }

    if (!integralLog2(lhsType.getScale(), lhsLog2Scale) ||
        !integralLog2(rhsType.getScale(), rhsLog2Scale) ||
        !integralLog2(resultType.getScale(), resultLog2Scale)) {
      return false;
    }

    return true;
  }

  /// Gets the result integer clamp range given the result quantized type
  // and any explicit clamp provided as attributes.
  std::pair<IntegerAttr, IntegerAttr> getClampMinMax(IntegerType ty) const {
    int64_t typeMin = resultType.getStorageTypeMin();
    int64_t typeMax = resultType.getStorageTypeMax();

    if (clampMin || clampMax) {
      quant::UniformQuantizedValueConverter conv(resultType);
      if (clampMin) {
        typeMin = std::max(typeMin, conv.quantizeFloatToInt64(*clampMin));
      }
      if (clampMax) {
        typeMax = std::min(typeMax, conv.quantizeFloatToInt64(*clampMax));
      }
    }

    // The quantized, integral ops expect clamps as 32bit ints.
    return {
        IntegerAttr::get(ty, typeMin),
        IntegerAttr::get(ty, typeMax),
    };
  }

  Operation *op;
  Value lhs;
  Value rhs;
  Optional<APFloat> clampMin;
  Optional<APFloat> clampMax;

  // Element UniformQuantizedType for operands/result.
  quant::UniformQuantizedType lhsType;
  quant::UniformQuantizedType rhsType;
  quant::UniformQuantizedType resultType;

  // Full storage-based types.
  Type lhsStorageType;
  Type rhsStorageType;
  Type resultStorageType;
};

/// Derives a quantized multiplier and shift from a real valued multiplier
/// less than 1.
struct QuantizedMultiplierSmallerThanOneExp {
  QuantizedMultiplierSmallerThanOneExp(double realMultiplier) {
    assert(realMultiplier < 1.0);
    assert(realMultiplier > 0.0);

    const double q = std::frexp(realMultiplier, &exponent);
    auto qFixed = static_cast<int64_t>(std::round(q * (1ll << 31)));
    assert(qFixed <= (1ll << 31));
    if (qFixed == (1ll << 31)) {
      qFixed /= 2;
      ++exponent;
    }
    assert(qFixed <= std::numeric_limits<int32_t>::max());
    multiplier = static_cast<int32_t>(qFixed);
  }

  int32_t multiplier;
  int exponent;
};

/// Casts an integer or floating point based shaped type to a new element type.
inline Type castElementType(Type t, Type newElementType) {
  if (auto st = t.dyn_cast<ShapedType>()) {
    switch (st.getKind()) {
    case StandardTypes::Kind::Vector:
      return VectorType::get(st.getShape(), newElementType);
    case StandardTypes::Kind::RankedTensor:
      return RankedTensorType::get(st.getShape(), newElementType);
    case StandardTypes::Kind::UnrankedTensor:
      return UnrankedTensorType::get(newElementType);
    case StandardTypes::Kind::MemRef:
      return MemRefType::get(st.getShape(), newElementType,
                             st.cast<MemRefType>().getAffineMaps());
    }
  }
  assert(t.isIntOrFloat());
  return newElementType;
}

/// Creates an IntegerAttr with a type that matches the shape of 't' (which can
/// be a scalar primitive or a shaped type).
inline Attribute broadcastScalarConstIntValue(Type t, int64_t value) {
  if (auto st = t.dyn_cast<ShapedType>()) {
    assert(st.getElementType().isa<IntegerType>());
    return DenseElementsAttr::get(st,
                                  IntegerAttr::get(st.getElementType(), value));
  }

  auto integerType = t.cast<IntegerType>();
  assert(t.isa<IntegerType>() && "integer broadcast must be of integer type");
  return IntegerAttr::get(integerType, value);
}

/// Given an APFloat, converts it to the float semantics that matches the
/// given FloatType, silently ignoring inexact conversions.
inline APFloat convertFloatToType(FloatType ft, APFloat value) {
  bool losesInfo;
  auto status = value.convert(ft.getFloatSemantics(),
                              APFloat::rmNearestTiesToEven, &losesInfo);
  (void)status; // unused in opt mode
  assert((status & (APFloat::opDivByZero | APFloat::opInvalidOp)) == 0 &&
         "could not convert to float const");
  return value;
}

/// Creates a FloatAttr with a type that matches the shape of 't' (which can be
/// a scalar primitive or a shaped type).
inline Attribute broadcastScalarConstFloatValue(Type t, APFloat value) {
  if (auto st = t.dyn_cast<ShapedType>()) {
    FloatType floatElementType = st.getElementType().dyn_cast<FloatType>();
    assert(floatElementType &&
           "float broadcast element type must be float like");
    APFloat apValue = convertFloatToType(floatElementType, value);
    return DenseElementsAttr::get(st,
                                  FloatAttr::get(st.getElementType(), apValue));
  } else {
    auto floatType = t.dyn_cast<FloatType>();
    assert(floatType && "float broadcast must be of float type");
    APFloat apValue = convertFloatToType(floatType, value);
    return FloatAttr::get(floatType, apValue);
  }
}

} // namespace detail
} // namespace fxpmath
} // namespace mlir

#endif // MLIR_FXPMATH_UNIFORM_KERNEL_UTILS_H_
