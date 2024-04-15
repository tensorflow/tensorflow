/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_IR_UNIFORMSUPPORT_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_IR_UNIFORMSUPPORT_H_

#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project

namespace mlir {
namespace quantfork {

/// Performs type conversion from an arbitrary input type to a type
/// that is expressed by a QuantizedType.
///
/// This handles cases where the inputType is a supported primitive type
/// (i.e. f32, bf16, etc) or a vector/tensor type based on a supported
/// elemental type.
///
/// Since conversion often involves introspecting some attributes of the
/// input type in order to determine how to represent it, this is a two step
/// process.
struct ExpressedToQuantizedConverter {
  /// Creates a converter for the given input type.
  static ExpressedToQuantizedConverter forInputType(Type inputType);

  /// Converts the inputType to be based on the given elemental type,
  /// returning the new type (or nullptr and emit an error on failure).
  Type convert(quant::QuantizedType elementalType) const;

  /// Whether the conversion is legal.
  explicit operator bool() const { return (bool)expressedType; }

  /// The input type that is being converted from.
  /// This may be an elemental or composite type.
  const Type inputType;

  /// Supported, elemental expressed type (i.e. f32).
  /// Will be nullptr if conversion is not supported.
  const Type expressedType;
};

/// Reference implementation of converting between real numbers and values
/// represented by a UniformQuantizedType.
/// Note that this is not expected to be speedy and may be superseded eventually
/// by a more optimal implementation.
/// Also, the interface assumes that quantization is done per-layer and will
/// need to be wider for various per-channel schemes. As such, this is a
/// placeholder.
class UniformQuantizedValueConverter {
 public:
  explicit UniformQuantizedValueConverter(
      quant::UniformQuantizedType uniformType)
      : UniformQuantizedValueConverter(
            uniformType.getScale(),
            static_cast<double>(uniformType.getZeroPoint()),
            static_cast<double>(uniformType.getStorageTypeMin()),
            static_cast<double>(uniformType.getStorageTypeMax()),
            uniformType.getStorageTypeIntegralWidth(), uniformType.isSigned()) {
    assert(uniformType.getExpressedType().isa<FloatType>());
    assert(uniformType.getStorageType().isSignlessInteger());
  }

  UniformQuantizedValueConverter(double scale, double zeroPoint,
                                 double clampMin, double clampMax,
                                 uint32_t storageBitWidth, bool isSigned)
      : scale(scale),
        zeroPoint(zeroPoint),
        clampMin(clampMin),
        clampMax(clampMax),
        scaleDouble(scale),
        zeroPointDouble(zeroPoint),
        clampMinDouble(clampMin),
        clampMaxDouble(clampMax),
        storageBitWidth(storageBitWidth),
        isSigned(isSigned),
        roundMode(APFloat::rmNearestTiesToAway) {}

  UniformQuantizedValueConverter(double scale, double zeroPoint,
                                 const APFloat &clampMin,
                                 const APFloat &clampMax,
                                 uint32_t storageBitWidth, bool isSigned)
      : scale(scale),
        zeroPoint(zeroPoint),
        clampMin(clampMin),
        clampMax(clampMax),
        scaleDouble(scale),
        zeroPointDouble(zeroPoint),
        clampMinDouble(clampMin.convertToDouble()),
        clampMaxDouble(clampMax.convertToDouble()),
        storageBitWidth(storageBitWidth),
        isSigned(isSigned),
        roundMode(APFloat::rmNearestTiesToAway) {}

  virtual APInt quantizeFloatToInt(APFloat expressedValue) const {
    // This function is a performance critical code path in quantization
    // since it runs for each single float parameter value.

    // Specialize f32->u8/i8 case to optimize performance.
    if (&expressedValue.getSemantics() == &APFloat::IEEEsingle() &&
        storageBitWidth == 8 &&
        roundMode == llvm::APFloatBase::rmNearestTiesToAway) {
      return quantizeF32ToInt8(expressedValue);
    }

    bool lossy;
    expressedValue.convert(scale.getSemantics(), roundMode, &lossy);
    // fixedpoint = clamp(clampMin, clampMax, (
    //   roundHalfToEven(expressed / scale) + zeroPoint))
    APFloat scaled = (expressedValue / scale);
    scaled.roundToIntegral(roundMode);
    scaled.add(zeroPoint, roundMode);
    APFloat fixedpoint = llvm::minimum(scaled, clampMax);
    fixedpoint = llvm::maximum(fixedpoint, clampMin);

    llvm::APSInt result(storageBitWidth, !isSigned);
    fixedpoint.convertToInteger(result, roundMode, &lossy);

    return std::move(result);
  }

  int64_t quantizeFloatToInt64(APFloat expressedValue) const {
    APInt qValue = quantizeFloatToInt(std::move(expressedValue));
    return isSigned ? qValue.getSExtValue() : qValue.getZExtValue();
  }

  virtual ~UniformQuantizedValueConverter() = default;

 private:
  // An optimized implementation to quantize f32 to i8/u8 with C++ native
  // arithmetic.
  virtual APInt quantizeF32ToInt8(APFloat expressedValue) const {
    assert(&expressedValue.getSemantics() == &APFloat::IEEEsingle());
    assert(storageBitWidth == 8);
    assert(roundMode == llvm::APFloatBase::rmNearestTiesToAway);

    const float realValue = expressedValue.convertToFloat();

    const double scaled = realValue / scaleDouble + zeroPointDouble;
    // Round to nearest integer with halfway cases rounded away from zero.
    const double scaledRounded = std::round(scaled);
    const double clamped =
        std::min(std::max(scaledRounded, clampMinDouble), clampMaxDouble);

    uint64_t signlessResult;
    if (isSigned) {
      int64_t clampedInt = static_cast<int8_t>(clamped);
      memcpy(&signlessResult, &clampedInt, sizeof(clampedInt));
    } else {
      signlessResult = static_cast<uint8_t>(clamped);
    }
    return APInt(storageBitWidth, signlessResult);
  }

  // Keep both APFloat and double versions of the quantization parameters
  // around since they will be used in generic and specialized arithmetic,
  // respectively.
  const APFloat scale;
  const APFloat zeroPoint;
  const APFloat clampMin;
  const APFloat clampMax;

  const double scaleDouble;
  const double zeroPointDouble;
  const double clampMinDouble;
  const double clampMaxDouble;

  const uint32_t storageBitWidth;
  const bool isSigned;
  const llvm::APFloat::roundingMode roundMode;
};

/// An utility class to quantize an attribute by the per-axis quantization
/// parameters. The size of the quantization dim in the converted elements
/// attribute should matche the size of of scales/zeroPoints vectors in the
/// quantization parameters.
class UniformQuantizedPerAxisValueConverter {
 public:
  explicit UniformQuantizedPerAxisValueConverter(
      quant::UniformQuantizedPerAxisType uniformType)
      : scales(uniformType.getScales()),
        zeroPoints(uniformType.getZeroPoints()),
        clampMin(static_cast<double>(uniformType.getStorageTypeMin())),
        clampMax(static_cast<double>(uniformType.getStorageTypeMax())),
        storageBitWidth(uniformType.getStorageTypeIntegralWidth()),
        isSigned(uniformType.isSigned()),
        quantizationDim(uniformType.getQuantizedDimension()) {
    assert(uniformType.getExpressedType().isa<FloatType>());
    assert(uniformType.getStorageType().isSignlessInteger());
    assert(scales.size() == zeroPoints.size());
  }

  /// Quantize an Attribute by the quantization parameters. Return nullptr if
  /// the conversion fails or the input array isn't an ElementsAttr.
  ElementsAttr convert(Attribute realValue);

 private:
  /// Quantize an DenseFPElementsAttr by the quantization parameters.
  DenseElementsAttr convert(DenseFPElementsAttr attr);

  /// Get a uniform converter for the index-th chunk along the quantizationDim.
  /// All the elements in this chunk is quantized by the returned converter.
  UniformQuantizedValueConverter getPerChunkConverter(int index) const {
    UniformQuantizedValueConverter converter(scales[index], zeroPoints[index],
                                             clampMin, clampMax,
                                             storageBitWidth, isSigned);
    return converter;
  }

  const ArrayRef<double> scales;
  const ArrayRef<int64_t> zeroPoints;
  const APFloat clampMin;
  const APFloat clampMax;
  const uint32_t storageBitWidth;
  const bool isSigned;
  int32_t quantizationDim;
};

}  // namespace quantfork
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_IR_UNIFORMSUPPORT_H_
