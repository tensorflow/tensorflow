//===- UniformSupport.h - Support utilities for uniform quant ---*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef MLIR_DIALECT_QUANTOPS_UNIFORMSUPPORT_H_
#define MLIR_DIALECT_QUANTOPS_UNIFORMSUPPORT_H_

#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"

namespace mlir {
namespace quant {

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
  static const ExpressedToQuantizedConverter forInputType(Type inputType);

  /// Converts the inputType to be based on the given elemental type,
  /// returning the new type (or nullptr and emit an error on failure).
  Type convert(QuantizedType elementalType) const;

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
/// Note that this is not expected to be speedy and may be superceded eventually
/// by a more optimal implementation.
/// Also, the interface assumes that quantization is done per-layer and will
/// need to be wider for various per-channel schemes. As such, this is a
/// placeholder.
class UniformQuantizedValueConverter {
public:
  explicit UniformQuantizedValueConverter(UniformQuantizedType uniformType)
      : scale(uniformType.getScale()),
        zeroPoint(static_cast<double>(uniformType.getZeroPoint())),
        clampMin(static_cast<double>(uniformType.getStorageTypeMin())),
        clampMax(static_cast<double>(uniformType.getStorageTypeMax())),
        storageBitWidth(uniformType.getStorageTypeIntegralWidth()),
        isSigned(uniformType.isSigned()) {
    assert(uniformType.getExpressedType().isa<FloatType>());
    assert(uniformType.getStorageType().isa<IntegerType>());
  }

  UniformQuantizedValueConverter(double scale, double zeroPoint,
                                 APFloat clampMin, APFloat clampMax,
                                 uint32_t storageBitWidth, bool isSigned)
      : scale(scale), zeroPoint(zeroPoint), clampMin(clampMin),
        clampMax(clampMax), storageBitWidth(storageBitWidth),
        isSigned(isSigned) {}

  virtual APInt quantizeFloatToInt(APFloat expressedValue) const {
    bool lossy;
    expressedValue.convert(scale.getSemantics(), APFloat::rmNearestTiesToEven,
                           &lossy);
    // fixedpoint = clamp(clampMin, clampMax, (
    //   roundHalfToEven(expressed / scale) + zeroPoint))
    APFloat scaled = (expressedValue / scale);
    scaled.roundToIntegral(APFloat::rmNearestTiesToEven);
    scaled.add(zeroPoint, APFloat::rmNearestTiesToEven);
    APFloat fixedpoint = llvm::minimum(scaled, clampMax);
    fixedpoint = llvm::maximum(fixedpoint, clampMin);

    llvm::APSInt result(storageBitWidth, !isSigned);
    fixedpoint.convertToInteger(result, APFloat::rmNearestTiesToEven, &lossy);

    return std::move(result);
  }

  int64_t quantizeFloatToInt64(APFloat expressedValue) const {
    APInt qValue = quantizeFloatToInt(expressedValue);
    return isSigned ? qValue.getSExtValue() : qValue.getZExtValue();
  }

  virtual ~UniformQuantizedValueConverter() {}

private:
  const APFloat scale;
  const APFloat zeroPoint;
  const APFloat clampMin;
  const APFloat clampMax;
  const uint32_t storageBitWidth;
  const bool isSigned;
};

/// An utility class to quantize an attribute by the per-axis quantization
/// parameters. The size of the quantization dim in the converted elements
/// attribute should matche the size of of scales/zeroPoints vectors in the
/// quantization parameters.
class UniformQuantizedPerAxisValueConverter {
public:
  explicit UniformQuantizedPerAxisValueConverter(
      UniformQuantizedPerAxisType uniformType)
      : scales(uniformType.getScales()),
        zeroPoints(uniformType.getZeroPoints()),
        clampMin(static_cast<double>(uniformType.getStorageTypeMin())),
        clampMax(static_cast<double>(uniformType.getStorageTypeMax())),
        storageBitWidth(uniformType.getStorageTypeIntegralWidth()),
        isSigned(uniformType.isSigned()),
        quantizationDim(uniformType.getQuantizedDimension()) {
    assert(uniformType.getExpressedType().isa<FloatType>());
    assert(uniformType.getStorageType().isa<IntegerType>());
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

} // namespace quant
} // namespace mlir

#endif // MLIR_DIALECT_QUANTOPS_UNIFORMSUPPORT_H_
