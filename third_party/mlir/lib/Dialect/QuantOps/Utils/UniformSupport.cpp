//===- UniformSupport.cpp - Support utilities for uniform quant -----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QuantOps/UniformSupport.h"
#include "mlir/IR/StandardTypes.h"
#include <numeric>

using namespace mlir;
using namespace mlir::quant;

static bool isQuantizablePrimitiveType(Type inputType) {
  return inputType.isa<FloatType>();
}

const ExpressedToQuantizedConverter
ExpressedToQuantizedConverter::forInputType(Type inputType) {
  switch (inputType.getKind()) {
  default:
    if (isQuantizablePrimitiveType(inputType)) {
      // Supported primitive type (which just is the expressed type).
      return ExpressedToQuantizedConverter{inputType, inputType};
    }
    // Unsupported.
    return ExpressedToQuantizedConverter{inputType, nullptr};
  case StandardTypes::RankedTensor:
  case StandardTypes::UnrankedTensor:
  case StandardTypes::Vector: {
    Type elementType = inputType.cast<ShapedType>().getElementType();
    if (!isQuantizablePrimitiveType(elementType)) {
      // Unsupported.
      return ExpressedToQuantizedConverter{inputType, nullptr};
    }
    return ExpressedToQuantizedConverter{
        inputType, inputType.cast<ShapedType>().getElementType()};
  }
  }
}

Type ExpressedToQuantizedConverter::convert(QuantizedType elementalType) const {
  assert(expressedType && "convert() on unsupported conversion");

  switch (inputType.getKind()) {
  default:
    if (isQuantizablePrimitiveType(elementalType)) {
      // For primitives, just use the new elemental type.
      return elementalType;
    }
    // Unsupported.
    return nullptr;
  case StandardTypes::RankedTensor:
    return RankedTensorType::get(inputType.cast<RankedTensorType>().getShape(),
                                 elementalType);
  case StandardTypes::UnrankedTensor:
    return UnrankedTensorType::get(elementalType);
  case StandardTypes::Vector:
    return VectorType::get(inputType.cast<VectorType>().getShape(),
                           elementalType);
  }
}

ElementsAttr
UniformQuantizedPerAxisValueConverter::convert(Attribute realValue) {
  if (auto attr = realValue.dyn_cast<DenseFPElementsAttr>()) {
    return convert(attr);
  }
  // TODO(fengliuai): handles sparse elements attribute
  return nullptr;
}

DenseElementsAttr
UniformQuantizedPerAxisValueConverter::convert(DenseFPElementsAttr attr) {
  // Creates the converter for each chunk. Normally the size of the
  // quantization dim is 3, so we can cache all the converters.
  ShapedType type = attr.getType();
  size_t dimSize = type.getDimSize(quantizationDim);
  if (dimSize != scales.size()) {
    return {};
  }
  SmallVector<UniformQuantizedValueConverter, 4> converters;
  converters.reserve(dimSize);
  for (int i = 0, e = dimSize; i != e; ++i) {
    converters.push_back(getPerChunkConverter(i));
  }

  // Scan the elements of the dense elements attributes and quantize them by
  // using the right quantization parameters.
  int64_t flattenIndex = 0;
  auto shape = type.getShape();
  int64_t chunkSize =
      std::accumulate(std::next(shape.begin(), quantizationDim + 1),
                      shape.end(), 1, std::multiplies<int64_t>());
  Type newElementType = IntegerType::get(storageBitWidth, attr.getContext());
  return attr.mapValues(newElementType, [&](const APFloat &old) {
    int chunkIndex = (flattenIndex++) / chunkSize;
    return converters[chunkIndex % dimSize].quantizeFloatToInt(old);
  });
}
