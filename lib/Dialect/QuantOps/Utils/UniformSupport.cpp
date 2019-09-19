//===- UniformSupport.cpp - Support utilities for uniform quant -----------===//
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
