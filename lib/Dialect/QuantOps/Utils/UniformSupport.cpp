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

using namespace mlir;
using namespace mlir::quant;

static bool isQuantizablePrimitiveType(Type inputType) {
  return inputType.isa<FloatType>();
}

const ExpressedToUniformQuantizedConverter
ExpressedToUniformQuantizedConverter::forInputType(Type inputType) {
  switch (inputType.getKind()) {
  default:
    if (isQuantizablePrimitiveType(inputType)) {
      // Supported primitive type (which just is the expressed type).
      return ExpressedToUniformQuantizedConverter{inputType, inputType};
    }
    // Unsupported.
    return ExpressedToUniformQuantizedConverter{inputType, nullptr};
  case StandardTypes::RankedTensor:
  case StandardTypes::UnrankedTensor:
  case StandardTypes::Vector: {
    Type elementType = inputType.cast<VectorOrTensorType>().getElementType();
    if (!isQuantizablePrimitiveType(elementType)) {
      // Unsupported.
      return ExpressedToUniformQuantizedConverter{inputType, nullptr};
    }
    return ExpressedToUniformQuantizedConverter{
        inputType, inputType.cast<VectorOrTensorType>().getElementType()};
  }
  }
}

Type ExpressedToUniformQuantizedConverter::convert(
    UniformQuantizedType elementalType) const {
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
