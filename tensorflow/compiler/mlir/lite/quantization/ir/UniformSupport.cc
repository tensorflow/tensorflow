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

#include "tensorflow/compiler/mlir/lite/quantization/ir/UniformSupport.h"

#include <numeric>

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project

using namespace mlir;
using namespace mlir::quantfork;

static bool isQuantizablePrimitiveType(Type inputType) {
  return inputType.isa<FloatType>();
}

ExpressedToQuantizedConverter ExpressedToQuantizedConverter::forInputType(
    Type inputType) {
  if (inputType.isa<TensorType, VectorType>()) {
    Type elementType = inputType.cast<ShapedType>().getElementType();
    if (!isQuantizablePrimitiveType(elementType))
      return ExpressedToQuantizedConverter{inputType, nullptr};
    return ExpressedToQuantizedConverter{inputType, elementType};
  }
  // Supported primitive type (which just is the expressed type).
  if (isQuantizablePrimitiveType(inputType))
    return ExpressedToQuantizedConverter{inputType, inputType};
  // Unsupported.
  return ExpressedToQuantizedConverter{inputType, nullptr};
}

Type ExpressedToQuantizedConverter::convert(
    quant::QuantizedType elementalType) const {
  assert(expressedType && "convert() on unsupported conversion");
  if (auto tensorType = inputType.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), elementalType);
  if (auto tensorType = inputType.dyn_cast<UnrankedTensorType>())
    return UnrankedTensorType::get(elementalType);
  if (auto vectorType = inputType.dyn_cast<VectorType>())
    return VectorType::get(vectorType.getShape(), elementalType);

  // If the expressed types match, just use the new elemental type.
  if (elementalType.getExpressedType() == expressedType) return elementalType;
  // Unsupported.
  return nullptr;
}

ElementsAttr UniformQuantizedPerAxisValueConverter::convert(
    Attribute realValue) {
  if (auto attr = realValue.dyn_cast<DenseFPElementsAttr>()) {
    return convert(attr);
  }
  // TODO: handles sparse elements attribute
  return nullptr;
}

DenseElementsAttr UniformQuantizedPerAxisValueConverter::convert(
    DenseFPElementsAttr attr) {
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
  Type newElementType = IntegerType::get(attr.getContext(), storageBitWidth);
  return attr.mapValues(newElementType, [&](const APFloat &old) {
    int chunkIndex = (flattenIndex++) / chunkSize;
    return converters[chunkIndex % dimSize].quantizeFloatToInt(old);
  });
}
