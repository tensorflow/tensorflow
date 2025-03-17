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

#include "tensorflow/compiler/mlir/quantization/common/ir/UniformSupport.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>

#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir::quantfork {

static bool isQuantizablePrimitiveType(Type input_type) {
  return isa<FloatType>(input_type);
}

ExpressedToQuantizedConverter ExpressedToQuantizedConverter::forInputType(
    Type input_type) {
  if (isa<TensorType, VectorType>(input_type)) {
    Type element_type = cast<ShapedType>(input_type).getElementType();
    if (!isQuantizablePrimitiveType(element_type))
      return ExpressedToQuantizedConverter{input_type, nullptr};
    return ExpressedToQuantizedConverter{input_type, element_type};
  }
  // Supported primitive type (which just is the expressed type).
  if (isQuantizablePrimitiveType(input_type))
    return ExpressedToQuantizedConverter{input_type, input_type};
  // Unsupported.
  return ExpressedToQuantizedConverter{input_type, nullptr};
}

Type ExpressedToQuantizedConverter::convert(
    quant::QuantizedType elemental_type) const {
  assert(expressed_type && "convert() on unsupported conversion");
  if (auto tensor_type = dyn_cast<RankedTensorType>(input_type))
    return RankedTensorType::get(tensor_type.getShape(), elemental_type);
  if (auto tensor_type = dyn_cast<UnrankedTensorType>(input_type))
    return UnrankedTensorType::get(elemental_type);
  if (auto vector_type = dyn_cast<VectorType>(input_type))
    return VectorType::get(vector_type.getShape(), elemental_type);

  // If the expressed types match, just use the new elemental type.
  if (elemental_type.getExpressedType() == expressed_type) {
    return elemental_type;
  }
  // Unsupported.
  return nullptr;
}

ElementsAttr UniformQuantizedPerAxisValueConverter::convert(
    Attribute real_value) {
  if (auto attr = dyn_cast<DenseFPElementsAttr>(real_value)) {
    return convert(attr);
  }
  return nullptr;
}

DenseElementsAttr UniformQuantizedPerAxisValueConverter::convert(
    DenseFPElementsAttr attr) {
  // Creates the converter for each chunk. Normally the size of the
  // quantization dim is 3, so we can cache all the converters.
  ShapedType type = attr.getType();
  std::size_t dim_size = type.getDimSize(quantization_dim_);
  if (dim_size != scales_.size()) {
    return {};
  }
  SmallVector<UniformQuantizedValueConverter, 4> converters;
  converters.reserve(dim_size);
  for (int i = 0, e = dim_size; i != e; ++i) {
    converters.push_back(getPerChunkConverter(i));
  }

  // Scan the elements of the dense elements attributes and quantize them by
  // using the right quantization parameters.
  int64_t flatten_index = 0;
  auto shape = type.getShape();
  int64_t chunk_size =
      std::accumulate(std::next(shape.begin(), quantization_dim_ + 1),
                      shape.end(), 1, std::multiplies<int64_t>());
  Type new_element_type =
      IntegerType::get(attr.getContext(), storage_bit_width_);
  return attr.mapValues(new_element_type, [&](const APFloat &old) {
    int chunk_index = flatten_index / chunk_size;
    flatten_index++;
    return converters[chunk_index % dim_size].quantizeFloatToInt(old);
  });
}

}  // namespace mlir::quantfork
