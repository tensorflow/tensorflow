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

#include "tensorflow/compiler/mlir/quantization/common/ir/QuantizeUtils.h"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/ir/UniformSupport.h"

namespace mlir {
namespace quant::ir {

/// Converts a possible primitive, real expressed value attribute to a
/// corresponding storage attribute (typically FloatAttr -> IntegerAttr).
/// quantizedElementType is the QuantizedType that describes the expressed
/// origValue.
/// Returns a converter Attribute or nullptr if conversion is not possible.
static Attribute convertPrimitiveValueAttr(
    Attribute origRealValue, quant::QuantizedType quantizedElementType,
    const UniformQuantizedValueConverter &converter, Type &outConvertedType) {
  if (mlir::isa<FloatAttr>(origRealValue)) {
    FloatAttr floatAttr = mlir::cast<FloatAttr>(origRealValue);
    outConvertedType = quantizedElementType.getStorageType();
    return IntegerAttr::get(quantizedElementType.getStorageType(),
                            converter.quantizeFloatToInt(floatAttr.getValue()));
  }

  return nullptr;
}

/// Converts a real expressed DenseFPElementsAttr to a corresponding
/// DenseElementsAttr (typically DenseIntElementsAttr) containing quantized
/// storage values assuming the given quantizedElementType and converter.
static DenseElementsAttr convertDenseFPElementsAttr(
    DenseFPElementsAttr realFPElementsAttr,
    quant::QuantizedType quantizedElementType,
    const UniformQuantizedValueConverter &converter) {
  return realFPElementsAttr.mapValues(
      quantizedElementType.getStorageType(),
      [&converter](const APFloat &realVal) {
        return converter.quantizeFloatToInt(realVal);
      });
}

/// Converts a real expressed SplatElementsAttr to a corresponding
/// SplatElementsAttr containing quantized storage values assuming the given
/// quantizedElementType and converter.
static SparseElementsAttr convertSparseElementsAttr(
    SparseElementsAttr realSparseAttr,
    quant::QuantizedType quantizedElementType,
    const UniformQuantizedValueConverter &converter) {
  DenseElementsAttr realDenseAttr = realSparseAttr.getValues();
  if (!mlir::isa<DenseFPElementsAttr>(realDenseAttr)) {
    return nullptr;
  }
  DenseElementsAttr quantDenseAttr =
      convertDenseFPElementsAttr(mlir::cast<DenseFPElementsAttr>(realDenseAttr),
                                 quantizedElementType, converter);
  if (!quantDenseAttr) {
    return nullptr;
  }

  // Cast from an expressed-type-based type to storage-type-based type,
  // preserving the sparse shape (i.e. tensor<4xf32> -> tensor<4xi8>).
  ShapedType newSparseType = mlir::dyn_cast_or_null<ShapedType>(
      quantizedElementType.castExpressedToStorageType(
          realSparseAttr.getType()));
  if (!newSparseType) {
    return nullptr;
  }
  return SparseElementsAttr::get(newSparseType, realSparseAttr.getIndices(),
                                 quantDenseAttr);
}

/// Converts a real expressed Attribute to a corresponding Attribute containing
/// quantized storage values assuming the given uniform quantizedElementType and
/// converter.
Attribute quantizeAttrUniform(Attribute realValue,
                              quant::UniformQuantizedType quantizedElementType,
                              const UniformQuantizedValueConverter &converter,
                              Type &outConvertedType) {
  // Fork to handle different variants of constants supported.
  if (mlir::isa<DenseFPElementsAttr>(realValue)) {
    // Dense tensor or vector constant.
    auto converted =
        convertDenseFPElementsAttr(mlir::cast<DenseFPElementsAttr>(realValue),
                                   quantizedElementType, converter);
    outConvertedType = converted.getType();
    return converted;
  }
  if (mlir::isa<SparseElementsAttr>(realValue)) {
    // Sparse tensor or vector constant.
    auto converted =
        convertSparseElementsAttr(mlir::cast<SparseElementsAttr>(realValue),
                                  quantizedElementType, converter);
    outConvertedType = converted.getType();
    return converted;
  }
  // Nothing else matched: try to convert a primitive.
  return convertPrimitiveValueAttr(realValue, quantizedElementType, converter,
                                   outConvertedType);
}

/// Convert an attribute from a type based on
/// quantizedElementType.getExpressedType() to one based on
/// quantizedElementType.getStorageType().
/// Returns nullptr if the conversion is not supported.
/// On success, stores the converted type in outConvertedType.
Attribute quantizeAttr(Attribute realValue,
                       quant::QuantizedType quantizedElementType,
                       Type &outConvertedType) {
  if (auto uniformQuantized =
          mlir::dyn_cast<quant::UniformQuantizedType>(quantizedElementType)) {
    UniformQuantizedValueConverter converter(uniformQuantized);
    return quantizeAttrUniform(realValue, uniformQuantized, converter,
                               outConvertedType);
  }
  if (auto uniformQuantizedPerAxis =
          mlir::dyn_cast<quant::UniformQuantizedPerAxisType>(
              quantizedElementType)) {
    UniformQuantizedPerAxisValueConverter converter(uniformQuantizedPerAxis);
    auto converted = converter.convert(realValue);
    // TODO: why we need this outConvertedType? remove it?
    if (converted) {
      outConvertedType = converted.getType();
    }
    return converted;
  }
  return nullptr;
}

}  // namespace quant::ir
}  // namespace mlir
