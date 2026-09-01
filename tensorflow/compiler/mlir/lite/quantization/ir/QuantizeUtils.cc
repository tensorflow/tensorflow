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

#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantizeUtils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/DialectResourceBlobManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/ir/UniformSupport.h"

namespace mlir {
namespace quantfork {

/// Converts a possible primitive, real expressed value attribute to a
/// corresponding storage attribute (typically FloatAttr -> IntegerAttr).
/// quantizedElementType is the QuantizedType that describes the expressed
/// origValue.
/// Returns a converter Attribute or nullptr if conversion is not possible.
static Attribute convertPrimitiveValueAttr(
    Attribute origRealValue, quant::QuantizedType quantizedElementType,
    const mlir::quant::ir::UniformQuantizedValueConverter& converter,
    Type& outConvertedType) {
  if (mlir::isa<FloatAttr>(origRealValue)) {
    const FloatAttr floatAttr = mlir::cast<FloatAttr>(origRealValue);
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
    const mlir::quant::ir::UniformQuantizedValueConverter& converter) {
  return realFPElementsAttr.mapValues(
      quantizedElementType.getStorageType(),
      [&converter](const APFloat& realVal) {
        return converter.quantizeFloatToInt(realVal);
      });
}

/// Converts a real expressed SplatElementsAttr to a corresponding
/// SplatElementsAttr containing quantized storage values assuming the given
/// quantizedElementType and converter.
static SparseElementsAttr convertSparseElementsAttr(
    SparseElementsAttr realSparseAttr,
    quant::QuantizedType quantizedElementType,
    const mlir::quant::ir::UniformQuantizedValueConverter& converter) {
  DenseElementsAttr realDenseAttr = realSparseAttr.getValues();
  if (!mlir::isa<DenseFPElementsAttr>(realDenseAttr)) {
    return nullptr;
  }
  const DenseElementsAttr quantDenseAttr =
      convertDenseFPElementsAttr(mlir::cast<DenseFPElementsAttr>(realDenseAttr),
                                 quantizedElementType, converter);
  if (!quantDenseAttr) {
    return nullptr;
  }

  // Cast from an expressed-type-based type to storage-type-based type,
  // preserving the sparse shape (i.e. tensor<4xf32> -> tensor<4xi8>).
  const ShapedType newSparseType = mlir::dyn_cast_or_null<ShapedType>(
      quantizedElementType.castExpressedToStorageType(
          realSparseAttr.getType()));
  if (!newSparseType) {
    return nullptr;
  }
  return SparseElementsAttr::get(newSparseType, realSparseAttr.getIndices(),
                                 quantDenseAttr);
}

static Attribute quantizeResourceAttrPerAxisLegacy(
    DenseResourceElementsAttr resourceAttr,
    quant::UniformQuantizedPerAxisType quantizedElementType,
    Type& outConvertedType) {
  const ShapedType type = resourceAttr.getType();
  const int32_t quantDim = quantizedElementType.getQuantizedDimension();
  const uint32_t storageBitWidth =
      quantizedElementType.getStorageTypeIntegralWidth();
  const bool isSigned = quantizedElementType.isSigned();
  const ArrayRef<double> scales = quantizedElementType.getScales();
  const ArrayRef<int64_t> zeroPoints = quantizedElementType.getZeroPoints();
  const double clampMin =
      static_cast<double>(quantizedElementType.getStorageTypeMin());
  const double clampMax =
      static_cast<double>(quantizedElementType.getStorageTypeMax());

  const std::string newKey =
      (llvm::Twine(resourceAttr.getRawHandle().getKey()) + "_quant_axis_" +
       llvm::Twine(quantDim) + "_w_" + llvm::Twine(storageBitWidth))
          .str();

  const Type storageElemType =
      IntegerType::get(resourceAttr.getContext(), storageBitWidth,
                       isSigned ? IntegerType::Signed : IntegerType::Signless);
  const auto resType = RankedTensorType::get(type.getShape(), storageElemType);

  auto& manager = DenseResourceElementsHandle::getManagerInterface(
      resourceAttr.getContext());
  if (const auto* entry = manager.getBlobManager().lookup(newKey)) {
    if (entry->getBlob()) {
      auto* dialect =
          resourceAttr.getContext()->getLoadedDialect<BuiltinDialect>();
      const DenseResourceElementsHandle handle(
          const_cast<DialectResourceBlobManager::BlobEntry*>(entry), dialect);
      outConvertedType = resType;
      return DenseResourceElementsAttr::get(resType, handle);
    }
  }

  const AsmResourceBlob* blob = resourceAttr.getRawHandle().getBlob();
  if (!blob && resourceAttr.getRawHandle().getResource()) {
    blob = resourceAttr.getRawHandle().getResource()->getBlob();
  }
  if (!blob || blob->getData().empty()) return nullptr;

  const size_t numElements = type.getNumElements();
  const size_t elemByteSize = std::max<size_t>(1, storageBitWidth / 8);
  const size_t outByteSize = (storageBitWidth == 4) ? (numElements + 1) / 2
                             : (storageBitWidth == 2)
                                 ? (numElements + 3) / 4
                                 : numElements * elemByteSize;

  auto rawOutputBlob = mlir::HeapAsmResourceBlob::allocate(
      outByteSize, /*align=*/64, /*dataIsMutable=*/true);

  const std::size_t dimSize = type.getDimSize(quantDim);
  if (dimSize != scales.size()) {
    return nullptr;
  }
  SmallVector<mlir::quant::ir::UniformQuantizedValueConverter, 4> converters;
  converters.reserve(dimSize);
  for (int i = 0, e = dimSize; i != e; ++i) {
    converters.emplace_back(scales[i], zeroPoints[i], APFloat(clampMin),
                            APFloat(clampMax), storageBitWidth, isSigned);
  }

  const auto shape = type.getShape();
  const int64_t chunkSize =
      std::accumulate(std::next(shape.begin(), quantDim + 1), shape.end(),
                      int64_t{1}, std::multiplies<int64_t>());

  const ArrayRef<float> rawFloat(
      reinterpret_cast<const float*>(blob->getData().data()), numElements);
  char* outData = const_cast<char*>(rawOutputBlob.getDataAs<char>().data());

  if (storageBitWidth == 8) {
    const MutableArrayRef<int8_t> outInt8(reinterpret_cast<int8_t*>(outData),
                                          numElements);
    for (size_t elemIdx = 0; elemIdx < numElements; ++elemIdx) {
      const int chunkIndex = (elemIdx / chunkSize) % dimSize;
      const APFloat old(rawFloat[elemIdx]);
      const APInt q = converters[chunkIndex].quantizeFloatToInt(old);
      outInt8[elemIdx] = static_cast<int8_t>(q.getSExtValue());
    }
  } else if (storageBitWidth == 4) {
    const MutableArrayRef<int8_t> outInt8(reinterpret_cast<int8_t*>(outData),
                                          outByteSize);
    llvm::fill(outInt8, int8_t{0});
    for (size_t elemIdx = 0; elemIdx < numElements; ++elemIdx) {
      const int chunkIndex = (elemIdx / chunkSize) % dimSize;
      const APFloat old(rawFloat[elemIdx]);
      const APInt q = converters[chunkIndex].quantizeFloatToInt(old);
      const int8_t val = static_cast<int8_t>(q.getSExtValue()) & 0x0F;
      if (elemIdx % 2 == 0) {
        outInt8[elemIdx / 2] |= val;
      } else {
        outInt8[elemIdx / 2] |= (val << 4);
      }
    }
  } else if (storageBitWidth == 2) {
    const MutableArrayRef<int8_t> outInt8(reinterpret_cast<int8_t*>(outData),
                                          outByteSize);
    llvm::fill(outInt8, int8_t{0});
    for (size_t elemIdx = 0; elemIdx < numElements; ++elemIdx) {
      const int chunkIndex = (elemIdx / chunkSize) % dimSize;
      const APFloat old(rawFloat[elemIdx]);
      const APInt q = converters[chunkIndex].quantizeFloatToInt(old);
      const int8_t val = static_cast<int8_t>(q.getSExtValue()) & 0x03;
      const int shift = (elemIdx % 4) * 2;
      outInt8[elemIdx / 4] |= (val << shift);
    }
  } else {
    const MutableArrayRef<int16_t> outInt16(reinterpret_cast<int16_t*>(outData),
                                            numElements);
    for (size_t elemIdx = 0; elemIdx < numElements; ++elemIdx) {
      const int chunkIndex = (elemIdx / chunkSize) % dimSize;
      const APFloat old(rawFloat[elemIdx]);
      const APInt q = converters[chunkIndex].quantizeFloatToInt(old);
      outInt16[elemIdx] = static_cast<int16_t>(q.getSExtValue());
    }
  }

  outConvertedType = resType;
  return DenseResourceElementsAttr::get(resType, newKey,
                                        std::move(rawOutputBlob));
}

static Attribute quantizeResourceAttrPerAxisFast(
    DenseResourceElementsAttr resourceAttr,
    quant::UniformQuantizedPerAxisType quantizedElementType,
    Type& outConvertedType) {
  const ShapedType type = resourceAttr.getType();
  const int32_t quantDim = quantizedElementType.getQuantizedDimension();
  const uint32_t storageBitWidth =
      quantizedElementType.getStorageTypeIntegralWidth();
  const bool isSigned = quantizedElementType.isSigned();
  const ArrayRef<double> scales = quantizedElementType.getScales();
  const ArrayRef<int64_t> zeroPoints = quantizedElementType.getZeroPoints();
  const double clampMin =
      static_cast<double>(quantizedElementType.getStorageTypeMin());
  const double clampMax =
      static_cast<double>(quantizedElementType.getStorageTypeMax());

  const std::string newKey =
      (llvm::Twine(resourceAttr.getRawHandle().getKey()) + "_quant_axis_" +
       llvm::Twine(quantDim) + "_w_" + llvm::Twine(storageBitWidth))
          .str();

  const Type storageElemType =
      IntegerType::get(resourceAttr.getContext(), storageBitWidth,
                       isSigned ? IntegerType::Signed : IntegerType::Signless);
  const auto resType = RankedTensorType::get(type.getShape(), storageElemType);

  auto& manager = DenseResourceElementsHandle::getManagerInterface(
      resourceAttr.getContext());
  if (const auto* entry = manager.getBlobManager().lookup(newKey)) {
    if (entry->getBlob()) {
      auto* dialect =
          resourceAttr.getContext()->getLoadedDialect<BuiltinDialect>();
      const DenseResourceElementsHandle handle(
          const_cast<DialectResourceBlobManager::BlobEntry*>(entry), dialect);
      outConvertedType = resType;
      return DenseResourceElementsAttr::get(resType, handle);
    }
  }

  const AsmResourceBlob* blob = resourceAttr.getRawHandle().getBlob();
  if (!blob && resourceAttr.getRawHandle().getResource()) {
    blob = resourceAttr.getRawHandle().getResource()->getBlob();
  }
  if (!blob || blob->getData().empty()) return nullptr;

  const size_t numElements = type.getNumElements();
  const size_t elemByteSize = std::max<size_t>(1, storageBitWidth / 8);
  const size_t outByteSize = (storageBitWidth == 4) ? (numElements + 1) / 2
                             : (storageBitWidth == 2)
                                 ? (numElements + 3) / 4
                                 : numElements * elemByteSize;

  auto rawOutputBlob = mlir::HeapAsmResourceBlob::allocate(
      outByteSize, /*align=*/64, /*dataIsMutable=*/true);

  const std::size_t dimSize = type.getDimSize(quantDim);
  if (dimSize != scales.size()) {
    return nullptr;
  }

  std::vector<float> invScales(dimSize);
  std::vector<float> zps(dimSize);
  for (size_t i = 0; i < dimSize; ++i) {
    invScales[i] = 1.0f / static_cast<float>(scales[i]);
    zps[i] = static_cast<float>(zeroPoints[i]);
  }
  const float fMin = static_cast<float>(clampMin);
  const float fMax = static_cast<float>(clampMax);

  const auto shape = type.getShape();
  const int64_t chunkSize =
      std::accumulate(std::next(shape.begin(), quantDim + 1), shape.end(),
                      int64_t{1}, std::multiplies<int64_t>());

  const ArrayRef<float> rawFloat(
      reinterpret_cast<const float*>(blob->getData().data()), numElements);
  char* outData = const_cast<char*>(rawOutputBlob.getDataAs<char>().data());

  if (storageBitWidth == 8) {
    const MutableArrayRef<int8_t> outInt8(reinterpret_cast<int8_t*>(outData),
                                          numElements);
    for (size_t elemIdx = 0; elemIdx < numElements; ++elemIdx) {
      const int chunkIndex = (elemIdx / chunkSize) % dimSize;
      const float val =
          std::clamp(std::nearbyint(rawFloat[elemIdx] * invScales[chunkIndex]) +
                         zps[chunkIndex],
                     fMin, fMax);
      outInt8[elemIdx] = static_cast<int8_t>(val);
    }
  } else if (storageBitWidth == 4) {
    const MutableArrayRef<int8_t> outInt8(reinterpret_cast<int8_t*>(outData),
                                          outByteSize);
    llvm::fill(outInt8, int8_t{0});
    for (size_t elemIdx = 0; elemIdx < numElements; ++elemIdx) {
      const int chunkIndex = (elemIdx / chunkSize) % dimSize;
      const float val =
          std::clamp(std::nearbyint(rawFloat[elemIdx] * invScales[chunkIndex]) +
                         zps[chunkIndex],
                     fMin, fMax);
      const int8_t nibble = static_cast<int8_t>(val) & 0x0F;
      if (elemIdx % 2 == 0) {
        outInt8[elemIdx / 2] |= nibble;
      } else {
        outInt8[elemIdx / 2] |= (nibble << 4);
      }
    }
  } else if (storageBitWidth == 2) {
    const MutableArrayRef<int8_t> outInt8(reinterpret_cast<int8_t*>(outData),
                                          outByteSize);
    llvm::fill(outInt8, int8_t{0});
    for (size_t elemIdx = 0; elemIdx < numElements; ++elemIdx) {
      const int chunkIndex = (elemIdx / chunkSize) % dimSize;
      const float val =
          std::clamp(std::nearbyint(rawFloat[elemIdx] * invScales[chunkIndex]) +
                         zps[chunkIndex],
                     fMin, fMax);
      const int8_t twoBit = static_cast<int8_t>(val) & 0x03;
      const int shift = (elemIdx % 4) * 2;
      outInt8[elemIdx / 4] |= (twoBit << shift);
    }
  } else {
    const MutableArrayRef<int16_t> outInt16(reinterpret_cast<int16_t*>(outData),
                                            numElements);
    for (size_t elemIdx = 0; elemIdx < numElements; ++elemIdx) {
      const int chunkIndex = (elemIdx / chunkSize) % dimSize;
      const float val =
          std::clamp(std::nearbyint(rawFloat[elemIdx] * invScales[chunkIndex]) +
                         zps[chunkIndex],
                     fMin, fMax);
      outInt16[elemIdx] = static_cast<int16_t>(val);
    }
  }

  outConvertedType = resType;
  return DenseResourceElementsAttr::get(resType, newKey,
                                        std::move(rawOutputBlob));
}

static Attribute quantizeResourceAttrPerAxis(
    DenseResourceElementsAttr resourceAttr,
    quant::UniformQuantizedPerAxisType quantizedElementType,
    Type& outConvertedType, bool useLegacySlowQuantize = false) {
  if (useLegacySlowQuantize) {
    return quantizeResourceAttrPerAxisLegacy(resourceAttr, quantizedElementType,
                                             outConvertedType);
  }
  return quantizeResourceAttrPerAxisFast(resourceAttr, quantizedElementType,
                                         outConvertedType);
}

/// Converts a real expressed Attribute to a corresponding Attribute containing
/// quantized storage values assuming the given uniform quantizedElementType and
/// converter.
Attribute quantizeAttrUniform(
    Attribute realValue, quant::UniformQuantizedType quantizedElementType,
    const mlir::quant::ir::UniformQuantizedValueConverter& converter,
    Type& outConvertedType) {
  // Fork to handle different variants of constants supported.
  if (mlir::isa<DenseFPElementsAttr>(realValue)) {
    // Dense tensor or vector constant.
    const auto converted =
        convertDenseFPElementsAttr(mlir::cast<DenseFPElementsAttr>(realValue),
                                   quantizedElementType, converter);
    outConvertedType = converted.getType();
    return converted;
  }
  if (mlir::isa<SparseElementsAttr>(realValue)) {
    // Sparse tensor or vector constant.
    const auto converted =
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
                       Type& outConvertedType) {
  if (const auto uniformQuantized =
          mlir::dyn_cast<quant::UniformQuantizedType>(quantizedElementType)) {
    const mlir::quant::ir::UniformQuantizedValueConverter converter(
        uniformQuantized);
    return quantizeAttrUniform(realValue, uniformQuantized, converter,
                               outConvertedType);
  }
  if (const auto uniformQuantizedPerAxis =
          mlir::dyn_cast<quant::UniformQuantizedPerAxisType>(
              quantizedElementType)) {
    if (const auto resourceAttr =
            mlir::dyn_cast<DenseResourceElementsAttr>(realValue)) {
      return quantizeResourceAttrPerAxis(resourceAttr, uniformQuantizedPerAxis,
                                         outConvertedType);
    }
    mlir::quant::ir::UniformQuantizedPerAxisValueConverter converter(
        uniformQuantizedPerAxis);
    const auto converted = converter.convert(realValue);
    if (converted) {
      outConvertedType = converted.getType();
    }
    return converted;
  }
  return nullptr;
}

}  // namespace quantfork
}  // namespace mlir
