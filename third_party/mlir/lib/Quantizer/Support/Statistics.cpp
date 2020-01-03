//===- Statistics.cpp - Collects statistics over tensors ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Quantizer/Support/Statistics.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::quantizer;

//===----------------------------------------------------------------------===//
// AttributeTensorStatistics implementation
//===----------------------------------------------------------------------===//

static void collectElementsStatisticsDim(ElementsAttr attr,
                                         unsigned numElements,
                                         ArrayRef<int64_t> shape,
                                         SmallVectorImpl<uint64_t> &indices,
                                         uint64_t dim,
                                         TensorAxisStatistics &statistics) {
  // Recursive terminating condition.
  if (dim >= shape.size())
    return;

  if (dim < (shape.size() - 1)) {
    // Recurse past dim.
    for (uint64_t i = 0, s = shape[dim]; i < s; ++i) {
      indices[dim] = i;
      collectElementsStatisticsDim(attr, numElements, shape, indices, dim + 1,
                                   statistics);
    }
    return;
  }

  // Collection dim.
  for (uint64_t i = 0, s = shape[dim]; i < s; ++i) {
    indices[dim] = i;
    double value = attr.getValue<FloatAttr>(indices).getValueAsDouble();
    statistics.minValue = std::min(statistics.minValue, value);
    statistics.maxValue = std::max(statistics.maxValue, value);
    statistics.mean += value / numElements;
    // TODO: Calculate a running variance.
  }
}

static bool getElementsStatistics(ElementsAttr attr,
                                  TensorAxisStatistics &statistics) {
  statistics.clear();
  statistics.minValue = std::numeric_limits<double>::infinity();
  statistics.maxValue = -std::numeric_limits<double>::infinity();

  ShapedType sType = attr.getType();
  if (!sType.hasStaticShape())
    return false;
  Type elementTy = sType.getElementType();
  if (!elementTy.isa<FloatType>())
    return false;

  SmallVector<uint64_t, 4> indices;
  indices.resize(sType.getRank());
  ArrayRef<int64_t> shape = sType.getShape();

  auto numElements = sType.getNumElements();
  collectElementsStatisticsDim(attr, numElements, shape, indices, 0,
                               statistics);
  statistics.sampleSize = numElements;

  return true;
}

bool AttributeTensorStatistics::get(TensorAxisStatistics &stats) const {
  if (FloatAttr floatAttr = attr.dyn_cast<FloatAttr>()) {
    double value = floatAttr.getValueAsDouble();
    stats = TensorAxisStatistics(1, value, value, value, 0);
    return true;
  } else if (auto eltAttr = attr.dyn_cast<ElementsAttr>()) {
    return getElementsStatistics(eltAttr, stats);
  }
  return false;
}

raw_ostream &mlir::quantizer::operator<<(raw_ostream &os,
                                         const TensorAxisStatistics &stats) {
  os << "STATS[sampleSize=" << stats.sampleSize << ", min=" << stats.minValue
     << ", maxValue=" << stats.maxValue << ", mean=" << stats.mean
     << ", variance=" << stats.variance << "]";
  return os;
}
