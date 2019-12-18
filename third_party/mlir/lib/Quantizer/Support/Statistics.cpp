//===- Statistics.cpp - Collects statistics over tensors ------------------===//
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

namespace mlir {
namespace quantizer {

raw_ostream &operator<<(raw_ostream &os, const TensorAxisStatistics &stats) {
  os << "STATS[sampleSize=" << stats.sampleSize << ", min=" << stats.minValue
     << ", maxValue=" << stats.maxValue << ", mean=" << stats.mean
     << ", variance=" << stats.variance << "]";
  return os;
}

} // end namespace quantizer
} // end namespace mlir
