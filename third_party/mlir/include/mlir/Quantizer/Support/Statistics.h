//===- Statistics.h - Collects statistics over tensors ----------*- C++ -*-===//
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
//
// This file defines adapters for extracting various (per layer and per axis)
// statistics over tensors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_QUANTIZER_SUPPORT_STATISTICS_H
#define MLIR_QUANTIZER_SUPPORT_STATISTICS_H

#include "mlir/IR/Attributes.h"

namespace mlir {
namespace quantizer {

/// Statistics about a tensor axis (or the whole tensor).
struct TensorAxisStatistics {
  int64_t sampleSize = 0;
  double minValue = 0;
  double maxValue = 0;
  double mean = 0;
  double variance = 0;

  TensorAxisStatistics() {}
  TensorAxisStatistics(int64_t sampleSize, double minValue, double maxValue,
                       double mean, double variance)
      : sampleSize(sampleSize), minValue(minValue), maxValue(maxValue),
        mean(mean), variance(variance) {}
  void clear() { *this = TensorAxisStatistics(); }
};

/// Base class for querying statistics about a tensor.
class AbstractTensorStatistics {
public:
  virtual ~AbstractTensorStatistics() = default;

  /// Gets statistics across the whole tensor.
  /// Returns true if statistics are valid and were populated.
  virtual bool get(TensorAxisStatistics &stats) const { return false; }

  /// Whether this instance supports querying per axis statistics. If true,
  /// then getForAxis(...) can be used.
  virtual bool supportsPerAxis() const { return false; }

  /// Count of axes supported in a per-axis query.
  virtual unsigned getAxisCount() const { return 0; }

  /// Gets statistics for a specific axis (0..getAxisCount() - 1).
  /// Returns true if statistics are valid and were populated.
  virtual bool getForAxis(unsigned axis, TensorAxisStatistics &stats) const {
    return false;
  }
};

/// Wraps an MLIR Attribute and returns statistics about it.
/// It is expected that the attribute be one of:
///   FloatAttr (scalar)
///   DenseFPElementsAttr
///   OpaqueElementsAttr (with Float based type)
///   SparseElementAttr  (with Float based type)
class AttributeTensorStatistics : public AbstractTensorStatistics {
public:
  AttributeTensorStatistics(Attribute attr) : attr(attr) {}

  bool get(TensorAxisStatistics &stats) const override;

  // TODO: Implement per-axis.

private:
  Attribute attr;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const TensorAxisStatistics &stats);

} // end namespace quantizer
} // end namespace mlir

#endif // MLIR_QUANTIZER_SUPPORT_STATISTICS_H
