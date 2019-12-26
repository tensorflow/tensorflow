//===- Statistics.h - Collects statistics over tensors ----------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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

raw_ostream &operator<<(raw_ostream &os, const TensorAxisStatistics &stats);

} // end namespace quantizer
} // end namespace mlir

#endif // MLIR_QUANTIZER_SUPPORT_STATISTICS_H
