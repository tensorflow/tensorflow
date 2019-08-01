//===- UniformConstraints.h - Constraints for uniform quant -----*- C++ -*-===//
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
// This file defines a builder that lets you attach constraints necessary to
// perform a variety of uniform quantization conversions to CAG anchors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_QUANTIZER_SUPPORT_UNIFORMCONSTRAINTS_H
#define MLIR_QUANTIZER_SUPPORT_UNIFORMCONSTRAINTS_H

#include "mlir/Quantizer/Support/Statistics.h"

namespace mlir {
namespace quantizer {

class CAGAnchorNode;
class CAGSlice;

/// Factory methods for adding CAG constraints of various kinds suitable
/// for solving for uniform quantization.
class UniformConstraintsBuilder {
public:
  UniformConstraintsBuilder(CAGSlice &slice) : slice(slice) {}

  /// Adds a coupling constraint between two nodes, effectively treating
  /// them as a hard identity relationship.
  void coupleAnchors(CAGAnchorNode *a, CAGAnchorNode *b);

  /// Applies statistics constraints to the given anchor, such that the solver
  /// ensures that the statistics are representable by chosen types.
  void applyStats(CAGAnchorNode *a, TensorAxisStatistics stats);

  /// Applies a constraint to a node which allows solutions that do not extend
  /// beyond given min/max bounds (this is a hint that the tensor will not
  /// take values outside of these bounds). If either minValue or maxValue is
  /// NAN, then that side is considered open.
  void clamp(CAGAnchorNode *a, APFloat minValue, APFloat maxValue);

  /// Propagates an explicit scale from an anchor that may have a uniform
  /// |selectedType| to the |explicitScaleZeroPoint| field of the to node.
  /// This is typically used with a to node that has a candidate quantized
  /// type of |UniformExplicitFixedPointScale|, indicating that it can be
  /// an arbitrary (signed) type that is expected to share the same scale
  /// as the originating node.
  void propagateExplicitScale(CAGAnchorNode *from, CAGAnchorNode *to);

private:
  CAGSlice &slice;
};

} // namespace quantizer
} // namespace mlir

#endif // MLIR_QUANTIZER_SUPPORT_UNIFORMCONSTRAINTS_H
