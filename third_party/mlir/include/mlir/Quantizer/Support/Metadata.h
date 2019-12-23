//===- Metadata.h - Top level types and metadata ----------------*- C++ -*-===//
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
// This file contains top level types needed to construct constraint graphs,
// including context/allocator support and concrete metadata structs for
// different quantization schemes (which must be attached to anchor nodes).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_QUANTIZER_SUPPORT_METADATA_H
#define MLIR_QUANTIZER_SUPPORT_METADATA_H

#include <limits>

#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Quantizer/Support/Rules.h"
#include "llvm/ADT/SmallBitVector.h"

namespace mlir {
namespace quantizer {

class SolverContext {
public:
  SolverContext(MLIRContext &mlirContext) : mlirContext(mlirContext) {}

  MLIRContext &getMlirContext() { return mlirContext; }

  llvm::BumpPtrAllocator &getAllocator() { return allocator; }

  // Optional path to write a debug DOT file for the CAG.
  StringRef getDebugCAGDotPath() const { return debugCAGDotPath; }
  void setDebugCAGDotPath(StringRef p) { debugCAGDotPath = p; }

private:
  MLIRContext &mlirContext;
  llvm::BumpPtrAllocator allocator;
  std::string debugCAGDotPath;
};

/// Candidate for a quantized type conversion.
struct CandidateQuantizedType {
  // Note that scheme encodes more than just the target type: it also encodes
  // additional constraints.
  enum class Scheme {
    // Uses aggregate range information for all nodes in the cluster to
    // solve for uniform scale and zero point.
    UniformPerLayer,
    // Uses aggregate per-axis range information for all nodes in the cluster
    // to solve for per-axis uniform scale and zero point.
    UniformPerAxisFixedPoint,
    // Uses the |explicitScaleZeroPoint| to set the scale (and zero point = 0)
    // for the uniform type. This typically overrides all other constraints
    // and is used for wide accumulator types (i.e. i32 bias vectors).
    UniformExplicitFixedPointScale,
  };
  unsigned ordinal;
  quant::AnyQuantizedType quantizedType;
  Scheme scheme;
};

struct CAGUniformMetadata {
  /// Default salience for facts that are derived from data either statically
  /// discovered in the computation or observed from an outside source.
  static constexpr int SalienceDefault = 0;

  /// Highest salience level for facts derived from overrides provided
  /// explicitly.
  static constexpr int SalienceForced = 100;

  /// Salience for facts derived from constraints in how the math is
  /// expressed which must be satisfied.
  static constexpr int SalienceRequired = 200;

  /// The range that the scheme must represent in order to accommodate the
  /// underlying data.
  ExpandingMinMaxFact requiredRange;

  /// Bool vector of scheme ordinals that are disabled.
  llvm::SmallBitVector disabledCandidateTypes;

  /// If set, then a solution has converged for the given per-layer scheme.
  quant::QuantizedType selectedType;

  /// Optional scale and zero point to be used by types which solve via the
  /// UniformExplicitFixedPointScale scheme.
  DiscreteScaleZeroPointFact explicitScaleZeroPoint;

  /// Prints a summary of the metadata suitable for display in a graph label.
  void printSummary(raw_ostream &os) const;
};

} // end namespace quantizer
} // end namespace mlir

#endif // MLIR_QUANTIZER_SUPPORT_METADATA_H
