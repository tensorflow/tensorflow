//===- AffineMap.h - MLIR Affine Map Class ----------------------*- C++ -*-===//
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
// Affine maps are mathematical functions which map a list of dimension
// identifiers and symbols, to multidimensional affine expressions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_AFFINE_MAP_H
#define MLIR_IR_AFFINE_MAP_H

#include <vector>

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {

namespace detail {

class AffineExpr;

} // namespace detail
template <typename T> class AffineExprBaseRef;
using AffineExprRef = AffineExprBaseRef<detail::AffineExpr>;
class Attribute;
class MLIRContext;

/// A multi-dimensional affine map
/// Affine map's are immutable like Type's, and they are uniqued.
/// Eg: (d0, d1) -> (d0/128, d0 mod 128, d1)
/// The names used (d0, d1) don't matter - it's the mathematical function that
/// is unique to this affine map.
class AffineMap {
public:
  static AffineMap *get(unsigned dimCount, unsigned symbolCount,
                        ArrayRef<AffineExprRef> results,
                        ArrayRef<AffineExprRef> rangeSizes,
                        MLIRContext *context);

  /// Returns a single constant result affine map.
  static AffineMap *getConstantMap(int64_t val, MLIRContext *context);

  /// Returns true if the co-domain (or more loosely speaking, range) of this
  /// map is bounded. Bounded affine maps have a size (extent) for each of
  /// their range dimensions (more accurately co-domain dimensions).
  bool isBounded() { return !rangeSizes.empty(); }

  /// Returns true if this affine map is an identity affine map.
  /// An identity affine map corresponds to an identity affine function on the
  /// dimensional identifiers.
  bool isIdentity();

  /// Returns true if this affine map is a single result constant function.
  bool isSingleConstant();

  /// Returns the constant result of this map. This methods asserts that the map
  /// has a single constant result.
  int64_t getSingleConstantResult();

  // Prints affine map to 'os'.
  void print(raw_ostream &os);
  void dump();

  unsigned getNumDims() { return numDims; }
  unsigned getNumSymbols() { return numSymbols; }
  unsigned getNumResults() { return numResults; }
  unsigned getNumInputs() { return numDims + numSymbols; }

  ArrayRef<AffineExprRef> getResults() { return results; }

  AffineExprRef getResult(unsigned idx);

  ArrayRef<AffineExprRef> getRangeSizes() { return rangeSizes; }

  /// Folds the results of the application of an affine map on the provided
  /// operands to a constant if possible. Returns false if the folding happens,
  /// true otherwise.
  bool constantFold(ArrayRef<Attribute *> operandConstants,
                    SmallVectorImpl<Attribute *> &results);

private:
  AffineMap(unsigned numDims, unsigned numSymbols, unsigned numResults,
            ArrayRef<AffineExprRef> results,
            ArrayRef<AffineExprRef> rangeSizes);

  AffineMap(const AffineMap &) = delete;
  void operator=(const AffineMap &) = delete;

  unsigned numDims;
  unsigned numSymbols;
  unsigned numResults;

  /// The affine expressions for this (multi-dimensional) map.
  /// TODO: use trailing objects for this.
  ArrayRef<AffineExprRef> results;

  /// The extents along each of the range dimensions if the map is bounded,
  /// nullptr otherwise.
  ArrayRef<AffineExprRef> rangeSizes;
};

} // end namespace mlir

#endif // MLIR_IR_AFFINE_MAP_H
