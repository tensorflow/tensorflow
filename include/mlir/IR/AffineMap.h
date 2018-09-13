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

class MLIRContext;
class AffineExpr;

/// A multi-dimensional affine map
/// Affine map's are immutable like Type's, and they are uniqued.
/// Eg: (d0, d1) -> (d0/128, d0 mod 128, d1)
/// The names used (d0, d1) don't matter - it's the mathematical function that
/// is unique to this affine map.
class AffineMap {
public:
  static AffineMap *get(unsigned dimCount, unsigned symbolCount,
                        ArrayRef<AffineExpr *> results,
                        ArrayRef<AffineExpr *> rangeSizes,
                        MLIRContext *context);

  /// Returns a single constant result affine map.
  static AffineMap *getConstantMap(int64_t val, MLIRContext *context);

  /// Returns true if the co-domain (or more loosely speaking, range) of this
  /// map is bounded. Bounded affine maps have a size (extent) for each of
  /// their range dimensions (more accurately co-domain dimensions).
  bool isBounded() const { return rangeSizes != nullptr; }

  /// Returns true if this affine map is an identity affine map.
  /// An identity affine map corresponds to an identity affine function on the
  /// dimensional identifiers.
  bool isIdentity() const;

  /// Returns true if this affine map is a single result constant function.
  bool isSingleConstant() const;

  /// Returns the constant result of this map. This methods asserts that the map
  /// has a single constant result.
  int64_t getSingleConstantResult() const;

  // Prints affine map to 'os'.
  void print(raw_ostream &os) const;
  void dump() const;

  unsigned getNumDims() const { return numDims; }
  unsigned getNumSymbols() const { return numSymbols; }
  unsigned getNumResults() const { return numResults; }
  unsigned getNumInputs() const { return numDims + numSymbols; }

  ArrayRef<AffineExpr *> getResults() const {
    return ArrayRef<AffineExpr *>(results, numResults);
  }

  AffineExpr *getResult(unsigned idx) const { return results[idx]; }

  ArrayRef<AffineExpr *> getRangeSizes() const {
    return rangeSizes ? ArrayRef<AffineExpr *>(rangeSizes, numResults)
                      : ArrayRef<AffineExpr *>();
  }

private:
  AffineMap(unsigned numDims, unsigned numSymbols, unsigned numResults,
            AffineExpr *const *results, AffineExpr *const *rangeSizes);

  AffineMap(const AffineMap &) = delete;
  void operator=(const AffineMap &) = delete;

  const unsigned numDims;
  const unsigned numSymbols;
  const unsigned numResults;

  /// The affine expressions for this (multi-dimensional) map.
  /// TODO: use trailing objects for this.
  AffineExpr *const *const results;

  /// The extents along each of the range dimensions if the map is bounded,
  /// nullptr otherwise.
  AffineExpr *const *const rangeSizes;
};

}  // end namespace mlir

#endif  // MLIR_IR_AFFINE_MAP_H
