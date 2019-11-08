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

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace mlir {

namespace detail {
struct AffineMapStorage;
} // end namespace detail

class AffineExpr;
class Attribute;
struct LogicalResult;
class MLIRContext;

/// A multi-dimensional affine map
/// Affine map's are immutable like Type's, and they are uniqued.
/// Eg: (d0, d1) -> (d0/128, d0 mod 128, d1)
/// The names used (d0, d1) don't matter - it's the mathematical function that
/// is unique to this affine map.
class AffineMap {
public:
  using ImplType = detail::AffineMapStorage;

  AffineMap() : map(nullptr) {}
  explicit AffineMap(ImplType *map) : map(map) {}
  AffineMap(const AffineMap &other) : map(other.map) {}
  AffineMap &operator=(const AffineMap &other) = default;

  /// Returns a zero result affine map with no dimensions or symbols: () -> ().
  static AffineMap get(MLIRContext *context);

  static AffineMap get(unsigned dimCount, unsigned symbolCount,
                       ArrayRef<AffineExpr> results);

  /// Returns a single constant result affine map.
  static AffineMap getConstantMap(int64_t val, MLIRContext *context);

  /// Returns an AffineMap with 'numDims' identity result dim exprs.
  static AffineMap getMultiDimIdentityMap(unsigned numDims,
                                          MLIRContext *context);

  MLIRContext *getContext() const;

  explicit operator bool() { return map != nullptr; }
  bool operator==(AffineMap other) const { return other.map == map; }
  bool operator!=(AffineMap other) const { return !(other.map == map); }

  /// Returns true if this affine map is an identity affine map.
  /// An identity affine map corresponds to an identity affine function on the
  /// dimensional identifiers.
  bool isIdentity() const;

  /// Returns true if this affine map is an empty map, i.e., () -> ().
  bool isEmpty() const;

  /// Returns true if this affine map is a single result constant function.
  bool isSingleConstant() const;

  /// Returns the constant result of this map. This methods asserts that the map
  /// has a single constant result.
  int64_t getSingleConstantResult() const;

  // Prints affine map to 'os'.
  void print(raw_ostream &os) const;
  void dump() const;

  unsigned getNumDims() const;
  unsigned getNumSymbols() const;
  unsigned getNumResults() const;
  unsigned getNumInputs() const;

  ArrayRef<AffineExpr> getResults() const;
  AffineExpr getResult(unsigned idx) const;

  /// Walk all of the AffineExpr's in this mapping. Each node in an expression
  /// tree is visited in postorder.
  void walkExprs(std::function<void(AffineExpr)> callback) const;

  /// This method substitutes any uses of dimensions and symbols (e.g.
  /// dim#0 with dimReplacements[0]) in subexpressions and returns the modified
  /// expression mapping.  Because this can be used to eliminate dims and
  /// symbols, the client needs to specify the number of dims and symbols in
  /// the result.  The returned map always has the same number of results.
  AffineMap replaceDimsAndSymbols(ArrayRef<AffineExpr> dimReplacements,
                                  ArrayRef<AffineExpr> symReplacements,
                                  unsigned numResultDims,
                                  unsigned numResultSyms);

  /// Folds the results of the application of an affine map on the provided
  /// operands to a constant if possible.
  LogicalResult constantFold(ArrayRef<Attribute> operandConstants,
                             SmallVectorImpl<Attribute> &results) const;

  /// Returns the AffineMap resulting from composing `this` with `map`.
  /// The resulting AffineMap has as many AffineDimExpr as `map` and as many
  /// AffineSymbolExpr as the concatenation of `this` and `map` (in which case
  /// the symbols of `this` map come first).
  ///
  /// Prerequisites:
  /// The maps are composable, i.e. that the number of AffineDimExpr of `this`
  /// matches the number of results of `map`.
  ///
  /// Example:
  ///   map1: `(d0, d1)[s0, s1] -> (d0 + 1 + s1, d1 - 1 - s0)`
  ///   map2: `(d0)[s0] -> (d0 + s0, d0 - s0)`
  ///   map1.compose(map2):
  ///     `(d0)[s0, s1, s2] -> (d0 + s1 + s2 + 1, d0 - s0 - s2 - 1)`
  AffineMap compose(AffineMap map);

  /// Returns true if the AffineMap represents a subset (i.e. a projection) of a
  /// symbol-less permutation map.
  bool isProjectedPermutation();

  /// Returns true if the AffineMap represents a symbol-less permutation map.
  bool isPermutation();

  /// Returns the map consisting of the `resultPos` subset.
  AffineMap getSubMap(ArrayRef<unsigned> resultPos);

  friend ::llvm::hash_code hash_value(AffineMap arg);

private:
  ImplType *map;

  static AffineMap getImpl(unsigned dimCount, unsigned symbolCount,
                           ArrayRef<AffineExpr> results, MLIRContext *context);
};

// Make AffineExpr hashable.
inline ::llvm::hash_code hash_value(AffineMap arg) {
  return ::llvm::hash_value(arg.map);
}

/// Simplify an affine map by simplifying its underlying AffineExpr results.
AffineMap simplifyAffineMap(AffineMap map);

/// Returns a map of codomain to domain dimensions such that the first codomain
/// dimension for a particular domain dimension is selected.
/// Returns an empty map if the input map is empty or if `map` is not invertible
/// (i.e. `map` does not contain a subset that is a permutation of full domain
/// rank).
///
/// Prerequisites:
///   1. `map` has no symbols.
///
/// Example 1:
///
/// ```{.mlir}
///    (d0, d1, d2) -> (d1, d1, d0, d2, d1, d2, d1, d0)
///                      0       2   3
/// ```
///
/// returns:
///
/// ```{.mlir}
///    (d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d0, d3)
/// ```
///
/// Example 2:
///
/// ```{.mlir}
///    (d0, d1, d2) -> (d1, d0 + d1, d0, d2, d1, d2, d1, d0)
///                      0            2   3
/// ```
///
/// returns:
///
/// ```{.mlir}
///    (d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d0, d3)
/// ```
AffineMap inversePermutation(AffineMap map);

/// Concatenates a list of `maps` into a single AffineMap, stepping over
/// potentially empty maps. Assumes each of the underlying map has 0 symbols.
/// The resulting map has a number of dims equal to the max of `maps`' dims and
/// the concatenated results as its results.
/// Returns an empty map if all input `maps` are empty.
///
/// Example:
/// When applied to the following list of 3 affine maps,
///
/// ```{.mlir}
///    {
///      (i, j, k) -> (i, k),
///      (i, j, k) -> (k, j),
///      (i, j, k) -> (i, j)
///    }
/// ```
///
/// Returns the map:
///
/// ```{.mlir}
///     (i, j, k) -> (i, k, k, j, i, j)
/// ```
AffineMap concatAffineMaps(llvm::ArrayRef<AffineMap> maps);

inline raw_ostream &operator<<(raw_ostream &os, AffineMap map) {
  map.print(os);
  return os;
}
} // end namespace mlir

namespace llvm {

// AffineExpr hash just like pointers
template <> struct DenseMapInfo<mlir::AffineMap> {
  static mlir::AffineMap getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::AffineMap(static_cast<mlir::AffineMap::ImplType *>(pointer));
  }
  static mlir::AffineMap getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::AffineMap(static_cast<mlir::AffineMap::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::AffineMap val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::AffineMap LHS, mlir::AffineMap RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // MLIR_IR_AFFINE_MAP_H
