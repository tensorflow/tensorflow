//===- HyperRectangularSet.h - MLIR HyperRectangle Class --------*- C++ -*-===//
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
// A symbolic hyper-rectangular set of integer points for analysis.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_HYPER_RECTANGULAR_SET_H
#define MLIR_ANALYSIS_HYPER_RECTANGULAR_SET_H

#include <vector>

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IntegerSet.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"

namespace mlir {

class AffineApplyOp;
class AffineBound;
class AffineCondition;
class AffineMap;
class IntegerSet;
class MLIRContext;
class MutableIntegerSet;
class FlatAffineConstraints;
class HyperRectangleList;
class Value;

/// A list of affine bounds.
// Not using a MutableAffineMap here since numSymbols is the same as the
// containing HyperRectangularSet's numSymbols, and its numDims is 0.
using AffineBoundExprList = SmallVector<AffineExpr, 4>;

/// A HyperRectangularSet is a symbolic set of integer points contained in a
/// hyper-rectangular space. It supports set manipulation operations
/// and other queries to aid analysis of multi-dimensional integer sets that can
/// be represented as integer points inside a symbolic hyper-rectangle, i.e.,
/// an interval is associated with each dimension, and the lower and upper
/// bounds of each interval are symbolic affine expressions. The bounds on
/// a 'dimension' can't depend on other 'dimensions'. The fields of this set are
/// always maintained in an irredundant form (no redundant bounds), and the
/// bounds are simplified under its context field.
//
//  Example: dims: (d0, d1), symbols: (M, N)
//   0          <= d0 <=  511
//   max(128,M) <= d1 <=  min(N-1,256)
//
// Symbols here aren't necessarily associated with Function's symbols; they
// could also correspond to outer loop IVs for example or anything abstract. The
// binding to SSA values for dimensions/symbols is optional, and these are in an
// abstract integer domain. As an example, to describe data accessed in a tile
// surrounded by loop i0, i1, the following set symbolic in i0, i1 is a
// hyper-rectangular set:
//
//  128*i  <= d0 <=  min(128*i0 + 127, N-1)
//  128*i  <= d1 <=  min(128*i1 + 127, N-1)
//
// The context field specifies constraints on the symbols, and the set is always
// kept in a form simplified under 'context', i.e., information implied by
// context is used to simplify bounds. For eg., if the context includes (N >=
// 0), a bound such as d0 >= max(0, N) will never arise. This would be
// simplified to d0 >= N at construction time or when the context is updated.
// As another example, if N%128 = 0, M <= N-1 floordiv 128 is specified, we will
// never have a bound such as d0 <= min(128*M + 127, N-1); this would be
// simplified to d0 <= 128*M + 127 (since 128*M + 127 is always <= N-1 under
// such circumstances). In the context of code generation, such simplification
// leads to code that explicitly scans "full" tiles / no boundary case and with
// lower control overhead.
//
class HyperRectangularSet
    : public llvm::ilist_node_with_parent<HyperRectangularSet,
                                          HyperRectangleList> {
public:
  /// Construct a hyper-rectangular set from FlatAffineConstraints if possible;
  /// returns nullptr if it cannot.
  static std::unique_ptr<HyperRectangularSet>
  getFromFlatAffineConstraints(const FlatAffineConstraints &cst);

  HyperRectangularSet(unsigned numDims, unsigned numSymbols,
                      ArrayRef<ArrayRef<AffineExpr>> lbs,
                      ArrayRef<ArrayRef<AffineExpr>> ubs, MLIRContext *context,
                      IntegerSet symbolContext = IntegerSet());

  unsigned getNumDims() const { return numDims; }
  unsigned getNumSymbols() const { return numSymbols; }

  ArrayRef<AffineBoundExprList> getLowerBounds() const { return lowerBounds; }
  ArrayRef<AffineBoundExprList> getUpperBounds() const { return upperBounds; }

  AffineBoundExprList &getLowerBound(unsigned idx) { return lowerBounds[idx]; }
  AffineBoundExprList &getUpperBound(unsigned idx) { return upperBounds[idx]; }

  const AffineBoundExprList &getLowerBound(unsigned idx) const {
    return lowerBounds[idx];
  }
  const AffineBoundExprList &getUpperBound(unsigned idx) const {
    return upperBounds[idx];
  }

  /// Intersects 'rhs' with this set.
  void intersect(const HyperRectangularSet &rhs);

  /// Performs a union of 'rhs' with this set.
  void unionize(const HyperRectangularSet &rhs);

  /// Project out num dimensions starting from 'idx'. This is equivalent to
  /// taking an image of this set on the remaining dimensions.
  void projectOut(unsigned idx, unsigned num);

  /// Returns true if the set has no integer points in it.
  bool empty() const;

  /// Add a lower bound expression to dimension position 'idx'.
  void addLowerBoundExpr(unsigned idx, AffineExpr expr);

  /// Add an upper bound expression to dimension position 'idx'.
  void addUpperBoundExpr(unsigned idx, AffineExpr expr);

  /// Clear this set's context, i.e., make it the universal set.
  void clearContext() { context.clear(); }

  void print(raw_ostream &os) const;
  void dump() const;

private:
  /// Simplify this set under the symbolic context 'context'.
  void simplifyUnderContext() {}

  /// The lower bound along any dimension is a max of several pure
  /// symbolic/constant affine expressions. A bound cannot be mutated from
  /// outside the class, it has to be to be updated through
  /// addLowerBoundExpr/addUpperBoundExpr.
  std::vector<AffineBoundExprList> lowerBounds;
  // Each upper bound is a min of several pure symbolic/constant affine
  // expressions.
  std::vector<AffineBoundExprList> upperBounds;

  Optional<SmallVector<Value *, 8>> dims = None;
  Optional<SmallVector<Value *, 4>> symbols = None;

  /// Number of real dimensions.
  unsigned numDims;

  /// Number of symbols (unknown but constant)
  unsigned numSymbols;

  // Constraints on the symbols. The representation of the set is kept
  // simplified under this context.
  MutableIntegerSet context;
};

//===--------------------------------------------------------------------===//
// Out of place operations.
//===--------------------------------------------------------------------===//

static std::unique_ptr<HyperRectangularSet>
intersection(const HyperRectangularSet &lhs, const HyperRectangularSet &rhs);

static std::unique_ptr<HyperRectangleList>
intersection(const HyperRectangleList &lhs, const HyperRectangleList &rhs);

/// Performs a union of 'lhs' and 'rhs'.
static std::unique_ptr<HyperRectangleList>
unionize(const HyperRectangularSet &lhs, const HyperRectangularSet &rhs);
static std::unique_ptr<HyperRectangleList>
unionize(const HyperRectangleList &lhs, const HyperRectangleList &rhs);

/// Subtract 'rhs' from this lhs and return the result.
static std::unique_ptr<HyperRectangleList>
difference(const HyperRectangularSet &lhs, const HyperRectangularSet &rhs);
static std::unique_ptr<HyperRectangleList>
difference(const HyperRectangleList &lhs, const HyperRectangleList &rhs);

/// Project out num dimensions starting from 'idx'. This is equivalent to
/// taking an image of this set on the remaining dimensions.
static std::unique_ptr<HyperRectangularSet>
projectOut(const HyperRectangularSet &set, unsigned idx, unsigned num);

} // namespace mlir

namespace llvm {

template <> struct ilist_traits<::mlir::HyperRectangularSet> {
  using HyperRectangularSet = ::mlir::HyperRectangularSet;
  using set_iterator = simple_ilist<HyperRectangularSet>::iterator;

  static void deleteNode(HyperRectangularSet *set) { delete set; }

  void addNodeToList(HyperRectangularSet *set);
  void removeNodeFromList(HyperRectangularSet *set);
  void transferNodesFromList(ilist_traits<HyperRectangularSet> &otherList,
                             set_iterator first, set_iterator last);

private:
  mlir::HyperRectangleList *getContainingBlock();
};

} // namespace llvm

namespace mlir {

/// A list of hyper-rectangular sets lying in the same space of dimensional
/// and symbolic identifiers. The individual set elements are always kept
/// disjoint (re-evaluate choice) and minimal, i.e., the union of any subset of
/// the contained hyperrectangles can't be coalesced into a single
/// hyper-rectangle.
class HyperRectangleList {
public:
  /// Construct a constraint system reserving memory for the specified number of
  /// constraints and identifiers.
  explicit HyperRectangleList(const FlatAffineConstraints &cst);

  HyperRectangleList(unsigned numDims, unsigned numSymbols,
                     ArrayRef<std::unique_ptr<HyperRectangularSet>> sets);

  unsigned getNumDims() const { return numDims; }
  unsigned getNumSymbols() const { return numSymbols; }

  // In-place operations.

  /// Intersects a hyper rectangular set list 'rhs' with this set.
  void intersect(const HyperRectangleList &rhs);

  /// Intersects 'rhs' with this set.
  void intersect(const HyperRectangularSet &rhs);

  /// Performs a union of 'rhs' with this set.
  void unionize(const HyperRectangleList &rhs);

  /// Performs a union of 'rhs' with this set.
  void unionize(const HyperRectangularSet &rhs);

  /// Project out num dimensions starting from 'idx'. This is equivalent to
  /// taking an image of this set on the remaining dimensions.
  void projectOut(unsigned idx, unsigned num);

  /// Returns true if all the sets are empty.
  bool empty() const;

  //===--------------------------------------------------------------------===//
  // Hyper-rectangular set list management.
  //===--------------------------------------------------------------------===//

  /// These are for the list of hyper-rectangular set elements.
  using HyperRectangleListTy = ::llvm::iplist<HyperRectangularSet>;
  HyperRectangleListTy &getRectangles() { return hyperRectangles; }

  // Iteration over the instructions in the block.
  using const_iterator = HyperRectangleListTy::const_iterator;

  const_iterator begin() const { return hyperRectangles.begin(); }
  const_iterator end() const { return hyperRectangles.end(); }

  bool listEmpty() const { return hyperRectangles.empty(); }

  void addSet(std::unique_ptr<HyperRectangularSet> set) {
    set->clearContext();
    hyperRectangles.push_back(set.release());
  }

private:
  // Mutable versions of the iterators are private.
  using iterator = HyperRectangleListTy::iterator;
  iterator begin() { return hyperRectangles.begin(); }
  iterator end() { return hyperRectangles.end(); }

  /// Simplify under the symbolic context 'context'.
  void simplifyUnderContext() {}

  /// Number of identifiers corresponding to real dimensions.
  unsigned numDims;

  /// Number of identifiers corresponding to symbols (unknown but constant)
  unsigned numSymbols;

  /// The list of hyper-rectangular sets contained.
  HyperRectangleListTy hyperRectangles;

  // Constraints on the symbols. The representation of the set is kept
  // simplified under this context.
  MutableIntegerSet context;
};

// Out of place operations.

// Return a bounding box of this list of hyper-rectangles. This is notionally
// equivanelt to a rectangular/convex hull.
std::unique_ptr<HyperRectangularSet> boundingBox();

/// Intersects and returns the result.
static std::unique_ptr<HyperRectangleList>
intersection(const HyperRectangleList &lhs, const HyperRectangleList &rhs);

/// Performs a union and returns the result.
static std::unique_ptr<HyperRectangleList>
unionize(const HyperRectangleList &lhs, const HyperRectangleList &rhs);

/// Subtracts 'rhs' from this lhs and return the result.
static std::unique_ptr<HyperRectangleList>
difference(const HyperRectangleList &lhs, const HyperRectangleList &rhs);

} // end namespace mlir.

#endif // MLIR_ANALYSIS_HYPER_RECTANGULAR_SET_H
