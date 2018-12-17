//===- Utils.h - General analysis utilities ---------------------*- C++ -*-===//
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
// This header file defines prototypes for various transformation utilities for
// memref's and non-loop IR structures. These are not passes by themselves but
// are used either by passes, optimization sequences, or in turn by other
// transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_UTILS_H
#define MLIR_ANALYSIS_UTILS_H

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir {

class FlatAffineConstraints;
class ForStmt;
class MLValue;
class MemRefAccess;
class OperationStmt;
class Statement;

/// Returns true if statement 'a' dominates statement b.
bool dominates(const Statement &a, const Statement &b);

/// Returns true if statement 'a' properly dominates statement b.
bool properlyDominates(const Statement &a, const Statement &b);

/// Populates 'loops' with IVs of the loops surrounding 'stmt' ordered from
/// the outermost 'for' statement to the innermost one.
void getLoopIVs(const Statement &stmt, SmallVectorImpl<ForStmt *> *loops);

/// A region of a memref's data space; this is typically constructed by
/// analyzing load/store op's on this memref and the index space of loops
/// surrounding such op's.
// For example, the memref region for a load operation at loop depth = 1:
//
//    for %i = 0 to 32 {
//      for %ii = %i to (d0) -> (d0 + 8) (%i) {
//        load %A[%ii]
//      }
//    }
//
// Region:  {memref = %A, write = false, {%i <= m0 <= %i + 7} }
// The last field is a 2-d FlatAffineConstraints symbolic in %i.
//
struct MemRefRegion {
  FlatAffineConstraints *getConstraints() { return &cst; }
  const FlatAffineConstraints *getConstraints() const { return &cst; }
  bool isWrite() const { return write; }
  void setWrite(bool flag) { write = flag; }

  /// Returns a constant upper bound on the number of elements in this region if
  /// bounded by a known constant, None otherwise. The 'shape' vector is set to
  /// the corresponding dimension-wise bounds major to minor. We use int64_t
  /// instead of uint64_t since index types can be at most int64_t.
  Optional<int64_t> getBoundingConstantSizeAndShape(
      SmallVectorImpl<int> *shape = nullptr,
      std::vector<SmallVector<int64_t, 4>> *lbs = nullptr) const;

  /// A wrapper around FlatAffineConstraints::getConstantBoundOnDimSize(). 'pos'
  /// corresponds to the position of the memref shape's dimension (major to
  /// minor) which matches 1:1 with the dimensional identifier positions in
  //'cst'.
  Optional<int64_t>
  getConstantBoundOnDimSize(unsigned pos, SmallVectorImpl<int64_t> *lb) const {
    assert(pos < getRank() && "invalid position");
    return cst.getConstantBoundOnDimSize(pos, lb);
  }

  /// Returns the rank of the memref that this region corresponds to.
  unsigned getRank() const;

  /// Memref that this region corresponds to.
  MLValue *memref;

private:
  /// Read or write.
  bool write;

  /// Region (data space) of the memref accessed. This set will thus have at
  /// least as many dimensional identifiers as the shape dimensionality of the
  /// memref, and these are the leading dimensions of the set appearing in that
  /// order (major to minor / outermost to innermost). There may be additional
  /// identifiers since getMemRefRegion() is called with a specific loop depth,
  /// and thus the region is symbolic in the outer surrounding loops at that
  /// depth.
  // TODO(bondhugula): Replace this to exploit HyperRectangularSet.
  FlatAffineConstraints cst;
};

/// Computes the memory region accessed by this memref with the region
/// represented as constraints symbolic/parameteric in 'loopDepth' loops
/// surrounding opStmt. Returns false if this fails due to yet unimplemented
/// cases. The computed region's 'cst' field has exactly as many dimensional
/// identifiers as the rank of the memref, and *potentially* additional symbolic
/// identifiers which could include any of the loop IVs surrounding opStmt up
/// until 'loopDepth' and another additional MLFunction symbols involved with
/// the access (for eg., those appear in affine_apply's, loop bounds, etc.).
///  For example, the memref region for this operation at loopDepth = 1 will be:
///
///    for %i = 0 to 32 {
///      for %ii = %i to (d0) -> (d0 + 8) (%i) {
///        load %A[%ii]
///      }
///    }
///
///   {memref = %A, write = false, {%i <= m0 <= %i + 7} }
/// The last field is a 2-d FlatAffineConstraints symbolic in %i.
///
bool getMemRefRegion(OperationStmt *opStmt, unsigned loopDepth,
                     MemRefRegion *region);

/// Returns the size of memref data in bytes if it's statically shaped, None
/// otherwise.
Optional<uint64_t> getMemRefSizeInBytes(MemRefType memRefType);

/// Checks a load or store op for an out of bound access; returns true if the
/// access is out of bounds along any of the dimensions, false otherwise. Emits
/// a diagnostic error (with location information) if emitError is true.
template <typename LoadOrStoreOpPointer>
bool boundCheckLoadOrStoreOp(LoadOrStoreOpPointer loadOrStoreOp,
                             bool emitError = true);

/// Creates a clone of the computation contained in the loop nest surrounding
/// 'srcAccess', and inserts it at the beginning of the statement block of the
/// loop containing 'dstAccess'. Returns the top-level loop of the computation
/// slice on success, returns nullptr otherwise.
// Computes memref dependence between 'srcAccess' and 'dstAccess' and uses the
// dependence constraint system to create AffineMaps with which to adjust the
// loop bounds of the inserted compution slice so that they are functions of the
// loop IVs and symbols of the loops surrounding 'dstAccess'.
// TODO(andydavis) Add 'dstLoopDepth' argument for computation slice insertion.
// Loop depth is a crucial optimization choice that determines where to
// materialize the results of the backward slice - presenting a trade-off b/w
// storage and redundant computation in several cases
// TODO(andydavis) Support computation slices with common surrounding loops.
ForStmt *insertBackwardComputationSlice(MemRefAccess *srcAccess,
                                        MemRefAccess *dstAccess);
} // end namespace mlir

#endif // MLIR_ANALYSIS_UTILS_H
