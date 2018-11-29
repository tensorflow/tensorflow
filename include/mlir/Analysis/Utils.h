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
class MLValue;
class OperationStmt;
class Statement;

/// Returns true if statement 'a' dominates statement b.
bool dominates(const Statement &a, const Statement &b);

/// Returns true if statement 'a' properly dominates statement b.
bool properlyDominates(const Statement &a, const Statement &b);

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

  // Computes the shape if the extents are known constants, returns false
  // otherwise.
  bool getConstantShape(llvm::SmallVectorImpl<int> *shape) const;

  // Returns the number of elements in this region if it's a known constant. We
  // use int64_t instead of uint64_t since index types can be at most int64_t.
  Optional<int64_t> getConstantSize() const;

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
/// cases.
//  For example, the memref region for this operation at loopDepth = 1 will be:
//
//    for %i = 0 to 32 {
//      for %ii = %i to (d0) -> (d0 + 8) (%i) {
//        load %A[%ii]
//      }
//    }
//
//   {memref = %A, write = false, {%i <= m0 <= %i + 7} }
// The last field is a 2-d FlatAffineConstraints symbolic in %i.
//
bool getMemRefRegion(OperationStmt *opStmt, unsigned loopDepth,
                     MemRefRegion *region);

/// Returns the size of memref data in bytes if it's statically shaped, None
/// otherwise.
Optional<uint64_t> getMemRefSizeInBytes(MemRefType memRefType);

} // end namespace mlir

#endif // MLIR_ANALYSIS_UTILS_H
