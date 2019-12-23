//===- AffineAnalysis.h - analyses for affine structures --------*- C++ -*-===//
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
// This header file defines prototypes for methods that perform analysis
// involving affine structures (AffineExprStorage, AffineMap, IntegerSet, etc.)
// and other IR structures that in turn use these.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_AFFINE_ANALYSIS_H
#define MLIR_ANALYSIS_AFFINE_ANALYSIS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

class AffineApplyOp;
class AffineForOp;
class AffineValueMap;
class FlatAffineConstraints;
class Operation;

/// Returns in `affineApplyOps`, the sequence of those AffineApplyOp
/// Operations that are reachable via a search starting from `operands` and
/// ending at those operands that are not the result of an AffineApplyOp.
void getReachableAffineApplyOps(ArrayRef<Value> operands,
                                SmallVectorImpl<Operation *> &affineApplyOps);

/// Builds a system of constraints with dimensional identifiers corresponding to
/// the loop IVs of the forOps appearing in that order. Bounds of the loop are
/// used to add appropriate inequalities. Any symbols founds in the bound
/// operands are added as symbols in the system. Returns failure for the yet
/// unimplemented cases.
//  TODO(bondhugula): handle non-unit strides.
LogicalResult getIndexSet(MutableArrayRef<AffineForOp> forOps,
                          FlatAffineConstraints *domain);

/// Encapsulates a memref load or store access information.
struct MemRefAccess {
  Value memref;
  Operation *opInst;
  SmallVector<Value, 4> indices;

  /// Constructs a MemRefAccess from a load or store operation.
  // TODO(b/119949820): add accessors to standard op's load, store, DMA op's to
  // return MemRefAccess, i.e., loadOp->getAccess(), dmaOp->getRead/WriteAccess.
  explicit MemRefAccess(Operation *opInst);

  // Returns the rank of the memref associated with this access.
  unsigned getRank() const;
  // Returns true if this access is of a store op.
  bool isStore() const;

  /// Populates 'accessMap' with composition of AffineApplyOps reachable from
  /// 'indices'.
  void getAccessMap(AffineValueMap *accessMap) const;

  /// Equal if both affine accesses can be proved to be equivalent at compile
  /// time (considering the memrefs, their respective affine access maps  and
  /// operands). The equality of access functions + operands is checked by
  /// subtracting fully composed value maps, and then simplifying the difference
  /// using the expression flattener.
  /// TODO: this does not account for aliasing of memrefs.
  bool operator==(const MemRefAccess &rhs) const;
  bool operator!=(const MemRefAccess &rhs) const { return !(*this == rhs); }
};

// DependenceComponent contains state about the direction of a dependence as an
// interval [lb, ub] for an AffineForOp.
// Distance vectors components are represented by the interval [lb, ub] with
// lb == ub.
// Direction vectors components are represented by the interval [lb, ub] with
// lb < ub. Note that ub/lb == None means unbounded.
struct DependenceComponent {
  // The AffineForOp Operation associated with this dependence component.
  Operation *op;
  // The lower bound of the dependence distance.
  Optional<int64_t> lb;
  // The upper bound of the dependence distance (inclusive).
  Optional<int64_t> ub;
  DependenceComponent() : lb(llvm::None), ub(llvm::None) {}
};

/// Checks whether two accesses to the same memref access the same element.
/// Each access is specified using the MemRefAccess structure, which contains
/// the operation, indices and memref associated with the access. Returns
/// 'NoDependence' if it can be determined conclusively that the accesses do not
/// access the same memref element. If 'allowRAR' is true, will consider
/// read-after-read dependences (typically used by applications trying to
/// optimize input reuse).
// TODO(andydavis) Wrap 'dependenceConstraints' and 'dependenceComponents' into
// a single struct.
// TODO(andydavis) Make 'dependenceConstraints' optional arg.
struct DependenceResult {
  enum ResultEnum {
    HasDependence, // A dependence exists between 'srcAccess' and 'dstAccess'.
    NoDependence,  // No dependence exists between 'srcAccess' and 'dstAccess'.
    Failure,       // Dependence check failed due to unsupported cases.
  } value;
  DependenceResult(ResultEnum v) : value(v) {}
};

DependenceResult checkMemrefAccessDependence(
    const MemRefAccess &srcAccess, const MemRefAccess &dstAccess,
    unsigned loopDepth, FlatAffineConstraints *dependenceConstraints,
    SmallVector<DependenceComponent, 2> *dependenceComponents,
    bool allowRAR = false);

/// Utility function that returns true if the provided DependenceResult
/// corresponds to a dependence result.
inline bool hasDependence(DependenceResult result) {
  return result.value == DependenceResult::HasDependence;
}

/// Returns in 'depCompsVec', dependence components for dependences between all
/// load and store ops in loop nest rooted at 'forOp', at loop depths in range
/// [1, maxLoopDepth].
void getDependenceComponents(
    AffineForOp forOp, unsigned maxLoopDepth,
    std::vector<SmallVector<DependenceComponent, 2>> *depCompsVec);

} // end namespace mlir

#endif // MLIR_ANALYSIS_AFFINE_ANALYSIS_H
