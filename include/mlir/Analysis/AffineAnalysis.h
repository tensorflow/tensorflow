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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

class AffineApplyOp;
class AffineForOp;
class AffineValueMap;
class FlatAffineConstraints;
class Instruction;
template <typename OpType> class OpPointer;
class Value;

/// Returns in `affineApplyOps`, the sequence of those AffineApplyOp
/// Instructions that are reachable via a search starting from `operands` and
/// ending at those operands that are not the result of an AffineApplyOp.
void getReachableAffineApplyOps(
    llvm::ArrayRef<Value *> operands,
    llvm::SmallVectorImpl<Instruction *> &affineApplyOps);

/// Builds a system of constraints with dimensional identifiers corresponding to
/// the loop IVs of the forOps appearing in that order. Bounds of the loop are
/// used to add appropriate inequalities. Any symbols founds in the bound
/// operands are added as symbols in the system. Returns false for the yet
/// unimplemented cases.
//  TODO(bondhugula): handle non-unit strides.
bool getIndexSet(llvm::MutableArrayRef<OpPointer<AffineForOp>> forOps,
                 FlatAffineConstraints *domain);

/// Encapsulates a memref load or store access information.
struct MemRefAccess {
  Value *memref;
  Instruction *opInst;
  llvm::SmallVector<Value *, 4> indices;

  /// Constructs a MemRefAccess from a load or store operation instruction.
  // TODO(b/119949820): add accessors to standard op's load, store, DMA op's to
  // return MemRefAccess, i.e., loadOp->getAccess(), dmaOp->getRead/WriteAccess.
  explicit MemRefAccess(Instruction *opInst);

  // Returns the rank of the memref associated with this access.
  unsigned getRank() const;
  // Returns true if this access is of a store op.
  bool isStore() const;

  /// Populates 'accessMap' with composition of AffineApplyOps reachable from
  // 'indices'.
  void getAccessMap(AffineValueMap *accessMap) const;
};

// DependenceComponent contains state about the direction of a dependence as an
// interval [lb, ub].
// Distance vectors components are represented by the interval [lb, ub] with
// lb == ub.
// Direction vectors components are represented by the interval [lb, ub] with
// lb < ub. Note that ub/lb == None means unbounded.
struct DependenceComponent {
  // The lower bound of the dependence distance.
  llvm::Optional<int64_t> lb;
  // The upper bound of the dependence distance (inclusive).
  llvm::Optional<int64_t> ub;
  DependenceComponent() : lb(llvm::None), ub(llvm::None) {}
};

/// Checks whether two accesses to the same memref access the same element.
/// Each access is specified using the MemRefAccess structure, which contains
/// the operation instruction, indices and memref associated with the access.
/// Returns 'false' if it can be determined conclusively that the accesses do
/// not access the same memref element. Returns 'true' otherwise.
// TODO(andydavis) Wrap 'dependenceConstraints' and 'dependenceComponents' into
// a single struct.
// TODO(andydavis) Make 'dependenceConstraints' optional arg.
bool checkMemrefAccessDependence(
    const MemRefAccess &srcAccess, const MemRefAccess &dstAccess,
    unsigned loopDepth, FlatAffineConstraints *dependenceConstraints,
    llvm::SmallVector<DependenceComponent, 2> *dependenceComponents);
} // end namespace mlir

#endif // MLIR_ANALYSIS_AFFINE_ANALYSIS_H
