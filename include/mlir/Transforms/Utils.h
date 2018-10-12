//===- Utils.h - General transformation utilities ---------------*- C++ -*-===//
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

#ifndef MLIR_TRANSFORMS_UTILS_H
#define MLIR_TRANSFORMS_UTILS_H

#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {

class Location;
class MLFuncBuilder;
class MLValue;
class OperationStmt;
class SSAValue;

/// Replace all uses of oldMemRef with newMemRef while optionally remapping the
/// old memref's indices using the supplied affine map and adding any additional
/// indices. The new memref could be of a different shape or rank. Returns true
/// on success and false if the replacement is not possible (whenever a memref
/// is used as an operand in a non-deferencing scenario).
/// Additional indices are added at the start.
// TODO(mlir-team): extend this for SSAValue / CFGFunctions. Can also be easily
// extended to add additional indices at any position.
bool replaceAllMemRefUsesWith(MLValue *oldMemRef, MLValue *newMemRef,
                              llvm::ArrayRef<MLValue *> extraIndices,
                              AffineMap indexRemap = AffineMap::Invalid());

/// Creates and inserts into 'builder' a new AffineApplyOp, with the number of
/// its results equal to the number of operands, as a composition
/// of all other AffineApplyOps reachable from input parameter 'operands'. If
/// different operands were drawing results from multiple affine apply ops,
/// these will also be collected into a single (multi-result) affine apply op.
/// The final results of the composed AffineApplyOp are returned in output
/// parameter 'results'. Returns the affine apply op created.
OperationStmt *
createComposedAffineApplyOp(MLFuncBuilder *builder, Location *loc,
                            ArrayRef<MLValue *> operands,
                            ArrayRef<OperationStmt *> affineApplyOps,
                            SmallVectorImpl<SSAValue *> &results);

/// Given an operation statement, inserts a new single affine apply operation,
/// that is exclusively used by this operation statement, and that provides all
/// operands that are results of an affine_apply as a function of loop iterators
/// and program parameters and whose results are.
///
/// Before
///
/// for %i = 0 to #map(%N)
///   %idx = affine_apply (d0) -> (d0 mod 2) (%i)
///   send %A[%idx], ...
///   %v = "compute"(%idx, ...)
///
/// After
///
/// for %i = 0 to #map(%N)
///   %idx = affine_apply (d0) -> (d0 mod 2) (%i)
///   send %A[%idx], ...
///   %idx_ = affine_apply (d0) -> (d0 mod 2) (%i)
///   %v = "compute"(%idx_, ...)

/// This allows the application of  different transformations on send and
/// compute (for eg.  / different shifts/delays)
///
/// Returns nullptr if none of the operands were the result of an affine_apply
/// and thus there was no affine computation slice to create. Returns the newly
/// affine_apply operation statement otherwise.
OperationStmt *createAffineComputationSlice(OperationStmt *opStmt);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_UTILS_H
