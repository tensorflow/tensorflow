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

class AffineExpr;
class AffineValueMap;
class ForStmt;
class MLIRContext;
class FlatAffineConstraints;
class MLValue;
class OperationStmt;
class Statement;

/// Simplify an affine expression through flattening and some amount of
/// simple analysis. This has complexity linear in the number of nodes in
/// 'expr'. Returns the simplified expression, which is the same as the input
//  expression if it can't be simplified.
AffineExpr simplifyAffineExpr(AffineExpr expr, unsigned numDims,
                              unsigned numSymbols);

/// Returns the sequence of AffineApplyOp OperationStmts operation in
/// 'affineApplyOps', which are reachable via a search starting from 'operands',
/// and ending at operands which are not defined by AffineApplyOps.
void getReachableAffineApplyOps(
    llvm::ArrayRef<MLValue *> operands,
    llvm::SmallVectorImpl<OperationStmt *> &affineApplyOps);

/// Forward substitutes into 'valueMap' all AffineApplyOps reachable from the
/// operands of 'valueMap'.
void forwardSubstituteReachableOps(AffineValueMap *valueMap);

/// Flattens 'expr' into 'flattenedExpr'. Returns true on success or false
/// if 'expr' was unable to be flattened (i.e., semi-affine is not yet handled).
/// 'cst' contains constraints that connect newly introduced local identifiers
/// to existing dimensional and / symbolic identifiers. See documentation for
/// AffineExprFlattener on how mod's and div's are flattened.
bool getFlattenedAffineExpr(AffineExpr expr, unsigned numDims,
                            unsigned numSymbols,
                            llvm::SmallVectorImpl<int64_t> *flattenedExpr,
                            FlatAffineConstraints *cst = nullptr);

/// Adds constraints capturing the index set of the ML values in indices to
/// 'domain'.
bool addIndexSet(llvm::ArrayRef<const MLValue *> indices,
                 FlatAffineConstraints *domain);

/// Checks whether two accesses to the same memref access the same element.
/// Each access is specified using the MemRefAccess structure, which contains
/// the operation statement, indices and memref associated with the access.
/// Returns 'false' if it can be determined conclusively that the accesses do
/// not access the same memref element. Returns 'true' otherwise.
struct MemRefAccess {
  const MLValue *memref;
  const OperationStmt *opStmt;
  llvm::SmallVector<MLValue *, 4> indices;
  // Populates 'accessMap' with composition of AffineApplyOps reachable from
  // 'indices'.
  void getAccessMap(AffineValueMap *accessMap) const;
};
bool checkMemrefAccessDependence(const MemRefAccess &srcAccess,
                                 const MemRefAccess &dstAccess);

} // end namespace mlir

#endif // MLIR_ANALYSIS_AFFINE_ANALYSIS_H
