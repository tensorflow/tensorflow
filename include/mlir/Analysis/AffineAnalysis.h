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
class AffineMap;
class AffineValueMap;
class ForStmt;
class MLIRContext;
class FlatAffineConstraints;
class IntegerSet;
class OperationInst;
class Statement;
class Value;

/// Simplify an affine expression through flattening and some amount of
/// simple analysis. This has complexity linear in the number of nodes in
/// 'expr'. Returns the simplified expression, which is the same as the input
///  expression if it can't be simplified.
AffineExpr simplifyAffineExpr(AffineExpr expr, unsigned numDims,
                              unsigned numSymbols);

/// Given 2 unbounded AffineMaps `f` and `g`, returns the AffineMap f o g
/// (i.e. f(g), i.e. f composed with g).
/// The resulting AffineMap has as many AffineDimExpr as `g` and as many
/// AffineSymbolExpr as the max of either `f` or `g`.
/// In particular, this does not try and compress the unused AffineDimExpr and
/// AffineSymbolExpr. This could be implemented later as a canonicalization if
/// needed.
///
/// Prerequisites:
/// The maps are composable, i.e. that the number of AffineDimExpr of `f`
/// matches the number of results of `g`.
///
/// Examples:
/// compose((d0, d1) -> (d0 + 1, d1 - 1),
///         (d0)[s0] -> (d0 + s0, d0 - s0))
/// Returns (d0)[s0] -> (d0 + s0 + 1, d0 - s0 - 1)
AffineMap composeUnboundedMaps(AffineMap f, AffineMap g);

/// Given an AffineExpr `e` and an unbounded AffineMap `g`, returns the
/// AffineExpr e o g (i.e. e(g), i.e. e composed with g).
///
/// Prerequisites:
/// `e` and `g` are composable, i.e. that the number of AffineDimExpr of `e` is
/// smaller than the number of results of `g`.
AffineExpr composeWithUnboundedMap(AffineExpr e, AffineMap g);

/// Returns the sequence of AffineApplyOp OperationInsts operation in
/// 'affineApplyOps', which are reachable via a search starting from 'operands',
/// and ending at operands which are not defined by AffineApplyOps.
void getReachableAffineApplyOps(
    llvm::ArrayRef<Value *> operands,
    llvm::SmallVectorImpl<OperationInst *> &affineApplyOps);

/// Forward substitutes into 'valueMap' all AffineApplyOps reachable from the
/// operands of 'valueMap'.
void forwardSubstituteReachableOps(AffineValueMap *valueMap);

/// Flattens 'expr' into 'flattenedExpr'. Returns true on success or false
/// if 'expr' could not be flattened (i.e., semi-affine is not yet handled).
/// 'cst' contains constraints that connect newly introduced local identifiers
/// to existing dimensional and / symbolic identifiers. See documentation for
/// AffineExprFlattener on how mod's and div's are flattened.
bool getFlattenedAffineExpr(AffineExpr expr, unsigned numDims,
                            unsigned numSymbols,
                            llvm::SmallVectorImpl<int64_t> *flattenedExpr,
                            FlatAffineConstraints *cst = nullptr);

/// Flattens the result expressions of the map to their corresponding flattened
/// forms and set in 'flattenedExprs'. Returns true on success or false
/// if any expression in the map could not be flattened (i.e., semi-affine is
/// not yet handled). 'cst' contains constraints that connect newly introduced
/// local identifiers to existing dimensional and / symbolic identifiers. See
/// documentation for AffineExprFlattener on how mod's and div's are flattened.
/// For all affine expressions that share the same operands (like those of an
/// affine map), this method should be used instead of repeatedly calling
/// getFlattenedAffineExpr since local variables added to deal with div's and
/// mod's will be reused across expressions.
bool getFlattenedAffineExprs(
    AffineMap map, std::vector<llvm::SmallVector<int64_t, 8>> *flattenedExprs,
    FlatAffineConstraints *cst = nullptr);
bool getFlattenedAffineExprs(
    IntegerSet set, std::vector<llvm::SmallVector<int64_t, 8>> *flattenedExprs,
    FlatAffineConstraints *cst = nullptr);

/// Builds a system of constraints with dimensional identifiers corresponding to
/// the loop IVs of the forStmts appearing in that order. Bounds of the loop are
/// used to add appropriate inequalities. Any symbols founds in the bound
/// operands are added as symbols in the system. Returns false for the yet
/// unimplemented cases.
//  TODO(bondhugula): handle non-unit strides.
bool getIndexSet(llvm::ArrayRef<ForStmt *> forStmts,
                 FlatAffineConstraints *domain);

struct MemRefAccess {
  const Value *memref;
  const OperationInst *opStmt;
  llvm::SmallVector<Value *, 4> indices;
  // Populates 'accessMap' with composition of AffineApplyOps reachable from
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
/// the operation statement, indices and memref associated with the access.
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
