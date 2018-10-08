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
// involving affine structures (AffineExprClass, AffineMap, IntegerSet, etc.)
// and other IR structures that in turn use these.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_AFFINE_ANALYSIS_H
#define MLIR_ANALYSIS_AFFINE_ANALYSIS_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

namespace detail {

class AffineExprClass;

} // namespace detail
template <typename T> class AffineExprBase;
using AffineExpr = AffineExprBase<detail::AffineExprClass>;
class MLIRContext;
class MLValue;
class OperationStmt;

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
    const llvm::SmallVector<MLValue *, 4> &operands,
    llvm::SmallVector<OperationStmt *, 4> *affineApplyOps);

} // end namespace mlir

#endif // MLIR_ANALYSIS_AFFINE_ANALYSIS_H
