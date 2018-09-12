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
// involving affine structures (AffineExpr, AffineMap, IntegerSet, etc.) and
// other IR structures that in turn use these.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_AFFINE_ANALYSIS_H
#define MLIR_ANALYSIS_AFFINE_ANALYSIS_H

#include "llvm/ADT/Optional.h"

namespace mlir {

class AffineExpr;
class MLIRContext;

/// Simplify an affine expression through flattening and some amount of
/// simple analysis. This has complexity linear in the number of nodes in
/// 'expr'. Return nullptr, if the expression can't be simplified.
AffineExpr *simplifyAffineExpr(AffineExpr *expr, unsigned numDims,
                               unsigned numSymbols, MLIRContext *context);

} // end namespace mlir

#endif // MLIR_ANALYSIS_AFFINE_ANALYSIS_H
