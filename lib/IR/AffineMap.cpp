//===- AffineMap.cpp - MLIR Affine Map Classes ----------------------------===//
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

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

AffineMap::AffineMap(unsigned numDims, unsigned numSymbols, unsigned numResults,
                     AffineExpr *const *results)
    : numDims(numDims), numSymbols(numSymbols), numResults(numResults),
      results(results) {}

AffineExpr *AffineAddExpr::simplify(AffineExpr *lhs, AffineExpr *rhs,
                                    MLIRContext *context) {
  AffineConstantExpr *l, *r;
  if ((l = dyn_cast<AffineConstantExpr>(lhs)) &&
      (r = dyn_cast<AffineConstantExpr>(rhs)))
    return AffineConstantExpr::get(l->getValue() + r->getValue(), context);
  return nullptr;
  // TODO(someone): implement more simplification.
}

// TODO(bondhugula): implement simplify for remaining affine binary op expr's
