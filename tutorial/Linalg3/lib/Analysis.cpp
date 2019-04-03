//===- Analysis.cpp - Implementation of analysis functions for Linalg -----===//
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
// This file implements a simple IR operation to create a new RangeType in the
// linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "linalg3/Analysis.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/StandardTypes.h"

using llvm::SmallVector;
using namespace mlir;

// Compute an inverse map (only works with permutations for now).
// Note that the mapping is generally non-full rank, so this returns the first
// seen entry for each dim.
static AffineMap inversePermutationMap(AffineMap map) {
  SmallVector<AffineExpr, 4> exprs(map.getNumDims());
  for (auto en : llvm::enumerate(map.getResults())) {
    auto expr = en.value();
    auto d = expr.dyn_cast<AffineDimExpr>();
    assert(d && "permutation map expected");
    if (exprs[d.getPosition()])
      continue;
    exprs[d.getPosition()] = getAffineDimExpr(en.index(), d.getContext());
  }
  SmallVector<AffineExpr, 4> seenExprs;
  seenExprs.reserve(map.getNumDims());
  for (auto expr : exprs)
    if (expr)
      seenExprs.push_back(expr);
  assert(map.getNumSymbols() == 0 && "expected map without symbols");
  assert(seenExprs.size() == map.getNumInputs() && "map is not invertible");
  return AffineMap::get(map.getNumResults(), 0, seenExprs, {});
}

mlir::AffineMap linalg::inverseSubMap(AffineMap map, unsigned beginResult,
                                      unsigned endResult) {
  if (beginResult == 0 && endResult == 0)
    endResult = map.getNumResults();
  auto subMap = AffineMap::get(
      map.getNumDims(), map.getNumSymbols(),
      map.getResults().slice(beginResult, endResult - beginResult), {});
  return inversePermutationMap(subMap);
}
