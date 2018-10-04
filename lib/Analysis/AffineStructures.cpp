//===- AffineStructures.cpp - MLIR Affine Structures Class-------*- C++ -*-===//
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
// Structures for affine/polyhedral analysis of MLIR functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/StandardOps.h"

using namespace mlir;

MutableAffineMap::MutableAffineMap(AffineMap *map, MLIRContext *context)
    : numDims(map->getNumDims()), numSymbols(map->getNumSymbols()),
      context(context) {
  for (auto result : map->getResults())
    results.push_back(result);
  for (auto rangeSize : map->getRangeSizes())
    results.push_back(rangeSize);
}

bool MutableAffineMap::isMultipleOf(unsigned idx, int64_t factor) const {
  if (const_cast<AffineExprRef &>(results[idx])->isMultipleOf(factor))
    return true;

  // TODO(bondhugula): use simplifyAffineExpr and FlatAffineConstraints to
  // complete this (for a more powerful analysis).
  return false;
}

// Simplifies the result affine expressions of this map. The expressions have to
// be pure for the simplification implemented.
void MutableAffineMap::simplify() {
  // Simplify each of the results if possible.
  // TODO(ntv): functional-style map
  for (unsigned i = 0, e = getNumResults(); i < e; i++) {
    results[i] = simplifyAffineExpr(getResult(i), numDims, numSymbols);
  }
}

MutableIntegerSet::MutableIntegerSet(IntegerSet *set, MLIRContext *context)
    : numDims(set->getNumDims()), numSymbols(set->getNumSymbols()),
      context(context) {
  // TODO(bondhugula)
}

// Universal set.
MutableIntegerSet::MutableIntegerSet(unsigned numDims, unsigned numSymbols,
                                     MLIRContext *context)
    : numDims(numDims), numSymbols(numSymbols), context(context) {}

AffineValueMap::AffineValueMap(const AffineApplyOp &op, MLIRContext *context)
    : map(op.getAffineMap(), context) {
  // TODO: pull operands and results in.
}

inline bool AffineValueMap::isMultipleOf(unsigned idx, int64_t factor) const {
  return map.isMultipleOf(idx, factor);
}

AffineValueMap::~AffineValueMap() {}

void FlatAffineConstraints::addEquality(ArrayRef<int64_t> eq) {
  assert(eq.size() == getNumCols());
  unsigned offset = equalities.size();
  equalities.resize(equalities.size() + eq.size());
  for (unsigned i = 0, e = eq.size(); i < e; i++) {
    equalities[offset + i] = eq[i];
  }
}
