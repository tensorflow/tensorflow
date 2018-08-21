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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardOps.h"

namespace mlir {

MutableAffineMap::MutableAffineMap(AffineMap *map) {
  for (auto *result : map->getResults())
    results.push_back(result);
  for (auto *rangeSize : map->getRangeSizes())
    results.push_back(rangeSize);
}

MutableIntegerSet::MutableIntegerSet(IntegerSet *set)
    : numDims(set->getNumDims()), numSymbols(set->getNumSymbols()) {
  // TODO(bondhugula)
}

AffineValueMap::AffineValueMap(const AffineApplyOp &op)
    : map(op.getAffineMap()) {
  // TODO: pull operands and results in.
}

bool AffineValueMap::isMultipleOf(unsigned idx, int64_t factor) const {
  /* Check if the (first result expr) % factor becomes 0. */
  if (auto *expr = dyn_cast<AffineConstantExpr>(AffineBinaryOpExpr::get(
          AffineExpr::Kind::Mod, map.getResult(idx),
          AffineConstantExpr::get(factor, context), context)))
    return expr->getValue() == 0;

  // TODO(bondhugula): use FlatAffineConstraints to complete this.
  assert(0 && "isMultipleOf implementation incomplete");
}

AffineValueMap::~AffineValueMap() {}

} // end namespace mlir
