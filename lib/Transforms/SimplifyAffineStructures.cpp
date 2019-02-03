//===- SimplifyAffineStructures.cpp - ---------------------------*- C++ -*-===//
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
// This file implements a pass to simplify affine structures.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Instruction.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "simplify-affine-structure"

using namespace mlir;

namespace {

/// Simplifies all affine expressions appearing in the operation instructions of
/// the Function. This is mainly to test the simplifyAffineExpr method.
/// TODO(someone): This should just be defined as a canonicalization pattern
/// on AffineMap and driven from the existing canonicalization pass.
struct SimplifyAffineStructures : public FunctionPass {
  explicit SimplifyAffineStructures()
      : FunctionPass(&SimplifyAffineStructures::passID) {}

  PassResult runOnFunction(Function *f) override;

  static char passID;
};

} // end anonymous namespace

char SimplifyAffineStructures::passID = 0;

FunctionPass *mlir::createSimplifyAffineStructuresPass() {
  return new SimplifyAffineStructures();
}

/// Performs basic integer set simplifications. Checks if it's empty, and
/// replaces it with the canonical empty set if it is.
static IntegerSet simplifyIntegerSet(IntegerSet set) {
  FlatAffineConstraints fac(set);
  if (fac.isEmpty())
    return IntegerSet::getEmptySet(set.getNumDims(), set.getNumSymbols(),
                                   set.getContext());
  return set;
}

PassResult SimplifyAffineStructures::runOnFunction(Function *f) {
  f->walkOps([&](OperationInst *opInst) {
    for (auto attr : opInst->getAttrs()) {
      if (auto mapAttr = attr.second.dyn_cast<AffineMapAttr>()) {
        MutableAffineMap mMap(mapAttr.getValue());
        mMap.simplify();
        auto map = mMap.getAffineMap();
        opInst->setAttr(attr.first, AffineMapAttr::get(map));
      } else if (auto setAttr = attr.second.dyn_cast<IntegerSetAttr>()) {
        auto simplified = simplifyIntegerSet(setAttr.getValue());
        opInst->setAttr(attr.first, IntegerSetAttr::get(simplified));
      }
    }
  });

  return success();
}

static PassRegistration<SimplifyAffineStructures>
    pass("simplify-affine-structures", "Simplify affine expressions");
