//===- SimplifyAffineExpr.cpp - MLIR Affine Structures Class-----*- C++ -*-===//
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
// This file implements a pass to simplify affine expressions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/InstVisitor.h"
#include "mlir/Pass.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "simplify-affine-structure"

using namespace mlir;
using llvm::report_fatal_error;

namespace {

/// Simplifies all affine expressions appearing in the operation instructions of
/// the Function. This is mainly to test the simplifyAffineExpr method.
//  TODO(someone): Gradually, extend this to all affine map references found in
//  ML functions and CFG functions.
struct SimplifyAffineStructures : public FunctionPass,
                                  InstWalker<SimplifyAffineStructures> {
  explicit SimplifyAffineStructures()
      : FunctionPass(&SimplifyAffineStructures::passID) {}

  PassResult runOnMLFunction(Function *f) override;
  // Does nothing on CFG functions for now. No reusable walkers/visitors exist
  // for this yet? TODO(someone).
  PassResult runOnCFGFunction(Function *f) override { return success(); }

  void visitIfInst(IfInst *ifInst);
  void visitOperationInst(OperationInst *opInst);

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

void SimplifyAffineStructures::visitIfInst(IfInst *ifInst) {
  auto set = ifInst->getCondition().getIntegerSet();
  ifInst->setIntegerSet(simplifyIntegerSet(set));
}

void SimplifyAffineStructures::visitOperationInst(OperationInst *opInst) {
  for (auto attr : opInst->getAttrs()) {
    if (auto mapAttr = attr.second.dyn_cast<AffineMapAttr>()) {
      MutableAffineMap mMap(mapAttr.getValue());
      mMap.simplify();
      auto map = mMap.getAffineMap();
      opInst->setAttr(attr.first, AffineMapAttr::get(map));
    }
  }
}

PassResult SimplifyAffineStructures::runOnMLFunction(Function *f) {
  walk(f);
  return success();
}

static PassRegistration<SimplifyAffineStructures>
    pass("simplify-affine-structures", "Simplify affine expressions");
