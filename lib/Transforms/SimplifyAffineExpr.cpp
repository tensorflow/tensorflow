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
#include "mlir/IR/StmtVisitor.h"
#include "mlir/Pass.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "simplify-affine-structure"

using namespace mlir;
using llvm::report_fatal_error;

namespace {

/// Simplifies all affine expressions appearing in the operation statements of
/// the MLFunction. This is mainly to test the simplifyAffineExpr method.
//  TODO(someone): Gradually, extend this to all affine map references found in
//  ML functions and CFG functions.
struct SimplifyAffineStructures : public FunctionPass,
                                  StmtWalker<SimplifyAffineStructures> {
  explicit SimplifyAffineStructures()
      : FunctionPass(&SimplifyAffineStructures::passID) {}

  PassResult runOnMLFunction(MLFunction *f) override;
  // Does nothing on CFG functions for now. No reusable walkers/visitors exist
  // for this yet? TODO(someone).
  PassResult runOnCFGFunction(CFGFunction *f) override { return success(); }

  void visitIfStmt(IfStmt *ifStmt);
  void visitOperationInst(OperationInst *opStmt);

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

void SimplifyAffineStructures::visitIfStmt(IfStmt *ifStmt) {
  auto set = ifStmt->getCondition().getIntegerSet();
  ifStmt->setIntegerSet(simplifyIntegerSet(set));
}

void SimplifyAffineStructures::visitOperationInst(OperationInst *opStmt) {
  for (auto attr : opStmt->getAttrs()) {
    if (auto mapAttr = attr.second.dyn_cast<AffineMapAttr>()) {
      MutableAffineMap mMap(mapAttr.getValue());
      mMap.simplify();
      auto map = mMap.getAffineMap();
      opStmt->setAttr(attr.first, AffineMapAttr::get(map));
    }
  }
}

PassResult SimplifyAffineStructures::runOnMLFunction(MLFunction *f) {
  walk(f);
  return success();
}

static PassRegistration<SimplifyAffineStructures>
    pass("simplify-affine-structures", "Simplify affine expressions");
