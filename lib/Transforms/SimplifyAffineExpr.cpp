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
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Statements.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/Transforms/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using llvm::report_fatal_error;

namespace {

/// Simplifies all affine expressions appearing in the operation statements of
/// the MLFunction. This is mainly to test the simplifyAffineExpr method.
//  TODO(someone): Gradually, extend this to all affine map references found in
//  ML functions and CFG functions.
struct SimplifyAffineStructures : public FunctionPass,
                                  StmtWalker<SimplifyAffineStructures> {
  explicit SimplifyAffineStructures() {}

  PassResult runOnMLFunction(MLFunction *f);
  // Does nothing on CFG functions for now. No reusable walkers/visitors exist
  // for this yet? TODO(someone).
  PassResult runOnCFGFunction(CFGFunction *f) { return success(); }

  void visitIfStmt(IfStmt *ifStmt);
  void visitOperationStmt(OperationStmt *opStmt);
};

} // end anonymous namespace

FunctionPass *mlir::createSimplifyAffineStructuresPass() {
  return new SimplifyAffineStructures();
}

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

void SimplifyAffineStructures::visitOperationStmt(OperationStmt *opStmt) {
  for (auto attr : opStmt->getAttrs()) {
    if (auto *mapAttr = dyn_cast<AffineMapAttr>(attr.second)) {
      MutableAffineMap mMap(mapAttr->getValue());
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
