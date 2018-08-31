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
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/StmtVisitor.h"

#include "mlir/Transforms/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using llvm::report_fatal_error;

namespace {

/// Simplify all affine expressions appearing in the operation statements of the
/// MLFunction.
//  TODO(someone): Gradually, extend this to all affine map references found in
//  ML functions and CFG functions.
struct SimplifyAffineExpr : public FunctionPass {
  explicit SimplifyAffineExpr() {}

  void runOnMLFunction(MLFunction *f);
  // Does nothing on CFG functions for now. No reusable walkers/visitors exist
  // for this yet? TODO(someone).
  void runOnCFGFunction(CFGFunction *f) {}
};

// This class is used to flatten a pure affine expression into a sum of products
// (w.r.t constants) when possible, and in that process accumulating
// contributions for each dimensional and symbolic identifier together. Note
// that an affine expression may not always be expressible that way due to the
// preesnce of modulo, floordiv, and ceildiv expressions. A simplification that
// this flattening naturally performs is to fold a modulo expression to a zero,
// if possible. Two examples are below:
//
// (d0 + 3 * d1) + d0) - 2 * d1) - d0 simplified to  d0 + d1
// (d0 - d0 mod 4 + 4) mod 4  simplified to 0.
//
// For modulo and floordiv expressions, an additional variable is introduced to
// rewrite it as a sum of products (w.r.t constants). For example, for the
// second example above, d0 % 4 is replaced by d0 - 4*q with q being introduced:
// the expression simplifies to:
// (d0 - (d0 - 4q) + 4) = 4q + 4, modulo of which w.r.t 4 simplifies to zero.
//
// This is a linear time post order walk for an affine expression that attempts
// the above simplifications through visit methods, with partial results being
// stored in 'operandExprStack'. When a parent expr is visited, the flattened
// expressions corresponding to its two operands would already be on the stack -
// the parent expr looks at the two flattened expressions and combines the two.
// It pops off the operand expressions and pushes the combined result (although
// this is done in-place on its LHS operand expr. When the walk is completed,
// the flattened form of the top-level expression would be left on the stack.
//
class AffineExprFlattener : public AffineExprVisitor<AffineExprFlattener> {
public:
  std::vector<SmallVector<int64_t, 32>> operandExprStack;

  // The layout of the flattened expressions is dimensions, symbols, locals,
  // and constant term.
  unsigned getNumCols() const { return numDims + numSymbols + numLocals + 1; }

  AffineExprFlattener(unsigned numDims, unsigned numSymbols)
      : numDims(numDims), numSymbols(numSymbols), numLocals(0) {}

  void visitMulExpr(AffineBinaryOpExpr *expr) {
    assert(expr->isPureAffine());
    // Get the RHS constant.
    auto rhsConst = operandExprStack.back()[getNumCols() - 1];
    operandExprStack.pop_back();
    // Update the LHS in place instead of pop and push.
    auto &lhs = operandExprStack.back();
    for (unsigned i = 0, e = lhs.size(); i < e; i++) {
      lhs[i] *= rhsConst;
    }
  }
  void visitAddExpr(AffineBinaryOpExpr *expr) {
    const auto &rhs = operandExprStack.back();
    auto &lhs = operandExprStack[operandExprStack.size() - 2];
    assert(lhs.size() == rhs.size());
    // Update the LHS in place.
    for (unsigned i = 0; i < rhs.size(); i++) {
      lhs[i] += rhs[i];
    }
    // Pop off the RHS.
    operandExprStack.pop_back();
  }
  void visitModExpr(AffineBinaryOpExpr *expr) {
    assert(expr->isPureAffine());
    // This is a pure affine expr; the RHS is a constant.
    auto rhsConst = operandExprStack.back()[getNumCols() - 1];
    operandExprStack.pop_back();
    auto &lhs = operandExprStack.back();
    assert(rhsConst != 0 && "RHS constant can't be zero");
    unsigned i;
    for (i = 0; i < lhs.size(); i++)
      if (lhs[i] % rhsConst != 0)
        break;
    if (i == lhs.size()) {
      // The modulo expression here simplifies to zero.
      lhs.assign(lhs.size(), 0);
      return;
    }
    // Add an existential quantifier. expr1 % expr2 is replaced by (expr1 -
    // q * expr2) where q is the existential quantifier introduced.
    addExistentialQuantifier();
    lhs = operandExprStack.back();
    lhs[numDims + numSymbols + numLocals - 1] = -rhsConst;
  }
  void visitConstantExpr(AffineConstantExpr *expr) {
    operandExprStack.emplace_back(SmallVector<int64_t, 32>(getNumCols(), 0));
    auto &eq = operandExprStack.back();
    eq[getNumCols() - 1] = expr->getValue();
  }
  void visitDimExpr(AffineDimExpr *expr) {
    SmallVector<int64_t, 32> eq(getNumCols(), 0);
    eq[expr->getPosition()] = 1;
    operandExprStack.push_back(eq);
  }
  void visitSymbolExpr(AffineSymbolExpr *expr) {
    SmallVector<int64_t, 32> eq(getNumCols(), 0);
    eq[numDims + expr->getPosition()] = 1;
    operandExprStack.push_back(eq);
  }
  void visitCeilDivExpr(AffineBinaryOpExpr *expr) {
    // TODO(bondhugula): handle ceildiv as well; won't simplify further through
    // this analysis but will be handled (rest of the expr will simplify).
    report_fatal_error("ceildiv expr simplification not supported here");
  }
  void visitFloorDivExpr(AffineBinaryOpExpr *expr) {
    // TODO(bondhugula): handle ceildiv as well; won't simplify further through
    // this analysis but will be handled (rest of the expr will simplify).
    report_fatal_error("floordiv expr simplification unimplemented");
  }
  // Add an existential quantifier (used to flatten a mod or a floordiv expr).
  void addExistentialQuantifier() {
    for (auto &subExpr : operandExprStack) {
      subExpr.insert(subExpr.begin() + numDims + numSymbols + numLocals, 0);
    }
    numLocals++;
  }

  unsigned numDims;
  unsigned numSymbols;
  unsigned numLocals;
};

} // end anonymous namespace

FunctionPass *mlir::createSimplifyAffineExprPass() {
  return new SimplifyAffineExpr();
}

AffineMap *MutableAffineMap::getAffineMap() {
  return AffineMap::get(numDims, numSymbols, results, rangeSizes, context);
}

void SimplifyAffineExpr::runOnMLFunction(MLFunction *f) {
  struct MapSimplifier : public StmtWalker<MapSimplifier> {
    MLIRContext *context;
    MapSimplifier(MLIRContext *context) : context(context) {}

    void visitOperationStmt(OperationStmt *opStmt) {
      for (auto attr : opStmt->getAttrs()) {
        if (auto *mapAttr = dyn_cast<AffineMapAttr>(attr.second)) {
          MutableAffineMap mMap(mapAttr->getValue(), context);
          mMap.simplify();
          auto *map = mMap.getAffineMap();
          opStmt->setAttr(attr.first, AffineMapAttr::get(map, context));
        }
      }
    }
  };

  MapSimplifier v(f->getContext());
  v.walkPostOrder(f);
}

/// Get an affine expression from a flat ArrayRef. If there are local variables
/// (existential quantifiers introduced during the flattening) that appear in
/// the sum of products expression, we can't readily express it as an affine
/// expression of dimension and symbol id's; return nullptr in such cases.
static AffineExpr *toAffineExpr(ArrayRef<int64_t> eq, unsigned numDims,
                                unsigned numSymbols, MLIRContext *context) {
  // Check if any local variable has a non-zero coefficient.
  for (unsigned j = numDims + numSymbols; j < eq.size() - 1; j++) {
    if (eq[j] != 0)
      return nullptr;
  }

  AffineExpr *expr = AffineConstantExpr::get(0, context);
  for (unsigned j = 0; j < numDims + numSymbols; j++) {
    if (eq[j] != 0) {
      AffineExpr *id =
          j < numDims
              ? static_cast<AffineExpr *>(AffineDimExpr::get(j, context))
              : AffineSymbolExpr::get(j - numDims, context);
      expr = AffineBinaryOpExpr::get(
          AffineExpr::Kind::Add, expr,
          AffineBinaryOpExpr::get(AffineExpr::Kind::Mul,
                                  AffineConstantExpr::get(eq[j], context), id,
                                  context),
          context);
    }
  }
  unsigned constTerm = eq[eq.size() - 1];
  if (constTerm != 0)
    expr = AffineBinaryOpExpr::get(AffineExpr::Kind::Add, expr,
                                   AffineConstantExpr::get(constTerm, context),
                                   context);
  return expr;
}

// Simplify the result affine expressions of this map. The expressions have to
// be pure for the simplification implemented.
void MutableAffineMap::simplify() {
  // Simplify each of the results if possible.
  for (unsigned i = 0, e = getNumResults(); i < e; i++) {
    AffineExpr *result = getResult(i);
    if (!result->isPureAffine())
      continue;

    AffineExprFlattener flattener(numDims, numSymbols);
    flattener.walkPostOrder(result);
    const auto &flattenedExpr = flattener.operandExprStack.back();
    auto *expr = toAffineExpr(flattenedExpr, numDims, numSymbols, context);
    if (expr)
      results[i] = expr;
    flattener.operandExprStack.pop_back();
    assert(flattener.operandExprStack.empty());
  }
}
