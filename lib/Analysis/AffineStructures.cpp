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

#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/StandardOps.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

/// Constructs an affine expression from a flat ArrayRef. If there are local
/// identifiers (neither dimensional nor symbolic) that appear in the sum of
/// products expression, 'localExprs' is expected to have the AffineExpr for it,
/// and is substituted into. The ArrayRef 'eq' is expected to be in the format
/// [dims, symbols, locals, constant term].
static AffineExpr *toAffineExpr(ArrayRef<int64_t> eq, unsigned numDims,
                                unsigned numSymbols,
                                ArrayRef<AffineExpr *> localExprs,
                                MLIRContext *context) {
  // Assert expected numLocals = eq.size() - numDims - numSymbols - 1
  assert(eq.size() - numDims - numSymbols - 1 == localExprs.size() &&
         "unexpected number of local expressions");

  AffineExpr *expr = AffineConstantExpr::get(0, context);
  // Dimensions and symbols.
  for (unsigned j = 0; j < numDims + numSymbols; j++) {
    if (eq[j] != 0) {
      AffineExpr *id =
          j < numDims
              ? static_cast<AffineExpr *>(AffineDimExpr::get(j, context))
              : AffineSymbolExpr::get(j - numDims, context);
      auto *term = AffineBinaryOpExpr::getMul(
          AffineConstantExpr::get(eq[j], context), id, context);
      expr = AffineBinaryOpExpr::getAdd(expr, term, context);
    }
  }

  // Local identifiers.
  for (unsigned j = numDims + numSymbols; j < eq.size() - 1; j++) {
    if (eq[j] != 0) {
      auto *term = AffineBinaryOpExpr::getMul(
          AffineConstantExpr::get(eq[j], context),
          localExprs[j - numDims - numSymbols], context);
      expr = AffineBinaryOpExpr::getAdd(expr, term, context);
    }
  }

  // Constant term.
  unsigned constTerm = eq[eq.size() - 1];
  if (constTerm != 0)
    expr = AffineBinaryOpExpr::getAdd(
        expr, AffineConstantExpr::get(constTerm, context), context);
  return expr;
}

namespace {

// This class is used to flatten a pure affine expression (AffineExpr *, which
// is in a tree form) into a sum of products (w.r.t constants) when possible,
// and in that process simplifying the expression. The simplification performed
// includes the accumulation of contributions for each dimensional and symbolic
// identifier together, the simplification of floordiv/ceildiv/mod exprssions
// and other simplifications that in turn happen as a result. A simplification
// that this flattening naturally performs is of simplifying the numerator and
// denominator of floordiv/ceildiv, and folding a modulo expression to a zero,
// if possible. Three examples are below:
//
// (d0 + 3 * d1) + d0) - 2 * d1) - d0 simplified to  d0 + d1
// (d0 - d0 mod 4 + 4) mod 4  simplified to 0.
// (3*d0 + 2*d1 + d0) floordiv 2 + d1 simplified to 2*d0 + 2*d1
//
// For a modulo, floordiv, or a ceildiv expression, an additional identifier
// (called a local identifier) is introduced to rewrite it as a sum of products
// (w.r.t constants). For example, for the second example above, d0 % 4 is
// replaced by d0 - 4*q with q being introduced: the expression then simplifies
// to: (d0 - (d0 - 4q) + 4) = 4q + 4, modulo of which w.r.t 4 simplifies to
// zero. Note that an affine expression may not always be expressible in a sum
// of products form due to the presence of modulo/floordiv/ceildiv expressions
// that may not be eliminated after simplification; in such cases, the final
// expression can be reconstructed by replacing the local identifier with its
// explicit form stored in localExprs (note that the explicit form itself would
// have been simplified and not necessarily the original form).
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
  // Flattend expression layout: [dims, symbols, locals, constant]
  // Stack that holds the LHS and RHS operands while visiting a binary op expr.
  // In future, consider adding a prepass to determine how big the SmallVector's
  // will be, and linearize this to std::vector<int64_t> to prevent
  // SmallVector moves on re-allocation.
  std::vector<SmallVector<int64_t, 32>> operandExprStack;

  inline unsigned getNumCols() const {
    return numDims + numSymbols + numLocals + 1;
  }

  unsigned numDims;
  unsigned numSymbols;
  // Number of newly introduced identifiers to flatten mod/floordiv/ceildiv
  // expressions that could not be simplified.
  unsigned numLocals;
  // AffineExpr's corresponding to the floordiv/ceildiv/mod expressions for
  // which new identifiers were introduced; if the latter do not get canceled
  // out, these expressions are needed to reconstruct the AffineExpr * / tree
  // form. Note that these expressions themselves would have been simplified
  // (recursively) by this pass. Eg. d0 + (d0 + 2*d1 + d0) ceildiv 4 will be
  // simplified to d0 + q, where q = (d0 + d1) ceildiv 2. (d0 + d1) ceildiv 2
  // would be the local expression stored for q.
  SmallVector<AffineExpr *, 4> localExprs;
  MLIRContext *context;

  AffineExprFlattener(unsigned numDims, unsigned numSymbols,
                      MLIRContext *context)
      : numDims(numDims), numSymbols(numSymbols), numLocals(0),
        context(context) {
    operandExprStack.reserve(8);
  }

  void visitMulExpr(AffineBinaryOpExpr *expr) {
    assert(operandExprStack.size() >= 2);
    // This is a pure affine expr; the RHS will be a constant.
    assert(isa<AffineConstantExpr>(expr->getRHS()));
    // Get the RHS constant.
    auto rhsConst = operandExprStack.back()[getConstantIndex()];
    operandExprStack.pop_back();
    // Update the LHS in place instead of pop and push.
    auto &lhs = operandExprStack.back();
    for (unsigned i = 0, e = lhs.size(); i < e; i++) {
      lhs[i] *= rhsConst;
    }
  }

  void visitAddExpr(AffineBinaryOpExpr *expr) {
    assert(operandExprStack.size() >= 2);
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
    assert(operandExprStack.size() >= 2);
    // This is a pure affine expr; the RHS will be a constant.
    assert(isa<AffineConstantExpr>(expr->getRHS()));
    auto rhsConst = operandExprStack.back()[getConstantIndex()];
    operandExprStack.pop_back();
    auto &lhs = operandExprStack.back();
    // TODO(bondhugula): handle modulo by zero case when this issue is fixed
    // at the other places in the IR.
    assert(rhsConst != 0 && "RHS constant can't be zero");

    // Check if the LHS expression is a multiple of modulo factor.
    unsigned i;
    for (i = 0; i < lhs.size(); i++)
      if (lhs[i] % rhsConst != 0)
        break;
    // If yes, modulo expression here simplifies to zero.
    if (i == lhs.size()) {
      lhs.assign(lhs.size(), 0);
      return;
    }

    // Add an existential quantifier. expr1 % expr2 is replaced by (expr1 -
    // q * expr2) where q is the existential quantifier introduced.
    addLocalId(AffineBinaryOpExpr::get(
        AffineExpr::Kind::FloorDiv,
        toAffineExpr(lhs, numDims, numSymbols, localExprs, context),
        AffineConstantExpr::get(rhsConst, context), context));
    lhs[getLocalVarStartIndex() + numLocals - 1] = -rhsConst;
  }
  void visitCeilDivExpr(AffineBinaryOpExpr *expr) {
    visitDivExpr(expr, /*isCeil=*/true);
  }
  void visitFloorDivExpr(AffineBinaryOpExpr *expr) {
    visitDivExpr(expr, /*isCeil=*/false);
  }
  void visitDimExpr(AffineDimExpr *expr) {
    operandExprStack.emplace_back(SmallVector<int64_t, 32>(getNumCols(), 0));
    auto &eq = operandExprStack.back();
    eq[getDimStartIndex() + expr->getPosition()] = 1;
  }
  void visitSymbolExpr(AffineSymbolExpr *expr) {
    operandExprStack.emplace_back(SmallVector<int64_t, 32>(getNumCols(), 0));
    auto &eq = operandExprStack.back();
    eq[getSymbolStartIndex() + expr->getPosition()] = 1;
  }
  void visitConstantExpr(AffineConstantExpr *expr) {
    operandExprStack.emplace_back(SmallVector<int64_t, 32>(getNumCols(), 0));
    auto &eq = operandExprStack.back();
    eq[getConstantIndex()] = expr->getValue();
  }

private:
  void visitDivExpr(AffineBinaryOpExpr *expr, bool isCeil) {
    assert(operandExprStack.size() >= 2);
    assert(isa<AffineConstantExpr>(expr->getRHS()));
    // This is a pure affine expr; the RHS is a positive constant.
    auto rhsConst = operandExprStack.back()[getConstantIndex()];
    // TODO(bondhugula): handle division by zero at the same time the issue is
    // fixed at other places.
    assert(rhsConst != 0 && "RHS constant can't be zero");
    operandExprStack.pop_back();
    auto &lhs = operandExprStack.back();

    // Simplify the floordiv, ceildiv if possible by canceling out the greatest
    // common divisors of the numerator and denominator.
    uint64_t gcd = std::abs(rhsConst);
    for (unsigned i = 0; i < lhs.size(); i++)
      gcd = llvm::GreatestCommonDivisor64(gcd, std::abs(lhs[i]));
    // Simplify the numerator and the denominator.
    if (gcd != 1) {
      for (unsigned i = 0; i < lhs.size(); i++)
        lhs[i] = lhs[i] / gcd;
    }
    int64_t denominator = rhsConst / gcd;
    // If the denominator becomes 1, the updated LHS is the result. (The
    // denominator can't be negative since rhsConst is positive).
    if (denominator == 1)
      return;

    // If the denominator cannot be simplified to one, we will have to retain
    // the ceil/floor expr (simplified up until here). Add an existential
    // quantifier to express its result, i.e., expr1 div expr2 is replaced
    // by a new identifier, q.
    auto divKind =
        isCeil ? AffineExpr::Kind::CeilDiv : AffineExpr::Kind::FloorDiv;
    addLocalId(AffineBinaryOpExpr::get(
        divKind, toAffineExpr(lhs, numDims, numSymbols, localExprs, context),
        AffineConstantExpr::get(denominator, context), context));
    lhs.assign(lhs.size(), 0);
    lhs[getLocalVarStartIndex() + numLocals - 1] = 1;
  }

  // Add an existential quantifier (used to flatten a mod, floordiv, ceildiv
  // expr). localExpr is the simplified tree expression (AffineExpr *)
  // corresponding to the quantifier.
  void addLocalId(AffineExpr *localExpr) {
    for (auto &subExpr : operandExprStack) {
      subExpr.insert(subExpr.begin() + getLocalVarStartIndex() + numLocals, 0);
    }
    localExprs.push_back(localExpr);
    numLocals++;
  }

  inline unsigned getConstantIndex() const { return getNumCols() - 1; }
  inline unsigned getLocalVarStartIndex() const { return numDims + numSymbols; }
  inline unsigned getSymbolStartIndex() const { return numDims; }
  inline unsigned getDimStartIndex() const { return 0; }
};

} // end anonymous namespace

AffineExpr *mlir::simplifyAffineExpr(AffineExpr *expr, unsigned numDims,
                                     unsigned numSymbols,
                                     MLIRContext *context) {
  // TODO(bondhugula): only pure affine for now. The simplification here can be
  // extended to semi-affine maps as well.
  if (!expr->isPureAffine())
    return nullptr;

  AffineExprFlattener flattener(numDims, numSymbols, context);
  flattener.walkPostOrder(expr);
  ArrayRef<int64_t> flattenedExpr = flattener.operandExprStack.back();
  auto *simplifiedExpr = toAffineExpr(flattenedExpr, numDims, numSymbols,
                                      flattener.localExprs, context);
  flattener.operandExprStack.pop_back();
  assert(flattener.operandExprStack.empty());
  if (simplifiedExpr == expr)
    return nullptr;
  return simplifiedExpr;
}

MutableAffineMap::MutableAffineMap(AffineMap *map, MLIRContext *context)
    : numDims(map->getNumDims()), numSymbols(map->getNumSymbols()),
      context(context) {
  for (auto *result : map->getResults())
    results.push_back(result);
  for (auto *rangeSize : map->getRangeSizes())
    results.push_back(rangeSize);
}

bool MutableAffineMap::isMultipleOf(unsigned idx, int64_t factor) const {
  if (results[idx]->isMultipleOf(factor))
    return true;

  // TODO(bondhugula): use simplifyAffineExpr and FlatAffineConstraints to
  // complete this (for a more powerful analysis).
  return false;
}

// Simplifies the result affine expressions of this map. The expressions have to
// be pure for the simplification implemented.
void MutableAffineMap::simplify() {
  // Simplify each of the results if possible.
  for (unsigned i = 0, e = getNumResults(); i < e; i++) {
    AffineExpr *sExpr =
        simplifyAffineExpr(getResult(i), numDims, numSymbols, context);
    if (sExpr)
      results[i] = sExpr;
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
