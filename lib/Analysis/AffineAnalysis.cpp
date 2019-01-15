//===- AffineAnalysis.cpp - Affine structures analysis routines -----------===//
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
// This file implements miscellaneous analysis routines for affine structures
// (expressions, maps, sets), and other utilities relying on such analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Instructions.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "affine-analysis"

using namespace mlir;

using llvm::dbgs;

/// Constructs an affine expression from a flat ArrayRef. If there are local
/// identifiers (neither dimensional nor symbolic) that appear in the sum of
/// products expression, 'localExprs' is expected to have the AffineExpr
/// for it, and is substituted into. The ArrayRef 'eq' is expected to be in the
/// format [dims, symbols, locals, constant term].
//  TODO(bondhugula): refactor getAddMulPureAffineExpr to reuse it from here.
static AffineExpr toAffineExpr(ArrayRef<int64_t> eq, unsigned numDims,
                               unsigned numSymbols,
                               ArrayRef<AffineExpr> localExprs,
                               MLIRContext *context) {
  // Assert expected numLocals = eq.size() - numDims - numSymbols - 1
  assert(eq.size() - numDims - numSymbols - 1 == localExprs.size() &&
         "unexpected number of local expressions");

  auto expr = getAffineConstantExpr(0, context);
  // Dimensions and symbols.
  for (unsigned j = 0; j < numDims + numSymbols; j++) {
    if (eq[j] == 0) {
      continue;
    }
    auto id = j < numDims ? getAffineDimExpr(j, context)
                          : getAffineSymbolExpr(j - numDims, context);
    expr = expr + id * eq[j];
  }

  // Local identifiers.
  for (unsigned j = numDims + numSymbols, e = eq.size() - 1; j < e; j++) {
    if (eq[j] == 0) {
      continue;
    }
    auto term = localExprs[j - numDims - numSymbols] * eq[j];
    expr = expr + term;
  }

  // Constant term.
  int64_t constTerm = eq[eq.size() - 1];
  if (constTerm != 0)
    expr = expr + constTerm;
  return expr;
}

AffineMap mlir::simplifyAffineMap(AffineMap map) {
  SmallVector<AffineExpr, 8> exprs, sizes;
  for (auto e : map.getResults()) {
    exprs.push_back(
        simplifyAffineExpr(e, map.getNumDims(), map.getNumSymbols()));
  }
  for (auto e : map.getRangeSizes()) {
    sizes.push_back(
        simplifyAffineExpr(e, map.getNumDims(), map.getNumSymbols()));
  }
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs, sizes);
}

namespace {

// This class is used to flatten a pure affine expression (AffineExpr,
// which is in a tree form) into a sum of products (w.r.t constants) when
// possible, and in that process simplifying the expression. For a modulo,
// floordiv, or a ceildiv expression, an additional identifier, called a local
// identifier, is introduced to rewrite the expression as a sum of product
// affine expression. Each local identifier is always and by construction a
// floordiv of a pure add/mul affine function of dimensional, symbolic, and
// other local identifiers, in a non-mutually recursive way. Hence, every local
// identifier can ultimately always be recovered as an affine function of
// dimensional and symbolic identifiers (involving floordiv's); note however
// that by AffineExpr construction, some floordiv combinations are converted to
// mod's. The result of the flattening is a flattened expression and a set of
// constraints involving just the local variables.
//
// d2 + (d0 + d1) floordiv 4  is flattened to d2 + q where 'q' is the local
// variable introduced, with localVarCst containing 4*q <= d0 + d1 <= 4*q + 3.
//
// The simplification performed includes the accumulation of contributions for
// each dimensional and symbolic identifier together, the simplification of
// floordiv/ceildiv/mod expressions and other simplifications that in turn
// happen as a result. A simplification that this flattening naturally performs
// is of simplifying the numerator and denominator of floordiv/ceildiv, and
// folding a modulo expression to a zero, if possible. Three examples are below:
//
// (d0 + 3 * d1) + d0) - 2 * d1) - d0    simplified to     d0 + d1
// (d0 - d0 mod 4 + 4) mod 4             simplified to     0
// (3*d0 + 2*d1 + d0) floordiv 2 + d1    simplified to     2*d0 + 2*d1
//
// The way the flattening works for the second example is as follows: d0 % 4 is
// replaced by d0 - 4*q with q being introduced: the expression then simplifies
// to: (d0 - (d0 - 4q) + 4) = 4q + 4, modulo of which w.r.t 4 simplifies to
// zero. Note that an affine expression may not always be expressible purely as
// a sum of products involving just the original dimensional and symbolic
// identifiers due to the presence of modulo/floordiv/ceildiv expressions that
// may not be eliminated after simplification; in such cases, the final
// expression can be reconstructed by replacing the local identifiers with their
// corresponding explicit form stored in 'localExprs' (note that each of the
// explicit forms itself would have been simplified).
//
// The expression walk method here performs a linear time post order walk that
// performs the above simplifications through visit methods, with partial
// results being stored in 'operandExprStack'. When a parent expr is visited,
// the flattened expressions corresponding to its two operands would already be
// on the stack - the parent expression looks at the two flattened expressions
// and combines the two. It pops off the operand expressions and pushes the
// combined result (although this is done in-place on its LHS operand expr).
// When the walk is completed, the flattened form of the top-level expression
// would be left on the stack.
//
// A flattener can be repeatedly used for multiple affine expressions that bind
// to the same operands, for example, for all result expressions of an
// AffineMap or AffineValueMap. In such cases, using it for multiple expressions
// is more efficient than creating a new flattener for each expression since
// common idenical div and mod expressions appearing across different
// expressions are mapped to the same local identifier (same column position in
// 'localVarCst').
struct AffineExprFlattener : public AffineExprVisitor<AffineExprFlattener> {
public:
  // Flattend expression layout: [dims, symbols, locals, constant]
  // Stack that holds the LHS and RHS operands while visiting a binary op expr.
  // In future, consider adding a prepass to determine how big the SmallVector's
  // will be, and linearize this to std::vector<int64_t> to prevent
  // SmallVector moves on re-allocation.
  std::vector<SmallVector<int64_t, 8>> operandExprStack;
  // Constraints connecting newly introduced local variables (for mod's and
  // div's) to existing (dimensional and symbolic) ones. These are always
  // inequalities.
  FlatAffineConstraints localVarCst;

  unsigned numDims;
  unsigned numSymbols;
  // Number of newly introduced identifiers to flatten mod/floordiv/ceildiv
  // expressions that could not be simplified.
  unsigned numLocals;
  // AffineExpr's corresponding to the floordiv/ceildiv/mod expressions for
  // which new identifiers were introduced; if the latter do not get canceled
  // out, these expressions can be readily used to reconstruct the AffineExpr
  // (tree) form. Note that these expressions themselves would have been
  // simplified (recursively) by this pass. Eg. d0 + (d0 + 2*d1 + d0) ceildiv 4
  // will be simplified to d0 + q, where q = (d0 + d1) ceildiv 2. (d0 + d1)
  // ceildiv 2 would be the local expression stored for q.
  SmallVector<AffineExpr, 4> localExprs;
  MLIRContext *context;

  AffineExprFlattener(unsigned numDims, unsigned numSymbols,
                      MLIRContext *context)
      : numDims(numDims), numSymbols(numSymbols), numLocals(0),
        context(context) {
    operandExprStack.reserve(8);
    localVarCst.reset(numDims, numSymbols, numLocals);
  }

  void visitMulExpr(AffineBinaryOpExpr expr) {
    assert(operandExprStack.size() >= 2);
    // This is a pure affine expr; the RHS will be a constant.
    assert(expr.getRHS().isa<AffineConstantExpr>());
    // Get the RHS constant.
    auto rhsConst = operandExprStack.back()[getConstantIndex()];
    operandExprStack.pop_back();
    // Update the LHS in place instead of pop and push.
    auto &lhs = operandExprStack.back();
    for (unsigned i = 0, e = lhs.size(); i < e; i++) {
      lhs[i] *= rhsConst;
    }
  }

  void visitAddExpr(AffineBinaryOpExpr expr) {
    assert(operandExprStack.size() >= 2);
    const auto &rhs = operandExprStack.back();
    auto &lhs = operandExprStack[operandExprStack.size() - 2];
    assert(lhs.size() == rhs.size());
    // Update the LHS in place.
    for (unsigned i = 0, e = rhs.size(); i < e; i++) {
      lhs[i] += rhs[i];
    }
    // Pop off the RHS.
    operandExprStack.pop_back();
  }

  //
  // t = expr mod c   <=>  t = expr - c*q and c*q <= expr <= c*q + c - 1
  //
  // A mod expression "expr mod c" is thus flattened by introducing a new local
  // variable q (= expr floordiv c), such that expr mod c is replaced with
  // 'expr - c * q' and c * q <= expr <= c * q + c - 1 are added to localVarCst.
  void visitModExpr(AffineBinaryOpExpr expr) {
    assert(operandExprStack.size() >= 2);
    // This is a pure affine expr; the RHS will be a constant.
    assert(expr.getRHS().isa<AffineConstantExpr>());
    auto rhsConst = operandExprStack.back()[getConstantIndex()];
    operandExprStack.pop_back();
    auto &lhs = operandExprStack.back();
    // TODO(bondhugula): handle modulo by zero case when this issue is fixed
    // at the other places in the IR.
    assert(rhsConst > 0 && "RHS constant has to be positive");

    // Check if the LHS expression is a multiple of modulo factor.
    unsigned i, e;
    for (i = 0, e = lhs.size(); i < e; i++)
      if (lhs[i] % rhsConst != 0)
        break;
    // If yes, modulo expression here simplifies to zero.
    if (i == lhs.size()) {
      std::fill(lhs.begin(), lhs.end(), 0);
      return;
    }

    // Add a local variable for the quotient, i.e., expr % c is replaced by
    // (expr - q * c) where q = expr floordiv c. Do this while canceling out
    // the GCD of expr and c.
    SmallVector<int64_t, 8> floorDividend(lhs);
    uint64_t gcd = rhsConst;
    for (unsigned i = 0, e = lhs.size(); i < e; i++)
      gcd = llvm::GreatestCommonDivisor64(gcd, std::abs(lhs[i]));
    // Simplify the numerator and the denominator.
    if (gcd != 1) {
      for (unsigned i = 0, e = floorDividend.size(); i < e; i++)
        floorDividend[i] = floorDividend[i] / static_cast<int64_t>(gcd);
    }
    int64_t floorDivisor = rhsConst / static_cast<int64_t>(gcd);

    // Construct the AffineExpr form of the floordiv to store in localExprs.
    auto dividendExpr =
        toAffineExpr(floorDividend, numDims, numSymbols, localExprs, context);
    auto divisorExpr = getAffineConstantExpr(floorDivisor, context);
    auto floorDivExpr = dividendExpr.floorDiv(divisorExpr);
    int loc;
    if ((loc = findLocalId(floorDivExpr)) == -1) {
      addLocalFloorDivId(floorDividend, floorDivisor, floorDivExpr);
      // Set result at top of stack to "lhs - rhsConst * q".
      lhs[getLocalVarStartIndex() + numLocals - 1] = -rhsConst;
    } else {
      // Reuse the existing local id.
      lhs[getLocalVarStartIndex() + loc] = -rhsConst;
    }
  }

  void visitCeilDivExpr(AffineBinaryOpExpr expr) {
    visitDivExpr(expr, /*isCeil=*/true);
  }
  void visitFloorDivExpr(AffineBinaryOpExpr expr) {
    visitDivExpr(expr, /*isCeil=*/false);
  }

  void visitDimExpr(AffineDimExpr expr) {
    operandExprStack.emplace_back(SmallVector<int64_t, 32>(getNumCols(), 0));
    auto &eq = operandExprStack.back();
    assert(expr.getPosition() < numDims && "Inconsistent number of dims");
    eq[getDimStartIndex() + expr.getPosition()] = 1;
  }

  void visitSymbolExpr(AffineSymbolExpr expr) {
    operandExprStack.emplace_back(SmallVector<int64_t, 32>(getNumCols(), 0));
    auto &eq = operandExprStack.back();
    assert(expr.getPosition() < numSymbols && "inconsistent number of symbols");
    eq[getSymbolStartIndex() + expr.getPosition()] = 1;
  }

  void visitConstantExpr(AffineConstantExpr expr) {
    operandExprStack.emplace_back(SmallVector<int64_t, 32>(getNumCols(), 0));
    auto &eq = operandExprStack.back();
    eq[getConstantIndex()] = expr.getValue();
  }

private:
  // t = expr floordiv c   <=> t = q, c * q <= expr <= c * q + c - 1
  // A floordiv is thus flattened by introducing a new local variable q, and
  // replacing that expression with 'q' while adding the constraints
  // c * q <= expr <= c * q + c - 1 to localVarCst (done by
  // FlatAffineConstraints::addLocalFloorDiv).
  //
  // A ceildiv is similarly flattened:
  // t = expr ceildiv c   <=> t =  (expr + c - 1) floordiv c
  void visitDivExpr(AffineBinaryOpExpr expr, bool isCeil) {
    assert(operandExprStack.size() >= 2);
    assert(expr.getRHS().isa<AffineConstantExpr>());

    // This is a pure affine expr; the RHS is a positive constant.
    int64_t rhsConst = operandExprStack.back()[getConstantIndex()];
    // TODO(bondhugula): handle division by zero at the same time the issue is
    // fixed at other places.
    assert(rhsConst > 0 && "RHS constant has to be positive");
    operandExprStack.pop_back();
    auto &lhs = operandExprStack.back();

    // Simplify the floordiv, ceildiv if possible by canceling out the greatest
    // common divisors of the numerator and denominator.
    uint64_t gcd = std::abs(rhsConst);
    for (unsigned i = 0, e = lhs.size(); i < e; i++)
      gcd = llvm::GreatestCommonDivisor64(gcd, std::abs(lhs[i]));
    // Simplify the numerator and the denominator.
    if (gcd != 1) {
      for (unsigned i = 0, e = lhs.size(); i < e; i++)
        lhs[i] = lhs[i] / static_cast<int64_t>(gcd);
    }
    int64_t divisor = rhsConst / static_cast<int64_t>(gcd);
    // If the divisor becomes 1, the updated LHS is the result. (The
    // divisor can't be negative since rhsConst is positive).
    if (divisor == 1)
      return;

    // If the divisor cannot be simplified to one, we will have to retain
    // the ceil/floor expr (simplified up until here). Add an existential
    // quantifier to express its result, i.e., expr1 div expr2 is replaced
    // by a new identifier, q.
    auto a = toAffineExpr(lhs, numDims, numSymbols, localExprs, context);
    auto b = getAffineConstantExpr(divisor, context);

    int loc;
    auto divExpr = isCeil ? a.ceilDiv(b) : a.floorDiv(b);
    if ((loc = findLocalId(divExpr)) == -1) {
      if (!isCeil) {
        SmallVector<int64_t, 8> dividend(lhs);
        addLocalFloorDivId(dividend, divisor, divExpr);
      } else {
        // lhs ceildiv c <=>  (lhs + c - 1) floordiv c
        SmallVector<int64_t, 8> dividend(lhs);
        dividend.back() += divisor - 1;
        addLocalFloorDivId(dividend, divisor, divExpr);
      }
    }
    // Set the expression on stack to the local var introduced to capture the
    // result of the division (floor or ceil).
    std::fill(lhs.begin(), lhs.end(), 0);
    if (loc == -1)
      lhs[getLocalVarStartIndex() + numLocals - 1] = 1;
    else
      lhs[getLocalVarStartIndex() + loc] = 1;
  }

  // Add a local identifier (needed to flatten a mod, floordiv, ceildiv expr).
  // The local identifier added is always a floordiv of a pure add/mul affine
  // function of other identifiers, coefficients of which are specified in
  // dividend and with respect to a positive constant divisor. localExpr is the
  // simplified tree expression (AffineExpr) corresponding to the quantifier.
  void addLocalFloorDivId(ArrayRef<int64_t> dividend, int64_t divisor,
                          AffineExpr localExpr) {
    assert(divisor > 0 && "positive constant divisor expected");
    for (auto &subExpr : operandExprStack)
      subExpr.insert(subExpr.begin() + getLocalVarStartIndex() + numLocals, 0);
    localExprs.push_back(localExpr);
    numLocals++;
    // Update localVarCst.
    localVarCst.addLocalFloorDiv(dividend, divisor);
  }

  int findLocalId(AffineExpr localExpr) {
    SmallVectorImpl<AffineExpr>::iterator it;
    if ((it = std::find(localExprs.begin(), localExprs.end(), localExpr)) ==
        localExprs.end())
      return -1;
    return it - localExprs.begin();
  }

  inline unsigned getNumCols() const {
    return numDims + numSymbols + numLocals + 1;
  }
  inline unsigned getConstantIndex() const { return getNumCols() - 1; }
  inline unsigned getLocalVarStartIndex() const { return numDims + numSymbols; }
  inline unsigned getSymbolStartIndex() const { return numDims; }
  inline unsigned getDimStartIndex() const { return 0; }
};

} // end anonymous namespace

/// Simplify the affine expression by flattening it and reconstructing it.
AffineExpr mlir::simplifyAffineExpr(AffineExpr expr, unsigned numDims,
                                    unsigned numSymbols) {
  // TODO(bondhugula): only pure affine for now. The simplification here can
  // be extended to semi-affine maps in the future.
  if (!expr.isPureAffine())
    return expr;

  AffineExprFlattener flattener(numDims, numSymbols, expr.getContext());
  flattener.walkPostOrder(expr);
  ArrayRef<int64_t> flattenedExpr = flattener.operandExprStack.back();
  auto simplifiedExpr = toAffineExpr(flattenedExpr, numDims, numSymbols,
                                     flattener.localExprs, expr.getContext());
  flattener.operandExprStack.pop_back();
  assert(flattener.operandExprStack.empty());

  return simplifiedExpr;
}

// Flattens the expressions in map. Returns true on success or false
// if 'expr' was unable to be flattened (i.e., semi-affine expressions not
// handled yet).
static bool getFlattenedAffineExprs(
    ArrayRef<AffineExpr> exprs, unsigned numDims, unsigned numSymbols,
    std::vector<llvm::SmallVector<int64_t, 8>> *flattenedExprs,
    FlatAffineConstraints *localVarCst) {
  if (exprs.empty()) {
    localVarCst->reset(numDims, numSymbols);
    return true;
  }

  flattenedExprs->clear();
  flattenedExprs->reserve(exprs.size());

  AffineExprFlattener flattener(numDims, numSymbols, exprs[0].getContext());
  // Use the same flattener to simplify each expression successively. This way
  // local identifiers / expressions are shared.
  for (auto expr : exprs) {
    if (!expr.isPureAffine())
      return false;

    flattener.walkPostOrder(expr);
  }

  assert(flattener.operandExprStack.size() == exprs.size());
  flattenedExprs->insert(flattenedExprs->end(),
                         flattener.operandExprStack.begin(),
                         flattener.operandExprStack.end());
  if (localVarCst)
    localVarCst->clearAndCopyFrom(flattener.localVarCst);

  return true;
}

// Flattens 'expr' into 'flattenedExpr'. Returns true on success or false
// if 'expr' was unable to be flattened (semi-affine expressions not handled
// yet).
bool mlir::getFlattenedAffineExpr(AffineExpr expr, unsigned numDims,
                                  unsigned numSymbols,
                                  llvm::SmallVectorImpl<int64_t> *flattenedExpr,
                                  FlatAffineConstraints *localVarCst) {
  std::vector<SmallVector<int64_t, 8>> flattenedExprs;
  bool ret = ::getFlattenedAffineExprs({expr}, numDims, numSymbols,
                                       &flattenedExprs, localVarCst);
  *flattenedExpr = flattenedExprs[0];
  return ret;
}

/// Flattens the expressions in map. Returns true on success or false
/// if 'expr' was unable to be flattened (i.e., semi-affine expressions not
/// handled yet).
bool mlir::getFlattenedAffineExprs(
    AffineMap map, std::vector<llvm::SmallVector<int64_t, 8>> *flattenedExprs,
    FlatAffineConstraints *localVarCst) {
  if (map.getNumResults() == 0) {
    localVarCst->reset(map.getNumDims(), map.getNumSymbols());
    return true;
  }
  return ::getFlattenedAffineExprs(map.getResults(), map.getNumDims(),
                                   map.getNumSymbols(), flattenedExprs,
                                   localVarCst);
}

bool mlir::getFlattenedAffineExprs(
    IntegerSet set, std::vector<llvm::SmallVector<int64_t, 8>> *flattenedExprs,
    FlatAffineConstraints *localVarCst) {
  if (set.getNumConstraints() == 0) {
    localVarCst->reset(set.getNumDims(), set.getNumSymbols());
    return true;
  }
  return ::getFlattenedAffineExprs(set.getConstraints(), set.getNumDims(),
                                   set.getNumSymbols(), flattenedExprs,
                                   localVarCst);
}

/// Returns the sequence of AffineApplyOp OperationInsts operation in
/// 'affineApplyOps', which are reachable via a search starting from 'operands',
/// and ending at operands which are not defined by AffineApplyOps.
// TODO(andydavis) Add a method to AffineApplyOp which forward substitutes
// the AffineApplyOp into any user AffineApplyOps.
void mlir::getReachableAffineApplyOps(
    ArrayRef<Value *> operands,
    SmallVectorImpl<OperationInst *> &affineApplyOps) {
  struct State {
    // The ssa value for this node in the DFS traversal.
    Value *value;
    // The operand index of 'value' to explore next during DFS traversal.
    unsigned operandIndex;
  };
  SmallVector<State, 4> worklist;
  for (auto *operand : operands) {
    worklist.push_back({operand, 0});
  }

  while (!worklist.empty()) {
    State &state = worklist.back();
    auto *opInst = state.value->getDefiningInst();
    // Note: getDefiningInst will return nullptr if the operand is not an
    // OperationInst (i.e. ForInst), which is a terminator for the search.
    if (opInst == nullptr || !opInst->isa<AffineApplyOp>()) {
      worklist.pop_back();
      continue;
    }
    if (auto affineApplyOp = opInst->dyn_cast<AffineApplyOp>()) {
      if (state.operandIndex == 0) {
        // Pre-Visit: Add 'opInst' to reachable sequence.
        affineApplyOps.push_back(opInst);
      }
      if (state.operandIndex < opInst->getNumOperands()) {
        // Visit: Add next 'affineApplyOp' operand to worklist.
        // Get next operand to visit at 'operandIndex'.
        auto *nextOperand = opInst->getOperand(state.operandIndex);
        // Increment 'operandIndex' in 'state'.
        ++state.operandIndex;
        // Add 'nextOperand' to worklist.
        worklist.push_back({nextOperand, 0});
      } else {
        // Post-visit: done visiting operands AffineApplyOp, pop off stack.
        worklist.pop_back();
      }
    }
  }
}

// Builds a system of constraints with dimensional identifiers corresponding to
// the loop IVs of the forInsts appearing in that order. Any symbols founds in
// the bound operands are added as symbols in the system. Returns false for the
// yet unimplemented cases.
// TODO(andydavis,bondhugula) Handle non-unit steps through local variables or
// stride information in FlatAffineConstraints. (For eg., by using iv - lb %
// step = 0 and/or by introducing a method in FlatAffineConstraints
// setExprStride(ArrayRef<int64_t> expr, int64_t stride)
bool mlir::getIndexSet(ArrayRef<ForInst *> forInsts,
                       FlatAffineConstraints *domain) {
  SmallVector<Value *, 4> indices(forInsts.begin(), forInsts.end());
  // Reset while associated Values in 'indices' to the domain.
  domain->reset(forInsts.size(), /*numSymbols=*/0, /*numLocals=*/0, indices);
  for (auto *forInst : forInsts) {
    // Add constraints from forInst's bounds.
    if (!domain->addForInstDomain(*forInst))
      return false;
  }
  return true;
}

// Computes the iteration domain for 'opInst' and populates 'indexSet', which
// encapsulates the constraints involving loops surrounding 'opInst' and
// potentially involving any Function symbols. The dimensional identifiers in
// 'indexSet' correspond to the loops surounding 'inst' from outermost to
// innermost.
// TODO(andydavis) Add support to handle IfInsts surrounding 'inst'.
static bool getInstIndexSet(const Instruction *inst,
                            FlatAffineConstraints *indexSet) {
  // TODO(andydavis) Extend this to gather enclosing IfInsts and consider
  // factoring it out into a utility function.
  SmallVector<ForInst *, 4> loops;
  getLoopIVs(*inst, &loops);
  return getIndexSet(loops, indexSet);
}

// ValuePositionMap manages the mapping from Values which represent dimension
// and symbol identifiers from 'src' and 'dst' access functions to positions
// in new space where some Values are kept separate (using addSrc/DstValue)
// and some Values are merged (addSymbolValue).
// Position lookups return the absolute position in the new space which
// has the following format:
//
//   [src-dim-identifiers] [dst-dim-identifiers] [symbol-identifers]
//
// Note: access function non-IV dimension identifiers (that have 'dimension'
// positions in the access function position space) are assigned as symbols
// in the output position space. Convienience access functions which lookup
// an Value in multiple maps are provided (i.e. getSrcDimOrSymPos) to handle
// the common case of resolving positions for all access function operands.
//
// TODO(andydavis) Generalize this: could take a template parameter for
// the number of maps (3 in the current case), and lookups could take indices
// of maps to check. So getSrcDimOrSymPos would be "getPos(value, {0, 2})".
class ValuePositionMap {
public:
  void addSrcValue(const Value *value) {
    if (addValueAt(value, &srcDimPosMap, numSrcDims))
      ++numSrcDims;
  }
  void addDstValue(const Value *value) {
    if (addValueAt(value, &dstDimPosMap, numDstDims))
      ++numDstDims;
  }
  void addSymbolValue(const Value *value) {
    if (addValueAt(value, &symbolPosMap, numSymbols))
      ++numSymbols;
  }
  unsigned getSrcDimOrSymPos(const Value *value) const {
    return getDimOrSymPos(value, srcDimPosMap, 0);
  }
  unsigned getDstDimOrSymPos(const Value *value) const {
    return getDimOrSymPos(value, dstDimPosMap, numSrcDims);
  }
  unsigned getSymPos(const Value *value) const {
    auto it = symbolPosMap.find(value);
    assert(it != symbolPosMap.end());
    return numSrcDims + numDstDims + it->second;
  }

  unsigned getNumSrcDims() const { return numSrcDims; }
  unsigned getNumDstDims() const { return numDstDims; }
  unsigned getNumDims() const { return numSrcDims + numDstDims; }
  unsigned getNumSymbols() const { return numSymbols; }

private:
  bool addValueAt(const Value *value, DenseMap<const Value *, unsigned> *posMap,
                  unsigned position) {
    auto it = posMap->find(value);
    if (it == posMap->end()) {
      (*posMap)[value] = position;
      return true;
    }
    return false;
  }
  unsigned getDimOrSymPos(const Value *value,
                          const DenseMap<const Value *, unsigned> &dimPosMap,
                          unsigned dimPosOffset) const {
    auto it = dimPosMap.find(value);
    if (it != dimPosMap.end()) {
      return dimPosOffset + it->second;
    }
    it = symbolPosMap.find(value);
    assert(it != symbolPosMap.end());
    return numSrcDims + numDstDims + it->second;
  }

  unsigned numSrcDims = 0;
  unsigned numDstDims = 0;
  unsigned numSymbols = 0;
  DenseMap<const Value *, unsigned> srcDimPosMap;
  DenseMap<const Value *, unsigned> dstDimPosMap;
  DenseMap<const Value *, unsigned> symbolPosMap;
};

// Builds a map from Value to identifier position in a new merged identifier
// list, which is the result of merging dim/symbol lists from src/dst
// iteration domains, the format of which is as follows:
//
//   [src-dim-identifiers, dst-dim-identifiers, symbol-identifiers, const_term]
//
// This method populates 'valuePosMap' with mappings from operand Values in
// 'srcAccessMap'/'dstAccessMap' (as well as those in 'srcDomain'/'dstDomain')
// to the position of these values in the merged list.
static void buildDimAndSymbolPositionMaps(
    const FlatAffineConstraints &srcDomain,
    const FlatAffineConstraints &dstDomain, const AffineValueMap &srcAccessMap,
    const AffineValueMap &dstAccessMap, ValuePositionMap *valuePosMap,
    FlatAffineConstraints *dependenceConstraints) {
  auto updateValuePosMap = [&](ArrayRef<Value *> values, bool isSrc) {
    for (unsigned i = 0, e = values.size(); i < e; ++i) {
      auto *value = values[i];
      if (!isa<ForInst>(values[i])) {
        assert(values[i]->isValidSymbol() &&
               "access operand has to be either a loop IV or a symbol");
        valuePosMap->addSymbolValue(value);
      } else if (isSrc) {
        valuePosMap->addSrcValue(value);
      } else {
        valuePosMap->addDstValue(value);
      }
    }
  };

  SmallVector<Value *, 4> srcValues, destValues;
  srcDomain.getAllIdValues(&srcValues);
  dstDomain.getAllIdValues(&destValues);

  // Update value position map with identifiers from src iteration domain.
  updateValuePosMap(srcValues, /*isSrc=*/true);
  // Update value position map with identifiers from dst iteration domain.
  updateValuePosMap(destValues, /*isSrc=*/false);
  // Update value position map with identifiers from src access function.
  updateValuePosMap(srcAccessMap.getOperands(), /*isSrc=*/true);
  // Update value position map with identifiers from dst access function.
  updateValuePosMap(dstAccessMap.getOperands(), /*isSrc=*/false);
}

// Sets up dependence constraints columns appropriately, in the format:
// [src-dim-identifiers, dst-dim-identifiers, symbol-identifiers, const_term]
void initDependenceConstraints(const FlatAffineConstraints &srcDomain,
                               const FlatAffineConstraints &dstDomain,
                               const AffineValueMap &srcAccessMap,
                               const AffineValueMap &dstAccessMap,
                               const ValuePositionMap &valuePosMap,
                               FlatAffineConstraints *dependenceConstraints) {
  // Calculate number of equalities/inequalities and columns required to
  // initialize FlatAffineConstraints for 'dependenceDomain'.
  unsigned numIneq =
      srcDomain.getNumInequalities() + dstDomain.getNumInequalities();
  AffineMap srcMap = srcAccessMap.getAffineMap();
  assert(srcMap.getNumResults() == dstAccessMap.getAffineMap().getNumResults());
  unsigned numEq = srcMap.getNumResults();
  unsigned numDims = srcDomain.getNumDimIds() + dstDomain.getNumDimIds();
  unsigned numSymbols = valuePosMap.getNumSymbols();
  unsigned numIds = numDims + numSymbols;
  unsigned numCols = numIds + 1;

  // Set flat affine constraints sizes and reserving space for constraints.
  dependenceConstraints->reset(numIneq, numEq, numCols, numDims, numSymbols,
                               /*numLocals=*/0);

  // Set values corresponding to dependence constraint identifiers.
  SmallVector<Value *, 4> srcLoopIVs, dstLoopIVs;
  srcDomain.getIdValues(0, srcDomain.getNumDimIds(), &srcLoopIVs);
  dstDomain.getIdValues(0, dstDomain.getNumDimIds(), &dstLoopIVs);

  dependenceConstraints->setIdValues(0, srcLoopIVs.size(), srcLoopIVs);
  dependenceConstraints->setIdValues(
      srcLoopIVs.size(), srcLoopIVs.size() + dstLoopIVs.size(), dstLoopIVs);

  // Set values for the symbolic identifier dimensions.
  auto setSymbolIds = [&](ArrayRef<Value *> values) {
    for (auto *value : values) {
      if (!isa<ForInst>(value)) {
        assert(value->isValidSymbol() && "expected symbol");
        dependenceConstraints->setIdValue(valuePosMap.getSymPos(value), value);
      }
    }
  };

  setSymbolIds(srcAccessMap.getOperands());
  setSymbolIds(dstAccessMap.getOperands());

  SmallVector<Value *, 8> srcSymbolValues, dstSymbolValues;
  srcDomain.getIdValues(srcDomain.getNumDimIds(),
                        srcDomain.getNumDimAndSymbolIds(), &srcSymbolValues);
  dstDomain.getIdValues(dstDomain.getNumDimIds(),
                        dstDomain.getNumDimAndSymbolIds(), &dstSymbolValues);
  setSymbolIds(srcSymbolValues);
  setSymbolIds(dstSymbolValues);

  for (unsigned i = 0, e = dependenceConstraints->getNumDimAndSymbolIds();
       i < e; i++)
    assert(dependenceConstraints->getIds()[i].hasValue());
}

// Adds iteration domain constraints from 'srcDomain' and 'dstDomain' into
// 'dependenceDomain'.
// Uses 'valuePosMap' to determine the position in 'dependenceDomain' to which a
// srcDomain/dstDomain Value maps.
static void addDomainConstraints(const FlatAffineConstraints &srcDomain,
                                 const FlatAffineConstraints &dstDomain,
                                 const ValuePositionMap &valuePosMap,
                                 FlatAffineConstraints *dependenceDomain) {
  unsigned srcNumIneq = srcDomain.getNumInequalities();
  unsigned srcNumDims = srcDomain.getNumDimIds();
  unsigned srcNumSymbols = srcDomain.getNumSymbolIds();
  unsigned srcNumIds = srcNumDims + srcNumSymbols;

  unsigned dstNumIneq = dstDomain.getNumInequalities();
  unsigned dstNumDims = dstDomain.getNumDimIds();
  unsigned dstNumSymbols = dstDomain.getNumSymbolIds();
  unsigned dstNumIds = dstNumDims + dstNumSymbols;

  SmallVector<int64_t, 4> ineq(dependenceDomain->getNumCols());
  // Add inequalities from src domain.
  for (unsigned i = 0; i < srcNumIneq; ++i) {
    // Zero fill.
    std::fill(ineq.begin(), ineq.end(), 0);
    // Set coefficients for identifiers corresponding to src domain.
    for (unsigned j = 0; j < srcNumIds; ++j)
      ineq[valuePosMap.getSrcDimOrSymPos(srcDomain.getIdValue(j))] =
          srcDomain.atIneq(i, j);
    // Set constant term.
    ineq[ineq.size() - 1] = srcDomain.atIneq(i, srcNumIds);
    // Add inequality constraint.
    dependenceDomain->addInequality(ineq);
  }
  // Add inequalities from dst domain.
  for (unsigned i = 0; i < dstNumIneq; ++i) {
    // Zero fill.
    std::fill(ineq.begin(), ineq.end(), 0);
    // Set coefficients for identifiers corresponding to dst domain.
    for (unsigned j = 0; j < dstNumIds; ++j)
      ineq[valuePosMap.getDstDimOrSymPos(dstDomain.getIdValue(j))] =
          dstDomain.atIneq(i, j);
    // Set constant term.
    ineq[ineq.size() - 1] = dstDomain.atIneq(i, dstNumIds);
    // Add inequality constraint.
    dependenceDomain->addInequality(ineq);
  }
}

// Adds equality constraints that equate src and dst access functions
// represented by 'srcAccessMap' and 'dstAccessMap' for each result.
// Requires that 'srcAccessMap' and 'dstAccessMap' have the same results count.
// For example, given the following two accesses functions to a 2D memref:
//
//   Source access function:
//     (a0 * d0 + a1 * s0 + a2, b0 * d0 + b1 * s0 + b2)
//
//   Destination acceses function:
//     (c0 * d0 + c1 * s0 + c2, f0 * d0 + f1 * s0 + f2)
//
// This method constructs the following equality constraints in
// 'dependenceDomain', by equating the access functions for each result
// (i.e. each memref dim). Notice that 'd0' for the destination access function
// is mapped into 'd0' in the equality constraint:
//
//   d0      d1      s0         c
//   --      --      --         --
//   a0     -c0      (a1 - c1)  (a1 - c2) = 0
//   b0     -f0      (b1 - f1)  (b1 - f2) = 0
//
// Returns false if any AffineExpr cannot be flattened (due to it being
// semi-affine). Returns true otherwise.
// TODO(bondhugula): assumes that dependenceDomain doesn't have local
// variables already. Fix this soon.
static bool
addMemRefAccessConstraints(const AffineValueMap &srcAccessMap,
                           const AffineValueMap &dstAccessMap,
                           const ValuePositionMap &valuePosMap,
                           FlatAffineConstraints *dependenceDomain) {
  if (dependenceDomain->getNumLocalIds() != 0)
    return false;
  AffineMap srcMap = srcAccessMap.getAffineMap();
  AffineMap dstMap = dstAccessMap.getAffineMap();
  assert(srcMap.getNumResults() == dstMap.getNumResults());
  unsigned numResults = srcMap.getNumResults();

  unsigned srcNumIds = srcMap.getNumDims() + srcMap.getNumSymbols();
  ArrayRef<Value *> srcOperands = srcAccessMap.getOperands();

  unsigned dstNumIds = dstMap.getNumDims() + dstMap.getNumSymbols();
  ArrayRef<Value *> dstOperands = dstAccessMap.getOperands();

  std::vector<SmallVector<int64_t, 8>> srcFlatExprs;
  std::vector<SmallVector<int64_t, 8>> destFlatExprs;
  FlatAffineConstraints srcLocalVarCst, destLocalVarCst;
  // Get flattened expressions for the source destination maps.
  if (!getFlattenedAffineExprs(srcMap, &srcFlatExprs, &srcLocalVarCst) ||
      !getFlattenedAffineExprs(dstMap, &destFlatExprs, &destLocalVarCst))
    return false;

  unsigned srcNumLocalIds = srcLocalVarCst.getNumLocalIds();
  unsigned dstNumLocalIds = destLocalVarCst.getNumLocalIds();
  unsigned numLocalIdsToAdd = srcNumLocalIds + dstNumLocalIds;
  for (unsigned i = 0; i < numLocalIdsToAdd; i++) {
    dependenceDomain->addLocalId(dependenceDomain->getNumLocalIds());
  }

  unsigned numDims = dependenceDomain->getNumDimIds();
  unsigned numSymbols = dependenceDomain->getNumSymbolIds();
  unsigned numSrcLocalIds = srcLocalVarCst.getNumLocalIds();

  // Equality to add.
  SmallVector<int64_t, 8> eq(dependenceDomain->getNumCols());
  for (unsigned i = 0; i < numResults; ++i) {
    // Zero fill.
    std::fill(eq.begin(), eq.end(), 0);

    // Flattened AffineExpr for src result 'i'.
    const auto &srcFlatExpr = srcFlatExprs[i];
    // Set identifier coefficients from src access function.
    for (unsigned j = 0, e = srcOperands.size(); j < e; ++j)
      eq[valuePosMap.getSrcDimOrSymPos(srcOperands[j])] = srcFlatExpr[j];
    // Local terms.
    for (unsigned j = 0, e = srcNumLocalIds; j < e; j++)
      eq[numDims + numSymbols + j] = srcFlatExpr[srcNumIds + j];
    // Set constant term.
    eq[eq.size() - 1] = srcFlatExpr[srcFlatExpr.size() - 1];

    // Flattened AffineExpr for dest result 'i'.
    const auto &destFlatExpr = destFlatExprs[i];
    // Set identifier coefficients from dst access function.
    for (unsigned j = 0, e = dstOperands.size(); j < e; ++j)
      eq[valuePosMap.getDstDimOrSymPos(dstOperands[j])] -= destFlatExpr[j];
    // Local terms.
    for (unsigned j = 0, e = dstNumLocalIds; j < e; j++)
      eq[numDims + numSymbols + numSrcLocalIds + j] =
          -destFlatExpr[dstNumIds + j];
    // Set constant term.
    eq[eq.size() - 1] -= destFlatExpr[destFlatExpr.size() - 1];

    // Add equality constraint.
    dependenceDomain->addEquality(eq);
  }

  // Add equality constraints for any operands that are defined by constant ops.
  auto addEqForConstOperands = [&](ArrayRef<const Value *> operands) {
    for (unsigned i = 0, e = operands.size(); i < e; ++i) {
      if (isa<ForInst>(operands[i]))
        continue;
      auto *symbol = operands[i];
      assert(symbol->isValidSymbol());
      // Check if the symbol is a constant.
      if (auto *opInst = symbol->getDefiningInst()) {
        if (auto constOp = opInst->dyn_cast<ConstantIndexOp>()) {
          dependenceDomain->setIdToConstant(valuePosMap.getSymPos(symbol),
                                            constOp->getValue());
        }
      }
    }
  };

  // Add equality constraints for any src symbols defined by constant ops.
  addEqForConstOperands(srcOperands);
  // Add equality constraints for any dst symbols defined by constant ops.
  addEqForConstOperands(dstOperands);

  // By construction (see flattener), local var constraints will not have any
  // equalities.
  assert(srcLocalVarCst.getNumEqualities() == 0 &&
         destLocalVarCst.getNumEqualities() == 0);
  // Add inequalities from srcLocalVarCst and destLocalVarCst into the
  // dependence domain.
  SmallVector<int64_t, 8> ineq(dependenceDomain->getNumCols());
  for (unsigned r = 0, e = srcLocalVarCst.getNumInequalities(); r < e; r++) {
    std::fill(ineq.begin(), ineq.end(), 0);

    // Set identifier coefficients from src local var constraints.
    for (unsigned j = 0, e = srcOperands.size(); j < e; ++j)
      ineq[valuePosMap.getSrcDimOrSymPos(srcOperands[j])] =
          srcLocalVarCst.atIneq(r, j);
    // Local terms.
    for (unsigned j = 0, e = srcNumLocalIds; j < e; j++)
      ineq[numDims + numSymbols + j] = srcLocalVarCst.atIneq(r, srcNumIds + j);
    // Set constant term.
    ineq[ineq.size() - 1] =
        srcLocalVarCst.atIneq(r, srcLocalVarCst.getNumCols() - 1);
    dependenceDomain->addInequality(ineq);
  }

  for (unsigned r = 0, e = destLocalVarCst.getNumInequalities(); r < e; r++) {
    std::fill(ineq.begin(), ineq.end(), 0);
    // Set identifier coefficients from dest local var constraints.
    for (unsigned j = 0, e = dstOperands.size(); j < e; ++j)
      ineq[valuePosMap.getDstDimOrSymPos(dstOperands[j])] =
          destLocalVarCst.atIneq(r, j);
    // Local terms.
    for (unsigned j = 0, e = dstNumLocalIds; j < e; j++)
      ineq[numDims + numSymbols + numSrcLocalIds + j] =
          destLocalVarCst.atIneq(r, dstNumIds + j);
    // Set constant term.
    ineq[ineq.size() - 1] =
        destLocalVarCst.atIneq(r, destLocalVarCst.getNumCols() - 1);

    dependenceDomain->addInequality(ineq);
  }
  return true;
}

// Returns the number of outer loop common to 'src/dstDomain'.
static unsigned getNumCommonLoops(const FlatAffineConstraints &srcDomain,
                                  const FlatAffineConstraints &dstDomain) {
  // Find the number of common loops shared by src and dst accesses.
  unsigned minNumLoops =
      std::min(srcDomain.getNumDimIds(), dstDomain.getNumDimIds());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if (!isa<ForInst>(srcDomain.getIdValue(i)) ||
        !isa<ForInst>(dstDomain.getIdValue(i)) ||
        srcDomain.getIdValue(i) != dstDomain.getIdValue(i))
      break;
    ++numCommonLoops;
  }
  return numCommonLoops;
}

// Returns Block common to 'srcAccess.opInst' and 'dstAccess.opInst'.
static const Block *getCommonBlock(const MemRefAccess &srcAccess,
                                   const MemRefAccess &dstAccess,
                                   const FlatAffineConstraints &srcDomain,
                                   unsigned numCommonLoops) {
  if (numCommonLoops == 0) {
    auto *block = srcAccess.opInst->getBlock();
    while (block->getContainingInst()) {
      block = block->getContainingInst()->getBlock();
    }
    return block;
  }
  auto *commonForValue = srcDomain.getIdValue(numCommonLoops - 1);
  assert(isa<ForInst>(commonForValue));
  return cast<ForInst>(commonForValue)->getBody();
}

// Returns true if the ancestor operation instruction of 'srcAccess' appears
// before the ancestor operation instruction of 'dstAccess' in the same
// instruction block. Returns false otherwise.
// Note that because 'srcAccess' or 'dstAccess' may be nested in conditionals,
// the function is named 'srcAppearsBeforeDstInCommonBlock'.
// Note that 'numCommonLoops' is the number of contiguous surrounding outer
// loops.
static bool srcAppearsBeforeDstInCommonBlock(
    const MemRefAccess &srcAccess, const MemRefAccess &dstAccess,
    const FlatAffineConstraints &srcDomain, unsigned numCommonLoops) {
  // Get Block common to 'srcAccess.opInst' and 'dstAccess.opInst'.
  auto *commonBlock =
      getCommonBlock(srcAccess, dstAccess, srcDomain, numCommonLoops);
  // Check the dominance relationship between the respective ancestors of the
  // src and dst in the Block of the innermost among the common loops.
  auto *srcInst = commonBlock->findAncestorInstInBlock(*srcAccess.opInst);
  assert(srcInst != nullptr);
  auto *dstInst = commonBlock->findAncestorInstInBlock(*dstAccess.opInst);
  assert(dstInst != nullptr);

  // Do a linear scan to determine whether dstInst comes after srcInst.
  auto aIter = Block::const_iterator(srcInst);
  auto bIter = Block::const_iterator(dstInst);
  auto aBlockStart = srcInst->getBlock()->begin();
  while (bIter != aBlockStart) {
    --bIter;
    if (bIter == aIter)
      return true;
  }
  return false;
}

// Adds ordering constraints to 'dependenceDomain' based on number of loops
// common to 'src/dstDomain' and requested 'loopDepth'.
// Note that 'loopDepth' cannot exceed the number of common loops plus one.
// EX: Given a loop nest of depth 2 with IVs 'i' and 'j':
// *) If 'loopDepth == 1' then one constraint is added: i' >= i + 1
// *) If 'loopDepth == 2' then two constraints are added: i == i' and j' > j + 1
// *) If 'loopDepth == 3' then two constraints are added: i == i' and j == j'
static void addOrderingConstraints(const FlatAffineConstraints &srcDomain,
                                   const FlatAffineConstraints &dstDomain,
                                   unsigned loopDepth,
                                   FlatAffineConstraints *dependenceDomain) {
  unsigned numCols = dependenceDomain->getNumCols();
  SmallVector<int64_t, 4> eq(numCols);
  unsigned numSrcDims = srcDomain.getNumDimIds();
  unsigned numCommonLoops = getNumCommonLoops(srcDomain, dstDomain);
  unsigned numCommonLoopConstraints = std::min(numCommonLoops, loopDepth);
  for (unsigned i = 0; i < numCommonLoopConstraints; ++i) {
    std::fill(eq.begin(), eq.end(), 0);
    eq[i] = -1;
    eq[i + numSrcDims] = 1;
    if (i == loopDepth - 1) {
      eq[numCols - 1] = -1;
      dependenceDomain->addInequality(eq);
    } else {
      dependenceDomain->addEquality(eq);
    }
  }
}

// Returns true if 'isEq' constraint in 'dependenceDomain' has a single
// non-zero coefficient at (rowIdx, idPos). Returns false otherwise.
// TODO(andydavis) Move this function to FlatAffineConstraints.
static bool hasSingleNonZeroAt(unsigned idPos, unsigned rowIdx, bool isEq,
                               FlatAffineConstraints *dependenceDomain) {
  unsigned numCols = dependenceDomain->getNumCols();
  for (unsigned j = 0; j < numCols - 1; ++j) {
    int64_t v = isEq ? dependenceDomain->atEq(rowIdx, j)
                     : dependenceDomain->atIneq(rowIdx, j);
    if ((j == idPos && v == 0) || (j != idPos && v != 0))
      return false;
  }
  return true;
}

// Computes distance and direction vectors in 'dependences', by adding
// variables to 'dependenceDomain' which represent the difference of the IVs,
// eliminating all other variables, and reading off distance vectors from
// equality constraints (if possible), and direction vectors from inequalities.
static void computeDirectionVector(
    const FlatAffineConstraints &srcDomain,
    const FlatAffineConstraints &dstDomain, unsigned loopDepth,
    FlatAffineConstraints *dependenceDomain,
    llvm::SmallVector<DependenceComponent, 2> *dependenceComponents) {
  // Find the number of common loops shared by src and dst accesses.
  unsigned numCommonLoops = getNumCommonLoops(srcDomain, dstDomain);
  if (numCommonLoops == 0)
    return;
  // Compute direction vectors for requested loop depth.
  unsigned numIdsToEliminate = dependenceDomain->getNumIds();
  // Add new variables to 'dependenceDomain' to represent the direction
  // constraints for each shared loop.
  for (unsigned j = 0; j < numCommonLoops; ++j) {
    dependenceDomain->addDimId(j);
  }

  // Add equality contraints for each common loop, setting newly introduced
  // variable at column 'j' to the 'dst' IV minus the 'src IV.
  SmallVector<int64_t, 4> eq;
  eq.resize(dependenceDomain->getNumCols());
  for (unsigned j = 0; j < numCommonLoops; ++j) {
    std::fill(eq.begin(), eq.end(), 0);
    eq[j] = 1;
    eq[j + numCommonLoops] = 1;
    eq[j + 2 * numCommonLoops] = -1;
    dependenceDomain->addEquality(eq);
  }

  // Eliminate all variables other than the direction variables just added.
  dependenceDomain->projectOut(numCommonLoops, numIdsToEliminate);

  // Scan each common loop variable column and add direction vectors based
  // on eliminated constraint system.
  unsigned numCols = dependenceDomain->getNumCols();
  dependenceComponents->reserve(numCommonLoops);
  for (unsigned j = 0; j < numCommonLoops; ++j) {
    DependenceComponent depComp;
    for (unsigned i = 0, e = dependenceDomain->getNumEqualities(); i < e; ++i) {
      // Check for equality constraint with single non-zero in column 'j'.
      if (!hasSingleNonZeroAt(j, i, /*isEq=*/true, dependenceDomain))
        continue;
      // Get direction variable coefficient at (i, j).
      int64_t d = dependenceDomain->atEq(i, j);
      // Get constant coefficient at (i, numCols - 1).
      int64_t c = -dependenceDomain->atEq(i, numCols - 1);
      assert(c % d == 0 && "No dependence should have existed");
      depComp.lb = depComp.ub = c / d;
      dependenceComponents->push_back(depComp);
      break;
    }
    // Skip checking inequalities if we set 'depComp' based on equalities.
    if (depComp.lb.hasValue() || depComp.ub.hasValue())
      continue;
    // TODO(andydavis) Call FlatAffineConstraints::getConstantLower/UpperBound
    // Check inequalities to track direction range for each 'j'.
    for (unsigned i = 0, e = dependenceDomain->getNumInequalities(); i < e;
         ++i) {
      // Check for inequality constraint with single non-zero in column 'j'.
      if (!hasSingleNonZeroAt(j, i, /*isEq=*/false, dependenceDomain))
        continue;
      // Get direction variable coefficient at (i, j).
      int64_t d = dependenceDomain->atIneq(i, j);
      // Get constant coefficient at (i, numCols - 1).
      int64_t c = dependenceDomain->atIneq(i, numCols - 1);
      if (d < 0) {
        // Upper bound: add tightest upper bound.
        auto ub = mlir::floorDiv(c, -d);
        if (!depComp.ub.hasValue() || ub < depComp.ub.getValue())
          depComp.ub = ub;
      } else {
        // Lower bound: add tightest lower bound.
        auto lb = mlir::ceilDiv(-c, d);
        if (!depComp.lb.hasValue() || lb > depComp.lb.getValue())
          depComp.lb = lb;
      }
    }
    if (depComp.lb.hasValue() || depComp.ub.hasValue()) {
      if (depComp.lb.hasValue() && depComp.ub.hasValue())
        assert(depComp.lb.getValue() <= depComp.ub.getValue());
      dependenceComponents->push_back(depComp);
    }
  }
}

// Populates 'accessMap' with composition of AffineApplyOps reachable from
// indices of MemRefAccess.
void MemRefAccess::getAccessMap(AffineValueMap *accessMap) const {
  auto memrefType = memref->getType().cast<MemRefType>();
  // Create identity map with same number of dimensions as 'memrefType' rank.
  auto map = AffineMap::getMultiDimIdentityMap(memrefType.getRank(),
                                               memref->getType().getContext());
  SmallVector<Value *, 8> operands(indices.begin(), indices.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  accessMap->reset(map, operands);
}

// Builds a flat affine constraint system to check if there exists a dependence
// between memref accesses 'srcAccess' and 'dstAccess'.
// Returns 'false' if the accesses can be definitively shown not to access the
// same element. Returns 'true' otherwise.
// If a dependence exists, returns in 'dependenceComponents' a direction
// vector for the dependence, with a component for each loop IV in loops
// common to both accesses (see Dependence in AffineAnalysis.h for details).
//
// The memref access dependence check is comprised of the following steps:
// *) Compute access functions for each access. Access functions are computed
//    using AffineValueMaps initialized with the indices from an access, then
//    composed with AffineApplyOps reachable from operands of that access,
//    until operands of the AffineValueMap are loop IVs or symbols.
// *) Build iteration domain constraints for each access. Iteration domain
//    constraints are pairs of inequality contraints representing the
//    upper/lower loop bounds for each ForInst in the loop nest associated
//    with each access.
// *) Build dimension and symbol position maps for each access, which map
//    Values from access functions and iteration domains to their position
//    in the merged constraint system built by this method.
//
// This method builds a constraint system with the following column format:
//
//  [src-dim-identifiers, dst-dim-identifiers, symbols, constant]
//
// For example, given the following MLIR code with with "source" and
// "destination" accesses to the same memref labled, and symbols %M, %N, %K:
//
//   for %i0 = 0 to 100 {
//     for %i1 = 0 to 50 {
//       %a0 = affine_apply
//         (d0, d1) -> (d0 * 2 - d1 * 4 + s1, d1 * 3 - s0) (%i0, %i1)[%M, %N]
//       // Source memref access.
//       store %v0, %m[%a0#0, %a0#1] : memref<4x4xf32>
//     }
//   }
//
//   for %i2 = 0 to 100 {
//     for %i3 = 0 to 50 {
//       %a1 = affine_apply
//         (d0, d1) -> (d0 * 7 + d1 * 9 - s1, d1 * 11 + s0) (%i2, %i3)[%K, %M]
//       // Destination memref access.
//       %v1 = load %m[%a1#0, %a1#1] : memref<4x4xf32>
//     }
//   }
//
// The access functions would be the following:
//
//   src: (%i0 * 2 - %i1 * 4 + %N, %i1 * 3 - %M)
//   dst: (%i2 * 7 + %i3 * 9 - %M, %i3 * 11 - %K)
//
// The iteration domains for the src/dst accesses would be the following:
//
//   src: 0 <= %i0 <= 100, 0 <= %i1 <= 50
//   dst: 0 <= %i2 <= 100, 0 <= %i3 <= 50
//
// The symbols by both accesses would be assigned to a canonical position order
// which will be used in the dependence constraint system:
//
//   symbol name: %M  %N  %K
//   symbol  pos:  0   1   2
//
// Equality constraints are built by equating each result of src/destination
// access functions. For this example, the following two equality constraints
// will be added to the dependence constraint system:
//
//   [src_dim0, src_dim1, dst_dim0, dst_dim1, sym0, sym1, sym2, const]
//      2         -4        -7        -9       1      1     0     0    = 0
//      0          3         0        -11     -1      0     1     0    = 0
//
// Inequality constraints from the iteration domain will be meged into
// the dependence constraint system
//
//   [src_dim0, src_dim1, dst_dim0, dst_dim1, sym0, sym1, sym2, const]
//       1         0         0         0        0     0     0     0    >= 0
//      -1         0         0         0        0     0     0     100  >= 0
//       0         1         0         0        0     0     0     0    >= 0
//       0        -1         0         0        0     0     0     50   >= 0
//       0         0         1         0        0     0     0     0    >= 0
//       0         0        -1         0        0     0     0     100  >= 0
//       0         0         0         1        0     0     0     0    >= 0
//       0         0         0        -1        0     0     0     50   >= 0
//
//
// TODO(andydavis) Support AffineExprs mod/floordiv/ceildiv.
bool mlir::checkMemrefAccessDependence(
    const MemRefAccess &srcAccess, const MemRefAccess &dstAccess,
    unsigned loopDepth, FlatAffineConstraints *dependenceConstraints,
    llvm::SmallVector<DependenceComponent, 2> *dependenceComponents) {
  LLVM_DEBUG(llvm::dbgs() << "Checking for dependence at depth: "
                          << Twine(loopDepth) << " between:\n";);
  LLVM_DEBUG(srcAccess.opInst->dump(););
  LLVM_DEBUG(dstAccess.opInst->dump(););

  // Return 'false' if these accesses do not acces the same memref.
  if (srcAccess.memref != dstAccess.memref)
    return false;
  // Return 'false' if one of these accesses is not a StoreOp.
  if (!srcAccess.opInst->isa<StoreOp>() && !dstAccess.opInst->isa<StoreOp>())
    return false;

  // Get composed access function for 'srcAccess'.
  AffineValueMap srcAccessMap;
  srcAccess.getAccessMap(&srcAccessMap);

  // Get composed access function for 'dstAccess'.
  AffineValueMap dstAccessMap;
  dstAccess.getAccessMap(&dstAccessMap);

  // Get iteration domain for the 'srcAccess' instruction.
  FlatAffineConstraints srcDomain;
  if (!getInstIndexSet(srcAccess.opInst, &srcDomain))
    return false;

  // Get iteration domain for 'dstAccess' instruction.
  FlatAffineConstraints dstDomain;
  if (!getInstIndexSet(dstAccess.opInst, &dstDomain))
    return false;

  // Return 'false' if loopDepth > numCommonLoops and if the ancestor operation
  // instruction of 'srcAccess' does not properly dominate the ancestor
  // operation instruction of 'dstAccess' in the same common instruction block.
  unsigned numCommonLoops = getNumCommonLoops(srcDomain, dstDomain);
  assert(loopDepth <= numCommonLoops + 1);
  if (loopDepth > numCommonLoops &&
      !srcAppearsBeforeDstInCommonBlock(srcAccess, dstAccess, srcDomain,
                                        numCommonLoops)) {
    return false;
  }
  // Build dim and symbol position maps for each access from access operand
  // Value to position in merged contstraint system.
  ValuePositionMap valuePosMap;
  buildDimAndSymbolPositionMaps(srcDomain, dstDomain, srcAccessMap,
                                dstAccessMap, &valuePosMap,
                                dependenceConstraints);

  initDependenceConstraints(srcDomain, dstDomain, srcAccessMap, dstAccessMap,
                            valuePosMap, dependenceConstraints);

  assert(valuePosMap.getNumDims() ==
         srcDomain.getNumDimIds() + dstDomain.getNumDimIds());

  // Create memref access constraint by equating src/dst access functions.
  // Note that this check is conservative, and will failure in the future
  // when local variables for mod/div exprs are supported.
  if (!addMemRefAccessConstraints(srcAccessMap, dstAccessMap, valuePosMap,
                                  dependenceConstraints))
    return true;

  // Add 'src' happens before 'dst' ordering constraints.
  addOrderingConstraints(srcDomain, dstDomain, loopDepth,
                         dependenceConstraints);
  // Add src and dst domain constraints.
  addDomainConstraints(srcDomain, dstDomain, valuePosMap,
                       dependenceConstraints);

  // Return false if the solution space is empty: no dependence.
  if (dependenceConstraints->isEmpty()) {
    return false;
  }

  // Compute dependence direction vector and return true.
  if (dependenceComponents != nullptr) {
    computeDirectionVector(srcDomain, dstDomain, loopDepth,
                           dependenceConstraints, dependenceComponents);
  }

  LLVM_DEBUG(llvm::dbgs() << "Dependence polyhedron:\n");
  LLVM_DEBUG(dependenceConstraints->dump());
  return true;
}

namespace {

/// An `AffineNormalizer` is a helper class that is not visible to the user and
/// supports renumbering operands of AffineApplyOp.
/// This acts as a reindexing map of Value* to positional dims or symbols and
/// allows simplifications such as:
///
/// ```mlir
///    %1 = affine_apply (d0, d1) -> (d0 - d1) (%0, %0)
/// ```
///
/// into:
///
/// ```mlir
///    %1 = affine_apply () -> (0)
/// ```
struct AffineNormalizer {
  AffineNormalizer(AffineMap map, ArrayRef<Value *> operands);

  /// Returns the AffineMap resulting from normalization.
  AffineMap getAffineMap() { return affineMap; }

  SmallVector<Value *, 8> getOperands() {
    SmallVector<Value *, 8> res(reorderedDims);
    res.append(concatenatedSymbols.begin(), concatenatedSymbols.end());
    return res;
  }

private:
  /// Helper function to insert `v` into the coordinate system of the current
  /// AffineNormalizer. Returns the AffineDimExpr with the corresponding
  /// renumbered position.
  AffineDimExpr applyOneDim(Value *v);

  /// Given an `other` normalizer, this rewrites `other.affineMap` in the
  /// coordinate system of the current AffineNormalizer.
  /// Returns the rewritten AffineMap and updates the dims and symbols of
  /// `this`.
  AffineMap renumber(const AffineNormalizer &other);

  /// Given an `app`, rewrites `app.getAffineMap()` in the coordinate system of
  /// the current AffineNormalizer.
  /// Returns the rewritten AffineMap and updates the dims and symbols of
  /// `this`.
  AffineMap renumber(const AffineApplyOp &app);

  /// Maps of Value* to position in `affineMap`.
  DenseMap<Value *, unsigned> dimValueToPosition;

  /// Ordered dims and symbols matching positional dims and symbols in
  /// `affineMap`.
  SmallVector<Value *, 8> reorderedDims;
  SmallVector<Value *, 8> concatenatedSymbols;

  AffineMap affineMap;

  /// Used with RAII to control the depth at which AffineApply are composed
  /// recursively. Only accepts depth 1 for now.
  /// Note that if one wishes to compose all AffineApply in the program and
  /// follows program order, maxdepth 1 is sufficient. This is as much as this
  /// abstraction is willing to support for now.
  static unsigned &affineApplyDepth() {
    static thread_local unsigned depth = 0;
    return depth;
  }
  static constexpr unsigned kMaxAffineApplyDepth = 1;

  AffineNormalizer() { affineApplyDepth()++; }

public:
  ~AffineNormalizer() { affineApplyDepth()--; }
};

} // namespace

AffineDimExpr AffineNormalizer::applyOneDim(Value *v) {
  DenseMap<Value *, unsigned>::iterator iterPos;
  bool inserted = false;
  std::tie(iterPos, inserted) =
      dimValueToPosition.insert(std::make_pair(v, dimValueToPosition.size()));
  if (inserted) {
    reorderedDims.push_back(v);
  }
  return getAffineDimExpr(iterPos->second, v->getFunction()->getContext())
      .cast<AffineDimExpr>();
}

AffineMap AffineNormalizer::renumber(const AffineNormalizer &other) {
  SmallVector<AffineExpr, 8> dimRemapping;
  for (auto *v : other.reorderedDims) {
    auto kvp = other.dimValueToPosition.find(v);
    if (dimRemapping.size() <= kvp->second)
      dimRemapping.resize(kvp->second + 1);
    dimRemapping[kvp->second] = applyOneDim(kvp->first);
  }
  unsigned numSymbols = concatenatedSymbols.size();
  unsigned numOtherSymbols = other.concatenatedSymbols.size();
  SmallVector<AffineExpr, 8> symRemapping(numOtherSymbols);
  for (unsigned idx = 0; idx < numOtherSymbols; ++idx) {
    symRemapping[idx] =
        getAffineSymbolExpr(idx + numSymbols, other.affineMap.getContext());
  }
  concatenatedSymbols.insert(concatenatedSymbols.end(),
                             other.concatenatedSymbols.begin(),
                             other.concatenatedSymbols.end());
  auto map = other.affineMap;
  return map.replaceDimsAndSymbols(dimRemapping, symRemapping,
                                   dimRemapping.size(), symRemapping.size());
}

AffineMap AffineNormalizer::renumber(const AffineApplyOp &app) {
  assert(app.getAffineMap().getRangeSizes().empty() && "Non-empty range sizes");

  // Create the AffineNormalizer for the operands of this
  // AffineApplyOp and combine it with the current AffineNormalizer.
  SmallVector<Value *, 8> operands(
      const_cast<AffineApplyOp &>(app).getOperands().begin(),
      const_cast<AffineApplyOp &>(app).getOperands().end());
  AffineNormalizer normalizer(app.getAffineMap(), operands);
  return renumber(normalizer);
}

static unsigned getIndexOf(Value *v, const AffineApplyOp &op) {
  unsigned numResults = op.getNumResults();
  for (unsigned i = 0; i < numResults; ++i) {
    if (v == op.getResult(i)) {
      return i;
    }
  }
  llvm_unreachable("value is not a result of AffineApply");
  return static_cast<unsigned>(-1);
}

AffineNormalizer::AffineNormalizer(AffineMap map, ArrayRef<Value *> operands)
    : AffineNormalizer() {
  assert(map.getRangeSizes().empty() && "Unbounded map expected");
  assert(map.getNumInputs() == operands.size() &&
         "number of operands does not match the number of map inputs");

  if (operands.empty()) {
    return;
  }

  SmallVector<AffineExpr, 8> exprs;
  for (auto en : llvm::enumerate(operands)) {
    auto *t = en.value();
    assert(t->getType().isIndex());
    bool operandNotFromAffineApply =
        !t->getDefiningInst() || !t->getDefiningInst()->isa<AffineApplyOp>();
    if (operandNotFromAffineApply ||
        affineApplyDepth() > kMaxAffineApplyDepth) {
      if (en.index() < map.getNumDims()) {
        exprs.push_back(applyOneDim(t));
      } else {
        concatenatedSymbols.push_back(t);
      }
    } else {
      auto *inst = t->getDefiningInst();
      auto app = inst->dyn_cast<AffineApplyOp>();
      unsigned idx = getIndexOf(t, *app);
      auto tmpMap = renumber(*app);
      exprs.push_back(tmpMap.getResult(idx));
    }
  }

  auto numDims = dimValueToPosition.size();
  auto numSymbols = concatenatedSymbols.size() - map.getNumSymbols();
  auto exprsMap = AffineMap::get(numDims, numSymbols, exprs, {});
  LLVM_DEBUG(map.print(dbgs() << "\nCompose map: "));
  LLVM_DEBUG(exprsMap.print(dbgs() << "\nWith map: "));

  affineMap = simplifyAffineMap(map.compose(exprsMap));
  LLVM_DEBUG(affineMap.print(dbgs() << "\nResult: "));
}

/// Implements `map` and `operands` composition and simplification to support
/// `makeComposedAffineApply`. This can be called to achieve the same effects
/// on `map` and `operands` without creating an AffineApplyOp that needs to be
/// immediately deleted.
static void composeAffineMapAndOperands(AffineMap *map,
                                        SmallVectorImpl<Value *> *operands) {
  AffineNormalizer normalizer(*map, *operands);
  auto normalizedMap = normalizer.getAffineMap();
  auto normalizedOperands = normalizer.getOperands();
  canonicalizeMapAndOperands(&normalizedMap, &normalizedOperands);
  *map = normalizedMap;
  *operands = normalizedOperands;
  assert(*map);
}

void mlir::fullyComposeAffineMapAndOperands(
    AffineMap *map, SmallVectorImpl<Value *> *operands) {
  while (llvm::any_of(*operands, [](Value *v) {
    return v->getDefiningInst() && v->getDefiningInst()->isa<AffineApplyOp>();
  })) {
    composeAffineMapAndOperands(map, operands);
  }
}

OpPointer<AffineApplyOp>
mlir::makeComposedAffineApply(FuncBuilder *b, Location loc, AffineMap map,
                              ArrayRef<Value *> operands) {
  AffineMap normalizedMap = map;
  SmallVector<Value *, 8> normalizedOperands(operands.begin(), operands.end());
  composeAffineMapAndOperands(&normalizedMap, &normalizedOperands);
  assert(normalizedMap);
  return b->create<AffineApplyOp>(loc, normalizedMap, normalizedOperands);
}

Value *
mlir::makeSingleValueFromComposedAffineApply(FuncBuilder *b, Location loc,
                                             AffineMap map,
                                             llvm::ArrayRef<Value *> operands) {
  auto app = mlir::makeComposedAffineApply(b, loc, map, operands);
  return app->getResult(0);
}
