//===- AffineExpr.cpp - MLIR Affine Expr Classes --------------------------===//
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

#include "mlir/IR/AffineExpr.h"
#include "AffineExprDetail.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AffineStructures.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::detail;

MLIRContext *AffineExpr::getContext() const {
  return expr->contextAndKind.getPointer();
}

AffineExprKind AffineExpr::getKind() const {
  return expr->contextAndKind.getInt();
}

/// Walk all of the AffineExprs in this subgraph in postorder.
void AffineExpr::walk(std::function<void(AffineExpr)> callback) const {
  struct AffineExprWalker : public AffineExprVisitor<AffineExprWalker> {
    std::function<void(AffineExpr)> callback;

    AffineExprWalker(std::function<void(AffineExpr)> callback)
        : callback(callback) {}

    void visitAffineBinaryOpExpr(AffineBinaryOpExpr expr) { callback(expr); }
    void visitConstantExpr(AffineConstantExpr expr) { callback(expr); }
    void visitDimExpr(AffineDimExpr expr) { callback(expr); }
    void visitSymbolExpr(AffineSymbolExpr expr) { callback(expr); }
  };

  AffineExprWalker(callback).walkPostOrder(*this);
}

/// This method substitutes any uses of dimensions and symbols (e.g.
/// dim#0 with dimReplacements[0]) and returns the modified expression tree.
AffineExpr
AffineExpr::replaceDimsAndSymbols(ArrayRef<AffineExpr> dimReplacements,
                                  ArrayRef<AffineExpr> symReplacements) const {
  switch (getKind()) {
  case AffineExprKind::Constant:
    return *this;
  case AffineExprKind::DimId: {
    unsigned dimId = cast<AffineDimExpr>().getPosition();
    if (dimId >= dimReplacements.size())
      return *this;
    return dimReplacements[dimId];
  }
  case AffineExprKind::SymbolId: {
    unsigned symId = cast<AffineSymbolExpr>().getPosition();
    if (symId >= symReplacements.size())
      return *this;
    return symReplacements[symId];
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod:
    auto binOp = cast<AffineBinaryOpExpr>();
    auto lhs = binOp.getLHS(), rhs = binOp.getRHS();
    auto newLHS = lhs.replaceDimsAndSymbols(dimReplacements, symReplacements);
    auto newRHS = rhs.replaceDimsAndSymbols(dimReplacements, symReplacements);
    if (newLHS == lhs && newRHS == rhs)
      return *this;
    return getAffineBinaryOpExpr(getKind(), newLHS, newRHS);
  }
}

/// Returns true if this expression is made out of only symbols and
/// constants (no dimensional identifiers).
bool AffineExpr::isSymbolicOrConstant() const {
  switch (getKind()) {
  case AffineExprKind::Constant:
    return true;
  case AffineExprKind::DimId:
    return false;
  case AffineExprKind::SymbolId:
    return true;

  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    auto expr = this->cast<AffineBinaryOpExpr>();
    return expr.getLHS().isSymbolicOrConstant() &&
           expr.getRHS().isSymbolicOrConstant();
  }
  }
}

/// Returns true if this is a pure affine expression, i.e., multiplication,
/// floordiv, ceildiv, and mod is only allowed w.r.t constants.
bool AffineExpr::isPureAffine() const {
  switch (getKind()) {
  case AffineExprKind::SymbolId:
  case AffineExprKind::DimId:
  case AffineExprKind::Constant:
    return true;
  case AffineExprKind::Add: {
    auto op = cast<AffineBinaryOpExpr>();
    return op.getLHS().isPureAffine() && op.getRHS().isPureAffine();
  }

  case AffineExprKind::Mul: {
    // TODO: Canonicalize the constants in binary operators to the RHS when
    // possible, allowing this to merge into the next case.
    auto op = cast<AffineBinaryOpExpr>();
    return op.getLHS().isPureAffine() && op.getRHS().isPureAffine() &&
           (op.getLHS().template isa<AffineConstantExpr>() ||
            op.getRHS().template isa<AffineConstantExpr>());
  }
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    auto op = cast<AffineBinaryOpExpr>();
    return op.getLHS().isPureAffine() &&
           op.getRHS().template isa<AffineConstantExpr>();
  }
  }
}

/// Returns the greatest known integral divisor of this affine expression.
uint64_t AffineExpr::getLargestKnownDivisor() const {
  AffineBinaryOpExpr binExpr(nullptr);
  switch (getKind()) {
  case AffineExprKind::SymbolId:
    LLVM_FALLTHROUGH;
  case AffineExprKind::DimId:
    return 1;
  case AffineExprKind::Constant:
    return std::abs(this->cast<AffineConstantExpr>().getValue());
  case AffineExprKind::Mul: {
    binExpr = this->cast<AffineBinaryOpExpr>();
    return binExpr.getLHS().getLargestKnownDivisor() *
           binExpr.getRHS().getLargestKnownDivisor();
  }
  case AffineExprKind::Add:
    LLVM_FALLTHROUGH;
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    binExpr = cast<AffineBinaryOpExpr>();
    return llvm::GreatestCommonDivisor64(
        binExpr.getLHS().getLargestKnownDivisor(),
        binExpr.getRHS().getLargestKnownDivisor());
  }
  }
}

bool AffineExpr::isMultipleOf(int64_t factor) const {
  AffineBinaryOpExpr binExpr(nullptr);
  uint64_t l, u;
  switch (getKind()) {
  case AffineExprKind::SymbolId:
    LLVM_FALLTHROUGH;
  case AffineExprKind::DimId:
    return factor * factor == 1;
  case AffineExprKind::Constant:
    return cast<AffineConstantExpr>().getValue() % factor == 0;
  case AffineExprKind::Mul: {
    binExpr = cast<AffineBinaryOpExpr>();
    // It's probably not worth optimizing this further (to not traverse the
    // whole sub-tree under - it that would require a version of isMultipleOf
    // that on a 'false' return also returns the largest known divisor).
    return (l = binExpr.getLHS().getLargestKnownDivisor()) % factor == 0 ||
           (u = binExpr.getRHS().getLargestKnownDivisor()) % factor == 0 ||
           (l * u) % factor == 0;
  }
  case AffineExprKind::Add:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    binExpr = cast<AffineBinaryOpExpr>();
    return llvm::GreatestCommonDivisor64(
               binExpr.getLHS().getLargestKnownDivisor(),
               binExpr.getRHS().getLargestKnownDivisor()) %
               factor ==
           0;
  }
  }
}

bool AffineExpr::isFunctionOfDim(unsigned position) const {
  if (getKind() == AffineExprKind::DimId) {
    return *this == mlir::getAffineDimExpr(position, getContext());
  }
  if (auto expr = this->dyn_cast<AffineBinaryOpExpr>()) {
    return expr.getLHS().isFunctionOfDim(position) ||
           expr.getRHS().isFunctionOfDim(position);
  }
  return false;
}

AffineBinaryOpExpr::AffineBinaryOpExpr(AffineExpr::ImplType *ptr)
    : AffineExpr(ptr) {}
AffineExpr AffineBinaryOpExpr::getLHS() const {
  return static_cast<ImplType *>(expr)->lhs;
}
AffineExpr AffineBinaryOpExpr::getRHS() const {
  return static_cast<ImplType *>(expr)->rhs;
}

AffineDimExpr::AffineDimExpr(AffineExpr::ImplType *ptr) : AffineExpr(ptr) {}
unsigned AffineDimExpr::getPosition() const {
  return static_cast<ImplType *>(expr)->position;
}

AffineSymbolExpr::AffineSymbolExpr(AffineExpr::ImplType *ptr)
    : AffineExpr(ptr) {}
unsigned AffineSymbolExpr::getPosition() const {
  return static_cast<ImplType *>(expr)->position;
}

AffineConstantExpr::AffineConstantExpr(AffineExpr::ImplType *ptr)
    : AffineExpr(ptr) {}
int64_t AffineConstantExpr::getValue() const {
  return static_cast<ImplType *>(expr)->constant;
}

AffineExpr AffineExpr::operator+(int64_t v) const {
  return AffineBinaryOpExprStorage::get(AffineExprKind::Add, expr,
                                        getAffineConstantExpr(v, getContext()));
}
AffineExpr AffineExpr::operator+(AffineExpr other) const {
  return AffineBinaryOpExprStorage::get(AffineExprKind::Add, expr, other.expr);
}
AffineExpr AffineExpr::operator*(int64_t v) const {
  return AffineBinaryOpExprStorage::get(AffineExprKind::Mul, expr,
                                        getAffineConstantExpr(v, getContext()));
}
AffineExpr AffineExpr::operator*(AffineExpr other) const {
  return AffineBinaryOpExprStorage::get(AffineExprKind::Mul, expr, other.expr);
}
// Unary minus, delegate to operator*.
AffineExpr AffineExpr::operator-() const {
  return AffineBinaryOpExprStorage::get(
      AffineExprKind::Mul, expr, getAffineConstantExpr(-1, getContext()));
}
// Delegate to operator+.
AffineExpr AffineExpr::operator-(int64_t v) const { return *this + (-v); }
AffineExpr AffineExpr::operator-(AffineExpr other) const {
  return *this + (-other);
}
AffineExpr AffineExpr::floorDiv(uint64_t v) const {
  return AffineBinaryOpExprStorage::get(AffineExprKind::FloorDiv, expr,
                                        getAffineConstantExpr(v, getContext()));
}
AffineExpr AffineExpr::floorDiv(AffineExpr other) const {
  return AffineBinaryOpExprStorage::get(AffineExprKind::FloorDiv, expr,
                                        other.expr);
}
AffineExpr AffineExpr::ceilDiv(uint64_t v) const {
  return AffineBinaryOpExprStorage::get(AffineExprKind::CeilDiv, expr,
                                        getAffineConstantExpr(v, getContext()));
}
AffineExpr AffineExpr::ceilDiv(AffineExpr other) const {
  return AffineBinaryOpExprStorage::get(AffineExprKind::CeilDiv, expr,
                                        other.expr);
}
AffineExpr AffineExpr::operator%(uint64_t v) const {
  return AffineBinaryOpExprStorage::get(AffineExprKind::Mod, expr,
                                        getAffineConstantExpr(v, getContext()));
}
AffineExpr AffineExpr::operator%(AffineExpr other) const {
  return AffineBinaryOpExprStorage::get(AffineExprKind::Mod, expr, other.expr);
}
AffineExpr AffineExpr::compose(AffineMap map) const {
  SmallVector<AffineExpr, 8> dimReplacements(map.getResults().begin(),
                                             map.getResults().end());
  return replaceDimsAndSymbols(dimReplacements, {});
}
raw_ostream &operator<<(raw_ostream &os, AffineExpr &expr) {
  expr.print(os);
  return os;
}

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
