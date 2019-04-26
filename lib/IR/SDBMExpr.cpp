//===- SDBMExpr.h - MLIR SDBM Expression implementation -------------------===//
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
// A striped difference-bound matrix (SDBM) expression is a constant expression,
// an identifier, a binary expression with constant RHS and +, stripe operators
// or a difference expression between two identifiers.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/SDBMExpr.h"
#include "SDBMExprDetail.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// SDBMExpr
//===----------------------------------------------------------------------===//

SDBMExprKind SDBMExpr::getKind() const { return impl->getKind(); }

MLIRContext *SDBMExpr::getContext() const { return impl->getContext(); }

template <typename Derived, typename Result = void> class SDBMVisitor {
public:
  /// Visit the given SDBM expression, dispatching to kind-specific functions.
  Result visit(SDBMExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    switch (expr.getKind()) {
    case SDBMExprKind::Add:
    case SDBMExprKind::Diff:
    case SDBMExprKind::DimId:
    case SDBMExprKind::SymbolId:
    case SDBMExprKind::Neg:
    case SDBMExprKind::Stripe:
      return derived->visitVarying(expr.cast<SDBMVaryingExpr>());
    case SDBMExprKind::Constant:
      return derived->visitConstant(expr.cast<SDBMConstantExpr>());
    }
  }

protected:
  /// Default visitors do nothing.
  void visitSum(SDBMSumExpr) {}
  void visitDiff(SDBMDiffExpr) {}
  void visitStripe(SDBMStripeExpr) {}
  void visitDim(SDBMDimExpr) {}
  void visitSymbol(SDBMSymbolExpr) {}
  void visitNeg(SDBMNegExpr) {}
  void visitConstant(SDBMConstantExpr) {}

  /// Default implementation of visitPositive dispatches to the special
  /// functions for stripes and other variables.  Concrete visitors can override
  /// it.
  Result visitPositive(SDBMPositiveExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    if (expr.getKind() == SDBMExprKind::Stripe)
      return derived->visitStripe(expr.cast<SDBMStripeExpr>());
    else
      return derived->visitInput(expr.cast<SDBMInputExpr>());
  }

  /// Default implementation of visitInput dispatches to the special
  /// functions for dimensions or symbols.  Concrete visitors can override it to
  /// visit all variables instead.
  Result visitInput(SDBMInputExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    if (expr.getKind() == SDBMExprKind::DimId)
      return derived->visitDim(expr.cast<SDBMDimExpr>());
    else
      return derived->visitSymbol(expr.cast<SDBMSymbolExpr>());
  }

  /// Default implementation of visitVarying dispatches to the special
  /// functions for variables and negations thereof.  Concerete visitors can
  /// override it to visit all variables and negations isntead.
  Result visitVarying(SDBMVaryingExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    if (auto var = expr.dyn_cast<SDBMPositiveExpr>())
      return derived->visitPositive(var);
    else if (auto neg = expr.dyn_cast<SDBMNegExpr>())
      return derived->visitNeg(neg);
    else if (auto sum = expr.dyn_cast<SDBMSumExpr>())
      return derived->visitSum(sum);
    else if (auto diff = expr.dyn_cast<SDBMDiffExpr>())
      return derived->visitDiff(diff);

    llvm_unreachable("unhandled subtype of varying SDBM expression");
  }
};

void SDBMExpr::print(raw_ostream &os) const {
  struct Printer : public SDBMVisitor<Printer> {
    Printer(raw_ostream &ostream) : prn(ostream) {}

    void visitSum(SDBMSumExpr expr) {
      visitVarying(expr.getLHS());
      prn << " + ";
      visitConstant(expr.getRHS());
    }
    void visitDiff(SDBMDiffExpr expr) {
      visitPositive(expr.getLHS());
      prn << " - ";
      visitPositive(expr.getRHS());
    }
    void visitDim(SDBMDimExpr expr) { prn << 'd' << expr.getPosition(); }
    void visitSymbol(SDBMSymbolExpr expr) { prn << 's' << expr.getPosition(); }
    void visitStripe(SDBMStripeExpr expr) {
      visitPositive(expr.getVar());
      prn << " # ";
      visitConstant(expr.getStripeFactor());
    }
    void visitNeg(SDBMNegExpr expr) {
      prn << '-';
      visitPositive(expr.getVar());
    }
    void visitConstant(SDBMConstantExpr expr) { prn << expr.getValue(); }

    raw_ostream &prn;
  };
  Printer printer(os);
  printer.visit(*this);
}

void SDBMExpr::dump() const {
  print(llvm::errs());
  llvm::errs() << '\n';
}

//===----------------------------------------------------------------------===//
// SDBMSumExpr
//===----------------------------------------------------------------------===//

SDBMVaryingExpr SDBMSumExpr::getLHS() const {
  return static_cast<ImplType *>(impl)->lhs;
}

SDBMConstantExpr SDBMSumExpr::getRHS() const {
  return static_cast<ImplType *>(impl)->rhs;
}

AffineExpr SDBMExpr::getAsAffineExpr() const {
  struct Converter : public SDBMVisitor<Converter, AffineExpr> {
    AffineExpr visitSum(SDBMSumExpr expr) {
      AffineExpr lhs = visit(expr.getLHS()), rhs = visit(expr.getRHS());
      return lhs + rhs;
    }

    AffineExpr visitStripe(SDBMStripeExpr expr) {
      AffineExpr lhs = visit(expr.getVar()),
                 rhs = visit(expr.getStripeFactor());
      return lhs - (lhs % rhs);
    }

    AffineExpr visitDiff(SDBMDiffExpr expr) {
      AffineExpr lhs = visit(expr.getLHS()), rhs = visit(expr.getRHS());
      return lhs - rhs;
    }

    AffineExpr visitDim(SDBMDimExpr expr) {
      return getAffineDimExpr(expr.getPosition(), expr.getContext());
    }

    AffineExpr visitSymbol(SDBMSymbolExpr expr) {
      return getAffineSymbolExpr(expr.getPosition(), expr.getContext());
    }

    AffineExpr visitNeg(SDBMNegExpr expr) {
      return getAffineBinaryOpExpr(AffineExprKind::Mul,
                                   getAffineConstantExpr(-1, expr.getContext()),
                                   visit(expr.getVar()));
    }

    AffineExpr visitConstant(SDBMConstantExpr expr) {
      return getAffineConstantExpr(expr.getValue(), expr.getContext());
    }
  } converter;
  return converter.visit(*this);
}

Optional<SDBMExpr> SDBMExpr::tryConvertAffineExpr(AffineExpr affine) {
  struct Converter : public AffineExprVisitor<Converter, SDBMExpr> {
    // Try matching the definition of the stripe operation as x - x mod C where
    // `pos` should match "x" and `neg` should match "- (x mod C)".
    SDBMExpr matchStripeAddPattern(AffineExpr pos, AffineExpr neg) {
      // Check that the "pos" part is a variable expression and that the "neg"
      // part is a mul expression.
      auto convertedLHS = visit(pos);
      if (!convertedLHS || !convertedLHS.isa<SDBMPositiveExpr>())
        return {};

      auto outerBinExpr = neg.dyn_cast<AffineBinaryOpExpr>();
      if (!outerBinExpr || outerBinExpr.getKind() != AffineExprKind::Mul)
        return {};

      // In affine mul expressions, the constant part is always on the RHS.
      // If there had been two constants, they would have been folded away.
      assert(!outerBinExpr.getLHS().isa<AffineConstantExpr>() &&
             "expected a constant on the RHS of an affine mul expression");
      // Check if the RHS of mul is -1.
      auto multiplierExpr =
          outerBinExpr.getRHS().dyn_cast<AffineConstantExpr>();
      if (!multiplierExpr || multiplierExpr.getValue() != -1)
        return {};

      // Check if the LHS of mul is ("pos" mod constant).
      auto binExpr = outerBinExpr.getLHS().dyn_cast<AffineBinaryOpExpr>();
      if (!binExpr || binExpr.getKind() != AffineExprKind::Mod ||
          !binExpr.getRHS().isa<AffineConstantExpr>())
        return {};

      if (convertedLHS != visit(binExpr.getLHS()))
        return {};

      // If all checks pass, we have a stripe.
      return SDBMStripeExpr::get(
          convertedLHS.cast<SDBMPositiveExpr>(),
          visit(binExpr.getRHS()).cast<SDBMConstantExpr>());
    }

    SDBMExpr visitAddExpr(AffineBinaryOpExpr expr) {
      // Attempt to recover a stripe expression.  Because AffineExprs don't have
      // a first-class difference kind, we check for both x + -1 * (x mod C) and
      // -1 * (x mod C) + x cases.
      if (auto stripe = matchStripeAddPattern(expr.getLHS(), expr.getRHS()))
        return stripe;
      if (auto stripe = matchStripeAddPattern(expr.getRHS(), expr.getLHS()))
        return stripe;

      auto lhs = visit(expr.getLHS()), rhs = visit(expr.getRHS());
      if (!lhs || !rhs)
        return {};

      // In a "add" AffineExpr, the constant always appears on the right.  If
      // there were two constants, they would have been folded away.
      assert(!lhs.isa<SDBMConstantExpr>() && "non-canonical affine expression");
      auto rhsConstant = rhs.dyn_cast<SDBMConstantExpr>();

      // SDBM accepts LHS variables and RHS constants in a sum.
      auto lhsVar = lhs.dyn_cast<SDBMVaryingExpr>();
      auto rhsVar = rhs.dyn_cast<SDBMVaryingExpr>();
      if (rhsConstant && lhsVar)
        return SDBMSumExpr::get(lhsVar, rhsConstant);

      // The sum of a negated variable and a non-negated variable is a
      // difference, supported as a special kind in SDBM.  Because AffineExprs
      // don't have first-class difference kind, check both LHS and RHS for
      // negation.
      auto lhsPos = lhs.dyn_cast<SDBMPositiveExpr>();
      auto rhsPos = rhs.dyn_cast<SDBMPositiveExpr>();
      auto lhsNeg = lhs.dyn_cast<SDBMNegExpr>();
      auto rhsNeg = rhs.dyn_cast<SDBMNegExpr>();
      if (lhsNeg && rhsVar)
        return SDBMDiffExpr::get(rhsPos, lhsNeg.getVar());
      if (rhsNeg && lhsVar)
        return SDBMDiffExpr::get(lhsPos, rhsNeg.getVar());

      // Other cases don't fit into SDBM.
      return {};
    }

    // Try matching the stripe pattern "(x floordiv C) * C" where `lhs`
    // corresponds to "(x floordiv C)" and `rhs` corresponds to "C".
    SDBMExpr matchStripeMulPattern(AffineExpr lhs, AffineExpr rhs) {
      // Check if LHS is a floordiv expression and rhs is a constant.
      auto lhsBinary = lhs.dyn_cast<AffineBinaryOpExpr>();
      auto rhsConstant = rhs.dyn_cast<AffineConstantExpr>();
      if (!lhsBinary || !rhsConstant ||
          lhsBinary.getKind() != AffineExprKind::FloorDiv)
        return {};

      // Check if the floordiv divides by the constant equal to RHS.
      auto lhsRhsConstant = lhsBinary.getRHS().dyn_cast<AffineConstantExpr>();
      if (!lhsRhsConstant || lhsRhsConstant != rhsConstant)
        return {};

      // Check if LHS can be converted to a single variable.
      SDBMExpr converted = visit(lhsBinary.getLHS());
      if (!converted)
        return {};
      auto varConverted = converted.dyn_cast<SDBMPositiveExpr>();
      if (!varConverted)
        return {};

      // If all checks pass, we have a stripe.
      return SDBMStripeExpr::get(
          varConverted, SDBMConstantExpr::get(varConverted.getContext(),
                                              rhsConstant.getValue()));
    }

    SDBMExpr visitMulExpr(AffineBinaryOpExpr expr) {
      // Attempt to recover a stripe expression "x # C = (x floordiv C) * C".
      if (auto stripe = matchStripeMulPattern(expr.getLHS(), expr.getRHS()))
        return stripe;

      auto lhs = visit(expr.getLHS()), rhs = visit(expr.getRHS());
      if (!lhs || !rhs)
        return {};

      // In a "mul" AffineExpr, the constant always appears on the right.  If
      // there were two constants, they would have been folded away.
      assert(!lhs.isa<SDBMConstantExpr>() && "non-canonical affine expression");
      auto rhsConstant = rhs.dyn_cast<SDBMConstantExpr>();
      if (!rhsConstant)
        return {};

      // The only supported "multiplication" expression is an SDBM is dimension
      // negation, that is a product of dimension and constant -1.
      auto lhsVar = lhs.dyn_cast<SDBMPositiveExpr>();
      if (lhsVar && rhsConstant.getValue() == -1)
        return SDBMNegExpr::get(lhsVar);

      // Other multiplications are not allowed in SDBM.
      return {};
    }

    SDBMExpr visitModExpr(AffineBinaryOpExpr expr) {
      auto lhs = visit(expr.getLHS()), rhs = visit(expr.getRHS());
      if (!lhs || !rhs)
        return {};

      // 'mod' can only be converted to SDBM if its LHS is a variable
      // and its RHS is a constant.  Then it `x mod c = x - x stripe c`.
      auto rhsConstant = rhs.dyn_cast<SDBMConstantExpr>();
      auto lhsVar = rhs.dyn_cast<SDBMPositiveExpr>();
      if (!lhsVar || !rhsConstant)
        return {};
      return SDBMDiffExpr::get(lhsVar,
                               SDBMStripeExpr::get(lhsVar, rhsConstant));
    }

    // `a floordiv b = (a stripe b) / b`, but we have no division in SDBM
    SDBMExpr visitFloorDivExpr(AffineBinaryOpExpr expr) { return {}; }
    SDBMExpr visitCeilDivExpr(AffineBinaryOpExpr expr) { return {}; }

    // Dimensions, symbols and constants are converted trivially.
    SDBMExpr visitConstantExpr(AffineConstantExpr expr) {
      return SDBMConstantExpr::get(expr.getContext(), expr.getValue());
    }
    SDBMExpr visitDimExpr(AffineDimExpr expr) {
      return SDBMDimExpr::get(expr.getContext(), expr.getPosition());
    }
    SDBMExpr visitSymbolExpr(AffineSymbolExpr expr) {
      return SDBMSymbolExpr::get(expr.getContext(), expr.getPosition());
    }
  } converter;

  if (auto result = converter.visit(affine))
    return result;
  return None;
}

//===----------------------------------------------------------------------===//
// SDBMDiffExpr
//===----------------------------------------------------------------------===//

SDBMPositiveExpr SDBMDiffExpr::getLHS() const {
  return static_cast<ImplType *>(impl)->lhs;
}

SDBMPositiveExpr SDBMDiffExpr::getRHS() const {
  return static_cast<ImplType *>(impl)->rhs;
}

//===----------------------------------------------------------------------===//
// SDBMStripeExpr
//===----------------------------------------------------------------------===//

SDBMPositiveExpr SDBMStripeExpr::getVar() const {
  if (SDBMVaryingExpr lhs = static_cast<ImplType *>(impl)->lhs)
    return lhs.cast<SDBMPositiveExpr>();
  return {};
}

SDBMConstantExpr SDBMStripeExpr::getStripeFactor() const {
  return static_cast<ImplType *>(impl)->rhs;
}

//===----------------------------------------------------------------------===//
// SDBMInputExpr
//===----------------------------------------------------------------------===//

unsigned SDBMInputExpr::getPosition() const {
  return static_cast<ImplType *>(impl)->position;
}

//===----------------------------------------------------------------------===//
// SDBMConstantExpr
//===----------------------------------------------------------------------===//

int64_t SDBMConstantExpr::getValue() const {
  return static_cast<ImplType *>(impl)->constant;
}

//===----------------------------------------------------------------------===//
// SDBMNegExpr
//===----------------------------------------------------------------------===//

SDBMPositiveExpr SDBMNegExpr::getVar() const {
  return static_cast<ImplType *>(impl)->dim;
}
