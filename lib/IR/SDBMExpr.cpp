//===- SDBMExpr.cpp - MLIR SDBM Expression implementation -----------------===//
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

namespace {
/// A simple compositional matcher for AffineExpr
///
/// Example usage:
///
/// ```c++
///    AffineExprMatcher x, C, m;
///    AffineExprMatcher pattern1 = ((x % C) * m) + x;
///    AffineExprMatcher pattern2 = x + ((x % C) * m);
///    if (pattern1.match(expr) || pattern2.match(expr)) {
///      ...
///    }
/// ```
class AffineExprMatcherStorage;
class AffineExprMatcher {
public:
  AffineExprMatcher();
  AffineExprMatcher(const AffineExprMatcher &other);

  AffineExprMatcher operator+(AffineExprMatcher other) {
    return AffineExprMatcher(AffineExprKind::Add, *this, other);
  }
  AffineExprMatcher operator*(AffineExprMatcher other) {
    return AffineExprMatcher(AffineExprKind::Mul, *this, other);
  }
  AffineExprMatcher floorDiv(AffineExprMatcher other) {
    return AffineExprMatcher(AffineExprKind::FloorDiv, *this, other);
  }
  AffineExprMatcher ceilDiv(AffineExprMatcher other) {
    return AffineExprMatcher(AffineExprKind::CeilDiv, *this, other);
  }
  AffineExprMatcher operator%(AffineExprMatcher other) {
    return AffineExprMatcher(AffineExprKind::Mod, *this, other);
  }

  AffineExpr match(AffineExpr expr);
  AffineExpr matched();
  Optional<int> getMatchedConstantValue();

private:
  AffineExprMatcher(AffineExprKind k, AffineExprMatcher a, AffineExprMatcher b);
  AffineExprKind kind; // only used to match in binary op cases.
  // A shared_ptr allows multiple references to same matcher storage without
  // worrying about ownership or dealing with an arena. To be cleaned up if we
  // go with this.
  std::shared_ptr<AffineExprMatcherStorage> storage;
};

class AffineExprMatcherStorage {
public:
  AffineExprMatcherStorage() {}
  AffineExprMatcherStorage(const AffineExprMatcherStorage &other)
      : subExprs(other.subExprs.begin(), other.subExprs.end()),
        matched(other.matched) {}
  AffineExprMatcherStorage(ArrayRef<AffineExprMatcher> exprs)
      : subExprs(exprs.begin(), exprs.end()) {}
  AffineExprMatcherStorage(AffineExprMatcher &a, AffineExprMatcher &b)
      : subExprs({a, b}) {}
  llvm::SmallVector<AffineExprMatcher, 0> subExprs;
  AffineExpr matched;
};
} // namespace

AffineExprMatcher::AffineExprMatcher()
    : kind(AffineExprKind::Constant), storage(new AffineExprMatcherStorage()) {}

AffineExprMatcher::AffineExprMatcher(const AffineExprMatcher &other)
    : kind(other.kind), storage(other.storage) {}

Optional<int> AffineExprMatcher::getMatchedConstantValue() {
  if (auto cst = storage->matched.dyn_cast<AffineConstantExpr>())
    return cst.getValue();
  return None;
}

AffineExpr AffineExprMatcher::match(AffineExpr expr) {
  if (kind > AffineExprKind::LAST_AFFINE_BINARY_OP) {
    if (storage->matched)
      if (storage->matched != expr)
        return AffineExpr();
    storage->matched = expr;
    return storage->matched;
  }
  if (kind != expr.getKind()) {
    return AffineExpr();
  }
  if (auto bin = expr.dyn_cast<AffineBinaryOpExpr>()) {
    if (!storage->subExprs.empty() &&
        !storage->subExprs[0].match(bin.getLHS())) {
      return AffineExpr();
    }
    if (!storage->subExprs.empty() &&
        !storage->subExprs[1].match(bin.getRHS())) {
      return AffineExpr();
    }
    if (storage->matched)
      if (storage->matched != expr)
        return AffineExpr();
    storage->matched = expr;
    return storage->matched;
  }
  llvm_unreachable("binary expected");
}

AffineExpr AffineExprMatcher::matched() { return storage->matched; }

AffineExprMatcher::AffineExprMatcher(AffineExprKind k, AffineExprMatcher a,
                                     AffineExprMatcher b)
    : kind(k), storage(new AffineExprMatcherStorage(a, b)) {
  storage->subExprs.push_back(a);
  storage->subExprs.push_back(b);
}

//===----------------------------------------------------------------------===//
// SDBMExpr
//===----------------------------------------------------------------------===//

SDBMExprKind SDBMExpr::getKind() const { return impl->getKind(); }

MLIRContext *SDBMExpr::getContext() const { return impl->getContext(); }

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

namespace {
// Helper class to perform negation of an SDBM expression.
struct SDBMNegator : public SDBMVisitor<SDBMNegator, SDBMExpr> {
  // Any positive expression is wrapped into a negation expression.
  //  -(x) = -x
  SDBMExpr visitPositive(SDBMPositiveExpr expr) {
    return SDBMNegExpr::get(expr);
  }
  // A negation expression is unwrapped.
  //  -(-x) = x
  SDBMExpr visitNeg(SDBMNegExpr expr) { return expr.getVar(); }
  // The value of the constant is negated.
  SDBMExpr visitConstant(SDBMConstantExpr expr) {
    return SDBMConstantExpr::get(expr.getContext(), -expr.getValue());
  }
  // Both terms of the sum are negated recursively.
  SDBMExpr visitSum(SDBMSumExpr expr) {
    return SDBMSumExpr::get(visit(expr.getLHS()).cast<SDBMVaryingExpr>(),
                            visit(expr.getRHS()).cast<SDBMConstantExpr>());
  }
  // Terms of a difference are interchanged.
  //  -(x - y) = y - x
  SDBMExpr visitDiff(SDBMDiffExpr expr) {
    return SDBMDiffExpr::get(expr.getRHS(), expr.getLHS());
  }
};
} // namespace

SDBMExpr SDBMExpr::operator-() { return SDBMNegator().visit(*this); }

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
    SDBMExpr visitAddExpr(AffineBinaryOpExpr expr) {
      // Attempt to recover a stripe expression.  Because AffineExprs don't have
      // a first-class difference kind, we check for both x + -1 * (x mod C) and
      // -1 * (x mod C) + x cases.
      AffineExprMatcher x, C, m;
      AffineExprMatcher pattern1 = ((x % C) * m) + x;
      AffineExprMatcher pattern2 = x + ((x % C) * m);
      if ((pattern1.match(expr) && m.getMatchedConstantValue() == -1) ||
          (pattern2.match(expr) && m.getMatchedConstantValue() == -1)) {
        if (auto convertedLHS = visit(x.matched())) {
          // TODO(ntv): return convertedLHS.stripe(C);
          return SDBMStripeExpr::get(
              convertedLHS.cast<SDBMPositiveExpr>(),
              visit(C.matched()).cast<SDBMConstantExpr>());
        }
      }
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

    SDBMExpr visitMulExpr(AffineBinaryOpExpr expr) {
      // Attempt to recover a stripe expression "x # C = (x floordiv C) * C".
      AffineExprMatcher x, C;
      AffineExprMatcher pattern = (x.floorDiv(C)) * C;
      if (pattern.match(expr)) {
        if (SDBMExpr converted = visit(x.matched())) {
          if (auto varConverted = converted.dyn_cast<SDBMPositiveExpr>())
            // TODO(ntv): return varConverted.stripe(C.getConstantValue());
            return SDBMStripeExpr::get(
                varConverted,
                SDBMConstantExpr::get(varConverted.getContext(),
                                      C.getMatchedConstantValue().getValue()));
        }
      }

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
