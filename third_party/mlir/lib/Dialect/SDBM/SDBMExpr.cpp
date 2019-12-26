//===- SDBMExpr.cpp - MLIR SDBM Expression implementation -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A striped difference-bound matrix (SDBM) expression is a constant expression,
// an identifier, a binary expression with constant RHS and +, stripe operators
// or a difference expression between two identifiers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SDBM/SDBMExpr.h"
#include "SDBMExprDetail.h"
#include "mlir/Dialect/SDBM/SDBMDialect.h"
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
  SmallVector<AffineExprMatcher, 0> subExprs;
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

MLIRContext *SDBMExpr::getContext() const {
  return impl->dialect->getContext();
}

SDBMDialect *SDBMExpr::getDialect() const { return impl->dialect; }

void SDBMExpr::print(raw_ostream &os) const {
  struct Printer : public SDBMVisitor<Printer> {
    Printer(raw_ostream &ostream) : prn(ostream) {}

    void visitSum(SDBMSumExpr expr) {
      visit(expr.getLHS());
      prn << " + ";
      visit(expr.getRHS());
    }
    void visitDiff(SDBMDiffExpr expr) {
      visit(expr.getLHS());
      prn << " - ";
      visit(expr.getRHS());
    }
    void visitDim(SDBMDimExpr expr) { prn << 'd' << expr.getPosition(); }
    void visitSymbol(SDBMSymbolExpr expr) { prn << 's' << expr.getPosition(); }
    void visitStripe(SDBMStripeExpr expr) {
      SDBMDirectExpr lhs = expr.getLHS();
      bool isTerm = lhs.isa<SDBMTermExpr>();
      if (!isTerm)
        prn << '(';
      visit(lhs);
      if (!isTerm)
        prn << ')';
      prn << " # ";
      visitConstant(expr.getStripeFactor());
    }
    void visitNeg(SDBMNegExpr expr) {
      bool isSum = expr.getVar().isa<SDBMSumExpr>();
      prn << '-';
      if (isSum)
        prn << '(';
      visit(expr.getVar());
      if (isSum)
        prn << ')';
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
  // Any term expression is wrapped into a negation expression.
  //  -(x) = -x
  SDBMExpr visitDirect(SDBMDirectExpr expr) { return SDBMNegExpr::get(expr); }
  // A negation expression is unwrapped.
  //  -(-x) = x
  SDBMExpr visitNeg(SDBMNegExpr expr) { return expr.getVar(); }
  // The value of the constant is negated.
  SDBMExpr visitConstant(SDBMConstantExpr expr) {
    return SDBMConstantExpr::get(expr.getDialect(), -expr.getValue());
  }

  // Terms of a difference are interchanged. Since only the LHS of a diff
  // expression is allowed to be a sum with a constant, we need to recreate the
  // sum with the negated value:
  //   -((x + C) - y) = (y - C) - x.
  SDBMExpr visitDiff(SDBMDiffExpr expr) {
    // If the LHS is just a term, we can do straightforward interchange.
    if (auto term = expr.getLHS().dyn_cast<SDBMTermExpr>())
      return SDBMDiffExpr::get(expr.getRHS(), term);

    auto sum = expr.getLHS().cast<SDBMSumExpr>();
    auto cst = visitConstant(sum.getRHS()).cast<SDBMConstantExpr>();
    return SDBMDiffExpr::get(SDBMSumExpr::get(expr.getRHS(), cst),
                             sum.getLHS());
  }
};
} // namespace

SDBMExpr SDBMExpr::operator-() { return SDBMNegator().visit(*this); }

//===----------------------------------------------------------------------===//
// SDBMSumExpr
//===----------------------------------------------------------------------===//

SDBMSumExpr SDBMSumExpr::get(SDBMTermExpr lhs, SDBMConstantExpr rhs) {
  assert(lhs && "expected SDBM variable expression");
  assert(rhs && "expected SDBM constant");

  // If LHS of a sum is another sum, fold the constant RHS parts.
  if (auto lhsSum = lhs.dyn_cast<SDBMSumExpr>()) {
    lhs = lhsSum.getLHS();
    rhs = SDBMConstantExpr::get(rhs.getDialect(),
                                rhs.getValue() + lhsSum.getRHS().getValue());
  }

  StorageUniquer &uniquer = lhs.getDialect()->getUniquer();
  return uniquer.get<detail::SDBMBinaryExprStorage>(
      /*initFn=*/{}, static_cast<unsigned>(SDBMExprKind::Add), lhs, rhs);
}

SDBMTermExpr SDBMSumExpr::getLHS() const {
  return static_cast<ImplType *>(impl)->lhs.cast<SDBMTermExpr>();
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
      AffineExpr lhs = visit(expr.getLHS()),
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

// Given a direct expression `expr`, add the given constant to it and pass the
// resulting expression to `builder` before returning its result.  If the
// expression is already a sum expression, update its constant and extract the
// LHS if the constant becomes zero.  Otherwise, construct a sum expression.
template <typename Result>
Result addConstantAndSink(SDBMDirectExpr expr, int64_t constant, bool negated,
                          function_ref<Result(SDBMDirectExpr)> builder) {
  SDBMDialect *dialect = expr.getDialect();
  if (auto sumExpr = expr.dyn_cast<SDBMSumExpr>()) {
    if (negated)
      constant = sumExpr.getRHS().getValue() - constant;
    else
      constant += sumExpr.getRHS().getValue();

    if (constant != 0) {
      auto sum = SDBMSumExpr::get(sumExpr.getLHS(),
                                  SDBMConstantExpr::get(dialect, constant));
      return builder(sum);
    } else {
      return builder(sumExpr.getLHS());
    }
  }
  if (constant != 0)
    return builder(SDBMSumExpr::get(
        expr.cast<SDBMTermExpr>(),
        SDBMConstantExpr::get(dialect, negated ? -constant : constant)));
  return expr;
}

// Construct an expression lhs + constant while maintaining the canonical form
// of the SDBM expressions, in particular sink the constant expression to the
// nearest sum expression in the left subtree of the expression tree.
static SDBMExpr addConstant(SDBMVaryingExpr lhs, int64_t constant) {
  if (auto lhsDiff = lhs.dyn_cast<SDBMDiffExpr>())
    return addConstantAndSink<SDBMExpr>(
        lhsDiff.getLHS(), constant, /*negated=*/false,
        [lhsDiff](SDBMDirectExpr e) {
          return SDBMDiffExpr::get(e, lhsDiff.getRHS());
        });
  if (auto lhsNeg = lhs.dyn_cast<SDBMNegExpr>())
    return addConstantAndSink<SDBMExpr>(
        lhsNeg.getVar(), constant, /*negated=*/true,
        [](SDBMDirectExpr e) { return SDBMNegExpr::get(e); });
  if (auto lhsSum = lhs.dyn_cast<SDBMSumExpr>())
    return addConstantAndSink<SDBMExpr>(lhsSum, constant, /*negated=*/false,
                                        [](SDBMDirectExpr e) { return e; });
  if (constant != 0)
    return SDBMSumExpr::get(lhs.cast<SDBMTermExpr>(),
                            SDBMConstantExpr::get(lhs.getDialect(), constant));
  return lhs;
}

// Build a difference expression given a direct expression and a negation
// expression.
static SDBMExpr buildDiffExpr(SDBMDirectExpr lhs, SDBMNegExpr rhs) {
  // Fold (x + C) - (x + D) = C - D.
  if (lhs.getTerm() == rhs.getVar().getTerm())
    return SDBMConstantExpr::get(
        lhs.getDialect(), lhs.getConstant() - rhs.getVar().getConstant());

  return SDBMDiffExpr::get(
      addConstantAndSink<SDBMDirectExpr>(lhs, -rhs.getVar().getConstant(),
                                         /*negated=*/false,
                                         [](SDBMDirectExpr e) { return e; }),
      rhs.getVar().getTerm());
}

// Try folding an expression (lhs + rhs) where at least one of the operands
// contains a negated variable, i.e. is a negation or a difference expression.
static SDBMExpr foldSumDiff(SDBMExpr lhs, SDBMExpr rhs) {
  // If exactly one of LHS, RHS is a negation expression, we can construct
  // a difference expression, which is a special kind in SDBM.
  auto lhsDirect = lhs.dyn_cast<SDBMDirectExpr>();
  auto rhsDirect = rhs.dyn_cast<SDBMDirectExpr>();
  auto lhsNeg = lhs.dyn_cast<SDBMNegExpr>();
  auto rhsNeg = rhs.dyn_cast<SDBMNegExpr>();

  if (lhsDirect && rhsNeg)
    return buildDiffExpr(lhsDirect, rhsNeg);
  if (lhsNeg && rhsDirect)
    return buildDiffExpr(rhsDirect, lhsNeg);

  // If a subexpression appears in a diff expression on the LHS(RHS) of a
  // sum expression where it also appears on the RHS(LHS) with the opposite
  // sign, we can simplify it away and obtain the SDBM form.
  auto lhsDiff = lhs.dyn_cast<SDBMDiffExpr>();
  auto rhsDiff = rhs.dyn_cast<SDBMDiffExpr>();

  // -(x + A) + ((x + B) - y) = -(y + (A - B))
  if (lhsNeg && rhsDiff &&
      lhsNeg.getVar().getTerm() == rhsDiff.getLHS().getTerm()) {
    int64_t constant =
        lhsNeg.getVar().getConstant() - rhsDiff.getLHS().getConstant();
    // RHS of the diff is a term expression, its sum with a constant is a direct
    // expression.
    return SDBMNegExpr::get(
        addConstant(rhsDiff.getRHS(), constant).cast<SDBMDirectExpr>());
  }

  // (x + A) + ((y + B) - x) = (y + B) + A.
  if (lhsDirect && rhsDiff && lhsDirect.getTerm() == rhsDiff.getRHS())
    return addConstant(rhsDiff.getLHS(), lhsDirect.getConstant());

  // ((x + A) - y) + (-(x + B)) = -(y + (B - A)).
  if (lhsDiff && rhsNeg &&
      lhsDiff.getLHS().getTerm() == rhsNeg.getVar().getTerm()) {
    int64_t constant =
        rhsNeg.getVar().getConstant() - lhsDiff.getLHS().getConstant();
    // RHS of the diff is a term expression, its sum with a constant is a direct
    // expression.
    return SDBMNegExpr::get(
        addConstant(lhsDiff.getRHS(), constant).cast<SDBMDirectExpr>());
  }

  // ((x + A) - y) + (y + B) = (x + A) + B.
  if (rhsDirect && lhsDiff && rhsDirect.getTerm() == lhsDiff.getRHS())
    return addConstant(lhsDiff.getLHS(), rhsDirect.getConstant());

  return {};
}

Optional<SDBMExpr> SDBMExpr::tryConvertAffineExpr(AffineExpr affine) {
  struct Converter : public AffineExprVisitor<Converter, SDBMExpr> {
    SDBMExpr visitAddExpr(AffineBinaryOpExpr expr) {
      auto lhs = visit(expr.getLHS()), rhs = visit(expr.getRHS());
      if (!lhs || !rhs)
        return {};

      // In a "add" AffineExpr, the constant always appears on the right.  If
      // there were two constants, they would have been folded away.
      assert(!lhs.isa<SDBMConstantExpr>() && "non-canonical affine expression");

      // If RHS is a constant, we can always extend the SDBM expression to
      // include it by sinking the constant into the nearest sum expression.
      if (auto rhsConstant = rhs.dyn_cast<SDBMConstantExpr>()) {
        int64_t constant = rhsConstant.getValue();
        auto varying = lhs.dyn_cast<SDBMVaryingExpr>();
        assert(varying && "unexpected uncanonicalized sum of constants");
        return addConstant(varying, constant);
      }

      // Try building a difference expression if one of the values is negated,
      // or check if a difference on either hand side cancels out the outer term
      // so as to remain correct within SDBM. Return null otherwise.
      return foldSumDiff(lhs, rhs);
    }

    SDBMExpr visitMulExpr(AffineBinaryOpExpr expr) {
      // Attempt to recover a stripe expression "x # C = (x floordiv C) * C".
      AffineExprMatcher x, C;
      AffineExprMatcher pattern = (x.floorDiv(C)) * C;
      if (pattern.match(expr)) {
        if (SDBMExpr converted = visit(x.matched())) {
          if (auto varConverted = converted.dyn_cast<SDBMTermExpr>())
            // TODO(ntv): return varConverted.stripe(C.getConstantValue());
            return SDBMStripeExpr::get(
                varConverted,
                SDBMConstantExpr::get(dialect,
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
      if (rhsConstant.getValue() != -1)
        return {};

      if (auto lhsVar = lhs.dyn_cast<SDBMTermExpr>())
        return SDBMNegExpr::get(lhsVar);
      if (auto lhsDiff = lhs.dyn_cast<SDBMDiffExpr>())
        return SDBMNegator().visitDiff(lhsDiff);

      // Other multiplications are not allowed in SDBM.
      return {};
    }

    SDBMExpr visitModExpr(AffineBinaryOpExpr expr) {
      auto lhs = visit(expr.getLHS()), rhs = visit(expr.getRHS());
      if (!lhs || !rhs)
        return {};

      // 'mod' can only be converted to SDBM if its LHS is a direct expression
      // and its RHS is a constant.  Then it `x mod c = x - x stripe c`.
      auto rhsConstant = rhs.dyn_cast<SDBMConstantExpr>();
      auto lhsVar = lhs.dyn_cast<SDBMDirectExpr>();
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
      return SDBMConstantExpr::get(dialect, expr.getValue());
    }
    SDBMExpr visitDimExpr(AffineDimExpr expr) {
      return SDBMDimExpr::get(dialect, expr.getPosition());
    }
    SDBMExpr visitSymbolExpr(AffineSymbolExpr expr) {
      return SDBMSymbolExpr::get(dialect, expr.getPosition());
    }

    SDBMDialect *dialect;
  } converter;
  converter.dialect = affine.getContext()->getRegisteredDialect<SDBMDialect>();

  if (auto result = converter.visit(affine))
    return result;
  return None;
}

//===----------------------------------------------------------------------===//
// SDBMDiffExpr
//===----------------------------------------------------------------------===//

SDBMDiffExpr SDBMDiffExpr::get(SDBMDirectExpr lhs, SDBMTermExpr rhs) {
  assert(lhs && "expected SDBM dimension");
  assert(rhs && "expected SDBM dimension");

  StorageUniquer &uniquer = lhs.getDialect()->getUniquer();
  return uniquer.get<detail::SDBMDiffExprStorage>(
      /*initFn=*/{}, static_cast<unsigned>(SDBMExprKind::Diff), lhs, rhs);
}

SDBMDirectExpr SDBMDiffExpr::getLHS() const {
  return static_cast<ImplType *>(impl)->lhs;
}

SDBMTermExpr SDBMDiffExpr::getRHS() const {
  return static_cast<ImplType *>(impl)->rhs;
}

//===----------------------------------------------------------------------===//
// SDBMDirectExpr
//===----------------------------------------------------------------------===//

SDBMTermExpr SDBMDirectExpr::getTerm() {
  if (auto sum = dyn_cast<SDBMSumExpr>())
    return sum.getLHS();
  return cast<SDBMTermExpr>();
}

int64_t SDBMDirectExpr::getConstant() {
  if (auto sum = dyn_cast<SDBMSumExpr>())
    return sum.getRHS().getValue();
  return 0;
}

//===----------------------------------------------------------------------===//
// SDBMStripeExpr
//===----------------------------------------------------------------------===//

SDBMStripeExpr SDBMStripeExpr::get(SDBMDirectExpr var,
                                   SDBMConstantExpr stripeFactor) {
  assert(var && "expected SDBM variable expression");
  assert(stripeFactor && "expected non-null stripe factor");
  if (stripeFactor.getValue() <= 0)
    llvm::report_fatal_error("non-positive stripe factor");

  StorageUniquer &uniquer = var.getDialect()->getUniquer();
  return uniquer.get<detail::SDBMBinaryExprStorage>(
      /*initFn=*/{}, static_cast<unsigned>(SDBMExprKind::Stripe), var,
      stripeFactor);
}

SDBMDirectExpr SDBMStripeExpr::getLHS() const {
  if (SDBMVaryingExpr lhs = static_cast<ImplType *>(impl)->lhs)
    return lhs.cast<SDBMDirectExpr>();
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
// SDBMDimExpr
//===----------------------------------------------------------------------===//

SDBMDimExpr SDBMDimExpr::get(SDBMDialect *dialect, unsigned position) {
  assert(dialect && "expected non-null dialect");

  auto assignDialect = [dialect](detail::SDBMTermExprStorage *storage) {
    storage->dialect = dialect;
  };

  StorageUniquer &uniquer = dialect->getUniquer();
  return uniquer.get<detail::SDBMTermExprStorage>(
      assignDialect, static_cast<unsigned>(SDBMExprKind::DimId), position);
}

//===----------------------------------------------------------------------===//
// SDBMSymbolExpr
//===----------------------------------------------------------------------===//

SDBMSymbolExpr SDBMSymbolExpr::get(SDBMDialect *dialect, unsigned position) {
  assert(dialect && "expected non-null dialect");

  auto assignDialect = [dialect](detail::SDBMTermExprStorage *storage) {
    storage->dialect = dialect;
  };

  StorageUniquer &uniquer = dialect->getUniquer();
  return uniquer.get<detail::SDBMTermExprStorage>(
      assignDialect, static_cast<unsigned>(SDBMExprKind::SymbolId), position);
}

//===----------------------------------------------------------------------===//
// SDBMConstantExpr
//===----------------------------------------------------------------------===//

SDBMConstantExpr SDBMConstantExpr::get(SDBMDialect *dialect, int64_t value) {
  assert(dialect && "expected non-null dialect");

  auto assignCtx = [dialect](detail::SDBMConstantExprStorage *storage) {
    storage->dialect = dialect;
  };

  StorageUniquer &uniquer = dialect->getUniquer();
  return uniquer.get<detail::SDBMConstantExprStorage>(
      assignCtx, static_cast<unsigned>(SDBMExprKind::Constant), value);
}

int64_t SDBMConstantExpr::getValue() const {
  return static_cast<ImplType *>(impl)->constant;
}

//===----------------------------------------------------------------------===//
// SDBMNegExpr
//===----------------------------------------------------------------------===//

SDBMNegExpr SDBMNegExpr::get(SDBMDirectExpr var) {
  assert(var && "expected non-null SDBM direct expression");

  StorageUniquer &uniquer = var.getDialect()->getUniquer();
  return uniquer.get<detail::SDBMNegExprStorage>(
      /*initFn=*/{}, static_cast<unsigned>(SDBMExprKind::Neg), var);
}

SDBMDirectExpr SDBMNegExpr::getVar() const {
  return static_cast<ImplType *>(impl)->expr;
}

SDBMExpr mlir::ops_assertions::operator+(SDBMExpr lhs, SDBMExpr rhs) {
  if (auto folded = foldSumDiff(lhs, rhs))
    return folded;
  assert(!(lhs.isa<SDBMNegExpr>() && rhs.isa<SDBMNegExpr>()) &&
         "a sum of negated expressions is a negation of a sum of variables and "
         "not a correct SDBM");

  // Fold (x - y) + (y - x) = 0.
  auto lhsDiff = lhs.dyn_cast<SDBMDiffExpr>();
  auto rhsDiff = rhs.dyn_cast<SDBMDiffExpr>();
  if (lhsDiff && rhsDiff) {
    if (lhsDiff.getLHS() == rhsDiff.getRHS() &&
        lhsDiff.getRHS() == rhsDiff.getLHS())
      return SDBMConstantExpr::get(lhs.getDialect(), 0);
  }

  // If LHS is a constant and RHS is not, swap the order to get into a supported
  // sum case.  From now on, RHS must be a constant.
  auto lhsConstant = lhs.dyn_cast<SDBMConstantExpr>();
  auto rhsConstant = rhs.dyn_cast<SDBMConstantExpr>();
  if (!rhsConstant && lhsConstant) {
    std::swap(lhs, rhs);
    std::swap(lhsConstant, rhsConstant);
  }
  assert(rhsConstant && "at least one operand must be a constant");

  // Constant-fold if LHS is also a constant.
  if (lhsConstant)
    return SDBMConstantExpr::get(lhs.getDialect(), lhsConstant.getValue() +
                                                       rhsConstant.getValue());
  return addConstant(lhs.cast<SDBMVaryingExpr>(), rhsConstant.getValue());
}

SDBMExpr mlir::ops_assertions::operator-(SDBMExpr lhs, SDBMExpr rhs) {
  // Fold x - x == 0.
  if (lhs == rhs)
    return SDBMConstantExpr::get(lhs.getDialect(), 0);

  // LHS and RHS may be constants.
  auto lhsConstant = lhs.dyn_cast<SDBMConstantExpr>();
  auto rhsConstant = rhs.dyn_cast<SDBMConstantExpr>();

  // Constant fold if both LHS and RHS are constants.
  if (lhsConstant && rhsConstant)
    return SDBMConstantExpr::get(lhs.getDialect(), lhsConstant.getValue() -
                                                       rhsConstant.getValue());

  // Replace a difference with a sum with a negated value if one of LHS and RHS
  // is a constant:
  //   x - C == x + (-C);
  //   C - x == -x + C.
  // This calls into operator+ for further simplification.
  if (rhsConstant)
    return lhs + (-rhsConstant);
  if (lhsConstant)
    return -rhs + lhsConstant;

  return buildDiffExpr(lhs.cast<SDBMDirectExpr>(), (-rhs).cast<SDBMNegExpr>());
}

SDBMExpr mlir::ops_assertions::stripe(SDBMExpr expr, SDBMExpr factor) {
  auto constantFactor = factor.cast<SDBMConstantExpr>();
  assert(constantFactor.getValue() > 0 && "non-positive stripe");

  // Fold x # 1 = x.
  if (constantFactor.getValue() == 1)
    return expr;

  return SDBMStripeExpr::get(expr.cast<SDBMDirectExpr>(), constantFactor);
}
