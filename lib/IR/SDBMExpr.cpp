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

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// SDBMExpr
//===----------------------------------------------------------------------===//

SDBMExprKind SDBMExpr::getKind() const { return impl->getKind(); }

MLIRContext *SDBMExpr::getContext() const { return impl->getContext(); }

template <typename Derived> class SDBMVisitor {
public:
  /// Visit the given SDBM expression, dispatching to kind-specific functions.
  void visit(SDBMExpr expr) {
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
  void visitPositive(SDBMPositiveExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    if (expr.getKind() == SDBMExprKind::Stripe)
      derived->visitStripe(expr.cast<SDBMStripeExpr>());
    else
      derived->visitInput(expr.cast<SDBMInputExpr>());
  }

  /// Default implementation of visitInput dispatches to the special
  /// functions for dimensions or symbols.  Concrete visitors can override it to
  /// visit all variables instead.
  void visitInput(SDBMInputExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    if (expr.getKind() == SDBMExprKind::DimId)
      derived->visitDim(expr.cast<SDBMDimExpr>());
    else
      derived->visitSymbol(expr.cast<SDBMSymbolExpr>());
  }

  /// Default implementation of visitVarying dispatches to the special
  /// functions for variables and negations thereof.  Concerete visitors can
  /// override it to visit all variables and negations isntead.
  void visitVarying(SDBMVaryingExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    if (auto var = expr.dyn_cast<SDBMPositiveExpr>())
      derived->visitPositive(var);
    else if (auto neg = expr.dyn_cast<SDBMNegExpr>())
      derived->visitNeg(neg);
    else if (auto sum = expr.dyn_cast<SDBMSumExpr>())
      derived->visitSum(sum);
    else if (auto diff = expr.dyn_cast<SDBMDiffExpr>())
      derived->visitDiff(diff);

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

void SDBMExpr::dump() const { print(llvm::errs()); }

//===----------------------------------------------------------------------===//
// SDBMSumExpr
//===----------------------------------------------------------------------===//

SDBMVaryingExpr SDBMSumExpr::getLHS() const {
  return static_cast<ImplType *>(impl)->lhs;
}

SDBMConstantExpr SDBMSumExpr::getRHS() const {
  return static_cast<ImplType *>(impl)->rhs;
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
