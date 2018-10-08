//===- HyperRectangularSet.cpp - MLIR HyperRectangularSet Class -----------===//
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

#include "mlir/Analysis/HyperRectangularSet.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IntegerSet.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace mlir;

// TODO(bondhugula): clean this code up.
// Get the constant bound that is either the min or max (depending on 'cmp').
static Optional<int64_t>
getReducedConstBound(const HyperRectangularSet &set, unsigned *idx,
                     std::function<bool(int64_t, int64_t)> const &cmp) {
  Optional<int64_t> val = None;

  for (unsigned i = 0, n = set.getNumDims(); i < n; i++) {
    auto &ubs = set.getLowerBound(i);
    unsigned j = 0;
    AffineBoundExprList::const_iterator it, e;
    for (it = ubs.begin(), e = ubs.end(); it != e; it++, j++) {
      if (auto cExpr = it->dyn_cast<AffineConstantExpr>()) {
        if (val == None) {
          val = cExpr->getValue();
          *idx = j;
        } else if (cmp(cExpr->getValue(), val.getValue())) {
          val = cExpr->getValue();
          *idx = j;
        }
      }
    }
  }
  return val;
}

// Merge the two lists of AffineExprClass's into a single one, avoiding
// duplicates. lb specifies whether the bound lists are for a lower bound or an
// upper bound.
// TODO(bondhugula): clean this code up.
static void mergeBounds(const HyperRectangularSet &set,
                        AffineBoundExprList &lhsList,
                        const AffineBoundExprList &rhsList, bool lb) {
  // The list of bounds is going to be small. Just a linear search
  // should be enough to create a list without duplicates.
  for (auto expr : rhsList) {
    AffineBoundExprList::const_iterator it;
    for (it = lhsList.begin(); it != lhsList.end(); it++) {
      if (expr == *it)
        break;
    }
    if (it == lhsList.end()) {
      // There can only be one constant affine expr in this bound list.
      if (auto cExpr = expr.dyn_cast<AffineConstantExpr>()) {
        unsigned idx;
        if (lb) {
          auto cb = getReducedConstBound(
              set, &idx,
              [](int64_t newVal, int64_t oldVal) { return newVal < oldVal; });
          if (!cb.hasValue()) {
            lhsList.push_back(expr);
            continue;
          }
          if (cExpr->getValue() < cb)
            lhsList[idx] = expr;
          // A constant value >= the existing bound constant.
          continue;
        }
        // Upper bound case.
        auto cb =
            getReducedConstBound(set, &idx, [](int64_t newVal, int64_t oldVal) {
              return newVal > oldVal;
            });
        if (!cb.hasValue()) {
          lhsList.push_back(expr);
          continue;
        }
        if (cExpr->getValue() > cb)
          lhsList[idx] = expr;
        continue;
      }
      // Not a constant expression; push it.
      // TODO(bondhugula): check if this was implied by an existing symbolic
      // expression or by the context.
      lhsList.push_back(expr);
    }
  }
}

HyperRectangularSet::HyperRectangularSet(unsigned numDims, unsigned numSymbols,
                                         ArrayRef<ArrayRef<AffineExpr>> lbs,
                                         ArrayRef<ArrayRef<AffineExpr>> ubs,
                                         MLIRContext *context,
                                         IntegerSet *symbolContext)
    : context(symbolContext ? MutableIntegerSet(symbolContext, context)
                            : MutableIntegerSet(numDims, numSymbols, context)) {
  unsigned d = 0;
  for (auto boundList : lbs) {
    AffineBoundExprList lb;
    for (auto expr : boundList) {
      assert(expr->isSymbolicOrConstant() &&
             "bound expression should be symbolic or constant");
      lb.push_back(expr);
    }
    mergeBounds(*this, lowerBounds[d++], lb, true);
  }

  d = 0;
  for (auto boundList : ubs) {
    AffineBoundExprList ub;
    for (auto expr : boundList) {
      assert(expr->isSymbolicOrConstant() &&
             "bound expression should be symbolic or constant");
      ub.push_back(expr);
    }
    mergeBounds(*this, upperBounds[d++], ub, false);
  }

  simplifyUnderContext();
}

void HyperRectangularSet::projectOut(unsigned idx, unsigned num) {
  // Erase the bounds along the projected out dimensions.
  lowerBounds.erase(lowerBounds.begin() + idx, lowerBounds.begin() + idx + num);
  upperBounds.erase(upperBounds.begin() + idx, upperBounds.begin() + idx + num);
  numDims -= num;
}

void HyperRectangularSet::intersect(const HyperRectangularSet &rhs) {
  assert(rhs.getNumSymbols() == getNumSymbols() &&
         rhs.getNumDims() == getNumDims() && "operand space does not match");

  // Intersection is just a concatenation of distinct bounds.
  for (unsigned i = 0, n = getNumDims(); i < n; i++) {
    mergeBounds(*this, getLowerBound(i), rhs.getLowerBound(i), true);
    mergeBounds(*this, getUpperBound(i), rhs.getUpperBound(i), false);
  }
}

void HyperRectangularSet::print(raw_ostream &os) const {
  os << "Hyper rectangular set: " << numDims << "dimensions, " << numSymbols
     << "symbols\n";
  os << "Lower bounds\n";
  unsigned d = 0;
  for (auto &lb : lowerBounds) {
    os << "Dim " << d++ << "\n";
    for (auto expr : lb) {
      expr->print(os);
    }
  }
  d = 0;
  os << "Upper bounds\n";
  for (auto &lb : upperBounds) {
    os << "Dim " << d++ << "\n";
    for (auto expr : lb) {
      expr->print(os);
    }
  }
}

void HyperRectangleList::projectOut(unsigned idx, unsigned num) {
  for (auto &elt : hyperRectangles) {
    elt.projectOut(idx, num);
  }
  // TODO: after a project out, some of the sets may be identical. Remove those.
}

bool HyperRectangleList::empty() const {
  for (auto &set : hyperRectangles) {
    if (!set.empty())
      return false;
  }
  return true;
}

bool HyperRectangularSet::empty() const {
  assert(0 && "unimplemented");
  return false;
}

void HyperRectangularSet::dump() const { print(llvm::errs()); }
