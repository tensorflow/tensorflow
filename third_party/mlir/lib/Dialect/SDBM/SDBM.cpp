//===- SDBM.cpp - MLIR SDBM implementation --------------------------------===//
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
// A striped difference-bound matrix (SDBM) is a set in Z^N (or R^N) defined
// as {(x_1, ... x_n) | f(x_1, ... x_n) >= 0} where f is an SDBM expression.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SDBM/SDBM.h"
#include "mlir/Dialect/SDBM/SDBMExpr.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

// Helper function for SDBM construction that collects information necessary to
// start building an SDBM in one sweep.  In particular, it records the largest
// position of a dimension in `dim`, that of a symbol in `symbol` as well as
// collects all unique stripe expressions in `stripes`.  Uses SetVector to
// ensure these expressions always have the same order.
static void collectSDBMBuildInfo(SDBMExpr expr, int &dim, int &symbol,
                                 llvm::SmallSetVector<SDBMExpr, 8> &stripes) {
  struct Visitor : public SDBMVisitor<Visitor> {
    void visitDim(SDBMDimExpr dimExpr) {
      int p = dimExpr.getPosition();
      if (p > maxDimPosition)
        maxDimPosition = p;
    }
    void visitSymbol(SDBMSymbolExpr symbExpr) {
      int p = symbExpr.getPosition();
      if (p > maxSymbPosition)
        maxSymbPosition = p;
    }
    void visitStripe(SDBMStripeExpr stripeExpr) { stripes.insert(stripeExpr); }

    Visitor(llvm::SmallSetVector<SDBMExpr, 8> &stripes) : stripes(stripes) {}

    int maxDimPosition = -1;
    int maxSymbPosition = -1;
    llvm::SmallSetVector<SDBMExpr, 8> &stripes;
  };

  Visitor visitor(stripes);
  visitor.walkPostorder(expr);
  dim = std::max(dim, visitor.maxDimPosition);
  symbol = std::max(symbol, visitor.maxSymbPosition);
}

namespace {
// Utility class for SDBMBuilder.  Represents a value that can be inserted in
// the SDB matrix that corresponds to "v0 - v1 + C <= 0", where v0 and v1 is
// any combination of the positive and negative positions.  Since multiple
// variables can be declared equal to the same stripe expression, the
// constraints on this expression must be reflected to all these variables.  For
// example, if
//   d0 = s0 # 42
//   d1 = s0 # 42
//   d2 = s1 # 2
//   d3 = s1 # 2
// the constraint
//   s0 # 42 - s1 # 2 <= C
// should be reflected in the DB matrix as
//   d0 - d2 <= C
//   d1 - d2 <= C
//   d0 - d3 <= C
//   d1 - d3 <= C
// since the DB matrix has no knowledge of the transitive equality between d0,
// d1 and s0 # 42 as well as between d2, d3 and s1 # 2.  This knowledge can be
// obtained by computing a transitive closure, which is impossible until the
// DBM is actually built.
struct SDBMBuilderResult {
  // Positions in the matrix of the variables taken with the "+" sign in the
  // difference expression, 0 if it is a constant rather than a variable.
  llvm::SmallVector<unsigned, 2> positivePos;

  // Positions in the matrix of the variables taken with the "-" sign in the
  // difference expression, 0 if it is a constant rather than a variable.
  llvm::SmallVector<unsigned, 2> negativePos;

  // Constant value in the difference expression.
  int64_t value = 0;
};

// Visitor for building an SDBM from SDBM expressions.  After traversing an SDBM
// expression, produces an update to the SDB matrix specifying the positions in
// the matrix and the negated value that should be stored.  Both the positive
// and the negative positions may be lists of indices in cases where multiple
// variables are equal to the same stripe expression.  In such cases, the update
// applies to the cross product of positions because elements involved in the
// update are (transitively) equal and should have the same constraints, but we
// may not have an explicit equality for them.
struct SDBMBuilder : public SDBMVisitor<SDBMBuilder, SDBMBuilderResult> {
public:
  // A difference expression produces both the positive and the negative
  // coordinate in the matrix, recursively traversing the LHS and the RHS. The
  // value is the difference between values obtained from LHS and RHS.
  SDBMBuilderResult visitDiff(SDBMDiffExpr diffExpr) {
    auto lhs = visit(diffExpr.getLHS());
    auto rhs = visit(diffExpr.getRHS());
    assert(lhs.negativePos.size() == 1 && lhs.negativePos[0] == 0 &&
           "unexpected negative expression in a difference expression");
    assert(rhs.negativePos.size() == 1 && lhs.negativePos[0] == 0 &&
           "unexpected negative expression in a difference expression");

    SDBMBuilderResult result;
    result.positivePos = lhs.positivePos;
    result.negativePos = rhs.positivePos;
    result.value = lhs.value - rhs.value;
    return result;
  }

  // An input expression is always taken with the "+" sign and therefore
  // produces a positive coordinate keeping the negative coordinate zero for an
  // eventual constant.
  SDBMBuilderResult visitInput(SDBMInputExpr expr) {
    SDBMBuilderResult r;
    r.positivePos.push_back(linearPosition(expr));
    r.negativePos.push_back(0);
    return r;
  }

  // A stripe expression is always equal to one or more variables, which may be
  // temporaries, and appears with a "+" sign in the SDBM expression tree. Take
  // the positions of the corresponding variables as positive coordinates.
  SDBMBuilderResult visitStripe(SDBMStripeExpr expr) {
    SDBMBuilderResult r;
    assert(pointExprToStripe.count(expr));
    r.positivePos = pointExprToStripe[expr];
    r.negativePos.push_back(0);
    return r;
  }

  // A constant expression has both coordinates at zero.
  SDBMBuilderResult visitConstant(SDBMConstantExpr expr) {
    SDBMBuilderResult r;
    r.positivePos.push_back(0);
    r.negativePos.push_back(0);
    r.value = expr.getValue();
    return r;
  }

  // A negation expression swaps the positive and the negative coordinates
  // and also negates the constant value.
  SDBMBuilderResult visitNeg(SDBMNegExpr expr) {
    SDBMBuilderResult result = visit(expr.getVar());
    std::swap(result.positivePos, result.negativePos);
    result.value = -result.value;
    return result;
  }

  // The RHS of a sum expression must be a constant and therefore must have both
  // positive and negative coordinates at zero.  Take the sum of the values
  // between LHS and RHS and keep LHS coordinates.
  SDBMBuilderResult visitSum(SDBMSumExpr expr) {
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    for (auto pos : rhs.negativePos) {
      (void)pos;
      assert(pos == 0 && "unexpected variable on the RHS of SDBM sum");
    }
    for (auto pos : rhs.positivePos) {
      (void)pos;
      assert(pos == 0 && "unexpected variable on the RHS of SDBM sum");
    }

    lhs.value += rhs.value;
    return lhs;
  }

  SDBMBuilder(llvm::DenseMap<SDBMExpr, llvm::SmallVector<unsigned, 2>>
                  &pointExprToStripe,
              llvm::function_ref<unsigned(SDBMInputExpr)> callback)
      : pointExprToStripe(pointExprToStripe), linearPosition(callback) {}

  llvm::DenseMap<SDBMExpr, llvm::SmallVector<unsigned, 2>> &pointExprToStripe;
  llvm::function_ref<unsigned(SDBMInputExpr)> linearPosition;
};
} // namespace

SDBM SDBM::get(ArrayRef<SDBMExpr> inequalities, ArrayRef<SDBMExpr> equalities) {
  SDBM result;

  // TODO(zinenko): consider detecting equalities in the list of inequalities.
  // This is potentially expensive and requires to
  //   - create a list of negated inequalities (may allocate under lock);
  //   - perform a pairwise comparison of direct and negated inequalities;
  //   - copy the lists of equalities and inequalities, and move entries between
  //     them;
  // only for the purpose of sparing a temporary variable in cases where an
  // implicit equality between a variable and a stripe expression is present in
  // the input.

  // Do the first sweep over (in)equalities to collect the information necessary
  // to allocate the SDB matrix (number of dimensions, symbol and temporary
  // variables required for stripe expressions).
  llvm::SmallSetVector<SDBMExpr, 8> stripes;
  int maxDim = -1;
  int maxSymbol = -1;
  for (auto expr : inequalities)
    collectSDBMBuildInfo(expr, maxDim, maxSymbol, stripes);
  for (auto expr : equalities)
    collectSDBMBuildInfo(expr, maxDim, maxSymbol, stripes);
  // Indexing of dimensions starts with 0, obtain the number of dimensions by
  // incrementing the maximal position of the dimension seen in expressions.
  result.numDims = maxDim + 1;
  result.numSymbols = maxSymbol + 1;
  result.numTemporaries = 0;

  // Helper function that returns the position of the variable represented by
  // an SDBM input expression.
  auto linearPosition = [result](SDBMInputExpr expr) {
    if (expr.isa<SDBMDimExpr>())
      return result.getDimPosition(expr.getPosition());
    return result.getSymbolPosition(expr.getPosition());
  };

  // Check if some stripe expressions are equal to another variable. In
  // particular, look for the equalities of the form
  //   d0 - stripe-expression = 0, or
  //   stripe-expression - d0 = 0.
  // There may be multiple variables that are equal to the same stripe
  // expression.  Keep track of those in pointExprToStripe.
  // There may also be multiple stripe expressions equal to the same variable.
  // Introduce a temporary variable for each of those.
  llvm::DenseMap<SDBMExpr, llvm::SmallVector<unsigned, 2>> pointExprToStripe;
  unsigned numTemporaries = 0;

  auto updateStripePointMaps = [&numTemporaries, &result, &pointExprToStripe,
                                linearPosition](SDBMInputExpr input,
                                                SDBMExpr expr) {
    unsigned position = linearPosition(input);
    if (result.stripeToPoint.count(position) &&
        result.stripeToPoint[position] != expr) {
      position = result.getNumVariables() + numTemporaries++;
    }
    pointExprToStripe[expr].push_back(position);
    result.stripeToPoint.insert(std::make_pair(position, expr));
  };

  for (auto eq : equalities) {
    auto diffExpr = eq.dyn_cast<SDBMDiffExpr>();
    if (!diffExpr)
      continue;

    auto lhs = diffExpr.getLHS();
    auto rhs = diffExpr.getRHS();
    auto lhsInput = lhs.dyn_cast<SDBMInputExpr>();
    auto rhsInput = rhs.dyn_cast<SDBMInputExpr>();

    if (lhsInput && stripes.count(rhs))
      updateStripePointMaps(lhsInput, rhs);
    if (rhsInput && stripes.count(lhs))
      updateStripePointMaps(rhsInput, lhs);
  }

  // Assign the remaining stripe expressions to temporary variables.  These
  // expressions are the ones that could not be associated with an existing
  // variable in the previous step.
  for (auto expr : stripes) {
    if (pointExprToStripe.count(expr))
      continue;
    unsigned position = result.getNumVariables() + numTemporaries++;
    pointExprToStripe[expr].push_back(position);
    result.stripeToPoint.insert(std::make_pair(position, expr));
  }

  // Create the DBM matrix, initialized to infinity values for the least tight
  // possible bound (x - y <= infinity is always true).
  result.numTemporaries = numTemporaries;
  result.matrix.resize(result.getNumVariables() * result.getNumVariables(),
                       IntInfty::infinity());

  SDBMBuilder builder(pointExprToStripe, linearPosition);

  // Only keep the tightest constraint.  Since we transform everything into
  // less-than-or-equals-to inequalities, keep the smallest constant.  For
  // example, if we have d0 - d1 <= 42 and d0 - d1 <= 2, we keep the latter.
  // Note that the input expressions are in the shape of d0 - d1 + -42 <= 0
  // so we negate the value before storing it.
  // In case where the positive and the negative positions are equal, the
  // corresponding expression has the form d0 - d0 + -42 <= 0.  If the constant
  // value is positive, the set defined by SDBM is trivially empty.  We store
  // this value anyway and continue processing to maintain the correspondence
  // between the matrix form and the list-of-SDBMExpr form.
  // TODO(zinenko): we may want to reconsider this once we have canonicalization
  // or simplification in place
  auto updateMatrix = [](SDBM &sdbm, const SDBMBuilderResult &r) {
    for (auto positivePos : r.positivePos) {
      for (auto negativePos : r.negativePos) {
        auto &m = sdbm.at(negativePos, positivePos);
        m = m < -r.value ? m : -r.value;
      }
    }
  };

  // Do the second sweep on (in)equalities, updating the SDB matrix to reflect
  // the constraints.
  for (auto ineq : inequalities)
    updateMatrix(result, builder.visit(ineq));

  // An equality f(x) = 0 is represented as a pair of inequalities {f(x) >= 0;
  // f(x) <= 0} or, alternatively, {-f(x) <= 0 and f(x) <= 0}.
  for (auto eq : equalities) {
    updateMatrix(result, builder.visit(eq));
    updateMatrix(result, builder.visit(-eq));
  }

  // Add the inequalities induced by stripe equalities.
  //   t = x # C  =>  t <= x <= t + C - 1
  // which is equivalent to
  //   {t - x <= 0;
  //    x - t - (C - 1) <= 0}.
  for (const auto &pair : result.stripeToPoint) {
    auto stripe = pair.second.cast<SDBMStripeExpr>();
    SDBMBuilderResult update = builder.visit(stripe.getLHS());
    assert(update.negativePos.size() == 1 && update.negativePos[0] == 0 &&
           "unexpected negated variable in stripe expression");
    assert(update.value == 0 &&
           "unexpected non-zero value in stripe expression");
    update.negativePos.clear();
    update.negativePos.push_back(pair.first);
    update.value = -(stripe.getStripeFactor().getValue() - 1);
    updateMatrix(result, update);

    std::swap(update.negativePos, update.positivePos);
    update.value = 0;
    updateMatrix(result, update);
  }

  return result;
}

// Given a row and a column position in the square DBM, insert one equality
// or up to two inequalities that correspond the entries (col, row) and (row,
// col) in the DBM.  `rowExpr` and `colExpr` contain the expressions such that
// colExpr - rowExpr <= V where V is the value at (row, col) in the DBM.
// If one of the expressions is derived from another using a stripe operation,
// check if the inequalities induced by the stripe operation subsume the
// inequalities defined in the DBM and if so, elide these inequalities.
void SDBM::convertDBMElement(unsigned row, unsigned col, SDBMTermExpr rowExpr,
                             SDBMTermExpr colExpr,
                             SmallVectorImpl<SDBMExpr> &inequalities,
                             SmallVectorImpl<SDBMExpr> &equalities) {
  using ops_assertions::operator+;
  using ops_assertions::operator-;

  auto diffIJValue = at(col, row);
  auto diffJIValue = at(row, col);

  // If symmetric entries are opposite, the corresponding expressions are equal.
  if (diffIJValue.isFinite() &&
      diffIJValue.getValue() == -diffJIValue.getValue()) {
    equalities.push_back(rowExpr - colExpr - diffIJValue.getValue());
    return;
  }

  // Given an inequality x0 - x1 <= A, check if x0 is a stripe variable derived
  // from x1: x0 = x1 # B.  If so, it would imply the constraints
  // x0 <= x1 <= x0 + (B - 1) <=> x0 - x1 <= 0 and x1 - x0 <= (B - 1).
  // Therefore, if A >= 0, this inequality is subsumed by that implied
  // by the stripe equality and thus can be elided.
  // Similarly, check if x1 is a stripe variable derived from x0: x1 = x0 # C.
  // If so, it would imply the constraints x1 <= x0 <= x1 + (C - 1) <=>
  // <=> x1 - x0 <= 0 and x0 - x1 <= (C - 1).  Therefore, if A >= (C - 1), this
  // inequality can be elided.
  //
  // Note: x0 and x1 may be a stripe expressions themselves, we rely on stripe
  // expressions being stored without temporaries on the RHS and being passed
  // into this function as is.
  auto canElide = [this](unsigned x0, unsigned x1, SDBMExpr x0Expr,
                         SDBMExpr x1Expr, int64_t value) {
    if (stripeToPoint.count(x0)) {
      auto stripe = stripeToPoint[x0].cast<SDBMStripeExpr>();
      SDBMDirectExpr var = stripe.getLHS();
      if (x1Expr == var && value >= 0)
        return true;
    }
    if (stripeToPoint.count(x1)) {
      auto stripe = stripeToPoint[x1].cast<SDBMStripeExpr>();
      SDBMDirectExpr var = stripe.getLHS();
      if (x0Expr == var && value >= stripe.getStripeFactor().getValue() - 1)
        return true;
    }
    return false;
  };

  // Check row - col.
  if (diffIJValue.isFinite() &&
      !canElide(row, col, rowExpr, colExpr, diffIJValue.getValue())) {
    inequalities.push_back(rowExpr - colExpr - diffIJValue.getValue());
  }
  // Check col - row.
  if (diffJIValue.isFinite() &&
      !canElide(col, row, colExpr, rowExpr, diffJIValue.getValue())) {
    inequalities.push_back(colExpr - rowExpr - diffJIValue.getValue());
  }
}

// The values on the main diagonal correspond to the upper bound on the
// difference between a variable and itself: d0 - d0 <= C, or alternatively
// to -C <= 0.  Only construct the inequalities when C is negative, which
// are trivially false but necessary for the returned system of inequalities
// to indicate that the set it defines is empty.
void SDBM::convertDBMDiagonalElement(unsigned pos, SDBMTermExpr expr,
                                     SmallVectorImpl<SDBMExpr> &inequalities) {
  auto selfDifference = at(pos, pos);
  if (selfDifference.isFinite() && selfDifference < 0) {
    auto selfDifferenceValueExpr =
        SDBMConstantExpr::get(expr.getDialect(), -selfDifference.getValue());
    inequalities.push_back(selfDifferenceValueExpr);
  }
}

void SDBM::getSDBMExpressions(SDBMDialect *dialect,
                              SmallVectorImpl<SDBMExpr> &inequalities,
                              SmallVectorImpl<SDBMExpr> &equalities) {
  using ops_assertions::operator-;
  using ops_assertions::operator+;

  // Helper function that creates an SDBMInputExpr given the linearized position
  // of variable in the DBM.
  auto getInput = [dialect, this](unsigned matrixPos) -> SDBMInputExpr {
    if (matrixPos < numDims)
      return SDBMDimExpr::get(dialect, matrixPos);
    return SDBMSymbolExpr::get(dialect, matrixPos - numDims);
  };

  // The top-left value corresponds to inequality 0 <= C.  If C is negative, the
  // set defined by SDBM is trivially empty and we add the constraint -C <= 0 to
  // the list of inequalities.  Otherwise, the constraint is trivially true and
  // we ignore it.
  auto difference = at(0, 0);
  if (difference.isFinite() && difference < 0) {
    inequalities.push_back(
        SDBMConstantExpr::get(dialect, -difference.getValue()));
  }

  // Traverse the segment of the matrix that involves non-temporary variables.
  unsigned numTrueVariables = numDims + numSymbols;
  for (unsigned i = 0; i < numTrueVariables; ++i) {
    // The first row and column represent numerical upper and lower bound on
    // each variable.  Transform them into inequalities if they are finite.
    auto upperBound = at(0, 1 + i);
    auto lowerBound = at(1 + i, 0);
    auto inputExpr = getInput(i);
    if (upperBound.isFinite() &&
        upperBound.getValue() == -lowerBound.getValue()) {
      equalities.push_back(inputExpr - upperBound.getValue());
    } else if (upperBound.isFinite()) {
      inequalities.push_back(inputExpr - upperBound.getValue());
    } else if (lowerBound.isFinite()) {
      inequalities.push_back(-inputExpr - lowerBound.getValue());
    }

    // Introduce trivially false inequalities if required by diagonal elements.
    convertDBMDiagonalElement(1 + i, inputExpr, inequalities);

    // Introduce equalities or inequalities between non-temporary variables.
    for (unsigned j = 0; j < i; ++j) {
      convertDBMElement(1 + i, 1 + j, getInput(i), getInput(j), inequalities,
                        equalities);
    }
  }

  // Add equalities for stripe expressions that define non-temporary
  // variables.  Temporary variables will be substituted into their uses and
  // should not appear in the resulting equalities.
  for (const auto &stripePair : stripeToPoint) {
    unsigned position = stripePair.first;
    if (position < 1 + numTrueVariables) {
      equalities.push_back(getInput(position - 1) - stripePair.second);
    }
  }

  // Add equalities / inequalities involving temporaries by replacing the
  // temporaries with stripe expressions that define them.
  for (unsigned i = 1 + numTrueVariables, e = getNumVariables(); i < e; ++i) {
    // Mixed constraints involving one temporary (j) and one non-temporary (i)
    // variable.
    for (unsigned j = 0; j < numTrueVariables; ++j) {
      convertDBMElement(i, 1 + j, stripeToPoint[i].cast<SDBMStripeExpr>(),
                        getInput(j), inequalities, equalities);
    }

    // Constraints involving only temporary variables.
    for (unsigned j = 1 + numTrueVariables; j < i; ++j) {
      convertDBMElement(i, j, stripeToPoint[i].cast<SDBMStripeExpr>(),
                        stripeToPoint[j].cast<SDBMStripeExpr>(), inequalities,
                        equalities);
    }

    // Introduce trivially false inequalities if required by diagonal elements.
    convertDBMDiagonalElement(i, stripeToPoint[i].cast<SDBMStripeExpr>(),
                              inequalities);
  }
}

void SDBM::print(llvm::raw_ostream &os) {
  unsigned numVariables = getNumVariables();

  // Helper function that prints the name of the variable given its linearized
  // position in the DBM.
  auto getVarName = [this](unsigned matrixPos) -> std::string {
    if (matrixPos == 0)
      return "cst";
    matrixPos -= 1;
    if (matrixPos < numDims)
      return llvm::formatv("d{0}", matrixPos);
    matrixPos -= numDims;
    if (matrixPos < numSymbols)
      return llvm::formatv("s{0}", matrixPos);
    matrixPos -= numSymbols;
    return llvm::formatv("t{0}", matrixPos);
  };

  // Header row.
  os << "      cst";
  for (unsigned i = 1; i < numVariables; ++i) {
    os << llvm::formatv(" {0,4}", getVarName(i));
  }
  os << '\n';

  // Data rows.
  for (unsigned i = 0; i < numVariables; ++i) {
    os << llvm::formatv("{0,-4}", getVarName(i));
    for (unsigned j = 0; j < numVariables; ++j) {
      IntInfty value = operator()(i, j);
      if (!value.isFinite())
        os << "  inf";
      else
        os << llvm::formatv(" {0,4}", value.getValue());
    }
    os << '\n';
  }

  // Explanation of temporaries.
  for (const auto &pair : stripeToPoint) {
    os << getVarName(pair.first) << " = ";
    pair.second.print(os);
    os << '\n';
  }
}

void SDBM::dump() { print(llvm::errs()); }
