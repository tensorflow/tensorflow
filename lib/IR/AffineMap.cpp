//===- AffineMap.cpp - MLIR Affine Map Classes ----------------------------===//
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

#include "mlir/IR/AffineMap.h"
#include "AffineMapDetail.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

// AffineExprConstantFolder evaluates an affine expression using constant
// operands passed in 'operandConsts'. Returns a pointer to an IntegerAttr
// attribute representing the constant value of the affine expression
// evaluated on constant 'operandConsts'.
class AffineExprConstantFolder {
public:
  AffineExprConstantFolder(unsigned numDims, ArrayRef<Attribute> operandConsts)
      : numDims(numDims), operandConsts(operandConsts) {}

  /// Attempt to constant fold the specified affine expr, or return null on
  /// failure.
  IntegerAttr constantFold(AffineExpr expr) {
    switch (expr.getKind()) {
    case AffineExprKind::Add:
      return constantFoldBinExpr(
          expr, [](int64_t lhs, int64_t rhs) { return lhs + rhs; });
    case AffineExprKind::Mul:
      return constantFoldBinExpr(
          expr, [](int64_t lhs, int64_t rhs) { return lhs * rhs; });
    case AffineExprKind::Mod:
      return constantFoldBinExpr(
          expr, [](int64_t lhs, uint64_t rhs) { return mod(lhs, rhs); });
    case AffineExprKind::FloorDiv:
      return constantFoldBinExpr(
          expr, [](int64_t lhs, uint64_t rhs) { return floorDiv(lhs, rhs); });
    case AffineExprKind::CeilDiv:
      return constantFoldBinExpr(
          expr, [](int64_t lhs, uint64_t rhs) { return ceilDiv(lhs, rhs); });
    case AffineExprKind::Constant:
      return IntegerAttr::get(Type::getIndex(expr.getContext()),
                              expr.cast<AffineConstantExpr>().getValue());
    case AffineExprKind::DimId:
      return operandConsts[expr.cast<AffineDimExpr>().getPosition()]
          .dyn_cast_or_null<IntegerAttr>();
    case AffineExprKind::SymbolId:
      return operandConsts[numDims +
                           expr.cast<AffineSymbolExpr>().getPosition()]
          .dyn_cast_or_null<IntegerAttr>();
    }
  }

private:
  // TODO: Change these to operate on APInts too.
  IntegerAttr
  constantFoldBinExpr(AffineExpr expr,
                      std::function<uint64_t(int64_t, uint64_t)> op) {
    auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
    auto lhs = constantFold(binOpExpr.getLHS());
    auto rhs = constantFold(binOpExpr.getRHS());
    if (!lhs || !rhs)
      return nullptr;
    return IntegerAttr::get(lhs.getType(), op(lhs.getInt(), rhs.getInt()));
  }

  // The number of dimension operands in AffineMap containing this expression.
  unsigned numDims;
  // The constant valued operands used to evaluate this AffineExpr.
  ArrayRef<Attribute> operandConsts;
};

} // end anonymous namespace

/// Returns a single constant result affine map.
AffineMap AffineMap::getConstantMap(int64_t val, MLIRContext *context) {
  return get(/*dimCount=*/0, /*symbolCount=*/0,
             {getAffineConstantExpr(val, context)}, {});
}

AffineMap AffineMap::getMultiDimIdentityMap(unsigned numDims,
                                            MLIRContext *context) {
  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(numDims);
  for (unsigned i = 0; i < numDims; ++i)
    dimExprs.push_back(mlir::getAffineDimExpr(i, context));
  return get(/*dimCount=*/numDims, /*symbolCount=*/0, dimExprs, {});
}

MLIRContext *AffineMap::getContext() const { return getResult(0).getContext(); }

bool AffineMap::isBounded() const {
  assert(map && "uninitialized AffineMap");
  return !map->rangeSizes.empty();
}

bool AffineMap::isIdentity() const {
  if (getNumDims() != getNumResults())
    return false;
  ArrayRef<AffineExpr> results = getResults();
  for (unsigned i = 0, numDims = getNumDims(); i < numDims; ++i) {
    auto expr = results[i].dyn_cast<AffineDimExpr>();
    if (!expr || expr.getPosition() != i)
      return false;
  }
  return true;
}

bool AffineMap::isSingleConstant() const {
  return getNumResults() == 1 && getResult(0).isa<AffineConstantExpr>();
}

int64_t AffineMap::getSingleConstantResult() const {
  assert(isSingleConstant() && "map must have a single constant result");
  return getResult(0).cast<AffineConstantExpr>().getValue();
}

unsigned AffineMap::getNumDims() const {
  assert(map && "uninitialized map storage");
  return map->numDims;
}
unsigned AffineMap::getNumSymbols() const {
  assert(map && "uninitialized map storage");
  return map->numSymbols;
}
unsigned AffineMap::getNumResults() const {
  assert(map && "uninitialized map storage");
  return map->results.size();
}
unsigned AffineMap::getNumInputs() const {
  assert(map && "uninitialized map storage");
  return map->numDims + map->numSymbols;
}

ArrayRef<AffineExpr> AffineMap::getResults() const {
  assert(map && "uninitialized map storage");
  return map->results;
}
AffineExpr AffineMap::getResult(unsigned idx) const {
  assert(map && "uninitialized map storage");
  return map->results[idx];
}
ArrayRef<AffineExpr> AffineMap::getRangeSizes() const {
  assert(map && "uninitialized map storage");
  return map->rangeSizes;
}

/// Folds the results of the application of an affine map on the provided
/// operands to a constant if possible. Returns false if the folding happens,
/// true otherwise.
bool AffineMap::constantFold(ArrayRef<Attribute> operandConstants,
                             SmallVectorImpl<Attribute> &results) const {
  assert(getNumInputs() == operandConstants.size());

  // Fold each of the result expressions.
  AffineExprConstantFolder exprFolder(getNumDims(), operandConstants);
  // Constant fold each AffineExpr in AffineMap and add to 'results'.
  for (auto expr : getResults()) {
    auto folded = exprFolder.constantFold(expr);
    // If we didn't fold to a constant, then folding fails.
    if (!folded)
      return true;

    results.push_back(folded);
  }
  assert(results.size() == getNumResults() &&
         "constant folding produced the wrong number of results");
  return false;
}

/// Walk all of the AffineExpr's in this mapping.  The results are visited
/// first, and then the range sizes (if present).  Each node in an expression
/// tree is visited in postorder.
void AffineMap::walkExprs(std::function<void(AffineExpr)> callback) const {
  for (auto expr : getResults())
    expr.walk(callback);

  for (auto expr : getRangeSizes())
    expr.walk(callback);
}

/// This method substitutes any uses of dimensions and symbols (e.g.
/// dim#0 with dimReplacements[0]) in subexpressions and returns the modified
/// expression mapping.  Because this can be used to eliminate dims and
/// symbols, the client needs to specify the number of dims and symbols in
/// the result.  The returned map always has the same number of results.
AffineMap AffineMap::replaceDimsAndSymbols(ArrayRef<AffineExpr> dimReplacements,
                                           ArrayRef<AffineExpr> symReplacements,
                                           unsigned numResultDims,
                                           unsigned numResultSyms) {
  SmallVector<AffineExpr, 8> results;
  results.reserve(getNumResults());
  for (auto expr : getResults())
    results.push_back(
        expr.replaceDimsAndSymbols(dimReplacements, symReplacements));

  SmallVector<AffineExpr, 8> resultRanges;
  resultRanges.reserve(getRangeSizes().size());
  for (auto expr : getRangeSizes())
    resultRanges.push_back(
        expr.replaceDimsAndSymbols(dimReplacements, symReplacements));

  return get(numResultDims, numResultSyms, results, resultRanges);
}

AffineMap AffineMap::compose(AffineMap map) {
  assert(getNumDims() == map.getNumResults() && "Number of results mismatch");
  assert(getRangeSizes().empty() && "TODO: support bounded AffineMap");
  assert(map.getRangeSizes().empty() && "TODO: support bounded AffineMap");
  // Prepare `map` by concatenating the symbols and rewriting its exprs.
  unsigned numDims = map.getNumDims();
  unsigned numSymbolsThisMap = getNumSymbols();
  unsigned numSymbols = numSymbolsThisMap + map.getNumSymbols();
  SmallVector<AffineExpr, 8> newDims(numDims);
  for (unsigned idx = 0; idx < numDims; ++idx) {
    newDims[idx] = getAffineDimExpr(idx, getContext());
  }
  SmallVector<AffineExpr, 8> newSymbols(numSymbols);
  for (unsigned idx = numSymbolsThisMap; idx < numSymbols; ++idx) {
    newSymbols[idx - numSymbolsThisMap] =
        getAffineSymbolExpr(idx, getContext());
  }
  auto newMap =
      map.replaceDimsAndSymbols(newDims, newSymbols, numDims, numSymbols);
  SmallVector<AffineExpr, 8> exprs;
  exprs.reserve(getResults().size());
  for (auto expr : getResults())
    exprs.push_back(expr.compose(newMap));
  return AffineMap::get(numDims, numSymbols, exprs, {});
}
