//===- VectorAnalysis.cpp - Analysis for Vectorization --------------------===//
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

#include "mlir/Analysis/VectorAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Statements.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/STLExtras.h"

///
/// Implements Analysis functions specific to vectors which support
/// the vectorization and vectorization materialization passes.
///

using namespace mlir;

Optional<SmallVector<unsigned, 4>> mlir::shapeRatio(ArrayRef<int> superShape,
                                                    ArrayRef<int> subShape) {
  if (superShape.size() < subShape.size()) {
    return Optional<SmallVector<unsigned, 4>>();
  }

  // Starting from the end, compute the integer divisors.
  // Set the boolean `divides` if integral division is not possible.
  std::vector<unsigned> result;
  result.reserve(superShape.size());
  bool divides = true;
  auto divide = [&divides, &result](int superSize, int subSize) {
    assert(superSize > 0 && "superSize must be > 0");
    assert(subSize > 0 && "subSize must be > 0");
    divides &= (superSize % subSize == 0);
    result.push_back(superSize / subSize);
  };
  functional::zipApply(
      divide, SmallVector<int, 8>{superShape.rbegin(), superShape.rend()},
      SmallVector<int, 8>{subShape.rbegin(), subShape.rend()});

  // If integral division does not occur, return and let the caller decide.
  if (!divides) {
    return None;
  }

  // At this point we computed the ratio (in reverse) for the common
  // size. Fill with the remaining entries from the super-vector shape (still in
  // reverse).
  int commonSize = subShape.size();
  std::copy(superShape.rbegin() + commonSize, superShape.rend(),
            std::back_inserter(result));

  assert(result.size() == superShape.size() &&
         "super to sub shape ratio is not of the same size as the super rank");

  // Reverse again to get it back in the proper order and return.
  return SmallVector<unsigned, 4>{result.rbegin(), result.rend()};
}

Optional<SmallVector<unsigned, 4>> mlir::shapeRatio(VectorType superVectorType,
                                                    VectorType subVectorType) {
  assert(superVectorType.getElementType() == subVectorType.getElementType() &&
         "NYI: vector types must be of the same elemental type");
  return shapeRatio(superVectorType.getShape(), subVectorType.getShape());
}

AffineMap mlir::makePermutationMap(MemRefType memrefType,
                                   VectorType vectorType) {
  unsigned memRefRank = memrefType.getRank();
  unsigned vectorRank = vectorType.getRank();
  assert(memRefRank >= vectorRank && "Broadcast not supported");
  unsigned offset = memRefRank - vectorRank;
  SmallVector<AffineExpr, 4> perm;
  perm.reserve(memRefRank);
  for (unsigned i = 0; i < vectorRank; ++i) {
    perm.push_back(getAffineDimExpr(offset + i, memrefType.getContext()));
  }
  return AffineMap::get(memRefRank, 0, perm, {});
}

bool mlir::matcher::operatesOnStrictSuperVectors(const OperationStmt &opStmt,
                                                 VectorType subVectorType) {
  // First, extract the vector type and ditinguish between:
  //   a. ops that *must* lower a super-vector (i.e. vector_transfer_read,
  //      vector_transfer_write); and
  //   b. ops that *may* lower a super-vector (all other ops).
  // The ops that *may* lower a super-vector only do so if the super-vector to
  // sub-vector ratio is striclty greater than 1. The ops that *must* lower a
  // super-vector are explicitly checked for this property.
  /// TODO(ntv): there should be a single function for all ops to do this so we
  /// do not have to special case. Maybe a trait, or just a method, unclear atm.
  bool mustDivide = false;
  VectorType superVectorType;
  if (auto read = opStmt.dyn_cast<VectorTransferReadOp>()) {
    superVectorType = read->getResultType();
    mustDivide = true;
  } else if (auto write = opStmt.dyn_cast<VectorTransferWriteOp>()) {
    superVectorType = write->getVectorType();
    mustDivide = true;
  } else if (opStmt.getNumResults() == 0) {
    assert(opStmt.isa<ReturnOp>() &&
           "NYI: assuming only return statements can have 0 results at this "
           "point");
    return false;
  } else if (opStmt.getNumResults() == 1) {
    if (auto v = opStmt.getResult(0)->getType().dyn_cast<VectorType>()) {
      superVectorType = v;
    } else {
      // Not a vector type.
      return false;
    }
  } else {
    // Not a vector_transfer and has more than 1 result, fail hard for now to
    // wake us up when something changes.
    assert(false && "NYI: statement has more than 1 result");
    return false;
  }

  // Get the ratio.
  auto ratio = shapeRatio(superVectorType, subVectorType);

  // Sanity check.
  assert((ratio.hasValue() || !mustDivide) &&
         "NYI: vector_transfer instruction in which super-vector size is not an"
         " integer multiple of sub-vector size");

  // This catches cases that are not strictly necessary to have multiplicity but
  // still aren't divisible by the sub-vector shape.
  // This could be useful information if we wanted to reshape at the level of
  // the vector type (but we would have to look at the compute and distinguish
  // between parallel, reduction and possibly other cases.
  if (!ratio.hasValue()) {
    return false;
  }

  // A strict super-vector is at least 2 sub-vectors.
  for (auto m : *ratio) {
    if (m > 1) {
      return true;
    }
  }

  // Not a strict super-vector.
  return false;
}
