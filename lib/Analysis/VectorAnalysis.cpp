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
#include "mlir/Support/Functional.h"
#include "mlir/Support/STLExtras.h"

///
/// Implements Analysis functions specific to vectors which support
/// the vectorization and vectorization materialization passes.
///

using namespace mlir;

bool mlir::isaVectorTransferRead(const OperationStmt &stmt) {
  return stmt.getName().getStringRef().str() == kVectorTransferReadOpName;
}

bool mlir::isaVectorTransferWrite(const OperationStmt &stmt) {
  return stmt.getName().getStringRef().str() == kVectorTransferWriteOpName;
}

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
  functional::zip(divide,
                  SmallVector<int, 8>{superShape.rbegin(), superShape.rend()},
                  SmallVector<int, 8>{subShape.rbegin(), subShape.rend()});

  // If integral division does not occur, return and let the caller decide.
  if (!divides) {
    return Optional<SmallVector<unsigned, 4>>();
  }

  // At this point we computed the multiplicity (in reverse) for the common
  // size. Fill with the remaining entries from the super-vector shape (still in
  // reverse).
  int commonSize = subShape.size();
  std::copy(superShape.rbegin() + commonSize, superShape.rend(),
            std::back_inserter(result));

  assert(result.size() == superShape.size() &&
         "multiplicity must be of the same size as the super-vector rank");

  // Reverse again to get it back in the proper order and return.
  return SmallVector<unsigned, 4>{result.rbegin(), result.rend()};
}

Optional<SmallVector<unsigned, 4>> mlir::shapeRatio(VectorType superVectorType,
                                                    VectorType subVectorType) {
  assert(superVectorType.getElementType() == subVectorType.getElementType() &&
         "NYI: vector types must be of the same elemental type");
  assert(superVectorType.getElementType() ==
             Type::getF32(superVectorType.getContext()) &&
         "Only f32 supported for now");
  return shapeRatio(superVectorType.getShape(), subVectorType.getShape());
}

/// Matches vector_transfer_read, vector_transfer_write and ops that return a
/// vector type that is at least a 2-multiple of the sub-vector type size.
/// This allows leaving other vector types in the function untouched and avoids
/// interfering with operations on those.
/// This is a first approximation, it can easily be extended in the future.
/// TODO(ntv): this could all be much simpler if we added a bit that a vector
/// type to mark that a vector is a strict super-vector but it is not strictly
/// needed so let's avoid adding even 1 extra bit in the IR for now.
bool mlir::matcher::operatesOnStrictSuperVectors(const OperationStmt &opStmt,
                                                 VectorType subVectorType) {
  // First, extract the vector type and ditinguish between:
  //   a. ops that *must* lower a super-vector (i.e. vector_transfer_read,
  //      vector_transfer_write); and
  //   b. ops that *may* lower a super-vector (all other ops).
  // The ops that *may* lower a super-vector only do so if the vector size is
  // an integer multiple of the HW vector size, with multiplicity 1.
  // The ops that *must* lower a super-vector are explicitly checked for this
  // property.
  /// TODO(ntv): there should be a single function for all ops to do this so we
  /// do not have to special case. Maybe a trait, or just a method, unclear atm.
  bool mustDivide = false;
  VectorType superVectorType;
  if (isaVectorTransferRead(opStmt)) {
    superVectorType = opStmt.getResult(0)->getType().cast<VectorType>();
    mustDivide = true;
  } else if (isaVectorTransferWrite(opStmt)) {
    // TODO(ntv): if vector_transfer_write had store-like semantics we could
    // have written something similar to:
    //   auto store = storeOp->cast<StoreOp>();
    //   auto *value = store->getValueToStore();
    superVectorType = opStmt.getOperand(0)->getType().cast<VectorType>();
    mustDivide = true;
  } else if (opStmt.getNumResults() == 0) {
    assert(opStmt.dyn_cast<ReturnOp>() &&
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

  // Get the multiplicity.
  auto multiplicity = shapeRatio(superVectorType, subVectorType);

  // Sanity check.
  assert((multiplicity.hasValue() || !mustDivide) &&
         "NYI: vector_transfer instruction in which super-vector size is not an"
         " integer multiple of sub-vector size");

  // This catches cases that are not strictly necessary to have multiplicity but
  // still aren't divisible by the sub-vector shape.
  // This could be useful information if we wanted to reshape at the level of
  // the vector type (but we would have to look at the compute and distinguish
  // between parallel, reduction and possibly other cases.
  if (!multiplicity.hasValue()) {
    return false;
  }

  // A strict super-vector is at least 2 sub-vectors.
  for (auto m : *multiplicity) {
    if (m > 1) {
      return true;
    }
  }

  // Not a strict super-vector.
  return false;
}
