//===- Utils.cpp ---- Misc utilities for code and data transformation -----===//
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
// This file implements miscellaneous transformation routines for non-loop IR
// structures.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Utils.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardOps.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

/// Return true if this operation dereferences one or more memref's.
// Temporary utility: will be replaced when this is modeled through
// side-effects/op traits. TODO(b/117228571)
static bool isMemRefDereferencingOp(const Operation &op) {
  if (op.is<LoadOp>() || op.is<StoreOp>() || op.is<DmaStartOp>() ||
      op.is<DmaWaitOp>())
    return true;
  return false;
}

/// Replaces all uses of oldMemRef with newMemRef while optionally remapping
/// old memref's indices to the new memref using the supplied affine map
/// and adding any additional indices. The new memref could be of a different
/// shape or rank, but of the same elemental type. Additional indices are added
/// at the start for now.
// TODO(mlir-team): extend this for SSAValue / CFGFunctions. Can also be easily
// extended to add additional indices at any position.
bool mlir::replaceAllMemRefUsesWith(MLValue *oldMemRef, MLValue *newMemRef,
                                    ArrayRef<SSAValue *> extraIndices,
                                    AffineMap *indexRemap) {
  unsigned newMemRefRank = cast<MemRefType>(newMemRef->getType())->getRank();
  (void)newMemRefRank; // unused in opt mode
  unsigned oldMemRefRank = cast<MemRefType>(oldMemRef->getType())->getRank();
  (void)newMemRefRank;
  if (indexRemap) {
    assert(indexRemap->getNumInputs() == oldMemRefRank);
    assert(indexRemap->getNumResults() + extraIndices.size() == newMemRefRank);
  } else {
    assert(oldMemRefRank + extraIndices.size() == newMemRefRank);
  }

  // Assert same elemental type.
  assert(cast<MemRefType>(oldMemRef->getType())->getElementType() ==
         cast<MemRefType>(newMemRef->getType())->getElementType());

  // Check if memref was used in a non-deferencing context.
  for (const StmtOperand &use : oldMemRef->getUses()) {
    auto *opStmt = cast<OperationStmt>(use.getOwner());
    // Failure: memref used in a non-deferencing op (potentially escapes); no
    // replacement in these cases.
    if (!isMemRefDereferencingOp(*opStmt))
      return false;
  }

  // Walk all uses of old memref. Statement using the memref gets replaced.
  for (auto it = oldMemRef->use_begin(); it != oldMemRef->use_end();) {
    StmtOperand &use = *(it++);
    auto *opStmt = cast<OperationStmt>(use.getOwner());
    assert(isMemRefDereferencingOp(*opStmt) &&
           "memref deferencing op expected");

    auto getMemRefOperandPos = [&]() -> unsigned {
      unsigned i;
      for (i = 0; i < opStmt->getNumOperands(); i++) {
        if (opStmt->getOperand(i) == oldMemRef)
          break;
      }
      assert(i < opStmt->getNumOperands() && "operand guaranteed to be found");
      return i;
    };
    unsigned memRefOperandPos = getMemRefOperandPos();

    // Construct the new operation statement using this memref.
    SmallVector<MLValue *, 8> operands;
    operands.reserve(opStmt->getNumOperands() + extraIndices.size());
    // Insert the non-memref operands.
    operands.insert(operands.end(), opStmt->operand_begin(),
                    opStmt->operand_begin() + memRefOperandPos);
    operands.push_back(newMemRef);

    MLFuncBuilder builder(opStmt);
    // Normally, we could just use extraIndices as operands, but we will
    // clone it so that each op gets its own "private" index. See b/117159533.
    for (auto *extraIndex : extraIndices) {
      OperationStmt::OperandMapTy operandMap;
      // TODO(mlir-team): An operation/SSA value should provide a method to
      // return the position of an SSA result in its defining
      // operation.
      assert(extraIndex->getDefiningStmt()->getNumResults() == 1 &&
             "single result op's expected to generate these indices");
      // TODO: actually check if this is a result of an affine_apply op.
      assert((cast<MLValue>(extraIndex)->isValidDim() ||
              cast<MLValue>(extraIndex)->isValidSymbol()) &&
             "invalid memory op index");
      auto *clonedExtraIndex =
          cast<OperationStmt>(
              builder.clone(*extraIndex->getDefiningStmt(), operandMap))
              ->getResult(0);
      operands.push_back(cast<MLValue>(clonedExtraIndex));
    }

    // Construct new indices. The indices of a memref come right after it, i.e.,
    // at position memRefOperandPos + 1.
    SmallVector<SSAValue *, 4> indices(
        opStmt->operand_begin() + memRefOperandPos + 1,
        opStmt->operand_begin() + memRefOperandPos + 1 + oldMemRefRank);
    if (indexRemap) {
      auto remapOp =
          builder.create<AffineApplyOp>(opStmt->getLoc(), indexRemap, indices);
      // Remapped indices.
      for (auto *index : remapOp->getOperation()->getResults())
        operands.push_back(cast<MLValue>(index));
    } else {
      // No remapping specified.
      for (auto *index : indices)
        operands.push_back(cast<MLValue>(index));
    }

    // Insert the remaining operands unmodified.
    operands.insert(operands.end(),
                    opStmt->operand_begin() + memRefOperandPos + 1 +
                        oldMemRefRank,
                    opStmt->operand_end());

    // Result types don't change. Both memref's are of the same elemental type.
    SmallVector<Type *, 8> resultTypes;
    resultTypes.reserve(opStmt->getNumResults());
    for (const auto *result : opStmt->getResults())
      resultTypes.push_back(result->getType());

    // Create the new operation.
    auto *repOp =
        builder.createOperation(opStmt->getLoc(), opStmt->getName(), operands,
                                resultTypes, opStmt->getAttrs());
    // Replace old memref's deferencing op's uses.
    unsigned r = 0;
    for (auto *res : opStmt->getResults()) {
      res->replaceAllUsesWith(repOp->getResult(r++));
    }
    opStmt->eraseFromBlock();
  }
  return true;
}
