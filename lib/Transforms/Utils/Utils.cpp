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

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/DenseMap.h"
using namespace mlir;

/// Return true if this operation dereferences one or more memref's.
// Temporary utility: will be replaced when this is modeled through
// side-effects/op traits. TODO(b/117228571)
static bool isMemRefDereferencingOp(const OperationInst &op) {
  if (op.isa<LoadOp>() || op.isa<StoreOp>() || op.isa<DmaStartOp>() ||
      op.isa<DmaWaitOp>())
    return true;
  return false;
}

/// Replaces all uses of oldMemRef with newMemRef while optionally remapping
/// old memref's indices to the new memref using the supplied affine map
/// and adding any additional indices. The new memref could be of a different
/// shape or rank, but of the same elemental type. Additional indices are added
/// at the start. 'extraOperands' is another optional argument that corresponds
/// to additional operands (inputs) for indexRemap at the beginning of its input
/// list. An optional argument 'domOpFilter' restricts the replacement to only
/// those operations that are dominated by the former. The replacement succeeds
/// and returns true if all uses of the memref in the region where the
/// replacement is asked for are "dereferencing" memref uses.
//  Ex: to replace load %A[%i, %j] with load %Abuf[%t mod 2, %ii - %i, %j]:
//  The SSA value corresponding to '%t mod 2' should be in 'extraIndices', and
//  index remap will (%i, %j) -> (%ii - %i, %j), i.e., (d0, d1, d2) -> (d0 - d1,
//  d2) will be the 'indexRemap', and %ii is the extra operand. Without any
//  extra operands, note that 'indexRemap' would just be applied to the existing
//  indices (%i, %j).
//
// TODO(mlir-team): extend this for CFG Functions. Can also be easily
// extended to add additional indices at any position.
bool mlir::replaceAllMemRefUsesWith(const Value *oldMemRef, Value *newMemRef,
                                    ArrayRef<Value *> extraIndices,
                                    AffineMap indexRemap,
                                    ArrayRef<Value *> extraOperands,
                                    const Statement *domStmtFilter) {
  unsigned newMemRefRank = newMemRef->getType().cast<MemRefType>().getRank();
  (void)newMemRefRank; // unused in opt mode
  unsigned oldMemRefRank = oldMemRef->getType().cast<MemRefType>().getRank();
  (void)newMemRefRank;
  if (indexRemap) {
    assert(indexRemap.getNumInputs() == extraOperands.size() + oldMemRefRank);
    assert(indexRemap.getNumResults() + extraIndices.size() == newMemRefRank);
  } else {
    assert(oldMemRefRank + extraIndices.size() == newMemRefRank);
  }

  // Assert same elemental type.
  assert(oldMemRef->getType().cast<MemRefType>().getElementType() ==
         newMemRef->getType().cast<MemRefType>().getElementType());

  // Walk all uses of old memref. Operation using the memref gets replaced.
  for (auto it = oldMemRef->use_begin(); it != oldMemRef->use_end();) {
    StmtOperand &use = *(it++);
    auto *opStmt = cast<OperationInst>(use.getOwner());

    // Skip this use if it's not dominated by domStmtFilter.
    if (domStmtFilter && !dominates(*domStmtFilter, *opStmt))
      continue;

    // Check if the memref was used in a non-deferencing context. It is fine for
    // the memref to be used in a non-deferencing way outside of the region
    // where this replacement is happening.
    if (!isMemRefDereferencingOp(*opStmt))
      // Failure: memref used in a non-deferencing op (potentially escapes); no
      // replacement in these cases.
      return false;

    auto getMemRefOperandPos = [&]() -> unsigned {
      unsigned i, e;
      for (i = 0, e = opStmt->getNumOperands(); i < e; i++) {
        if (opStmt->getOperand(i) == oldMemRef)
          break;
      }
      assert(i < opStmt->getNumOperands() && "operand guaranteed to be found");
      return i;
    };
    unsigned memRefOperandPos = getMemRefOperandPos();

    // Construct the new operation statement using this memref.
    OperationState state(opStmt->getContext(), opStmt->getLoc(),
                         opStmt->getName());
    state.operands.reserve(opStmt->getNumOperands() + extraIndices.size());
    // Insert the non-memref operands.
    state.operands.insert(state.operands.end(), opStmt->operand_begin(),
                          opStmt->operand_begin() + memRefOperandPos);
    state.operands.push_back(newMemRef);

    FuncBuilder builder(opStmt);
    for (auto *extraIndex : extraIndices) {
      // TODO(mlir-team): An operation/SSA value should provide a method to
      // return the position of an SSA result in its defining
      // operation.
      assert(extraIndex->getDefiningInst()->getNumResults() == 1 &&
             "single result op's expected to generate these indices");
      assert((extraIndex->isValidDim() || extraIndex->isValidSymbol()) &&
             "invalid memory op index");
      state.operands.push_back(extraIndex);
    }

    // Construct new indices as a remap of the old ones if a remapping has been
    // provided. The indices of a memref come right after it, i.e.,
    // at position memRefOperandPos + 1.
    SmallVector<Value *, 4> remapOperands;
    remapOperands.reserve(oldMemRefRank + extraOperands.size());
    remapOperands.insert(remapOperands.end(), extraOperands.begin(),
                         extraOperands.end());
    remapOperands.insert(
        remapOperands.end(), opStmt->operand_begin() + memRefOperandPos + 1,
        opStmt->operand_begin() + memRefOperandPos + 1 + oldMemRefRank);
    if (indexRemap) {
      auto remapOp = builder.create<AffineApplyOp>(opStmt->getLoc(), indexRemap,
                                                   remapOperands);
      // Remapped indices.
      for (auto *index : remapOp->getOperation()->getResults())
        state.operands.push_back(index);
    } else {
      // No remapping specified.
      for (auto *index : remapOperands)
        state.operands.push_back(index);
    }

    // Insert the remaining operands unmodified.
    state.operands.insert(state.operands.end(),
                          opStmt->operand_begin() + memRefOperandPos + 1 +
                              oldMemRefRank,
                          opStmt->operand_end());

    // Result types don't change. Both memref's are of the same elemental type.
    state.types.reserve(opStmt->getNumResults());
    for (const auto *result : opStmt->getResults())
      state.types.push_back(result->getType());

    // Attributes also do not change.
    state.attributes.insert(state.attributes.end(), opStmt->getAttrs().begin(),
                            opStmt->getAttrs().end());

    // Create the new operation.
    auto *repOp = builder.createOperation(state);
    // Replace old memref's deferencing op's uses.
    unsigned r = 0;
    for (auto *res : opStmt->getResults()) {
      res->replaceAllUsesWith(repOp->getResult(r++));
    }
    opStmt->erase();
  }
  return true;
}

// Creates and inserts into 'builder' a new AffineApplyOp, with the number of
// its results equal to the number of 'operands, as a composition
// of all other AffineApplyOps reachable from input parameter 'operands'. If the
// operands were drawing results from multiple affine apply ops, this also leads
// to a collapse into a single affine apply op. The final results of the
// composed AffineApplyOp are returned in output parameter 'results'.
OperationInst *
mlir::createComposedAffineApplyOp(FuncBuilder *builder, Location loc,
                                  ArrayRef<Value *> operands,
                                  ArrayRef<OperationInst *> affineApplyOps,
                                  SmallVectorImpl<Value *> *results) {
  // Create identity map with same number of dimensions as number of operands.
  auto map = builder->getMultiDimIdentityMap(operands.size());
  // Initialize AffineValueMap with identity map.
  AffineValueMap valueMap(map, operands);

  for (auto *opStmt : affineApplyOps) {
    assert(opStmt->isa<AffineApplyOp>());
    auto affineApplyOp = opStmt->cast<AffineApplyOp>();
    // Forward substitute 'affineApplyOp' into 'valueMap'.
    valueMap.forwardSubstitute(*affineApplyOp);
  }
  // Compose affine maps from all ancestor AffineApplyOps.
  // Create new AffineApplyOp from 'valueMap'.
  unsigned numOperands = valueMap.getNumOperands();
  SmallVector<Value *, 4> outOperands(numOperands);
  for (unsigned i = 0; i < numOperands; ++i) {
    outOperands[i] = valueMap.getOperand(i);
  }
  // Create new AffineApplyOp based on 'valueMap'.
  auto affineApplyOp =
      builder->create<AffineApplyOp>(loc, valueMap.getAffineMap(), outOperands);
  results->resize(operands.size());
  for (unsigned i = 0, e = operands.size(); i < e; ++i) {
    (*results)[i] = affineApplyOp->getResult(i);
  }
  return cast<OperationInst>(affineApplyOp->getOperation());
}

/// Given an operation statement, inserts a new single affine apply operation,
/// that is exclusively used by this operation statement, and that provides all
/// operands that are results of an affine_apply as a function of loop iterators
/// and program parameters and whose results are.
///
/// Before
///
/// for %i = 0 to #map(%N)
///   %idx = affine_apply (d0) -> (d0 mod 2) (%i)
///   "send"(%idx, %A, ...)
///   "compute"(%idx)
///
/// After
///
/// for %i = 0 to #map(%N)
///   %idx = affine_apply (d0) -> (d0 mod 2) (%i)
///   "send"(%idx, %A, ...)
///   %idx_ = affine_apply (d0) -> (d0 mod 2) (%i)
///   "compute"(%idx_)
///
/// This allows applying different transformations on send and compute (for eg.
/// different shifts/delays).
///
/// Returns nullptr either if none of opStmt's operands were the result of an
/// affine_apply and thus there was no affine computation slice to create, or if
/// all the affine_apply op's supplying operands to this opStmt do not have any
/// uses besides this opStmt. Returns the new affine_apply operation statement
/// otherwise.
OperationInst *mlir::createAffineComputationSlice(OperationInst *opStmt) {
  // Collect all operands that are results of affine apply ops.
  SmallVector<Value *, 4> subOperands;
  subOperands.reserve(opStmt->getNumOperands());
  for (auto *operand : opStmt->getOperands()) {
    auto *defStmt = operand->getDefiningInst();
    if (defStmt && defStmt->isa<AffineApplyOp>()) {
      subOperands.push_back(operand);
    }
  }

  // Gather sequence of AffineApplyOps reachable from 'subOperands'.
  SmallVector<OperationInst *, 4> affineApplyOps;
  getReachableAffineApplyOps(subOperands, affineApplyOps);
  // Skip transforming if there are no affine maps to compose.
  if (affineApplyOps.empty())
    return nullptr;

  // Check if all uses of the affine apply op's lie only in this op stmt, in
  // which case there would be nothing to do.
  bool localized = true;
  for (auto *op : affineApplyOps) {
    for (auto *result : op->getResults()) {
      for (auto &use : result->getUses()) {
        if (use.getOwner() != opStmt) {
          localized = false;
          break;
        }
      }
    }
  }
  if (localized)
    return nullptr;

  FuncBuilder builder(opStmt);
  SmallVector<Value *, 4> results;
  auto *affineApplyStmt = createComposedAffineApplyOp(
      &builder, opStmt->getLoc(), subOperands, affineApplyOps, &results);
  assert(results.size() == subOperands.size() &&
         "number of results should be the same as the number of subOperands");

  // Construct the new operands that include the results from the composed
  // affine apply op above instead of existing ones (subOperands). So, they
  // differ from opStmt's operands only for those operands in 'subOperands', for
  // which they will be replaced by the corresponding one from 'results'.
  SmallVector<Value *, 4> newOperands(opStmt->getOperands());
  for (unsigned i = 0, e = newOperands.size(); i < e; i++) {
    // Replace the subOperands from among the new operands.
    unsigned j, f;
    for (j = 0, f = subOperands.size(); j < f; j++) {
      if (newOperands[i] == subOperands[j])
        break;
    }
    if (j < subOperands.size()) {
      newOperands[i] = results[j];
    }
  }

  for (unsigned idx = 0, e = newOperands.size(); idx < e; idx++) {
    opStmt->setOperand(idx, newOperands[idx]);
  }

  return affineApplyStmt;
}

void mlir::forwardSubstitute(OpPointer<AffineApplyOp> affineApplyOp) {
  if (!affineApplyOp->getOperation()->getFunction()->isML()) {
    // TODO: Support forward substitution for CFG style functions.
    return;
  }
  auto *opStmt = cast<OperationInst>(affineApplyOp->getOperation());
  // Iterate through all uses of all results of 'opStmt', forward substituting
  // into any uses which are AffineApplyOps.
  for (unsigned resultIndex = 0, e = opStmt->getNumResults(); resultIndex < e;
       ++resultIndex) {
    const Value *result = opStmt->getResult(resultIndex);
    for (auto it = result->use_begin(); it != result->use_end();) {
      StmtOperand &use = *(it++);
      auto *useStmt = use.getOwner();
      auto *useOpStmt = dyn_cast<OperationInst>(useStmt);
      // Skip if use is not AffineApplyOp.
      if (useOpStmt == nullptr || !useOpStmt->isa<AffineApplyOp>())
        continue;
      // Advance iterator past 'opStmt' operands which also use 'result'.
      while (it != result->use_end() && it->getOwner() == useStmt)
        ++it;

      FuncBuilder builder(useOpStmt);
      // Initialize AffineValueMap with 'affineApplyOp' which uses 'result'.
      auto oldAffineApplyOp = useOpStmt->cast<AffineApplyOp>();
      AffineValueMap valueMap(*oldAffineApplyOp);
      // Forward substitute 'result' at index 'i' into 'valueMap'.
      valueMap.forwardSubstituteSingle(*affineApplyOp, resultIndex);

      // Create new AffineApplyOp from 'valueMap'.
      unsigned numOperands = valueMap.getNumOperands();
      SmallVector<Value *, 4> operands(numOperands);
      for (unsigned i = 0; i < numOperands; ++i) {
        operands[i] = valueMap.getOperand(i);
      }
      auto newAffineApplyOp = builder.create<AffineApplyOp>(
          useOpStmt->getLoc(), valueMap.getAffineMap(), operands);

      // Update all uses to use results from 'newAffineApplyOp'.
      for (unsigned i = 0, e = useOpStmt->getNumResults(); i < e; ++i) {
        oldAffineApplyOp->getResult(i)->replaceAllUsesWith(
            newAffineApplyOp->getResult(i));
      }
      // Erase 'oldAffineApplyOp'.
      oldAffineApplyOp->getOperation()->erase();
    }
  }
}

/// Folds the specified (lower or upper) bound to a constant if possible
/// considering its operands. Returns false if the folding happens for any of
/// the bounds, true otherwise.
bool mlir::constantFoldBounds(ForStmt *forStmt) {
  auto foldLowerOrUpperBound = [forStmt](bool lower) {
    // Check if the bound is already a constant.
    if (lower && forStmt->hasConstantLowerBound())
      return true;
    if (!lower && forStmt->hasConstantUpperBound())
      return true;

    // Check to see if each of the operands is the result of a constant.  If so,
    // get the value.  If not, ignore it.
    SmallVector<Attribute, 8> operandConstants;
    auto boundOperands = lower ? forStmt->getLowerBoundOperands()
                               : forStmt->getUpperBoundOperands();
    for (const auto *operand : boundOperands) {
      Attribute operandCst;
      if (auto *operandOp = operand->getDefiningInst()) {
        if (auto operandConstantOp = operandOp->dyn_cast<ConstantOp>())
          operandCst = operandConstantOp->getValue();
      }
      operandConstants.push_back(operandCst);
    }

    AffineMap boundMap =
        lower ? forStmt->getLowerBoundMap() : forStmt->getUpperBoundMap();
    assert(boundMap.getNumResults() >= 1 &&
           "bound maps should have at least one result");
    SmallVector<Attribute, 4> foldedResults;
    if (boundMap.constantFold(operandConstants, foldedResults))
      return true;

    // Compute the max or min as applicable over the results.
    assert(!foldedResults.empty() && "bounds should have at least one result");
    auto maxOrMin = foldedResults[0].cast<IntegerAttr>().getValue();
    for (unsigned i = 1, e = foldedResults.size(); i < e; i++) {
      auto foldedResult = foldedResults[i].cast<IntegerAttr>().getValue();
      maxOrMin = lower ? llvm::APIntOps::smax(maxOrMin, foldedResult)
                       : llvm::APIntOps::smin(maxOrMin, foldedResult);
    }
    lower ? forStmt->setConstantLowerBound(maxOrMin.getSExtValue())
          : forStmt->setConstantUpperBound(maxOrMin.getSExtValue());

    // Return false on success.
    return false;
  };

  bool ret = foldLowerOrUpperBound(/*lower=*/true);
  ret &= foldLowerOrUpperBound(/*lower=*/false);
  return ret;
}

void mlir::remapFunctionAttrs(
    OperationInst &op,
    const DenseMap<Attribute, FunctionAttr> &remappingTable) {
  for (auto attr : op.getAttrs()) {
    // Do the remapping, if we got the same thing back, then it must contain
    // functions that aren't getting remapped.
    auto newVal =
        attr.second.remapFunctionAttrs(remappingTable, op.getContext());
    if (newVal == attr.second)
      continue;

    // Otherwise, replace the existing attribute with the new one.  It is safe
    // to mutate the attribute list while we walk it because underlying
    // attribute lists are uniqued and immortal.
    op.setAttr(attr.first, newVal);
  }
}

void mlir::remapFunctionAttrs(
    Function &fn, const DenseMap<Attribute, FunctionAttr> &remappingTable) {
  // Look at all instructions in a CFGFunction.
  if (fn.isCFG()) {
    for (auto &bb : fn.getBlockList()) {
      for (auto &inst : bb) {
        if (auto *op = dyn_cast<OperationInst>(&inst))
          remapFunctionAttrs(*op, remappingTable);
      }
    }
    return;
  }

  // Otherwise, look at MLFunctions.  We ignore external functions.
  if (!fn.isML())
    return;

  struct MLFnWalker : public StmtWalker<MLFnWalker> {
    MLFnWalker(const DenseMap<Attribute, FunctionAttr> &remappingTable)
        : remappingTable(remappingTable) {}
    void visitOperationInst(OperationInst *opStmt) {
      remapFunctionAttrs(*opStmt, remappingTable);
    }

    const DenseMap<Attribute, FunctionAttr> &remappingTable;
  };

  MLFnWalker(remappingTable).walk(&fn);
}

void mlir::remapFunctionAttrs(
    Module &module, const DenseMap<Attribute, FunctionAttr> &remappingTable) {
  for (auto &fn : module) {
    remapFunctionAttrs(fn, remappingTable);
  }
}
