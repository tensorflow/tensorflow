//===- FoldUtils.cpp ---- Fold Utilities ----------------------------------===//
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
// This file defines various operation fold utilities. These utilities are
// intended to be used by passes to unify and simply their logic.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/FoldUtils.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

/// Given an operation, find the parent region that folded constants should be
/// inserted into.
static Region *getInsertionRegion(
    DialectInterfaceCollection<OpFolderDialectInterface> &interfaces,
    Operation *op) {
  while (Region *region = op->getParentRegion()) {
    // Insert in this region for any of the following scenarios:
    //  * The parent is unregistered, or is known to be isolated from above.
    //  * The parent is a top-level operation.
    auto *parentOp = region->getParentOp();
    if (!parentOp->isRegistered() || parentOp->isKnownIsolatedFromAbove() ||
        !parentOp->getBlock())
      return region;

    // Otherwise, check if this region is a desired insertion region.
    auto *interface = interfaces.getInterfaceFor(parentOp);
    if (LLVM_UNLIKELY(interface && interface->shouldMaterializeInto(region)))
      return region;

    // Traverse up the parent looking for an insertion region.
    op = parentOp;
  }
  llvm_unreachable("expected valid insertion region");
}

/// A utility function used to materialize a constant for a given attribute and
/// type. On success, a valid constant value is returned. Otherwise, null is
/// returned
static Operation *materializeConstant(Dialect *dialect, OpBuilder &builder,
                                      Attribute value, Type type,
                                      Location loc) {
  auto insertPt = builder.getInsertionPoint();
  (void)insertPt;

  // Ask the dialect to materialize a constant operation for this value.
  if (auto *constOp = dialect->materializeConstant(builder, value, type, loc)) {
    assert(insertPt == builder.getInsertionPoint());
    assert(matchPattern(constOp, m_Constant(&value)));
    return constOp;
  }

  // If the dialect is unable to materialize a constant, check to see if the
  // standard constant can be used.
  if (ConstantOp::isBuildableWith(value, type))
    return builder.create<ConstantOp>(loc, type, value);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// OperationFolder
//===----------------------------------------------------------------------===//

LogicalResult OperationFolder::tryToFold(
    Operation *op, function_ref<void(Operation *)> processGeneratedConstants,
    function_ref<void(Operation *)> preReplaceAction) {
  // If this is a unique'd constant, return failure as we know that it has
  // already been folded.
  if (referencedDialects.count(op))
    return failure();

  // Try to fold the operation.
  SmallVector<ValuePtr, 8> results;
  if (failed(tryToFold(op, results, processGeneratedConstants)))
    return failure();

  // Constant folding succeeded. We will start replacing this op's uses and
  // eventually erase this op. Invoke the callback provided by the caller to
  // perform any pre-replacement action.
  if (preReplaceAction)
    preReplaceAction(op);

  // Check to see if the operation was just updated in place.
  if (results.empty())
    return success();

  // Otherwise, replace all of the result values and erase the operation.
  for (unsigned i = 0, e = results.size(); i != e; ++i)
    op->getResult(i)->replaceAllUsesWith(results[i]);
  op->erase();
  return success();
}

/// Notifies that the given constant `op` should be remove from this
/// OperationFolder's internal bookkeeping.
void OperationFolder::notifyRemoval(Operation *op) {
  // Check to see if this operation is uniqued within the folder.
  auto it = referencedDialects.find(op);
  if (it == referencedDialects.end())
    return;

  // Get the constant value for this operation, this is the value that was used
  // to unique the operation internally.
  Attribute constValue;
  matchPattern(op, m_Constant(&constValue));
  assert(constValue);

  // Get the constant map that this operation was uniqued in.
  auto &uniquedConstants = foldScopes[getInsertionRegion(interfaces, op)];

  // Erase all of the references to this operation.
  auto type = op->getResult(0)->getType();
  for (auto *dialect : it->second)
    uniquedConstants.erase(std::make_tuple(dialect, constValue, type));
  referencedDialects.erase(it);
}

/// Tries to perform folding on the given `op`. If successful, populates
/// `results` with the results of the folding.
LogicalResult OperationFolder::tryToFold(
    Operation *op, SmallVectorImpl<ValuePtr> &results,
    function_ref<void(Operation *)> processGeneratedConstants) {
  SmallVector<Attribute, 8> operandConstants;
  SmallVector<OpFoldResult, 8> foldResults;

  // Check to see if any operands to the operation is constant and whether
  // the operation knows how to constant fold itself.
  operandConstants.assign(op->getNumOperands(), Attribute());
  for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
    matchPattern(op->getOperand(i), m_Constant(&operandConstants[i]));

  // If this is a commutative binary operation with a constant on the left
  // side move it to the right side.
  if (operandConstants.size() == 2 && operandConstants[0] &&
      !operandConstants[1] && op->isCommutative()) {
    std::swap(op->getOpOperand(0), op->getOpOperand(1));
    std::swap(operandConstants[0], operandConstants[1]);
  }

  // Attempt to constant fold the operation.
  if (failed(op->fold(operandConstants, foldResults)))
    return failure();

  // Check to see if the operation was just updated in place.
  if (foldResults.empty())
    return success();
  assert(foldResults.size() == op->getNumResults());

  // Create a builder to insert new operations into the entry block of the
  // insertion region.
  auto *insertRegion = getInsertionRegion(interfaces, op);
  auto &entry = insertRegion->front();
  OpBuilder builder(&entry, entry.begin());

  // Get the constant map for the insertion region of this operation.
  auto &uniquedConstants = foldScopes[insertRegion];

  // Create the result constants and replace the results.
  auto *dialect = op->getDialect();
  for (unsigned i = 0, e = op->getNumResults(); i != e; ++i) {
    assert(!foldResults[i].isNull() && "expected valid OpFoldResult");

    // Check if the result was an SSA value.
    if (auto repl = foldResults[i].dyn_cast<ValuePtr>()) {
      results.emplace_back(repl);
      continue;
    }

    // Check to see if there is a canonicalized version of this constant.
    auto res = op->getResult(i);
    Attribute attrRepl = foldResults[i].get<Attribute>();
    if (auto *constOp =
            tryGetOrCreateConstant(uniquedConstants, dialect, builder, attrRepl,
                                   res->getType(), op->getLoc())) {
      results.push_back(constOp->getResult(0));
      continue;
    }
    // If materialization fails, cleanup any operations generated for the
    // previous results and return failure.
    for (Operation &op : llvm::make_early_inc_range(
             llvm::make_range(entry.begin(), builder.getInsertionPoint()))) {
      notifyRemoval(&op);
      op.erase();
    }
    return failure();
  }

  // Process any newly generated operations.
  if (processGeneratedConstants) {
    for (auto i = entry.begin(), e = builder.getInsertionPoint(); i != e; ++i)
      processGeneratedConstants(&*i);
  }

  return success();
}

/// Try to get or create a new constant entry. On success this returns the
/// constant operation value, nullptr otherwise.
Operation *OperationFolder::tryGetOrCreateConstant(
    ConstantMap &uniquedConstants, Dialect *dialect, OpBuilder &builder,
    Attribute value, Type type, Location loc) {
  // Check if an existing mapping already exists.
  auto constKey = std::make_tuple(dialect, value, type);
  auto *&constInst = uniquedConstants[constKey];
  if (constInst)
    return constInst;

  // If one doesn't exist, try to materialize one.
  if (!(constInst = materializeConstant(dialect, builder, value, type, loc)))
    return nullptr;

  // Check to see if the generated constant is in the expected dialect.
  auto *newDialect = constInst->getDialect();
  if (newDialect == dialect) {
    referencedDialects[constInst].push_back(dialect);
    return constInst;
  }

  // If it isn't, then we also need to make sure that the mapping for the new
  // dialect is valid.
  auto newKey = std::make_tuple(newDialect, value, type);

  // If an existing operation in the new dialect already exists, delete the
  // materialized operation in favor of the existing one.
  if (auto *existingOp = uniquedConstants.lookup(newKey)) {
    constInst->erase();
    referencedDialects[existingOp].push_back(dialect);
    return constInst = existingOp;
  }

  // Otherwise, update the new dialect to the materialized operation.
  referencedDialects[constInst].assign({dialect, newDialect});
  auto newIt = uniquedConstants.insert({newKey, constInst});
  return newIt.first->second;
}
