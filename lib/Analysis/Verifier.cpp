//===- Verifier.cpp - MLIR Verifier Implementation ------------------------===//
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
// This file implements the verify() methods on the various IR types, performing
// (potentially expensive) checks on the holistic structure of the code.  This
// can be used for detecting bugs in compiler transformations and hand written
// .mlir files.
//
// The checks in this file are only for things that can occur as part of IR
// transformations: e.g. violation of dominance information, malformed operation
// attributes, etc.  MLIR supports transformations moving IR through locally
// invalid states (e.g. unlinking an operation from a block before re-inserting
// it in a new place), but each transformation must complete with the IR in a
// valid form.
//
// This should not check for things that are always wrong by construction (e.g.
// attributes or other immutable structures that are incorrect), because those
// are not mutable and can be checked at time of construction.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Dominance.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;

namespace {
/// This class encapsulates all the state used to verify an operation region.
class OperationVerifier {
public:
  explicit OperationVerifier(MLIRContext *ctx)
      : ctx(ctx), identifierRegex("^[a-zA-Z_][a-zA-Z_0-9\\.\\$]*$") {}

  /// Verify the body of the given function.
  LogicalResult verify(Function &fn);

  /// Verify the given operation.
  LogicalResult verify(Operation &op);

  /// Returns the registered dialect for a dialect-specific attribute.
  Dialect *getDialectForAttribute(const NamedAttribute &attr) {
    assert(attr.first.strref().contains('.') && "expected dialect attribute");
    auto dialectNamePair = attr.first.strref().split('.');
    return ctx->getRegisteredDialect(dialectNamePair.first);
  }

  /// Returns if the given string is valid to use as an identifier name.
  bool isValidName(StringRef name) { return identifierRegex.match(name); }

private:
  /// Verify the given potentially nested region or block.
  LogicalResult verifyRegion(Region &region);
  LogicalResult verifyBlock(Block &block);
  LogicalResult verifyOperation(Operation &op);

  /// Verify the dominance within the given IR unit.
  LogicalResult verifyDominance(Region &region);
  LogicalResult verifyDominance(Operation &op);

  /// Emit an error for the given block.
  InFlightDiagnostic emitError(Block &bb, const Twine &message) {
    // Take the location information for the first operation in the block.
    if (!bb.empty())
      return bb.front().emitError(message);

    // Worst case, fall back to using the parent's location.
    return mlir::emitError(bb.getParent()->getLoc(), message);
  }

  /// The current context for the verifier.
  MLIRContext *ctx;

  /// Dominance information for this function, when checking dominance.
  DominanceInfo *domInfo = nullptr;

  /// Regex checker for attribute names.
  llvm::Regex identifierRegex;

  /// Mapping between dialect namespace and if that dialect supports
  /// unregistered operations.
  llvm::StringMap<bool> dialectAllowsUnknownOps;
};
} // end anonymous namespace

/// Verify the body of the given function.
LogicalResult OperationVerifier::verify(Function &fn) {
  // Verify the body first.
  if (failed(verifyRegion(fn.getBody())))
    return failure();

  // Since everything looks structurally ok to this point, we do a dominance
  // check.  We do this as a second pass since malformed CFG's can cause
  // dominator analysis constructure to crash and we want the verifier to be
  // resilient to malformed code.
  DominanceInfo theDomInfo(&fn);
  domInfo = &theDomInfo;
  if (failed(verifyDominance(fn.getBody())))
    return failure();

  domInfo = nullptr;
  return success();
}

/// Verify the given operation.
LogicalResult OperationVerifier::verify(Operation &op) {
  // Verify the operation first.
  if (failed(verifyOperation(op)))
    return failure();

  // Since everything looks structurally ok to this point, we do a dominance
  // check for any nested regions. We do this as a second pass since malformed
  // CFG's can cause dominator analysis constructure to crash and we want the
  // verifier to be resilient to malformed code.
  DominanceInfo theDomInfo(&op);
  domInfo = &theDomInfo;
  for (auto &region : op.getRegions())
    if (failed(verifyDominance(region)))
      return failure();

  domInfo = nullptr;
  return success();
}

LogicalResult OperationVerifier::verifyRegion(Region &region) {
  if (region.empty())
    return success();

  // Verify the first block has no predecessors.
  auto *firstBB = &region.front();
  if (!firstBB->hasNoPredecessors())
    return mlir::emitError(region.getLoc(),
                           "entry block of region may not have predecessors");

  // Verify each of the blocks within the region.
  for (auto &block : region)
    if (failed(verifyBlock(block)))
      return failure();
  return success();
}

LogicalResult OperationVerifier::verifyBlock(Block &block) {
  for (auto *arg : block.getArguments())
    if (arg->getOwner() != &block)
      return emitError(block, "block argument not owned by block");

  // Verify that this block has a terminator.
  if (block.empty())
    return emitError(block, "block with no terminator");

  // Verify the non-terminator operations separately so that we can verify
  // they has no successors.
  for (auto &op : llvm::make_range(block.begin(), std::prev(block.end()))) {
    if (op.getNumSuccessors() != 0)
      return op.emitError(
          "operation with block successors must terminate its parent block");

    if (failed(verifyOperation(op)))
      return failure();
  }

  // Verify the terminator.
  if (failed(verifyOperation(block.back())))
    return failure();
  if (block.back().isKnownNonTerminator())
    return emitError(block, "block with no terminator");

  // Verify that this block is not branching to a block of a different
  // region.
  for (Block *successor : block.getSuccessors())
    if (successor->getParent() != block.getParent())
      return block.back().emitOpError(
          "branching to block of a different region");

  return success();
}

LogicalResult OperationVerifier::verifyOperation(Operation &op) {
  // Check that operands are non-nil and structurally ok.
  for (auto *operand : op.getOperands())
    if (!operand)
      return op.emitError("null operand found");

  /// Verify that all of the attributes are okay.
  for (auto attr : op.getAttrs()) {
    if (!identifierRegex.match(attr.first))
      return op.emitError("invalid attribute name '") << attr.first << "'";

    // Check for any optional dialect specific attributes.
    if (!attr.first.strref().contains('.'))
      continue;
    if (auto *dialect = getDialectForAttribute(attr))
      if (failed(dialect->verifyOperationAttribute(&op, attr)))
        return failure();
  }

  // If we can get operation info for this, check the custom hook.
  auto *opInfo = op.getAbstractOperation();
  if (opInfo && failed(opInfo->verifyInvariants(&op)))
    return failure();

  // Verify that all child regions are ok.
  for (auto &region : op.getRegions())
    if (failed(verifyRegion(region)))
      return failure();

  // If this is a registered operation, there is nothing left to do.
  if (opInfo)
    return success();

  // Otherwise, verify that the parent dialect allows un-registered operations.
  auto dialectPrefix = op.getName().getDialect();

  // Check for an existing answer for the operation dialect.
  auto it = dialectAllowsUnknownOps.find(dialectPrefix);
  if (it == dialectAllowsUnknownOps.end()) {
    // If the operation dialect is registered, query it directly.
    if (auto *dialect = ctx->getRegisteredDialect(dialectPrefix))
      it = dialectAllowsUnknownOps
               .try_emplace(dialectPrefix, dialect->allowsUnknownOperations())
               .first;
    // Otherwise, conservatively allow unknown operations.
    else
      it = dialectAllowsUnknownOps.try_emplace(dialectPrefix, true).first;
  }

  if (!it->second) {
    return op.emitError("unregistered operation '")
           << op.getName() << "' found in dialect ('" << dialectPrefix
           << "') that does not allow unknown operations";
  }

  return success();
}

LogicalResult OperationVerifier::verifyDominance(Region &region) {
  // Verify the dominance of each of the held operations.
  for (auto &block : region)
    for (auto &op : block)
      if (failed(verifyDominance(op)))
        return failure();
  return success();
}

LogicalResult OperationVerifier::verifyDominance(Operation &op) {
  // Check that operands properly dominate this use.
  for (unsigned operandNo = 0, e = op.getNumOperands(); operandNo != e;
       ++operandNo) {
    auto *operand = op.getOperand(operandNo);
    if (domInfo->properlyDominates(operand, &op))
      continue;

    auto diag = op.emitError("operand #")
                << operandNo << " does not dominate this use";
    if (auto *useOp = operand->getDefiningOp())
      diag.attachNote(useOp->getLoc()) << "operand defined here";
    return failure();
  }

  // Verify the dominance of each of the nested blocks within this operation.
  for (auto &region : op.getRegions())
    if (failed(verifyDominance(region)))
      return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Entrypoints
//===----------------------------------------------------------------------===//

/// Perform (potentially expensive) checks of invariants, used to detect
/// compiler bugs.  On error, this reports the error through the MLIRContext and
/// returns failure.
LogicalResult Function::verify() {
  OperationVerifier opVerifier(getContext());
  llvm::PrettyStackTraceFormat fmt("MLIR Verifier: func @%s",
                                   getName().c_str());

  // Check that the function name is valid.
  if (!opVerifier.isValidName(getName().strref()))
    return emitError("invalid function name '") << getName() << "'";

  /// Verify that all of the attributes are okay.
  for (auto attr : getAttrs()) {
    if (!opVerifier.isValidName(attr.first))
      return emitError("invalid attribute name '") << attr.first << "'";

    /// Check that the attribute is a dialect attribute, i.e. contains a '.' for
    /// the namespace.
    if (!attr.first.strref().contains('.'))
      return emitError("functions may only have dialect attributes");

    // Verify this attribute with the defining dialect.
    if (auto *dialect = opVerifier.getDialectForAttribute(attr))
      if (failed(dialect->verifyFunctionAttribute(this, attr)))
        return failure();
  }

  /// Verify that all of the argument attributes are okay.
  for (unsigned i = 0, e = getNumArguments(); i != e; ++i) {
    for (auto attr : getArgAttrs(i)) {
      if (!opVerifier.isValidName(attr.first))
        return emitError("invalid attribute name '")
               << attr.first << "' on argument " << i;

      /// Check that the attribute is a dialect attribute, i.e. contains a '.'
      /// for the namespace.
      if (!attr.first.strref().contains('.'))
        return emitError("function arguments may only have dialect attributes");

      // Verify this attribute with the defining dialect.
      if (auto *dialect = opVerifier.getDialectForAttribute(attr))
        if (failed(dialect->verifyFunctionArgAttribute(this, i, attr)))
          return failure();
    }
  }

  // External functions have nothing more to check.
  if (isExternal())
    return success();

  // Verify that the argument list of the function and the arg list of the first
  // block line up.
  auto *firstBB = &front();
  auto fnInputTypes = getType().getInputs();
  if (fnInputTypes.size() != firstBB->getNumArguments())
    return emitError("first block of function must have ")
           << fnInputTypes.size() << " arguments to match function signature";
  for (unsigned i = 0, e = firstBB->getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != firstBB->getArgument(i)->getType())
      return emitError("type of argument #")
             << i << " must match corresponding argument in function signature";

  // Finally, verify the body of the function.
  return opVerifier.verify(*this);
}

/// Perform (potentially expensive) checks of invariants, used to detect
/// compiler bugs.  On error, this reports the error through the MLIRContext and
/// returns failure.
LogicalResult Operation::verify() {
  return OperationVerifier(getContext()).verify(*this);
}

/// Perform (potentially expensive) checks of invariants, used to detect
/// compiler bugs.  On error, this reports the error through the MLIRContext and
/// returns failure.
LogicalResult Module::verify() {
  /// Check that each function is correct.
  for (auto &fn : *this)
    if (failed(fn.verify()))
      return failure();

  return success();
}
