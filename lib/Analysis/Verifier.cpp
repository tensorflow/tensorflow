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
// invalid states (e.g. unlinking an instruction from an instruction before
// re-inserting it in a new place), but each transformation must complete with
// the IR in a valid form.
//
// This should not check for things that are always wrong by construction (e.g.
// affine maps or other immutable structures that are incorrect), because those
// are not mutable and can be checked at time of construction.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Dominance.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Instruction.h"
#include "mlir/IR/Module.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;

namespace {
/// This class encapsulates all the state used to verify a function body.  It is
/// a pervasive truth that this file treats "true" as an error that needs to be
/// recovered from, and "false" as success.
///
class FuncVerifier {
public:
  bool failure(const Twine &message, const Instruction &value) {
    return value.emitError(message);
  }

  bool failure(const Twine &message, const Function &fn) {
    return fn.emitError(message);
  }

  bool failure(const Twine &message, const Block &bb) {
    // Take the location information for the first instruction in the block.
    if (!bb.empty())
      return failure(message, bb.front());

    // Worst case, fall back to using the function's location.
    return failure(message, fn);
  }

  bool verifyAttribute(Attribute attr, const Instruction &op);

  bool verify();
  bool verifyBlock(const Block &block, bool isTopLevel);
  bool verifyOperation(const Instruction &op);
  bool verifyDominance(const Block &block);
  bool verifyInstDominance(const Instruction &inst);

  explicit FuncVerifier(const Function &fn) : fn(fn) {}

private:
  /// The function being checked.
  const Function &fn;

  /// Dominance information for this function, when checking dominance.
  DominanceInfo *domInfo = nullptr;
};
} // end anonymous namespace

bool FuncVerifier::verify() {
  llvm::PrettyStackTraceFormat fmt("MLIR Verifier: func @%s",
                                   fn.getName().c_str());

  // External functions have nothing more to check.
  if (fn.empty())
    return false;

  // Verify the first block has no predecessors.
  auto *firstBB = &fn.front();
  if (!firstBB->hasNoPredecessors())
    return failure("entry block of function may not have predecessors", fn);

  // Verify that the argument list of the function and the arg list of the first
  // block line up.
  auto fnInputTypes = fn.getType().getInputs();
  if (fnInputTypes.size() != firstBB->getNumArguments())
    return failure("first block of function must have " +
                       Twine(fnInputTypes.size()) +
                       " arguments to match function signature",
                   fn);
  for (unsigned i = 0, e = firstBB->getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != firstBB->getArgument(i)->getType())
      return failure(
          "type of argument #" + Twine(i) +
              " must match corresponding argument in function signature",
          fn);

  for (auto &block : fn) {
    if (verifyBlock(block, /*isTopLevel*/ true))
      return true;
  }

  // Since everything looks structurally ok to this point, we do a dominance
  // check.  We do this as a second pass since malformed CFG's can cause
  // dominator analysis constructure to crash and we want the verifier to be
  // resilient to malformed code.
  DominanceInfo theDomInfo(const_cast<Function *>(&fn));
  domInfo = &theDomInfo;
  for (auto &block : fn) {
    if (verifyDominance(block))
      return true;
  }

  domInfo = nullptr;
  return false;
}

// Check that function attributes are all well formed.
bool FuncVerifier::verifyAttribute(Attribute attr, const Instruction &op) {
  if (!attr.isOrContainsFunction())
    return false;

  // If we have a function attribute, check that it is non-null and in the
  // same module as the operation that refers to it.
  if (auto fnAttr = attr.dyn_cast<FunctionAttr>()) {
    if (!fnAttr.getValue())
      return failure("attribute refers to deallocated function!", op);

    if (fnAttr.getValue()->getModule() != fn.getModule())
      return failure("attribute refers to function '" +
                         Twine(fnAttr.getValue()->getName()) +
                         "' defined in another module!",
                     op);
    return false;
  }

  // Otherwise, we must have an array attribute, remap the elements.
  for (auto elt : attr.cast<ArrayAttr>().getValue()) {
    if (verifyAttribute(elt, op))
      return true;
  }

  return false;
}

bool FuncVerifier::verifyBlock(const Block &block, bool isTopLevel) {
  for (auto *arg : block.getArguments()) {
    if (arg->getOwner() != &block)
      return failure("block argument not owned by block", block);
  }

  for (auto &inst : block)
    if (verifyOperation(inst))
      return true;

  // If this block is at the function level, then verify that it has a
  // terminator.
  if (isTopLevel) {
    if (!block.getTerminator())
      return failure("block with no terminator", block);

    // Verify that this block is not branching to a block of a different
    // region.
    for (const Block *successor : block.getSuccessors())
      if (successor->getParent() != block.getParent())
        return failure("branching to a block of a different region",
                       *block.getTerminator());
  } else if (block.getTerminator()) {
    // TODO(riverriddle) Blocks in an IfInst/ForInst aren't allowed to have
    // terminators.
    return failure("non function block with terminator", block);
  }

  return false;
}

/// Check the invariants of the specified operation.
bool FuncVerifier::verifyOperation(const Instruction &op) {
  if (op.getFunction() != &fn)
    return failure("operation in the wrong function", op);

  // Check that operands are non-nil and structurally ok.
  for (const auto *operand : op.getOperands()) {
    if (!operand)
      return failure("null operand found", op);

    if (operand->getFunction() != &fn)
      return failure("reference to operand defined in another function", op);
  }

  // Verify all attributes are ok.  We need to check Function attributes, since
  // they are actually mutable (the function they refer to can be deleted), and
  // we have to check array attributes that can refer to them.
  for (auto attr : op.getAttrs()) {
    if (verifyAttribute(attr.second, op))
      return true;
  }

  // If we can get operation info for this, check the custom hook.
  if (auto *opInfo = op.getAbstractOperation()) {
    if (opInfo->verifyInvariants(&op))
      return true;
  }

  // Verify that all child blocks are ok.
  for (auto &blockList : op.getBlockLists())
    for (auto &b : blockList)
      if (verifyBlock(b, /*isTopLevel=*/false))
        return true;

  return false;
}

bool FuncVerifier::verifyDominance(const Block &block) {
  for (auto &inst : block) {
    // Check that all operands on the instruction are ok.
    if (verifyInstDominance(inst))
      return true;
    if (verifyOperation(inst))
      return true;
    for (auto &blockList : inst.getBlockLists())
      for (auto &block : blockList)
        if (verifyDominance(block))
          return true;
  }
  return false;
}

bool FuncVerifier::verifyInstDominance(const Instruction &inst) {
  // Check that operands properly dominate this use.
  for (unsigned operandNo = 0, e = inst.getNumOperands(); operandNo != e;
       ++operandNo) {
    auto *op = inst.getOperand(operandNo);
    if (domInfo->properlyDominates(op, &inst))
      continue;

    inst.emitError("operand #" + Twine(operandNo) +
                   " does not dominate this use");
    if (auto *useInst = op->getDefiningInst())
      useInst->emitNote("operand defined here");
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Entrypoints
//===----------------------------------------------------------------------===//

/// Perform (potentially expensive) checks of invariants, used to detect
/// compiler bugs.  On error, this reports the error through the MLIRContext and
/// returns true.
bool Function::verify() const { return FuncVerifier(*this).verify(); }

/// Perform (potentially expensive) checks of invariants, used to detect
/// compiler bugs.  On error, this reports the error through the MLIRContext and
/// returns true.
bool Module::verify() const {

  /// Check that each function is correct.
  for (auto &fn : *this) {
    if (fn.verify())
      return true;
  }

  return false;
}
