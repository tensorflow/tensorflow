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
#include "mlir/IR/Module.h"
#include "mlir/IR/Statements.h"
#include "mlir/IR/StmtVisitor.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;

namespace {
/// Base class for the verifiers in this file.  It is a pervasive truth that
/// this file treats "true" as an error that needs to be recovered from, and
/// "false" as success.
///
class Verifier {
public:
  bool failure(const Twine &message, const OperationInst &value) {
    return value.emitError(message);
  }

  bool failure(const Twine &message, const Function &fn) {
    return fn.emitError(message);
  }

  bool failure(const Twine &message, const BasicBlock &bb) {
    // Take the location information for the first instruction in the block.
    if (!bb.empty())
      if (auto *op = dyn_cast<OperationInst>(&bb.front()))
        return failure(message, *op);

    // Worst case, fall back to using the function's location.
    return failure(message, fn);
  }

  bool verifyOperation(const OperationInst &op);
  bool verifyAttribute(Attribute attr, const OperationInst &op);

protected:
  explicit Verifier(const Function &fn) : fn(fn) {}

private:
  /// The function being checked.
  const Function &fn;
};
} // end anonymous namespace

// Check that function attributes are all well formed.
bool Verifier::verifyAttribute(Attribute attr, const OperationInst &op) {
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

/// Check the invariants of the specified operation.
bool Verifier::verifyOperation(const OperationInst &op) {
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

  return false;
}

//===----------------------------------------------------------------------===//
// CFG Functions
//===----------------------------------------------------------------------===//

namespace {
struct CFGFuncVerifier : public Verifier {
  const CFGFunction &fn;
  DominanceInfo domInfo;

  CFGFuncVerifier(const CFGFunction &fn)
      : Verifier(fn), fn(fn), domInfo(const_cast<CFGFunction *>(&fn)) {}

  bool verify();
  bool verifyBlock(const BasicBlock &block);
  bool verifyInstOperands(const Instruction &inst);
};
} // end anonymous namespace

bool CFGFuncVerifier::verify() {
  llvm::PrettyStackTraceFormat fmt("MLIR Verifier: cfgfunc @%s",
                                   fn.getName().c_str());

  // TODO: Lots to be done here, including verifying dominance information when
  // we have uses and defs.

  if (fn.empty())
    return failure("cfgfunc must have at least one basic block", fn);

  // Verify the first block has no predecessors.
  auto *firstBB = &fn.front();
  if (!firstBB->hasNoPredecessors()) {
    return failure("first block of cfgfunc must not have predecessors", fn);
  }

  // Verify that the argument list of the function and the arg list of the first
  // block line up.
  auto fnInputTypes = fn.getType().getInputs();
  if (fnInputTypes.size() != firstBB->getNumArguments())
    return failure("first block of cfgfunc must have " +
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
    if (verifyBlock(block))
      return true;
  }
  return false;
}

bool CFGFuncVerifier::verifyInstOperands(const Instruction &inst) {
  // Check that operands properly dominate this use.
  for (unsigned operandNo = 0, e = inst.getNumOperands(); operandNo != e;
       ++operandNo) {
    auto *op = inst.getOperand(operandNo);
    if (domInfo.properlyDominates(op, &inst))
      continue;

    inst.emitError("operand #" + Twine(operandNo) +
                   " does not dominate this use");
    if (auto *useInst = op->getDefiningInst())
      useInst->emitNote("operand defined here");
    return true;
  }

  return false;
}

bool CFGFuncVerifier::verifyBlock(const BasicBlock &block) {
  if (!block.getTerminator())
    return failure("basic block with no terminator", block);

  for (auto *arg : block.getArguments()) {
    if (arg->getOwner() != &block)
      return failure("basic block argument not owned by block", block);
  }

  for (auto &inst : block) {
    if (auto *opInst = dyn_cast<OperationInst>(&inst))
      if (verifyOperation(*opInst))
        return true;

    if (verifyInstOperands(inst))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// ML Functions
//===----------------------------------------------------------------------===//

namespace {
struct MLFuncVerifier : public Verifier, public StmtWalker<MLFuncVerifier> {
  const MLFunction &fn;
  bool hadError = false;

  MLFuncVerifier(const MLFunction &fn) : Verifier(fn), fn(fn) {}

  void visitOperationInst(OperationInst *opStmt) {
    hadError |= verifyOperation(*opStmt);
  }

  bool verify() {
    llvm::PrettyStackTraceFormat fmt("MLIR Verifier: mlfunc @%s",
                                     fn.getName().c_str());

    // Check basic structural properties.
    walk(const_cast<MLFunction *>(&fn));
    if (hadError)
      return true;

    // TODO: check that loop bounds and if conditions are properly formed.
    if (verifyReturn())
      return true;

    return verifyDominance();
  }

  /// Walk all of the code in this MLFunc and verify that the operands of any
  /// operations are properly dominated by their definitions.
  bool verifyDominance();

  /// Verify that function has a return statement that matches its signature.
  bool verifyReturn();
};
} // end anonymous namespace

/// Walk all of the code in this MLFunc and verify that the operands of any
/// operations are properly dominated by their definitions.
bool MLFuncVerifier::verifyDominance() {
  using HashTable = llvm::ScopedHashTable<const Value *, bool>;
  HashTable liveValues;
  HashTable::ScopeTy topScope(liveValues);

  // All of the arguments to the function are live for the whole function.
  for (auto *arg : fn.getArguments())
    liveValues.insert(arg, true);

  // This recursive function walks the statement list pushing scopes onto the
  // stack as it goes, and popping them to remove them from the table.
  std::function<bool(const StmtBlock &block)> walkBlock;
  walkBlock = [&](const StmtBlock &block) -> bool {
    HashTable::ScopeTy blockScope(liveValues);

    // The induction variable of a for statement is live within its body.
    if (auto *forStmt = dyn_cast_or_null<ForStmt>(block.getContainingStmt()))
      liveValues.insert(forStmt, true);

    for (auto &stmt : block) {
      // Verify that each of the operands are live.
      unsigned operandNo = 0;
      for (auto *opValue : stmt.getOperands()) {
        if (!liveValues.count(opValue)) {
          stmt.emitError("operand #" + Twine(operandNo) +
                         " does not dominate this use");
          if (auto *useStmt = opValue->getDefiningInst())
            useStmt->emitNote("operand defined here");
          return true;
        }
        ++operandNo;
      }

      if (auto *opStmt = dyn_cast<OperationInst>(&stmt)) {
        // Operations define values, add them to the hash table.
        for (auto *result : opStmt->getResults())
          liveValues.insert(result, true);
        continue;
      }

      // If this is an if or for, recursively walk the block they contain.
      if (auto *ifStmt = dyn_cast<IfStmt>(&stmt)) {
        if (walkBlock(*ifStmt->getThen()))
          return true;

        if (auto *elseClause = ifStmt->getElse())
          if (walkBlock(*elseClause))
            return true;
      }
      if (auto *forStmt = dyn_cast<ForStmt>(&stmt))
        if (walkBlock(*forStmt->getBody()))
          return true;
    }

    return false;
  };

  // Check the whole function out.
  return walkBlock(*fn.getBody());
}

bool MLFuncVerifier::verifyReturn() {
  // TODO: fold return verification in the pass that verifies all statements.
  const char missingReturnMsg[] = "ML function must end with return statement";
  if (fn.getBody()->getStatements().empty())
    return failure(missingReturnMsg, fn);

  const auto &stmt = fn.getBody()->getStatements().back();
  if (const auto *op = dyn_cast<OperationInst>(&stmt)) {
    if (!op->isReturn())
      return failure(missingReturnMsg, fn);

    return false;
  }
  return failure(missingReturnMsg, fn);
}

//===----------------------------------------------------------------------===//
// Entrypoints
//===----------------------------------------------------------------------===//

/// Perform (potentially expensive) checks of invariants, used to detect
/// compiler bugs.  On error, this reports the error through the MLIRContext and
/// returns true.
bool Function::verify() const {
  switch (getKind()) {
  case Kind::ExtFunc:
    // No body, nothing can be wrong here.
    return false;
  case Kind::CFGFunc:
    return CFGFuncVerifier(*cast<CFGFunction>(this)).verify();
  case Kind::MLFunc:
    return MLFuncVerifier(*cast<MLFunction>(this)).verify();
  }
}

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
