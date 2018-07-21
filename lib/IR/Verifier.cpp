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

#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OperationSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;

namespace {
/// Base class for the verifiers in this file.  It is a pervasive truth that
/// this file treats "true" as an error that needs to be recovered from, and
/// "false" as success.
///
class Verifier {
public:
  template <typename T>
  static void failure(const Twine &message, const T &value, raw_ostream &os) {
    // Print the error message and flush the stream in case printing the value
    // causes a crash.
    os << "MLIR verification failure: " + message + "\n";
    os.flush();
    value.print(os);
  }

  template <typename T>
  bool failure(const Twine &message, const T &value) {
    // If the caller isn't trying to collect failure information, just print
    // the result and abort.
    if (!errorResult) {
      failure(message, value, llvm::errs());
      abort();
    }

    // Otherwise, emit the error into the string and return true.
    llvm::raw_string_ostream os(*errorResult);
    failure(message, value, os);
    os.flush();
    return true;
  }

protected:
  explicit Verifier(std::string *errorResult) : errorResult(errorResult) {}

private:
  std::string *errorResult;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// CFG Functions
//===----------------------------------------------------------------------===//

namespace {
class CFGFuncVerifier : public Verifier {
public:
  const CFGFunction &fn;
  OperationSet &operationSet;

  CFGFuncVerifier(const CFGFunction &fn, std::string *errorResult)
      : Verifier(errorResult), fn(fn),
        operationSet(OperationSet::get(fn.getContext())) {}

  bool verify();
  bool verifyBlock(const BasicBlock &block);
  bool verifyOperation(const OperationInst &inst);
  bool verifyTerminator(const TerminatorInst &term);
  bool verifyReturn(const ReturnInst &inst);
};
} // end anonymous namespace

bool CFGFuncVerifier::verify() {
  // TODO: Lots to be done here, including verifying dominance information when
  // we have uses and defs.

  for (auto &block : fn) {
    if (verifyBlock(block))
      return true;
  }
  return false;
}

bool CFGFuncVerifier::verifyBlock(const BasicBlock &block) {
  if (!block.getTerminator())
    return failure("basic block with no terminator", block);

  if (verifyTerminator(*block.getTerminator()))
    return true;

  for (auto &inst : block) {
    if (verifyOperation(inst))
      return true;
  }
  return false;
}

bool CFGFuncVerifier::verifyTerminator(const TerminatorInst &term) {
  if (term.getFunction() != &fn)
    return failure("terminator in the wrong function", term);

  // TODO: Check that operands are structurally ok.
  // TODO: Check that successors are in the right function.

  if (auto *ret = dyn_cast<ReturnInst>(&term))
    return verifyReturn(*ret);

  return false;
}

bool CFGFuncVerifier::verifyReturn(const ReturnInst &inst) {
  // Verify that the return operands match the results of the function.
  auto results = fn.getType()->getResults();
  if (inst.getNumOperands() != results.size())
    return failure("return has " + Twine(inst.getNumOperands()) +
                       " operands, but enclosing function returns " +
                       Twine(results.size()),
                   inst);

  return false;
}

bool CFGFuncVerifier::verifyOperation(const OperationInst &inst) {
  if (inst.getFunction() != &fn)
    return failure("operation in the wrong function", inst);

  // TODO: Check that operands are structurally ok.

  // See if we can get operation info for this.
  if (auto *opInfo = inst.getAbstractOperation(fn.getContext())) {
    if (auto errorMessage = opInfo->verifyInvariants(&inst))
      return failure(errorMessage, inst);
  }

  return false;
}

//===----------------------------------------------------------------------===//
// ML Functions
//===----------------------------------------------------------------------===//

namespace {
class MLFuncVerifier : public Verifier {
public:
  const MLFunction &fn;

  MLFuncVerifier(const MLFunction &fn, std::string *errorResult)
      : Verifier(errorResult), fn(fn) {}

  bool verify() {
    // TODO.
    return false;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Entrypoints
//===----------------------------------------------------------------------===//

/// Perform (potentially expensive) checks of invariants, used to detect
/// compiler bugs.  On error, this fills in the string and return true,
/// or aborts if the string was not provided.
bool Function::verify(std::string *errorResult) const {
  switch (getKind()) {
  case Kind::ExtFunc:
    // No body, nothing can be wrong here.
    return false;
  case Kind::CFGFunc:
    return CFGFuncVerifier(*cast<CFGFunction>(this), errorResult).verify();
  case Kind::MLFunc:
    return MLFuncVerifier(*cast<MLFunction>(this), errorResult).verify();
  }
}

/// Perform (potentially expensive) checks of invariants, used to detect
/// compiler bugs.  On error, this fills in the string and return true,
/// or aborts if the string was not provided.
bool Module::verify(std::string *errorResult) const {

  /// Check that each function is correct.
  for (auto fn : functionList) {
    if (fn->verify(errorResult))
      return true;
  }

  // Make sure the error string is empty on success.
  if (errorResult)
    errorResult->clear();
  return false;
}
