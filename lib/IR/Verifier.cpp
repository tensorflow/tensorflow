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

template <typename T>
static void failure(const Twine &message, const T &value) {
  // Print the error message and flush the stream in case printing the value
  // causes a crash.
  llvm::errs() << "MLIR verification failure: " << message << "\n";
  llvm::errs().flush();
  value.dump();
}

//===----------------------------------------------------------------------===//
// CFG Functions
//===----------------------------------------------------------------------===//

namespace {
class CFGFuncVerifier {
public:
  const CFGFunction &fn;
  OperationSet &operationSet;

  CFGFuncVerifier(const CFGFunction &fn)
      : fn(fn), operationSet(OperationSet::get(fn.getContext())) {}

  void verify();
  void verifyBlock(const BasicBlock &block);
  void verifyTerminator(const TerminatorInst &term);
  void verifyOperation(const OperationInst &inst);
};
} // end anonymous namespace

void CFGFuncVerifier::verify() {
  // TODO: Lots to be done here, including verifying dominance information when
  // we have uses and defs.

  for (auto &block : fn) {
    verifyBlock(block);
  }
}

void CFGFuncVerifier::verifyBlock(const BasicBlock &block) {
  if (!block.getTerminator())
    failure("basic block with no terminator", block);
  verifyTerminator(*block.getTerminator());

  for (auto &inst : block) {
    verifyOperation(inst);
  }
}

void CFGFuncVerifier::verifyTerminator(const TerminatorInst &term) {
  if (term.getFunction() != &fn)
    failure("terminator in the wrong function", term);

  // TODO: Check that operands are structurally ok.
  // TODO: Check that successors are in the right function.
}

void CFGFuncVerifier::verifyOperation(const OperationInst &inst) {
  if (inst.getFunction() != &fn)
    failure("operation in the wrong function", inst);

  // TODO: Check that operands are structurally ok.

  // See if we can get operation info for this.
  if (auto *opInfo = inst.getAbstractOperation(fn.getContext())) {
    if (auto errorMessage = opInfo->verifyInvariants(&inst))
      failure(errorMessage, inst);
  }
}

//===----------------------------------------------------------------------===//
// ML Functions
//===----------------------------------------------------------------------===//

namespace {
class MLFuncVerifier {
public:
  const MLFunction &fn;

  MLFuncVerifier(const MLFunction &fn) : fn(fn) {}

  void verify() {
    // TODO.
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Entrypoints
//===----------------------------------------------------------------------===//

/// Perform (potentially expensive) checks of invariants, used to detect
/// compiler bugs.  This aborts on failure.
void Function::verify() const {
  switch (getKind()) {
  case Kind::ExtFunc:
    // No body, nothing can be wrong here.
    break;
  case Kind::CFGFunc:
    return CFGFuncVerifier(*cast<CFGFunction>(this)).verify();
  case Kind::MLFunc:
    return MLFuncVerifier(*cast<MLFunction>(this)).verify();
  }
}

/// Perform (potentially expensive) checks of invariants, used to detect
/// compiler bugs.  This aborts on failure.
void Module::verify() const {
  /// Check that each function is correct.
  for (auto fn : functionList) {
    fn->verify();
  }
}
