//===- Deaffinator.cpp - Convert affine_apply to primitives -----*- C++ -*-===//
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
// This file defines an MLIR function pass that replaces affine_apply operations
// in CFGFunctions with sequences of corresponding elementary arithmetic
// operations.
//
//===----------------------------------------------------------------------===//
//
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/Pass.h"
#include "mlir/Transforms/LoweringUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

struct Deaffinator : public FunctionPass {

  explicit Deaffinator() : FunctionPass(&Deaffinator::passID) {}

  PassResult runOnMLFunction(MLFunction *f) override;
  PassResult runOnCFGFunction(CFGFunction *f) override;

  static char passID;
};

} // end anonymous namespace

char Deaffinator::passID = 0;

PassResult Deaffinator::runOnMLFunction(MLFunction *f) {
  f->emitError("ML Functions contain syntactically hidden affine_apply's that "
               "cannot be expanded");
  return failure();
}

PassResult Deaffinator::runOnCFGFunction(CFGFunction *f) {
  for (BasicBlock &bb : *f) {
    // Handle iterators with care because we erase in the same loop.
    // In particular, step to the next element before erasing the current one.
    for (auto it = bb.begin(); it != bb.end();) {
      Instruction &inst = *it;
      auto affineApplyOp = inst.dyn_cast<AffineApplyOp>();
      ++it;

      if (!affineApplyOp)
        continue;
      if (expandAffineApply(*affineApplyOp))
        return failure();
    }
  }

  return success();
}

static PassRegistration<Deaffinator>
    pass("deaffinator", "Decompose affine_applies into primitive operations");

FunctionPass *createDeaffinatorPass() { return new Deaffinator(); }
