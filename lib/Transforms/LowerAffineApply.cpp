//===- LowerAffineApply.cpp - Convert affine_apply to primitives ----------===//
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
#include "mlir/Pass.h"
#include "mlir/Transforms/LoweringUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

struct LowerAffineApply : public FunctionPass {

  explicit LowerAffineApply() : FunctionPass(&LowerAffineApply::passID) {}

  PassResult runOnMLFunction(Function *f) override;
  PassResult runOnCFGFunction(Function *f) override;

  static char passID;
};

} // end anonymous namespace

char LowerAffineApply::passID = 0;

PassResult LowerAffineApply::runOnMLFunction(Function *f) {
  f->emitError("ML Functions contain syntactically hidden affine_apply's that "
               "cannot be expanded");
  return failure();
}

PassResult LowerAffineApply::runOnCFGFunction(Function *f) {
  for (Block &bb : *f) {
    // Handle iterators with care because we erase in the same loop.
    // In particular, step to the next element before erasing the current one.
    for (auto it = bb.begin(); it != bb.end();) {
      auto *inst = dyn_cast<OperationInst>(&*it++);
      if (!inst)
        continue;

      auto affineApplyOp = inst->dyn_cast<AffineApplyOp>();
      if (!affineApplyOp)
        continue;
      if (expandAffineApply(&*affineApplyOp))
        return failure();
    }
  }

  return success();
}

static PassRegistration<LowerAffineApply>
    pass("lower-affine-apply",
         "Decompose affine_applies into primitive operations");

FunctionPass *mlir::createLowerAffineApplyPass() {
  return new LowerAffineApply();
}
