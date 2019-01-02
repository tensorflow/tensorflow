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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass.h"
#include "mlir/Transforms/LoweringUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

// TODO: This shouldn't be its own pass, it should be a legalization (once we
// have the proper infra).
struct LowerAffineApply : public FunctionPass {
  explicit LowerAffineApply() : FunctionPass(&LowerAffineApply::passID) {}
  PassResult runOnFunction(Function *f) override;
  static char passID;
};

} // end anonymous namespace

char LowerAffineApply::passID = 0;

// Given an affine expression `expr` extracted from `op`, build the sequence of
// primitive instructions that correspond to the affine expression in the
// `builder`.
static bool expandAffineApply(AffineApplyOp *op) {
  if (!op)
    return true;

  FuncBuilder builder(op->getInstruction());
  auto maybeExpandedMap =
      expandAffineMap(&builder, op->getLoc(), op->getAffineMap(),
                      llvm::to_vector<8>(op->getOperands()));
  if (!maybeExpandedMap)
    return true;
  for (auto pair : llvm::zip(op->getResults(), *maybeExpandedMap)) {
    Value *original = std::get<0>(pair);
    Value *expanded = std::get<1>(pair);
    if (!expanded)
      return true;
    original->replaceAllUsesWith(expanded);
  }
  op->erase();
  return false;
}

PassResult LowerAffineApply::runOnFunction(Function *f) {
  SmallVector<OpPointer<AffineApplyOp>, 8> affineApplyInsts;

  // Find all the affine_apply operations.
  f->walkOps([&](OperationInst *inst) {
    auto applyOp = inst->dyn_cast<AffineApplyOp>();
    if (applyOp)
      affineApplyInsts.push_back(applyOp);
  });

  // Rewrite them in a second pass, avoiding invalidation of the walker
  // iterator.
  for (auto applyOp : affineApplyInsts)
    if (expandAffineApply(applyOp))
      return failure();

  return success();
}

static PassRegistration<LowerAffineApply>
    pass("lower-affine-apply",
         "Decompose affine_applies into primitive operations");

FunctionPass *mlir::createLowerAffineApplyPass() {
  return new LowerAffineApply();
}
