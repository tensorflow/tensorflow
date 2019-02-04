//===- ComposeAffineMaps.cpp - MLIR Affine Transform Class-----*- C++ -*-===//
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
// This file implements a testing pass which composes affine maps from
// AffineApplyOps in a Function, by forward subtituting results from an
// AffineApplyOp into any of its users which are also AffineApplyOps.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/InstVisitor.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

// ComposeAffineMaps walks all affine apply op's in a function, and for each
// such op, composes into it the results of any other AffineApplyOps - so
// that all operands of the composed AffineApplyOp are guaranteed to be either
// loop IVs or terminal symbols, (i.e., Values that are themselves not the
// result of any AffineApplyOp). After this composition, AffineApplyOps with no
// remaining uses are erased.
// TODO(andydavis) Remove this when Chris adds instruction combiner pass.
struct ComposeAffineMaps : public FunctionPass, InstWalker<ComposeAffineMaps> {
  explicit ComposeAffineMaps() : FunctionPass(&ComposeAffineMaps::passID) {}
  PassResult runOnFunction(Function *f) override;
  void visitInstruction(OperationInst *opInst);

  SmallVector<OpPointer<AffineApplyOp>, 8> affineApplyOps;

  static char passID;
};

} // end anonymous namespace

char ComposeAffineMaps::passID = 0;

FunctionPass *mlir::createComposeAffineMapsPass() {
  return new ComposeAffineMaps();
}

static bool affineApplyOp(const Instruction &inst) {
  const auto &opInst = cast<OperationInst>(inst);
  return opInst.isa<AffineApplyOp>();
}

void ComposeAffineMaps::visitInstruction(OperationInst *opInst) {
  if (auto afOp = opInst->dyn_cast<AffineApplyOp>()) {
    affineApplyOps.push_back(afOp);
  }
}

PassResult ComposeAffineMaps::runOnFunction(Function *f) {
  // If needed for future efficiency, reserve space based on a pre-walk.
  affineApplyOps.clear();
  walk(f);
  for (auto afOp : affineApplyOps) {
    SmallVector<Value *, 8> operands(afOp->getOperands());
    FuncBuilder b(afOp->getInstruction());
    auto newAfOp = makeComposedAffineApply(&b, afOp->getLoc(),
                                           afOp->getAffineMap(), operands);
    afOp->replaceAllUsesWith(newAfOp);
  }

  // Erase dead affine apply ops.
  affineApplyOps.clear();
  walk(f);
  for (auto it = affineApplyOps.rbegin(); it != affineApplyOps.rend(); ++it) {
    if ((*it)->use_empty()) {
      (*it)->erase();
    }
  }

  return success();
}

static PassRegistration<ComposeAffineMaps> pass("compose-affine-maps",
                                                "Compose affine maps");
