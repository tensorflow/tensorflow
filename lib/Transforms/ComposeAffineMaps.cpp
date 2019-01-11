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
#include "mlir/Analysis/MLFunctionMatcher.h"
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

// ComposeAffineMaps walks inst blocks in a Function, and for each
// AffineApplyOp, forward substitutes its results into any users which are
// also AffineApplyOps. After forward subtituting its results, AffineApplyOps
// with no remaining uses are collected and erased after the walk.
// TODO(andydavis) Remove this when Chris adds instruction combiner pass.
struct ComposeAffineMaps : public FunctionPass {
  explicit ComposeAffineMaps() : FunctionPass(&ComposeAffineMaps::passID) {}
  PassResult runOnFunction(Function *f) override;

  // Thread-safe RAII contexts local to pass, BumpPtrAllocator freed on exit.
  MLFunctionMatcherContext MLContext;

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

PassResult ComposeAffineMaps::runOnFunction(Function *f) {
  using matcher::Op;

  auto pattern = Op(affineApplyOp);
  auto apps = pattern.match(f);
  for (auto m : apps) {
    auto app = cast<OperationInst>(m.first)->cast<AffineApplyOp>();
    SmallVector<Value *, 8> operands(app->getOperands().begin(),
                                     app->getOperands().end());
    FuncBuilder b(m.first);
    auto newApp = makeComposedAffineApply(&b, app->getLoc(),
                                          app->getAffineMap(), operands);
    unsigned idx = 0;
    for (auto *v : app->getResults()) {
      v->replaceAllUsesWith(newApp->getResult(idx++));
    }
  }
  {
    auto pattern = Op(affineApplyOp);
    auto apps = pattern.match(f);
    std::reverse(apps.begin(), apps.end());
    for (auto m : apps) {
      auto app = cast<OperationInst>(m.first)->cast<AffineApplyOp>();
      bool hasNonEmptyUse = llvm::any_of(
          app->getResults(), [](Value *r) { return !r->use_empty(); });
      if (!hasNonEmptyUse) {
        m.first->erase();
      }
    }
  }
  return success();
}

static PassRegistration<ComposeAffineMaps> pass("compose-affine-maps",
                                                "Compose affine maps");
