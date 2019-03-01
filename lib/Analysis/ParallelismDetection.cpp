//===- ParallelismDetection.cpp - Parallelism Detection pass ------------*-===//
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
// This file implements a pass to detect parallel affine 'for' ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/Analysis/Passes.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct LoopParallelismDetection
    : public FunctionPass<LoopParallelismDetection> {
  void runOnFunction() override;
};

} // end anonymous namespace

FunctionPassBase *mlir::createLoopParallelismDetectionPass() {
  return new LoopParallelismDetection();
}

// Walks the function and marks all parallel 'for' ops with an attribute.
void LoopParallelismDetection::runOnFunction() {
  Function *f = &getFunction();
  FuncBuilder b(f);
  f->walk<AffineForOp>([&](OpPointer<AffineForOp> forOp) {
    forOp->getInstruction()->setAttr("parallel",
                                     b.getBoolAttr(isLoopParallel(forOp)));
  });
}

static PassRegistration<LoopParallelismDetection> pass("detect-parallel",
                                                       "Detect parallel loops");
