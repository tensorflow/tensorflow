//===- ParallelismDetection.cpp - Parallelism Detection pass ------------*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to detect parallel affine 'affine.for' ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Passes.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestParallelismDetection
    : public FunctionPass<TestParallelismDetection> {
  void runOnFunction() override;
};

} // end anonymous namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createParallelismDetectionTestPass() {
  return std::make_unique<TestParallelismDetection>();
}

// Walks the function and emits a note for all 'affine.for' ops detected as
// parallel.
void TestParallelismDetection::runOnFunction() {
  FuncOp f = getFunction();
  OpBuilder b(f.getBody());
  f.walk([&](AffineForOp forOp) {
    if (isLoopParallel(forOp))
      forOp.emitRemark("parallel loop");
    else
      forOp.emitRemark("sequential loop");
  });
}

static PassRegistration<TestParallelismDetection>
    pass("test-detect-parallel", "Test parallelism detection ");
