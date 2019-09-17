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
