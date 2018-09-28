//===- PipelineDataTransfer.cpp --- Pass for pipelining data movement ---*-===//
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
// This file implements a pass to pipeline data transfers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Statements.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Pass.h"

using namespace mlir;

namespace {

struct PipelineDataTransfer : public MLFunctionPass {
  explicit PipelineDataTransfer() {}
  PassResult runOnMLFunction(MLFunction *f) override;
};

} // end anonymous namespace

/// Creates a pass to pipeline explicit movement of data across levels of the
/// memory hierarchy.
MLFunctionPass *mlir::createPipelineDataTransferPass() {
  return new PipelineDataTransfer();
}

// For testing purposes, this just runs on the first statement of the MLFunction
// if that statement is a for stmt, and shifts the second half of its body by
// one.
PassResult PipelineDataTransfer::runOnMLFunction(MLFunction *f) {
  if (f->empty())
    return PassResult::Success;
  auto *forStmt = dyn_cast<ForStmt>(&f->front());
  if (!forStmt)
    return PassResult::Failure;

  unsigned numStmts = forStmt->getStatements().size();
  if (numStmts == 0)
    return PassResult::Success;

  std::vector<uint64_t> delays(numStmts);
  for (unsigned i = 0; i < numStmts; i++)
    delays[i] = (i < numStmts / 2) ? 0 : 1;

  if (!checkDominancePreservationOnShift(*forStmt, delays))
    // Violates SSA dominance.
    return PassResult::Failure;

  if (stmtBodySkew(forStmt, delays))
    return PassResult::Failure;

  return PassResult::Success;
}
