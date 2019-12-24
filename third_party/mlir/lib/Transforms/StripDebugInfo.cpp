//===- StripDebugInfo.cpp - Pass to strip debug information ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct StripDebugInfo : public FunctionPass<StripDebugInfo> {
  void runOnFunction() override;
};
} // end anonymous namespace

void StripDebugInfo::runOnFunction() {
  FuncOp func = getFunction();
  auto unknownLoc = UnknownLoc::get(&getContext());

  // Strip the debug info from the function and its operations.
  func.setLoc(unknownLoc);
  func.walk([&](Operation *op) { op->setLoc(unknownLoc); });
}

/// Creates a pass to strip debug information from a function.
std::unique_ptr<OpPassBase<FuncOp>> mlir::createStripDebugInfoPass() {
  return std::make_unique<StripDebugInfo>();
}

static PassRegistration<StripDebugInfo>
    pass("strip-debuginfo", "Strip debug info from functions and operations");
