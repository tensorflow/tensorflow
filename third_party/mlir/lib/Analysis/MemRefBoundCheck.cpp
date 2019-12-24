//===- MemRefBoundCheck.cpp - MLIR Affine Structures Class ----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to check memref accesses for out of bound
// accesses.
//
//===----------------------------------------------------------------------===//

#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Passes.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "memref-bound-check"

using namespace mlir;

namespace {

/// Checks for out of bound memef access subscripts..
struct MemRefBoundCheck : public FunctionPass<MemRefBoundCheck> {
  void runOnFunction() override;
};

} // end anonymous namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createMemRefBoundCheckPass() {
  return std::make_unique<MemRefBoundCheck>();
}

void MemRefBoundCheck::runOnFunction() {
  getFunction().walk([](Operation *opInst) {
    TypeSwitch<Operation *>(opInst).Case<AffineLoadOp, AffineStoreOp>(
        [](auto op) { boundCheckLoadOrStoreOp(op); });

    // TODO(bondhugula): do this for DMA ops as well.
  });
}

static PassRegistration<MemRefBoundCheck>
    memRefBoundCheck("memref-bound-check",
                     "Check memref access bounds in a Function");
