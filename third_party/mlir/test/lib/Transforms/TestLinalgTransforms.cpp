//===- TestLinalgTransforms.cpp - Test Linalg transformation patterns -----===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Linalg transformations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/LinalgTransforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace linalg {
namespace {
#include "TestLinalgTransformPatterns.h.inc"
} // end namespace
} // end namespace linalg
} // end namespace mlir

namespace {
struct TestLinalgTransforms : public FunctionPass<TestLinalgTransforms> {
  void runOnFunction() override;
};
} // end anonymous namespace

/// Apply transformations specified as patterns.
void TestLinalgTransforms::runOnFunction() {
  OwningRewritePatternList patterns;
  auto funcOp = getFunction();

  // Add the generated patterns to the list.
  linalg::populateWithGenerated(&getContext(), &patterns);
  applyPatternsGreedily(funcOp, patterns);

  // Drop the marker.
  funcOp.walk([](LinalgOp op) {
    op.removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
}

static PassRegistration<TestLinalgTransforms>
    pass("test-linalg-transform-patterns",
         "Test Linalg transformation patterns by applying them greedily.");
