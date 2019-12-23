//===- TestVectorToVectorConversion.cpp - Test VectorTransfers lowering ---===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Dialect/VectorOps/VectorOps.h"
#include "mlir/Dialect/VectorOps/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::vector;

namespace {
#include "TestVectorTransformPatterns.h.inc"

struct TestVectorToVectorConversion
    : public FunctionPass<TestVectorToVectorConversion> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    populateWithGenerated(context, &patterns);
    populateVectorToVectorCanonicalizationPatterns(patterns, context);
    populateVectorToVectorTransformationPatterns(patterns, context);
    applyPatternsGreedily(getFunction(), patterns);
  }
};
} // end anonymous namespace

static PassRegistration<TestVectorToVectorConversion>
    pass("test-vector-to-vector-conversion",
         "Test conversion patterns between ops in the vector dialect");
