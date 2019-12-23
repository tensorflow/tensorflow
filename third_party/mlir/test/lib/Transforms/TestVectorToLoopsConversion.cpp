//===- TestVectorToLoopsConversion.cpp - Test VectorTransfers lowering ----===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "mlir/Conversion/VectorToLoops/ConvertVectorToLoops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

struct TestVectorToLoopsPass
    : public FunctionPass<TestVectorToLoopsPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    populateVectorToAffineLoopsConversionPatterns(context, patterns);
    applyPatternsGreedily(getFunction(), patterns);
  }
};

} // end anonymous namespace

static PassRegistration<TestVectorToLoopsPass>
    pass("test-convert-vector-to-loops",
         "Converts vector transfer ops to loops over scalars and vector casts");
