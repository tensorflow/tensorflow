//===- TestVectorToVectorConversion.cpp - Test VectorTransfers lowering ---===//
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
    applyPatternsGreedily(getFunction(), patterns);
  }
};
} // end anonymous namespace

static PassRegistration<TestVectorToVectorConversion>
    pass("test-vector-to-vector-conversion",
         "Test conversion patterns between ops in the vector dialect");
