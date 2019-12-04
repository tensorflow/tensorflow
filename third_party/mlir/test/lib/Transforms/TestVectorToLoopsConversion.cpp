//===- TestVectorToLoopsConversion.cpp - Test VectorTransfers lowering ----===//
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
