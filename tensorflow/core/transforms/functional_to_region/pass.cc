/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/transforms/functional_to_region/pass.h"

#include <memory>
#include <utility>

#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/transforms/functional_to_region/impl.h"

namespace mlir {
namespace tfg {
namespace {

#define GEN_PASS_DEF_FUNCTIONALTOREGION
#include "tensorflow/core/transforms/passes.h.inc"

struct FunctionalToRegionPass
    : public impl::FunctionalToRegionBase<FunctionalToRegionPass> {
  void runOnOperation() override {
    SymbolTable table(getOperation());
    RewritePatternSet patterns(&getContext());
    PopulateFunctionalToRegionPatterns(patterns, table);

    GreedyRewriteConfig config;
    // Use top-down traversal for more efficient conversion. Disable region
    // simplification as all regions are single block.
    config.setUseTopDownTraversal(true);
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);
    // If there are deeply nested conditionals, instantiating them too deep will
    // cause the verifiers, which are implemented recursively, to stack
    // overflow. Set a relatively low iteration limit.
    config.setMaxIterations(16);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<Pass> CreateFunctionalToRegionPass() {
  return std::make_unique<FunctionalToRegionPass>();
}

}  // namespace tfg
}  // namespace mlir
