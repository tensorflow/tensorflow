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

#include <utility>

#include "mhlo_tosa/Transforms/passes.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "passes_detail.h"

#define PASS_NAME "tosa-legalize-tf"
#define DEBUG_TYPE PASS_NAME

#include "mhlo_tosa/Transforms/legalize_mhlo.pdll.h.inc"

namespace mlir {
namespace tosa {
namespace {

struct LegalizeMhlo : TosaLegalizeMhloPassBase<LegalizeMhlo> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

 private:
  FrozenRewritePatternSet patterns;
};

}  // namespace

LogicalResult LegalizeMhlo::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);
  populateGeneratedPDLLPatterns(patternList);
  patterns = std::move(patternList);
  return success();
}

void LegalizeMhlo::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeMhloPass() {
  return std::make_unique<LegalizeMhlo>();
}

}  // namespace tosa
}  // namespace mlir
