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

#include <memory>
#include <utility>

#include "./passes.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_TOSAPREPAREMHLOPASS
#include "./passes.h.inc"

#define PASS_NAME "tosa-prepare-mhlo"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {
namespace {

class PrepareMhlo : public ::impl::TosaPrepareMhloPassBase<PrepareMhlo> {
 public:
  explicit PrepareMhlo() = default;
  void runOnOperation() override;
};

void PrepareMhlo::runOnOperation() {
  auto* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  mhlo::DotGeneralOp::getCanonicalizationPatterns(patterns, ctx);
  mhlo::populateGeneralDotOpLoweringPatterns(&patterns, ctx);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createPrepareMhloPass() {
  return std::make_unique<PrepareMhlo>();
}

}  // namespace tosa
}  // namespace mlir
