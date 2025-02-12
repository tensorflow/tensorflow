/* Copyright 2019 The OpenXLA Authors.

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

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_TESTUNFUSEBATCHNORMPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

struct TestUnfuseBatchNormPass
    : public impl::TestUnfuseBatchNormPassBase<TestUnfuseBatchNormPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateUnfuseBatchNormInferencePattern(&getContext(), &patterns);
    populateUnfuseBatchNormTrainingPattern(&getContext(), &patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<::mlir::Pass> createTestUnfuseBatchNormPass() {
  return std::make_unique<TestUnfuseBatchNormPass>();
}

}  // namespace mhlo
}  // namespace mlir
