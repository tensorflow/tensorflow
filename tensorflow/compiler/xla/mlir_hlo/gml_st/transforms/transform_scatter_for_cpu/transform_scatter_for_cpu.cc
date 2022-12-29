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

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/interfaces/tiling_interface_impl.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMSCATTERFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

struct TransformScatterForCpuPass
    : public impl::TransformScatterForCpuPassBase<TransformScatterForCpuPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<mlir::gml_st::GmlStDialect, arith::ArithDialect,
                    tensor::TensorDialect>();
    mlir::gml_st::registerGmlStTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    mlir::gml_st::TilingOptions opts;
    opts.distribute = false;  // Tile to `for` loops.

    // Tile everything to points.
    opts.tileSizeComputationFn = [](OpBuilder &b, Operation *op) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(
          &op->getParentOfType<func::FuncOp>().getBody().front());

      auto loops = cast<gml_st::TilingInterface>(op).getLoopIteratorTypes();
      return SmallVector<Value>(
          loops.size(), b.create<arith::ConstantIndexOp>(op->getLoc(), 1));
    };

    auto filterFn = [&](Operation *op) {
      if (isa<mlir::thlo::ScatterOp>(op))
        return success();
      return failure();
    };

    RewritePatternSet patterns(ctx);
    populateTilingPatterns(ctx, filterFn, opts, &patterns);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    removeTilingLabels(f);
  }
};

}  // namespace
}  // namespace mlir::gml_st

namespace mlir::gml_st {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformScatterForCpuPass() {
  return std::make_unique<mlir::gml_st::TransformScatterForCpuPass>();
}

}  // namespace mlir::gml_st
