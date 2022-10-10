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

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/gml_st/transforms/tiling.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/gml_st/transforms/tiling_interface_impl.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/thlo/IR/thlo_ops.h"

namespace mlir::cpu {
namespace {

#define GEN_PASS_DEF_TILETHLOFORCPUPASS
#include "tensorflow/compiler/xla/mlir/transforms/cpu/passes.h.inc"

struct TileThloForCpuPass
    : public impl::TileThloForCpuPassBase<TileThloForCpuPass> {
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
    // TODO(kramerb): This can be made a lot smarter for ops other than scatter.
    // TODO(kramerb): Tile scatter when we have scatter with slices.
    opts.tileSizeComputationFn = [](OpBuilder &b, Operation *op) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(
          &op->getParentOfType<func::FuncOp>().getBody().front());

      auto loops = cast<gml_st::TilingInterface>(op).getLoopIteratorTypes();
      return SmallVector<Value>(
          loops.size(), b.create<arith::ConstantIndexOp>(op->getLoc(), 1));
    };

    auto filterFn = [&](Operation *op) {
      if (op->getDialect()->getNamespace() !=
          thlo::THLODialect::getDialectNamespace())
        return failure();
      return success();
    };

    RewritePatternSet patterns(ctx);
    populateTilingPatterns(ctx, filterFn, opts, &patterns);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::cpu

namespace xla::cpu {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTileThloForCpuPass() {
  return std::make_unique<mlir::cpu::TileThloForCpuPass>();
}

}  // namespace xla::cpu
