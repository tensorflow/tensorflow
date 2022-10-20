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

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir-hlo/Dialect/gml_st/transforms/tiling.h"
#include "mlir-hlo/Dialect/gml_st/transforms/tiling_interface_impl.h"
#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"
#include "mlir-hlo/Dialect/thlo/IR/thlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMMATMULFORCPUPASS
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h.inc"

struct TransformMatmulForCpuPass
    : public impl::TransformMatmulForCpuPassBase<TransformMatmulForCpuPass> {
  TransformMatmulForCpuPass() = default;
  explicit TransformMatmulForCpuPass(
      llvm::ArrayRef<int64_t> matmulTileSizes) {
    tileSizes = matmulTileSizes;
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<mlir::gml_st::GmlStDialect, arith::ArithDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
    mlir::gml_st::registerGmlStTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    mlir::gml_st::TilingOptions opts;

    if ((*tileSizes).empty()) {
      tileSizes = {2, 2, 2};
    }

    assert(tileSizes.size() == 3 &&
           "Tiling sizes for MatMul should have 3 elements");

    auto filter_fn = [&](Operation *op) {
      return success(isa<mlir::linalg::MatmulOp>(op));
    };

    ///////////////////////////////
    // Tiling parallel dimensions
    opts.setTileSizeComputationFn({(*tileSizes)[0], (*tileSizes)[1], 0});

    RewritePatternSet patterns(ctx);
    populateTilingPatterns(ctx, filter_fn, opts, &patterns);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Ensure we drop the marker in the end.
    f.walk([](linalg::LinalgOp op) { gml_st::removeTransformationAttr(op); });

    ///////////////////////////////
    // Tiling reduction dimension
    opts.setTileSizeComputationFn({0, 0, (*tileSizes).back()});
    opts.distribute = false;

    RewritePatternSet newpatterns(ctx);
    populateTilingPatterns(ctx, filter_fn, opts, &newpatterns);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(newpatterns)))) {
      return signalPassFailure();
    }

    // Ensure we drop the marker in the end.
    f.walk([](linalg::LinalgOp op) { gml_st::removeTransformationAttr(op); });
  }
};

}  // namespace
}  // namespace mlir::gml_st

namespace mlir::gml_st {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformMatmulForCpuPass() {
  return std::make_unique<mlir::gml_st::TransformMatmulForCpuPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformMatmulForCpuPass(llvm::ArrayRef<int64_t> matmulTileSizes) {
  return std::make_unique<mlir::gml_st::TransformMatmulForCpuPass>(
      matmulTileSizes);
}

}  // namespace mlir::gml_st
