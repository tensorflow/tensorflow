/* Copyright 2026 The StableHLO Authors.

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

#include <cstdint>
#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo_ext/transforms/passes.h"  // NOLINT: Used in passes.h.inc

namespace mlir::stablehlo_ext {

#define GEN_PASS_DEF_CHLOSCANTOREDUCEWINDOWPASS
#include "stablehlo_ext/transforms/passes.h.inc"

// TODO(b/469020791): Remove this pass after 1.14 is the minimum supported
// version (i.e. after 2026-05-24).

namespace {

struct ConvertChloScanToReduceWindow
    : public mlir::OpRewritePattern<mlir::chlo::ScanOp> {
  using OpRewritePattern<mlir::chlo::ScanOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::chlo::ScanOp op, mlir::PatternRewriter& rewriter) const override {
    if (op.getInputs().size() != 1 || op.getInits().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Only single input/init scans are supported.");
    }

    mlir::Value input = op.getInputs()[0];
    mlir::Value init = op.getInits()[0];
    auto inputType = llvm::cast<mlir::RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    int64_t axis = op.getDimension();
    int64_t n = inputType.getDimSize(axis);
    bool reverse = op.getIsReverse();

    llvm::SmallVector<int64_t> windowDims(rank, 1);
    windowDims[axis] = n;

    llvm::SmallVector<int64_t> padding(rank * 2, 0);
    if (reverse) {
      padding[axis * 2 + 1] = n - 1;
    } else {
      padding[axis * 2] = n - 1;
    }

    auto windowDimsAttr = rewriter.getDenseI64ArrayAttr(windowDims);
    auto stridesAttr =
        rewriter.getDenseI64ArrayAttr(llvm::SmallVector<int64_t>(rank, 1));

    auto paddingType =
        mlir::RankedTensorType::get({rank, 2}, rewriter.getI64Type());
    auto paddingAttr = mlir::DenseIntElementsAttr::get(paddingType, padding);

    auto dilationsAttr =
        rewriter.getDenseI64ArrayAttr(llvm::SmallVector<int64_t>(rank, 1));

    // Create a scalar zero constant (rank-0 tensor) for reduce_window init
    auto scalarType =
        mlir::RankedTensorType::get({}, inputType.getElementType());
    mlir::Attribute zeroAttr;
    if (auto floatType =
            llvm::dyn_cast<mlir::FloatType>(inputType.getElementType())) {
      zeroAttr = rewriter.getFloatAttr(floatType, 0.0);
    } else {
      zeroAttr = rewriter.getIntegerAttr(inputType.getElementType(), 0);
    }
    auto scalarZero = mlir::stablehlo::ConstantOp::create(
        rewriter, op.getLoc(),
        mlir::DenseElementsAttr::get(scalarType, zeroAttr));

    auto reduceWindowOp = mlir::stablehlo::ReduceWindowOp::create(
        rewriter, op.getLoc(), op.getResultTypes()[0], input, scalarZero,
        windowDimsAttr, stridesAttr, dilationsAttr, dilationsAttr, paddingAttr);

    // Create a scalar block for the reduce_window reducer (using rank-0
    // tensors)
    auto* block = rewriter.createBlock(&reduceWindowOp.getBody());
    block->addArguments({scalarType, scalarType}, {op.getLoc(), op.getLoc()});

    rewriter.setInsertionPointToStart(block);
    auto addOp = mlir::stablehlo::AddOp::create(
        rewriter, op.getLoc(), block->getArgument(0), block->getArgument(1));
    mlir::stablehlo::ReturnOp::create(rewriter, op.getLoc(), addOp.getResult());

    rewriter.replaceOp(op, {reduceWindowOp.getResult(0), init});

    return mlir::success();
  }
};

struct ChloScanToReduceWindowPass
    : public impl::ChloScanToReduceWindowPassBase<ChloScanToReduceWindowPass> {
  using ChloScanToReduceWindowPassBase::ChloScanToReduceWindowPassBase;

 protected:
  void runOnOperation() override {
    if (!targetVersionOption.empty()) {
      auto targetVersion = mlir::vhlo::Version::fromString(targetVersionOption);
      if (mlir::succeeded(targetVersion) &&
          !(*targetVersion < mlir::vhlo::Version(1, 14, 0))) {
        return;
      }
    }

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ConvertChloScanToReduceWindow>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::stablehlo_ext
