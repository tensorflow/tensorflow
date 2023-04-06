/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <iterator>
#include <memory>
#include <utility>

#include "gml_st/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_REWRITEDOTASREDUCEPASS
#include "gml_st/transforms/passes.h.inc"

LogicalResult rewriteDotAsReduce(linalg::DotOp dotOp,
                                 PatternRewriter &rewriter) {
  Location loc = dotOp.getLoc();

  // Create empty tensor for linalg.map.
  Value lhs = dotOp.getInputs().front();
  FailureOr<OpFoldResult> inputSizeOfr =
      tensor::createDimValue(rewriter, loc, lhs, 0);

  if (failed(inputSizeOfr)) {
    return rewriter.notifyMatchFailure(
        dotOp, "cannot get the size of the input tensor");
  }

  // For small linalg.dots it is more beneficial to use the usual matmul
  // codegen.
  auto inputSize = getConstantIntValue(*inputSizeOfr);
  constexpr int64_t kInputSizeThreshold = 32;
  if (inputSize && *inputSize <= kInputSizeThreshold) return failure();

  Type elementType = getElementTypeOrSelf(lhs.getType());
  Value emptyTensor =
      rewriter.create<tensor::EmptyOp>(loc, *inputSizeOfr, elementType);

  // Create linalg.map.
  Operation *arithMul = &dotOp.getBody()->front();
  auto mul = rewriter.create<linalg::MapOp>(
      loc, dotOp.getOperands().take_front(2), emptyTensor,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        auto *n = mlir::clone(b, arithMul, arithMul->getResultTypes(),
                              args.take_front(2));
        b.create<linalg::YieldOp>(loc, n->getResults());
      });

  // Create linalg.reduce.
  Operation *arithAdd = &(*std::next(dotOp.getBody()->begin()));
  auto add = rewriter.create<linalg::ReduceOp>(
      loc, ValueRange{mul.getResult()}, ValueRange{dotOp.getOperand(2)},
      SmallVector<int64_t>{0},
      [&](OpBuilder &b, Location loc, ValueRange args) {
        auto *n = mlir::clone(b, arithAdd, arithAdd->getResultTypes(),
                              {args[1], args[0]});
        b.create<linalg::YieldOp>(loc, n->getResults());
      });

  rewriter.replaceOp(dotOp, add->getResults());
  return success();
}

struct RewriteDotAsReducePass
    : public impl::RewriteDotAsReducePassBase<RewriteDotAsReducePass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add(rewriteDotAsReduce);
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createRewriteDotAsReducePass() {
  return std::make_unique<RewriteDotAsReducePass>();
}

}  // namespace mlir::gml_st
