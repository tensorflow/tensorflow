/* Copyright 2025 The OpenXLA Authors.

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
#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace ttir = ::mlir::triton;

#define GEN_PASS_DEF_TENSORLOWERTOTRITONPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

class LowerBitcast : public mlir::OpRewritePattern<tensor::BitcastOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      tensor::BitcastOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<ttir::BitcastOp>(op, op.getResult().getType(),
                                                 op.getOperand());
    return mlir::success();
  }
};

class LowerExtractOnOneElementTensorToReshapeReduce
    : public mlir::OpRewritePattern<tensor::ExtractOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      tensor::ExtractOp op, mlir::PatternRewriter& rewriter) const override {
    auto input_tensor_type = op.getTensor().getType();
    auto input_tensor_shape = input_tensor_type.getShape();

    if (input_tensor_shape.empty() ||
        !absl::c_all_of(input_tensor_shape,
                        [](int64_t dim) { return dim == 1; })) {
      return rewriter.notifyMatchFailure(
          op,
          "Extract will only be lowered for tensors with all dimensions equal "
          "to 1.");
    }

    // First, reshape to a 1D tensor if not already the case. This is needed
    // because triton::ReduceOp can only reduce 1 dimension at a time.
    auto single_dim_tensor = op.getTensor();
    if (input_tensor_type.getRank() > 1) {
      Type output_tensor_type =
          mlir::RankedTensorType::get({1}, input_tensor_type.getElementType());
      single_dim_tensor = ttir::ReshapeOp::create(
          rewriter, op.getLoc(), output_tensor_type, single_dim_tensor,
          /*allow_reorder=*/true);
    }

    // Second, reduce to a scalar.
    ttir::ReduceOp reduction = ttir::ReduceOp::create(
        rewriter, op.getLoc(), single_dim_tensor, /*axis=*/0);

    auto element_type = input_tensor_type.getElementType();
    mlir::Location loc = op.getLoc();
    mlir::Block* reducer =
        rewriter.createBlock(&reduction->getRegion(0), /*insertPt=*/{},
                             /*argTypes=*/
                             {element_type, element_type},
                             /*locs=*/{loc, loc});

    rewriter.setInsertionPointToStart(reducer);
    Value result = mlir::isa<mlir::IntegerType>(element_type)
                       ? arith::AddIOp::create(
                             rewriter, reducer->getArgument(0).getLoc(),
                             reducer->getArgument(0), reducer->getArgument(1))
                             .getResult()
                       : arith::AddFOp::create(
                             rewriter, reducer->getArgument(0).getLoc(),
                             reducer->getArgument(0), reducer->getArgument(1))
                             .getResult();
    ttir::ReduceReturnOp::create(rewriter, result.getLoc(),
                                 SmallVector<Value>({result}));
    rewriter.setInsertionPointAfter(reduction);
    rewriter.replaceOp(op, reduction);

    return mlir::success();
  }
};

// TODO(basioli): Consider fusing this with the stablehlo lowering pass into a
// single xtile to triton lowering pass.
class TensorLowerToTritonPass
    : public impl::TensorLowerToTritonPassBase<TensorLowerToTritonPass> {
 public:
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<LowerBitcast, LowerExtractOnOneElementTensorToReshapeReduce>(
        mlir_context);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTensorLowerToTritonPass() {
  return std::make_unique<TensorLowerToTritonPass>();
}

}  // namespace mlir::triton::xla
