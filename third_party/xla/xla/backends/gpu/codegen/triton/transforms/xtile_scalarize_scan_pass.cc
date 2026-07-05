/* Copyright 2026 The OpenXLA Authors.

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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_XTILESCALARIZESCANPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Changes the scan region to have scalar arguments instead of tensor arguments.
// The region's operations are rewritten to single-element tensor operations.
class ScalarizeScanRegionPattern
    : public mlir::OpRewritePattern<::xla::xtile::ScanOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      ::xla::xtile::ScanOp op, mlir::PatternRewriter& rewriter) const override {
    if (llvm::none_of(op.getBody().getArgumentTypes(), [](mlir::Type type) {
          return mlir::isa<mlir::RankedTensorType>(type);
        })) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "No tensor arguments in scan region");
    }

    rewriter.modifyOpInPlace(op, [&]() {
      mlir::Block& block = op.getBody().front();
      RewriteBlockArguments(block, rewriter);
      RewriteRegionOps(block, rewriter);
      RewriteTerminator(block, rewriter);
    });

    return mlir::success();
  }

  static void RewriteBlockArguments(mlir::Block& block,
                                    mlir::PatternRewriter& rewriter) {
    rewriter.setInsertionPointToStart(&block);
    for (mlir::BlockArgument arg : block.getArguments()) {
      auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
      if (!tensor_type) {
        continue;
      }

      arg.setType(tensor_type.getElementType());
      auto from_elements = mlir::tensor::FromElementsOp::create(
          rewriter, arg.getLoc(), tensor_type, arg);
      rewriter.replaceAllUsesExcept(arg, from_elements, from_elements);
    }
  }

  static void RewriteRegionOps(mlir::Block& block,
                               mlir::PatternRewriter& rewriter) {
    for (mlir::Operation& nested_op : block.without_terminator()) {
      rewriter.modifyOpInPlace(&nested_op, [&]() {
        for (mlir::Value result : nested_op.getResults()) {
          if (auto tensor_type =
                  mlir::dyn_cast<mlir::RankedTensorType>(result.getType())) {
            result.setType(
                mlir::RankedTensorType::get({}, tensor_type.getElementType()));
          }
        }
      });
    }
  }

  static void RewriteTerminator(mlir::Block& block,
                                mlir::PatternRewriter& rewriter) {
    mlir::Operation* terminator = block.getTerminator();
    rewriter.setInsertionPoint(terminator);
    llvm::SmallVector<mlir::Value> operands;
    operands.reserve(terminator->getNumOperands());
    for (mlir::Value operand : terminator->getOperands()) {
      auto tensor_type =
          mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
      if (tensor_type) {
        operand = mlir::tensor::ExtractOp::create(
            rewriter, terminator->getLoc(), tensor_type.getElementType(),
            operand, /*indices=*/{});
      }
      operands.push_back(operand);
    }
    rewriter.replaceOpWithNewOp<::xla::xtile::YieldOp>(terminator, operands);
  }
};

class XTileScalarizeScanPass
    : public impl::XTileScalarizeScanPassBase<XTileScalarizeScanPass> {
 public:
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet scalarize_patterns(mlir_context);
    scalarize_patterns.add<ScalarizeScanRegionPattern>(mlir_context);
    mlir::walkAndApplyPatterns(getOperation(), std::move(scalarize_patterns));
  }
};

}  // namespace
}  // namespace mlir::triton::xla
