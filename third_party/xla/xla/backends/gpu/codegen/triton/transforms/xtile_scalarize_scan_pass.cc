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

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_XTILESCALARIZESCANPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Changes the scan region to have scalar arguments instead of tensor arguments.
class ScalarizeScanRegionPattern
    : public mlir::OpRewritePattern<::xla::xtile::ScanOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ::xla::xtile::ScanOp op, mlir::PatternRewriter& rewriter) const override {
    if (op.getRegion().empty() ||
        op.getRegion().front().getNumArguments() == 0) {
      return mlir::failure();
    }

    mlir::Block& block = op.getRegion().front();
    bool has_tensor_arg = false;
    for (auto arg : block.getArguments()) {
      if (mlir::isa<mlir::RankedTensorType>(arg.getType())) {
        has_tensor_arg = true;
        break;
      }
    }
    if (!has_tensor_arg) {
      return mlir::failure();
    }

    auto new_scan_op = ::xla::xtile::ScanOp::create(
        rewriter, op.getLoc(), op.getOutputs().getTypes(),
        op.getCarries().getTypes(), op.getInputs(), op.getInits(),
        op.getDimension(), op.getScanDimSize(), op.getIsReverse());

    rewriter.inlineRegionBefore(op.getRegion(), new_scan_op.getRegion(),
                                new_scan_op.getRegion().end());

    mlir::Block& new_block = new_scan_op.getRegion().front();
    int num_args = new_block.getNumArguments();
    rewriter.setInsertionPointToStart(&new_block);

    for (int i = 0; i < num_args; ++i) {
      mlir::BlockArgument old_arg = new_block.getArgument(i);
      auto tensor_type =
          mlir::dyn_cast<mlir::RankedTensorType>(old_arg.getType());
      if (!tensor_type) {
        new_block.addArgument(old_arg.getType(), old_arg.getLoc());
        continue;
      }

      mlir::Type element_type = tensor_type.getElementType();
      mlir::BlockArgument new_arg =
          new_block.addArgument(element_type, old_arg.getLoc());

      auto single_elem_type = mlir::RankedTensorType::get({}, element_type);
      mlir::Value from_elements = mlir::tensor::FromElementsOp::create(
          rewriter, old_arg.getLoc(), single_elem_type,
          mlir::ValueRange{new_arg});
      rewriter.replaceAllUsesWith(old_arg, from_elements);
    }

    for (int i = 0; i < num_args; ++i) {
      new_block.eraseArgument(0);
    }

    for (auto& op_ref :
         llvm::make_early_inc_range(new_block.without_terminator())) {
      mlir::Operation* op = &op_ref;
      if (op->hasTrait<mlir::OpTrait::Elementwise>()) {
        rewriter.setInsertionPoint(op);
        llvm::SmallVector<mlir::Type> new_result_types;
        for (mlir::Type res_type : op->getResultTypes()) {
          if (auto t_type = mlir::dyn_cast<mlir::RankedTensorType>(res_type)) {
            new_result_types.push_back(
                mlir::RankedTensorType::get({}, t_type.getElementType()));
          } else {
            new_result_types.push_back(res_type);
          }
        }
        mlir::OperationState state(op->getLoc(), op->getName());
        state.addOperands(op->getOperands());
        state.addTypes(new_result_types);
        state.addAttributes(op->getAttrs());
        mlir::Operation* new_op = rewriter.create(state);
        rewriter.replaceOp(op, new_op->getResults());
      }
    }

    mlir::Operation* terminator = new_block.getTerminator();
    rewriter.setInsertionPoint(terminator);
    llvm::SmallVector<mlir::Value> new_returns;
    for (mlir::Value ret_val : terminator->getOperands()) {
      auto tensor_type =
          mlir::dyn_cast<mlir::RankedTensorType>(ret_val.getType());
      if (tensor_type) {
        mlir::Value extracted = mlir::tensor::ExtractOp::create(
            rewriter, terminator->getLoc(), tensor_type.getElementType(),
            ret_val, mlir::ValueRange{});
        new_returns.push_back(extracted);
      } else {
        new_returns.push_back(ret_val);
      }
    }
    ::xla::xtile::YieldOp::create(rewriter, terminator->getLoc(), new_returns);
    rewriter.eraseOp(terminator);

    rewriter.replaceOp(op, new_scan_op.getResults());
    return mlir::success();
  }
};

class XTileScalarizeScanPass
    : public impl::XTileScalarizeScanPassBase<XTileScalarizeScanPass> {
 public:
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet scalarize_patterns(mlir_context);
    scalarize_patterns.add<ScalarizeScanRegionPattern>(mlir_context);
    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(scalarize_patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::triton::xla
