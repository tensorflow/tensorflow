/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TF {

namespace {

/// Convert the `tf.IfRegion` op to the `scf.if` op.
class ConvertIfRegionOp : public OpRewritePattern<IfRegionOp> {
 public:
  using OpRewritePattern<IfRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfRegionOp op,
                                PatternRewriter& rewriter) const override {
    // Creates the `then` or `else` region of the `scf.if` op. Note that
    // `tf_then_or_else_region` is the `then` or `else` region of the
    // `tf.IfRegion` op and `scf_then_or_else_region` is the `then` or `else`
    // region of the new `scf.if` op. Further, `tf_if_region_return_type` is the
    // list of return types of the `tf.IfRegion` op.
    auto createScfThenOrElse = [](Region& tf_then_or_else_region,
                                  Region& scf_then_or_else_region,
                                  TypeRange tf_if_region_return_type,
                                  PatternRewriter& rewriter) {
      // Clone all the ops of `tf_then_or_else_region` into
      // `scf_then_or_else_region`.
      rewriter.cloneRegionBefore(tf_then_or_else_region,
                                 &scf_then_or_else_region.front());
      rewriter.eraseBlock(&scf_then_or_else_region.back());

      Block* first_block_of_scf_then_or_else_region =
          &scf_then_or_else_region.front();

      // Replace the current terminator (a `tf.Yield` op) with an `scf.yield`
      // op. The input of the `scf.yield` op is a list of results of `tf.Cast`
      // ops, each of which casts an operand of the current terminator to the
      // corresponding result type of the `tf.IfRegion` op.
      Operation* current_terminator =
          first_block_of_scf_then_or_else_region->getTerminator();
      rewriter.setInsertionPoint(current_terminator);
      SmallVector<Value, 4> scf_yield_input;
      for (auto it : llvm::zip(tf_if_region_return_type,
                               current_terminator->getOperands())) {
        scf_yield_input.push_back(rewriter.create<CastOp>(
            current_terminator->getLoc(), std::get<0>(it), std::get<1>(it)));
      }

      rewriter.replaceOpWithNewOp<scf::YieldOp>(current_terminator,
                                                scf_yield_input);
    };

    Location loc = op.getLoc();

    // The condition of an `scf.if` op is a 1-bit signless integer. Whereas, the
    // condition of the `tf.IfRegion` op is a 0-D tensor of 1-bit signless
    // integers. Thus, we use the `tensor.extract` op to compute the condition
    // of `scf.if` from that of `tf.IfRegion`.
    auto scf_if_condition = rewriter.create<tensor::ExtractOp>(loc, op.cond());

    TypeRange tf_if_region_return_type = op.getResultTypes();

    // Create the `scf.if` op.
    auto scf_if_op =
        rewriter.create<scf::IfOp>(loc, tf_if_region_return_type,
                                   scf_if_condition, /*withElseRegion=*/true);

    Region& then_region = op.then_branch();
    Region& else_region = op.else_branch();

    // Create the `then` and `else` regions of the `scf.if` op.
    createScfThenOrElse(then_region, scf_if_op.thenRegion(),
                        tf_if_region_return_type, rewriter);
    createScfThenOrElse(else_region, scf_if_op.elseRegion(),
                        tf_if_region_return_type, rewriter);

    // Replace the `tf.IfRegion` op with the results of the `scf.if` op.
    rewriter.replaceOp(op, scf_if_op.getResults());
    return success();
  }
};

}  // end anonymous namespace

void populateTfControlFlowToScfPatterns(MLIRContext* context,
                                        OwningRewritePatternList* patterns) {
  patterns->insert<ConvertIfRegionOp>(context);
}

struct ConvertTfControlFlowToScf
    : public ConvertTfControlFlowToScfPassBase<ConvertTfControlFlowToScf> {
  void runOnOperation() override {
    OwningRewritePatternList patterns(&getContext());
    populateTfControlFlowToScfPatterns(&getContext(), &patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createConvertTfControlFlowToScfPass() {
  return std::make_unique<ConvertTfControlFlowToScf>();
}

}  // namespace TF
}  // end namespace mlir
