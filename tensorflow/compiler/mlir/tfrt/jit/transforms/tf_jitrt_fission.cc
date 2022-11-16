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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_DEF_FISSION
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

struct FusedMatMulFission
    : public mlir::OpRewritePattern<mlir::TF::_FusedMatMulOp> {
  using OpRewritePattern<mlir::TF::_FusedMatMulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::_FusedMatMulOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto type = op.getResult().getType();

    size_t n = op.getFusedOps().size();

    // Extract fused operations from the operation attributes.
    mlir::StringAttr fusion0 =
        n > 0 ? op.getFusedOps()[0].dyn_cast<mlir::StringAttr>() : nullptr;
    mlir::StringAttr fusion1 =
        n > 1 ? op.getFusedOps()[1].dyn_cast<mlir::StringAttr>() : nullptr;

    // Match to supported operations
    bool is_bias_add = fusion0 && fusion0.getValue() == "BiasAdd";
    bool is_relu_activation = fusion1 && fusion1.getValue() == "Relu";

    // Create a simple MatMul operation from the fused one.
    auto matmul = [&]() -> mlir::TF::MatMulOp {
      auto lhs = op.getOperand(0);
      auto rhs = op.getOperand(1);
      return rewriter.create<mlir::TF::MatMulOp>(
          loc, type, lhs, rhs, op.getTransposeA(), op.getTransposeB());
    };

    // FusedMatMul[BiasAdd].
    if (n == 1 && is_bias_add) {
      rewriter.replaceOpWithNewOp<mlir::TF::BiasAddOp>(op, type, matmul(),
                                                       op.getOperand(2));
      return mlir::success();
    }

    // FusedMatMul[BiasAdd, Relu].
    if (n == 2 && is_bias_add && is_relu_activation) {
      auto biased = rewriter.create<mlir::TF::BiasAddOp>(loc, type, matmul(),
                                                         op.getOperand(2));
      rewriter.replaceOpWithNewOp<mlir::TF::ReluOp>(op, type, biased);
      return mlir::success();
    }

    return mlir::failure();
  }
};

}  // namespace

// -------------------------------------------------------------------------- //
// Break Tensorflow _Fused{Op} operations into primitive ones.
// -------------------------------------------------------------------------- //
struct FissionPass : public impl::FissionBase<FissionPass> {
  void runOnOperation() override {
    mlir::func::FuncOp function = getOperation();
    mlir::MLIRContext* ctx = function.getContext();

    mlir::RewritePatternSet patterns(ctx);
    patterns.add<FusedMatMulFission>(ctx);

    (void)mlir::applyPatternsAndFoldGreedily(function, std::move(patterns));
  }
};

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateFissionPass() {
  return std::make_unique<FissionPass>();
}

}  // namespace tensorflow
