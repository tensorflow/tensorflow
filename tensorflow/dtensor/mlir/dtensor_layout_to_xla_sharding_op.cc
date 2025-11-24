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

#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/dtensor/cc/xla_spmd/layout_to_xla_sharding.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

namespace tensorflow {
namespace dtensor {
namespace {

#define GEN_PASS_DECL_DTENSORLAYOUTTOXLASHARDINGOPPASS
#define GEN_PASS_DEF_DTENSORLAYOUTTOXLASHARDINGOPPASS
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

using mlir::TF::DTensorLayout;

class RemoveDTensorLayoutAfterConstOrBlockArgPattern
    : public mlir::OpRewritePattern<DTensorLayout> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      DTensorLayout layout_op, mlir::PatternRewriter& rewriter) const override {
    if (match(layout_op).failed()) {
      return mlir::failure();
    }
    rewriter.replaceAllUsesWith(layout_op, layout_op.getInput());
    rewriter.eraseOp(layout_op);
    return mlir::success();
  }

 private:
  mlir::LogicalResult match(DTensorLayout layout_op) const {
    auto input = layout_op.getInput();
    if (mlir::isa<mlir::BlockArgument>(input)) {
      return mlir::success();
    }
    mlir::Operation* input_op = input.getDefiningOp();
    if (input_op != nullptr) {
      return mlir::success(input_op->hasTrait<mlir::OpTrait::ConstantLike>());
    } else {
      return layout_op->emitOpError() << "Can't find defining op for " << input;
    }
  }
};

class DTensorLayoutToXlaShardingOpPass
    : public impl::DTensorLayoutToXlaShardingOpPassBase<
          DTensorLayoutToXlaShardingOpPass> {
 public:
  void runOnOperation() override;
};

void DTensorLayoutToXlaShardingOpPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  // Some patterns in tf2xla requires operands to be ConstantLike.
  // Inserting tf.XlaSharding between them will fail the pattern match.
  // We remove all tf.DTensorLayout after constants so no tf.XlaSharding is
  // inserted in the above case. XLA will figure out the sharding of constants
  // without DTensor guidance.
  //
  // For BlockArgument, the sharding is already attached to function attribute
  // by DTensorSetHloShardingPass. No additional tf.XlaSharding is needed.
  patterns.add<RemoveDTensorLayoutAfterConstOrBlockArgPattern>(&getContext());
  if (mlir::failed(
          mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }

  auto result =
      getOperation().walk([](DTensorLayout layout_op) -> mlir::WalkResult {
        Layout layout = layout_op.getLayout();
        StatusOr<xla::OpSharding> sharding =
            ConvertLayoutToXlaOpSharding(layout);
        if (!sharding.ok()) {
          return layout_op.emitOpError()
                 << "Failed to convert layout to sharding for "
                 << layout.ToString() << ": " << sharding.status().message();
        }
        mlir::OpBuilder builder(layout_op);
        auto sharding_attr =
            builder.getStringAttr(sharding->SerializeAsString());
        // TODO(b/414807890): It seems that the dtensor path later on clear up
        // the V1 sharding attr, so set V2 sharding to "" here. It may be better
        // to set the V2 sharding attr here and then removed it when V1 is
        // removed.
        auto sharding_op = builder.create<mlir::TF::XlaShardingOp>(
            layout_op.getLoc(), layout_op.getOutput().getType(),
            layout_op.getInput(),
            /*sharding=*/builder.getStringAttr(""),  // Not used by tf2xla.
            /*_xlaSharding=*/sharding_attr,
            /*_xla_sharding_v2=*/builder.getStringAttr(""));
        layout_op.getOutput().replaceAllUsesWith(sharding_op);
        layout_op.erase();
        return mlir::WalkResult::advance();
      });

  if (result.wasInterrupted()) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorLayoutToXlaShardingOpPass() {
  return std::make_unique<DTensorLayoutToXlaShardingOpPass>();
}

}  // namespace dtensor
}  // namespace tensorflow
