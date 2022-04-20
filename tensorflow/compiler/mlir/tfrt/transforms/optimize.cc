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
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

// Fold tf.DeviceIndex to tf.Const if it has device assigned.
class FoldDeviceIndex : public mlir::OpRewritePattern<mlir::TF::DeviceIndexOp> {
 public:
  using mlir::OpRewritePattern<mlir::TF::DeviceIndexOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::DeviceIndexOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto device = op->getAttrOfType<mlir::StringAttr>("device");
    if (!device) return mlir::failure();

    DeviceNameUtils::ParsedName parsed_name;
    if (!DeviceNameUtils::ParseFullName(device.getValue().str(),
                                        &parsed_name) ||
        !parsed_name.has_type)
      return mlir::failure();

    int32_t i = 0;
    mlir::ArrayAttr device_names = op.device_names();
    for (; i < device_names.size(); ++i) {
      auto device_name = device_names[i].cast<mlir::StringAttr>().getValue();
      if (device_name == parsed_name.type) break;
    }

    rewriter.replaceOpWithNewOp<mlir::TF::ConstOp>(
        op,
        mlir::DenseIntElementsAttr::get(
            mlir::RankedTensorType::get(/*shape=*/{}, rewriter.getI32Type()),
            i));

    return mlir::success();
  }
};

// Optimization pass for TFRT-specific rewrite patterns.
class OptimizeTfForTfrt
    : public mlir::PassWrapper<OptimizeTfForTfrt,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeTfForTfrt)

  llvm::StringRef getArgument() const final { return "optimize-tf-for-tfrt"; }

  llvm::StringRef getDescription() const final {
    return "optmize TF MLIR for TFRT workflow.";
  }

  mlir::LogicalResult initialize(mlir::MLIRContext* context) override {
    mlir::RewritePatternSet pattern_list(context);
    pattern_list.add<FoldDeviceIndex>(context);
    patterns_ = std::move(pattern_list);
    return mlir::success();
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, patterns_)))
      signalPassFailure();
  }

 private:
  mlir::FrozenRewritePatternSet patterns_;
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateOptimizeTfForTfrtPass() {
  return std::make_unique<OptimizeTfForTfrt>();
}

static mlir::PassRegistration<OptimizeTfForTfrt> register_pass;

}  // namespace tfrt_compiler
}  // namespace tensorflow
