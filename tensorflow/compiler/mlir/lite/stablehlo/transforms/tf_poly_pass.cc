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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/tf_poly_pass.h"

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/mhlo_util.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/tf_mhlo_pass.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/register.h"

namespace mlir {
namespace TFL {
namespace mhlo {

static bool isTFOp(Operation *op) {
  return op->getDialect()->getNamespace() == "tf";
}

class TFPolyPass
    : public mlir::PassWrapper<TFPolyPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 private:
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::mhlo::registerAllMhloDialects(registry);
    registry.insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                    mlir::TFL::TensorFlowLiteDialect>();
  }

  // Options that control what transfomraitons are applied.

  PolyCallOptions options_;

  // A transformation to be applied to TF ops. The transformation is specified
  // by patterns and target.
  struct Tranformation {
    std::string name;
    FrozenRewritePatternSet patterns;  // The patterns to be applied.
    ConversionTarget target;           // The target of the transformation.
  };

  // Creates a list of transformaiton to be applied to TF ops.
  // The transformations apply on replicated list of TF ops so there is no
  // interaction between the transformations.
  std::vector<Tranformation> LoadTransformations(PolyCallOptions options,
                                                 MLIRContext *context) {
    std::vector<Tranformation> transformations;
    // Optionally add TF to MHLO pass.
    if (options.enable_tf_mhlo_conversion) {
      RewritePatternSet patterns(context);
      PopulateTFToMhloPatterns(
          context, /*legalize_chlo=*/true,
          /*tf2xla_fallback_device_type=*/llvm::StringRef("DEFAULT"),
          /*prefer_tf2xla=*/false, &patterns);

      ConversionTarget target(*context);
      target.addLegalDialect<arith::ArithmeticDialect>();
      target.addLegalDialect<func::FuncDialect>();
      target.addLegalDialect<::mlir::mhlo::MhloDialect>();

      FrozenRewritePatternSet frozen_patterns(std::move(patterns));
      Tranformation tf_mhlo_transform = {"tf_mhlo", frozen_patterns, target};
      transformations.push_back(tf_mhlo_transform);
    }
    return transformations;
  }

  // Copies an `op` and put the copy into `region`, and return the copied op.
  Operation *CopyTfAndCreateRegion(OpBuilder *builder, Operation *op,
                                   Region *region) {
    Block *block = new Block;
    region->push_back(block);
    builder->setInsertionPointToEnd(&region->front());
    Operation *tf_op = builder->clone(*op);
    Location loc = op->getLoc();
    block->addArguments(op->getOperandTypes(),
                        SmallVector<Location>(op->getNumOperands(), loc));
    for (auto &idx_args : llvm::enumerate(block->getArguments())) {
      tf_op->setOperand(idx_args.index(), idx_args.value());
    }
    builder->create<YieldOp>(loc, tf_op->getResults());
    return tf_op;
  }

 public:
  StringRef getArgument() const final { return "tf-poly"; }
  StringRef getDescription() const final {
    return "This pass will legalize TF ops to poly call.";
  }
  explicit TFPolyPass(PolyCallOptions options) { options_ = options; }
};

void TFPolyPass::runOnOperation() {
  func::FuncOp fn = getOperation();
  MLIRContext *context = fn->getContext();
  const std::vector<Tranformation> transformations =
      LoadTransformations(options_, context);
  const int num_transformation = transformations.size();
  std::vector<std::vector<Operation *>> to_transform(num_transformation);
  fn.walk([&](Operation *op) {
    // Process only TF ops.
    if (!isTFOp(op)) return;

    // Create polycall op. Need to call setInsertionPoint to avoid recurrsion.
    OpBuilder builder(op->getContext());
    builder.setInsertionPoint(op);
    auto poly_op = builder.create<TFL::PolyCallOp>(
        op->getLoc(), op->getResultTypes(), op->getOperands(),
        num_transformation + 1);
    poly_op->setAttrs(op->getAttrs());

    // Create TF region.
    Region tf_region;
    (void)CopyTfAndCreateRegion(&builder, op, &tf_region);
    poly_op.calls()
        .take_back(num_transformation + 1)
        .data()
        ->takeBody(tf_region);

    // Create regions according to the transformations.
    for (int i = 0; i < num_transformation; i++) {
      Region region;
      to_transform[i].push_back(CopyTfAndCreateRegion(&builder, op, &region));
      poly_op.calls().take_back(i + 1).data()->takeBody(region);
    }

    // Replace original func with polycall.
    op->replaceAllUsesWith(poly_op);
    op->erase();
  });

  // Apply transformations.
  for (int i = 0; i < num_transformation; i++) {
    auto transformation = transformations[i];
    auto op_to_transform = to_transform[i];
    if (failed(applyPartialConversion(op_to_transform, transformation.target,
                                      transformation.patterns))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateTFPolyPass(
    PolyCallOptions options) {
  return std::make_unique<TFPolyPass>(options);
}

static PassRegistration<TFPolyPass> pass([] {
  PolyCallOptions options;
  options.enable_tf_mhlo_conversion = true;
  return CreateTFPolyPass(options);
});

}  // namespace mhlo
}  // namespace TFL
}  // namespace mlir
