/* Copyright 2021 Google Inc. All Rights Reserved.

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

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Reducer/ReductionPatternInterface.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Tools/mlir-reduce/MlirReduceMain.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/register_common_dialects.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace {

struct CollapseAndErase : public mlir::OpRewritePattern<mlir::TF::IdentityOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(
      mlir::TF::IdentityOp op, mlir::PatternRewriter& rewriter) const override {
    auto arg = op.getInput().getDefiningOp<mlir::TF::IdentityOp>();
    if (!arg) return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::TF::IdentityOp>(op, op.getType(),
                                                      arg.getInput());
    if (arg.use_empty()) {
      rewriter.eraseOp(arg);
    }
    return mlir::success();
  }
};

struct TFReductionPatternInterface
    : public mlir::DialectReductionPatternInterface {
 public:
  explicit TFReductionPatternInterface(mlir::Dialect *dialect)
      : DialectReductionPatternInterface(dialect) {}

  void populateReductionPatterns(
      mlir::RewritePatternSet &patterns) const final {
    patterns.add<CollapseAndErase>(patterns.getContext());
  }
};

}  // namespace

int main(int argc, char *argv[]) {
  tensorflow::InitMlir y(&argc, &argv);

  mlir::DialectRegistry registry;
  mlir::RegisterCommonToolingDialects(registry);

  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::TF::TensorFlowDialect *dialect) {
        dialect->addInterfaces<TFReductionPatternInterface>();
      });

  mlir::MLIRContext context(registry);

  return failed(mlirReduceMain(argc, argv, context));
}
