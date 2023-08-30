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
#include <string>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/path.h"

namespace mlir {
namespace quant {
namespace {

constexpr StringRef kCompositeFuncPrefix = "composite_";

// AddDumpTensorOp pass adds DumpTensorOp - which saves entire value of its
// input into a file - to quantizable layer's output.
class AddDumpTensorOpPass
    : public PassWrapper<AddDumpTensorOpPass, OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddDumpTensorOpPass)

  explicit AddDumpTensorOpPass() = default;

  explicit AddDumpTensorOpPass(std::string log_dir_path)
      : log_dir_path_(std::move(log_dir_path)) {}

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in the textual format (on
    // the commandline for example).
    return "quant-add-dump-tensor-op";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Add DumpTensor ops after quantizable ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TF::TensorFlowDialect>();
    registry.insert<quant::QuantizationDialect>();
    registry.insert<quantfork::QuantizationForkDialect>();
  }

 private:
  void runOnOperation() override;

  std::string log_dir_path_ = "/tmp/dumps";
};

class AddDumpTensorOp : public OpRewritePattern<TF::PartitionedCallOp> {
 public:
  // Does not take ownership of context, which must refer to a valid value that
  // outlives this object.
  explicit AddDumpTensorOp(MLIRContext *context, std::string log_dir_path)
      : OpRewritePattern(context), log_dir_path_(std::move(log_dir_path)) {}

 private:
  std::string log_dir_path_;

  LogicalResult matchAndRewrite(TF::PartitionedCallOp call_op,
                                PatternRewriter &rewriter) const override {
    const auto f_attr = call_op.getFAttr().dyn_cast<FlatSymbolRefAttr>();
    if (!call_op->hasAttr(kQuantTraitAttrName)) {
      return failure();
    }
    if (!f_attr.getValue().startswith(kCompositeFuncPrefix)) {
      return failure();
    }

    // For now, only support ops with 1 results
    if (call_op->getNumResults() != 1) return failure();

    Value result = call_op->getResult(0);

    // If one of the user is DumpTensorOp, do nothing
    for (auto user : result.getUsers()) {
      if (dyn_cast_or_null<TF::DumpTensorOp>(user)) return failure();
    }

    rewriter.setInsertionPointAfterValue(result);

    auto folder_name =
        tensorflow::io::JoinPath(log_dir_path_, f_attr.getValue());
    SmallVector<NamedAttribute> dump_attributes{
        rewriter.getNamedAttr("log_dir_path",
                              rewriter.getStringAttr(folder_name)),
        // The file_name will be changed from unquantized_tensor_data.pb to
        // quantized_tensor_data.pb after the calibration.
        rewriter.getNamedAttr(
            "file_name", rewriter.getStringAttr("unquantized_tensor_data.pb")),
        // The op is disabled by default. Otherwise, values will be saved
        // during calibration.
        rewriter.getNamedAttr("enabled", rewriter.getBoolAttr(false)),
    };

    rewriter.create<TF::DumpTensorOp>(call_op->getLoc(), TypeRange{}, result,
                                      dump_attributes);

    return success();
  }
};

static PassRegistration<AddDumpTensorOpPass> pass;

void AddDumpTensorOpPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();

  patterns.add<AddDumpTensorOp>(ctx, log_dir_path_);
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    func.emitError() << "quant-add-dump-tensor-op failed.";
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateAddDumpTensorOpPass(
    std::string log_dir_path) {
  return std::make_unique<AddDumpTensorOpPass>(std::move(log_dir_path));
}

}  // namespace quant
}  // namespace mlir
