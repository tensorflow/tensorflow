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
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/quantization_unit_loc.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/path.h"

namespace mlir {
namespace quant {
namespace {

using DebuggerType = tensorflow::quantization::DebuggerOptions::DebuggerType;
using DebuggerOptions = tensorflow::quantization::DebuggerOptions;

constexpr StringRef kCompositeFuncPrefix = "composite_";

// AddDumpTensorOp pass adds DumpTensorOp - which saves entire value of its
// input into a file - to quantizable layer's output.
class AddDumpTensorOpPass
    : public PassWrapper<AddDumpTensorOpPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddDumpTensorOpPass)

  explicit AddDumpTensorOpPass() = default;

  explicit AddDumpTensorOpPass(DebuggerType debugger_type,
                               std::string log_dir_path)
      : log_dir_path_(std::move(log_dir_path)) {
    debugger_type_ = debugger_type;
  }

  AddDumpTensorOpPass(const AddDumpTensorOpPass &other) {
    debugger_type_ = other.debugger_type_;
    log_dir_path_ = other.log_dir_path_;
  }

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

  Option<DebuggerType> debugger_type_{
      *this, "debugger_type",
      llvm::cl::init(DebuggerOptions::DEBUGGER_TYPE_UNSPECIFIED),
      llvm::cl::values(clEnumValN(DebuggerOptions::DEBUGGER_TYPE_WHOLE_MODEL,
                                  "whole_model", "Whole model verify"),
                       clEnumValN(DebuggerOptions::DEBUGGER_TYPE_PER_LAYER,
                                  "per_layer", "Per-layer verify"))};

  std::string log_dir_path_ = "/tmp/dumps";
};

class AddDumpTensorOp : public OpRewritePattern<TF::PartitionedCallOp> {
 public:
  // Does not take ownership of context, which must refer to a valid value that
  // outlives this object.
  explicit AddDumpTensorOp(MLIRContext *context, DebuggerType debugger_type,
                           std::string log_dir_path)
      : OpRewritePattern(context),
        debugger_type_(debugger_type),
        log_dir_path_(std::move(log_dir_path)) {}

 private:
  DebuggerType debugger_type_;
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

    std::optional<QuantizationUnitLoc::QuantizationUnit> quant_unit =
        FindQuantizationUnitFromLoc(call_op->getLoc());

    if (!quant_unit.has_value()) return failure();

    auto folder_name =
        tensorflow::io::JoinPath(log_dir_path_, f_attr.getValue());
    // In Whole model, we first need to set file_name as
    // unquantized_tensor_data.pb as it is used by unquantized dump model.
    // After saving unquantized dump model, the file name will be changed to
    // quantized_tensor_data.pb.
    // Since this process doesn't happen for per layer, we need to set file_name
    // as quantized_tensor_data.pb here.
    // TODO: b/296933893 - Refactor the debugger code when no quantize option
    // is added
    auto file_name = debugger_type_ == DebuggerOptions::DEBUGGER_TYPE_PER_LAYER
                         ? "quantized_tensor_data.pb"
                         : "unquantized_tensor_data.pb";

    SmallVector<NamedAttribute> dump_attributes{
        rewriter.getNamedAttr("log_dir_path",
                              rewriter.getStringAttr(folder_name)),
        rewriter.getNamedAttr("file_name", rewriter.getStringAttr(file_name)),
        // The op is disabled by default. Otherwise, values will be saved
        // during calibration.
        rewriter.getNamedAttr("enabled", rewriter.getBoolAttr(false)),
        rewriter.getNamedAttr("func_name",
                              rewriter.getStringAttr(quant_unit->func_name())),
        rewriter.getNamedAttr("node_name",
                              rewriter.getStringAttr(quant_unit->node_name())),
    };

    rewriter.create<TF::DumpTensorOp>(call_op->getLoc(), TypeRange{}, result,
                                      dump_attributes);

    // Per-layer mode.
    if (debugger_type_ == DebuggerOptions::DEBUGGER_TYPE_PER_LAYER) {
      auto module = call_op->getParentOfType<ModuleOp>();
      SymbolTable symbol_table(module);

      // Copy composite function of quantizable layer.
      const mlir::func::FuncOp ref_func = dyn_cast_or_null<func::FuncOp>(
          symbol_table.lookup(f_attr.getValue()));
      mlir::func::FuncOp new_ref_func =
          dyn_cast<func::FuncOp>(ref_func->clone());
      const StringAttr new_ref_func_name = symbol_table.insert(new_ref_func);

      // Create PartitionedCallOp to the copied composite function.
      // This PartitionedCallOp does not have kQuantTraitAttrName, and therefore
      // won't get quantized.
      auto ref_call_op = rewriter.create<TF::PartitionedCallOp>(
          call_op.getLoc(), call_op.getResultTypes(), call_op.getOperands(),
          FlatSymbolRefAttr::get(new_ref_func_name));

      // Attach DumpTensorOp to its output unquantized layer.
      SmallVector<NamedAttribute> dump_attributes{
          rewriter.getNamedAttr("log_dir_path",
                                rewriter.getStringAttr(folder_name)),
          rewriter.getNamedAttr("file_name", rewriter.getStringAttr(
                                                 "unquantized_tensor_data.pb")),
          rewriter.getNamedAttr("enabled", rewriter.getBoolAttr(false)),
          rewriter.getNamedAttr(
              "func_name", rewriter.getStringAttr(quant_unit->func_name())),
          rewriter.getNamedAttr(
              "node_name", rewriter.getStringAttr(quant_unit->node_name())),
      };

      rewriter.create<TF::DumpTensorOp>(call_op->getLoc(), TypeRange{},
                                        ref_call_op.getResult(0),
                                        dump_attributes);
    }

    return success();
  }
};

static PassRegistration<AddDumpTensorOpPass> pass;

void AddDumpTensorOpPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ModuleOp module = getOperation();

  patterns.add<AddDumpTensorOp>(ctx, debugger_type_, log_dir_path_);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
    module.emitError() << "quant-add-dump-tensor-op failed.";
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateAddDumpTensorOpPass(
    DebuggerType debugger_type, std::string log_dir_path) {
  return std::make_unique<AddDumpTensorOpPass>(debugger_type,
                                               std::move(log_dir_path));
}

}  // namespace quant
}  // namespace mlir
