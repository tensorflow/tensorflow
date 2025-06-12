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
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/common/tf_attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/quantization_unit_loc.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_call_module_attrs.h"
#include "tensorflow/core/platform/path.h"

namespace mlir {
namespace quant {
namespace {

using ::stablehlo::quantization::DebuggerConfig;
using DebuggerType = DebuggerConfig::DebuggerType;
using ::mlir::tf_quant::GetFuncAttr;

constexpr StringRef kOriginalEntryFuncAttrName = "_original_entry_function";
constexpr StringRef kCompositeFuncPrefix = "composite_";
constexpr StringRef kEmptyNodeName = "_empty_node";

// Returns a pair: `func_name` and `node_name` for the lifted function. In TF
// quantizer, both are filled. For StableHLO quantizer, the func_name is only
// filled and node_name is always set to "_empty_node".
std::pair<std::string, std::string> GetFuncNameAndNodeName(
    TF::PartitionedCallOp call_op, const FlatSymbolRefAttr &f_attr) {
  std::optional<QuantizationUnitLoc::QuantizationUnit> quant_unit =
      FindQuantizationUnitFromLoc(call_op->getLoc());
  return std::make_pair(quant_unit->func_name(), quant_unit->node_name());
}

std::pair<std::string, std::string> GetFuncNameAndNodeName(
    TF::XlaCallModuleOp call_op, const FlatSymbolRefAttr &f_attr) {
  return std::make_pair(f_attr.getValue().str(), kEmptyNodeName.str());
}

Operation *DuplicateOp(TF::PartitionedCallOp call_op, PatternRewriter &rewriter,
                       const StringAttr &new_ref_func_name) {
  // Create PartitionedCallOp to the copied composite function. This
  // PartitionedCallOp does not have kQuantTraitAttrName, and therefore won't
  // get quantized.
  auto new_call_op = rewriter.create<TF::PartitionedCallOp>(
      call_op.getLoc(), call_op.getResultTypes(), call_op.getOperands(),
      call_op.getArgAttrsAttr(), call_op.getResAttrsAttr(),
      FlatSymbolRefAttr::get(new_ref_func_name));
  return new_call_op;
}

Operation *DuplicateOp(TF::XlaCallModuleOp call_op, PatternRewriter &rewriter,
                       const StringAttr &new_ref_func_name) {
  // Create XlaCallModuleOp to the copied composite function. This
  // XlaCallModuleOp does not have kQuantTraitAttrName, and therefore won't get
  // quantized.
  auto new_call_op = rewriter.create<TF::XlaCallModuleOp>(
      call_op.getLoc(), call_op.getResultTypes(), call_op.getOperands(),
      call_op.getVersionAttr(), call_op.getModuleAttr(), call_op.getSoutAttr());
  new_call_op->setAttr(TF::kStablehloEntryFunctionAttrName,
                       rewriter.getStringAttr(new_ref_func_name.getValue()));
  new_call_op->setAttrs(call_op->getAttrs());
  new_call_op->setAttr(TF::kStablehloVersionAttrName,
                       call_op->getAttr(TF::kStablehloVersionAttrName));
  new_call_op->removeAttr(rewriter.getStringAttr(kQuantTraitAttrName));

  FlatSymbolRefAttr new_func_name_attr =
      FlatSymbolRefAttr::get(rewriter.getContext(), new_ref_func_name);
  new_call_op->setAttr(TF::kStablehloEntryFunctionAttrName, new_func_name_attr);
  new_call_op->setAttr(kOriginalEntryFuncAttrName, new_ref_func_name);
  return new_call_op;
}

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
    registry.insert<quant::QuantDialect>();
    registry.insert<mlir::quant::ir::TFQuantDialect>();
  }

 private:
  void runOnOperation() override;

  Option<DebuggerType> debugger_type_{
      *this, "debugger_type",
      llvm::cl::init(DebuggerConfig::DEBUGGER_TYPE_UNSPECIFIED),
      llvm::cl::values(
          clEnumValN(DebuggerConfig::DEBUGGER_TYPE_WHOLE_MODEL, "whole_model",
                     "Whole model verify"),
          clEnumValN(DebuggerConfig::DEBUGGER_TYPE_INT_PER_LAYER,
                     "int_per_layer", "Int Per-layer verify"),
          clEnumValN(DebuggerConfig::DEBUGGER_TYPE_FLOAT_PER_LAYER,
                     "float_per_layer", "Float Per-layer verify"))};

  std::string log_dir_path_ = "/tmp/dumps";
};

template <typename LiftedOpT>
class AddDumpTensorOp : public OpRewritePattern<LiftedOpT> {
 public:
  // Does not take ownership of context, which must refer to a valid value that
  // outlives this object.
  explicit AddDumpTensorOp(MLIRContext *context, DebuggerType debugger_type,
                           std::string log_dir_path)
      : OpRewritePattern<LiftedOpT>(context),
        debugger_type_(debugger_type),
        log_dir_path_(std::move(log_dir_path)) {}

  LogicalResult matchAndRewrite(LiftedOpT op,
                                PatternRewriter &rewriter) const override {
    if (match(op).failed()) {
      return failure();
    }
    rewrite(op, rewriter);
    return success();
  }

 private:
  SmallVector<NamedAttribute> CreateDumpAttributes(
      PatternRewriter &rewriter, const StringRef folder_name,
      const StringRef file_name, const bool enabled, const StringRef func_name,
      const StringRef node_name) const {
    SmallVector<NamedAttribute> dump_attributes{
        rewriter.getNamedAttr("log_dir_path",
                              rewriter.getStringAttr(folder_name)),
        rewriter.getNamedAttr("file_name", rewriter.getStringAttr(file_name)),
        // The op is disabled by default. Otherwise, values will be saved
        // during calibration.
        rewriter.getNamedAttr("enabled", rewriter.getBoolAttr(enabled)),
        rewriter.getNamedAttr("func_name", rewriter.getStringAttr(func_name)),
        rewriter.getNamedAttr("node_name", rewriter.getStringAttr(node_name)),
    };
    return dump_attributes;
  }

  StringAttr DuplicateFunction(Operation *op,
                               const FlatSymbolRefAttr &f_attr) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    SymbolTable symbol_table(module);

    const func::FuncOp ref_func =
        dyn_cast_or_null<func::FuncOp>(symbol_table.lookup(f_attr.getValue()));
    func::FuncOp new_ref_func = dyn_cast<func::FuncOp>(ref_func->clone());
    return symbol_table.insert(new_ref_func);
  }

  LogicalResult match(LiftedOpT op) const {
    if (!op->hasAttr(kQuantTraitAttrName) || op->getNumResults() != 1) {
      return failure();
    }

    Value result = op->getResult(0);
    for (auto user : result.getUsers()) {
      if (dyn_cast_or_null<TF::DumpTensorOp>(user)) return failure();
    }

    const FlatSymbolRefAttr f_attr = GetFuncAttr(op);
    if (!f_attr.getValue().starts_with(kCompositeFuncPrefix)) return failure();
    return success();
  }

  void rewrite(LiftedOpT op, PatternRewriter &rewriter) const {
    // Only support ops with 1 results
    Value result = op->getResult(0);
    rewriter.setInsertionPointAfterValue(result);

    // In Whole model, we first need to set file_name as
    // unquantized_tensor_data.pb as it is used by unquantized dump model.
    // After saving unquantized dump model, the file name will be changed to
    // quantized_tensor_data.pb.
    // Since this process doesn't happen for per layer, we need to set file_name
    // as quantized_tensor_data.pb here.
    // TODO: b/296933893 - Refactor the debugger code when no quantize option
    // is added
    std::string file_name =
        debugger_type_ == DebuggerConfig::DEBUGGER_TYPE_WHOLE_MODEL
            ? "unquantized_tensor_data.pb"
            : "quantized_tensor_data.pb";

    const FlatSymbolRefAttr f_attr = GetFuncAttr(op);

    // In TF::PartitionedCallOp case, func_name and node_name are filled.
    // But in TF::XlaCallModuleOp case, node_name is `kEmptyNodeName` since
    // debugging and selective quantization of StableHLO Quantizer only uses
    // func_name for op matching.
    auto [func_name, node_name] = GetFuncNameAndNodeName(op, f_attr);
    std::string folder_name =
        tensorflow::io::JoinPath(log_dir_path_, f_attr.getValue());

    // Attach DumpTensorOp to its output layer.
    SmallVector<NamedAttribute> dump_attributes =
        CreateDumpAttributes(rewriter, folder_name, file_name,
                             /*enabled=*/true, func_name, node_name);
    rewriter.create<TF::DumpTensorOp>(op->getLoc(), TypeRange{}, result,
                                      dump_attributes);

    // Per-layer mode.
    if (debugger_type_ == DebuggerConfig::DEBUGGER_TYPE_INT_PER_LAYER ||
        debugger_type_ == DebuggerConfig::DEBUGGER_TYPE_FLOAT_PER_LAYER) {
      // Duplicate composite function and op of quantizable layer for creating
      // unquantized layer.
      StringAttr new_ref_func_name = DuplicateFunction(op, f_attr);
      Operation *new_op = DuplicateOp(op, rewriter, new_ref_func_name);

      // Attach second DumpTensorOp to its output unquantized layer.
      SmallVector<NamedAttribute> dump_attributes = CreateDumpAttributes(
          rewriter, folder_name, /*file_name=*/"unquantized_tensor_data.pb",
          /*enabled=*/true, func_name, node_name);
      rewriter.create<TF::DumpTensorOp>(op.getLoc(), TypeRange{},
                                        new_op->getResult(0), dump_attributes);

      if (debugger_type_ == DebuggerConfig::DEBUGGER_TYPE_FLOAT_PER_LAYER) {
        // Swap all uses between call_op and ref_call_op, except for the
        // particular use that owns DumpTensor.
        rewriter.replaceUsesWithIf(
            op.getResult(0), new_op->getResult(0), [](OpOperand &use) -> bool {
              return !isa<TF::DumpTensorOp>(use.getOwner());
            });
      }
    }
  }

  DebuggerType debugger_type_;
  std::string log_dir_path_;
};

static PassRegistration<AddDumpTensorOpPass> pass;

void AddDumpTensorOpPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ModuleOp module = getOperation();

  patterns.add<AddDumpTensorOp<TF::PartitionedCallOp>,
               AddDumpTensorOp<TF::XlaCallModuleOp>>(ctx, debugger_type_,
                                                     log_dir_path_);

  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
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
