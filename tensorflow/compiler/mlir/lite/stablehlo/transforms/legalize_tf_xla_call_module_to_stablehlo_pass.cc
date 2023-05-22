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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_tf_xla_call_module_to_stablehlo_pass.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace odml {

static constexpr std::string_view kStablehloModuleDefaultEntryFuncName = "main";
static constexpr std::string_view kStablehloFuncNamePrefix = "XlaCallModule";

class ConvertTFXlaCallModuleOp
    : public mlir::OpRewritePattern<mlir::TF::XlaCallModuleOp> {
 public:
  explicit ConvertTFXlaCallModuleOp(MLIRContext *context, ModuleOp module_op)
      : OpRewritePattern<mlir::TF::XlaCallModuleOp>(context),
        module_op_(module_op) {}
  using OpRewritePattern<mlir::TF::XlaCallModuleOp>::OpRewritePattern;

 private:
  ModuleOp module_op_;
  mlir::LogicalResult matchAndRewrite(
      mlir::TF::XlaCallModuleOp op, PatternRewriter &rewriter) const override {
    mlir::OwningOpRef<ModuleOp> stablehlo_module_op =
        mlir::stablehlo::deserializePortableArtifact(op.getModuleAttr(),
                                                     getContext());
    if (stablehlo_module_op.get() == nullptr) {
      return mlir::failure();
    }
    SymbolTable parent_module_symbol_table(module_op_);
    SymbolTable stablehlo_module_symbol_table(stablehlo_module_op.get());
    if (stablehlo_module_symbol_table.lookup<mlir::func::FuncOp>(
            kStablehloModuleDefaultEntryFuncName) == nullptr) {
      return rewriter.notifyMatchFailure(
          op, "could not find main function in XlaCallModuleOp");
    }
    mlir::Builder stablehlo_builder(stablehlo_module_op.get().getContext());
    // Rename XlaCallModuleOp's functions to avoid naming conflicts.
    for (auto func_op :
         stablehlo_module_op.get().getOps<mlir::func::FuncOp>()) {
      const std::string new_func_name =
          CreateNewFuncName(func_op.getSymName(), parent_module_symbol_table);
      if (failed(stablehlo_module_symbol_table.replaceAllSymbolUses(
              func_op, stablehlo_builder.getStringAttr(new_func_name),
              stablehlo_module_op.get()))) {
        return mlir::failure();
      }
      mlir::SymbolTable::setSymbolName(func_op, new_func_name);
    }
    // Move all functions from XlaCallModuleOp's stablehlo module, to parent
    // module. Also marks the stablehlo module entry function as private.
    mlir::func::FuncOp main_fn;
    for (auto func_op :
         stablehlo_module_op.get().getOps<mlir::func::FuncOp>()) {
      mlir::func::FuncOp cloned_func_op = func_op.clone();
      if (cloned_func_op.getSymName().contains(
              kStablehloModuleDefaultEntryFuncName)) {
        main_fn = cloned_func_op;
        main_fn.setSymVisibility(stablehlo_builder.getStringAttr("private"));
      }
      parent_module_symbol_table.insert(cloned_func_op);
    }

    // The stablehlo module main function's input tensor types might be
    // different from the XlaCallModuleOp's input tensor types. For example,
    // The XlaCallModuleOp's input is tensor<*xf32> while the function's
    // argument type is tensor<1x2f32>.
    llvm::SmallVector<Value, 4> casted_operands;
    casted_operands.reserve(main_fn.getNumArguments());
    for (const auto &operand_and_type :
         zip(op.getOperands(), main_fn.getFunctionType().getInputs())) {
      Value operand = std::get<0>(operand_and_type);
      Type expected_type = std::get<1>(operand_and_type);
      if (operand.getType() != expected_type) {
        operand = rewriter.create<TF::CastOp>(
            op.getLoc(), expected_type, operand,
            /*Truncate=*/rewriter.getBoolAttr(false));
      }
      casted_operands.push_back(operand);
    }

    auto call = rewriter.create<func::CallOp>(
        op->getLoc(), main_fn.getSymName(), main_fn.getResultTypes(),
        casted_operands);
    rewriter.replaceOp(op, call->getResults());

    return mlir::success();
  }

  // Creates a new function name to avoid collision. The naming scheme is
  // XlaCallModule_%s_%d where %s is the original function name and %d is the
  // counter.
  std::string CreateNewFuncName(const StringRef func_name,
                                SymbolTable &symbol_table) const {
    int suffix_id = 0;
    std::string new_func_name = absl::StrCat(kStablehloFuncNamePrefix, "_",
                                             func_name.str(), "_", suffix_id);
    while (symbol_table.lookup(new_func_name)) {
      suffix_id++;
      new_func_name = absl::StrCat(kStablehloFuncNamePrefix, "_",
                                   func_name.str(), "_", suffix_id);
    }
    return new_func_name;
  }
};

class TFXlaCallModuleOpToStablehloPass
    : public PassWrapper<TFXlaCallModuleOpToStablehloPass,
                         OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const final {
    return "tf-xla-call-module-op-to-stablehlo-pass";
  }
  StringRef getDescription() const final {
    return "Legalize TF_XlaCallModule Op to stablehlo";
  }
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect,
                    mlir::quant::QuantizationDialect, shape::ShapeDialect>();
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertTFXlaCallModuleOp>(&getContext(), module_op);
    if (failed(applyPatternsAndFoldGreedily(module_op, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::OperationPass<ModuleOp>>
CreateLegalizeTFXlaCallModuleToStablehloPass() {
  return std::make_unique<TFXlaCallModuleOpToStablehloPass>();
}

static PassRegistration<TFXlaCallModuleOpToStablehloPass> pass;

}  // namespace odml
}  // namespace mlir
