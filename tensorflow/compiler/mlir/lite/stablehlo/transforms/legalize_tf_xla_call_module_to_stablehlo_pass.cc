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

#include <cassert>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace odml {

static constexpr absl::string_view kStablehloModuleDefaultEntryFuncName =
    "main";
static constexpr absl::string_view kStablehloFuncNamePrefix = "XlaCallModule";
static constexpr char kShardingAttr[] = "mhlo.sharding";
static constexpr char kShardingName[] = "Sharding";

class RemoveCustomCallWithSharding
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern<stablehlo::CustomCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    // Removes the custom call with sharding op if the operand type is the
    // same as the result type.
    if (op->hasAttr(kShardingAttr) && op.getCallTargetName() == kShardingName &&
        op.getNumOperands() == 1 && op.getNumResults() == 1 &&
        op.getOperands().front().getType() ==
            op.getResults().front().getType()) {
      rewriter.replaceOp(op, op.getOperands());
      return success();
    }
    return failure();
  }
};

namespace {

bool IsShloMainFuncOp(func::FuncOp func_op) {
  if (func_op == nullptr) {
    return false;
  }

  if (!func_op.getSymName().contains(kStablehloModuleDefaultEntryFuncName)) {
    return false;
  }

  if (func_op.getSymVisibility() == "nested" ||
      func_op.getSymVisibility() == "private") {
    return false;
  }

  return true;
}

// Returns true if XlaCallModuleOp has the "platform index argument". The
// platform index argument is an extra 0-dimensional i32 tensor argument at
// index 0 when the XlaCallModuleOp contains more than one platform specified at
// the "platform" attribute.
//
// See:
// https://github.com/tensorflow/tensorflow/blob/eba24f41ba9d661d2f58a515921720cf90708cd4/tensorflow/compiler/tf2xla/ops/xla_ops.cc#L1376-L1385
bool ContainsPlatformIndexArg(TF::XlaCallModuleOp xla_call_module_op) {
  return xla_call_module_op.getPlatforms().size() > 1;
}

}  // namespace

class ConvertTFXlaCallModuleOp : public OpRewritePattern<TF::XlaCallModuleOp> {
 public:
  explicit ConvertTFXlaCallModuleOp(MLIRContext *context, ModuleOp module_op)
      : OpRewritePattern<TF::XlaCallModuleOp>(context), module_op_(module_op) {}
  using OpRewritePattern<TF::XlaCallModuleOp>::OpRewritePattern;

 private:
  ModuleOp module_op_;
  LogicalResult matchAndRewrite(TF::XlaCallModuleOp op,
                                PatternRewriter &rewriter) const override {
    OwningOpRef<ModuleOp> stablehlo_module_op =
        stablehlo::deserializePortableArtifact(op.getModuleAttr(),
                                               getContext());
    if (stablehlo_module_op.get() == nullptr) {
      return failure();
    }
    SymbolTable parent_module_symbol_table(module_op_);
    SymbolTable stablehlo_module_symbol_table(stablehlo_module_op.get());
    {
      auto main_func_op = stablehlo_module_symbol_table.lookup<func::FuncOp>(
          kStablehloModuleDefaultEntryFuncName);
      // TODO(b/291988976): move enforcement of this variable outside of this
      // rewrite pattern such that it's only checked once. Currently, this
      // approach results in duplicate error messages as this pattern executes
      // more than once.
      if (!IsShloMainFuncOp(main_func_op)) {
        auto error_msg =
            "'main' FuncOp in XlaCallModuleOp missing or has visibility other "
            "than 'public'";
        if (main_func_op) {
          main_func_op->emitError(error_msg);
        }
        return rewriter.notifyMatchFailure(op, error_msg);
      }
    }
    Builder stablehlo_builder(stablehlo_module_op.get().getContext());
    // Rename XlaCallModuleOp's functions to avoid naming conflicts.
    for (auto func_op : stablehlo_module_op.get().getOps<func::FuncOp>()) {
      const std::string new_func_name =
          CreateNewFuncName(func_op.getSymName(), parent_module_symbol_table);
      if (failed(stablehlo_module_symbol_table.replaceAllSymbolUses(
              func_op, stablehlo_builder.getStringAttr(new_func_name),
              stablehlo_module_op.get()))) {
        return failure();
      }
      SymbolTable::setSymbolName(func_op, new_func_name);
    }
    // Move all functions from XlaCallModuleOp's stablehlo module, to parent
    // module. Also marks the stablehlo module entry function as private.
    func::FuncOp main_fn;
    for (auto func_op : stablehlo_module_op.get().getOps<func::FuncOp>()) {
      func::FuncOp cloned_func_op = func_op.clone();
      if (IsShloMainFuncOp(cloned_func_op)) {
        main_fn = cloned_func_op;
      }
      cloned_func_op.setSymVisibility(
          stablehlo_builder.getStringAttr("private"));
      parent_module_symbol_table.insert(cloned_func_op);
    }

    // When the `XlaCallModuleOp`'s callee accepts a platform index argument,
    // add a dummy platform index argument in order to match the number of
    // the arguments of the callee function.
    //
    // This is because `XlaCallModuleOp` doesn't explicitly take it as an
    // operand. See:
    // https://github.com/tensorflow/tensorflow/blob/eba24f41ba9d661d2f58a515921720cf90708cd4/tensorflow/compiler/tf2xla/ops/xla_ops.cc#L1376-L1385

    SmallVector<Value, 4> call_op_operands(op.getOperands());
    if (ContainsPlatformIndexArg(op)) {
      Value dummy_const = rewriter.create<TF::ConstOp>(
          op.getLoc(),
          DenseIntElementsAttr::get(
              RankedTensorType::get({}, rewriter.getIntegerType(32)), {0}));
      call_op_operands.insert(call_op_operands.begin(), dummy_const);
    }

    // The stablehlo module main function's input tensor types might be
    // different from the XlaCallModuleOp's input tensor types. For example,
    // The XlaCallModuleOp's input is tensor<*xf32> while the function's
    // argument type is tensor<1x2f32>.
    SmallVector<Value, 4> casted_operands;
    casted_operands.reserve(main_fn.getNumArguments());
    assert(call_op_operands.size() == main_fn.getNumArguments());
    for (const auto &operand_and_type :
         zip(call_op_operands, main_fn.getFunctionType().getInputs())) {
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

    return success();
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
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<stablehlo::StablehloDialect, vhlo::VhloDialect,
                    quant::QuantDialect, shape::ShapeDialect>();
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertTFXlaCallModuleOp>(&getContext(), module_op);
    patterns.add<RemoveCustomCallWithSharding>(&getContext());
    if (failed(applyPatternsGreedily(module_op, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
CreateLegalizeTFXlaCallModuleToStablehloPass() {
  return std::make_unique<TFXlaCallModuleOpToStablehloPass>();
}

static PassRegistration<TFXlaCallModuleOpToStablehloPass> pass;

}  // namespace odml
}  // namespace mlir
