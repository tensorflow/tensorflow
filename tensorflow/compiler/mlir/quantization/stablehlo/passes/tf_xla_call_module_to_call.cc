/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/tf_attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir::tf_quant::stablehlo {

#define GEN_PASS_DEF_XLACALLMODULETOCALLPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/tf_passes.h.inc"

namespace {

// Converts XlaCallModuleOps to func.call.
class XlaCallModuleToCallPass
    : public impl::XlaCallModuleToCallPassBase<XlaCallModuleToCallPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XlaCallModuleToCallPass)

  explicit XlaCallModuleToCallPass() = default;

 private:
  void runOnOperation() override;
};

// Converts XlaCallModuleOps to func.call.
class XlaCallModuleOpToCallOp : public OpRewritePattern<TF::XlaCallModuleOp> {
 public:
  explicit XlaCallModuleOpToCallOp(MLIRContext* context)
      : OpRewritePattern<TF::XlaCallModuleOp>(context) {}

  LogicalResult matchAndRewrite(TF::XlaCallModuleOp op,
                                PatternRewriter& rewriter) const override {
    auto module_op = op->getParentOfType<ModuleOp>();
    SymbolTable symbol_table(module_op);

    auto entry_func_op = dyn_cast_or_null<func::FuncOp>(
        symbol_table.lookup(GetEntryFunctionName(op)));
    if (!entry_func_op) return failure();

    // Replace the XlaCallModuleOp with a new CallOp.
    rewriter.replaceOpWithNewOp<func::CallOp>(op, entry_func_op, op.getArgs());
    return success();
  }
};

void XlaCallModuleToCallPass::runOnOperation() {
  ModuleOp module_op = getOperation();
  MLIRContext* ctx = module_op.getContext();
  RewritePatternSet patterns(&getContext());
  patterns.add<XlaCallModuleOpToCallOp>(ctx);
  if (failed(applyPatternsGreedily(module_op, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace
}  // namespace mlir::tf_quant::stablehlo
