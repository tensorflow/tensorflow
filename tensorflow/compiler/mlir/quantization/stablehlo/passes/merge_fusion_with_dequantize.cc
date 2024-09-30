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

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_MERGEFUSIONWITHDEQUANTIZEPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

class MergeFusionWithDequantizePass
    : public impl::MergeFusionWithDequantizePassBase<
          MergeFusionWithDequantizePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeFusionWithDequantizePass)

  explicit MergeFusionWithDequantizePass() = default;

 private:
  void runOnOperation() override;
};

class MergeFusionWithUniformDequantizePattern
    : public OpRewritePattern<func::CallOp> {
 public:
  explicit MergeFusionWithUniformDequantizePattern(MLIRContext* context)
      : OpRewritePattern<func::CallOp>(context) {}
  LogicalResult matchAndRewrite(func::CallOp call_op,
                                PatternRewriter& rewriter) const override {
    if (call_op.getNumResults() != 1) return failure();
    auto users = call_op->getUsers();
    for (auto user : users) {
      if (!llvm::isa<mlir::stablehlo::UniformDequantizeOp>(user)) {
        return failure();
      }
    }
    auto func_name = call_op.getCallee();
    if (!func_name.starts_with("quantized_")) return failure();
    if (call_op->getNumResults() != 1) return failure();
    if (!mlir::isa<UniformQuantizedType>(
            getElementTypeOrSelf(call_op->getResult(0).getType())))
      return failure();

    // Fetch the callee function.
    SymbolTable symbol_table(call_op->getParentOfType<ModuleOp>());
    auto func_op =
        dyn_cast_or_null<func::FuncOp>(symbol_table.lookup(func_name));
    if (!func_op) return failure();
    // The quantized fusion should have requantize and return ops at the end.
    auto return_op = dyn_cast_or_null<func::ReturnOp>(
        func_op.getRegion().getBlocks().front().getTerminator());
    if (!return_op) return failure();
    auto req_op = llvm::dyn_cast_or_null<mlir::stablehlo::UniformQuantizeOp>(
        return_op.getOperands()[0].getDefiningOp());
    if (!req_op) return failure();

    // Create a new func.call op with f32 output.
    auto new_call_op = call_op.clone();
    new_call_op->getResult(0).setType(
        mlir::cast<ShapedType>(call_op.getResult(0).getType())
            .clone(rewriter.getF32Type()));
    rewriter.setInsertionPoint(call_op);
    rewriter.insert(new_call_op);

    // Remove the dequantize ops and replace uses by the new func.call op.
    SmallVector<Operation*> users_to_erase;
    for (auto user : users) {
      llvm::dyn_cast<mlir::stablehlo::UniformDequantizeOp>(user)
          .replaceAllUsesWith(new_call_op.getResult(0));
      users_to_erase.push_back(user);
    }
    for (auto user : users_to_erase) rewriter.eraseOp(user);
    rewriter.eraseOp(call_op);
    func_op.eraseResult(0);
    func_op.insertResult(0, new_call_op.getResult(0).getType(),
                         /*resultAttrs=*/nullptr);

    // Modify the quantized fused function to do dequantize+relu(6).
    rewriter.setInsertionPoint(req_op);
    Value new_result = rewriter.create<mlir::stablehlo::UniformDequantizeOp>(
        req_op.getLoc(), func_op.getResultTypes()[0], req_op.getOperand());
    if (func_name.contains("_relu6_")) {
      auto min = rewriter.create<mlir::stablehlo::ConstantOp>(
          req_op.getLoc(), rewriter.getF32FloatAttr(0));
      auto max = rewriter.create<mlir::stablehlo::ConstantOp>(
          req_op.getLoc(), rewriter.getF32FloatAttr(6));
      new_result = rewriter.create<mlir::stablehlo::ClampOp>(
          req_op.getLoc(), min, new_result, max);
    } else if (func_name.contains("_relu_")) {
      auto min = rewriter.create<mlir::stablehlo::ConstantOp>(
          req_op.getLoc(), rewriter.getF32FloatAttr(0));
      new_result = rewriter.create<mlir::chlo::BroadcastMaxOp>(
          req_op.getLoc(), min, new_result, nullptr);
    }
    return_op->setOperand(0, new_result);
    rewriter.eraseOp(req_op);

    return success();
  }
};

void MergeFusionWithDequantizePass::runOnOperation() {
  ModuleOp module_op = getOperation();
  MLIRContext* ctx = module_op.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<MergeFusionWithUniformDequantizePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(module_op, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace
}  // namespace mlir::quant::stablehlo
