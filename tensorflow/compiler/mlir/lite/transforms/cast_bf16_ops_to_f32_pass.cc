/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/cast_bf16_ops_to_f32_pass.h"

#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"

namespace mlir {
namespace TFL {
namespace {

Type CloneTypeWithNewElementType(Type old_type, Type element_type) {
  if (old_type.isBF16() || old_type.isF32()) {
    return element_type;
  }
  if (auto shaped_type = mlir::dyn_cast<ShapedType>(old_type)) {
    return shaped_type.clone(element_type);
  }
  return {};
}

class CastBf16OpsToF32 : public RewritePattern {
 public:
  explicit CastBf16OpsToF32(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    if (match(op).failed()) {
      return failure();
    }
    rewrite(op, rewriter);
    return success();
  }

 private:
  LogicalResult match(Operation* op) const {
    // Skip cast ops, constants, zero-operand ops, terminators, return-like ops,
    // region-bearing ops (e.g. tfl.while, tfl.if), and call ops to prevent type
    // mismatches with nested block arguments or function signatures.
    if (isa<mlir::TFL::CastOp>(op) || op->hasTrait<OpTrait::ConstantLike>() ||
        op->getName().hasTrait<OpTrait::ZeroOperands>() ||
        op->hasTrait<OpTrait::IsTerminator>() ||
        op->hasTrait<OpTrait::ReturnLike>() || op->getNumRegions() > 0 ||
        isa<CallOpInterface>(op)) {
      return failure();
    }
    for (Value input : op->getOperands()) {
      if (getElementTypeOrSelf(input).isBF16()) {
        return success();
      }
    }
    for (Value value : op->getResults()) {
      if (getElementTypeOrSelf(value).isBF16()) {
        return success();
      }
    }
    return failure();
  }

  void rewrite(Operation* op, PatternRewriter& rewriter) const {
    rewriter.modifyOpInPlace(op, [&]() {
      // Casts inputs of the operation from BF16 to F32.
      for (int i = 0; i < op->getNumOperands(); i++) {
        Value input = op->getOperand(i);
        if (getElementTypeOrSelf(input).isBF16()) {
          Type f32_type = CloneTypeWithNewElementType(input.getType(),
                                                      rewriter.getF32Type());
          if (f32_type) {
            Value f32_cast = mlir::TFL::CastOp::create(rewriter, op->getLoc(),
                                                       f32_type, input);
            op->setOperand(i, f32_cast);
          }
        }
      }

      // Casts BF16 outputs of the operation to F32, and inserts casts back to
      // BF16 for downstream consumers.
      for (Value value : op->getResults()) {
        if (getElementTypeOrSelf(value).isBF16()) {
          Type f32_type = CloneTypeWithNewElementType(value.getType(),
                                                      rewriter.getF32Type());
          if (!f32_type) continue;
          value.setType(f32_type);
          rewriter.setInsertionPointAfterValue(value);
          Type bf16_type = CloneTypeWithNewElementType(value.getType(),
                                                       rewriter.getBF16Type());
          Value bf16_cast = mlir::TFL::CastOp::create(rewriter, op->getLoc(),
                                                      bf16_type, value);
          rewriter.replaceAllUsesExcept(value, bf16_cast,
                                        bf16_cast.getDefiningOp());
        }
      }
    });
  }
};

// Remove unneeded redundant cast ops like (f32 -> bf16 -> f32).
class RemoveUnneededCastOps : public OpRewritePattern<mlir::TFL::CastOp> {
 public:
  explicit RemoveUnneededCastOps(MLIRContext* context)
      : OpRewritePattern<mlir::TFL::CastOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(mlir::TFL::CastOp op,
                                PatternRewriter& rewriter) const override {
    Value input = op.getInput();
    auto prev_cast = input.getDefiningOp<mlir::TFL::CastOp>();
    if (!prev_cast) {
      return failure();
    }
    Value orig_input = prev_cast.getInput();
    if (orig_input.getType() != op.getType()) {
      return failure();
    }
    rewriter.replaceOp(op, orig_input);
    return success();
  }
};

}  // namespace

void CastBf16OpsToF32Pass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func_op = getOperation();

  patterns.add<CastBf16OpsToF32, RemoveUnneededCastOps>(ctx);

  if (failed(applyPatternsGreedily(func_op, std::move(patterns)))) {
    func_op.emitError() << "tfl-cast-bf16-ops-to-f32 failed.";
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateCastBf16OpsToF32Pass() {
  return std::make_unique<CastBf16OpsToF32Pass>();
}

}  // namespace TFL
}  // namespace mlir
