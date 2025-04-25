/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass applies some clean up steps after quantization.

#include <memory>
#include <string>
#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project

//===----------------------------------------------------------------------===//
// The post-quantize Passes.
//
namespace mlir {
namespace quant {
namespace {

// Applies all the clean up steps after quantization.
class TFPostQuantizePass
    : public PassWrapper<TFPostQuantizePass, OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TFPostQuantizePass)

  // Constructor used by the PassRegistration. This will remove the adaptor ops.
  explicit TFPostQuantizePass() = default;

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tf-quant-post-quantize";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Apply post quantization clean up after quantization";
  }

  void runOnOperation() override;
};

enum RemoveVolatileOpsType {
  // Remove all volatile quant-dequant ops.
  kPreserveNone,
  // Preserve volatile quant-dequants for input and output ops.
  kPreserveInputsAndOutputs,
};

// Remove the back-to-back quantize and dequantize ops with volatile attribute.
template <RemoveVolatileOpsType remove_volatile_ops_type>
struct RemoveVolatileOps
    : public OpRewritePattern<mlir::quant::ir::DequantizeCastOp> {
  explicit RemoveVolatileOps(MLIRContext* context)
      : OpRewritePattern<mlir::quant::ir::DequantizeCastOp>(context, 1) {}

  LogicalResult matchAndRewrite(mlir::quant::ir::DequantizeCastOp op,
                                PatternRewriter& rewriter) const override {
    auto input_op = op.getArg().getDefiningOp();
    if (auto q = llvm::dyn_cast_or_null<mlir::quant::ir::QuantizeCastOp>(
      input_op)) {
      if (!q->getAttr(kVolatileOpAttrName)) return failure();

      if (remove_volatile_ops_type == kPreserveInputsAndOutputs) {
        // Don't remove leading and trailing QDQ for PTQ workflow, so the io
        // modifying lib can work correctly.
        if (!q.getArg().getDefiningOp()) return failure();
        if (op->hasOneUse() &&
            op->user_begin()->hasTrait<OpTrait::IsTerminator>())
          return failure();
      }
      // If the quantize op is a requantize op, it is being used in other scale
      // adjustments and should be kept. Instead, moving dequantize op before
      // the requantize op to remove the unnecessary requantize op.
      if (auto qtype =
              QuantizedType::getQuantizedElementType(q.getArg().getType())) {
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<mlir::quant::ir::DequantizeCastOp>(
            op, op.getResult().getType(), q.getArg());
        return success();
      }

      op.replaceAllUsesWith(q.getArg());
      return success();
    }
    return failure();
  }
};

// The StorageCastOp is used to cast from a quantized type to its storage type
// or the opposite. If none of its input and output is quantized, the op has
// no effect and should be removed.
class RemoveRedundantScast
    : public mlir::OpRewritePattern<mlir::quant::ir::StorageCastOp> {
 public:
  explicit RemoveRedundantScast(MLIRContext* context)
      : OpRewritePattern<mlir::quant::ir::StorageCastOp>(context) {}

 private:
  LogicalResult matchAndRewrite(mlir::quant::ir::StorageCastOp scast_op,
                                PatternRewriter& rewriter) const override {
    if (QuantizedType::getQuantizedElementType(scast_op.getArg().getType()) ||
        QuantizedType::getQuantizedElementType(scast_op.getType())) {
      return failure();
    }

    scast_op.replaceAllUsesWith(scast_op.getArg());
    return success();
  }
};

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_post_quantize.inc"

void TFPostQuantizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto* ctx = func.getContext();
  patterns.add<FoldTrivalRequantizeOp<mlir::quant::ir::QuantizeCastOp>,
               RemoveVolatileOps<kPreserveNone>, RemoveRedundantScast>(ctx);
  populateWithGenerated(patterns);
  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

// Creates an instance of the TensorFlow dialect PostQuantize pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTFPostQuantizePass() {
  return std::make_unique<TFPostQuantizePass>();
}

static PassRegistration<TFPostQuantizePass> pass;

}  // namespace quant
}  // namespace mlir
