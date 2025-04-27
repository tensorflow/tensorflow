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

#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/tf_quantization_lib/tf_quantization_traits.h"
#include "tensorflow/compiler/mlir/quantization/common/tf_quantization_lib/tf_quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/tf_passes.h"  // IWYU pragma: keep

namespace mlir::tf_quant::stablehlo {

#define GEN_PASS_DEF_POSTQUANTIZEPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/tf_passes.h.inc"

namespace {

// Applies clean-up patterns after quantization.
class PostQuantizePass : public impl::PostQuantizePassBase<PostQuantizePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PostQuantizePass)

  explicit PostQuantizePass() = default;

 private:
  void runOnOperation() override;
};

// TODO: b/305815328 - Consider preserving leading and trailing QDQs for
// ModifyIONodesPass in TFLite use cases.
// Removes the back-to-back quantize and dequantize ops with volatile attribute.
class RemoveVolatileQdqPattern
    : public OpRewritePattern<mlir::quant::ir::DequantizeCastOp> {
 public:
  explicit RemoveVolatileQdqPattern(MLIRContext* context)
      : OpRewritePattern<mlir::quant::ir::DequantizeCastOp>(context) {}

  LogicalResult matchAndRewrite(mlir::quant::ir::DequantizeCastOp op,
                                PatternRewriter& rewriter) const override {
    auto input_op = op.getArg().getDefiningOp();
    if (auto q =
            llvm::dyn_cast_or_null<mlir::quant::ir::QuantizeCastOp>(input_op)) {
      if (!q->getAttr(kVolatileOpAttrName)) return failure();

      // If the quantize op is a requantize op, it is being used in other scale
      // adjustments and should be kept. Instead, move dequantize op before the
      // requantize op to remove the unnecessary requantize op.
      if (const QuantizedType qtype =
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

// Replaces constant and uniform_quantize ops with single quantized constant op.
class QuantizeConstPattern
    : public OpRewritePattern<mlir::stablehlo::UniformQuantizeOp> {
 public:
  explicit QuantizeConstPattern(MLIRContext* context)
      : OpRewritePattern<mlir::stablehlo::UniformQuantizeOp>(context) {}

  LogicalResult matchAndRewrite(mlir::stablehlo::UniformQuantizeOp op,
                                PatternRewriter& rewriter) const override {
    DenseFPElementsAttr attr;
    if (matchPattern(op.getOperand(), m_Constant(&attr))) {
      const Type qtype = op.getResult().getType();
      ElementsAttr quantized_attr = Quantize(attr, qtype);
      if (quantized_attr) {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
            op, qtype, quantized_attr);
        return success();
      }
    }
    return failure();
  }
};

// Replaces quantfork.dcast with stablehlo.uniform_dequantize.
class ConvertDequantizeCastToUniformDequantizePattern
    : public OpRewritePattern<mlir::quant::ir::DequantizeCastOp> {
 public:
  explicit ConvertDequantizeCastToUniformDequantizePattern(MLIRContext* context)
      : OpRewritePattern<mlir::quant::ir::DequantizeCastOp>(context) {}
  LogicalResult matchAndRewrite(mlir::quant::ir::DequantizeCastOp dq_op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::stablehlo::UniformDequantizeOp>(
        dq_op, dq_op.getResult().getType(), dq_op.getArg());
    return success();
  }
};

// Replaces quantfork.qcast with stablehlo.uniform_quantize.
class ConvertQuantizeCastToUniformQuantizePattern
    : public OpRewritePattern<mlir::quant::ir::QuantizeCastOp> {
 public:
  explicit ConvertQuantizeCastToUniformQuantizePattern(MLIRContext* context)
      : OpRewritePattern<mlir::quant::ir::QuantizeCastOp>(context) {}
  LogicalResult matchAndRewrite(mlir::quant::ir::QuantizeCastOp q_op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::stablehlo::UniformQuantizeOp>(
        q_op, q_op.getResult().getType(), q_op.getArg());
    return success();
  }
};

void PostQuantizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  func::FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();
  // TODO: b/307463853 - Consider splitting passes for each pattern set.
  patterns.add<FoldTrivalRequantizeOp<mlir::quant::ir::QuantizeCastOp>,
               RemoveVolatileQdqPattern>(ctx);
  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }

  RewritePatternSet patterns_2(&getContext());
  patterns_2
      .add<QuantizeConstPattern, ConvertQuantizeCastToUniformQuantizePattern,
           ConvertDequantizeCastToUniformDequantizePattern>(ctx);
  if (failed(applyPatternsGreedily(func, std::move(patterns_2)))) {
    signalPassFailure();
  }
}

}  // namespace
}  // namespace mlir::tf_quant::stablehlo
