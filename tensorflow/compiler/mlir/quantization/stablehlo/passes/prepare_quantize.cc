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
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_driver.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/ops/stablehlo_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace stablehlo {

#define GEN_PASS_DEF_PREPAREQUANTIZEPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

// Applies prepare quantization on the model in TF dialect. This pass runs
// before the quantization pass and propagate the quantization parameters
// across ops. This step is necessary for post-training quantization and also
// making the quantization rule for some operations in the quantization-aware
// training quantization simpler.
class PrepareQuantizePass
    : public impl::PrepareQuantizePassBase<PrepareQuantizePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareQuantizePass)

  using impl::PrepareQuantizePassBase<
      PrepareQuantizePass>::PrepareQuantizePassBase;

  explicit PrepareQuantizePass(const bool enable_per_channel_quantized_weight,
                               const int bit_width) {
    enable_per_channel_quantized_weight_ = enable_per_channel_quantized_weight;
    bit_width_ = bit_width;
  }

  void runOnOperation() override;
};

// Merges consecutive QuantizeCast ops. See b/246655213 for details.
// For example, the following case:
// %1 = quantfork.QuantizeCastOp(%0) : f32 -> qtype1
// %2 = quantfork.QuantizeCastOp(%1) : qtype1 -> qtype2
// %3 = quantfork.QuantizedOp1(%1)
// %4 = quantfork.QuantizedOp2(%2)
// will be tranformed to:
// %1 = quantfork.QuantizeCastOp(%0) : f32 -> qtype1
// %2 = quantfork.QuantizeCastOp(%0) : f32 -> qtype2
// %3 = quantfork.QuantizedOp1(%1)
// %4 = quantfork.QuantizedOp2(%2)
// Converting from f32 -> qtype1 -> qtype2 will add unexpected quantization
// lost for %2. This pattern avoids that by converting from f32 -> qtype2
// directly.
class MergeConsecutiveQuantizeCast
    : public mlir::OpRewritePattern<quantfork::QuantizeCastOp> {
 public:
  explicit MergeConsecutiveQuantizeCast(MLIRContext* context)
      : OpRewritePattern<quantfork::QuantizeCastOp>(context) {}

 private:
  LogicalResult matchAndRewrite(quantfork::QuantizeCastOp q_op,
                                PatternRewriter& rewriter) const override {
    auto preceding_qcast =
        q_op.getArg().getDefiningOp<quantfork::QuantizeCastOp>();
    if (!preceding_qcast) return failure();

    auto new_qcast = rewriter.create<quantfork::QuantizeCastOp>(
        q_op.getLoc(), q_op.getType(), preceding_qcast.getArg());
    new_qcast->setAttr(kVolatileOpAttrName, rewriter.getUnitAttr());
    q_op->replaceAllUsesWith(new_qcast);
    return success();
  }
};

class ConvertTFConstOpToArithConstOp : public OpRewritePattern<TF::ConstOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::ConstOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValue());
    return success();
  }
};

class ConvertStablehloConstToArithConstOp
    : public OpRewritePattern<mlir::stablehlo::ConstantOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::stablehlo::ConstantOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValue());
    return success();
  }
};

class ConvertArithConstToStablehloConstOp
    : public OpRewritePattern<arith::ConstantOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, op.getValue());
    return success();
  }
};

void PrepareQuantizePass::runOnOperation() {
  ModuleOp module_op = getOperation();
  MLIRContext* ctx = module_op.getContext();

  auto func_op_quant_spec = GetStableHloOpQuantSpec;
  auto func_op_quant_scale_spec = GetStableHloQuantConstraints;

  for (auto func_op : module_op.getOps<func::FuncOp>()) {
    // The function might contain more stats ops than required, and it will
    // introduce requantize if the calibration stats have conflicts. This tries
    // to remove all the redundant stats ops.
    RemoveRedundantStatsOps(func_op, func_op_quant_spec,
                            func_op_quant_scale_spec);

    RewritePatternSet patterns(ctx);
    // Convert quant stats to int8 quantization parameters.
    // Currently, only activation stats are imported, so narrow_range = false.
    patterns.add<quant::ConvertStatsToQDQs<quantfork::QuantizeCastOp,
                                           quantfork::DequantizeCastOp>>(
        bit_width_,
        /*narrow_range=*/false,
        /*is_signed=*/true,
        /*legacy_float_scale=*/false, ctx);
    // Convert all constants to arith::ConstantOp as quantization driver can
    // deal with the arith::ConstantOp instances.
    patterns.add<ConvertTFConstOpToArithConstOp>(ctx);
    patterns.add<ConvertStablehloConstToArithConstOp>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func_op, std::move(patterns)))) {
      signalPassFailure();
    }

    // Finally, the quantization parameters can be propagated to the rest of the
    // values (tensors).
    ApplyQuantizationParamsPropagation(
        func_op, /*is_signed=*/true, bit_width_,
        !enable_per_channel_quantized_weight_, func_op_quant_spec,
        func_op_quant_scale_spec,
        /*infer_tensor_ranges=*/true, /*legacy_float_scale=*/false,
        /*is_qdq_conversion=*/false);

    // Restore constants as stablehlo::ConstantOp.
    RewritePatternSet patterns_2(ctx);
    patterns_2
        .add<MergeConsecutiveQuantizeCast, ConvertArithConstToStablehloConstOp>(
            ctx);
    if (failed(applyPatternsAndFoldGreedily(func_op, std::move(patterns_2)))) {
      signalPassFailure();
    }
  }
}

}  // namespace

// Creates an instance of the TensorFlow dialect PrepareQuantize pass.
std::unique_ptr<OperationPass<ModuleOp>> CreatePrepareQuantizePass(
    const bool enable_per_channel_quantized_weight, const int bit_width) {
  return std::make_unique<PrepareQuantizePass>(
      enable_per_channel_quantized_weight, bit_width);
}

}  // namespace stablehlo
}  // namespace quant
}  // namespace mlir
