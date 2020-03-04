/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "mlir/Dialect/QuantOps/QuantOps.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// The pass to legalize the quantization emulation ops from TF.
//
namespace {

// Legalize TF quantization emulation ops to that in Quant ops dialect.
struct LegalizeTFToQuant : public FunctionPass<LegalizeTFToQuant> {
  explicit LegalizeTFToQuant() = default;
  LegalizeTFToQuant(const LegalizeTFToQuant &) {}

  /// Performs the lowering to Quant ops dialect.
  void runOnFunction() override;
};

// TODO(fengliuai): move this rule to PreparePatterns.td
// TODO(b/140968741): propagate the sign from the command line. Currently all
// the FakeQuant is assumed to targeting UIN8, but per-channel kernel is
// actually INT8.
// Inserts a "tfl.quantize" and "tfl.dequantize" op pair (QDQs) after the
// "tf.FakeQuantWithMinMaxVarsOp" to be constant folded. Since the constant
// folding logic will use a "std.constant" op to replace the
// "tf.FakeQuantWithMinMaxVarsOp", the "tfl.quantize" op is used to preserve
// the quantization parameters as a TypeAttr and "tfl.dequantize" op used to
// convert the output type to the next op. Here are the transformations:
//
// input   min cst       max cst          input   min cst       max cst
//  \       |             |                \       |             |
//   \  (tf.Identity) (tf.Identity)   =>    \  (tf.Identity) (tf.Identity)
//    \     |             |                  \     |             |
//       tf.FakeQuantWithMinMaxVars       tf.FakeQuantWithMinMaxVars
//                   |                                 |
//                                                tf.quantize
//                                                     |
//                                                tf.dequantize
//                                                     |
// If the input is a constant, the result pattern will eventually converted to
//
//            quant-emulated input
//                   |
//               tf.quantize
//                   |
//              tf.dequantize
//                   |
template <typename TFFakeQuantOp, bool PerAxis>
struct InsertQuantOpsAfterTFFakeQuantOp
    : public OpRewritePattern<TFFakeQuantOp> {
  using BaseType = InsertQuantOpsAfterTFFakeQuantOp<TFFakeQuantOp, PerAxis>;

  explicit InsertQuantOpsAfterTFFakeQuantOp<TFFakeQuantOp, PerAxis>(
      MLIRContext *ctx)
      : OpRewritePattern<TFFakeQuantOp>(ctx) {}

  PatternMatchResult matchAndRewrite(TFFakeQuantOp tf_op,
                                     PatternRewriter &rewriter) const override {
    // We don't want to insert quantize/dequantize if the quantize op exists.
    auto res = tf_op.outputs();
    if (!res.hasOneUse() || isa<quant::QuantizeCastOp>(*res.user_begin()))
      return this->matchFailure();

    // Extract the min/max constant values from the operands. We also consider
    // a special case that there are tf.Identity ops between the min/max
    // constants and the tf.FakeQuantWithMinMaxVarsOp.
    Value min = tf_op.min(), max = tf_op.max();
    DenseFPElementsAttr min_value, max_value;
    if (auto id1 = dyn_cast_or_null<TF::IdentityOp>(min.getDefiningOp())) {
      id1.replaceAllUsesWith(id1.input());
      min = tf_op.min();
      rewriter.eraseOp(id1);
    }
    if (auto id2 = dyn_cast_or_null<TF::IdentityOp>(max.getDefiningOp())) {
      id2.replaceAllUsesWith(id2.input());
      max = tf_op.max();
      rewriter.eraseOp(id2);
    }
    if (!matchPattern(min, m_Constant(&min_value))) return this->matchFailure();
    if (!matchPattern(max, m_Constant(&max_value))) return this->matchFailure();

    int quant_dim = -1;
    if (PerAxis) {
      // This is a special case that the quant_dim is the last dimensions
      // according to the tf.FakeQuantWithMinMaxPerChannel.
      quant_dim = res.getType().template cast<ShapedType>().getRank() - 1;
    }
    // Use the min/max from the operands and the num_bits and narrow_range
    // attribute to create the quantization parameter for the new quantize op.
    rewriter.setInsertionPointAfter(tf_op);
    IntegerAttr num_bits =
        rewriter.getI64IntegerAttr(tf_op.num_bits().getSExtValue());
    BoolAttr narrow_range = rewriter.getBoolAttr(tf_op.narrow_range());
    Type res_type = tf_op.getType();
    TypeAttr qtype = quant::GetQuantizedTypeAttr(
        rewriter, res_type, min_value, max_value, quant_dim, num_bits,
        narrow_range, /*is_signed=*/true);
    if (!qtype) this->matchFailure();

    // Finally, use the quantization parameter to create the quantize and
    // dequantize ops, and insert them between the tf.FakeQuantWithMinMaxVarsOp
    // and its users.
    Value value = tf_op.outputs();
    auto quantize = rewriter.create<quant::QuantizeCastOp>(
        tf_op.getLoc(), qtype.getValue(), value);
    auto dequantize = rewriter.create<quant::DequantizeCastOp>(
        tf_op.getLoc(), res_type, quantize.getResult());
    value.replaceAllUsesWith(dequantize);
    quantize.getOperation()->replaceUsesOfWith(dequantize, value);

    return this->matchSuccess();
  }
};

using PreparePerTensorFakeQuant =
    InsertQuantOpsAfterTFFakeQuantOp<TF::FakeQuantWithMinMaxVarsOp, false>;

using PreparePerChannelFakeQuant =
    InsertQuantOpsAfterTFFakeQuantOp<TF::FakeQuantWithMinMaxVarsPerChannelOp,
                                     true>;

// TODO(fengliuai): add the support of the tf.QuantizeAndDequantize*
// legalization.

void LegalizeTFToQuant::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();
  auto *ctx = func.getContext();
  patterns.insert<PreparePerTensorFakeQuant, PreparePerChannelFakeQuant>(ctx);
  applyPatternsGreedily(func, patterns);
}
}  // namespace

// Creates an instance of the TensorFlow dialect to QuantOps dialect pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateLegalizeTFToQuantPass() {
  return std::make_unique<LegalizeTFToQuant>();
}

static PassRegistration<LegalizeTFToQuant> pass(
    "tf-to-quant", "Legalize TF to quant ops dialect");

}  // namespace TF
}  // namespace mlir
