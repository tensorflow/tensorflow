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
#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// The pass to legalize the quantization emulation ops from TF.
//
namespace {

// Legalize TF quantization emulation ops to that in Quant ops dialect.
struct LegalizeTFToQuant
    : public PassWrapper<LegalizeTFToQuant, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeTFToQuant)

  explicit LegalizeTFToQuant() = default;
  LegalizeTFToQuant(const LegalizeTFToQuant &) {}

  /// Performs the lowering to Quant ops dialect.
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<quant::QuantDialect,
                    quantfork::QuantizationForkDialect>();
  }

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tf-to-quant";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Legalize TF to quant ops dialect";
  }
};

// Inserts a "tfl.quantize" and "tfl.dequantize" op pair (QDQs) after the
// "tf.FakeQuantWithMinMaxVarsOp" to be constant folded. Since the constant
// folding logic will use a "arith.constant" op to replace the
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

  LogicalResult matchAndRewrite(TFFakeQuantOp tf_op,
                                PatternRewriter &rewriter) const override {
    // We don't want to insert quantize/dequantize if the quantize op exists.
    auto res = tf_op.getOutputs();
    if (!res.hasOneUse() || isa<quantfork::QuantizeCastOp>(*res.user_begin()))
      return failure();

    // Extract the min/max constant values from the operands. We also consider
    // a special case that there are tf.Identity ops between the min/max
    // constants and the tf.FakeQuantWithMinMaxVarsOp.
    Value min = tf_op.getMin(), max = tf_op.getMax();
    DenseFPElementsAttr min_value, max_value;
    if (auto id1 = dyn_cast_or_null<TF::IdentityOp>(min.getDefiningOp())) {
      id1.replaceAllUsesWith(id1.getInput());
      min = tf_op.getMin();
      rewriter.eraseOp(id1);
    }
    if (auto id2 = dyn_cast_or_null<TF::IdentityOp>(max.getDefiningOp())) {
      id2.replaceAllUsesWith(id2.getInput());
      max = tf_op.getMax();
      rewriter.eraseOp(id2);
    }
    if (!matchPattern(min, m_Constant(&min_value))) return failure();
    if (!matchPattern(max, m_Constant(&max_value))) return failure();

    int quant_dim = -1;
    if (PerAxis) {
      // This is a special case that the quant_dim is the last dimensions
      // according to the tf.FakeQuantWithMinMaxPerChannel.
      quant_dim = mlir::cast<ShapedType>(res.getType()).getRank() - 1;
    }
    // Use the min/max from the operands and the num_bits and narrow_range
    // attribute to create the quantization parameter for the new quantize op.
    rewriter.setInsertionPointAfter(tf_op.getOperation());
    IntegerAttr num_bits = rewriter.getI64IntegerAttr(tf_op.getNumBits());
    BoolAttr narrow_range = rewriter.getBoolAttr(tf_op.getNarrowRange());
    Type res_type = tf_op.getType();
    TypeAttr qtype = quant::GetQuantizedTypeAttr(
        rewriter, res_type, min_value, max_value, quant_dim, num_bits,
        narrow_range, /*is_signed=*/true);
    if (!qtype) return failure();

    // Finally, use the quantization parameter to create the quantize and
    // dequantize ops, and insert them between the tf.FakeQuantWithMinMaxVarsOp
    // and its users.
    Value value = tf_op.getOutputs();
    auto quantize = rewriter.create<quantfork::QuantizeCastOp>(
        tf_op.getLoc(), qtype.getValue(), value);
    auto dequantize = rewriter.create<quantfork::DequantizeCastOp>(
        tf_op.getLoc(), res_type, quantize.getResult());
    value.replaceAllUsesWith(dequantize);
    quantize.getOperation()->replaceUsesOfWith(dequantize, value);

    return success();
  }
};

using PreparePerTensorFakeQuant =
    InsertQuantOpsAfterTFFakeQuantOp<TF::FakeQuantWithMinMaxVarsOp, false>;

using PreparePerChannelFakeQuant =
    InsertQuantOpsAfterTFFakeQuantOp<TF::FakeQuantWithMinMaxVarsPerChannelOp,
                                     true>;

// TODO(fengliuai): add the support of the tf.QuantizeAndDequantize*
// legalization.

void LegalizeTFToQuant::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto *ctx = func.getContext();
  patterns.add<PreparePerTensorFakeQuant, PreparePerChannelFakeQuant>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
}  // namespace

// Creates an instance of the TensorFlow dialect to QuantOps dialect pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateLegalizeTFToQuantPass() {
  return std::make_unique<LegalizeTFToQuant>();
}

static PassRegistration<LegalizeTFToQuant> pass([] {
  return CreateLegalizeTFToQuantPass();
});

}  // namespace TF
}  // namespace mlir
