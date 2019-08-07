/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass prepares for legalization to the TFLite dialect by
// converting operations in TensorFlow dialect into operations that can be
// legalized to TensorFlow Lite dialect with simple replacements.  The newly
// created operations are in the TensorFlow dialect if the operation can be
// represented using a TensorFlow op.  Otherwise, TensorFlow Lite dialect op is
// used.  For example, Conv2D in TFLite which uses OHWI data format for filters
// is not supported in TensorFlow because TensorFlow requires filters in the
// HWIO data format.
//
// Motivation to prepare for the TFLite legalization before the actual
// legalization is to exploit constant folding opportunities in any newly
// created ops by leveraging constant folding support for the TensorFlow ops.
// This way TFLite can be used as a serialization format only and does not
// require access to the TFLite runtime for optimizations as required by the
// TFLite team.

#include <climits>
#include <cstdint>

#include "absl/memory/memory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/LoopAnalysis.h"  // TF:local_config_mlir
#include "mlir/Dialect/QuantOps/FakeQuantSupport.h"  // TF:local_config_mlir
#include "mlir/Dialect/QuantOps/UniformSupport.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Support/Functional.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

#define DEBUG_TYPE "tf-tfl-legalization"

namespace mlir {
namespace TFL {
//===----------------------------------------------------------------------===//
// The actual PrepareTF Pass.
//
// TODO(hinsu): Add and use TensorFlow dialect ops for the ops created in this
// pass.
namespace {

// Prepare TF operations in functions for subsequent legalization.
struct PrepareTFPass : public FunctionPass<PrepareTFPass> {
  void runOnFunction() override;
};

// TODO(fengliuai): move this rule to PreparePatterns.td
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

//            quant-emulated input
//                   |
//               tf.quantize
//                   |
//              tf.dequantize
//                   |
struct InsertTFLQuantOpsAfterTFFakeQuantOp : public RewritePattern {
  InsertTFLQuantOpsAfterTFFakeQuantOp(MLIRContext *context)
      : RewritePattern(TF::FakeQuantWithMinMaxVarsOp::getOperationName(), 3,
                       context) {}
  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto tf_op = cast<TF::FakeQuantWithMinMaxVarsOp>(op);
    // We don't want to insert quantize/dequantize if the quantize op exists.
    auto res = tf_op.outputs();
    if (!res->hasOneUse() || isa<QuantizeOp>(*res->user_begin()))
      return matchFailure();

    // Extract the min/max constant values from the operands. We also consider
    // a special case that there are tf.Identity ops between the min/max
    // constants and the tf.FakeQuantWithMinMaxVarsOp.
    Value *min = tf_op.min(), *max = tf_op.max();
    ElementsAttr min_value, max_value;
    if (auto id1 = dyn_cast_or_null<TF::IdentityOp>(min->getDefiningOp()))
      min = id1.input();
    if (auto id2 = dyn_cast_or_null<TF::IdentityOp>(max->getDefiningOp()))
      max = id2.input();
    if (!matchPattern(min, m_Constant(&min_value))) return matchFailure();
    if (!matchPattern(max, m_Constant(&max_value))) return matchFailure();
    FloatAttr min_attr = ExtractSingleElementAsFloat(min_value);
    FloatAttr max_attr = ExtractSingleElementAsFloat(max_value);
    if (!min_attr || !max_attr) return matchFailure();

    // Use the min/max from the operands and the num_bits and narrow_range
    // attribute to create the quantization parameter for the new quantize op.
    rewriter.setInsertionPoint(op->getBlock(), ++Block::iterator(op));
    Type num_bits = rewriter.getIntegerType(tf_op.num_bits().getSExtValue());
    bool narrow_range = tf_op.narrow_range();
    Type res_type = tf_op.getType();
    TypeAttr qtype = GetQuantizedTypeAttr(rewriter, res_type, min_attr,
                                          max_attr, num_bits, narrow_range);

    // Finally, use the quantization parameter to create the quantize and
    // dequantize ops, and insert them between the tf.FakeQuantWithMinMaxVarsOp
    // and its users.
    Value *value = tf_op.outputs();
    auto quantize = rewriter.create<TFL::QuantizeOp>(
        op->getLoc(), qtype.getValue(), value, qtype);
    auto dequantize = rewriter.create<TFL::DequantizeOp>(op->getLoc(), res_type,
                                                         quantize.output());
    value->replaceAllUsesWith(dequantize);
    quantize.getOperation()->replaceUsesOfWith(dequantize, value);

    return matchSuccess();
  }
};

// Templated class for declaring a converter from some TensorFlow convolution
// op into its counterpart in TensorFlow Lite.
//
// The `ConcreteType` deriving from this template must provide the following
// method for constructing TensorFlow Lite op:
//
//   TFL::[op] createTFLOp(ConvertTFConvOpMatchState *state,
//                         PatternRewriter &rewriter, Location loc,
//                         Type result_type, Value *input,
//                         Value *filter, Value *bias) const;
//
// And also the following method for getting the dimension for bias tensor:
//
//  int64_t getBiasDim(ArrayRef<int64_t> filterShape) const;
template <typename ConcreteType, typename TFConvOpType>
struct ConvertTFConvOp : public RewritePattern {
  // Transient state for preserving data from match to rewrite
  struct ConvertTFConvOpMatchState : public PatternState {
    IntegerAttr dilation_height_factor;
    IntegerAttr dilation_width_factor;
    StringAttr padding;
    IntegerAttr stride_height;
    IntegerAttr stride_width;
  };

  ConvertTFConvOp(MLIRContext *context)
      : RewritePattern(TFConvOpType::getOperationName(), 1, context),
        intAttrOne(Builder(context).getI32IntegerAttr(1)) {}

  PatternMatchResult match(Operation *op) const override {
    // Assumes TensorFlow convolution op is already verified to be
    // in valid form.

    // Match a TFConvOpType under the following conditions:
    // * The 'T' attribute must exist and be of value DT_FLOAT.
    // * The 'data_format' attribute must exist and be of value "NHWC".
    // * The 'strides' attribute must exist and is of the form [1, X, Y, 1].
    // * The 'dilations' attribute is optional, but it must be of the form
    //   [1, X, Y, 1] if exists.

    TFConvOpType tf_op = cast<TFConvOpType>(op);

    if (!TFTypeIsFloatTensor(tf_op.input()) || !TFDataFormatIsNHWC(op))
      return matchFailure();

    IntegerAttr height, width;
    if (!TFIntListIs1XY1(op, "strides", &height, &width)) return matchFailure();

    auto state = llvm::make_unique<ConvertTFConvOpMatchState>();

    state->stride_height = height;
    state->stride_width = width;

    if (TFIntListIs1XY1(op, "dilations", &height, &width)) {
      state->dilation_height_factor = height;
      state->dilation_width_factor = width;
    } else {
      // If the 'dilations' attribute is missing, we use the default value (1)
      // for both dilation height and width factor.
      state->dilation_height_factor = intAttrOne;
      state->dilation_width_factor = intAttrOne;
    }

    StringAttr padding_attr;
    if (!TFPaddingIsSameOrValid(op, &padding_attr)) return matchFailure();
    state->padding = padding_attr;

    // Additionally, we require the filter operand to be of 4-D tensor type so
    // that we can extract info from the shape (e.g., for constructing bias
    // tensor, for setting depth_multiplier attribute, etc.).
    auto filter_type =
        tf_op.filter()->getType().template dyn_cast<RankedTensorType>();
    if (filter_type && filter_type.getRank() == 4)
      return matchSuccess(std::move(state));

    return matchFailure();
  }

  void rewrite(Operation *op, std::unique_ptr<PatternState> state,
               PatternRewriter &rewriter) const override {
    // TensorFlow convolution op only has two inputs, while the TFLite one has
    // three, with the bias vector marked as optional. However, TOCO has a
    // dedicated pass, EnsureBiasVectors, to create default bias vectors for all
    // those missing. So we model TFLite convolution op as requiring three
    // inputs to achieve the legalization task of EnsureBiasVector. this
    // requires the filter tensor to have static shape.

    // TODO(antiagainst): also handle the case of tf.Add(tf.[op], <bias>)

    TFConvOpType tf_op = cast<TFConvOpType>(op);

    // Get a splat zero tensor with the expected dimension for the bias tensor
    auto filter = tf_op.filter();
    auto filter_type = filter->getType().template cast<RankedTensorType>();
    auto elem_type = filter_type.getElementType();
    auto bias_dim = static_cast<const ConcreteType *>(this)->getBiasDim(
        filter_type.getShape());
    auto bias_type = rewriter.getTensorType({bias_dim}, elem_type);
    auto bias_attr = rewriter.getZeroAttr(bias_type);
    auto bias = rewriter.create<ConstantOp>(op->getLoc(), bias_type, bias_attr);

    auto *conv_state = static_cast<ConvertTFConvOpMatchState *>(state.get());
    auto conv_op = static_cast<const ConcreteType *>(this)->createTFLOp(
        conv_state, rewriter, op->getLoc(), tf_op.getType(), tf_op.input(),
        filter, bias);

    rewriter.replaceOp(op, conv_op.getResult());
  }

  const IntegerAttr intAttrOne;
};

class ConvertTFConv2D : public ConvertTFConvOp<ConvertTFConv2D, TF::Conv2DOp> {
 public:
  using BaseType = ConvertTFConvOp<ConvertTFConv2D, TF::Conv2DOp>;

  ConvertTFConv2D(MLIRContext *context) : BaseType(context) {}

  int64_t getBiasDim(ArrayRef<int64_t> filterShape) const {
    return filterShape.back();
  }

  TFL::Conv2DOp createTFLOp(ConvertTFConvOpMatchState *state,
                            PatternRewriter &rewriter, Location loc,
                            Type result_type, Value *input, Value *filter,
                            Value *bias) const {
    filter = legalizeFilter(rewriter, loc, filter);
    return rewriter.create<TFL::Conv2DOp>(
        loc, result_type, input, filter, bias,
        /*dilation_h_factor=*/state->dilation_height_factor,
        /*dilation_w_factor=*/state->dilation_width_factor,
        /*fused_activation_function=*/rewriter.getStringAttr("NONE"),
        /*padding=*/state->padding,
        /*stride_h=*/state->stride_height,
        /*stride_w=*/state->stride_width);
  }

 private:
  // Legalize the given filter by converting it from TensorFlow filter data
  // format HWIO to TFLite Conv2D op filter data format OHWI and return Value
  // for the converted filter.  Requires that filter is verified by the match
  // method that it is a 4-D RankedTensorType.
  Value *legalizeFilter(PatternRewriter &rewriter, Location loc,
                        Value *filter) const {
    // Create a constant op for HWIO to OHWI transpose permutation.
    SmallVector<int, 4> perm = {3, 0, 1, 2};
    auto perm_type = rewriter.getTensorType({static_cast<int>(perm.size())},
                                            rewriter.getIntegerType(32));
    auto perm_attr =
        DenseElementsAttr::get(perm_type, llvm::makeArrayRef<int>(perm));
    auto perm_op = rewriter.create<ConstantOp>(loc, perm_type, perm_attr);

    // Create tensor type for the transpose result.
    auto filter_type = filter->getType().cast<RankedTensorType>();
    auto result_shape = functional::map(
        [filter_type](int64_t dim) { return filter_type.getDimSize(dim); },
        perm);
    auto elem_type = filter_type.getElementType();
    auto result_type = rewriter.getTensorType(result_shape, elem_type);

    return rewriter.create<TF::TransposeOp>(loc, result_type, filter, perm_op);
  }
};

class ConvertTFDepthwiseConv2dNative
    : public ConvertTFConvOp<ConvertTFDepthwiseConv2dNative,
                             TF::DepthwiseConv2dNativeOp> {
 public:
  using BaseType = ConvertTFConvOp<ConvertTFDepthwiseConv2dNative,
                                   TF::DepthwiseConv2dNativeOp>;

  ConvertTFDepthwiseConv2dNative(MLIRContext *context) : BaseType(context) {}

  int64_t getBiasDim(ArrayRef<int64_t> filterShape) const {
    return filterShape[2] * filterShape[3];
  }

  TFL::DepthwiseConv2DOp createTFLOp(ConvertTFConvOpMatchState *state,
                                     PatternRewriter &rewriter, Location loc,
                                     Type result_type, Value *input,
                                     Value *filter, Value *bias) const {
    // Compared to tfl.conv_2d, tfl.depthwise_conv_2d has an additional
    // 'depth_multiplier' attribute. However, tf.DepthwiseConv2dNative does not
    // have a corresponding 'depth_multiplier' attribute; the multiplier is the
    // fourth dimension in the 4-D filter tensor. We query the multiplier from
    // tf.DepthwiseConv2dNative and set it as the attribute value accordingly.
    auto multiplier = filter->getType().cast<RankedTensorType>().getDimSize(3);

    filter = legalizeFilter(rewriter, loc, filter);
    return rewriter.create<TFL::DepthwiseConv2DOp>(
        loc, result_type, input, filter, bias,
        /*dilation_h_factor=*/state->dilation_height_factor,
        /*dilation_w_factor=*/state->dilation_width_factor,
        /*fused_activation_function=*/rewriter.getStringAttr("NONE"),
        /*padding=*/state->padding,
        /*stride_h=*/state->stride_height,
        /*stride_w=*/state->stride_width,
        /*depth_multiplier=*/rewriter.getI32IntegerAttr(multiplier));
  }

 private:
  /// Legalize the given filter by converting it from TensorFlow filter data
  /// format to TFLite DepthwiseConv2D op filter data format and return Value
  /// for the converted filter.  TensorFlow filter data format is
  /// [filter_height, filter_width, in_channels, channel_multiplier] and TFLite
  /// filter data format is [1, filter_height, filter_width, out_channels].
  /// Requires that filter is verified by the match method that it is a 4-D
  /// RankedTensorType.
  Value *legalizeFilter(PatternRewriter &rewriter, Location loc,
                        Value *filter) const {
    auto filter_type = filter->getType().cast<RankedTensorType>();
    auto filterShape = filter_type.getShape();
    SmallVector<int64_t, 4> result_shape = {1, filterShape[0], filterShape[1],
                                            filterShape[2] * filterShape[3]};
    auto elem_type = filter_type.getElementType();
    auto result_type = rewriter.getTensorType(result_shape, elem_type);
    auto shape_type = rewriter.getTensorType({4}, rewriter.getIntegerType(64));
    auto shape_attr =
        DenseElementsAttr::get(shape_type, llvm::makeArrayRef(result_shape));
    auto shape = rewriter.create<ConstantOp>(loc, shape_type, shape_attr);

    return rewriter.create<TF::ReshapeOp>(loc, result_type, filter, shape);
  }
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_prepare_tf.inc"

void PrepareTFPass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();
  // This pattern was intented to uses TFL QDQs to preserve the quantization
  // parameters from the TF Quant ops, thus this pattern should run with the
  // first `applyPatternsGreedily` method, which would otherwise removes the
  // TF FakeQuant ops by the constant folding.
  patterns.insert<InsertTFLQuantOpsAfterTFFakeQuantOp>(&getContext());
  TFL::populateWithGenerated(&getContext(), &patterns);
  // TODO(karimnosseir): Split to separate pass probably after
  // deciding on long term plan for this optimization.
  // This will allow optimizing any TF_Mul->TF_Conv in the graph
  // and any expanded from FusedBatchNorm. We need to do this
  // before converting TF_Conv to TFL_Conv
  applyPatternsGreedily(func, std::move(patterns));

  // Load the generated pattern again, so new quantization pass-through
  // will be applied.
  TFL::populateWithGenerated(&getContext(), &patterns);
  patterns.insert<ConvertTFConv2D, ConvertTFDepthwiseConv2dNative>(
      &getContext());
  applyPatternsGreedily(func, std::move(patterns));
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect PrepareTF pass.
FunctionPassBase *CreatePrepareTFPass() { return new PrepareTFPass(); }

static PassRegistration<PrepareTFPass> pass(
    "tfl-prepare-tf", "Prepare TF for legalization to TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
