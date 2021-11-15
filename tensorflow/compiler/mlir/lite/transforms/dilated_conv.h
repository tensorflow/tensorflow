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
// This pass identifies patterns for dilated convolution and replace it with
// a real convolution op.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_DILATED_CONV_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_DILATED_CONV_H_

#include <cstdint>

#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

// A dilated convolution can be emulated with a regular convolution by chaining
// SpaceToBatch and BatchToSpace ops before and after it:
//
//     SpaceToBatchND -> Conv2D -> BatchToSpaceND
//
// This method was common before Conv2D fully supported dilated convolution in
// TensorFlow. This transformation detects this "emulation", and replaces it
// with a true dilated convolution, eliminating the SpaceToBatch and
// BatchtoSpace ops.
//
// Detecting this alone would be relatively easy. However, in practice some
// extra ops are used, so we detect the following patterns:
//
//
//   SpaceToBatchND -> Expand -> Conv2D -> Squeeze -> BatchToSpaceND -> BiasAdd
//
//   SpaceToBatchND -> Expand -> Conv2D -> Squeeze -> Pad -> BatchToSpaceND ->
//   BiasAdd
//
//   SpaceToBatchND -> Expand -> Conv2D -> Squeeze -> BiasAdd -> BatchToSpaceND
//
//   SpaceToBatchND -> Conv2D -> Pad -> BatchToSpaceND -> BiasAdd
//
//   SpaceToBatchND -> Conv2D -> BatchToSpaceND -> BiasAdd
//
//
// The Expand/Squeeze combination is used to adapt a 3D array (such as in
// WaveNet) to the 4D arrays that Conv2D requires. Padding and BiasAdd are
// thrown in just for the extra headache. Padding adapts non-conforming input
// sizes, and can be discarded. The bias is necessary, so is kept.
template <typename Conv2dOpTy>
class ConvertTFDilatedConvOp : public OpRewritePattern<Conv2dOpTy> {
 private:
  using OpRewritePattern<Conv2dOpTy>::OpRewritePattern;

  // Extract the dilation factor from `block_shape` and pack it in an ArrayAttr.
  llvm::Optional<ArrayAttr> ExtractDilationsAttrFromBlockShape(
      Value stb_block_shape, Value bts_block_shape, int64_t expand_axis,
      PatternRewriter& rewriter) const;

 public:
  LogicalResult matchAndRewrite(Conv2dOpTy op,
                                PatternRewriter& rewriter) const override;
};

template <typename Conv2dOpTy>
LogicalResult ConvertTFDilatedConvOp<Conv2dOpTy>::matchAndRewrite(
    Conv2dOpTy op, PatternRewriter& rewriter) const {
  if (!op.getResult().hasOneUse()) {
    return rewriter.notifyMatchFailure(
        op, "result for current op has more than 1 use");
  }
  // Make sure Conv2D has 'VALID' padding.
  if (op->template getAttrOfType<StringAttr>("padding").getValue() != "VALID") {
    return rewriter.notifyMatchFailure(op,
                                       "Conv2D op doesn't have valid padding");
  }
  // Make sure dilations are all ones if set.
  const ArrayAttr& dilations =
      op->template getAttrOfType<ArrayAttr>("dilations");
  if (dilations && !TFIntListIsAllOnes(dilations)) {
    return rewriter.notifyMatchFailure(op, "dilations should be all 1");
  }

  if (!TFTypeIsFloat32Tensor(op.input()) || !TFDataFormatIsNHWC(op)) {
    return rewriter.notifyMatchFailure(
        op, "op's input is not float or the data format isn't NHWC");
  }

  // Allow dynamic width and height dimensions only.
  auto result_ty = op.getResult().getType().template cast<TensorType>();
  if (!result_ty.hasRank() || result_ty.getRank() != 4 ||
      result_ty.isDynamicDim(0) || result_ty.isDynamicDim(3)) {
    return rewriter.notifyMatchFailure(
        op, "only dynamic width and height dimensions are allowed");
  }

  // Check if the ConvOp's input is defined by `Expand` op, and the output used
  // by `Squeeze` op.
  Operation* producer_op = op.getOperand(0).getDefiningOp();
  if (!producer_op || producer_op->getNumResults() != 1) {
    return rewriter.notifyMatchFailure(
        op, "op doesn't have a producer node that has a single result");
  }
  if (!producer_op->hasOneUse() ||
      *(producer_op->getResult(0).user_begin()) != op) {
    return rewriter.notifyMatchFailure(
        op, "op's input isn't produced by previous operation");
  }

  auto tryGetDirectConsumerOp =
      [&rewriter](Operation* current) -> std::pair<LogicalResult, Operation*> {
    // Check the current operation has a single result.
    if (current->getNumResults() != 1) {
      return {
          rewriter.notifyMatchFailure(current, "op doesn't have single result"),
          nullptr};
    }
    // Check the current operation has a consumer node.
    Operation* consumer_op =
        current->getResult(0).getUses().begin()->getOwner();
    if (!consumer_op) {
      return {
          rewriter.notifyMatchFailure(current, "op doesn't have consumer node"),
          nullptr};
    }
    // Check the current operation's result is used by its successor node.
    if (!current->hasOneUse() ||
        *(current->getResult(0).user_begin()) != consumer_op) {
      return {
          rewriter.notifyMatchFailure(
              current, "op's result isn't directly consumed by the next op"),
          nullptr};
    }
    return {LogicalResult::success(), consumer_op};
  };

  std::pair<LogicalResult, Operation*> maybeConsumer =
      tryGetDirectConsumerOp(op.getOperation());
  if (failed(maybeConsumer.first)) {
    return maybeConsumer.first;
  }
  Operation* consumer_op = maybeConsumer.second;

  TF::ExpandDimsOp expand_op;
  TF::SqueezeOp squeeze_op;
  int64_t expand_axis = -1;
  // Expand + Squeeze op.
  if (llvm::isa<TF::ExpandDimsOp>(producer_op)) {
    if (!llvm::isa<TF::SqueezeOp>(consumer_op)) {
      // Expand/Squeeze op must come in pair.
      return rewriter.notifyMatchFailure(
          op, "ExpandDimsOp and SqueezeOp should come in pair");
    }
    expand_op = llvm::cast<TF::ExpandDimsOp>(producer_op);
    squeeze_op = llvm::cast<TF::SqueezeOp>(consumer_op);
    if (!expand_op.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(
          expand_op, "result for current op has more than 1 use");
    }
    if (!squeeze_op.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(
          squeeze_op, "result for current op has more than 1 use");
    }
    // Make sure that the axis in `expand_op` is constant.
    if (auto const_op =
            llvm::dyn_cast<TF::ConstOp>(expand_op.dim().getDefiningOp())) {
      expand_axis = (*const_op.value()
                          .cast<DenseElementsAttr>()
                          .getValues<APInt>()
                          .begin())
                        .getSExtValue();
      // Canonicalize axis. Some TF python functions, such as
      // `tf.nn.convolution`, use negative axis.
      if (expand_axis < 0) {
        // Always expand 3D input to 4D input.
        expand_axis += 4;
      }
    } else {
      return rewriter.notifyMatchFailure(
          expand_op, "ExpandDimsOp doesn't have a constant axis");
    }
    // Make sure that the `squeeze_dims` is equal to `expand_axis`.
    auto squeeze_dims = squeeze_op.squeeze_dims();
    if (squeeze_dims.size() != 1) {
      return rewriter.notifyMatchFailure(
          squeeze_op, "squeeze dims should have exactly 1 dimension specified");
    }
    int64_t squeeze_axis = squeeze_dims[0].cast<IntegerAttr>().getInt();
    if (squeeze_axis < 0) {
      // Always squeeze 4D input to 3D input.
      squeeze_axis += 4;
    }
    if (squeeze_axis != expand_axis) {
      return rewriter.notifyMatchFailure(
          op, "squeeze axis and expand axis doesn't match");
    }

    // Update previous/next op pointer.
    Operation* tmp = expand_op.input().getDefiningOp();
    if (!tmp || tmp->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          producer_op,
          "op doesn't have a producer node that has a single result");
    }
    if (!tmp->hasOneUse() || *(tmp->getResult(0).user_begin()) != producer_op) {
      return rewriter.notifyMatchFailure(
          producer_op, "op's input isn't defined by its previous node");
    }
    producer_op = tmp;
    std::pair<LogicalResult, Operation*> maybeConsumer =
        tryGetDirectConsumerOp(consumer_op);
    if (failed(maybeConsumer.first)) {
      return maybeConsumer.first;
    }
    consumer_op = maybeConsumer.second;
  }

  // SpaceToBatchND op.
  if (!llvm::isa<TF::SpaceToBatchNDOp>(producer_op)) {
    return rewriter.notifyMatchFailure(producer_op,
                                       "op should be a SpaceToBatchND op");
  }
  // TODO(b/149936532): Check `padding` input, currently ignored.
  TF::SpaceToBatchNDOp stb_op = llvm::cast<TF::SpaceToBatchNDOp>(producer_op);
  if (!stb_op.getResult().hasOneUse()) {
    return rewriter.notifyMatchFailure(
        stb_op, "result for current op has more than 1 use");
  }

  // Pad op.
  TF::PadOp pad_op;
  ElementsAttr pad_attr;
  if (llvm::isa<TF::PadOp>(consumer_op)) {
    pad_op = llvm::cast<TF::PadOp>(consumer_op);
    if (!pad_op.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(
          pad_op, "result for current op has more than 1 use");
    }
    std::pair<LogicalResult, Operation*> maybeConsumer =
        tryGetDirectConsumerOp(consumer_op);
    if (failed(maybeConsumer.first)) {
      return maybeConsumer.first;
    }
    consumer_op = maybeConsumer.second;
    if (!matchPattern(pad_op.paddings(), m_Constant(&pad_attr))) {
      // If the padding value isn't constant, we can't determine the padding
      // scheme for Conv2D below, in this case just reject the pattern.
      return rewriter.notifyMatchFailure(
          pad_op, "PadOp's padding value isn't constant");
    }
  }

  // BatchToSpaceND + BiasAdd.
  TF::BatchToSpaceNDOp bts_op;
  TF::BiasAddOp biasadd_op;
  bool final_op_is_bts = true;
  if (llvm::isa<TF::BiasAddOp>(consumer_op)) {
    // Must be BiasAdd + BatchToSpaceND.
    biasadd_op = llvm::cast<TF::BiasAddOp>(consumer_op);
    if (!biasadd_op.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(
          biasadd_op, "result for current op has more than 1 use");
    }
    std::pair<LogicalResult, Operation*> maybeConsumer =
        tryGetDirectConsumerOp(consumer_op);
    if (failed(maybeConsumer.first)) {
      return maybeConsumer.first;
    }
    if (!llvm::isa<TF::BatchToSpaceNDOp>(maybeConsumer.second)) {
      return rewriter.notifyMatchFailure(
          consumer_op, "op's next node isn't BatchToSpaceND op");
    }
    consumer_op = maybeConsumer.second;
    bts_op = llvm::cast<TF::BatchToSpaceNDOp>(consumer_op);
  } else if (llvm::isa<TF::BatchToSpaceNDOp>(consumer_op)) {
    // BatchToSpaceND + (optional) BiasAdd.
    bts_op = llvm::cast<TF::BatchToSpaceNDOp>(consumer_op);
    std::pair<LogicalResult, Operation*> maybeConsumer =
        tryGetDirectConsumerOp(consumer_op);
    Operation* tmp = maybeConsumer.second;
    if (tmp && llvm::isa<TF::BiasAddOp>(tmp)) {
      consumer_op = tmp;
      biasadd_op = llvm::cast<TF::BiasAddOp>(consumer_op);
      final_op_is_bts = false;
    }
  } else {
    return rewriter.notifyMatchFailure(
        consumer_op, "next op is neither BiasAdd nor BatchToSpaceND");
  }

  llvm::Optional<ArrayAttr> dilations_attr = ExtractDilationsAttrFromBlockShape(
      stb_op.block_shape(), bts_op.block_shape(), expand_axis, rewriter);
  if (!dilations_attr.hasValue()) {
    return rewriter.notifyMatchFailure(op, "failed to extract dilation rate");
  }

  if (expand_op) {
    if (stb_op.input().getType().dyn_cast<RankedTensorType>() == nullptr) {
      return rewriter.notifyMatchFailure(
          stb_op, "SpaceToBatchND op's input should have RankedTensorType");
    }
  }

  // TODO(b/149936532): Check that the input width & height are multiples of
  // dilation rate.
  // TF python library will rewrite dilated conv to
  // "SpaceToBatch->Conv->BatchToSpace" pattern, and the Conv in the middle
  // always has 'VALID' padding. The padding tensor in `SpaceToBatch` has two
  // parts of contributions, one is to reduce padding of CONV from 'SAME' to
  // 'VALID', and another is to make input shape multiples of dilation rate. The
  // first part of padding, which is also called `base_padding` will be used
  // here to determine if the original padding format is 'SAME' or 'VALID'.
  // According to the following formula we will compute the `base_padding` if
  // it's a constant. Basically, `paddings` tensor in `SpaceToBatch` and `crops`
  // tensor  in `BatchToSpace` must satisfy the following:
  //  paddings[i, 0] = base_paddings[i, 0].
  //  0 <= paddings[i, 1] - base_paddings[i, 1] < block_shape[i]
  // (input_shape[i] + paddings[i, 0] + paddings[i, 1]) % block_shape[i] == 0.
  //  crops[i, 0] = 0.
  //  crops[i, 1] = paddings[i, 1] - base_paddings[i, 1].

  //  If `paddings` - `crops` != 0, this means that `base_paddings` != 0, which
  // tells us the original padding is 'SAME' (with one caveat presented below).
  // Here we need to reset the padding back to `SAME` if `base_padding`
  // != 0.
  // TODO(b/149936532): We might not simply rely on `paddings - crops != 0` to
  // determine the original padding format. For example, users can build
  // arbitrary valid examples of `STB->Conv->BTS` which doesn't represent a
  // dilated conv, hence we shouldn't pattern match here. Instead, we need to
  // check values of `paddings` and `crops` to make sure it really stands for
  // a dilated conv.
  auto stb_paddings = stb_op.paddings();
  auto bts_crops = bts_op.crops();
  ElementsAttr stb_paddings_attr, bts_crops_attr;
  if (!matchPattern(stb_paddings, m_Constant(&stb_paddings_attr)) ||
      !matchPattern(bts_crops, m_Constant(&bts_crops_attr))) {
    return rewriter.notifyMatchFailure(
        op,
        "either SpaceToBatchND or BatchToSpaceND "
        "doesn't have constant padding/crops value");
  }
  if (stb_paddings_attr.getType() != bts_crops_attr.getType()) {
    return rewriter.notifyMatchFailure(
        stb_op,
        "SpaceToBatchND op's padding doesn't have same shape/type with "
        "BatchToSpaceND op's crops");
  }
  int64_t m = stb_paddings_attr.getType().getDimSize(0);
  // padding - crop.
  for (uint64_t i = 0; i < m; ++i) {
    for (uint64_t j = 0; j < 2; ++j) {
      // `crops` tensor has shape [M, 2], crops[i] = [crop_start, crop_end]
      // specifies the amount to crop from input dimension i + 1. If the input
      // of `BatchToSpaceND` has been padded explicitly, then we need to
      // take into account the additional padding when determining the padding
      // scheme for `Conv2D`.
      int64_t addtional_pad =
          pad_attr ? pad_attr.getValues<APInt>()[{i + 1, j}].getSExtValue() : 0;
      if (stb_paddings_attr.getValues<APInt>()[{i, j}].getSExtValue() +
              addtional_pad !=
          bts_crops_attr.getValues<APInt>()[{i, j}].getSExtValue()) {
        op->setAttr("padding", rewriter.getStringAttr("SAME"));
        break;
      }
    }
  }

  // Set dilations
  op->setAttr("dilations", dilations_attr.getValue());

  if (expand_op) {
    // If there is `expand_op`, we need to rewire the inputs to bypass the
    // `SpaceToBatch`, `BatchToSpace` and `Pad` op. E.g, turning
    // 'SpaceToBatchND -> Expand -> Conv2D -> Squeeze -> BatchToSpaceND ->
    // BiasAdd' to 'Expand -> Conv2D ->Squeeze -> BiasAdd'.

    // Connect `expand_op` with the input of `stb_op`.
    expand_op.setOperand(0, stb_op.input());
    // Calculate the shape for expand.
    auto input_shape = stb_op.input().getType().cast<ShapedType>().getShape();
    SmallVector<int64_t, 4> expand_shape(input_shape.begin(),
                                         input_shape.end());
    expand_shape.insert(expand_shape.begin() + expand_axis, 1);

    auto expand_result_type = RankedTensorType::get(
        expand_shape, getElementTypeOrSelf(stb_op.input()));
    expand_op.getResult().setType(expand_result_type);

    // Update the conv op's output shape.
    auto bts_output_shape =
        bts_op.output().getType().cast<ShapedType>().getShape();
    SmallVector<int64_t, 4> conv_result_shape(bts_output_shape.begin(),
                                              bts_output_shape.end());
    conv_result_shape.insert(conv_result_shape.begin() + expand_axis, 1);
    auto conv_result_type = RankedTensorType::get(
        conv_result_shape, getElementTypeOrSelf(stb_op.input()));
    op.getResult().setType(conv_result_type);

    squeeze_op.getResult().setType(bts_op.output().getType());

    // Connect `biasadd_op` with the output of `squeeze_op`.
    if (biasadd_op) {
      biasadd_op.setOperand(0, squeeze_op.output());
      biasadd_op.output().setType(squeeze_op.output().getType());
    }
  } else {
    if (biasadd_op) biasadd_op.setOperand(0, op.output());
    op.setOperand(0, stb_op.input());
    op.getResult().setType(bts_op.getResult().getType());
  }

  if (final_op_is_bts) {
    if (bts_op.input().getDefiningOp<TF::PadOp>()) {
      bts_op.getResult().replaceAllUsesWith(pad_op.input());
    } else {
      bts_op.getResult().replaceAllUsesWith(bts_op.input());
    }
  }

  stb_op.getResult().dropAllUses();
  return success();
}

template <typename Conv2dOpTy>
llvm::Optional<ArrayAttr>
ConvertTFDilatedConvOp<Conv2dOpTy>::ExtractDilationsAttrFromBlockShape(
    Value stb_block_shape, Value bts_block_shape, int64_t expand_axis,
    PatternRewriter& rewriter) const {
  ElementsAttr stb_bs_attr, bts_bs_attr;
  if (!matchPattern(stb_block_shape, m_Constant(&stb_bs_attr)) ||
      !matchPattern(bts_block_shape, m_Constant(&bts_bs_attr))) {
    // Returns failure status if block_shape is not a constant.
    return {};
  }
  // Check that the block_shape of `stb_op` and `bts_op` are equal.
  if (stb_bs_attr.getNumElements() != bts_bs_attr.getNumElements()) return {};
  for (uint64_t i = 0, end = stb_bs_attr.getNumElements(); i < end; ++i) {
    if (stb_bs_attr.getValues<Attribute>()[i] !=
        bts_bs_attr.getValues<Attribute>()[i])
      return {};
  }

  int dilation_h_factor = -1, dilation_w_factor = -1;
  // Set dilation factor.
  if (stb_bs_attr.getNumElements() >= 2) {
    dilation_h_factor = stb_bs_attr.getValues<APInt>()[0].getSExtValue();
    dilation_w_factor = stb_bs_attr.getValues<APInt>()[1].getSExtValue();
  } else if (stb_bs_attr.getNumElements() == 1) {
    // For 1d conv, `tf.nn.convolution` expands NWC to NHWC format after
    // `SpaceToBatchND`. Therefore, `block_shape` of `stb_op` only has one
    // dilation factor of W dim, and dilation factor of H dim is set to 1.
    if (expand_axis == 1) {
      // NWC -> NHWC
      dilation_h_factor = 1;
      dilation_w_factor = stb_bs_attr.getValues<APInt>()[0].getSExtValue();
    } else if (expand_axis == 2) {
      // NHC -> NHWC
      dilation_h_factor = stb_bs_attr.getValues<APInt>()[0].getSExtValue();
      dilation_w_factor = 1;
    }
  }

  if (dilation_h_factor == -1 || dilation_w_factor == -1) {
    return {};
  }

  return rewriter.getI64ArrayAttr({1, dilation_h_factor, dilation_w_factor, 1});
}

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_DILATED_CONV_H_
