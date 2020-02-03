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
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Matchers.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
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
      Value stb_block_shape, Value bts_block_shape,
      PatternRewriter& rewriter) const;

 public:
  PatternMatchResult matchAndRewrite(Conv2dOpTy op,
                                     PatternRewriter& rewriter) const override;
};

template <typename Conv2dOpTy>
PatternMatchResult ConvertTFDilatedConvOp<Conv2dOpTy>::matchAndRewrite(
    Conv2dOpTy op, PatternRewriter& rewriter) const {
  // Check if the ConvOp is preceded by a `Expand` op and succeeded by a
  // `Squeeze` op.
  Operation* prev_op = op.getOperation()->getPrevNode();
  if (!prev_op) return Pattern::matchFailure();

  Operation* next_op = op.getOperation()->getNextNode();
  if (!next_op) return Pattern::matchFailure();

  TF::ExpandDimsOp expand_op;
  TF::SqueezeOp squeeze_op;
  // Expand + Squeeze op.
  if (llvm::isa<TF::ExpandDimsOp>(prev_op)) {
    if (!llvm::isa<TF::SqueezeOp>(next_op)) {
      // Expand/Squeeze op must come in pair.
      return Pattern::matchFailure();
    }
    expand_op = llvm::cast<TF::ExpandDimsOp>(prev_op);
    squeeze_op = llvm::cast<TF::SqueezeOp>(next_op);

    // Update previous/next op pointer.
    prev_op = prev_op->getPrevNode();
    if (!prev_op) return Pattern::matchFailure();
    next_op = next_op->getNextNode();
    if (!next_op) return Pattern::matchFailure();
  }

  // SpaceToBatchND op.
  if (!llvm::isa<TF::SpaceToBatchNDOp>(prev_op)) return Pattern::matchFailure();
  TF::SpaceToBatchNDOp stb_op = llvm::cast<TF::SpaceToBatchNDOp>(prev_op);

  // Pad op.
  TF::PadOp pad_op;
  if (llvm::isa<TF::PadOp>(next_op)) {
    pad_op = llvm::cast<TF::PadOp>(next_op);
    next_op = next_op->getNextNode();
    if (!next_op) return Pattern::matchFailure();
  }

  // BatchToSpaceND + BiasAdd.
  TF::BatchToSpaceNDOp bts_op;
  TF::BiasAddOp biasadd_op;
  bool final_op_is_bts = true;
  if (llvm::isa<TF::BiasAddOp>(next_op)) {
    // Must be BiasAdd + BatchToSpaceND.
    biasadd_op = llvm::cast<TF::BiasAddOp>(next_op);
    next_op = next_op->getNextNode();
    if (!next_op || !llvm::isa<TF::BatchToSpaceNDOp>(next_op))
      return Pattern::matchFailure();
    bts_op = llvm::cast<TF::BatchToSpaceNDOp>(next_op);
  } else if (llvm::isa<TF::BatchToSpaceNDOp>(next_op)) {
    // BatchToSpaceND + (optional) BiasAdd.
    bts_op = llvm::cast<TF::BatchToSpaceNDOp>(next_op);
    next_op = next_op->getNextNode();
    if (next_op && llvm::isa<TF::BiasAddOp>(next_op)) {
      biasadd_op = llvm::cast<TF::BiasAddOp>(next_op);
      final_op_is_bts = false;
    }
  } else {
    return Pattern::matchFailure();
  }

  llvm::Optional<ArrayAttr> dilations_attr = ExtractDilationsAttrFromBlockShape(
      stb_op.block_shape(), bts_op.block_shape(), rewriter);
  if (!dilations_attr.hasValue()) return Pattern::matchFailure();
  op.setAttr("dilations", dilations_attr.getValue());

  // Here we need to set the correct padding for Conv op. In TF, the conv op
  // inserted after 'SpaceToBatch' always has 'VALID' padding. This might
  // become a problem here if the original Conv op has 'SAME' padding. When
  // the original conv has 'SAME' padding, TF will set a non-zero padding for
  // the 'SpaceToBatch' op, so we rely on this information to check if we need
  // to change the padding from 'VALID' to 'SAME' (a.k.a when we see non-zero
  // values in `stb_op.paddings`, we change the current Conv's padding to
  // 'SAME').
  auto stb_paddings = stb_op.paddings();
  ElementsAttr stb_paddings_attr;
  if (matchPattern(stb_paddings, m_Constant(&stb_paddings_attr))) {
    if (llvm::any_of(stb_paddings_attr.getValues<IntegerAttr>(),
                     [](IntegerAttr attr) { return attr.getInt() != 0; })) {
      op.setAttr("padding", rewriter.getStringAttr("SAME"));
    }
  }

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
    expand_shape.push_back(1);
    auto expand_result_type = RankedTensorType::get(
        expand_shape, getElementTypeOrSelf(stb_op.input()));
    expand_op.getResult().setType(expand_result_type);
    op.getResult().setType(expand_result_type);

    squeeze_op.getResult().setType(bts_op.output().getType());

    // Connect `biasadd_op` with the output of `squeeze_op`.
    biasadd_op.setOperand(0, squeeze_op.output());
    biasadd_op.output().setType(squeeze_op.output().getType());
  } else {
    if (biasadd_op) biasadd_op.setOperand(0, op.output());
    op.setOperand(0, stb_op.input());
    op.getResult().setType(bts_op.getResult().getType());
  }

  if (final_op_is_bts) {
    bts_op.getResult().replaceAllUsesWith(bts_op.input());
  }

  stb_op.getResult().dropAllUses();
  return Pattern::matchSuccess();
}

template <typename Conv2dOpTy>
llvm::Optional<ArrayAttr>
ConvertTFDilatedConvOp<Conv2dOpTy>::ExtractDilationsAttrFromBlockShape(
    Value stb_block_shape, Value bts_block_shape,
    PatternRewriter& rewriter) const {
  ElementsAttr stb_bs_attr, bts_bs_attr;
  if (!matchPattern(stb_block_shape, m_Constant(&stb_bs_attr)) ||
      !matchPattern(bts_block_shape, m_Constant(&bts_bs_attr))) {
    // Returns failure status if block shape is not a constant.
    return {};
  }
  // Check that the block_shape of `stb_op` and `bts_op` are equal.
  if (stb_bs_attr.getNumElements() != bts_bs_attr.getNumElements()) return {};
  for (uint64_t i = 0; i < stb_bs_attr.getNumElements(); ++i) {
    if (stb_bs_attr.getValue({i}) != bts_bs_attr.getValue({i})) return {};
  }

  // TODO(haoliang): support 1-D dilated conv.
  if (stb_bs_attr.getNumElements() < 2) return {};

  int dilation_h_factor =
      stb_bs_attr.getValue({0}).cast<IntegerAttr>().getInt();
  int dilation_w_factor =
      stb_bs_attr.getValue({1}).cast<IntegerAttr>().getInt();

  return rewriter.getI64ArrayAttr({1, dilation_h_factor, dilation_w_factor, 1});
}

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_DILATED_CONV_H_
