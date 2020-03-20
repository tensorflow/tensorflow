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
      Value stb_block_shape, Value bts_block_shape,
      PatternRewriter& rewriter) const;

 public:
  LogicalResult matchAndRewrite(Conv2dOpTy op,
                                PatternRewriter& rewriter) const override;
};

template <typename Conv2dOpTy>
LogicalResult ConvertTFDilatedConvOp<Conv2dOpTy>::matchAndRewrite(
    Conv2dOpTy op, PatternRewriter& rewriter) const {
  // Make sure Conv2D has 'VALID' padding.
  if (op.template getAttrOfType<StringAttr>("padding").getValue() != "VALID") {
    return failure();
  }
  // Make sure dilations are all ones if set.
  const ArrayAttr& dilations =
      op.template getAttrOfType<ArrayAttr>("dilations");
  if (dilations && !TFIntListIsAllOnes(dilations)) {
    return failure();
  }

  // Check if the ConvOp is preceded by a `Expand` op and succeeded by a
  // `Squeeze` op.
  Operation* prev_op = op.getOperation()->getPrevNode();
  if (!prev_op) return failure();

  Operation* next_op = op.getOperation()->getNextNode();
  if (!next_op) return failure();

  TF::ExpandDimsOp expand_op;
  TF::SqueezeOp squeeze_op;
  int64_t expand_axis;
  // Expand + Squeeze op.
  if (llvm::isa<TF::ExpandDimsOp>(prev_op)) {
    if (!llvm::isa<TF::SqueezeOp>(next_op)) {
      // Expand/Squeeze op must come in pair.
      return failure();
    }
    expand_op = llvm::cast<TF::ExpandDimsOp>(prev_op);
    squeeze_op = llvm::cast<TF::SqueezeOp>(next_op);

    // Make sure that the axis in `expand_op` is constant.
    if (auto const_op =
            llvm::dyn_cast<TF::ConstOp>(expand_op.dim().getDefiningOp())) {
      expand_axis =
          (*const_op.value().cast<DenseElementsAttr>().getIntValues().begin())
              .getSExtValue();
    } else {
      return failure();
    }
    // Make sure that the `squeeze_dims` is equal to `expand_axis`.
    auto squeeze_dims = squeeze_op.squeeze_dims();
    if (squeeze_dims.size() != 1 ||
        squeeze_dims[0].cast<IntegerAttr>().getInt() != expand_axis) {
      return failure();
    }

    // Update previous/next op pointer.
    prev_op = prev_op->getPrevNode();
    if (!prev_op) return failure();
    next_op = next_op->getNextNode();
    if (!next_op) return failure();
  }

  // SpaceToBatchND op.
  if (!llvm::isa<TF::SpaceToBatchNDOp>(prev_op)) return failure();
  // TODO(b/149936532): Check `padding` input, currently ignored.
  TF::SpaceToBatchNDOp stb_op = llvm::cast<TF::SpaceToBatchNDOp>(prev_op);

  // Pad op.
  TF::PadOp pad_op;
  // TODO(b/149936532): Currently we just ignore the PadOp. However note that
  // in real scenarios this may not always be correct: user can put a PadOp here
  // with non-trivial consequences.
  if (llvm::isa<TF::PadOp>(next_op)) {
    pad_op = llvm::cast<TF::PadOp>(next_op);
    next_op = next_op->getNextNode();
    if (!next_op) return failure();
  }

  // BatchToSpaceND + BiasAdd.
  // TODO(b/149936532): Check the `crops` input, currently ignored.
  TF::BatchToSpaceNDOp bts_op;
  TF::BiasAddOp biasadd_op;
  bool final_op_is_bts = true;
  if (llvm::isa<TF::BiasAddOp>(next_op)) {
    // Must be BiasAdd + BatchToSpaceND.
    biasadd_op = llvm::cast<TF::BiasAddOp>(next_op);
    next_op = next_op->getNextNode();
    if (!next_op || !llvm::isa<TF::BatchToSpaceNDOp>(next_op)) return failure();
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
    return failure();
  }

  llvm::Optional<ArrayAttr> dilations_attr = ExtractDilationsAttrFromBlockShape(
      stb_op.block_shape(), bts_op.block_shape(), rewriter);
  if (!dilations_attr.hasValue()) return failure();
  op.setAttr("dilations", dilations_attr.getValue());

  // Padding is set to 'SAME' when `stb_op` has non-zero paddings.
  // TODO(b/149936532): This assumption only holds when the input width & height
  // is multiple of dilation width & height. We should fix it in order to
  // support other use cases.
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
    expand_shape.insert(expand_shape.begin() + expand_axis, 1);

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
  return success();
}

template <typename Conv2dOpTy>
llvm::Optional<ArrayAttr>
ConvertTFDilatedConvOp<Conv2dOpTy>::ExtractDilationsAttrFromBlockShape(
    Value stb_block_shape, Value bts_block_shape,
    PatternRewriter& rewriter) const {
  ElementsAttr stb_bs_attr, bts_bs_attr;
  if (!matchPattern(stb_block_shape, m_Constant(&stb_bs_attr)) ||
      !matchPattern(bts_block_shape, m_Constant(&bts_bs_attr))) {
    // Returns failure status if block_shape is not a constant.
    return {};
  }
  // Check that the block_shape of `stb_op` and `bts_op` are equal.
  if (stb_bs_attr.getNumElements() != bts_bs_attr.getNumElements()) return {};
  for (uint64_t i = 0; i < stb_bs_attr.getNumElements(); ++i) {
    if (stb_bs_attr.getValue({i}) != bts_bs_attr.getValue({i})) return {};
  }

  // Set dilation factor.
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
