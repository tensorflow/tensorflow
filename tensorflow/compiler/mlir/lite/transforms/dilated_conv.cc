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
#include "tensorflow/compiler/mlir/lite/transforms/dilated_conv.h"

#include <optional>
#include <utility>

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace {

#define GEN_PASS_DEF_IDENTIFYDILATEDCONVPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

struct FuseDilatedConv3DPattern : public OpRewritePattern<TF::Conv3DOp> {
  using OpRewritePattern<TF::Conv3DOp>::OpRewritePattern;

  std::optional<ArrayAttr> ExtractDilationsAttrFromBlockShape(
      Value stb_block_shape, Value bts_block_shape,
      PatternRewriter& rewriter) const {
    DenseIntElementsAttr stb_bs_attr, bts_bs_attr;
    if (!matchPattern(stb_block_shape, m_Constant(&stb_bs_attr)) ||
        !matchPattern(bts_block_shape, m_Constant(&bts_bs_attr))) {
      return {};
    }
    if (stb_bs_attr.size() != bts_bs_attr.size() || stb_bs_attr.size() != 3) {
      return {};
    }
    llvm::SmallVector<int64_t> stb_bs;
    for (int64_t i = 0, end = stb_bs_attr.size(); i < end; i++) {
      if (stb_bs_attr.getValues<APInt>()[i].getSExtValue() !=
          bts_bs_attr.getValues<APInt>()[i].getSExtValue()) {
        return {};
      }
      stb_bs.push_back(stb_bs_attr.getValues<APInt>()[i].getSExtValue());
    }

    return rewriter.getI64ArrayAttr({1, stb_bs[0], stb_bs[1], stb_bs[2], 1});
  }

  LogicalResult matchAndRewrite(TF::Conv3DOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(
          op, "result for current op has more than 1 use");
    }
    // Make sure Conv3D has 'VALID' padding.
    if (op->template getAttrOfType<StringAttr>("padding").getValue() !=
        "VALID") {
      return rewriter.notifyMatchFailure(
          op, "Conv3D op doesn't have valid padding");
    }
    // Make sure dilations are all ones if set.
    const ArrayAttr& dilations =
        op->template getAttrOfType<ArrayAttr>("dilations");
    if (dilations && !TFL::TFIntListIsAllOnes(dilations)) {
      return rewriter.notifyMatchFailure(op, "dilations should be all 1");
    }

    if (!TFL::TFTypeIsFloat32Tensor(op.getInput()) &&
        !TFL::TFTypeIsBFloat16OrHalfTensor(op.getInput())) {
      return rewriter.notifyMatchFailure(
          op, "op's input is not float or half or bfloat16");
    }
    if (!TFL::TFDataFormatIsNDHWC(op)) {
      return rewriter.notifyMatchFailure(op, "op's data format isn't NDHWC");
    }
    // Allow dynamic batch, width and height dimensions only.
    auto result_ty = op.getResult().getType().template cast<TensorType>();
    if (!result_ty.hasRank() || result_ty.getRank() != 5 ||
        result_ty.isDynamicDim(4)) {
      return rewriter.notifyMatchFailure(
          op, "only dynamic batch, width and height dimensions are allowed");
    }

    TF::SpaceToBatchNDOp stb_op =
        op.getInput().getDefiningOp<TF::SpaceToBatchNDOp>();
    if (!stb_op || !stb_op.getResult().hasOneUse()) {
      return failure();
    }
    if (!stb_op.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(
          stb_op, "stb_op's result doesn't have one use");
    }

    Operation* consumer_op = op.getResult().getUses().begin()->getOwner();
    if (!consumer_op) {
      return rewriter.notifyMatchFailure(op, "op doesn't have consumer node");
    }

    TF::BatchToSpaceNDOp bts_op =
        llvm::dyn_cast<TF::BatchToSpaceNDOp>(consumer_op);
    if (!bts_op) {
      return rewriter.notifyMatchFailure(consumer_op,
                                         "consumer op isn't BatchToSpace Op");
    }

    DenseIntElementsAttr paddings, crops;
    if (!matchPattern(stb_op.getPaddings(), m_Constant(&paddings)) ||
        !matchPattern(bts_op.getCrops(), m_Constant(&crops))) {
      return rewriter.notifyMatchFailure(
          stb_op,
          "either SpaceToBatch or BatchToSpaceND doesn't have constant "
          "paddings/crops value");
    }
    if (paddings.getType() != crops.getType()) {
      return rewriter.notifyMatchFailure(
          stb_op,
          "SpaceToBatchND op's padding doesn't have same shape/type with "
          "BatchToSpaceND op's crops");
    }
    std::optional<ArrayAttr> dilations_attr =
        ExtractDilationsAttrFromBlockShape(stb_op.getBlockShape(),
                                           bts_op.getBlockShape(), rewriter);
    if (!dilations_attr.has_value()) {
      return rewriter.notifyMatchFailure(stb_op,
                                         "failed to extract dilation rate");
    }

    // note: the padding setting copied from dilated_conv.h
    int64_t m = paddings.getType().getDimSize(0);
    for (uint64_t i = 0; i < m; i++) {
      for (uint64_t j = 0; j < 2; j++) {
        if (paddings.getValues<APInt>()[{i, j}].getSExtValue() !=
            crops.getValues<APInt>()[{i, j}].getSExtValue()) {
          op->setAttr("padding", rewriter.getStringAttr("SAME"));
          break;
        }
      }
    }

    op->setAttr("dilations", dilations_attr.value());
    op.setOperand(0, stb_op.getInput());
    op.getResult().setType(bts_op.getResult().getType());
    bts_op.getResult().replaceAllUsesWith(bts_op.getInput());
    stb_op.getResult().dropAllUses();
    return success();
  }
};

struct IdentifyDilatedConvPass
    : public impl::IdentifyDilatedConvPassBase<IdentifyDilatedConvPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IdentifyDilatedConvPass)
  void runOnOperation() override;
};

void IdentifyDilatedConvPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();

  patterns.add<ConvertTFDilatedConvOp<TF::Conv2DOp>,
               ConvertTFDilatedConvOp<TF::DepthwiseConv2dNativeOp>,
               FuseDilatedConv3DPattern>(&getContext());
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
}  // namespace
std::unique_ptr<OperationPass<func::FuncOp>> CreateIdentifyDilatedConvPass() {
  return std::make_unique<IdentifyDilatedConvPass>();
}

}  // namespace TFL
}  // namespace mlir
