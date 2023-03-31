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

#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMCONVFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

using tensor::CollapseShapeOp;
using tensor::ExpandShapeOp;

// The Conv2D is transformable into a matmul, if it has the following shape
//
// linalg.conv_2d_nhwc_hwcf
//   {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
//   ins(%input, %kernel : tensor<1x(N+L-1)xKx1xf32>, tensor<LxKx1xMxf32>)
//   outs(%fill : tensor<1xNx1xM>) -> tensor<1xNx1xMxf32>
//
// in that case we can tile w.r.t. L to bring it to the following form
//
// linalg.conv_2d_nhwc_hwcf
//   {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
//   ins(%input, %kernel : tensor<1xNxKx1xf32>, tensor<1xKx1xMxf32>)
//   outs(%fill : tensor<1xNx1xM>) -> tensor<1xNx1xMxf32>
bool isTransformableIntoMatmul(linalg::Conv2DNhwcHwcfOp convOp) {
  if (!convOp.hasTensorSemantics()) return false;

  Value input = convOp.getInputs()[0];
  auto inputType = input.getType().cast<RankedTensorType>();

  Value kernel = convOp.getInputs()[1];
  auto kernelType = kernel.getType().cast<RankedTensorType>();

  Value init = convOp.getOutputs()[0];
  auto initType = init.getType().cast<RankedTensorType>();

  if (!inputType.hasStaticShape() || !kernelType.hasStaticShape() ||
      !initType.hasStaticShape()) {
    return false;
  }

  auto allOnes = [](DenseIntElementsAttr attr) {
    return attr.isSplat() && attr.getValues<int64_t>()[0] == 1;
  };
  if (!allOnes(convOp.getDilations()) || !allOnes(convOp.getStrides()))
    return false;

  if (inputType.getDimSize(0) != 1 || inputType.getDimSize(3) != 1 ||
      kernelType.getDimSize(2) != 1 || initType.getDimSize(0) != 1 ||
      initType.getDimSize(2) != 1)
    return false;
  return true;
}

// linalg.conv_2d_nhwc_hwcf
//   {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
//   ins(%input, %kernel : tensor<1xNxKx1xf32>, tensor<1xKx1xMxf32>)
//   outs(%fill : tensor<1xNx1xM>) -> tensor<1xNx1xMxf32>
//
//  into
//
// linalg.matmul
//   ins(%lhs, %rhs : tensor<NxKxf32>, tensor<KxMxf32>)
//   outs(%fill : tensor<NxM>) -> tensor<1xNx1xMxf32>
FailureOr<linalg::MatmulOp> rewriteConvAsMatmul(linalg::Conv2DNhwcHwcfOp convOp,
                                                PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(convOp);
  Value input = convOp.getInputs()[0];
  Value kernel = convOp.getInputs()[1];
  Value init = convOp.getOutputs()[0];

  auto kernelType = kernel.getType().cast<RankedTensorType>();
  if (!isTransformableIntoMatmul(convOp) || kernelType.getDimSize(0) != 1)
    return failure();

  Location loc = convOp.getLoc();
  SmallVector<ReassociationIndices> map{{0, 1}, {2, 3}};
  Value newInput = rewriter.create<CollapseShapeOp>(loc, input, map);
  Value newKernel = rewriter.create<CollapseShapeOp>(loc, kernel, map);
  Value newInit = rewriter.create<CollapseShapeOp>(loc, init, map);

  auto matmul = rewriter.create<linalg::MatmulOp>(
      loc, newInit.getType(), ValueRange{newInput, newKernel},
      ValueRange{newInit});

  rewriter.replaceOpWithNewOp<ExpandShapeOp>(convOp, convOp.getType(0),
                                             matmul.getResult(0), map);
  return matmul;
}

struct Conv2DNhwcHwcfOpTransformPattern
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using OpRewritePattern<linalg::Conv2DNhwcHwcfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    if (!isTransformableIntoMatmul(convOp)) return failure();
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCFForOp(rewriter, convOp.getOperation(),
                               getSCFTilingOptions({0, 0, 0, 0, 1, 0, 0}));
    rewriter.replaceOp(convOp, tilingResult->replacements);

    auto tiledConv =
        cast<linalg::Conv2DNhwcHwcfOp>(tilingResult->tiledOps.front());
    return rewriteConvAsMatmul(tiledConv, rewriter);
  }
};

struct TransformConvForCpuPass
    : public impl::TransformConvForCpuPassBase<TransformConvForCpuPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry
        .insert<arith::ArithDialect, tensor::TensorDialect, scf::SCFDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<Conv2DNhwcHwcfOpTransformPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformConvForCpuPass() {
  return std::make_unique<mlir::gml_st::TransformConvForCpuPass>();
}

}  // namespace mlir::gml_st
