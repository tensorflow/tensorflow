
/* Copyright 2025 The StableHLO Authors.
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

#include <cstdint>
#include <iterator>
#include <utility>

#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/PassUtils.h"

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_STABLEHLOADDQDQAFTERCONVPASS
#include "stablehlo_ext/transforms/passes.h.inc"

namespace {

Type getQuantizedType(Location loc, PatternRewriter& rewriter,
                      ShapedType type) {
  // Create per-tensor uniform quantize op
  auto storageType = rewriter.getIntegerType(32);
  auto expressedType = type.getElementType();
  double scale = 1.0;
  int64_t zeroPoint = 0;
  auto storageTypeMin = quant::QuantizedType::getDefaultMinimumForInteger(
      !storageType.isUnsignedInteger(), storageType.getIntOrFloatBitWidth());
  auto storageTypeMax = quant::QuantizedType::getDefaultMaximumForInteger(
      !storageType.isUnsignedInteger(), storageType.getIntOrFloatBitWidth());

  auto quantizedElementType = stablehlo::getQuantizedElementType(
      loc, storageType, expressedType, {scale}, {zeroPoint}, -1, storageTypeMin,
      storageTypeMax);
  return RankedTensorType::get(type.getShape(), quantizedElementType);
}

struct AddQuantDeQuantAfterConvolutionOp final
    : OpRewritePattern<mlir::stablehlo::ConvolutionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConvolutionOp op,
                                PatternRewriter& rewriter) const override {
    // Match a stablehlo.convolution op if
    // 1. Its operands are defined by stablehlo.uniform_dequantize op,
    // 2. It has a single user.
    // 3. The user is NOT a stablehlo.uniform_quantize op.

    for (auto operand : op.getOperands()) {
      if (!operand.getDefiningOp<stablehlo::UniformDequantizeOp>()) {
        return failure();
      }
    }

    int numUsers = std::distance(op->getUsers().begin(), op->getUsers().end());
    if (numUsers != 1) return failure();

    if (isa<stablehlo::UniformQuantizeOp>(*op->getUsers().begin()))
      return failure();

    auto* clonedConvOp = rewriter.clone(*op);
    auto convResultType =
        cast<ShapedType>(clonedConvOp->getResult(0).getType());
    auto loc = clonedConvOp->getLoc();
    auto quantizedType = getQuantizedType(loc, rewriter, convResultType);
    auto stablehloQuantizeOp = rewriter.create<stablehlo::UniformQuantizeOp>(
        op.getLoc(), quantizedType, clonedConvOp->getResult(0));
    auto stablehloDeQuantizeOp =
        rewriter.create<stablehlo::UniformDequantizeOp>(
            op.getLoc(), op.getType(),
            /*input=*/stablehloQuantizeOp.getResult());
    rewriter.replaceAllUsesWith(op, stablehloDeQuantizeOp.getResult());
    return success();
  }
};

class StablehloAddQDQAfterConvPass
    : public impl::StablehloAddQDQAfterConvPassBase<
          StablehloAddQDQAfterConvPass> {
 public:
  void runOnOperation() override {
    MLIRContext& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AddQuantDeQuantAfterConvolutionOp>(&ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace stablehlo_ext
}  // namespace mlir
