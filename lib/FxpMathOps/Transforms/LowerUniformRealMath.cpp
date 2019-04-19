//===- LowerUniformRealMath.cpp  ------------------------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "UniformKernelUtils.h"

#include "mlir/FxpMathOps/FxpMathOps.h"
#include "mlir/FxpMathOps/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/Ops.h"

using namespace mlir;
using namespace mlir::fxpmath;
using namespace mlir::fxpmath::detail;
using namespace mlir::quant;

namespace {

struct LowerUniformRealMathPass
    : public FunctionPass<LowerUniformRealMathPass> {
  void runOnFunction() override;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Elementwise add
//===----------------------------------------------------------------------===//

static LogicalResult
tryRewriteAffineAddEwIsomorphicSigned(const UniformBinaryOpInfo &info,
                                      PatternRewriter &rewriter) {
  if (!info.resultType.isSigned() || info.lhsType != info.resultType ||
      info.rhsType != info.resultType) {
    return failure();
  }

  // Choose a byte aligned intermediate width big enough to perform the
  // calculation without overflow.
  // TODO: This should probably be made just big enough to avoid overflow and
  // leave the downstream tooling to decide how to align that to machine
  // word sizes.
  unsigned intermediateWidth =
      info.resultType.getStorageTypeIntegralWidth() <= 8 ? 16 : 32;
  IntegerType intermediateElementType =
      IntegerType::get(intermediateWidth, rewriter.getContext());
  Type intermediateType =
      castElementType(info.resultStorageType, intermediateElementType);

  // Cast operands to storage type.
  Value *lhsValue = rewriter
                        .create<StorageCastOp>(info.op->getLoc(),
                                               info.lhsStorageType, info.lhs)
                        .getResult();
  Value *rhsValue = rewriter
                        .create<StorageCastOp>(info.op->getLoc(),
                                               info.rhsStorageType, info.rhs)
                        .getResult();

  // Cast to the intermediate sized type.
  lhsValue = rewriter.create<ConvertISOp>(info.op->getLoc(), intermediateType,
                                          lhsValue);
  rhsValue = rewriter.create<ConvertISOp>(info.op->getLoc(), intermediateType,
                                          rhsValue);

  // Add.
  Value *resultValue =
      rewriter.create<AddIOp>(info.op->getLoc(), lhsValue, rhsValue);

  // Zero point offset adjustment.
  // result = (lhs - zp) + (rhs - zp) + zp
  // zpOffset = -zp
  int zpOffset = -1 * info.resultType.getZeroPoint();
  if (zpOffset != 0) {
    Value *zpOffsetConst = rewriter.create<ConstantOp>(
        info.op->getLoc(),
        broadcastScalarConstIntValue(intermediateType, zpOffset));
    resultValue =
        rewriter.create<AddIOp>(info.op->getLoc(), resultValue, zpOffsetConst);
  }

  // Clamp.
  auto clampMinMax = info.getClampMinMax(intermediateElementType);
  resultValue = rewriter.create<ClampISOp>(
      info.op->getLoc(), resultValue, clampMinMax.first, clampMinMax.second);

  // Convert back to original type.
  resultValue = rewriter.create<ConvertISOp>(
      info.op->getLoc(), info.resultStorageType, resultValue);

  // Cast back for new result.
  rewriter.replaceOpWithNewOp<StorageCastOp>(
      info.op, info.getQuantizedResultType(), resultValue);

  return success();
}

//===----------------------------------------------------------------------===//
// Elementwise mul
//===----------------------------------------------------------------------===//

static LogicalResult
tryRewriteAffineMulEwSigned(const UniformBinaryOpInfo &info,
                            PatternRewriter &rewriter) {
  if (!info.resultType.isSigned()) {
    return failure();
  }

  double outputMultiplierReal = info.lhsType.getScale() *
                                info.rhsType.getScale() /
                                info.resultType.getScale();
  if (outputMultiplierReal > 1.0) {
    info.op->emitWarning("unimplemented: cannot multiply with multipler > 1.0");
    return failure();
  }

  // TODO: Choose an appropriate intermediate width for muls > 8 bits to
  // avoid overflow.
  unsigned intermediateWidth = 32;
  IntegerType intermediateElementType =
      IntegerType::get(intermediateWidth, rewriter.getContext());
  Type intermediateType =
      castElementType(info.resultStorageType, intermediateElementType);

  // Cast operands to storage type.
  Value *lhsValue = rewriter
                        .create<StorageCastOp>(info.op->getLoc(),
                                               info.lhsStorageType, info.lhs)
                        .getResult();
  Value *rhsValue = rewriter
                        .create<StorageCastOp>(info.op->getLoc(),
                                               info.rhsStorageType, info.rhs)
                        .getResult();

  // Cast to the intermediate sized type.
  lhsValue = rewriter.create<ConvertISOp>(info.op->getLoc(), intermediateType,
                                          lhsValue);
  rhsValue = rewriter.create<ConvertISOp>(info.op->getLoc(), intermediateType,
                                          rhsValue);

  // Apply argument zeroPoints.
  if (info.lhsType.getZeroPoint() != 0) {
    Value *zpOffsetConst = rewriter.create<ConstantOp>(
        info.op->getLoc(), broadcastScalarConstIntValue(
                               intermediateType, -info.lhsType.getZeroPoint()));
    lhsValue =
        rewriter.create<AddIOp>(info.op->getLoc(), lhsValue, zpOffsetConst);
  }

  if (info.rhsType.getZeroPoint() != 0) {
    Value *zpOffsetConst = rewriter.create<ConstantOp>(
        info.op->getLoc(), broadcastScalarConstIntValue(
                               intermediateType, -info.rhsType.getZeroPoint()));
    rhsValue =
        rewriter.create<AddIOp>(info.op->getLoc(), rhsValue, zpOffsetConst);
  }

  // Mul.
  Value *resultValue =
      rewriter.create<MulIOp>(info.op->getLoc(), lhsValue, rhsValue);

  // Scale output.
  QuantizedMultiplierSmallerThanOneExp outputMultiplier(outputMultiplierReal);
  resultValue = rewriter.create<VecScalarSaturatingRoundingDoublingHighMulISOp>(
      info.op->getLoc(), resultValue,
      IntegerAttr::get(intermediateElementType, outputMultiplier.multiplier));
  resultValue = rewriter.create<RoundingDivideByPotISOp>(
      info.op->getLoc(), resultValue,
      IntegerAttr::get(intermediateElementType, -outputMultiplier.exponent));

  // Zero point offset adjustment.
  if (info.resultType.getZeroPoint() != 0) {
    Value *zpOffsetConst = rewriter.create<ConstantOp>(
        info.op->getLoc(),
        broadcastScalarConstIntValue(intermediateType,
                                     info.resultType.getZeroPoint()));
    resultValue =
        rewriter.create<AddIOp>(info.op->getLoc(), resultValue, zpOffsetConst);
  }

  // Clamp.
  auto clampMinMax = info.getClampMinMax(intermediateElementType);
  resultValue = rewriter.create<ClampISOp>(
      info.op->getLoc(), resultValue, clampMinMax.first, clampMinMax.second);

  // Convert back to original type.
  resultValue = rewriter.create<ConvertISOp>(
      info.op->getLoc(), info.resultStorageType, resultValue);

  // Cast back for new result.
  rewriter.replaceOpWithNewOp<StorageCastOp>(
      info.op, info.getQuantizedResultType(), resultValue);

  return success();
}

namespace {

struct UniformRealAddEwPattern : public RewritePattern {
  UniformRealAddEwPattern(MLIRContext *context)
      : RewritePattern(RealAddEwOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const {
    auto addOp = op->cast<RealAddEwOp>();
    const UniformBinaryOpInfo info(op, addOp.x(), addOp.y(), addOp.clamp_min(),
                                   addOp.clamp_max());
    if (!info.isValid()) {
      return matchFailure();
    }

    // Try all of the permutations we support.
    if (succeeded(tryRewriteAffineAddEwIsomorphicSigned(info, rewriter))) {
      return matchSuccess();
    }

    return matchFailure();
  }
};

struct UniformRealMulEwPattern : public RewritePattern {
  UniformRealMulEwPattern(MLIRContext *context)
      : RewritePattern(RealMulEwOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const {
    auto mulOp = op->cast<RealMulEwOp>();
    const UniformBinaryOpInfo info(op, mulOp.x(), mulOp.y(), mulOp.clamp_min(),
                                   mulOp.clamp_max());
    if (!info.isValid()) {
      return matchFailure();
    }

    // Try all of the permutations we support.
    if (succeeded(tryRewriteAffineMulEwSigned(info, rewriter))) {
      return matchSuccess();
    }

    return matchFailure();
  }
};

} // end anonymous namespace

void LowerUniformRealMathPass::runOnFunction() {
  auto &fn = getFunction();
  OwningRewritePatternList patterns;
  auto *context = &getContext();
  patterns.push_back(llvm::make_unique<UniformRealAddEwPattern>(context));
  patterns.push_back(llvm::make_unique<UniformRealMulEwPattern>(context));
  applyPatternsGreedily(fn, std::move(patterns));
}

FunctionPassBase *createLowerUniformRealMathPass() {
  return new LowerUniformRealMathPass();
}

static PassRegistration<LowerUniformRealMathPass>
    pass("fxpmath-lower-uniform-real-math",
         "Lowers uniform-quantized real math ops to integer arithmetic.");
