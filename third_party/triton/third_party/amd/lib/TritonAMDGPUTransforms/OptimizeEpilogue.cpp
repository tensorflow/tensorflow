/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;

namespace {

bool isOneOperandElementwiseOp(Operation *op) {
  if (llvm::isa<arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::FPToSIOp,
                arith::FPToUIOp, arith::NegFOp, arith::SIToFPOp,
                arith::TruncFOp, arith::TruncIOp, arith::UIToFPOp>(op))
    return true;
  if (llvm::isa<math::AbsFOp, math::AbsIOp, math::AtanOp, math::Atan2Op,
                math::CeilOp, math::CosOp, math::SinOp,
                math::CountLeadingZerosOp, math::CountTrailingZerosOp,
                math::CtPopOp, math::ErfOp, math::ExpOp, math::Exp2Op,
                math::ExpM1Op, math::FloorOp, math::LogOp, math::Log10Op,
                math::Log1pOp, math::Log2Op, math::SqrtOp, math::RsqrtOp,
                math::TanhOp>(op))
    return true;
  if (llvm::isa<triton::IntToPtrOp, triton::PtrToIntOp, triton::BitcastOp,
                triton::FpToFpOp>(op))
    return true;
  if (auto externElementwiseOp = dyn_cast<triton::ExternElementwiseOp>(op))
    return op->getNumOperands() == 1 && op->getNumResults() == 1 &&
           externElementwiseOp.getPure();
  return false;
}

// convert(val) : xmma -> blocked
// elementWiseOp(val) : blocked
// ...
// elementWiseOp(val) : blocked
// tt.store(ptr, val, mask, ...) : blocked
// ==>
// convert(ptr) : blocked -> xmma
// convert(mask) : blocked -> xmma
// elementWiseOp(val) : xmma
// ...
// elementWiseOp(val) : xmma
// tt.store(ptr, val, mask, ...) : xmma
//
// Store with xmma layout directly
//
// xmma layout is either MFMA or WMMA
class BypassEpilogueSMEM : public mlir::RewritePattern {

public:
  explicit BypassEpilogueSMEM(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::StoreOp::getOperationName(), 1, context) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {

    auto stOp = dyn_cast<triton::StoreOp>(op);
    if (!stOp)
      return mlir::failure();
    Value ptr = stOp.getPtr();
    Value val = stOp.getValue();
    Value mask = stOp.getMask();
    auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());
    auto valType = dyn_cast<RankedTensorType>(val.getType());
    if (!ptrType || !valType ||
        !isa<triton::gpu::BlockedEncodingAttr>(ptrType.getEncoding()) ||
        !isa<triton::gpu::BlockedEncodingAttr>(valType.getEncoding()))
      return mlir::failure();

    llvm::SmallVector<mlir::Operation *> chainedOps;
    while (true) {
      auto chainedOp = val.getDefiningOp();
      if (!chainedOp)
        return mlir::failure();
      if (llvm::isa<triton::gpu::ConvertLayoutOp>(chainedOp))
        break;
      if (!chainedOp->hasOneUse())
        return mlir::failure();
      if (!isOneOperandElementwiseOp(chainedOp))
        return mlir::failure();
      val = chainedOp->getOperand(0);
      chainedOps.push_back(chainedOp);
    }

    auto cvtOp = val.getDefiningOp<triton::gpu::ConvertLayoutOp>();
    if (!cvtOp)
      return mlir::failure();

    auto encoding = cvtOp.getSrc().getType().getEncoding();
    if (!isa<triton::gpu::MmaEncodingTrait>(encoding))
      return mlir::failure();

    if (!cvtOp.getResult().hasOneUse())
      return mlir::failure();

    auto newEncoding =
        cast<RankedTensorType>(cvtOp.getSrc().getType()).getEncoding();

    auto newVal = cvtOp.getSrc();

    auto newPtrType = RankedTensorType::get(
        ptrType.getShape(), ptrType.getElementType(), newEncoding);
    Value newPtr = rewriter.create<triton::gpu::ConvertLayoutOp>(
        ptr.getLoc(), newPtrType, ptr);

    for (auto chainedOp : llvm::reverse(chainedOps)) {
      auto oldType =
          cast<mlir::RankedTensorType>(chainedOp->getResult(0).getType());
      chainedOp->setOperand(0, newVal);
      newVal = llvm::cast<mlir::TypedValue<RankedTensorType>>(
          chainedOp->getResult(0));
      auto newType = mlir::RankedTensorType::get(
          oldType.getShape(), oldType.getElementType(), newEncoding);
      newVal.setType(newType);
    }

    Value newMask = mask;
    if (mask) {
      auto maskType = dyn_cast<RankedTensorType>(mask.getType());
      auto newMaskType = RankedTensorType::get(
          maskType.getShape(), maskType.getElementType(), newEncoding);
      newMask = rewriter.create<triton::gpu::ConvertLayoutOp>(
          mask.getLoc(), newMaskType, mask);
    }

    rewriter.replaceOpWithNewOp<triton::StoreOp>(
        stOp, newPtr, newVal, newMask, stOp.getCache(), stOp.getEvict());
    return mlir::success();
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPUOptimizeEpiloguePass
    : public TritonAMDGPUOptimizeEpilogueBase<
          TritonAMDGPUOptimizeEpiloguePass> {

public:
  TritonAMDGPUOptimizeEpiloguePass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);

    patterns.add<BypassEpilogueSMEM>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUOptimizeEpiloguePass() {
  return std::make_unique<TritonAMDGPUOptimizeEpiloguePass>();
}
