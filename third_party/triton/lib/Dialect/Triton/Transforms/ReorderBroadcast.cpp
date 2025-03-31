#include <memory>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

// TODO(jlebar): Move this and all other generatede code into namespace
// mlir::triton.
#define GEN_PASS_DEF_TRITONREORDERBROADCAST
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace mlir::triton {
namespace {

Operation *cloneWithNewArgsAndResultTypes(PatternRewriter &rewriter,
                                          Operation *op, ValueRange newOperands,
                                          TypeRange newTypes) {
  OperationState newElementwiseState(op->getLoc(), op->getName());
  newElementwiseState.addOperands(newOperands);
  newElementwiseState.addTypes(newTypes);
  newElementwiseState.addAttributes(op->getAttrs());
  return rewriter.create(newElementwiseState);
}

bool isSplat(Operation *op) {
  if (auto splatOp = llvm::dyn_cast<SplatOp>(op)) {
    return true;
  }
  DenseElementsAttr constAttr;
  return (matchPattern(op, m_Constant(&constAttr)) && constAttr.isSplat());
}

// elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))
struct MoveSplatAfterElementwisePattern
    : public OpTraitRewritePattern<OpTrait::Elementwise> {

  MoveSplatAfterElementwisePattern(MLIRContext *context)
      : OpTraitRewritePattern(context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isMemoryEffectFree(op)) {
      return failure();
    }

    for (auto operand : op->getOperands()) {
      auto definingOp = operand.getDefiningOp();
      if (!definingOp)
        return failure();

      if (!isSplat(definingOp)) {
        return failure();
      }
    }

    if (op->getNumOperands() <= 0)
      return failure();

    auto loc = op->getLoc();
    auto operands = op->getOperands();

    llvm::SmallVector<Value, 4> scalarOperands(operands.size());
    for (unsigned iOp = 0; iOp < operands.size(); ++iOp) {
      auto definingOp = operands[iOp].getDefiningOp();

      DenseElementsAttr constAttr;
      if (auto splatOp = llvm::dyn_cast<SplatOp>(definingOp)) {
        scalarOperands[iOp] = splatOp.getSrc();
      } else if (matchPattern(definingOp, m_Constant(&constAttr)) &&
                 constAttr.isSplat()) {
        auto value = constAttr.getSplatValue<Attribute>();
        scalarOperands[iOp] = arith::ConstantOp::materialize(
            rewriter, value, constAttr.getElementType(), loc);
      } else {
        llvm_unreachable("Expected a splat");
      }
    }

    auto resultTypes = op->getResultTypes();
    llvm::SmallVector<Type, 4> scalarResultTys;
    for (auto resultTy : resultTypes) {
      auto elemTy = dyn_cast<TensorType>(resultTy).getElementType();
      scalarResultTys.push_back(elemTy);
    }

    auto newOp = cloneWithNewArgsAndResultTypes(rewriter, op, scalarOperands,
                                                scalarResultTys);

    for (unsigned iRes = 0; iRes < resultTypes.size(); ++iRes) {
      auto newResult = rewriter.create<SplatOp>(loc, resultTypes[iRes],
                                                newOp->getResult(iRes));
      rewriter.replaceAllUsesWith(op->getResult(iRes), newResult);
    }
    return success();
  }
};

// elementwise(broadcast(a)) => broadcast(elementwise(a))
// This also generalizes to multiple arguments when the rest are splat-like
// Not handled: multiple broadcasted arguments
struct MoveBroadcastAfterElementwisePattern
    : public OpTraitRewritePattern<OpTrait::Elementwise> {

  MoveBroadcastAfterElementwisePattern(MLIRContext *context)
      : OpTraitRewritePattern(context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isMemoryEffectFree(op)) {
      return failure();
    }

    auto operands = op->getOperands();
    bool seenBroadcast = false;
    ArrayRef<int64_t> srcShape;
    for (auto operand : operands) {
      auto definingOp = operand.getDefiningOp();
      if (!definingOp) {
        return failure();
      }
      auto getSrcShape = [](BroadcastOp b) {
        return b.getSrc().getType().getShape();
      };
      if (auto broadcastOp = llvm::dyn_cast<BroadcastOp>(definingOp)) {
        if (!seenBroadcast) {
          seenBroadcast = true;
          srcShape = getSrcShape(broadcastOp);
        } else if (srcShape != getSrcShape(broadcastOp)) {
          // If the broadcast have different types we cannot re-order.
          return failure();
        }
      } else if (!isSplat(definingOp)) {
        // Not splat or broadcast
        return failure();
      }
    }
    if (!seenBroadcast)
      return failure();

    auto loc = op->getLoc();

    // Find broadcast op
    BroadcastOp broadcastOp;
    for (auto operand : operands) {
      broadcastOp = operand.getDefiningOp<BroadcastOp>();
      if (broadcastOp) {
        break;
      }
    }

    auto srcTy = broadcastOp.getSrc().getType();
    auto bcSrcShape = srcTy.getShape();
    auto srcEncoding = srcTy.getEncoding();

    // Reshape operands to match srcShape
    llvm::SmallVector<Value, 4> newOperands;
    for (auto operand : operands) {
      auto definingOp = operand.getDefiningOp();
      if (auto broadcastSrcOp = llvm::dyn_cast<BroadcastOp>(definingOp)) {
        newOperands.push_back(broadcastSrcOp.getSrc());
        continue;
      }
      auto elemTy =
          dyn_cast<RankedTensorType>(operand.getType()).getElementType();
      auto newTy = RankedTensorType::get(bcSrcShape, elemTy, srcEncoding);
      if (auto splatOp = llvm::dyn_cast<SplatOp>(definingOp)) {
        auto newSplat = rewriter.create<SplatOp>(loc, newTy, splatOp.getSrc());
        newOperands.push_back(newSplat);
        continue;
      }
      DenseElementsAttr constAttr;
      if (matchPattern(definingOp, m_Constant(&constAttr)) &&
          constAttr.isSplat()) {
        auto scalarValue = constAttr.getSplatValue<Attribute>();
        auto splatValue = SplatElementsAttr::get(newTy, scalarValue);
        auto newConstant =
            rewriter.create<arith::ConstantOp>(loc, newTy, splatValue);
        newOperands.push_back(newConstant);
        continue;
      }
      llvm_unreachable("Expected broadcast or splat");
    }

    // Reshape results to match srcShape
    llvm::SmallVector<Type, 4> newResultTypes;
    auto resultTypes = op->getResultTypes();
    for (auto resultTy : resultTypes) {
      auto elemTy = dyn_cast<RankedTensorType>(resultTy).getElementType();
      newResultTypes.push_back(
          RankedTensorType::get(bcSrcShape, elemTy, srcEncoding));
    }

    // Create new op and broadcast results
    auto newOp = cloneWithNewArgsAndResultTypes(rewriter, op, newOperands,
                                                newResultTypes);
    for (unsigned iRes = 0; iRes < newResultTypes.size(); ++iRes) {
      auto newResult = rewriter.create<BroadcastOp>(loc, resultTypes[iRes],
                                                    newOp->getResult(iRes));
      rewriter.replaceAllUsesWith(op->getResult(iRes), newResult);
    }
    return success();
  }
};

class ReorderBroadcastPass
    : public ::impl::TritonReorderBroadcastBase<ReorderBroadcastPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp m = getOperation();

    BroadcastOp::getCanonicalizationPatterns(patterns, context);
    ExpandDimsOp::getCanonicalizationPatterns(patterns, context);
    // elementwise(broadcast(a)) => broadcast(elementwise(a))
    patterns.add<MoveBroadcastAfterElementwisePattern>(context);
    // elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))
    patterns.add<MoveSplatAfterElementwisePattern>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createReorderBroadcastPass() {
  return std::make_unique<ReorderBroadcastPass>();
}

} // namespace mlir::triton
