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

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace mlir::triton {
namespace {

bool isZero(Value val) {
  return (matchPattern(val, m_Zero()) || matchPattern(val, m_AnyZeroFloat()));
}

bool isAddPtrOffsetCombinable(Value first, Value second) {
  auto GetConstantIntValue = [](Value val) -> std::optional<llvm::APInt> {
    DenseElementsAttr constAttr;
    auto defOp = val.getDefiningOp();
    if (defOp) {
      if (auto splatOp = llvm::dyn_cast<SplatOp>(defOp))
        val = splatOp.getSrc();
      else if (matchPattern(defOp, m_Constant(&constAttr)) &&
               constAttr.isSplat()) {
        auto attr = constAttr.getSplatValue<Attribute>();
        // Check IntegerAttr
        if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
          return intAttr.getValue();
      }
    }

    // Check constant value.
    llvm::APInt intVal;
    if (matchPattern(val, m_ConstantInt(&intVal)))
      return intVal;

    return std::nullopt;
  };

  if (first.getType() == second.getType()) {
    // Whether bitwidth of element type is equal to pointer
    if (getElementTypeOrSelf(first.getType()).getIntOrFloatBitWidth() == 64)
      return true;

    // first + second does not overflow
    auto firstVal = GetConstantIntValue(first);
    auto secondVal = GetConstantIntValue(second);
    if (firstVal && secondVal) {
      bool overflow = false;
      auto resVal = firstVal->sadd_ov(*secondVal, overflow);
      return !overflow;
    }
  }
  return false;
}

// TODO(csigg): remove after next LLVM integrate.
using FastMathFlags = arith::FastMathFlags;

#include "TritonCombine.inc"

// select(cond, load(ptrs, splat(cond), ???), other)
//   => load(ptrs, splat(cond), other)
class CombineSelectMaskedLoadPattern : public RewritePattern {
public:
  CombineSelectMaskedLoadPattern(MLIRContext *context)
      : RewritePattern(arith::SelectOp::getOperationName(), 3, context,
                       {LoadOp::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto selectOp = llvm::dyn_cast<arith::SelectOp>(op);
    if (!selectOp)
      return failure();

    Value trueValue = selectOp.getTrueValue();
    Value falseValue = selectOp.getFalseValue();
    Value condSelect = selectOp.getCondition();

    auto loadOp = trueValue.getDefiningOp<LoadOp>();
    if (!loadOp)
      return failure();

    Value mask = loadOp.getMask();
    if (!mask)
      return failure();

    auto splatOp = mask.getDefiningOp<SplatOp>();
    if (!splatOp)
      return failure();

    auto splatCond = splatOp.getSrc();
    if (splatCond != condSelect)
      return failure();

    rewriter.replaceOpWithNewOp<LoadOp>(
        op, loadOp.getPtr(), loadOp.getMask(), /*other=*/falseValue,
        loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
        loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    return success();
  }
};

// sum(x[:, :, None] * y[None, :, :], 1)
// -> dot(x, y)
class CombineBroadcastMulReducePattern : public RewritePattern {
private:
  static bool isAddF32(const Operation *op) {
    if (auto addf = dyn_cast_or_null<arith::AddFOp>(op))
      return addf.getType().getIntOrFloatBitWidth() <= 32;
    return false;
  }

  static SmallVector<int> getEqualIndices(ArrayRef<int64_t> x,
                                          ArrayRef<int64_t> y) {
    SmallVector<int> res;
    for (int i = 0; i < x.size(); ++i)
      if (x[i] == y[i])
        res.push_back(i);
    return res;
  }

public:
  CombineBroadcastMulReducePattern(MLIRContext *context)
      : RewritePattern(ReduceOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const {
    auto reduceOp = llvm::dyn_cast<ReduceOp>(op);
    if (!reduceOp)
      return failure();
    // only support reduce with simple addition
    Region &combineOp = reduceOp.getCombineOp();
    bool isReduceAdd = combineOp.hasOneBlock() &&
                       combineOp.front().getOperations().size() == 2 &&
                       isAddF32(&*combineOp.front().getOperations().begin());
    if (!isReduceAdd)
      return failure();
    // operand of reduce has to be mul
    auto mulOp = reduceOp.getOperand(0).getDefiningOp<arith::MulFOp>();
    if (!mulOp)
      return failure();
    // mul operand has to be broadcast
    auto broadcastLhsOp = mulOp.getOperand(0).getDefiningOp<BroadcastOp>();
    if (!broadcastLhsOp)
      return failure();
    auto broadcastRhsOp = mulOp.getOperand(1).getDefiningOp<BroadcastOp>();
    if (!broadcastRhsOp)
      return failure();
    // broadcast operand is expand dims
    auto expandLhsOp = broadcastLhsOp.getSrc().getDefiningOp<ExpandDimsOp>();
    if (!expandLhsOp)
      return failure();
    auto expandRhsOp = broadcastRhsOp.getSrc().getDefiningOp<ExpandDimsOp>();
    if (!expandRhsOp)
      return failure();
    // get not-broadcast dimensions
    int expandLhsAxis = expandLhsOp.getAxis();
    int expandRhsAxis = expandRhsOp.getAxis();
    if (expandLhsAxis != 2 || expandRhsAxis != 0)
      return failure();
    auto broadcastLhsShape =
        cast<ShapedType>(broadcastLhsOp.getType()).getShape();
    auto broadcastRhsShape =
        cast<ShapedType>(broadcastLhsOp.getType()).getShape();
    if (broadcastLhsShape[2] < 16 || broadcastRhsShape[0] < 16)
      return failure();
    Type newAccType = RankedTensorType::get(
        {broadcastLhsShape[0], broadcastRhsShape[2]},
        cast<ShapedType>(broadcastLhsOp.getSrc().getType()).getElementType());
    rewriter.setInsertionPoint(op);
    auto newAcc = rewriter.create<SplatOp>(
        op->getLoc(), newAccType,
        rewriter.create<arith::ConstantOp>(op->getLoc(),
                                           rewriter.getF32FloatAttr(0)));
    rewriter.replaceOpWithNewOp<DotOp>(op, expandLhsOp.getSrc(),
                                       expandRhsOp.getSrc(), newAcc,
                                       InputPrecision::TF32, 0);
    return success();
  }
};

// When reducing a 1D tensor the order of elements of the tensor doesn't matter.
// Therefore we can relax the reshape to allow it to re-order elements.
class CombineReshapeReducePatterns : public mlir::OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::ReshapeOp reshapeOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (reshapeOp.getAllowReorder())
      return failure();
    if (reshapeOp.getType().getRank() != 1)
      return failure();
    for (Operation *user : reshapeOp->getUsers()) {
      if (!isa<triton::ReduceOp, triton::HistogramOp>(user))
        return failure();
    }
    rewriter.modifyOpInPlace(reshapeOp,
                             [&]() { reshapeOp.setAllowReorder(true); });
    return success();
  }
};

class RankedReduceDescriptorLoads : public mlir::OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::ReshapeOp reshapeOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto loadDef = reshapeOp.getSrc().getDefiningOp<triton::DescriptorLoadOp>();
    if (!loadDef || !loadDef->hasOneUse())
      return failure();
    int loadRank = loadDef.getType().getRank();
    int reshapeRank = reshapeOp.getType().getRank();
    if (!(reshapeRank < loadRank))
      return failure();
    ArrayRef<int64_t> loadShape = loadDef.getType().getShape();
    ArrayRef<int64_t> reshapeShape = reshapeOp.getType().getShape();
    for (int i = 0; i < loadRank - reshapeRank; ++i) {
      // Only rank reduce unit dims.
      if (loadShape[i] != 1)
        return failure();
    }
    if (loadShape.take_back(reshapeRank) != reshapeShape)
      return failure();
    rewriter.modifyOpInPlace(
        loadDef, [&]() { loadDef.getResult().setType(reshapeOp.getType()); });
    rewriter.replaceOp(reshapeOp, loadDef.getResult());
    return success();
  }
};

class CombineOpsPass : public TritonCombineOpsBase<CombineOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp m = getOperation();

    // Dot Add %{
    patterns.add<CombineDotAddIPattern>(context);
    patterns.add<CombineDotAddFPattern>(context);
    patterns.add<CombineDotAddIRevPattern>(context);
    patterns.add<CombineDotAddFRevPattern>(context);
    // %}
    patterns.add<CombineSelectMaskedLoadPattern>(context);
    patterns.add<CombineAddPtrPattern>(context);
    patterns.add<CombineBroadcastMulReducePattern>(context);
    patterns.add<CombineReshapeReducePatterns>(context);
    patterns.add<RankedReduceDescriptorLoads>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createCombineOpsPass() {
  return std::make_unique<CombineOpsPass>();
}

} // namespace mlir::triton
