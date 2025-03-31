#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
namespace gpu {

namespace {

// Get the highest version supported for the hardware and the dot.
static int getMMAVersionSafe(int computeCapability, DotOp op) {
  // List supported mma version in order of preference.
  SmallVector<int> versionsSupported;
  if (computeCapability < 75) {
    versionsSupported = {1};
  } else if (computeCapability < 90) {
    versionsSupported = {2};
  } else if (computeCapability < 100) {
    versionsSupported = {3, 2};
  } else if (computeCapability < 110) {
    versionsSupported = {5, 2};
  } else if (computeCapability < 130) {
    versionsSupported = {2};
  } else {
    assert(false && "computeCapability not supported");
  }
  for (int baseVersion : versionsSupported) {
    if (supportMMA(op, baseVersion))
      return baseVersion;
    if (baseVersion == 3)
      op.emitRemark() << "Warning: can't use MMA V3 for the dot op";
  }
  return 0;
}

SmallVector<unsigned> warpsPerTileV2(DotOp dotOp, const ArrayRef<int64_t> shape,
                                     int numWarps) {
  auto rank = shape.size();
  // Early exit for batched matmul
  if (rank == 3)
    return {(unsigned)numWarps, 1, 1};

  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion() &&
           !isa<TransOp>(op);
  };
  auto slices = multiRootGetSlice(dotOp, {filter}, {filter});
  bool hasChainedDot = false;
  for (Operation *op : slices) {
    if (isa<DotOp>(op) && (op != dotOp)) {
      auto chainedDot = cast<DotOp>(op);
      auto resTy = chainedDot.getResult().getType();
      if (resTy.getRank() != rank) {
        continue;
      }
      if (auto mmaEncoding =
              dyn_cast<NvidiaMmaEncodingAttr>(resTy.getEncoding())) {
        return to_vector(mmaEncoding.getWarpsPerCTA());
      }
      hasChainedDot = true;
    }
  }
  if (hasChainedDot) {
    if (shape[0] >= shape[1]) {
      return {(unsigned)numWarps, 1};
    } else {
      return {1, (unsigned)numWarps};
    }
  }

  assert(rank == 2);
  SmallVector<int64_t> shapePerWarp = {16, 8};
  SmallVector<int64_t> warps = {1, 1};
  // Compute repM and repN
  SmallVector<int64_t> reps = {ceil(shape[0], shapePerWarp[0]),
                               ceil(shape[1], shapePerWarp[1])};
  // The formula for the number of registers given the reps is
  // repM * 4 * repK + repN * 2 * repK + regsC
  // where regsC = repM * repN * 4, which does not depend on the warp shape
  //
  // As such, to minimize the register pressure, we need to balance
  // repM and repN. We then untie towards M, as the lhs tile has 4 elements,
  // and the rhs tile has just 2.
  while (product(warps) < numWarps) {
    if (reps[0] >= reps[1]) {
      warps[0] *= 2;
      // Too many warps for this mma (repM == repN == 1).
      // We allocate the remaining warps to the left (arbitrary choice)
      if (reps[0] != 1) {
        reps[0] /= 2;
      }
    } else {
      warps[1] *= 2;
      reps[1] /= 2;
    }
  }
  return {(unsigned)warps[0], (unsigned)warps[1]};
}

SmallVector<unsigned, 2>
warpsPerTileV3(DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps,
               const SmallVector<unsigned, 3> &instrShape) {
  SetVector<Operation *> slices;
  mlir::getForwardSlice(dotOp.getResult(), &slices);
  // Contains a chained dot. We prefer to assign warps to one axis
  // to facilitate use cases like flash attention, allowing reductions within
  // the same warp.
  if (llvm::find_if(slices, [](Operation *op) {
        return isa<mlir::triton::DotOpInterface>(op);
      }) != slices.end())
    return {(unsigned)numWarps, 1};

  // For MMAv3, the smallest indivisible unit of warp shape is (4, 1).
  SmallVector<unsigned, 2> ret = {4, 1};
  SmallVector<int64_t, 2> shapePerWarp = {16, instrShape[1]};
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] > shapePerWarp[0] * ret[0]) {
      ret[0] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

// Returns a shared memory allocation that can be used by a dotMMA op for the
// given value.
static Value
getSharedMemoryMMAOperand(Value v, mlir::PatternRewriter &rewriter, int opIdx,
                          bool allowTranspose, bool isMMAv5Fp4Padded = false,
                          Operation *op = nullptr /*only for diagnostic*/) {
  OpBuilder::InsertionGuard g(rewriter);
  Value arg = v;
  if (auto cvtOp = v.getDefiningOp<ConvertLayoutOp>())
    arg = cvtOp.getSrc();
  auto argType = cast<RankedTensorType>(arg.getType());
  assert(argType.getEncoding() && "unexpected tensor type");
  auto order = getOrderForMemory(argType);

  // If the MMA op doesn't support transpose pick the layout expected by the MMA
  // op.
  llvm::SmallVector<unsigned> newOrder = order;
  if (!allowTranspose) {
    if (opIdx == 1) {
      newOrder = {0, 1};
    } else {
      newOrder = {1, 0};
    }
  }

  if (newOrder != order && op) {
    op->emitWarning("Warning: Forcing a different order [")
        << newOrder[0] << ", " << newOrder[1]
        << "] on SMEM than the register order for the opreand " << opIdx
        << ". Registers will be transposed before SMEM store and the pipelined "
           "load for this operand will be disabled, so poor performance is "
           "expected.";
  }

  Attribute SharedMemorySpace =
      SharedMemorySpaceAttr::get(argType.getContext());
  auto CTALayout = getCTALayout(argType.getEncoding());
  auto newLayout = NVMMASharedEncodingAttr::get(
      argType.getContext(), argType.getShape(), newOrder, CTALayout,
      argType.getElementType(), isMMAv5Fp4Padded);
  auto newType = MemDescType::get(argType.getShape(), argType.getElementType(),
                                  newLayout, SharedMemorySpace);
  rewriter.setInsertionPointAfterValue(arg);
  return rewriter.create<LocalAllocOp>(arg.getLoc(), newType, arg);
}

static LocalAllocOp
getSharedMemoryScale(Value arg, mlir::PatternRewriter &rewriter, Location loc) {
  OpBuilder::InsertionGuard g(rewriter);
  auto argType = cast<RankedTensorType>(arg.getType());
  assert(argType.getEncoding() && "unexpected tensor type");
  auto newOrder = getOrderForMemory(argType);

  Attribute SharedMemorySpace =
      SharedMemorySpaceAttr::get(argType.getContext());
  auto CTALayout = getCTALayout(argType.getEncoding());
  // No swizzling for scale for now
  auto newLayout = SwizzledSharedEncodingAttr::get(argType.getContext(), 1, 1,
                                                   1, newOrder, CTALayout);
  auto newType = MemDescType::get(argType.getShape(), argType.getElementType(),
                                  newLayout, SharedMemorySpace);
  rewriter.setInsertionPointAfterValue(arg);
  return rewriter.create<LocalAllocOp>(loc, newType, arg);
}

SmallVector<unsigned, 3>
getWarpsPerTile(DotOp dotOp, const ArrayRef<int64_t> shape, int version,
                int numWarps, const SmallVector<unsigned, 3> &instrShape) {
  switch (version) {
  case 2:
    return warpsPerTileV2(dotOp, shape, numWarps);
  case 3:
    return warpsPerTileV3(dotOp, shape, numWarps, instrShape);
  default:
    assert(false && "not supported version");
    return {0, 0};
  }
}

static bool bwdFilter(Operation *op) {
  // Dot operand layout assignment to Predicates are not currently supported
  // during lowering from TritonGPU to LLVM in Triton for MMA cases. This
  // condition limits visibility of the original bit-width so that predicate
  // are not considered, hence, kwidth can never be = 32.
  if (isa<arith::UIToFPOp>(op)) {
    Type srcType = getElementTypeOrSelf(op->getOperand(0));
    if (srcType.isInteger(1))
      return false;
  }

  // b/405045790: We don't want to propagate through the BroadcastOp because we
  // probably don't care about the load before a broadcast as it would likely be
  // small. This is just a heuristic to avoid a regression.
  return (op->hasTrait<OpTrait::Elementwise>() && isMemoryEffectFree(op)) ||
         isView(op) ||
         isa<Fp4ToFpOp, LoadOp, DescriptorLoadOp, /*BroadcastOp,*/
             ConvertLayoutOp>(op);
}

// Finds the bitwidth with which the value x is loaded
static int computeOrigBitWidth(Value x) {
  SetVector<Operation *> slice;
  mlir::BackwardSliceOptions opt;
  opt.omitBlockArguments = true;
  opt.filter = bwdFilter;
  getBackwardSlice(x, &slice, opt);

  // TODO: This heuristic may be a bit too coarse and may need improving
  // If the chain contains a fp4 to fp16/bf16 conversion, then the original
  // bitwidth is 4.
  if (llvm::any_of(slice, [](Operation *op) { return isa<Fp4ToFpOp>(op); }))
    return 4;

  int origBitWidth = getElementTypeOrSelf(x).getIntOrFloatBitWidth();
  for (auto op : slice) {
    if (isa<LoadOp, DescriptorLoadOp>(op)) {
      if (auto tensorTy =
              dyn_cast<RankedTensorType>(op->getResultTypes().front())) {
        origBitWidth =
            std::min<int>(origBitWidth, tensorTy.getElementTypeBitWidth());
      }
    }
  }

  // If JoinOp occurred at least once, in backward layout propagation,
  // the kWidth will be split in half as we pass through the JoinOp.
  // Hence we divide origBitWidth by 2 here to compensate for that and
  // improve our load width.
  // This won't be optimal if there is a tree of multiple JoinOps, which
  // would require counting the max number of JoinOp's along any path.
  //
  // In the future we might want to do something like trying a large kWidth,
  // run layout backpropagation and see what's the contiguity that you
  // get at the loads that feed into it.
  if (llvm::any_of(slice, [](Operation *op) { return isa<JoinOp>(op); }))
    origBitWidth /= 2;

  return origBitWidth;
}

class BlockedToMMA : public mlir::OpRewritePattern<DotOp> {
  int computeCapability;
  mutable llvm::DenseMap<Operation *, unsigned> dotOpInstNs;

public:
  BlockedToMMA(mlir::MLIRContext *context, int computeCapability, int benefit)
      : OpRewritePattern<DotOp>(context, benefit),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability < 70)
      return failure();
    if (computeCapability < 80) {
      dotOp.emitRemark()
          << "Dot op using MMA for compute capability " << computeCapability
          << " has been deprecated. It falls back to the FMA path.";
      return failure();
    }
    // TODO: Check data-types and SM compatibility
    if (!dotOp.getType().getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(dotOp.getType().getEncoding()))
      return failure();

    int numWarps = lookupNumWarps(dotOp);
    int versionMajor = getMMAVersionSafe(computeCapability, dotOp);
    if (!(versionMajor >= 1 && versionMajor <= 3))
      return failure();

    // If both of the operands are not loads, we fallback to MMAv2
    // otherwise the reg-smem roundtrip will tank the MMAv3 performance
    auto comesFromLoadOrBlockArg = [](Value v) -> bool {
      // Peel out the original cvt dot_op<..., #blocked>
      // and any other potential cvt/trans ops
      while (true) {
        if (auto cvtOp = v.getDefiningOp<ConvertLayoutOp>()) {
          v = cvtOp.getSrc();
          continue;
        }
        if (auto transOp = v.getDefiningOp<TransOp>()) {
          v = transOp.getSrc();
          continue;
        }
        break;
      }
      // We also accept block arguments as they appear in many MLIR tests
      // If this is problematic we can totally drop them
      return isa<BlockArgument>(v) ||
             (v.getDefiningOp() &&
              isa<LoadOp, DescriptorLoadOp>(v.getDefiningOp()));
    };

    bool aFromLoad = comesFromLoadOrBlockArg(dotOp.getA());
    bool bFromLoad = comesFromLoadOrBlockArg(dotOp.getB());
    auto origDotOp = dotOp;

    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());
    auto oldRetType = cast<RankedTensorType>(dotOp.getType());

    // get MMA encoding for the given number of warps
    auto CTALayout = getCTALayout(oldRetType.getEncoding());
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    auto instrShape = mmaVersionToInstrShape(
        versionMajor, retShapePerCTA, oldAType.getElementType(), numWarps);

    assert(versionMajor == 2 || versionMajor == 3);
    int versionMinor = computeCapability == 75 ? 1 : 0;
    auto warpsPerTile = getWarpsPerTile(dotOp, retShapePerCTA, versionMajor,
                                        numWarps, instrShape);
    auto mmaEnc = NvidiaMmaEncodingAttr::get(
        oldRetType.getContext(), versionMajor, versionMinor, warpsPerTile,
        CTALayout, instrShape);
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), mmaEnc);
    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc =
        rewriter.create<ConvertLayoutOp>(oldAcc.getLoc(), newRetType, oldAcc);

    auto getDotOperand = [&](Value v, int opIdx, int bitwidth) {
      auto minType =
          bitwidth > 0 ? rewriter.getIntegerType(bitwidth) : v.getType();
      auto vType = cast<RankedTensorType>(v.getType());
      auto newVEncoding = DotOperandEncodingAttr::get(
          v.getContext(), opIdx, newRetType.getEncoding(), minType);
      auto newVType = RankedTensorType::get(
          vType.getShape(), vType.getElementType(), newVEncoding);
      return rewriter.create<ConvertLayoutOp>(v.getLoc(), newVType, v);
    };

    Operation *newDot = nullptr;
    if (versionMajor == 3) {
      auto eltType = dotOp.getA().getType().getElementType();
      // In MMAV3 transpose is only supported for f16 and bf16.
      bool allowTranspose = eltType.isF16() || eltType.isBF16();
      if (!aFromLoad) {
        int bitwidth = getElementTypeOrSelf(a).getIntOrFloatBitWidth();
        a = getDotOperand(a, 0, bitwidth);
      } else {
        a = getSharedMemoryMMAOperand(a, rewriter, 0, allowTranspose);
      }
      b = getSharedMemoryMMAOperand(b, rewriter, 1, allowTranspose);
      newDot = rewriter.create<triton::nvidia_gpu::WarpGroupDotOp>(
          dotOp.getLoc(), newRetType, a, b, newAcc, nullptr,
          dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc(), false);
    } else {
      // convert operands
      int minBitwidth =
          std::min(computeOrigBitWidth(a), computeOrigBitWidth(b));

      a = getDotOperand(a, 0, minBitwidth);
      b = getDotOperand(b, 1, minBitwidth);
      newDot = rewriter.create<DotOp>(dotOp.getLoc(), newRetType, a, b, newAcc,
                                      dotOp.getInputPrecision(),
                                      dotOp.getMaxNumImpreciseAcc());
    }
    // convert dot instruction
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(origDotOp, origDotOp.getType(),
                                                 newDot->getResult(0));
    return success();
  }
};

// Pick the layout to match MXFP scales layout in register so that it can be
// copied directly using tmem st.
static Attribute getTmemScales(RankedTensorType type, unsigned numWarps) {
  return triton::gpu::LinearEncodingAttr::get(
      type.getContext(), getScaleTMEMStoreLinearLayout(type, numWarps));
}

static bool canUseTwoCTAs(triton::DotOp dotOp) {
  RankedTensorType retType = dotOp.getType();
  auto retShapePerCTA = getShapePerCTA(retType);
  // TODO: we could support 2 CTAs matmul with numCTAs > 2.
  SmallVector<unsigned> splitNum = getCTASplitNum(retType.getEncoding());
  if (splitNum.size() != 2 || splitNum[0] != 2 || splitNum[1] != 1)
    return false;
  int m = retShapePerCTA[0];
  int n = retShapePerCTA[1];
  // minimum size supported by 2CTAs mmav5.
  if (m < 64 || n < 32)
    return false;
  Value b = dotOp.getB();
  // Skip convert layouts.
  while (auto cvtOp = b.getDefiningOp<ConvertLayoutOp>())
    b = cvtOp.getSrc();
  if (!b.getDefiningOp<triton::LoadOp>())
    return false;
  return true;
}

static DistributedEncodingTrait
replaceCTALayout(DistributedEncodingTrait layout,
                 const triton::gpu::CTALayoutAttr &newCTALayout) {
  if (auto blockedLayout = mlir::dyn_cast<BlockedEncodingAttr>(layout)) {
    return BlockedEncodingAttr::get(
        layout.getContext(), blockedLayout.getSizePerThread(),
        blockedLayout.getThreadsPerWarp(), blockedLayout.getWarpsPerCTA(),
        blockedLayout.getOrder(), newCTALayout);
  } else if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(layout)) {
    return SliceEncodingAttr::get(
        layout.getContext(), sliceLayout.getDim(),
        replaceCTALayout(sliceLayout.getParent(), newCTALayout));
  } else {
    llvm::report_fatal_error("not implemented");
    return layout;
  }
}

static Value splitBOperand(Value b, mlir::PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  MLIRContext *ctx = b.getContext();
  while (auto cvtOp = b.getDefiningOp<ConvertLayoutOp>())
    b = cvtOp.getSrc();
  auto loadOp = b.getDefiningOp<triton::LoadOp>();
  assert(loadOp && "expected LoadOp");
  RankedTensorType bType = cast<RankedTensorType>(b.getType());
  auto currentLayout = cast<DistributedEncodingTrait>(bType.getEncoding());
  auto newCTALayout =
      CTALayoutAttr::get(ctx, {1, 2}, {1, 2}, getCTAOrder(currentLayout));
  Attribute newLayout = replaceCTALayout(currentLayout, newCTALayout);
  rewriter.setInsertionPoint(loadOp);
  for (OpOperand &operand : loadOp->getOpOperands()) {
    auto tensorType = dyn_cast<RankedTensorType>(operand.get().getType());
    if (!tensorType)
      continue;
    Value newOperand = rewriter.create<ConvertLayoutOp>(
        operand.get().getLoc(),
        RankedTensorType::get(tensorType.getShape(),
                              tensorType.getElementType(), newLayout),
        operand.get());
    loadOp.setOperand(operand.getOperandNumber(), newOperand);
  }
  loadOp.getResult().setType(RankedTensorType::get(
      bType.getShape(), bType.getElementType(), newLayout));
  Value newB = loadOp.getResult();
  rewriter.setInsertionPointAfter(loadOp);
  auto cvt =
      rewriter.create<ConvertLayoutOp>(b.getLoc(), bType, loadOp.getResult());
  rewriter.replaceAllUsesExcept(loadOp.getResult(), cvt.getResult(), cvt);
  return newB;
}

class BlockedToMMAv5 : public mlir::OpRewritePattern<DotOp> {
  int computeCapability;

public:
  BlockedToMMAv5(mlir::MLIRContext *context, int computeCapability, int benefit)
      : OpRewritePattern<DotOp>(context, benefit),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    int numWarps = lookupNumWarps(dotOp);
    auto CTALayout = getCTALayout(oldRetType.getEncoding());

    int versionMajor = getMMAVersionSafe(computeCapability, dotOp);
    if (versionMajor != 5)
      return failure();
    Location loc = dotOp.getLoc();
    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    if (std::min(computeOrigBitWidth(a), computeOrigBitWidth(b)) >= 32 &&
        dotOp.getInputPrecision() != InputPrecision::TF32)
      return failure();
    auto oldAType = dotOp.getA().getType();
    auto oldBType = dotOp.getB().getType();
    bool useTwoCTAs = canUseTwoCTAs(dotOp);
    if (useTwoCTAs) {
      b = splitBOperand(b, rewriter);
    }
    // TF32 transpose is only supported with 128 swizzle mode with 32B
    // atomicity. As we currently don't support this layout we disallow
    // transpose for TF32 inputs.
    bool allowTranspose = !dotOp.getA().getType().getElementType().isF32();
    a = getSharedMemoryMMAOperand(a, rewriter, 0, allowTranspose);
    b = getSharedMemoryMMAOperand(b, rewriter, 1, allowTranspose);
    MLIRContext *context = dotOp->getContext();
    auto instrShape = mmaVersionToInstrShape(
        versionMajor, retShapePerCTA, oldAType.getElementType(), numWarps);
    ArrayRef<unsigned> CTASplitNum = CTALayout.getCTASplitNum();
    Attribute accEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
        context, instrShape[0], instrShape[1], /*unpacked=*/true,
        CTASplitNum[0], CTASplitNum[1]);
    Attribute tensorMemorySpace =
        triton::nvidia_gpu::TensorMemorySpaceAttr::get(context);
    Type accMemDescType = triton::gpu::MemDescType::get(
        oldRetType.getShape(), oldRetType.getElementType(), accEncoding,
        tensorMemorySpace,
        /*mutableMemory=*/true);
    Attribute newDistributedEncoding = nvidia_gpu::getTmemCompatibleLayout(
        instrShape[0], instrShape[1], retShapePerCTA, numWarps, CTALayout);
    auto newAccType = RankedTensorType::get(oldRetType.getShape(),
                                            oldRetType.getElementType(),
                                            newDistributedEncoding);
    Value cvtAcc =
        rewriter.create<ConvertLayoutOp>(loc, newAccType, dotOp.getOperand(2));
    auto acc = rewriter.create<triton::nvidia_gpu::TMEMAllocOp>(
        loc, accMemDescType, cvtAcc);
    auto vTrue = rewriter.create<arith::ConstantIntOp>(dotOp.getLoc(), 1, 1);
    auto mma = rewriter.create<triton::nvidia_gpu::TCGen5MMAOp>(
        loc, a, b, acc, vTrue, vTrue, Value(), UnitAttr());
    mma.setTwoCtas(useTwoCTAs);

    auto ld =
        rewriter.create<triton::nvidia_gpu::TMEMLoadOp>(loc, newAccType, acc);
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, oldRetType, ld);
    return success();
  }
};

Value addSmemStageToScaleLoad(Value scale, mlir::PatternRewriter &rewriter) {
  /*
    Rewrite load(scale) -> local_load(local_alloc(load(scale))).
    This function does not add anything to the final IR when num_stages > 1,
    but it makes it easy to apply TMEM copy rewriting later.

    Since scales are stored in TMEM for MMAv5 scaled dot, loading of scales do
    not needs to be put into SMEM. But in practice, the software pipeliner puts
    loading of scales into multi-buffered SMEM. At that point, the SMEM
    allocation created here is eliminated.
   */
  OpBuilder::InsertionGuard g(rewriter);
  auto op = scale.getDefiningOp();
  Operation *loadConsumer = nullptr;

  if (!op)
    return scale;

  while (!isa<LoadOp>(op)) {
    if (auto reshape = dyn_cast<ReshapeOp>(op)) {
      op = reshape.getSrc().getDefiningOp();
      loadConsumer = reshape;
    } else if (auto trans = dyn_cast<TransOp>(op)) {
      op = trans.getSrc().getDefiningOp();
      loadConsumer = trans;
    } else if (auto cvt = dyn_cast<ConvertLayoutOp>(op)) {
      op = cvt.getSrc().getDefiningOp();
      loadConsumer = cvt;
    } else {
      // Unrecognized pattern, bail out. In practice, this implies that MMA
      // pipelining will not apply to the scaled dot op, since scales will not
      // be in passed through SMEM to tc_gen5_mma_scaled.
      return scale;
    }
  }

  auto scaleAfterLoad = op->getResult(0);
  auto scaleSmemAlloc =
      getSharedMemoryScale(scaleAfterLoad, rewriter, op->getLoc());

  rewriter.setInsertionPointAfterValue(scaleSmemAlloc);
  auto localLoad = rewriter.create<LocalLoadOp>(
      op->getLoc(), scaleAfterLoad.getType(), scaleSmemAlloc);

  rewriter.replaceAllUsesExcept(scaleAfterLoad, localLoad.getResult(),
                                scaleSmemAlloc);

  if (loadConsumer) {
    return scale;
  } else {
    return localLoad;
  }
}

class ScaledBlockedToMMAv5
    : public mlir::OpRewritePattern<triton::DotScaledOp> {
  int computeCapability;

public:
  ScaledBlockedToMMAv5(mlir::MLIRContext *context, int computeCapability,
                       int benefit)
      : mlir::OpRewritePattern<triton::DotScaledOp>(context, benefit),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotScaledOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    if (dotOp.getAScale() == nullptr || dotOp.getBScale() == nullptr) {
      return failure();
    }

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    int numWarps = lookupNumWarps(dotOp);
    auto CTALayout = getCTALayout(oldRetType.getEncoding());
    if (computeCapability < 100)
      return failure();
    if (retShapePerCTA[0] < 128 || retShapePerCTA[1] < 8)
      return failure();
    Location loc = dotOp.getLoc();
    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = a.getType();
    auto oldBType = b.getType();

    bool IsAMixedPrecFp4 = false;
    bool IsBMixedPrecFp4 = false;

    if (dotOp.getAElemType() != dotOp.getBElemType()) {
      if (dotOp.getAElemType() == ScaleDotElemType::E2M1)
        IsAMixedPrecFp4 = true;
      else if (dotOp.getBElemType() == ScaleDotElemType::E2M1)
        IsBMixedPrecFp4 = true;
    }

    // For mixed-precision fp4 operands, set allowTranspose = false, to force
    // the packed axis, K, to be contiguous in SMEM
    a = getSharedMemoryMMAOperand(a, rewriter, 0,
                                  /*allowTranspose=*/!IsAMixedPrecFp4,
                                  IsAMixedPrecFp4, dotOp);
    b = getSharedMemoryMMAOperand(b, rewriter, 1,
                                  /*allowTranspose=*/!IsBMixedPrecFp4,
                                  IsBMixedPrecFp4, dotOp);

    MLIRContext *context = dotOp->getContext();
    unsigned m = 128;
    unsigned n = retShapePerCTA[1] >= 256 ? 256 : retShapePerCTA[1];
    unsigned k = 32;
    // If both operands are E2M1, target the FP4 tensor core implicitly.
    // This may result in a downstream compile-time error if the scaled TC
    // descriptor requires options that are unavailable to the .kind=mxf4 mma.
    // This is likely preferable over a silent runtime performance degradation
    // from running f4xf4 via .kind=mxf8f6f4
    if (dotOp.getAElemType() == ScaleDotElemType::E2M1 &&
        dotOp.getBElemType() == ScaleDotElemType::E2M1) {
      k = 64;
    }
    SmallVector<unsigned> instrShape = {m, n, k};
    ArrayRef<unsigned> CTASplitNum = CTALayout.getCTASplitNum();
    Attribute accEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
        context, instrShape[0], instrShape[1], /*unpacked=*/true,
        CTASplitNum[0], CTASplitNum[1]);
    Attribute tensorMemorySpace =
        triton::nvidia_gpu::TensorMemorySpaceAttr::get(context);
    Type accMemDescType = triton::gpu::MemDescType::get(
        oldRetType.getShape(), oldRetType.getElementType(), accEncoding,
        tensorMemorySpace,
        /*mutableMemory=*/true);
    Attribute newDistributedEncoding = nvidia_gpu::getTmemCompatibleLayout(
        instrShape[0], instrShape[1], retShapePerCTA, numWarps, CTALayout);
    auto newAccType = RankedTensorType::get(oldRetType.getShape(),
                                            oldRetType.getElementType(),
                                            newDistributedEncoding);
    Value cvtAcc =
        rewriter.create<ConvertLayoutOp>(loc, newAccType, dotOp.getOperand(2));
    auto acc = rewriter.create<triton::nvidia_gpu::TMEMAllocOp>(
        loc, accMemDescType, cvtAcc);

    RankedTensorType oldScaleAType = dotOp.getAScale().getType();
    RankedTensorType oldScaleBType = dotOp.getBScale().getType();

    Attribute scaleEncoding =
        triton::nvidia_gpu::TensorMemoryScalesEncodingAttr::get(
            context, CTASplitNum[0], CTASplitNum[1]);
    Type scaleAType = triton::gpu::MemDescType::get(
        oldScaleAType.getShape(), oldScaleAType.getElementType(), scaleEncoding,
        tensorMemorySpace,
        /*mutableMemory=*/false);
    Type scaleBType = triton::gpu::MemDescType::get(
        oldScaleBType.getShape(), oldScaleBType.getElementType(), scaleEncoding,
        tensorMemorySpace,
        /*mutableMemory=*/false);
    Attribute scaleALayout = getTmemScales(oldScaleAType, numWarps);
    Attribute scaleBLayout = getTmemScales(oldScaleBType, numWarps);
    RankedTensorType newScaleAType = RankedTensorType::get(
        oldScaleAType.getShape(), oldScaleAType.getElementType(), scaleALayout);
    RankedTensorType newScaleBType = RankedTensorType::get(
        oldScaleBType.getShape(), oldScaleBType.getElementType(), scaleBLayout);

    auto lhsScale = addSmemStageToScaleLoad(dotOp.getAScale(), rewriter);
    auto rhsScale = addSmemStageToScaleLoad(dotOp.getBScale(), rewriter);

    Value newScaleA =
        rewriter.create<ConvertLayoutOp>(loc, newScaleAType, lhsScale);
    Value newScaleB =
        rewriter.create<ConvertLayoutOp>(loc, newScaleBType, rhsScale);
    Value scaleA = rewriter.create<triton::nvidia_gpu::TMEMAllocOp>(
        loc, scaleAType, newScaleA);
    Value scaleB = rewriter.create<triton::nvidia_gpu::TMEMAllocOp>(
        loc, scaleBType, newScaleB);
    auto vTrue = rewriter.create<arith::ConstantIntOp>(dotOp.getLoc(), 1, 1);
    rewriter.create<triton::nvidia_gpu::TCGen5MMAScaledOp>(
        loc, a, b, acc, scaleA, scaleB, dotOp.getAElemType(),
        dotOp.getBElemType(), vTrue, vTrue, Value());

    auto ld =
        rewriter.create<triton::nvidia_gpu::TMEMLoadOp>(loc, newAccType, acc);
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, oldRetType, ld);
    return success();
  }
};
} // namespace

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  Type tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                .cloneWith(std::nullopt, promotedType);
  return builder.create<FpToFpOp>(loc, tensorPromotedType, operand);
}

// promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(ModuleOp mod, int computeCapability) {
  mod.walk([=](DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    Type promoteType;
    NvidiaMmaEncodingAttr mmaLayout =
        dyn_cast<NvidiaMmaEncodingAttr>(D.getType().getEncoding());
    if (mmaLayout) {
      bool isNativeFP8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType>(AElType);
      // promote operands for sm < 89 since fp8 mma is not natively supported
      // promote operands for sm >= 90 when mma is not v3
      if (!isNativeFP8 ||
          (isNativeFP8 && (computeCapability == 89 || mmaLayout.isHopper())))
        return;
      promoteType = builder.getF16Type();
    } else {
      // FMA case.
      Type AElType = dotOp.getA().getType().getElementType();
      Type DElType = D.getType().getElementType();
      if (AElType == DElType)
        return;
      promoteType = DElType;
    }
    Location loc = dotOp.getLoc();
    Value promotedA = promoteOperand(builder, loc, dotOp.getA(), promoteType);
    Value promotedB = promoteOperand(builder, loc, dotOp.getB(), promoteType);
    dotOp.setOperand(0, promotedA);
    dotOp.setOperand(1, promotedB);
  });
}

// Transpose scaled_dot ops that have a scale on lhs.
static void transposeDotOp(DotScaledOp dotOp) {
  OpBuilder builder(dotOp);
  Value lhs = dotOp.getA();
  std::array<int, 2> transOrder = {1, 0};
  Value lhsTransposed = builder.create<TransOp>(lhs.getLoc(), lhs, transOrder);
  Value rhs = dotOp.getB();
  Value rhsTransposed = builder.create<TransOp>(rhs.getLoc(), rhs, transOrder);
  Value c = dotOp.getC();
  Value cTransposed = builder.create<TransOp>(c.getLoc(), c, transOrder);
  Value result = builder.create<DotScaledOp>(
      dotOp.getLoc(), cTransposed.getType(), rhsTransposed, lhsTransposed,
      cTransposed, dotOp.getBScale(), dotOp.getAScale(), dotOp.getBElemType(),
      dotOp.getAElemType(), dotOp.getFastMath());
  Operation *transposedResult =
      builder.create<TransOp>(result.getLoc(), result, transOrder);
  dotOp.replaceAllUsesWith(transposedResult);
  dotOp.erase();
}

static void transposeDots(ModuleOp m) {
  // TODO: extend to regular dot when it is profitable. For instance when we may
  // want to use rhs from register for mmav3.
  SmallVector<DotScaledOp> toTranspose;
  m.walk([&](DotScaledOp dotOp) -> void {
    if (dotOp.getAScale() == nullptr && dotOp.getBScale() != nullptr)
      toTranspose.push_back(dotOp);
  });
  for (DotScaledOp dotOp : toTranspose) {
    transposeDotOp(dotOp);
  }
}

#define GEN_PASS_DEF_TRITONGPUACCELERATEMATMUL
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUAccelerateMatmulPass
    : public impl::TritonGPUAccelerateMatmulBase<
          TritonGPUAccelerateMatmulPass> {
public:
  using impl::TritonGPUAccelerateMatmulBase<
      TritonGPUAccelerateMatmulPass>::TritonGPUAccelerateMatmulBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    auto computeCapability = getNVIDIAComputeCapability(m);
    // We could do this generically if we manage to improve the heuristics
    // reverted in these two PRs https://github.com/triton-lang/triton/pull/5834
    // https://github.com/triton-lang/triton/pull/5837
    transposeDots(m);

    mlir::RewritePatternSet patterns(context);
    constexpr int benefitDefault = 1;
    constexpr int benefitMMAv5 = 10;
    patterns.add<BlockedToMMA>(context, computeCapability, benefitDefault);
    populateDecomposeScaledBlockedPatterns(patterns, benefitDefault);
    patterns.add<BlockedToMMAv5, ScaledBlockedToMMAv5>(
        context, computeCapability, benefitMMAv5);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
    // Now that we have picked the mma type, decompose dot that are not natively
    // supported.
    decomposeMixedModeDotOp(m, computeCapability);
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
