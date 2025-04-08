#include "triton/Analysis/Utility.h"

#include <deque>

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/Sys/GetEnv.hpp"

namespace mlir {

using namespace triton;
using namespace triton::gpu;

SmallVector<unsigned> ReduceOpHelper::getOrderWithAxisAtBeginning() {
  auto order = toLinearEncoding(srcEncoding, srcShape).getOrder();
  auto it = std::find(order.begin(), order.end(), axis);
  // delete the axis from order
  order.erase(it);
  // insert axis at the beginning of order
  order.insert(order.begin(), axis);
  return order;
}

// Thread offset is the thread index offset of two adjacent threads on the
// reduction axis within the warp.
unsigned ReduceOpHelper::getThreadOffsetOnReductionAxis() {
  auto *ctx = srcEncoding.getContext();
  auto linearLayout = toLinearLayout(srcShape, srcEncoding);
  auto kLane = mlir::StringAttr::get(ctx, "lane");
  const auto &bases = linearLayout.getBases();
  const auto &lanes = bases.find(kLane)->second;
  auto offset = 1;
  for (const auto &lane : lanes) {
    if (lane[axis] != 0)
      break;
    offset *= 2;
  }
  return offset;
}

// Cases where distributed shared memory is not required in ConvertLayout:
// (1) numCTAs == 1
// (2) numCTAs > 1 but srcCTALayout == dstCTALayout
// TODO: Case with SliceLayout as srcLayout and numCTAs > 1 is to be implemented
// in the future
bool shouldUseDistSmem(Attribute srcLayout, Attribute dstLayout) {
  unsigned numCTAs = getNumCTAs(srcLayout);
  assert(numCTAs == getNumCTAs(dstLayout) &&
         "Invalid layout conversion: the numbers of CTAs of src and dst "
         "layouts are different");

  // Case (1): Never use dsmem when numCTAs == 1
  if (numCTAs == 1)
    return false;

  // Case where CTAsPerCGA of srcLayout in the sliced dim is not 1 is not
  // implemented yet
  if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(srcLayout)) {
    auto dim = sliceLayout.getDim();
    auto CTAsPerCGA = getCTAsPerCGA(sliceLayout.getParent());
    if (CTAsPerCGA[dim] != 1)
      llvm::report_fatal_error("Layout conversion to be implemented");
  }

  // Case where CTAsPerCGA of dstLayout in the sliced dim is not 1 is supported
  if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(dstLayout)) {
    auto dim = sliceLayout.getDim();
    auto CTAsPerCGA = getCTAsPerCGA(sliceLayout.getParent());
    if (CTAsPerCGA[dim] != 1)
      return true;
  }

  // The above two branches make sure that it is legal to call getCTALayout of
  // srcLayout and dstLayout

  // Case (2): Do not use dsmem when srcCTALayout == dstCTALayout
  auto srcCTALayout = getCTALayout(srcLayout);
  auto dstCTALayout = getCTALayout(dstLayout);
  if (srcCTALayout == dstCTALayout)
    return false;

  // Dsmem access is required when srcCTALayout != dstCTALayout
  return true;
}

unsigned ReduceOpHelper::getInterWarpSizeWithUniqueData() {
  return getWarpsPerCTA(srcEncoding, srcShape)[axis];
}

unsigned ReduceOpHelper::getIntraWarpSizeWithUniqueData() {
  return getThreadsPerWarp(srcEncoding, srcShape)[axis];
}

bool ReduceOpHelper::isWarpSynchronous() {
  return getWarpsPerCTA(srcEncoding, srcShape)[axis] == 1;
}

SmallVector<unsigned> ReduceOpHelper::getScratchRepShape() {
  SmallVector<unsigned> smemShape;
  // This case doesn't need inter-warp communication
  if (isWarpSynchronous())
    return {0, 0};

  smemShape = convertType<unsigned>(srcShape);
  smemShape[axis] = getInterWarpSizeWithUniqueData();

  return smemShape;
}

unsigned ReduceOpHelper::getScratchSizeInBytes() {
  auto smemShape = getScratchRepShape();
  auto elems = product<unsigned>(smemShape);

  unsigned bytesPerElem = 0;
  for (const auto &ty : srcElementTypes) {
    bytesPerElem += ceil<unsigned>(ty.getIntOrFloatBitWidth(), 8);
  }
  return bytesPerElem * elems;
}

bool ReduceOpHelper::isReduceWithinCTA() {
  // TODO: Support reduce across CTAS
  // Layout optimization passes such as PlanCTAPass and
  // RemoveLayoutConversionPass should avoid cross-CTA reduction
  return getCTASplitNum(srcEncoding)[axis] == 1;
}

unsigned ScanLoweringHelper::getAxisNumElementsPerThread() {
  return getEncoding().getContigPerThread()[getAxis()];
}

unsigned ScanLoweringHelper::getNonAxisNumElementsPerThread() {
  auto contigPerThread = getEncoding().getContigPerThread();
  contigPerThread[getAxis()] = 1;
  return product<unsigned>(contigPerThread);
}

Region &ScanLoweringHelper::getCombineOp() { return scanOp.getCombineOp(); }

unsigned ScanLoweringHelper::getAxisNumThreadsPerWarpWithUniqueData() {
  return getEncoding().getThreadsPerWarp()[getAxis()];
}

unsigned ScanLoweringHelper::getNonAxisNumThreadsPerWarp() {
  auto nThreads = product(getEncoding().getThreadsPerWarp());
  return nThreads / getAxisNumThreadsPerWarpWithUniqueData();
}

// Return the flat numbers of threads computing independent scan results.
unsigned ScanLoweringHelper::getNonAxisNumThreadsPerCTA() {
  auto nWarps = product(getEncoding().getWarpsPerCTA());
  return (nWarps / getAxisNumWarpsWithUniqueData()) *
         getNonAxisNumThreadsPerWarp();
}

unsigned ScanLoweringHelper::getAxisNumWarpsWithUniqueData() {
  return getEncoding().getWarpsPerCTA()[getAxis()];
}

unsigned ScanLoweringHelper::getAxisNumBlocks() {
  auto contigPerThread = getEncoding().getContigPerThread();
  auto threadsPerWarp = getEncoding().getThreadsPerWarp();
  auto warpsPerCTA = getEncoding().getWarpsPerCTA();
  unsigned axis = getAxis();
  return ceil<unsigned>(
      getShape()[axis],
      (contigPerThread[axis] * threadsPerWarp[axis] * warpsPerCTA[axis]));
}

unsigned ScanLoweringHelper::getNonAxisNumBlocks() {
  auto contigPerThread = getEncoding().getContigPerThread();
  auto threadsPerWarp = getEncoding().getThreadsPerWarp();
  auto warpsPerCTA = getEncoding().getWarpsPerCTA();
  auto rank = contigPerThread.size();
  unsigned axis = getAxis();
  unsigned numBlocks = 1;
  for (unsigned i = 0; i < rank; i++) {
    if (i == axis)
      continue;
    numBlocks *=
        ceil<unsigned>(getShape()[i], (contigPerThread[i] * threadsPerWarp[i] *
                                       warpsPerCTA[i]));
  }
  return numBlocks;
}

bool ScanLoweringHelper::isSupported() {
  // TODO: Support the following cases:
  // 1. Scan on non-blocking encodings
  if (!isa<BlockedEncodingAttr>(legacyEncoding))
    return false;
  return true;
}

unsigned ScanLoweringHelper::getScratchSizeInElems() {
  unsigned numWarps = product(getEncoding().getWarpsPerCTA());
  unsigned numNonAxisElementsPerWarp =
      getNonAxisNumThreadsPerWarp() * getNonAxisNumElementsPerThread();
  unsigned numElements = numWarps * numNonAxisElementsPerWarp *
                         getAxisNumBlocks() * getNonAxisNumBlocks();
  return numElements;
}

unsigned ScanLoweringHelper::getScratchSizeInBytes() {
  // Lowering will fail later if the layout is not supported.
  if (!isSupported())
    return 0;

  unsigned axisNumWarps = getAxisNumWarpsWithUniqueData();
  if (axisNumWarps == 1)
    return 0;
  unsigned elementSizeInBytes = 0;
  for (const auto &ty : srcElementTypes) {
    elementSizeInBytes += ceil<unsigned>(ty.getIntOrFloatBitWidth(), 8);
  }
  return elementSizeInBytes * getScratchSizeInElems();
}

std::optional<DecomposedWarpConversion>
getWarpLayoutConvertDecomposition(RankedTensorType srcTy,
                                  RankedTensorType dstTy) {
  auto conversion = minimalCvtLayout(srcTy, dstTy);

  MLIRContext *ctx = srcTy.getContext();
  auto kRegister = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");

  // We have already checked that data movement is only required within a warp,
  // thus we can discard the block and warp dimensions.
  LinearLayout C = conversion.sublayout({kLane, kRegister}, {kLane, kRegister});

  // `C` is map from `(dst_lane, dst_reg) -> (src_lane, src_reg)`. From the
  // perspetive of the destination lane, it tells us which register from which
  // lane to get the value. Since the source and destination layouts are
  // subpermutation matrices, the overall transformation amounts to permuting
  // data around (plus broadcasting, if necessary).
  //
  // Warp shuffles allow indexing into another lane, but does not allowing
  // selecting the register. Suppose we decompose `C` into `C = P1 ∘ W ∘ P2`,
  // where `W` is a warp shuffle and `P1` and `P2` are (lane-dependent) register
  // permutations within a lane. Start from `C` and work backwards.
  //
  // Given any `C`, is it possible that for a given destination register, two
  // destination lanes map to different source registers in the same source
  // lane. This is impossible to represent using a shuffle. This happens when,
  // with respect to the identity layout, a register base is swapped with a lane
  // base (when the destination lane changes, the source register changes but
  // the lane does not).
  //
  // Example:
  //
  //   src = {register = [[1,0], [2,0]], lane = [[0,1], [0,2]]}
  //   dst = {register = [[0,1], [2,0]], lane = [[1,0], [0,2]]}
  //   cvt = dst, since src is the identity layout
  //
  // The map from destination -> source looks like:
  //
  //             dst_lane
  // dst_reg       0      1      2      3
  //  0          T0:0   T0:1   T2:0   T2:1
  //  1          T1:0   T1:1   T3:0   T3:1
  //  2          T0:2   T0:3   T2:2   T2:3
  //  3          T1:2   T1:3   T3:2   T3:3
  //
  // Note for each destination register, two lanes want two different registers
  // in the same source lane (T0:0 -> T0:0, T1:0 -> T0:1). This is impossible to
  // represent with a warp shuffle, because the source lane (e.g. T0) can only
  // supply one of its registers as the shuffle value.
  //
  // The goal of `P2` is to permute registers within a thread so that this does
  // not happen. Specifically, pick `P2` such that bases in
  // `(P2^-1 ∘ C).sublayout(kLane, {kLane, kRegister})` has non-zero lane
  // components when the register components are non-zero.
  //
  // P2 can only change the register mapping within a thread. Constrain P2 as:
  //
  //   P2 = [ I 0 ]
  //        [ P I ]
  //
  // Then `P2^-1 ∘ C` is:
  //
  //   [ I  0 ] [ C(r,r) C(r,l) ] = [ C(r,r)             C(r,l)           ]
  //   [ P' I ] [ C(l,r) C(l,l) ]   [ P'*C(r,r)+C(l,r)   P'*C(r,l)+C(l,l) ]
  //
  // Where addition in GF(2) is xor.
  //
  // We can see that P' selects rows (i.e. bases) from the upper half (register)
  // and combines them with the lower half (lane). Because the goal is for P' to
  // select register bases `i` where C(r,l)[i] != 0, we know P'*C(r,r) = 0,
  // since the corresponding C(r,r)[i] element in the same row will be zero.
  //
  // Note that solutions for P' do not always exist (no register permutation
  // will decompose C to make the warp shuffle possible), and this happens when
  // there aren't enough non-zero bases in C(r,l).
  //
  // Find the indices of the missing lane bases: rows in the lower half where
  // the register component is non-zero but the lane component is zero.
  SmallVector<int> missingLaneRows;
  for (int i : llvm::seq(C.getInDimSizeLog2(kLane))) {
    ArrayRef<int32_t> /*C(l,(r,l))[i]*/ lowerHalfRow = C.getBasis(kLane, i);
    assert(lowerHalfRow.size() == 2);
    if (/*C(l,r)[i]*/ lowerHalfRow[0] != 0) {
      assert(/*C(l,l)[i]*/ lowerHalfRow[1] == 0);
      missingLaneRows.push_back(i);
    } else if (lowerHalfRow[1] == 0) {
      // If there is broadcasting along the lane, then C'(l,l) below won't be
      // invertible. Intuitively, the dst tensor contains a subset of the src
      // tensor's data, so recovering the src tensor through permutation alone
      // is impossible. We would need an affine component (bfly shuffle).
      return {};
    }
  }

  // Find rows in the upper-half  of C (i.e. the (reg) -> (reg, lane) submatrix)
  // that can be selected by P' to make the lane components in the lower half
  // (i.e. the (lane) -> (lane) submatrix) non-zero.
  std::vector<std::vector<int32_t>> PPrimeLaneBases(C.getInDimSizeLog2(kLane),
                                                    {0});
  for (int i : llvm::seq(C.getInDimSizeLog2(kRegister))) {
    ArrayRef<int32_t> /*C(r,(r,l))[i]*/ upperHalfRow = C.getBasis(kRegister, i);
    assert(upperHalfRow.size() == 2);
    if (/*C(r,l)[i]*/ upperHalfRow[1] == 0)
      continue;

    assert(upperHalfRow[0] == 0);
    int32_t laneBase = upperHalfRow[1];
    assert(/*C(r,r)[i]*/ upperHalfRow[0] == 0);
    if (!missingLaneRows.empty()) {
      // Select row i into row j from the missing rows. The order in which the
      // missing rows are selected doesn't really matter.
      PPrimeLaneBases[missingLaneRows.pop_back_val()][0] |= (1 << i);
    }
  }
  if (!missingLaneRows.empty()) {
    // The decomposition failed. No solution for P' is possible.
    return {};
  }

  // P' outputs the destination register.
  LinearLayout PPrime({{kLane, std::move(PPrimeLaneBases)}},
                      {{kRegister, C.getInDimSize(kRegister)}},
                      /*requiresSurjective=*/false);

  // Form P2^-1 from P'.
  unsigned dstRegSize = C.getInDimSize(kRegister);
  unsigned numLanes = C.getInDimSize(kLane);
  LinearLayout P2invTop =
      LinearLayout::identity1D(dstRegSize, kRegister, kRegister)
          .concatOuts(
              LinearLayout::zeros1D(dstRegSize, kRegister, kLane, numLanes));
  LinearLayout P2invBot =
      PPrime.concatOuts(LinearLayout::identity1D(numLanes, kLane, kLane));
  LinearLayout P2inv = P2invTop.concatIns(P2invBot);

  // Check that P2^-1 was formed correctly.
  assert(P2inv.sublayoutIsZero(kRegister, kLane));
  assert(squareSublayoutIsPermutation(P2inv, kLane));

  LinearLayout Cp = P2inv.compose(C);

  // Now we have C' = P2^-1 ∘ C = W ∘ P1. W is considerably easier to compute.
  // A warp shuffle is a function from `(register, lane) -> (lane)`, i.e.
  //
  //   W = [ I R' ]
  //       [ 0 L  ]
  //
  // `W^-1 ∘ C'` will be
  //
  //   [ I R ] [ C'(r,r) C'(r,l) ] = [ ... C'(r,l) + R*C'(l,l) ]
  //   [ 0 L ] [ C'(l,r) C'(l,l) ] = [ ... L*C'(l,l)           ]
  //
  // Since P1 cannot change lanes, we know that
  //
  //   W^-1 ∘ C' = [ ... 0 ]
  //               [ ... I ]
  //
  // Thus L = C'(l,l)^-1, and R = -C'(r,l) * C'(l,l)^-1. (0 - LL) = LL in GF(2).
  // We know that C'(l,l) has a suitable pseudo-inverse.
  LinearLayout L = Cp.sublayout(kLane, kLane).pseudoinvert();
  LinearLayout R = Cp.sublayout(kRegister, kLane).compose(L);

  // Now form W^-1.
  LinearLayout WinvLeft =
      LinearLayout::identity1D(dstRegSize, kRegister, kRegister)
          .concatIns(
              LinearLayout::zeros1D(numLanes, kLane, kRegister, dstRegSize));
  LinearLayout Winv = WinvLeft.concatOuts(R.concatIns(L));

  // Check that Winv was formed correctly. P1 is just what's left over.
  LinearLayout P1 = Winv.compose(Cp);
  assert(P1.sublayoutIsZero(kRegister, kLane));
  assert(squareSublayoutIsIdentity(P1, kLane));

  // Grab just the interesting parts of the decomposed layouts.
  P1 = P1.sublayout({kLane, kRegister}, kRegister);
  P2inv = P2inv.sublayout({kLane, kRegister}, kRegister);
  Cp = Cp.sublayout({kLane, kRegister}, kLane);

  // To minimize the number of selects emitted on the source side, determine the
  // minimum set of registers that could be selected from each thread.
  // InstCombine *might* be able to crush this, but if the sizePerThread is
  // large, it's truly a huge number of selects that get emitted.
  // If reducedP1 is trivial, then we will emit
  // shflSrc = select(i == i, src[i], undef) and this will get trivially folded,
  // so don't worry about this case.
  LinearLayout reducedP1 = P1.removeZeroBasesAlongDim(kLane);
  LinearLayout reducedP2 = P2inv.removeZeroBasesAlongDim(kLane);

  // The number of emitted selects can still be quite large if the layout is not
  // cooperative. This happens when the source register is more correlated
  // with the desination lane than the destination register (i.e. the number of
  // non-zero bases). The number of selects impacts performance and grows
  // exponentially with the number of non-zero bases. Experiments show that more
  // than 1 select causes performance to be slower than shared memory.
  if (reducedP1.getInDimSize(kLane) > 2 || reducedP2.getInDimSize(kLane) > 2)
    return {};

  // HACK: Workaround AMD codegen path generating transient invalid layouts.
  auto isInvalidDotEnc = [](RankedTensorType type) {
    auto dotEnc = dyn_cast<DotOperandEncodingAttr>(type.getEncoding());
    return dotEnc && dotEnc.getKWidth() == 0;
  };
  if (isInvalidDotEnc(srcTy) || isInvalidDotEnc(dstTy))
    return {};

  // When the element type is smaller than 32 bits, values are upcasted to i32
  // for shuffles. When the shared memory conversion can use vector stores of
  // sufficiently large length, the shared memory conversion is faster.
  // TODO: Implementing shuffling packed 16 and 8 bit values.
  auto [inVec, outVec] = getScratchCvtInOutVecLengths(srcTy, dstTy);
  if (!isa<PointerType>(srcTy.getElementType()) &&
      srcTy.getElementTypeBitWidth() < 32 && inVec > 4 && outVec > 4)
    return {};

  // Return just the interesting parts of the decomposed layouts.
  return {{std::move(P1), std::move(Cp), std::move(P2inv), std::move(reducedP1),
           std::move(reducedP2)}};
}

SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
getReshapeDecomposition(ArrayRef<int64_t> srcShape,
                        ArrayRef<int64_t> dstShape) {
  SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> ret;

  if (srcShape.empty()) {
    assert(dstShape.empty());
    return ret;
  }
  ret.push_back({});

  int srcIdx = 0;
  int dstIdx = 0;
  int srcNElems = 1;
  int dstNElems = 1;
  while (srcIdx < srcShape.size() || dstIdx < dstShape.size()) {
    if (srcNElems < dstNElems || //
        (srcIdx < srcShape.size() && srcNElems == 1) ||
        (srcIdx < srcShape.size() && srcShape[srcIdx] == 1)) {
      assert(srcIdx < srcShape.size());
      srcNElems *= srcShape[srcIdx];
      ret.back().first.push_back(srcIdx);
      srcIdx++;
    } else if (dstNElems < srcNElems ||
               (dstIdx < dstShape.size() && dstShape[dstIdx] == 1)) {
      assert(dstIdx < dstShape.size());
      dstNElems *= dstShape[dstIdx];
      ret.back().second.push_back(dstIdx);
      dstIdx++;
    } else {
      ret.push_back({});
      srcNElems = 1;
      dstNElems = 1;
    }
  }
  return ret;
}

unsigned ScanLoweringHelper::getAxisElementStride() {
  auto order = getOrder();
  unsigned stride = 1;
  for (unsigned dim : order) {
    if (dim == getAxis())
      return stride;
    stride *= getEncoding().getContigPerThread()[dim];
  }
  llvm_unreachable("Axis not found in order");
}

unsigned ScanLoweringHelper::getAxisThreadStride() {
  auto encoding = getEncoding();
  auto kThread = StringAttr::get(encoding.getContext(), "lane");
  // OOOGHHH This is nasty. We should implement this lowering via LLs natively
  // to avoid this
  auto threadsPerWarp = encoding.basesPerDim(kThread, /*skipBroadcast=*/false);
  auto order = getOrder();
  unsigned stride = 1;
  for (unsigned dim : order) {
    if (dim == getAxis())
      return stride;
    stride *= threadsPerWarp[dim];
  }
  llvm_unreachable("Axis not found in order");
}

unsigned ScanLoweringHelper::getAxisBlockStride() {
  auto order = getOrder();
  unsigned stride = 1;
  auto contigPerThread = getEncoding().getContigPerThread();
  auto threadsPerWarp = getEncoding().getThreadsPerWarp();
  auto warpsPerCTA = getEncoding().getWarpsPerCTA();
  for (unsigned dim : order) {
    if (dim == getAxis())
      return stride;
    stride *= ceil<unsigned int>(getShape()[dim], contigPerThread[dim] *
                                                      threadsPerWarp[dim] *
                                                      warpsPerCTA[dim]);
  }
  llvm_unreachable("Axis not found in order");
}

GatherLoweringHelper::GatherLoweringHelper(triton::GatherOp gatherOp)
    : gatherOp(gatherOp) {}

unsigned GatherLoweringHelper::getScratchSizeInBytes() {
  // If the gather is warp-local, no scratch space is needed.
  if (isWarpLocal())
    return 0;

  // Otherwise, performing the gather will require scratch space to communicate
  // the source tensor across threads. For now, assume the whole source tensor
  // is written back to shared memory.
  RankedTensorType srcType = gatherOp.getSrc().getType();
  return product(srcType.getShape()) *
         ceil<unsigned>(srcType.getElementTypeBitWidth(), 8);
}

bool GatherLoweringHelper::isWarpLocal() {
  // The gather is warp-local if for each column along the gather axis in the
  // source and index tensors, all the elements are owned by the same warp.
  RankedTensorType srcType = gatherOp.getSrc().getType();
  RankedTensorType idxType = gatherOp.getIndices().getType();
  LinearLayout srcLayout =
      toLinearLayout(srcType.getShape(), srcType.getEncoding());
  LinearLayout idxLayout =
      toLinearLayout(idxType.getShape(), idxType.getEncoding());

  Builder b(gatherOp.getContext());
  StringAttr kBlock = b.getStringAttr("block");
  StringAttr kWarp = b.getStringAttr("warp");
  StringAttr kLane = b.getStringAttr("lane");
  StringAttr kGatherDim =
      b.getStringAttr("dim" + std::to_string(gatherOp.getAxis()));

  // The tensor layouts must be distributed layouts, where the basis matrix is a
  // subpermutation matrix (permutation matrix plus zeros for broadcasting).
  // FIXME(jeff): Check this invariant somehow.
  //
  // We want to know if all elements of a column along the gather axis are
  // mapped to the same set of warps, which means the gather can be performed
  // entirely within the warp. We need to query
  //
  //   srcLayout.invert().sublayoutIsZero({kGatherDim}, {kBlock, kWarp})
  //
  // But due to broadcasting, the matrix might not be invertible. But since the
  // matrix is a permutation matrix (checked below), we can instead query
  //
  //   srcLayout.sublayoutIsZero({kBlock, kWarp}, {kGatherDim})
  //
  // Which implies that changing the warp will not change the gather dimension.
  // And since there is no swizzling, this applies to all warps.
  if (!srcLayout.sublayoutIsZero({kBlock, kWarp}, kGatherDim) ||
      !idxLayout.sublayoutIsZero({kBlock, kWarp}, kGatherDim))
    return false;

  SmallVector<StringAttr> otherDims;
  for (unsigned dim = 0, rank = srcType.getRank(); dim < rank; ++dim) {
    if (dim != gatherOp.getAxis()) {
      otherDims.push_back(b.getStringAttr("dim" + Twine(dim)));
    }
  }

  // If the gather axis `dimN` is invariant to the warp, but the `(block, warp)`
  // mapping to all other dimensions must be the same for both layouts. If so,
  // then the warp that owns a particular index element also owns all the source
  // elements it could index into.
  if (srcLayout.sublayout({kBlock, kWarp}, otherDims) !=
      idxLayout.sublayout({kBlock, kWarp}, otherDims))
    return false;

  // The two constraints above ensure that data-movement to perform the gather
  // operation are contained within a warp. The subsequent constraints simplify
  // codegen.

  // Require that for any given gather column, the threads mapped to the column
  // in the index and source tensors are the same. This means we don't need to
  // xor shuffle across threads before emitting index shuffles; we push warp
  // shuffling to layout conversions.
  return srcLayout.sublayout(kLane, otherDims) ==
         idxLayout.sublayout(kLane, otherDims);
}

unsigned getNumScratchElements(ArrayRef<unsigned> shape) {
  if (shape.empty())
    return 0;
  return product<unsigned>(shape);
}

bool supportMMA(triton::DotOp op, int version) {
  // Refer to mma section for the data type supported by Volta and Hopper
  // Tensor Core in
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f16
  auto aElemTy = op.getA().getType().getElementType();
  auto bElemTy = op.getB().getType().getElementType();
  if (version == 5) {
    if (triton::tools::getBoolEnv("DISABLE_MMA_V5"))
      return false;
    auto retType = op.getType();
    auto retShapePerCTA = getShapePerCTA(retType);
    auto rank = retShapePerCTA.size();
    int numWarps = lookupNumWarps(op);
    if (aElemTy.isInteger() || bElemTy.isInteger() ||
        retType.getElementType().isInteger())
      return false;
    if (op.getType().getRank() != 2)
      return false;
    if (numWarps != 4 && numWarps != 8) {
      // Currently only support numWarps 4 or 8 for TMEM load and store.
      return false;
    }
    if (!(retShapePerCTA[rank - 2] % 64 == 0 &&
          retShapePerCTA[rank - 1] % 8 == 0))
      return false;
    return true;
  }
  if (version == 3) {
    if (triton::tools::getBoolEnv("DISABLE_MMA_V3"))
      return false;
    auto retType = op.getType();
    RankedTensorType typeA = op.getA().getType();
    int k = typeA.getShape().back();
    // If k size is smaller than the native mma size, we cannot use MMA.
    if (k < 256 / aElemTy.getIntOrFloatBitWidth())
      return false;
    auto retShapePerCTA = getShapePerCTA(retType);
    auto rank = retShapePerCTA.size();
    int numWarps = lookupNumWarps(op);
    // TODO(Keren): for now, fallback to MMAv2 if handling batch matmul.
    if (rank == 3)
      return false;
    if (!(numWarps % 4 == 0 && retShapePerCTA[rank - 2] % 64 == 0 &&
          retShapePerCTA[rank - 1] % 8 == 0 &&
          (llvm::isa<Float8E5M2Type, Float8E4M3FNType>(aElemTy) ||
           aElemTy.isInteger(8) || aElemTy.isF16() || aElemTy.isBF16() ||
           aElemTy.isF32()))) {
      return false;
    }
    // We cannot use MMA_V3 if we need to accumulate in F32 within the MMA op.
    if (op.getMaxNumImpreciseAcc() < 32 &&
        (llvm::isa<Float8E5M2Type, Float8E4M3FNType>(aElemTy)) &&
        cast<RankedTensorType>(op.getType()).getElementType().isF32()) {
      return false;
    }
  }
  if (aElemTy.isF32() && bElemTy.isF32()) {
    return op.getInputPrecision() == InputPrecision::TF32 && version >= 2;
  }
  return supportMMA(op.getA(), version) && supportMMA(op.getB(), version);
}

bool supportMMA(Value value, int version) {
  // Tell whether a DotOp support MMA by the operand type(either $a or $b).
  // We cannot get both the operand types(in TypeConverter), here we assume the
  // types of both the operands are identical here.
  assert((version == 1 || version == 2 || version == 3) &&
         "Unexpected MMA layout version found");
  auto elemTy =
      cast<triton::gpu::TensorOrMemDesc>(value.getType()).getElementType();
  // FP8 is not natively supported on all mma versions but it can always be
  // promoted to fp16 therefore we can always support it.
  bool isFP8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType, Float8E5M2FNUZType,
                         Float8E4M3FNUZType>(elemTy);
  return isFP8 || elemTy.isF16() || elemTy.isBF16() ||
         (elemTy.isF32() && version >= 2) ||
         (elemTy.isInteger(8) && version >= 2);
}

// For MMAV3 dotOperand layout matches mma operand for f16 and bf16 cases.
bool matchMmaV3AndDotOperandLayout(RankedTensorType srcTy,
                                   RankedTensorType dstTy) {
  auto mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(srcTy.getEncoding());
  auto dotOperandLayout = dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
  if (!mmaLayout || !dotOperandLayout) {
    return false;
  }
  int elementTypeSize = srcTy.getElementType().getIntOrFloatBitWidth();
  auto parentTy = RankedTensorType::get(
      srcTy.getShape(), srcTy.getElementType(), dotOperandLayout.getParent());
  auto ans = mmaLayout.getVersionMajor() == 3 &&
             dotOperandLayout.getOpIdx() == 0 &&
             mmaLayout.getWarpsPerCTA()[1] == 1 &&
             !cvtNeedsSharedMemory(parentTy, srcTy) && elementTypeSize == 8 &&
             dotOperandLayout.getKWidth() == 32 / elementTypeSize;
  return ans;
}

bool matchMFMAAndDotOperandShuffleCase(RankedTensorType srcTy,
                                       RankedTensorType dstTy) {
  auto mfmaLayout = dyn_cast<AMDMfmaEncodingAttr>(srcTy.getEncoding());
  auto dotOperandLayout = dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
  if (!mfmaLayout || !dotOperandLayout)
    return false;

  // Currently supporting 32x32 and 16x16 FP8 MFMA -> dot operand case
  return dotOperandLayout.getParent() == mfmaLayout &&
         dotOperandLayout.getOpIdx() == 0 && mfmaLayout.getIsTransposed() &&
         dotOperandLayout.getKWidth() == 8 &&
         ((mfmaLayout.getMDim() == 16 && mfmaLayout.getNDim() == 16) ||
          (mfmaLayout.getMDim() == 32 && mfmaLayout.getNDim() == 32)) &&
         triton::type::isFloat8(srcTy.getElementType()) &&
         triton::type::isFloat8(dstTy.getElementType()) &&
         mfmaLayout.getWarpsPerCTA()[1] == 1;
}

// We get the smallest submap of srcTy^{-1} * dstTy that is not the identity
// under kBlock, kWarp or kLane (in that order). The idea here is that if we
// have a transformation that's the identity on kBlock, we don't need to use
// distributed shared memory. If it's also the identity on kWarp, we can
// transfer via warp-shuffles, and if it's the identity on kLane just have to
// reorder the registers
LinearLayout minimalCvtLayout(RankedTensorType srcTy, RankedTensorType dstTy) {
  MLIRContext *ctx = srcTy.getContext();
  LinearLayout srcLayout =
      toLinearLayout(srcTy.getShape(), srcTy.getEncoding());
  LinearLayout dstLayout =
      toLinearLayout(dstTy.getShape(), dstTy.getEncoding());
  StringAttr kRegister = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  StringAttr kBlock = StringAttr::get(ctx, "block");

  auto comp = dstLayout.invertAndCompose(srcLayout);
  // We try to quotient by the largest subspace first
  auto dims = SmallVector<StringRef>{"block", "warp", "lane", "register"};
  for (auto dim : dims) {
    auto quotient = comp.quotient(StringAttr::get(ctx, dim));
    if (!quotient.has_value()) {
      break;
    }
    comp = *quotient;
  }
  return comp;
}

bool cvtReordersRegisters(RankedTensorType srcTy, RankedTensorType dstTy) {
  auto layout = minimalCvtLayout(srcTy, dstTy);
  MLIRContext *ctx = srcTy.getContext();
  auto kRegister = StringAttr::get(ctx, "register");
  auto outDims = to_vector(layout.getOutDimNames());
  return outDims.empty() || ArrayRef(outDims) == ArrayRef({kRegister});
}

bool cvtNeedsWarpShuffle(RankedTensorType srcTy, RankedTensorType dstTy) {
  auto layout = minimalCvtLayout(srcTy, dstTy);
  MLIRContext *ctx = srcTy.getContext();
  auto kRegister = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  return to_vector(layout.getOutDimNames()) ==
         SmallVector<StringAttr, 2>{kRegister, kLane};
}

bool cvtNeedsSharedMemory(RankedTensorType srcTy, RankedTensorType dstTy) {
  // TODO(jlebar): Remove these special cases `isMfmaToDotShortcut` once
  // they're fully subsumed by the linear-layout checks.
  return !cvtReordersRegisters(srcTy, dstTy) &&
         !(cvtNeedsWarpShuffle(srcTy, dstTy) &&
           getWarpLayoutConvertDecomposition(srcTy, dstTy)) &&
         !matchMmaV3AndDotOperandLayout(srcTy, dstTy) &&
         // to be removed when generalized warp shuffle conversions
         // are ready:
         !matchMFMAAndDotOperandShuffleCase(srcTy, dstTy);
}

bool atomicNeedsSharedMemory(Value value) {
  auto type = value.getType();
  if (isa<RankedTensorType>(type) || value.use_empty())
    return false;
  return true;
}

namespace {

/// A data structure similar to SetVector but maintains
/// a deque instead of a vector to allow for efficient
/// push_back and pop_front operations.
/// Using SetVector doesn't suffice our needs because
/// it only pushes and pops from the back.
/// For example, if we have a queue like this:
/// 0->4 1->2->3
///    ^--------
/// where 3 depends on 4, once we pop 3, we found
/// 4 is not ready, so we check 2 and push 3 back
/// to the queue.
struct DFSSubgraphState {
  DFSSubgraphState() : set(), deque() {}
  DenseSet<Operation *> set;
  std::deque<Operation *> deque;

  bool push_back(Operation *op) {
    if (set.insert(op).second) {
      deque.push_back(op);
      return true;
    }
    return false;
  }

  Operation *pop_front() {
    Operation *op = deque.front();
    deque.pop_front();
    set.erase(op);
    return op;
  }

  bool empty() { return deque.empty(); }
};

/// DFS post-order implementation that maintains a global count to work across
/// multiple invocations, to help implement topological sort on multi-root DAGs.
/// We traverse all operations but only record the ones that appear in
/// `toSort` for the final result.
struct DFSState {
  DFSState(const SetVector<Operation *> &set) : toSort(set), seen() {}
  const SetVector<Operation *> &toSort;
  SmallVector<Operation *, 16> topologicalCounts;
  DenseSet<Operation *> seen;

  /// We mark each op as ready if all its operands and parents ops are seen. If
  /// an op is ready, we add it to the queue. Otherwise, we keep adding its
  /// operands to the ancestors set.
  /// We always want an op to be scheduled after all its parents to handle
  /// correctly cases with scf operations.
  void addToReadyQueue(Operation *op, DFSSubgraphState &subGraph,
                       SmallVector<Operation *, 4> &readyQueue) {
    bool ready = true;
    for (Value operand : op->getOperands()) {
      auto def = operand.getDefiningOp();
      if (def && !seen.count(def)) {
        subGraph.push_back(def);
        ready = false;
      }
    }
    Operation *parent = op->getParentOp();
    while (parent) {
      if (!seen.count(parent)) {
        subGraph.push_back(parent);
        ready = false;
      }
      parent = parent->getParentOp();
    }
    if (ready)
      readyQueue.push_back(op);
  }
};

void dfsPostorder(Operation *root, DFSState *state) {
  DFSSubgraphState subGraph;
  subGraph.push_back(root);
  SmallVector<Operation *> ops;
  while (!subGraph.empty()) {
    // Nodes in the ready queue are ready to be processed.
    // Meaning that either their operands are all seen or they have null
    // operands.
    SmallVector<Operation *, 4> readyQueue;
    auto *current = subGraph.pop_front();
    state->addToReadyQueue(current, subGraph, readyQueue);
    while (!readyQueue.empty()) {
      Operation *current = readyQueue.pop_back_val();
      if (!state->seen.insert(current).second)
        continue;
      ops.push_back(current);
      for (Value result : current->getResults()) {
        for (Operation *op : result.getUsers())
          state->addToReadyQueue(op, subGraph, readyQueue);
      }
      for (Region &region : current->getRegions()) {
        for (Operation &op : region.getOps())
          state->addToReadyQueue(&op, subGraph, readyQueue);
      }
    }
  }

  for (Operation *op : llvm::reverse(ops)) {
    if (state->toSort.count(op) > 0)
      state->topologicalCounts.push_back(op);
  }
}

} // namespace

SetVector<Operation *>
multiRootTopologicalSort(const SetVector<Operation *> &toSort) {
  if (toSort.empty()) {
    return toSort;
  }

  // Run from each root with global count and `seen` set.
  DFSState state(toSort);
  for (auto *s : toSort) {
    assert(toSort.count(s) == 1 && "NYI: multi-sets not supported");
    dfsPostorder(s, &state);
  }

  // Reorder and return.
  SetVector<Operation *> res;
  for (auto it = state.topologicalCounts.rbegin(),
            eit = state.topologicalCounts.rend();
       it != eit; ++it) {
    res.insert(*it);
  }
  return res;
}

SetVector<Operation *> multiRootGetSlice(Operation *op,
                                         TransitiveFilter backwardFilter,
                                         TransitiveFilter forwardFilter) {
  SetVector<Operation *> slice;
  slice.insert(op);

  unsigned currentIndex = 0;
  SetVector<Operation *> backwardSlice;
  SetVector<Operation *> forwardSlice;
  while (currentIndex != slice.size()) {
    auto *currentOp = (slice)[currentIndex];
    // Compute and insert the backwardSlice starting from currentOp.
    backwardSlice.clear();
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = backwardFilter;
    getBackwardSlice(currentOp, &backwardSlice, opt);
    slice.insert(backwardSlice.begin(), backwardSlice.end());

    // Compute and insert the forwardSlice starting from currentOp.
    forwardSlice.clear();
    getForwardSlice(currentOp, &forwardSlice, forwardFilter);
    slice.insert(forwardSlice.begin(), forwardSlice.end());
    ++currentIndex;
  }
  return multiRootTopologicalSort(slice);
}

namespace {
// Copied from TestDeadCodeAnalysis.cpp, because some dead code analysis
// interacts with constant propagation, but SparseConstantPropagation
// doesn't seem to be sufficient.
class ConstantAnalysis : public DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;

  LogicalResult initialize(Operation *top) override {
    WalkResult result = top->walk([&](Operation *op) {
      ProgramPoint programPoint(op);
      if (failed(visit(&programPoint)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return success(!result.wasInterrupted());
  }

  LogicalResult visit(ProgramPoint *point) override {
    Operation *op = point->getOperation();
    Attribute value;
    if (matchPattern(op, m_Constant(&value))) {
      auto *constant = getOrCreate<dataflow::Lattice<dataflow::ConstantValue>>(
          op->getResult(0));
      propagateIfChanged(constant, constant->join(dataflow::ConstantValue(
                                       value, op->getDialect())));
      return success();
    }
    // Dead code analysis requires every operands has initialized ConstantValue
    // state before it is visited.
    // https://github.com/llvm/llvm-project/blob/2ec1aba2b69faa1de5f71832a48e25aa3b5d5314/mlir/lib/Analysis/DataFlow/DeadCodeAnalysis.cpp#L322
    // That's why we need to set all operands to unknown constants.
    setAllToUnknownConstants(op->getResults());
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks())
        setAllToUnknownConstants(block.getArguments());
    }
    return success();
  }

private:
  /// Set all given values as not constants.
  void setAllToUnknownConstants(ValueRange values) {
    dataflow::ConstantValue unknownConstant(nullptr, nullptr);
    for (Value value : values) {
      auto *constant =
          getOrCreate<dataflow::Lattice<dataflow::ConstantValue>>(value);
      propagateIfChanged(constant, constant->join(unknownConstant));
    }
  }
};
} // namespace

std::unique_ptr<DataFlowSolver> createDataFlowSolver() {
  auto solver = std::make_unique<DataFlowSolver>();
  solver->load<dataflow::DeadCodeAnalysis>();
  solver->load<ConstantAnalysis>();
  return solver;
}

static MakeTensorPtrOp getMakeTensorPtrOpImpl(Operation *op, Value v) {

  if (auto makeTensorPtrOp = dyn_cast<MakeTensorPtrOp>(op)) {
    return makeTensorPtrOp;
  }

  if (auto advanceOp = dyn_cast<AdvanceOp>(op)) {
    return getMakeTensorPtrOp(advanceOp.getPtr());
  }

  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    auto idx = cast<OpResult>(v).getResultNumber();
    llvm::SmallVector<scf::YieldOp> yieldOps;
    op->walk([&](Operation *op) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op))
        yieldOps.push_back(yieldOp);
    });

    // benzh@ if multi yields, all yields operand should come from same arg.
    Value newValue = yieldOps[0].getOperands()[idx];
    return getMakeTensorPtrOp(newValue);
  }

  llvm_unreachable("Unable to getMakeTensorPtr()");
}

MakeTensorPtrOp getMakeTensorPtrOp(Value v) {
  using BranchOps = llvm::SetVector<std::pair<Operation *, int>>;
  llvm::DenseMap<Block *, BranchOps> blockToCFOps;
  auto moduleOp =
      v.getParentBlock()->getParentOp()->getParentOfType<ModuleOp>();

  moduleOp.walk([&](Operation *op) {
    if (auto br = dyn_cast<cf::BranchOp>(op)) {
      Block *block = br.getDest();
      blockToCFOps[block].insert({op, -1});
    }
    if (auto condBr = dyn_cast<cf::CondBranchOp>(op)) {
      Block *blockT = condBr.getTrueDest();
      Block *blockF = condBr.getFalseDest();
      blockToCFOps[blockT].insert({condBr, 1});
      blockToCFOps[blockF].insert({condBr, 0});
    }
  });

  if (Operation *definingOp = v.getDefiningOp())
    return getMakeTensorPtrOpImpl(definingOp, v);

  // If there is no defining op, v must be a BlockArgument.
  BlockArgument arg = cast<BlockArgument>(v);
  unsigned argNum = arg.getArgNumber();
  Operation *argOwner = arg.getOwner()->getParentOp();

  if (auto forOp = dyn_cast<scf::ForOp>(argOwner))
    return getMakeTensorPtrOp(
        forOp.getOperand(argNum + forOp.getNumControlOperands() - 1));
  if (auto funcOp = dyn_cast<FunctionOpInterface>(argOwner)) {
    Block *block = arg.getOwner();
    Operation *op;
    int tOrF;
    std::tie(op, tOrF) = blockToCFOps[block][0];
    if (auto br = dyn_cast<cf::BranchOp>(op))
      return getMakeTensorPtrOp(br.getDestOperands()[argNum]);
    if (auto condBr = dyn_cast<cf::CondBranchOp>(op))
      return getMakeTensorPtrOp(tOrF ? condBr.getTrueDestOperands()[argNum]
                                     : condBr.getFalseDestOperands()[argNum]);
    return getMakeTensorPtrOp(argOwner->getOperand(argNum));
  }
  llvm_unreachable("Unable to getMakeTensorPtr()");
}

} // namespace mlir
