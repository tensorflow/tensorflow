#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
class GatherOpConversion : public ConvertOpToLLVMPattern<GatherOp> {
public:
  GatherOpConversion(LLVMTypeConverter &typeConverter,
                     const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  // Codegen the gather by storing the source tensor into shared memory and then
  // gathering directly from shared memory.
  void emitGatherInShared(GatherOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const;
  // Codegen a warp-local gather by shuffling elements across the warp and
  // selecting from them.
  void emitWarpLocalGather(GatherOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const;

  const TargetInfoBase &targetInfo;
};

LogicalResult
GatherOpConversion::matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  GatherLoweringHelper helper(op);
  // Specialize the lowering based on the source layout. Given that the cost of
  // a warp shuffle is approximately half the cost of a roundtrip to shared
  // memory with zero bank conflicts, we will need a more precise heuristic to
  // choose between the two codegen paths and rely on the middle end to pick the
  // right layout.
  if (helper.isWarpLocal()) {
    emitWarpLocalGather(op, adaptor, rewriter);
  } else {
    emitGatherInShared(op, adaptor, rewriter);
  }
  return success();
}

static Value convertIndexToI32(Location loc, Value index,
                               ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned idxWidth = index.getType().getIntOrFloatBitWidth();
  // The LL index computations are performed with 32 bit integers. If the
  // indices are something else, cast them to i32.
  if (idxWidth > 32) {
    index = b.trunc(i32_ty, index);
  } else if (idxWidth < 32) {
    // Negative indices don't make sense, so zero-extend.
    index = b.zext(i32_ty, index);
  }
  return index;
}

void GatherOpConversion::emitGatherInShared(
    GatherOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  RankedTensorType srcType = op.getSrc().getType();

  // Compute the src subtensor shape owned by this CTA.
  SmallVector<unsigned> srcShapePerCTA =
      convertType<unsigned>(triton::gpu::getShapePerCTA(srcType));

  // Grab the src values in this thread.
  SmallVector<Value> srcValues =
      unpackLLElements(loc, adaptor.getSrc(), rewriter);

  // Emit the indices of the src values owned by this thread.
  SmallVector<SmallVector<Value>> srcIndices =
      emitIndices(loc, rewriter, targetInfo, srcType.getEncoding(),
                  op.getSrc().getType(), /*withCTAOffset=*/true);

  // Store the src values owned by the thread into their respective location in
  // the scratch memory.
  assert(srcValues.size() == srcIndices.size());

  // Get the base pointer to the scratch memory.
  Value smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);

  // For each src element owned by the thread, index into the scratch memory and
  // then store it.
  Type elemType = getTypeConverter()->convertType(srcType.getElementType());
  for (auto [value, indices] : llvm::zip(srcValues, srcIndices)) {
    // Convert the index at each dim into a single offset given the shape of the
    // tensor.
    Value offset = LLVM::linearize(rewriter, loc, indices, srcShapePerCTA);
    // Emit the offset into the shared memory and then store the value.
    Value ptr = b.gep(smemBase.getType(), elemType, smemBase, offset);
    b.store(value, ptr);
  }

  // Synchronize the whole CTA.
  b.barrier();

  // Grab the index values owned by this thread.
  SmallVector<Value> idxValues =
      unpackLLElements(loc, adaptor.getIndices(), rewriter);

  // Apply the layout of the destination tensor to obtain the indices of the
  // column to gather along, then for each column, replace the index along the
  // gather axis with the appropriate index value.
  //
  // I = LL(pid)
  // idx = indices[I]
  // I_gather = [I[d] if d != axis else idx for d in range(len(I))]
  // out[I] = src[I_gather]
  RankedTensorType dstType = op.getType();
  SmallVector<SmallVector<Value>> dstIndices =
      emitIndices(loc, rewriter, targetInfo, dstType.getEncoding(), dstType,
                  /*withCTAOffset=*/true);

  unsigned axis = op.getAxis();
  SmallVector<Value> results(dstIndices.size());
  for (auto [i, idx, indices] : llvm::enumerate(idxValues, dstIndices)) {
    indices[axis] = convertIndexToI32(loc, idx, rewriter);
    Value offset = LLVM::linearize(rewriter, loc, indices, srcShapePerCTA);
    Value ptr = b.gep(smemBase.getType(), elemType, smemBase, offset);
    results[i] = b.load(elemType, ptr);
  }

  Value packed =
      packLLElements(loc, getTypeConverter(), results, rewriter, dstType);
  rewriter.replaceOp(op, packed);
}

// High-level description of the algorithm:
//
// `isWarpLocal` checks that it is possible to compute each output element
// without data movement across warps.
//
// If the gather dim is `dimN`, then this means
//
//   ll^-1(dimN)[(block, warp)] == 0
//
// for both source and index tensors: moving along the gather axis does not
// change the warp. Broadcasted layouts are not supported, so we know the
// layouts are permutation matrices.
//
// We can check this with `ll((block, warp))[dimN] == 0`.
//
// Let `gatherCol` be a tuple of all dimensions except the gather dimension.
// We also check that the gather columns line up the same way with respect to
// the warp between the source and index tensors with
//
//   ll_src((block, warp))[gatherCol] == ll_idx((block, warp))[gatherCol]
//
// This means that for all index columns, the corresponding column in the source
// tensor is owned by the same warp.
//
// We also check
//
//   ll_src(lane)[gatherCol] == ll_idx(lane)[gatherCol]
//
// This boils down to the fact that the algorithm essentially emits a series of
// index shuffles for each index value owned by each thread, and then a pile of
// selects to pick the right value. We need to figure out given an index value
// in a particular column, what are the source register values it could read
// from and who owns them.
//
// If this relationship did not hold, then the possible source registers for
// each index value varies with the thread, meaning the value operand provided
// to each shuffle index instruction would depend on the thread ID. This isn't a
// big deal. It just means would have to emit a pile of selects before each
// shuffle as well, to pick the right source register value. But we choose not
// to handle this.
//
// The codegen algorithm emits code:
// - Given the thread ID and a particular index tensor register, figure out
//   which gather column it belongs to using a layout.
// - Using the index value itself as the value for `dimN`, use another layout to
//   figure out which lane in the warp owns the desired value and which register
//   in that lane it is.
// - For the gather column, figure out the source registers in that column, and
//   for each of them, emit an index shuffle with the same computed lane ID.
// - Use the register component to select the right value from the shuffle
//   results.
void GatherOpConversion::emitWarpLocalGather(
    GatherOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  RankedTensorType srcType = op.getSrc().getType();
  RankedTensorType idxType = op.getIndices().getType();

  // Layout dimension names.
  StringAttr kBlock = str_attr("block");
  StringAttr kWarp = str_attr("warp");
  StringAttr kLane = str_attr("lane");
  StringAttr kRegister = str_attr("register");
  StringAttr kGatherDim = rewriter.getStringAttr("dim" + Twine(op.getAxis()));
  SmallVector<StringAttr> allDims, otherDims;
  for (unsigned dim = 0, rank = srcType.getRank(); dim < rank; ++dim) {
    allDims.push_back(str_attr("dim" + Twine(dim)));
    if (dim != op.getAxis()) {
      otherDims.push_back(allDims.back());
    }
  }

  // Compute the src and idx layouts.
  LinearLayout srcLayout =
      toLinearLayout(srcType.getShape(), srcType.getEncoding());
  LinearLayout idxLayout =
      toLinearLayout(idxType.getShape(), idxType.getEncoding());

  // Let `ll_src` be the source layout and `ll_idx` be the index layout.
  // Let `src_col` be a tuple of dimensions except the gather dimension,
  // representing a specific column in the source tensor. Likewise for
  // `idx_col`. Let `src_idx` be the index into gather dimension in the source
  // tensor.
  //
  // `(src_lane, src_reg) = ll_src^-1(src_col, src_idx)`, where `src_lane` is
  // the thread that contains the required element and `src_reg` is the register
  // within that thread.
  //
  // Because `ll_src(block=0, warp=0, lane=0)[otherDims] ==
  // ll_idx(0, 0, 0)[otherDims]`, we know given any `idx_reg` (element in the
  // index tensor) the thread will need to read from the same column in the
  // source tensor.
  //
  // Thus, we can obtain
  //
  //   (src_lane, src_reg) = (ll_src^-1)(
  //       ll_idx(black, warp, lane, idx_reg)[otherDims],
  //       idxValues[idx_reg]
  //   )[{"lane", "register"}]
  //
  // And the mapping will be the correct for each thread.
  //
  // Given `src_reg \in [0, K*N)`, we just need to emit N index shuffles for
  // each `idx_reg` (the number of index shuffles is quadratic!) and
  // `llvm.select` using `src_reg` to get the right one. `K` is the number of
  // elements per column owned by a thread.

  // Invert the source layout. It doesn't matter whether it is fully invertible
  // with respect to anything except the register input dimension, since we know
  // those don't vary in ways that matter for codegen.
  LinearLayout invSrcLayout = srcLayout.pseudoinvert();

  // Sanity check: the warp must be invariant to the index because otherwise the
  // gather would need to read across warps!
  assert(invSrcLayout.sublayoutIsZero(kGatherDim, {kBlock, kWarp}) &&
         "expected a warp-local gather");
  invSrcLayout = invSrcLayout.sublayout(allDims, {kLane, kRegister});

  LinearLayout idxColLayout =
      idxLayout.sublayout({kBlock, kWarp, kLane, kRegister}, otherDims);

  SmallVector<Value> srcValues =
      unpackLLElements(loc, adaptor.getSrc(), rewriter);
  SmallVector<Value> idxValues =
      unpackLLElements(loc, adaptor.getIndices(), rewriter);

  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = targetInfo.getClusterCTAId(rewriter, loc);

  unsigned /*N=*/srcRegsPerThread = srcLayout.getInDimSize(kRegister);
  assert(srcRegsPerThread == srcValues.size());

  // Given a index value, we need to know which sources register values it could
  // index into. This is invariant to anything other than the register, which we
  // checked already. Compute the full reverse map from
  //
  //   idx_reg -> gather_column -> (src_reg0, src_reg1, ...)
  //
  LinearLayout invertSrcRegMap = invSrcLayout.sublayout(allDims, {kRegister});
  // Remove zero bases in the gather dimension to make the function injective
  // (for a given column) over the same codomain.
  invertSrcRegMap = invertSrcRegMap.removeZeroBasesAlongDim(kGatherDim);
  // We are left with only non-zero bases in the gather dimension, which means
  // the number of registers per column is the size of the "gather dimension".
  unsigned numRegsPerColumn = invertSrcRegMap.getInDimSize(kGatherDim);
  // Get a map from idx_reg to the column it indexes into.
  LinearLayout idxRegToCol = idxLayout.sublayout({kRegister}, otherDims);
  // Now given `idx_reg`, we can compute the column it belongs to in both src
  // and index tensors, then partially apply `invertSrcRegMap` with this to
  // obtain a function that outputs the corresponding registers in the src
  // tensor in the same column.

  // L(column, i) = L(column, 0) xor L(0, i)
  LinearLayout invertSrcRegMapColPart =
      invertSrcRegMap.sublayout(otherDims, {kRegister});
  LinearLayout invertSrcRegMapRest =
      invertSrcRegMap.sublayout({kGatherDim}, {kRegister});

  SmallVector<Value> results;
  for (auto [idxReg, idxVal] : llvm::enumerate(idxValues)) {
    SmallVector<std::pair<StringAttr, Value>> column =
        applyLinearLayout(loc, rewriter, idxColLayout,
                          {{kBlock, blockId},
                           {kWarp, warpId},
                           {kLane, laneId},
                           {kRegister, b.i32_val(idxReg)}});
    assert(column.size() == otherDims.size());

    // Combine the computed column with the data-dependent gather index.
    column.emplace_back(kGatherDim, convertIndexToI32(loc, idxVal, rewriter));
    SmallVector<std::pair<StringAttr, Value>> srcLaneAndReg =
        applyLinearLayout(loc, rewriter, invSrcLayout, column);

    auto [srcLaneName, srcLane] = srcLaneAndReg.back();
    auto [srcRegName, srcReg] = srcLaneAndReg.front();
    assert(srcLaneName == kLane && srcRegName == kRegister);

    assert(!srcValues.empty() && "can't gather from an empty tensor");

    // Figure out which src registers we need to index shuffle from. This is
    // invariant to anything else.
    SmallVector<std::pair<StringAttr, int32_t>> normalizedColumn =
        idxRegToCol.apply({{kRegister, idxReg}});
    int32_t srcBase =
        invertSrcRegMapColPart.apply(normalizedColumn).front().second;

    Value result = b.undef(srcValues.front().getType());
    for (unsigned i = 0; i != numRegsPerColumn; ++i) {
      int32_t rest =
          invertSrcRegMapRest.apply({{kGatherDim, i}}).front().second;
      int32_t srcRegIdx = srcBase ^ rest;

      Value value =
          targetInfo.shuffleIdx(rewriter, loc, srcValues[srcRegIdx], srcLane);
      result = b.select(b.icmp_eq(b.i32_val(srcRegIdx), srcReg), value, result);
    }

    results.push_back(result);
  }

  rewriter.replaceOp(op, packLLElements(loc, getTypeConverter(), results,
                                        rewriter, op.getType()));
}

} // namespace

void triton::populateGatherOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            const TargetInfoBase &targetInfo,
                                            PatternBenefit benefit) {
  patterns.insert<GatherOpConversion>(typeConverter, targetInfo, benefit);
}
