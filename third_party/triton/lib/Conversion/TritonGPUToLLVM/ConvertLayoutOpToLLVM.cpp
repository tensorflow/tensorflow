#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

namespace {

using namespace mlir;
using namespace mlir::triton::gpu;

struct ConvertLayoutOpUsingLinearLayoutsConversion
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
  const TargetInfoBase &targetInfo;

  // Set benefit to 2 so that this pattern applies before other convert-layout
  // conversions.  TODO(jlebar): Eventually we want this to be the only pattern.
  explicit ConvertLayoutOpUsingLinearLayoutsConversion(
      LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();

    const auto &shape = op.getType().getShape();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
    LinearLayout srcLayout =
        toLinearLayout(srcTy.getShape(), srcTy.getEncoding());
    LinearLayout dstLayout =
        toLinearLayout(dstTy.getShape(), dstTy.getEncoding());

    StringAttr kBlock = str_attr("block");
    StringAttr kWarp = str_attr("warp");
    StringAttr kLane = str_attr("lane");
    StringAttr kRegister = str_attr("register");

    assert(to_vector(conversion.getInDimNames()) ==
           to_vector(conversion.getOutDimNames()));
    auto dims = conversion.getInDimNames();
    if (llvm::is_contained(dims, kBlock)) {
      // Case 1: Transfer between values in different CTAs.
      //          This requires moving values through distributed shared memory.
      return rewriter.notifyMatchFailure(
          op, "NYI: Transfer between different CTAs");
    } else if (llvm::is_contained(dims, kWarp)) {
      // Case 2: Transfer between values in the same CTA, in which case we move
      //         values through shared memory.
      return transferWithinBlock(op, srcLayout, dstLayout, adaptor, rewriter);
    } else if (llvm::is_contained(dims, kLane)) {
      // Case 3. Transfer between values in the same warp, in which case we try
      //         to move values using warp shuffles, though if the pattern is
      //         complicated enough we may fall back to using shared memory
      if (auto decomposedCvt =
              getWarpLayoutConvertDecomposition(srcTy, dstTy)) {
        transferWithinWarp(op, *decomposedCvt, adaptor, rewriter);
        return success();
      }
      // TODO: Since data is only transferred within a warp over shared memory,
      // we should use `bar.warp.sync` instead of `barrier`, which will improve
      // latency when warps issue barriers on different cycles.
      return transferWithinBlock(op, srcLayout, dstLayout, adaptor, rewriter);
    } else if (llvm::is_contained(dims, kRegister)) {
      // Case 4. Transfer between values in the same thread, in which case we
      //         simply reorder the elements of adaptor.getSrc().
      return transferWithinThread(op, conversion, adaptor, rewriter);
    } else {
      // Cast 5. The two layouts are equivalent. We should probably remove
      // these in RemoveLayoutConversion.
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
  }

  LogicalResult
  transferWithinThread(ConvertLayoutOp op, const LinearLayout &conversion,
                       OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    StringAttr kRegister = str_attr("register");
    assert(!cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> outVals(conversion.getInDimSize(kRegister));
    for (int i = 0; i < outVals.size(); i++) {
      auto srcIdx = conversion.apply({{kRegister, i}}).begin()->second;
      outVals[i] = inVals[srcIdx];
    }
    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult transferWithinBlock(ConvertLayoutOp op,
                                    const LinearLayout &srcLayout,
                                    const LinearLayout &dstLayout,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    assert(cvtNeedsSharedMemory(srcTy, dstTy));

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    assert(!inVals.empty());

    // We munge the input values by converting i<n> (n<8) elements to i8 and
    // pointers to i64. This is necessary because TargetInfo::loadDShared and
    // storeDShared can't handle vectors of pointers or sub-byte elements.
    auto elemTy = srcTy.getElementType();
    auto isSubByteInt =
        elemTy.isInteger() && elemTy.getIntOrFloatBitWidth() < 8;
    auto isPtr = isa<triton::PointerType>(elemTy);
    auto llvmElemTyOrig = getTypeConverter()->convertType(elemTy);
    if (isSubByteInt)
      elemTy = IntegerType::get(elemTy.getContext(), 8);
    else if (isPtr)
      elemTy = IntegerType::get(elemTy.getContext(), 64);
    auto llvmElemTy = getTypeConverter()->convertType(elemTy);

    // Munge input values
    for (const auto &it : llvm::enumerate(inVals)) {
      if (isSubByteInt) {
        inVals[it.index()] = b.zext(llvmElemTy, it.value());
      } else if (isPtr) {
        inVals[it.index()] = b.ptrtoint(llvmElemTy, it.value());
      }
    }

    // Pretty sure this is the identity function ATM
    // It'd be better to simply call `quotient({kBlock})` and
    // remove kBlock from transferWithinBlockImpl
    auto srcLayoutWithinBlock = getLayoutWithinBlock(srcLayout);
    auto dstLayoutWithinBlock = getLayoutWithinBlock(dstLayout);
    SmallVector<Value> outVals =
        transferWithinBlockImpl(inVals, op, srcLayoutWithinBlock,
                                dstLayoutWithinBlock, adaptor, rewriter);

    // Unmunge output values
    for (const auto &it : llvm::enumerate(outVals)) {
      if (isSubByteInt) {
        outVals[it.index()] = b.trunc(llvmElemTyOrig, it.value());
      } else if (isPtr) {
        outVals[it.index()] = b.inttoptr(llvmElemTyOrig, it.value());
      }
    }

    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

  // Use warp shuffles to implement a layout conversion where data only needs to
  // be moved within warps.
  void transferWithinWarp(ConvertLayoutOp op,
                          DecomposedWarpConversion decomposed,
                          OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const;

  SmallVector<Value>
  transferWithinBlockImpl(ArrayRef<Value> inVals, ConvertLayoutOp op,
                          const LinearLayout &srcLayout,
                          const LinearLayout &dstLayout, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    StringAttr kOffset = str_attr("offset");
    StringAttr kIteration = str_attr("iteration");

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

    auto scratchConfig =
        getScratchConfigForCvt(op.getSrc().getType(), op.getType());
    auto tensorShapePerCTA = convertType<unsigned, int64_t>(getShapePerCTA(
        op.getSrc().getType().getEncoding(), op.getType().getShape()));
    // Input dims: [offset, iteration, block]
    // Output dims: dimN-1, dimN-2, ..., dim0, where N is obtained from repShape
    LinearLayout sharedLayout = chooseShemLayoutForRegToRegConversion(
        ctx, tensorShapePerCTA, scratchConfig.repShape, scratchConfig.order);

    // Layout for the store from registers to shared memory.
    //
    // Note: If two threads in the same warp write to the same shmem offset, the
    // hardware resolves that without a stall or a bank conflict.  Therefore we
    // don't need to avoid duplicate writes.
    // Input dims: [reg, lane, warp]
    // Output dims: [offset, iteration]
    bool isStMatrix = targetInfo.canUseStMatrix(
        op.getSrc().getType(), scratchConfig.repShape,
        scratchConfig.paddedRepShape, scratchConfig.order,
        /*swizzleByteSize=*/0);
    LinearLayout shmemStoreLayout =
        isStMatrix ? chooseStMatrixLayout(ctx, op.getSrc().getType(),
                                          /*swizzleByteSize=*/0)
                   : srcLayout.invertAndCompose(sharedLayout);

    const int shmemAllocatedNumElems =
        getNumScratchElements(scratchConfig.paddedRepShape);
    assert(shmemStoreLayout.getOutDimSize(kOffset) <= shmemAllocatedNumElems);

    // Layout for the load from shmem to registers.
    LinearLayout shmemLoadLayout = dstLayout.invertAndCompose(sharedLayout);

    // Check that the `register` fully determines the `iteration`.  That is,
    // each thread does exactly the same reads and writes to shmem on each
    // iteration, just with different input/output registers.
    assert(
        shmemStoreLayout.sublayoutIsZero({kLane, kWarp, kBlock}, {kIteration}));
    assert(
        shmemLoadLayout.sublayoutIsZero({kLane, kWarp, kBlock}, {kIteration}));

    // iteration -> registers
    SmallVector<SmallVector<int>> inRegsForIter =
        collectRegsForIter(ctx, shmemStoreLayout);
    SmallVector<SmallVector<int>> outRegsForIter =
        collectRegsForIter(ctx, shmemLoadLayout);

    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto sharedPtrTy = smemBase.getType();
    Type elemTy = inVals[0].getType();
    auto outSize = shmemLoadLayout.getInDimSize(kRegister);
    auto iterations = sharedLayout.getInDimSize(kIteration);
    assert(scratchConfig.inVec * iterations <= inVals.size());
    assert(scratchConfig.outVec * iterations <= outSize);

    // Check only one dimension has been padded.
    // This means the difference between the padded shape and the original shape
    // should only be in one dimension, specifically in
    // `scratchConfig.order[0]`.
    auto rank = scratchConfig.repShape.size();
    for (auto i = 0; i < rank; i++) {
      if (i == scratchConfig.order[0]) {
        continue;
      }
      assert(scratchConfig.repShape[i] == scratchConfig.paddedRepShape[i]);
    }
    auto paddedStride = scratchConfig.repShape[scratchConfig.order[0]];
    auto paddedSize =
        scratchConfig.paddedRepShape[scratchConfig.order[0]] - paddedStride;

    // Linear layout function is split in two parts below:
    //
    // L(r, t, w, b) = L(0, t, w, b) xor L(r, 0, 0, 0)
    //   offset      =    regBase   xor    regIdx
    //
    // It is the same hack as what we've done in the emitIndices function to get
    // around performance issues on AMD GPUs
    auto getVecAddr = [&](LinearLayout &layout, Value &regBase,
                          int regSlice) -> Value {
      auto regIdx = layout
                        .apply({{kRegister, regSlice},
                                {kLane, 0},
                                {kWarp, 0},
                                {kBlock, 0}})[0]
                        .second;
      Value offset = b.xor_(regBase, b.i32_val(regIdx));
      if (paddedSize > 0) {
        assert(llvm::isPowerOf2_32(paddedStride));
        assert(llvm::isPowerOf2_32(paddedSize));
        auto rshiftVal = llvm::Log2_32(paddedStride);
        auto lshiftVal = llvm::Log2_32(paddedSize);
        offset = b.add(
            b.shl(b.lshr(offset, b.i32_val(rshiftVal)), b.i32_val(lshiftVal)),
            offset);
      }
      auto vecAddr = b.gep(sharedPtrTy, elemTy, smemBase, offset);
      vecAddr.setInbounds(true);
      return vecAddr;
    };

    auto storeBase = applyLinearLayout(loc, rewriter, shmemStoreLayout,
                                       {{kRegister, b.i32_val(0)},
                                        {kLane, laneId},
                                        {kWarp, warpId},
                                        {kBlock, b.i32_val(0)}})[0]
                         .second;
    auto loadBase = applyLinearLayout(loc, rewriter, shmemLoadLayout,
                                      {{kRegister, b.i32_val(0)},
                                       {kLane, laneId},
                                       {kWarp, warpId},
                                       {kBlock, b.i32_val(0)}})[0]
                        .second;
    // register idx -> Value
    llvm::MapVector<int, Value> outVals;
    for (int i = 0; i < iterations; i++) {
      if (i != 0)
        b.barrier();

      auto &inRegs = inRegsForIter[i];
      auto &outRegs = outRegsForIter[i];

      // When using `stmatrix`, we can store `inVec` elements even if they are
      // not contiguous
      auto inVec = isStMatrix ? shmemStoreLayout.getNumConsecutiveInOut()
                              : scratchConfig.inVec;
      for (int j = 0; j < inVals.size() / iterations; j += inVec) {
        auto inRegSlice = inRegs[j];
        Value vecAddr = getVecAddr(shmemStoreLayout, storeBase, inRegSlice);
        SmallVector<Value> inValsVec;
        for (int k = 0; k < inVec; k++)
          inValsVec.push_back(inVals[inRegSlice + k]);
        Value valsVec = packLLVector(loc, inValsVec, rewriter);
        if (isStMatrix) {
          targetInfo.storeMatrixShared(rewriter, loc, vecAddr, valsVec);
        } else {
          targetInfo.storeDShared(rewriter, loc, vecAddr, std::nullopt, valsVec,
                                  /*pred=*/b.true_val());
        }
      }

      b.barrier();

      for (int j = 0; j < outSize / iterations; j += scratchConfig.outVec) {
        auto outRegSlice = outRegs[j];
        auto vecAddr = getVecAddr(shmemLoadLayout, loadBase, outRegSlice);
        Value valsVec =
            targetInfo.loadDShared(rewriter, loc, vecAddr, std::nullopt,
                                   vec_ty(elemTy, scratchConfig.outVec),
                                   /*pred=*/b.true_val());
        for (Value v : unpackLLVector(loc, valsVec, rewriter))
          outVals[outRegSlice++] = v;
      }
    }

    SmallVector<Value> outValsVec;
    for (size_t i = 0; i < outVals.size(); i++)
      outValsVec.push_back(outVals[i]);
    return outValsVec;
  }

  // Determine which registers are read/written in which iteration of the shmem
  // transfer specified by `layout`.
  SmallVector<SmallVector<int> /*registers*/>
  collectRegsForIter(MLIRContext *ctx, const LinearLayout &layout) const {
    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    StringAttr kIteration = str_attr("iteration");

    // The choice of iteration should be determined only by the register.  That
    // is, it should be correct to split the register dimension into iterations.
    assert(layout.sublayoutIsZero({kLane, kWarp, kBlock}, {kIteration}));

    LinearLayout sublayout = layout.sublayout({kRegister}, {kIteration});
    SmallVector<SmallVector<int>> ret(sublayout.getOutDimSize(kIteration));
    for (int reg = 0; reg < sublayout.getInDimSize(kRegister); reg++) {
      auto idx = sublayout.apply({{kRegister, reg}});
      ret[idx.begin()->second].push_back(reg);
    }
    return ret;
  }
};

} // namespace

void ConvertLayoutOpUsingLinearLayoutsConversion::transferWithinWarp(
    ConvertLayoutOp op, DecomposedWarpConversion decomposed, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  assert(!cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));
  auto [P1, Cp, P2inv, reducedP1, reducedP2] = std::move(decomposed);

  // Grab the source elements and prepare the outputs of just the shuffles.
  SmallVector<Value> srcValues =
      unpackLLElements(loc, adaptor.getSrc(), rewriter);
  SmallVector<Value> shflOuts(Cp.getInDimSize(kRegister));

  Value laneId = getLaneId(rewriter, loc);

  // Emit one shuffle per destination register.
  for (int i : llvm::seq(shflOuts.size())) {
    // 'Cp' maps a (dst_lane, dst_reg) -> (src_lane, src_reg), and we know that
    // for a register, it does not map to different registers in the same lane.
    // At the same time, for each register, P1 returns the source value index
    // to provide as the shuffle value.
    auto out = applyLinearLayout(loc, rewriter, P1,
                                 {{kLane, laneId}, {kRegister, b.i32_val(i)}});
    assert(out.size() == 1);
    Value srcRegIdx = out.front().second;
    // The size of the input lane dimension is the number of selects to emit.
    // TODO(jeff): For dtypes smaller than i32, we can use byte permutes and
    // shuffle multiple values at a time.
    Value shflSrc = b.undef(srcValues.front().getType());
    for (int j : llvm::seq(reducedP1.getInDimSize(kLane))) {
      int32_t check =
          reducedP1.apply({{kLane, j}, {kRegister, i}}).front().second;
      shflSrc = b.select(b.icmp_eq(srcRegIdx, b.i32_val(check)),
                         srcValues[check], shflSrc);
    }

    out = applyLinearLayout(loc, rewriter, Cp,
                            {{kLane, laneId}, {kRegister, b.i32_val(i)}});
    assert(out.size() == 1);
    Value shflIdx = out.front().second;
    shflOuts[i] = targetInfo.shuffleIdx(rewriter, loc, shflSrc, shflIdx);
  }

  // Finally, we just need to apply P2 to the shflOuts to permute the registers
  // into their final form. Use the same trick to reduce the number of emitted
  // selects.
  SmallVector<Value> results(shflOuts.size());
  for (int i : llvm::seq(results.size())) {
    Value result = b.undef(srcValues.front().getType());

    auto out = applyLinearLayout(loc, rewriter, P2inv,
                                 {{kLane, laneId}, {kRegister, b.i32_val(i)}});
    Value resultIdx = out.front().second;
    for (int j : llvm::seq(reducedP2.getInDimSize(kLane))) {
      int32_t check =
          reducedP2.apply({{kLane, j}, {kRegister, i}}).front().second;
      result = b.select(b.icmp_eq(resultIdx, b.i32_val(check)), shflOuts[check],
                        result);
    }
    results[i] = result;
  }

  Value result =
      packLLElements(loc, getTypeConverter(), results, rewriter, op.getType());
  rewriter.replaceOp(op, result);
}

void mlir::triton::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpUsingLinearLayoutsConversion>(
      typeConverter, targetInfo, benefit);
}
