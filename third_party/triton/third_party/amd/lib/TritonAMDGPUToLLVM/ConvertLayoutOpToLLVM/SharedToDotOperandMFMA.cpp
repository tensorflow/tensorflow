/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
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
#include "../PatternTritonGPUOpToLLVM.h"
#include "../TritonAMDGPUToLLVM/SchedInstructions.h"
#include "SharedToDotOperandHelper.h"
#include "Utility.h"

using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::SwizzledSharedEncodingAttr;

namespace SharedToDotOperandMFMA {

/// This function maps particular load of mfma dot operand to element
/// indexes(row, col)
///
/// Whole tensor is broken into "blocks" of warps along "non-K" axis.
/// One block could be processed by multiple warps.
/// One warp works on a piece of tensor size elemsPerInstr[0] x K.
/// Each of these pieces is broken into "tiles" of size elemsPerInstr[0] x
/// elemsPerInstr[1].
///
/// Total offset of element is a sum of following values:
/// 1. Offset of warp-block in tensor
/// 2. Offset of warp inside one warp-block
/// 3. Offset of tile in one warp
/// 4. Offset of one lane data in a tile
/// 5. Offset of particular element of tensor processed by one lane
///
/// This function computes these offsets for axies independently
/// Note that this function returns the offsets of elements in the first
/// warp-block. The offsets of elements in later warp-blocks can be computed
/// by adding a constant stride to the xor-ed offsets of elements in the
/// first warp-block.
///
/// \param rewriter
/// \param loc
/// \param elemsPerInstr operand tile shape consumed by one MFMA instruction
/// \param warpId id component of 2d warp grid along non-K axis
/// \param laneId lane id in warp [0..63]
/// \param numOfElems number of elements accessed by thread per repetition
/// \param reps number of instructions repetition to fully cover dot operand
/// \param smemStrides strides in LDS tensor
/// \param loadVecSize number of elements loaded by one operation
/// \param iNonKDim non-K dimension size of one MFMA instruction
/// \param iKDim K dimension size of one MFMA instruction
/// \returns vector (i-th element corresponds to i-th load instruction) of
/// 2-element vectors(tensor row and col).
llvm::SmallVector<llvm::SmallVector<Value>> computeTensorElemMappingInBlock(
    ConversionPatternRewriter &rewriter, Location loc,
    const ArrayRef<int64_t> &elemsPerInstr, Value warpId, Value laneId,
    int numOfElems, ArrayRef<int64_t> reps, ArrayRef<Value> smemOffsets,
    int loadVecSize, unsigned iNonKDim, unsigned iKDim) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto numM = reps[1];
  auto numK = reps[2];
  const int loadsPerThread = numOfElems / loadVecSize;
  llvm::SmallVector<llvm::SmallVector<Value>> mapping(numK * loadsPerThread);

  Value _0 = b.i32_val(0);
  Value _32 = b.i32_val(32);
  Value nonKDim = b.i32_val(iNonKDim);
  Value warpVOffset = b.mul(warpId, b.i32_val(elemsPerInstr[0]));

  auto rank = smemOffsets.size();

  for (int tile = 0; tile < numK; ++tile) {
    Value tileVOffset = _0;
    Value tileHOffset = b.i32_val(tile * elemsPerInstr[1]);

    Value laneVOffset = b.urem(laneId, nonKDim);
    Value laneHOffset;
    if (iNonKDim == 32) {
      laneHOffset =
          b.select(b.icmp_uge(laneId, _32), b.i32_val(numOfElems), _0);
    } else {
      // In this configuration warp contains 16 copies of same data
      if ((iKDim == 1 || iKDim == 4) && iNonKDim == 4) {
        laneHOffset = b.i32_val(0);
      } else {
        assert(iKDim * iNonKDim / numOfElems == 64 &&
               "seems no all threads in warp contain unique elements");
        laneHOffset = b.mul(b.udiv(laneId, nonKDim), b.i32_val(numOfElems));
      }
    }

    for (int loadId = 0; loadId < loadsPerThread; ++loadId) {
      Value elemVOffset = _0;
      Value elemHOffset = b.i32_val(loadId * loadVecSize);

      Value sliceVOffset = b.add(
          b.add(b.add(tileVOffset, laneVOffset), elemVOffset), warpVOffset);
      Value sliceHOffset = b.add(b.add(tileHOffset, laneHOffset), elemHOffset);

      Value row = b.add(sliceVOffset, smemOffsets[rank - 2]);
      Value col = b.add(sliceHOffset, smemOffsets[rank - 1]);

      mapping[loadsPerThread * tile + loadId] = {row, col};
    }
  }
  return mapping;
}

bool hasSwizzleEnabled(const SwizzledSharedEncodingAttr &srcEncoding) {
  return srcEncoding.getMaxPhase() > 1;
}

/// Computes offsets for operand B or transposed operand A
///
/// \param rewriter
/// \param loc
/// \param elemsPerInstr operand tile shape [K, nonK] consumed by one MFMA
/// instruction
/// \param warpId warp id for the "non K" axis
/// \param laneId lane id in warp [0..63]
/// \param warpsPerBlock number of warps per horizontal axis
/// \param numOfElems number of elements accessed by threads per repetition
/// \param reps number of instructions repretition to fully cover dot operand
/// \param cSwizzleOffset
llvm::SmallVector<Value>
fastPathComputeOffsets(ConversionPatternRewriter &rewriter, Location loc,
                       const ArrayRef<int64_t> &elemsPerInstr, Value warpId,
                       Value laneId, int warpsPerBlock, int numOfElems,
                       ArrayRef<int64_t> reps, Value cSwizzleOffset) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto numK = reps[1];
  auto numN = reps[2];
  SmallVector<Value> offsets(numK * numN * numOfElems);

  auto iKDim = elemsPerInstr[0];
  auto iNonKDim = elemsPerInstr[1];
  int lineSize = warpsPerBlock * iNonKDim * numN;
  Value _nonKDim = b.i32_val(iNonKDim);
  Value warpOffset = b.mul(warpId, b.i32_val(iNonKDim));
  Value colOffset = b.urem(laneId, _nonKDim);

  for (int block = 0; block < numN; ++block) {
    Value blockOffset = b.i32_val(block * iNonKDim * warpsPerBlock);
    for (int tile = 0; tile < numK; ++tile) {
      Value tileOffset = b.i32_val(tile * iKDim * lineSize);
      for (int elem = 0; elem < numOfElems; ++elem) {
        // halfOffset is an offset related to wrapping of warp in the tile.
        // for example, mfma 32 case (mapping of tensor elements to lane ids in
        // warp):
        //
        //  0  1  2  3 ... 31
        //  0  1  2  3 ... 31
        //  0  1  2  3 ... 31
        //  0  1  2  3 ... 31
        // 32 33 34 35 ... 63  <- at this point warp is wrapping
        // 32 33 34 35 ... 63
        // 32 33 34 35 ... 63
        // 32 33 34 35 ... 63
        Value halfOffset;
        if ((iKDim == 1 || iKDim == 4) && iNonKDim == 4)
          halfOffset = b.i32_val(0);
        else
          halfOffset =
              b.mul(b.udiv(laneId, _nonKDim), b.i32_val(numOfElems * lineSize));
        Value rowOffset = b.add(b.i32_val(elem * lineSize), halfOffset);
        Value elemOffset = b.add(rowOffset, colOffset);
        Value offset = b.add(b.add(b.add(warpOffset, blockOffset), tileOffset),
                             elemOffset);
        offsets[numK * numOfElems * block + numOfElems * tile + elem] = offset;
      }
    }
  }
  return offsets;
}

bool isColMajor(::llvm::ArrayRef<unsigned> order) {
  auto rank = order.size();
  return order[0] == (rank - 2);
}

Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread) {
  // We observe mismatches going down this path for scaled dot operations.
  // However, these mismatches do not occur when the conversion is performed via
  // LL, which is the preferred anyway.
  // TODO: Deprecate and remove this path once fixing LLM performance issues.
  for (auto op : tensor.getUsers()) {
    if (auto localLoadOp = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op)) {
      if (mlir::LLVM::AMD::isUsedByDotScaledOp(op))
        return Value();
    }
  }

  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  assert((opIdx == 0 || opIdx == 1) && "unexpected operand idx");
  auto aTensorTy = cast<triton::gpu::MemDescType>(tensor.getType());
  ArrayRef<int64_t> shape = aTensorTy.getShape();
  auto rank = shape.size();
  int kDimIdx = opIdx == 0 ? rank - 1 : rank - 2;
  int nonKDimIdx = opIdx == 0 ? rank - 2 : rank - 1;

  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(encoding.getParent());
  auto mDim = mfmaLayout.getMDim();
  auto nDim = mfmaLayout.getNDim();
  assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
         (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

  auto sharedLayout =
      dyn_cast<SwizzledSharedEncodingAttr>(aTensorTy.getEncoding());
  if (!sharedLayout)
    return Value();
  auto order = sharedLayout.getOrder();
  assert((rank == 2 || order[2] == 0) &&
         "expect batch to be the slowest dimension");

  auto elemTy = aTensorTy.getElementType();
  auto kWidth = encoding.getKWidth();
  auto elemsPerInstr = mfmaLayout.getInstrShapeForOperand(kWidth, opIdx);

  int64_t mfmaInstrNonK;
  int64_t mfmaInstrK;
  // TODO(Lixun): make it simpler
  // getInstrShapeForOperand always returns a 2D vector
  if (rank == 3) {
    mfmaInstrNonK = elemsPerInstr[nonKDimIdx - 1];
    mfmaInstrK = elemsPerInstr[kDimIdx - 1];
  } else {
    mfmaInstrNonK = elemsPerInstr[nonKDimIdx];
    mfmaInstrK = elemsPerInstr[kDimIdx];
  }

  if (mfmaInstrNonK > shape[nonKDimIdx] || mfmaInstrK > shape[kDimIdx]) {
    // This pattern does not support cases tensor shape is smaller than
    // one instruction size, it will be processed by LinearLayout converter
    return Value();
  }

  auto numReps = mfmaLayout.getRepForOperand(shape, kWidth, opIdx);
  auto numRepNonK = numReps[nonKDimIdx];
  auto numRepK = numReps[kDimIdx];
  auto repB = numReps[0];
  // TODO(Lixun): make it simpler
  // getRepForOperand always returns a 3D vector
  if (rank == 2) {
    numRepNonK = numReps[nonKDimIdx + 1];
    numRepK = numReps[kDimIdx + 1];
  }

  unsigned iWarpSize = triton::gpu::lookupThreadsPerWarp(rewriter);
  assert(iWarpSize == 64);
  Value warpSize = tb.i32_val(iWarpSize);
  Value linearWarpId = tb.udiv(thread, warpSize);
  Value lane = tb.urem(thread, warpSize);

  auto warpOrder = triton::gpu::getMatrixOrder(rank, /*rowMajor*/ true);

  Value spatialWarpId = AMD::getWarpIdInBlock(
      rewriter, loc, linearWarpId, warpsPerCTA, mfmaInstrNonK,
      shape[nonKDimIdx], nonKDimIdx, warpOrder);

  // number of duplicates of elements in warp
  // In case of 64x4 x 4x4 multiplication, 4x4 B operand is duplicated 16 times
  int numSubBlocks = 1;
  if ((mfmaInstrK == 4 || mfmaInstrK == 1) && mfmaInstrNonK == 4)
    numSubBlocks = 16;
  // numOfElemsPerThreadPerMfmaInstr
  int numOfElems = mfmaInstrNonK * mfmaInstrK * numSubBlocks / iWarpSize;
  assert(numOfElems >= 1);

  unsigned int maxNumWarps = shape[nonKDimIdx] / mfmaInstrNonK;
  int warpsPerBlockNonK = std::min(warpsPerCTA[nonKDimIdx], maxNumWarps);
  int warpsPerBatch =
      rank == 3 ? std::min<unsigned>(shape[0], warpsPerCTA[0]) : 1;
  Value warpIdInBatch = tb.urem(linearWarpId, tb.i32_val(warpsPerBatch));
  elemTy = typeConverter->convertType(elemTy);

  SmallVector<Value> loadedValues;
  SmallVector<Value> offsets;
  Value smemBase;
  auto smemStrides = smemObj.getStrides(aTensorTy, loc, rewriter);
  bool isFastPath =
      !AMD::isKContig(order, opIdx) && !hasSwizzleEnabled(sharedLayout);
  if (isFastPath) {
    // fast path handles tensors that are not k-major and have swizzling
    // disabled, in which case offsets computation can be simplified
    // TODO (zhanglx): later when we enable vector access to LDS for non k-major
    // tensors, we'll refactor the scope of fast and normal path
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
    if (opIdx == 0) {
      if (isColMajor(order)) {
        SmallVector<int64_t> elemsPerInstr{mfmaInstrK, mfmaInstrNonK};
        SmallVector<int64_t> reps{numReps[0], numReps[2], numReps[1]};
        offsets = fastPathComputeOffsets(rewriter, loc, elemsPerInstr,
                                         spatialWarpId, lane, warpsPerBlockNonK,
                                         numOfElems, reps, cSwizzleOffset);
      } else {
        llvm_unreachable(
            "row major operand A should be handled in the normal path");
      }
    } else {
      if (isColMajor(order)) {
        llvm_unreachable(
            "col major operand B should be handled in the normal path");
      } else {
        offsets = fastPathComputeOffsets(rewriter, loc, elemsPerInstr,
                                         spatialWarpId, lane, warpsPerBlockNonK,
                                         numOfElems, numReps, cSwizzleOffset);
      }
    }
    smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);
  } else { // normal path
    // Normal path handles tensors that fall into either of the following three
    // cases:
    //   1. k-major + swizzling is enabled <-- this should be the most
    //   performant case
    //   2. k-major + swizzling is disabled <-- for testing purpose only
    //   3. non k-major + swizzling is enabled <-- for testing purpose only
    if (opIdx == 0) {
      offsets = AMD::computeOffsetsAType(
          rewriter, loc, computeTensorElemMappingInBlock, elemsPerInstr,
          spatialWarpId, lane, warpsPerBlockNonK, numOfElems, numReps, smemObj,
          smemStrides, sharedLayout, mDim, mfmaInstrK);
    } else {
      assert(opIdx == 1);
      offsets = AMD::computeOffsetsBType(
          rewriter, loc, computeTensorElemMappingInBlock, elemsPerInstr,
          spatialWarpId, lane, warpsPerBlockNonK, numOfElems, numReps, smemObj,
          smemStrides, sharedLayout, nDim, mfmaInstrK);
    }
    smemBase = AMD::computeBasePtr(rewriter, loc, smemObj, smemStrides);
  }

  Type resElemTy = typeConverter->convertType(elemTy);
  Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);

  int loadsPerThread = offsets.size() / numRepK / numRepNonK;
  int elemsPerLoad = numOfElems / loadsPerThread;
  assert(numOfElems % loadsPerThread == 0);

  VectorType loadVecTy = vec_ty(elemTy, elemsPerLoad);
  for (int b = 0; b < repB; ++b) {
    int operandSize = shape[rank - 1] * shape[rank - 2];
    Value batchOffset =
        tb.mul(tb.i32_val(operandSize),
               tb.add(warpIdInBatch, tb.i32_val(b * warpsPerBatch)));
    for (int nonK = 0; nonK < numRepNonK; ++nonK) {
      int blockNonKOffset = nonK * mfmaInstrNonK * warpsPerBlockNonK;
      Value warpBlockOffAdjust = tb.i32_val(blockNonKOffset * shape[order[0]]);
      for (int k = 0; k < numRepK; ++k) {
        auto vecTy = vec_ty(resElemTy, numOfElems);
        for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
          Value loadOffset;
          loadOffset = offsets[nonK * loadsPerThread * numRepK +
                               k * loadsPerThread + loadId];
          loadOffset = tb.add(loadOffset, batchOffset);
          Value loadAddress = tb.gep(smemPtrTy, elemTy, smemBase, loadOffset);
          Value loadedValue = tb.load(loadVecTy, loadAddress);
          for (int elemId = 0; elemId < elemsPerLoad; ++elemId) {
            Value elemVal =
                tb.extract_element(elemTy, loadedValue, tb.i32_val(elemId));
            loadedValues.push_back(elemVal);
          }
        }
      }
    }
  }

  for (auto op : tensor.getUsers()) {
    if (auto localLoadOp = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op)) {
      const size_t numDsReadsCount =
          repB * numRepNonK * numRepK * loadsPerThread;
      setNumGeneratedDsReads(localLoadOp, numDsReadsCount, loadVecTy);
    }
  }

  MLIRContext *ctx = mfmaLayout.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(loadedValues.size(), loadedValues[0].getType()));
  auto result =
      packLLElements(loc, typeConverter, loadedValues, rewriter, structTy);
  return result;
}

} // namespace SharedToDotOperandMFMA
