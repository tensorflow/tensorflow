#include "SharedToDotOperandHelper.h"

using ::mlir::triton::gpu::SwizzledSharedEncodingAttr;

namespace mlir::triton::AMD {

// Get warpId inside block of warps.
Value getWarpIdInBlock(ConversionPatternRewriter &rewriter, Location loc,
                       Value warpId, const ArrayRef<unsigned int> &wpt,
                       int elemPerInstrNonK, int tensorSizeNonK, int nonKIdx,
                       const ArrayRef<unsigned int> &order) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, wpt, order);

  return b.urem(multiDimWarpId[nonKIdx],
                b.i32_val(tensorSizeNonK / elemPerInstrNonK));
}

bool isSwizzled(SwizzledSharedEncodingAttr layout) {
  return layout.getMaxPhase() != 1;
}

std::pair<mlir::Value, mlir::Value>
swizzleIndexes(ConversionPatternRewriter &rewriter, Location loc, Value row,
               Value col, SharedMemoryObject smemObj,
               SwizzledSharedEncodingAttr attr) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  (void)smemObj; // unused in current pattern
  const auto &order = attr.getOrder();
  auto rank = order.size();
  bool transposed = (order[rank - 2] != 1);
  if (transposed) {
    // tensor is column-wise, so swapping col and row in computations
    std::swap(row, col);
  }
  auto vec = b.i32_val(attr.getVec());
  auto perPhase = b.i32_val(attr.getPerPhase());
  auto maxPhase = b.i32_val(attr.getMaxPhase());

  // phase = (row // perPhase) % maxPhase
  // colOffSwizzled = ((col // vec) ^ phase) * vec
  // colOffOrdered = col % vec
  // colOff = colOffSwizzled + colOffOrdered
  auto phase = b.urem(b.udiv(row, perPhase), maxPhase);
  auto colOffSwizzled = b.mul(b.xor_(b.udiv(col, vec), phase), vec);
  auto colOffOrdered = b.urem(col, vec);
  auto colOff = b.add(colOffSwizzled, colOffOrdered);

  if (transposed)
    return {colOff, row};
  else
    return {row, colOff};
}

Value computeOffset(ConversionPatternRewriter &rewriter, Location loc,
                    Value row, Value col, SharedMemoryObject smemObj,
                    ArrayRef<Value> smemStrides,
                    SwizzledSharedEncodingAttr srcLayout) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto [swizzledRow, swizzledCol] =
      swizzleIndexes(rewriter, loc, row, col, smemObj, srcLayout);
  auto rank = smemStrides.size();
  assert(rank == 2 || rank == 3);
  Value rowOffset = b.mul(swizzledRow, smemStrides[rank - 2]);
  Value colOffset = b.mul(swizzledCol, smemStrides[rank - 1]);
  return b.add(rowOffset, colOffset);
}

Value computeBasePtr(ConversionPatternRewriter &rewriter, Location loc,
                     const SharedMemoryObject &smemObj,
                     ArrayRef<Value> smemStrides) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value base = smemObj.getBase();
  Type type = base.getType();
  Type elemType = smemObj.getBaseElemType();
  for (int i = 0; i < smemStrides.size(); ++i) {
    Value offset =
        b.sub(b.i32_val(0), b.mul(smemObj.getOffsets()[i], smemStrides[i]));
    base = b.gep(type, elemType, base, offset);
  }
  return base;
}

bool isKContig(llvm::ArrayRef<unsigned> order, int opIdx) {
  auto rank = order.size();
  int kdim = opIdx == 0 ? rank - 1 : rank - 2;
  return order[0] == kdim;
}

/// Checks that swizzle pattern fits into one warp block
/// and block size is a multiple of swizzle size along non-K dimension
///
/// \param sharedLayout
/// \param opIdx operand id 0 or 1
/// \param reps number of repetitions: [non-k, k] or [batch, non-k, k]
/// \param elemsPerInstr one instruction size
/// \param warpsPerBlockNonK number of warps along non-k Dim
/// \returns bool
bool isSwizzlePatternFitsIntoBlock(
    const SwizzledSharedEncodingAttr sharedLayout, int opIdx,
    const ArrayRef<int64_t> reps, const ArrayRef<int64_t> elemsPerInstr,
    unsigned warpsPerBlockNonK) {
  assert(elemsPerInstr.size() == 2);
  unsigned mfmaInstrNonK = elemsPerInstr[opIdx == 0 ? 0 : 1];
  unsigned mfmaInstrK = elemsPerInstr[opIdx == 0 ? 1 : 0];
  auto order = sharedLayout.getOrder();
  const auto swizzleFastDimSize =
      sharedLayout.getMaxPhase() * sharedLayout.getVec();
  const auto swizzleSlowDimSize =
      sharedLayout.getMaxPhase() * sharedLayout.getPerPhase();
  const auto swizzlePatternSizeK =
      isKContig(order, opIdx) ? swizzleFastDimSize : swizzleSlowDimSize;
  const auto swizzlePatternSizeNonK =
      !isKContig(order, opIdx) ? swizzleFastDimSize : swizzleSlowDimSize;

  const auto blockSizeK = mfmaInstrK * reps[reps.size() - 1];
  const auto blockSizeNonK = mfmaInstrNonK * warpsPerBlockNonK;
  return blockSizeK % swizzlePatternSizeK == 0 &&
         blockSizeNonK % swizzlePatternSizeNonK == 0;
}

llvm::SmallVector<Value> computeOffsetsAType(
    ConversionPatternRewriter &rewriter, Location loc,
    computeTensorElemMappingInBlockT fn, const ArrayRef<int64_t> &elemsPerInstr,
    Value warpId, Value laneId, int warpsPerBlock, int numOfElems,
    ArrayRef<int64_t> reps, SharedMemoryObject smemObj,
    ArrayRef<Value> smemStrides, SwizzledSharedEncodingAttr srcLayout,
    unsigned nonKDim, unsigned kDim) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> offsets = smemObj.getOffsets();
  auto order = srcLayout.getOrder();
  auto rank = offsets.size();

  int vectorSize = 1;
  if (order[0] == rank - 1) {
    if (isSwizzled(srcLayout))
      vectorSize = std::min(static_cast<int>(srcLayout.getVec()), numOfElems);
    else
      vectorSize = numOfElems;
  }

  auto mapping = fn(rewriter, loc, elemsPerInstr, warpId, laneId, numOfElems,
                    reps, offsets, vectorSize, nonKDim, kDim);
  const auto numBlocks = reps[reps.size() - 2];
  const auto blockSize = mapping.size();
  llvm::SmallVector<Value> aOffsets(blockSize * numBlocks);

  if (!isSwizzlePatternFitsIntoBlock(srcLayout, 0, reps, elemsPerInstr,
                                     warpsPerBlock)) {
    for (int block = 0; block < numBlocks; ++block) {
      int blockNonKOffset = block * nonKDim * warpsPerBlock;
      for (int i = 0; i < blockSize; ++i) {
        Value row = b.add(mapping[i][0], b.i32_val(blockNonKOffset));
        Value col = mapping[i][1];
        aOffsets[block * blockSize + i] = computeOffset(
            rewriter, loc, row, col, smemObj, smemStrides, srcLayout);
      }
    }
  } else {
    // compute inblock offsets once and reuse them for all blocks
    llvm::SmallVector<Value> inblockOffset(mapping.size());
    for (int i = 0; i < mapping.size(); ++i) {
      Value row = mapping[i][0];
      Value col = mapping[i][1];
      inblockOffset[i] = computeOffset(rewriter, loc, row, col, smemObj,
                                       smemStrides, srcLayout);
    }
    for (int block = 0; block < numBlocks; ++block) {
      int blockNonKOffset = block * nonKDim * warpsPerBlock;
      Value offAdjust =
          b.mul(b.i32_val(blockNonKOffset), smemStrides[rank - 2]);
      for (int i = 0; i < blockSize; ++i)
        aOffsets[block * blockSize + i] = b.add(offAdjust, inblockOffset[i]);
    }
  }
  return aOffsets;
}

template <typename Container>
static SmallVector<typename Container::value_type>
transposeSpatialDims(const Container &vec) {
  auto rank = vec.size();
  assert(rank == 2 || rank == 3);
  SmallVector<typename Container::value_type> res(rank, vec[0]);
  res[rank - 2] = vec[rank - 1];
  res[rank - 1] = vec[rank - 2];
  return res;
}

llvm::SmallVector<Value> computeOffsetsBType(
    ConversionPatternRewriter &rewriter, Location loc,
    computeTensorElemMappingInBlockT fn, const ArrayRef<int64_t> &elemsPerInstr,
    Value warpId, Value laneId, int warpsPerBlock, int numOfElems,
    ArrayRef<int64_t> reps, SharedMemoryObject smemObj,
    ArrayRef<Value> smemStrides, SwizzledSharedEncodingAttr srcLayout,
    unsigned nonKDim, unsigned kDim) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // transpose reps and offsets, because operand B has layout equal to
  // transposed operand A layout
  // this unifies axis order, so non-K dim is 0, k dim is 1
  auto rank = smemObj.getOffsets().size();
  auto order = srcLayout.getOrder();
  SmallVector<int64_t> tElemsPerInstr{elemsPerInstr[1], elemsPerInstr[0]};
  SmallVector<int64_t> tReps = transposeSpatialDims(reps);
  SmallVector<Value> tOffsets = transposeSpatialDims(smemObj.getOffsets());
  SmallVector<Value> tStrides = transposeSpatialDims(smemStrides);

  int vectorSize = 1;
  if (order[0] == rank - 2) {
    if (isSwizzled(srcLayout))
      vectorSize = std::min(static_cast<int>(srcLayout.getVec()), numOfElems);
    else
      vectorSize = numOfElems;
  }

  auto mapping = fn(rewriter, loc, tElemsPerInstr, warpId, laneId, numOfElems,
                    tReps, tOffsets, vectorSize, nonKDim, kDim);
  const auto numBlocks = tReps[tReps.size() - 2];
  const auto blockSize = mapping.size();
  llvm::SmallVector<Value> bOffsets(blockSize * numBlocks);

  if (!isSwizzlePatternFitsIntoBlock(srcLayout, 0, reps, elemsPerInstr,
                                     warpsPerBlock)) {
    for (int block = 0; block < numBlocks; ++block) {
      int blockNonKOffset = block * nonKDim * warpsPerBlock;
      for (int i = 0; i < mapping.size(); ++i) {
        // swap row and col, because operand B layout is
        // a transposed operand A layout
        Value row = mapping[i][1];
        Value col = b.add(mapping[i][0], b.i32_val(blockNonKOffset));
        bOffsets[block * blockSize + i] = computeOffset(
            rewriter, loc, row, col, smemObj, smemStrides, srcLayout);
      }
    }
  } else {
    // compute inblock offsets once and reuse them for all blocks
    llvm::SmallVector<Value> inblockOffset(mapping.size());
    for (int i = 0; i < mapping.size(); ++i) {
      // swap row and col, because operand B layout is a transposed operand A
      // layout
      Value row = mapping[i][1];
      Value col = mapping[i][0];
      inblockOffset[i] = computeOffset(rewriter, loc, row, col, smemObj,
                                       smemStrides, srcLayout);
    }
    for (int block = 0; block < numBlocks; ++block) {
      int blockNonKOffset = block * nonKDim * warpsPerBlock;
      Value offAdjust = b.mul(b.i32_val(blockNonKOffset), tStrides[rank - 2]);
      for (int i = 0; i < mapping.size(); ++i)
        bOffsets[block * blockSize + i] = b.add(offAdjust, inblockOffset[i]);
    }
  }
  return bOffsets;
}

} // namespace mlir::triton::AMD
