#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;

namespace {

/// OperandValueKey structure represents compile time part
/// of spatial coordinates of a value in a tensor.
///
/// Every Value spatial coordinates(i.e. [batch;nonK;k]) in tensor can be
/// defined as:
///
/// batch = (bRepIdx * CTABSize + bIdx) + (laneBCoord + warpBCoord)
/// nonK = (nonKRepIdx * CTANKSize + nonKIdx) + (laneNonKCoord + warpNonKCoord)
/// k = kIdx
///
/// Where:
/// CTABSize, CTANKSize: constants;
/// laneBCoord, warpBCoord, laneNonKCoord, warpNonKCoord: runtime components;
/// bRepIdx, nonKRepIdx, bIdx, nonKIdx, kIdx: compile time components.
struct OperandValueKey {
  unsigned bRepIdx, nonKRepIdx;
  unsigned bIdx, nonKIdx, kIdx;

  bool operator==(const OperandValueKey &other) const {
    return (bRepIdx == other.bRepIdx && nonKRepIdx == other.nonKRepIdx &&
            bIdx == other.bIdx && nonKIdx == other.nonKIdx &&
            kIdx == other.kIdx);
  }
};

} // namespace

template <> struct std::hash<OperandValueKey> {
  std::size_t operator()(const OperandValueKey &k) const {
    return llvm::hash_combine(k.bRepIdx, k.nonKRepIdx, k.bIdx, k.nonKIdx,
                              k.kIdx);
  }
};

namespace {

using ValueTableFMA = std::unordered_map<OperandValueKey, Value>;

ValueTableFMA getValueTableFromStructFMA(
    Value val, ArrayRef<unsigned> perRepShape, ArrayRef<unsigned> repetitions,
    unsigned kDim, unsigned nonKDim, ConversionPatternRewriter &rewriter,
    Location loc, ArrayRef<unsigned> inRepOrder, ArrayRef<unsigned> repOrder) {
  ValueTableFMA res;
  auto elems = unpackLLElements(loc, val, rewriter);
  assert(perRepShape.size() == 3);
  auto numElemsRep = product(perRepShape);
  assert(elems.size() == numElemsRep * product(repetitions));
  assert(kDim == 1 || kDim == 2);
  assert(nonKDim == 1 || nonKDim == 2);
  const unsigned bDim = 0;

  for (unsigned idx = 0; idx < elems.size(); ++idx) {
    auto inRepLinearIdx = idx % numElemsRep;
    auto repLinearIdx = idx / numElemsRep;
    auto inRepSpatialIdx =
        mlir::LLVM::delinearize(inRepLinearIdx, perRepShape, inRepOrder);
    auto repSpatialIdx =
        mlir::LLVM::delinearize(repLinearIdx, repetitions, repOrder);
    OperandValueKey key{repSpatialIdx[0], repSpatialIdx[nonKDim],
                        inRepSpatialIdx[0], inRepSpatialIdx[nonKDim],
                        inRepSpatialIdx[kDim]};
    res[key] = elems[idx];
  }
  return res;
}

} // namespace

namespace mlir::triton::gpu {

LogicalResult parametricConvertFMADot(DotOp op, DotOp::Adaptor adaptor,
                                      const LLVMTypeConverter *typeConverter,
                                      ConversionPatternRewriter &rewriter,
                                      FMAVectorMultiplier &multiplier) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
  auto D = op.getResult();

  auto aTensorTy = cast<RankedTensorType>(A.getType());
  auto dTensorTy = cast<RankedTensorType>(D.getType());

  SmallVector<int64_t> aShapePerCTA =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTA(aTensorTy)));
  auto dShapePerCTA =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTA(dTensorTy)));

  BlockedEncodingAttr dLayout =
      cast<BlockedEncodingAttr>(dTensorTy.getEncoding());
  // TODO process A and B operand separately
  auto inRepOrder = expandMatrixOrderWithBatch(dLayout.getOrder());
  auto repOrder = expandMatrixOrderWithBatch(dLayout.getRepOrder());
  auto cc = unpackLLElements(loc, adaptor.getC(), rewriter);

  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  auto sizePerThread = getContigPerThread(dTensorTy);
  auto numElemsPerThread = product(sizePerThread);
  SmallVector<unsigned> shapePerCTATile;
  for (auto [reg, thread, warp] :
       llvm::zip(sizePerThread, dLayout.getThreadsPerWarp(),
                 dLayout.getWarpsPerCTA())) {
    shapePerCTATile.push_back(reg * thread * warp);
  }
  shapePerCTATile = expandMatrixShapeWithBatch(ArrayRef(shapePerCTATile));
  sizePerThread = expandMatrixShapeWithBatch(ArrayRef(sizePerThread));

  unsigned K = aShapePerCTA[2];

  unsigned threadTileShape[3];
  unsigned repetitions[3];
  for (int i = 0; i < 3; ++i) {
    repetitions[i] =
        ceil(dShapePerCTA[i], static_cast<int64_t>(shapePerCTATile[i]));
  }

  auto has = getValueTableFromStructFMA(
      llA, {sizePerThread[0], sizePerThread[1], K},
      {repetitions[0], repetitions[1], 1},
      /*kDim*/ 2, /*nonKDim*/ 1, rewriter, loc, inRepOrder, repOrder);
  auto hbs = getValueTableFromStructFMA(
      llB, {sizePerThread[0], K, sizePerThread[2]},
      {repetitions[0], 1, repetitions[2]},
      /*kDim*/ 1, /*nonKDim*/ 2, rewriter, loc, inRepOrder, repOrder);

  SmallVector<Value> acc = cc;

  for (unsigned bRep = 0; bRep < repetitions[0]; ++bRep)
    for (unsigned mRep = 0; mRep < repetitions[1]; ++mRep)
      for (unsigned nRep = 0; nRep < repetitions[2]; ++nRep)
        for (unsigned b = 0; b < sizePerThread[0]; ++b)
          for (unsigned m = 0; m < sizePerThread[1]; ++m)
            for (unsigned n = 0; n < sizePerThread[2]; ++n) {
              SmallVector<unsigned> multiDimAccumIdx = {b, m, n};
              unsigned linearInRepIdx =
                  LLVM::linearize(multiDimAccumIdx, sizePerThread, inRepOrder);
              SmallVector<unsigned> multiDimRepIdx = {bRep, mRep, nRep};
              unsigned linearRepIdx =
                  LLVM::linearize(multiDimRepIdx, repetitions, repOrder);
              unsigned linearAccumIdx =
                  linearInRepIdx + linearRepIdx * numElemsPerThread;

              SmallVector<Value> aOpVector;
              SmallVector<Value> bOpVector;

              for (unsigned k = 0; k < K; ++k) {
                aOpVector.push_back(has.at({bRep, mRep, b, m, k}));
                bOpVector.push_back(hbs.at({bRep, nRep, b, n, k}));
              }

              acc[linearAccumIdx] = multiplier.multiplyVectors(
                  aOpVector, bOpVector, acc[linearAccumIdx]);
            }

  auto res = packLLElements(loc, typeConverter, acc, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);

  return success();
}

} // namespace mlir::triton::gpu
