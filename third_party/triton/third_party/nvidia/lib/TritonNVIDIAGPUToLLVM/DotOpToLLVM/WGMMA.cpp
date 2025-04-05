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

#include "MMAHelpers.h"
#include "Utility.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::NVIDIA;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::MemDescType;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::NVMMASharedEncodingAttr;

triton::nvgpu::WGMMAEltType getMmaRetType(Value d) {
  auto dTy = cast<RankedTensorType>(d.getType()).getElementType();
  if (dTy.isF32()) {
    return triton::nvgpu::WGMMAEltType::f32;
  } else if (dTy.isF16()) {
    return triton::nvgpu::WGMMAEltType::f16;
  } else if (dTy.isInteger(32)) {
    return triton::nvgpu::WGMMAEltType::s32;
  } else {
    llvm::report_fatal_error("Unsupported mma result type found");
  }
}

triton::nvgpu::WGMMAEltType getMmaOperandType(Value a, bool allowTF32) {
  auto aTy = cast<triton::gpu::TensorOrMemDesc>(a.getType()).getElementType();
  if (aTy.isF16()) {
    return triton::nvgpu::WGMMAEltType::f16;
  } else if (aTy.isBF16()) {
    return triton::nvgpu::WGMMAEltType::bf16;
  } else if (aTy.isF32() && allowTF32) {
    return triton::nvgpu::WGMMAEltType::tf32;
  } else if (aTy.isInteger(8)) {
    return triton::nvgpu::WGMMAEltType::s8;
  } else if (llvm::isa<Float8E5M2Type>(aTy)) {
    return triton::nvgpu::WGMMAEltType::e5m2;
  } else if (llvm::isa<Float8E4M3FNType>(aTy)) {
    return triton::nvgpu::WGMMAEltType::e4m3;
  } else {
    llvm::report_fatal_error("Unsupported mma operand type found");
  }
}

int64_t getSwizzlingFromLayout(const NVMMASharedEncodingAttr &layout,
                               uint32_t widthInByte) {
  uint32_t swizzlingByteWidth = layout.getSwizzlingByteWidth();
  // TODO[biaow]: remove it once we support swizzling size larger than matrix
  // width, which requires padding the matrix width to the swizzling size when
  // allocating shared memory.
  assert(swizzlingByteWidth <= widthInByte &&
         "swizzling size larger than matrix width is not supported.");
  return swizzlingByteWidth;
}

static Value createDescriptor(ConversionPatternRewriter &rewriter, Location loc,
                              int64_t swizzling, uint32_t stride) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  static_assert(sizeof(SMEMDescriptor) == 8,
                "Descriptor size should be 64 bits.");
  SMEMDescriptor desc;
  desc.descriptor = 0;
  switch (swizzling) {
  case 0:
    desc.swizzlingMode = 0;
    break;
  case 32:
    desc.swizzlingMode = 3;
    break;
  case 64:
    desc.swizzlingMode = 2;
    break;
  case 128:
    desc.swizzlingMode = 1;
    break;
  default:
    llvm::report_fatal_error("Unsupported swizzling size.");
  }
  desc.strideDimensionBaseOffset = swizzling >> 1;
  desc.leadDimensionBaseOffset = (swizzling * stride) >> 4;
  return b.int_val(64, desc.descriptor);
}

mlir::triton::NVIDIA::DotOpMmaV3SmemLoader::DotOpMmaV3SmemLoader(
    Value tensor, Value base, SmallVector<int64_t> shape,
    ArrayRef<int64_t> allocSwizzleShape, Value warpId, unsigned int dimWpt,
    bool trans, SmallVector<unsigned int> instrShape, int64_t elementBitwidth,
    ConversionPatternRewriter &rewriter, Location loc)
    : base(base), shape(shape), allocSwizzleShape(allocSwizzleShape),
      warpId(warpId), dimWpt(dimWpt), trans(trans), instrShape(instrShape),
      elemBits(elementBitwidth) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ty = cast<MemDescType>(tensor.getType());
  auto sharedLayout = cast<NVMMASharedEncodingAttr>(ty.getEncoding());
  fastMovingDim = sharedLayout.getTransposed() ? 0 : 1;
  const int swizzlingByteWidth = sharedLayout.getSwizzlingByteWidth();
  elemsPerSwizzlingRow = (swizzlingByteWidth * 8) / elemBits;
  elemsPerSwizzlingRowVal = b.i32_val(elemsPerSwizzlingRow);

  uint32_t widthInByte = allocSwizzleShape[fastMovingDim] * elemBits / 8;
  int64_t swizzling = getSwizzlingFromLayout(sharedLayout, widthInByte);

  descriptor =
      createDescriptor(rewriter, loc, swizzling, shape[1 - fastMovingDim]);
}

Value mlir::triton::NVIDIA::DotOpMmaV3SmemLoader::smemLoad(
    int a, int b, ConversionPatternRewriter &rewriter, Location loc) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  Value k = tb.i32_val(b * instrShape[1]);
  Value m = tb.add(tb.i32_val(a * dimWpt * instrShape[0]),
                   tb.mul(warpId, tb.i32_val(instrShape[0])));
  if (trans) {
    std::swap(k, m);
  }
  Value leading_offset =
      tb.mul(tb.udiv(k, elemsPerSwizzlingRowVal),
             tb.i32_val(shape[1 - fastMovingDim] * elemsPerSwizzlingRow));
  Value stride_offset = tb.mul(m, elemsPerSwizzlingRowVal);
  Value offset = tb.add(tb.add(leading_offset, stride_offset),
                        tb.urem(k, elemsPerSwizzlingRowVal));
  Value off1;
  // Avoid the runtime udiv if we know the elements are byte multiples
  if (elemBits % 8) {
    off1 = tb.udiv(tb.mul(tb.i32_val(elemBits), offset), tb.i32_val(8));
  } else {
    off1 = tb.mul(tb.i32_val(elemBits / 8), offset);
  }
  Value off_ = tb.zext(i64_ty, tb.udiv(off1, tb.i32_val(16)));

  Value loadDesc = tb.add(descriptor, off_);
  // Add the base at the end to make it easier to do loop invariant code
  // motion.
  loadDesc = tb.add(
      loadDesc, tb.lshr(tb.shl(tb.ptrtoint(i64_ty, base), tb.int_val(64, 46)),
                        tb.int_val(64, 50)));
  return loadDesc;
}

DotOpMmaV3SmemLoader loadA(const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Location loc,
                           const NvidiaMmaEncodingAttr &mmaEncoding,
                           Value tensor, Value smemObjBase, Value thread) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto aTy = cast<MemDescType>(tensor.getType());
  auto aSharedLayout = dyn_cast<NVMMASharedEncodingAttr>(aTy.getEncoding());
  assert(aSharedLayout && "only support load dot operand from shared.");
  auto instrShape = mmaEncoding.getInstrShape();
  auto wpt = mmaEncoding.getWarpsPerCTA();
  bool transA = aSharedLayout.getTransposed();
  auto shapePerCTA = getShapePerCTA(aTy);
  auto allocSwizzleShape = aTy.getAllocShape().take_back(shapePerCTA.size());

  // The descriptor should be calculated based on the first warp of the
  // warpgroup.
  Value warp =
      b.and_(rewriter.create<nvgpu::WarpIdOp>(loc), b.i32_val(0xFFFFFFFC));
  // Workaround for a bug in ptxas 12.3 that cause a failure in
  // test_core.py::test_dot. The shuffle will force the compiler to treat the
  // value as uniform and prevent wrong optimizations.
  warp = mlir::LLVM::NVIDIA::shuffleIdx(loc, rewriter, warp, 0);
  Value warpM = b.urem(warp, b.i32_val(wpt[0]));
  Value warpId = b.urem(warpM, b.i32_val(shapePerCTA[0] / instrShape[0]));

  return {tensor,
          smemObjBase,
          shapePerCTA,
          allocSwizzleShape,
          warpId,
          wpt[0],
          transA,
          {instrShape[0], instrShape[2]},
          aTy.getElementTypeBitWidth(),
          rewriter,
          loc};
}

DotOpMmaV3SmemLoader loadB(const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Location loc,
                           NvidiaMmaEncodingAttr &mmaEncoding, Value tensor,
                           Value base, Value thread) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto bTy = cast<MemDescType>(tensor.getType());
  auto bSharedLayout = cast<NVMMASharedEncodingAttr>(bTy.getEncoding());
  assert(bSharedLayout && "only support load B from shared.");
  auto instrShape = mmaEncoding.getInstrShape();
  auto wpt = mmaEncoding.getWarpsPerCTA();
  bool transB = !bSharedLayout.getTransposed();
  auto shapePerCTA = triton::gpu::getShapePerCTA(bTy);
  auto allocSwizzleShape = bTy.getAllocShape().take_back(shapePerCTA.size());

  Value warp =
      b.and_(rewriter.create<nvgpu::WarpIdOp>(loc), b.i32_val(0xFFFFFFFC));
  Value warpMN = b.udiv(warp, b.i32_val(wpt[0]));
  Value warpN = b.urem(warpMN, b.i32_val(wpt[1]));
  Value warpId = b.urem(warpN, b.i32_val(shapePerCTA[1] / instrShape[1]));

  return {tensor,
          base,
          shapePerCTA,
          allocSwizzleShape,
          warpId,
          wpt[1],
          transB,
          {instrShape[1], instrShape[2]},
          bTy.getElementTypeBitWidth(),
          rewriter,
          loc};
}

// Return a vector of Value of the accumulator start at startIndex and pack the
// values into 32bits in case the accumulator is fp16.
//
// `elements` contains all loaded register values for operand A.
// This consists of operand A for possibly multiple wgmma instructions.
// For each wgmma, each warp in a warp group feeds a single "warp matrix"
// Each warp matrix consists of 2x2 "quads".
// Each thread holds several elements in each quad. Right before a wgmma,
// the sum of bitwidth of
// the elements in each quad should add up to 32.
//
// These values are stored unrolled in `elements`.
// The ordering of dimensions is as follows:
// batch (only 1 batch for Hopper currently)
// matM (m-index of the "warp matrix")
// matK (k-index of the "warp matrix")
// quadK (k-index of the "quad" in the core matrix)
// quadM (m-index of the "quad" in the core matrix)
// vecIdx (index of the element in the quad; this is always along the k-dim)
//
// This ordering is decided when a tensor in DotOpEnc is lowered into llvm.
// For WGMMA this happens in both SharedToDotOperand and MMAToDotOperand.
// Thus, both lowerings must obey this above ordering for the below code to be
// correct.
llvm::SmallVector<Value> loadReg(ConversionPatternRewriter &rewriter,
                                 Location loc,
                                 const SmallVector<Value> &elements,
                                 int startIndex, int numElements,
                                 Operation *insertBefore) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(insertBefore);

  if (!elements[0].getType().isIntOrFloat() ||
      elements[0].getType().getIntOrFloatBitWidth() >= 32) {
    llvm::SmallVector<Value> mmaOut(numElements);
    for (int i = 0; i < numElements; ++i)
      mmaOut[i] = elements[startIndex + i];
    return mmaOut;
  }
  Type elementType = elements[0].getType();
  int numElemsPer32Bits = 32 / elementType.getIntOrFloatBitWidth();

  // For FP16 and BF16 we need to pack accumulator into 32-bit integers.
  int num32BitValues = numElements / numElemsPer32Bits;
  llvm::SmallVector<Value> mmaOut(num32BitValues);
  Type packTy = vec_ty(elementType, numElemsPer32Bits);
  for (int i = 0; i < num32BitValues; ++i) {
    Value pack = rewriter.create<LLVM::UndefOp>(loc, packTy);
    for (int j = 0; j < numElemsPer32Bits; ++j) {
      Value element = elements[startIndex + i * numElemsPer32Bits + j];
      pack = b.insert_element(packTy, pack, element, b.i32_val(j));
    }
    pack = b.bitcast(pack, rewriter.getIntegerType(32));
    mmaOut[i] = pack;
  }
  return mmaOut;
}

// If the accumulator is fp16 unpack it from 32-bit integers.
SmallVector<Value> unpackAccumulator(ConversionPatternRewriter &rewriter,
                                     Location loc,
                                     const SmallVector<Value> &packed,
                                     RankedTensorType tensorTy) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (!tensorTy.getElementType().isF16())
    return packed;
  // For fp16 the accumulator is pack into 32-bit integers so we need to unpack
  // it.
  SmallVector<Value> results;
  for (Value elem : packed) {
    elem = b.bitcast(elem, vec_ty(rewriter.getF16Type(), 2));
    results.push_back(
        b.extract_element(rewriter.getF16Type(), elem, b.i32_val(0)));
    results.push_back(
        b.extract_element(rewriter.getF16Type(), elem, b.i32_val(1)));
  }
  return results;
}

static Value faddAccumulate(ConversionPatternRewriter &rewriter, Location loc,
                            Value a, Value b) {
  int numEl = cast<LLVM::LLVMStructType>(a.getType()).getBody().size();
  Value newStruct = rewriter.create<LLVM::UndefOp>(loc, a.getType());
  for (int i = 0; i < numEl; ++i) {
    Value lhs = rewriter.create<LLVM::ExtractValueOp>(loc, a, i);
    Value rhs = rewriter.create<LLVM::ExtractValueOp>(loc, b, i);
    Value add = rewriter.create<LLVM::FAddOp>(loc, lhs, rhs);
    newStruct = rewriter.create<LLVM::InsertValueOp>(loc, newStruct, add, i);
  }
  return newStruct;
}

static SmallVector<Value> emitWait(ConversionPatternRewriter &rewriter,
                                   Location loc, SmallVector<Value> acc,
                                   int pendings) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Type> types(acc.size(), acc[0].getType());
  auto structTy =
      LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structTy);
  int i = 0;
  for (Value v : acc) {
    llvmStruct = b.insert_val(structTy, llvmStruct, v, i++);
  }
  Value res = rewriter.create<triton::nvgpu::WGMMAWaitGroupOp>(loc, llvmStruct,
                                                               pendings);
  SmallVector<Value> results;
  for (int i = 0; i < acc.size(); ++i) {
    results.push_back(b.extract_val(types[0], res, i));
  }
  return results;
}

LogicalResult convertDot(const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Operation *op, Value a, Value b, Value c, Value d,
                         Value useCOperand, Value loadedA, Value loadedB,
                         Value loadedC, bool allowTF32,
                         bool needsPartialAccumulator,
                         uint32_t maxNumImpreciseAcc, bool sync, Value thread) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  auto aTensorTy = cast<triton::gpu::TensorOrMemDesc>(a.getType());
  auto bTensorTy = cast<triton::gpu::TensorOrMemDesc>(b.getType());
  auto dTensorTy = cast<RankedTensorType>(d.getType());
  auto aSharedLayout =
      dyn_cast<NVMMASharedEncodingAttr>(aTensorTy.getEncoding());
  auto bSharedLayout = cast<NVMMASharedEncodingAttr>(bTensorTy.getEncoding());
  auto mmaEncoding = cast<NvidiaMmaEncodingAttr>(dTensorTy.getEncoding());
  bool transA = false;
  Value baseA;
  Value baseB;
  if (aSharedLayout)
    baseA =
        getSharedMemoryObjectFromStruct(
            loc, loadedA,
            typeConverter->convertType(aTensorTy.getElementType()), rewriter)
            .getBase();
  baseB = getSharedMemoryObjectFromStruct(
              loc, loadedB,
              typeConverter->convertType(bTensorTy.getElementType()), rewriter)
              .getBase();
  if (aSharedLayout) {
    transA = aSharedLayout.getTransposed();
  }
  bool transB = !bSharedLayout.getTransposed();
  auto dShapePerCTA = getShapePerCTA(dTensorTy);
  auto instrShape = mmaEncoding.getInstrShape();
  auto accSize = 2 * (instrShape[1] / 4);
  int M = 4 * instrShape[0];
  int N = instrShape[1];
  int K = instrShape[2];
  bool zeroAcc = isZeroConst(c);
  auto instrMNK = mmaEncoding.getInstrShape();
  auto warpSize = mmaEncoding.getWarpsPerCTA();
  auto shapePerCTATile = SmallVector<unsigned>{instrMNK[0] * warpSize[0],
                                               instrMNK[1] * warpSize[1]};
  int numRepM = ceil<unsigned>(dShapePerCTA[0], shapePerCTATile[0]);
  int numRepN = ceil<unsigned>(dShapePerCTA[1], shapePerCTATile[1]);
  int numRepK = ceil<unsigned>(aTensorTy.getShape()[1], instrShape[2]);
  DotOpMmaV3SmemLoader aLoader;
  SmallVector<Value> structA;
  if (aSharedLayout) {
    aLoader =
        loadA(typeConverter, rewriter, loc, mmaEncoding, a, baseA, thread);
  } else {
    structA = unpackLLElements(loc, loadedA, rewriter);
  }
  DotOpMmaV3SmemLoader bLoader =
      loadB(typeConverter, rewriter, loc, mmaEncoding, b, baseB, thread);

  auto fc = unpackLLElements(loc, loadedC, rewriter);

  triton::nvgpu::WGMMAEltType eltTypeC = getMmaRetType(d);
  triton::nvgpu::WGMMAEltType eltTypeA = getMmaOperandType(a, allowTF32);
  triton::nvgpu::WGMMAEltType eltTypeB = getMmaOperandType(b, allowTF32);

  triton::nvgpu::WGMMALayout layoutA = transA ? triton::nvgpu::WGMMALayout::col
                                              : triton::nvgpu::WGMMALayout::row;
  triton::nvgpu::WGMMALayout layoutB = transB ? triton::nvgpu::WGMMALayout::row
                                              : triton::nvgpu::WGMMALayout::col;

  auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
  Operation *startSequence = rewriter.create<triton::nvgpu::WGMMAFenceOp>(loc);
  SmallVector<Value> mmaResults;
  for (int m = 0; m < numRepM; ++m) {
    for (int n = 0; n < numRepN; ++n) {
      llvm::SmallVector<Value> mmaOut =
          loadReg(rewriter, loc, fc, (m * numRepN + n) * accSize, accSize,
                  startSequence);
      llvm::SmallVector<Type> elemTypes;
      for (Value accEl : mmaOut)
        elemTypes.push_back(accEl.getType());
      auto accTy =
          LLVM::LLVMStructType::getLiteral(rewriter.getContext(), elemTypes);
      Value d;
      Value useC = tb.i1_val(0);
      if (!zeroAcc) {
        d = packLLElements(loc, typeConverter, mmaOut, rewriter, accTy);
        useC = tb.i1_val(1);
      }
      if (useCOperand)
        useC = tb.and_(useC, useCOperand);
      uint32_t numLowPrecisionAcc = 0;
      Value partialAcc;
      for (int k = 0; k < numRepK; ++k) {
        Value a;
        if (aSharedLayout) {
          a = aLoader.smemLoad(m, k, rewriter, loc);
        } else {
          auto aDotOpEnc =
              cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
          assert(aDotOpEnc.getKWidth() ==
                 32 / aTensorTy.getElementTypeBitWidth());

          unsigned regASize = (instrShape[0] * instrShape[2]) / 32;
          llvm::SmallVector<Value> regA =
              loadReg(rewriter, loc, structA, (m * numRepK + k) * regASize,
                      regASize, startSequence);
          auto regATy = LLVM::LLVMStructType::getLiteral(
              rewriter.getContext(),
              SmallVector<Type>(regA.size(), regA[0].getType()));
          a = packLLElements(loc, typeConverter, regA, rewriter, regATy);
        }
        auto b = bLoader.smemLoad(n, k, rewriter, loc);
        numLowPrecisionAcc += K;
        // If using native accumulation would cause use to do more low precion
        // accumulation than allowed do a separate allocation.
        bool requireAddAccumulator =
            needsPartialAccumulator &&
            (numLowPrecisionAcc >= maxNumImpreciseAcc || k == numRepK - 1);
        Value mmaAcc = needsPartialAccumulator ? partialAcc : d;
        mmaAcc = rewriter.create<triton::nvgpu::WGMMAOp>(
            loc, accTy, a, b, useC, mmaAcc, M, N, K, eltTypeC, eltTypeA,
            eltTypeB, layoutA, layoutB);
        useC = tb.i1_val(1);
        if (needsPartialAccumulator)
          partialAcc = mmaAcc;
        else
          d = mmaAcc;
        // If we need accumulate separately to have higher precision, insert
        // adds.
        if (requireAddAccumulator) {
          d = d ? faddAccumulate(rewriter, loc, d, partialAcc) : partialAcc;
          numLowPrecisionAcc = 0;
          partialAcc = Value();
        }
      }
      auto acc = unpackLLElements(loc, d, rewriter);
      for (int i = 0; i < acc.size(); ++i) {
        mmaResults.push_back(acc[i]);
      }
    }
  }
  rewriter.create<triton::nvgpu::WGMMACommitGroupOp>(loc);

  if (sync)
    mmaResults = emitWait(rewriter, loc, mmaResults, 0);

  SmallVector<Value> results =
      unpackAccumulator(rewriter, loc, mmaResults, dTensorTy);

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      mmaEncoding.getContext(),
      SmallVector<Type>(results.size(), dTensorTy.getElementType()));
  auto res = packLLElements(loc, typeConverter, results, rewriter, structTy);
  rewriter.replaceOp(op, res);
  return success();
}

LogicalResult convertWGMMA(triton::nvidia_gpu::WarpGroupDotOp op,
                           triton::nvidia_gpu::WarpGroupDotOp::Adaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Value thread) {
  auto AEnc = op.getA().getType().getEncoding();
  auto BEnc = op.getB().getType().getEncoding();
  assert(mlir::isa<NVMMASharedEncodingAttr>(AEnc) ||
         mlir::isa<DotOperandEncodingAttr>(AEnc));
  assert(mlir::isa<NVMMASharedEncodingAttr>(BEnc) &&
         "Operand B should use Shared layout.");
  return convertDot(typeConverter, rewriter, op.getLoc(), op.getOperation(),  //
                    op.getA(), op.getB(), op.getC(), op.getD(), op.getUseC(), //
                    adaptor.getA(), adaptor.getB(), adaptor.getC(),           //
                    op.getInputPrecision() == InputPrecision::TF32,
                    op.needsPartialAccumulator(), op.getMaxNumImpreciseAcc(),
                    !op.getIsAsync(), thread);
}
