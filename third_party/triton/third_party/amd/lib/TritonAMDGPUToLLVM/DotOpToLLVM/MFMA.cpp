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
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

using ::mlir::LLVM::AMD::scaleDotElemTypeToMLIRType;
using ::mlir::LLVM::AMD::shuffleXor;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::LinearEncodingAttr;

using ValueTable = std::map<std::array<int, 3>, Value>;

/// Get matrix format flag passed through BLGP/CBSZ args in V_MFMA_*_F8F6F4
/// instructions.
///
/// Values:
/// - 0: E4M3(FP8)
/// - 1: E5M2(BF8)
/// - 2: E2M3(FP6)
/// - 3: E3M2(BF6)
/// - 4: E2M1(FP4)
static inline int32_t getMfmaF8F6F4MatrixFormat(Type t) {
  return llvm::TypeSwitch<Type, int32_t>(t)
      .Case<Float8E4M3FNType>([](Type) { return 0; })
      .Case<Float8E5M2Type>([](Type) { return 1; })
      .Case<Float6E3M2FNType>([](Type) { return 2; })
      .Case<Float6E2M3FNType>([](Type) { return 3; })
      .Case<Float4E2M1FNType>([](Type) { return 4; })
      .Default([](Type) { return -1; });
}

struct DotOpMFMAConversionHelper {
  AMDMfmaEncodingAttr mfmaLayout;

  ConversionPatternRewriter &rewriter;
  const LLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  virtual ~DotOpMFMAConversionHelper() = default;

  explicit DotOpMFMAConversionHelper(AMDMfmaEncodingAttr mfmaLayout,
                                     ConversionPatternRewriter &rewriter,
                                     const LLVMTypeConverter *typeConverter,
                                     Location loc)
      : mfmaLayout(mfmaLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(mfmaLayout.getContext()) {}

  Value generateMFMAOp(StringRef intrinsicName, Value valA, Value valB,
                       Value valC) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto resType = valC.getType();
    Value zeroFlag = b.i32_val(0);
    OperationState loweredOp(loc, intrinsicName);
    loweredOp.addTypes(resType);
    loweredOp.addOperands({valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    return rewriter.create(loweredOp)->getResult(0);
  }

  int getNumSubmatrices(Type elementType, int mDim, int nDim) const {
    if ((mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64))
      return 1;
    assert(mDim == nDim);
    switch (mDim) {
    case 32:
    case 16:
      return 1;
      break;
    case 4:
      assert(elementType.getIntOrFloatBitWidth() <= 32 &&
             "fp64 is not supported yet");
      assert(elementType.getIntOrFloatBitWidth() != 8 ||
             elementType.isInteger(8) && "fp8 is not supported yet");
      return 16;
      break;
    default:
      llvm::report_fatal_error("unsupported nonKDim in MFMA dot");
    }
    return -1;
  }

  Value processSubBlocks(int numSubBlocks, Value acc, bool reduceSubBlocks,
                         bool zeroSubBlocks) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    assert((numSubBlocks & (numSubBlocks - 1)) == 0 &&
           "numSubBlocks in not pow 2!");
    if (numSubBlocks == 1)
      return acc;
    constexpr int warpSize = 64;
    int subBlockSize = warpSize / numSubBlocks;
    Value laneId = getLaneId(rewriter, loc);
    auto vecTy = dyn_cast<VectorType>(acc.getType());
    auto elemType = vecTy.getElementType();
    assert(elemType.getIntOrFloatBitWidth() == 32);
    int numScalars = vecTy.getNumElements();
    std::vector<Value> accScalar(numScalars);
    for (int i = 0; i < numScalars; ++i)
      accScalar[i] = b.extract_element(elemType, acc, b.i32_val(i));

    if (reduceSubBlocks) {
      while (subBlockSize < warpSize) {
        for (int i = 0; i < numScalars; ++i) {
          Value other_acc =
              shuffleXor(loc, rewriter, accScalar[i], subBlockSize);
          if (elemType.isInteger(32))
            accScalar[i] = b.add(accScalar[i], other_acc);
          else
            accScalar[i] = b.fadd(accScalar[i], other_acc);
        }
        subBlockSize *= 2;
      }
    }
    if (zeroSubBlocks) {
      Value zero;
      if (elemType.isInteger(32))
        zero = b.i32_val(0);
      else
        zero = b.f32_val(0.0);
      auto cond = b.icmp_ult(laneId, b.i32_val(subBlockSize));
      for (int i = 0; i < numScalars; ++i)
        accScalar[i] = b.select(cond, accScalar[i], zero);
    }

    Value reducedAcc = b.undef(vecTy);
    for (int i = 0; i < numScalars; ++i)
      reducedAcc =
          b.insert_element(vecTy, reducedAcc, accScalar[i], b.i32_val(i));
    return reducedAcc;
  }

  /// @brief MFMA 4x4 is computes 16 matrix multiplications, this functions adds
  /// these 16 matrices to get final 4x4 matrix
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value reduceSubBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, true, false);
  }

  /// @brief Zeroes out redundant values in all sub-blocks except first one
  ///
  /// Every warp in mfma 4x4 layout holds only 4 unique values(scalar or
  /// vectors) in blocks of 4 consecutive threads, There are 16 copies of these
  /// 4 values across all threads of the warp. Need to zero out 15 copies to use
  /// accumulator between dot operations.
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value zeroAuxiliarBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, false, true);
  }

  /// Dot operand layout minimal tile is kDimInstrSize elements across
  /// K dimension. If dot operand K dimension is smaller, layout
  /// assigns tensor elements to multiple different hardware locations.
  /// In this case mfma instruction adds elements in accumulator
  /// multiple times.
  ///
  /// Let say A=[1,2]; B=[3,4], C = A*B = 1*3+2*4 = 11
  /// Consider instruction K size is 4,
  /// in this case operands will be duplicated:
  /// A' = [1,2,1,2] B' = [3,4,3,4]
  /// C' = (1*3+2*4) + (1*3+2*4) = 22
  ///
  /// Following code adjusts accumulator values in such cases.
  /// If accumulator is integer, shift accumulator right by
  /// log2(duplicationRate). If accumulator is float, multiply accum
  /// with 1/duplicationRate constant.
  void adjustAccForSmallKDim(SmallVector<Value> &fc, Value &acc, Type dstElemTy,
                             int b, int m, int n, int64_t numRepM,
                             int64_t numRepN, int64_t kDimInstrSize,
                             int64_t kDimOperandSize,
                             unsigned elemsPerVec) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    for (unsigned v = 0; v < elemsPerVec; ++v) {
      Value accElem = tb.extract_element(dstElemTy, acc, tb.i32_val(v));
      if (kDimInstrSize > kDimOperandSize) {
        assert(kDimInstrSize % kDimOperandSize == 0);
        int duplicationRate = kDimInstrSize / kDimOperandSize;
        assert(llvm::isPowerOf2_32(duplicationRate));
        if (dstElemTy.isInteger()) {
          auto shiftSize = llvm::Log2_32(duplicationRate);
          assert(!accElem.getType().isUnsignedInteger() &&
                 "MFMA uses signed accumulator");
          accElem = tb.ashr(accElem, tb.i32_val(shiftSize));
        } else {
          auto multiplierAttr =
              rewriter.getFloatAttr(dstElemTy, 1.0 / duplicationRate);
          auto multiplierVal =
              rewriter.create<LLVM::ConstantOp>(loc, dstElemTy, multiplierAttr);
          accElem = tb.fmul(accElem, multiplierVal);
        }
      }
      auto linearIdx = b * numRepM * numRepN * elemsPerVec +
                       m * numRepN * elemsPerVec + n * elemsPerVec + v;
      fc[linearIdx] = accElem;
    }
  }

  template <typename T>
  void packAndReplaceResult(T &op, SmallVector<Value> &fc,
                            const FailureOr<MfmaIntrinsic> &maybeMfmaIntrinsic,
                            Type dstElemTy, Type elemtTy,
                            size_t mmaCount) const {
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(fc.size(), dstElemTy));
    Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);

    setNumGeneratedMMAs(op, mmaCount, maybeMfmaIntrinsic->mDim,
                        maybeMfmaIntrinsic->nDim, maybeMfmaIntrinsic->kDim,
                        elemtTy);

    rewriter.replaceOp(op, res);
  }

  // Conduct the Dot conversion.
  LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    // Check if this dot has come with priority set by setprio.
    auto setPrioOp = dyn_cast_or_null<ROCDL::SetPrioOp>(op->getPrevNode());

    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto mDim = mfmaLayout.getMDim();
    auto nDim = mfmaLayout.getNDim();
    auto mfmaVersion = mfmaLayout.getVersionMajor();
    assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
           (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));

    Value a = op.getA();
    Value b = op.getB();
    Value d = op.getD();
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto dTensorTy = cast<RankedTensorType>(d.getType());
    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();

    const auto kDimOperandSize = aTensorTy.getShape().back();

    bool allowXF32 =
        op.getInputPrecision() == InputPrecision::TF32 && mfmaVersion == 3;
    StringRef intrinsicName;
    FailureOr<MfmaIntrinsic> maybeMfmaIntrinsic = MfmaIntrinsic::selectFor(
        mfmaVersion, mDim, nDim, kDimOperandSize, elemTyA, elemTyB,
        /*withScale=*/false, allowXF32);
    if (failed(maybeMfmaIntrinsic))
      llvm::report_fatal_error("No match found in MFMA database\n");

    intrinsicName = maybeMfmaIntrinsic->name;
    unsigned kBase = maybeMfmaIntrinsic->kBase;

    auto aEncoding = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    auto bEncoding = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
    int kWidth = aEncoding.getKWidth();

    // If we are using XF32, the kWidth (and kBase) is double that of F32.
    if (aTensorTy.getElementType().isF32() && allowXF32)
      kWidth *= 2;

    const auto kDimInstrSize = mfmaLayout.getInstrShapeForOperand(kWidth, 0)[1];

    auto repA = mfmaLayout.getRepForOperand(aTensorTy.getShape(), kWidth, 0);
    auto repB = mfmaLayout.getRepForOperand(bTensorTy.getShape(), kWidth, 1);

    assert(repA[2] == repB[1]);

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();

    auto numRepM = repA[1];
    auto numRepN = repB[2];
    auto numRepK = repA[2];
    auto numRepB = repA[0];
    assert(repA[0] == repB[0]);

    bool preserveBF16 = intrinsicName.contains(".bf16") && mfmaVersion >= 4;
    auto operandA = getValuesFromDotOperandLayoutStruct(
        loadedA, numRepB, numRepM, numRepK, kWidth, kBase,
        aTensorTy.getElementType(), allowXF32, preserveBF16);
    auto operandB = getValuesFromDotOperandLayoutStruct(
        loadedB, numRepB, numRepN, numRepK, kWidth, kBase,
        aTensorTy.getElementType(), allowXF32, preserveBF16);

    auto dstElemTy = dTensorTy.getElementType();
    auto fc = unpackLLElements(loc, loadedC, rewriter);

    unsigned warpSize = triton::gpu::lookupThreadsPerWarp(rewriter);
    // compute number of output elements that each thread holds for one MFMA
    // instruction.
    const int subBlocks =
        getNumSubmatrices(aTensorTy.getElementType(), mDim, nDim);
    auto elemsPerVec = mDim * nDim * subBlocks / warpSize;

    Value firstMfma;
    auto vecTy = vec_ty(dstElemTy, elemsPerVec);
    for (int b = 0; b < numRepB; ++b) {
      for (int m = 0; m < numRepM; ++m) {
        for (int n = 0; n < numRepN; ++n) {
          Value acc = tb.undef(vecTy);
          for (unsigned v = 0; v < elemsPerVec; ++v) {
            acc = tb.insert_element(
                vecTy, acc,
                fc[b * numRepM * numRepN * elemsPerVec +
                   m * numRepN * elemsPerVec + n * elemsPerVec + v],
                tb.i32_val(v));
          }
          acc = zeroAuxiliarBlocks(subBlocks, acc);
          for (int k = 0; k < numRepK; k++) {
            for (int kPack = 0; kPack < kWidth / kBase; ++kPack) {
              acc = mfmaLayout.getIsTransposed()
                        ? generateMFMAOp(intrinsicName,
                                         operandB[kPack][{b, n, k}],
                                         operandA[kPack][{b, m, k}], acc)
                        : generateMFMAOp(intrinsicName,
                                         operandA[kPack][{b, m, k}],
                                         operandB[kPack][{b, n, k}], acc);
              if (!firstMfma)
                firstMfma = acc;
            }
          }
          acc = reduceSubBlocks(subBlocks, acc);
          adjustAccForSmallKDim(fc, acc, dstElemTy, b, m, n, numRepM, numRepN,
                                kDimInstrSize, kDimOperandSize, elemsPerVec);
        }
      }
    }

    // Originally, setprio (high) is set to the high-level dot op. After dot is
    // being lowered to the series of mfma operations, it should be moved next
    // to the first mfma leaving the first mfma staying at the low priority. In
    // this way, incoming warp can be effectively waiting on the first mfma
    // instruction (low priority) while the other warp is executing mfma with
    // high priority. Otherwise, incoming warp can break the cluster.
    if (setPrioOp && firstMfma)
      setPrioOp->moveAfter(firstMfma.getDefiningOp());

    const size_t mmaCount =
        numRepB * numRepM * numRepN * numRepK * kWidth / kBase;
    packAndReplaceResult(op, fc, maybeMfmaIntrinsic, dstElemTy, elemTyA,
                         mmaCount);

    return success();
  }

  /// Extract vector from rawElems based on kWidth and kBase
  /// rawElems is a vector of kWidth elements. We need to prepare vector(s) of
  /// kBase elements for each mfma instruction
  SmallVector<Value> extractOperands(Value rawElems, int kWidth, int kBase,
                                     Type type, bool preserveBF16,
                                     bool isConstantScale = false) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int kpack = kWidth / kBase;
    SmallVector<Value> results;
    auto vecTy = vec_ty(type, kBase);
    if (type.isBF16() && !preserveBF16)
      vecTy = vec_ty(i16_ty, kBase);
    for (int k = 0; k < kpack; ++k) {
      Value vec = b.undef(vecTy);
      for (int elemId = 0; elemId < kBase; ++elemId) {
        auto val =
            b.extract_element(type, rawElems, b.i32_val(elemId + k * kBase));
        if (type.isBF16() && !preserveBF16) {
          // rocdl.mfma.f32.32x32x8bf16.1k calls for input of i16 type
          auto cast = b.bitcast(val, i16_ty);
          vec = b.insert_element(vecTy, vec, cast, b.i32_val(elemId));
        } else {
          vec = b.insert_element(vecTy, vec, val, b.i32_val(elemId));
        }
      }
      if (type.getIntOrFloatBitWidth() == 8) {
        if (1 == kBase) {
          // This is only for the scale operands of scaled mfma on CDNA4
          if (isConstantScale) {
            // If the scale is constant(created by arith::ConstantOp), it will
            // be put in a sgpr instead of vgpr. In that case, instead of
            // vgpr[7:0], the instruction reads sgpr[30:23] as the scale value.
            // So we need to manually left shift the scale by 23 bits to meet
            // the requirement.
            results.push_back(b.shl(
                i32_ty, b.zext(i32_ty, b.bitcast(vec, i8_ty)), b.i32_val(23)));
          } else {
            results.push_back(b.zext(i32_ty, b.bitcast(vec, i8_ty)));
          }
        }
        if (4 == kBase)
          // This is for int8 on pre- CDNA3 GPUs
          results.push_back(b.bitcast(vec, i32_ty));
        if (8 == kBase)
          results.push_back(b.bitcast(vec, i64_ty));
        if (16 == kBase)
          // This is only for the operands of scaled mfma on CDNA4
          results.push_back(b.bitcast(vec, vec_ty(i32_ty, 4)));
        if (32 == kBase)
          results.push_back(b.bitcast(vec, vec_ty(i32_ty, 8)));
      } else {
        results.push_back(vec);
      }
    }
    return results;
  }

  /// Converts dot operand structure to value table and converts types
  /// appropriate for mfma instructions
  virtual SmallVector<ValueTable> getValuesFromDotOperandLayoutStruct(
      Value value, int batch, int n0, int n1, int kWidth, int kBase, Type type,
      bool allowXF32, bool preserveBF16, bool isConstantScale = false) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    auto elems = unpackLLElements(loc, value, rewriter);
    int kpack = kWidth / kBase;
    SmallVector<ValueTable> dotOpVals(kpack);
    for (int b = 0; b < batch; ++b) {
      for (int i = 0; i < n0; i++) {
        for (int j = 0; j < n1; j++) {
          Type elemTy = typeConverter->convertType(type);
          Type ty = vec_ty(elemTy, kWidth);
          Value rawElems = tb.undef(ty);
          for (int k = 0; k < kWidth; ++k) {
            rawElems = tb.insert_element(
                ty, rawElems,
                elems[kWidth * n1 * n0 * b + kWidth * n1 * i + kWidth * j + k],
                tb.i32_val(k));
          }

          Value convertedElems;
          if (type.isF32() && !allowXF32) {
            for (int k = 0; k < kpack; ++k)
              dotOpVals[k][{b, i, j}] =
                  tb.extract_element(type, rawElems, tb.i32_val(k));
          } else {
            SmallVector<Value> vals;
            if (type.isF32() && allowXF32) {
              vals = extractOperands(rawElems, kWidth, kBase, f32_ty,
                                     preserveBF16);
            } else if (type.getIntOrFloatBitWidth() == 8) {
              vals = extractOperands(rawElems, kWidth, kBase, i8_ty,
                                     preserveBF16, isConstantScale);
            } else if (type.isBF16()) {
              vals = extractOperands(rawElems, kWidth, kBase, bf16_ty,
                                     preserveBF16);
            } else {
              assert(type.isF16() && "Unsupported data type");
              vals = extractOperands(rawElems, kWidth, kBase, f16_ty,
                                     preserveBF16);
            }
            for (int k = 0; k < kpack; ++k) {
              dotOpVals[k][{b, i, j}] = vals[k];
            }
          }
        }
      }
    }
    return dotOpVals;
  }
};

struct ScaledDotOpMFMAConversionHelper : DotOpMFMAConversionHelper {
  virtual ~ScaledDotOpMFMAConversionHelper() = default;

  ScaledDotOpMFMAConversionHelper(AMDMfmaEncodingAttr mfmaLayout,
                                  ConversionPatternRewriter &rewriter,
                                  const LLVMTypeConverter *typeConverter,
                                  Location loc)
      : DotOpMFMAConversionHelper(mfmaLayout, rewriter, typeConverter, loc) {}

  Value generateScaledMFMAOp(StringRef intrinsicName, Value valA, Value valB,
                             Value valC, Type elemTypeA, Type elemTypeB) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto resType = valC.getType();
    Value zeroFlag = b.i32_val(0);
    OperationState loweredOp(loc, intrinsicName);
    int32_t cbsz = getMfmaF8F6F4MatrixFormat(elemTypeA);
    int32_t blgp = getMfmaF8F6F4MatrixFormat(elemTypeB);
    assert((cbsz != -1) && (blgp != -1));
    loweredOp.addTypes(resType);
    // If both scales are constant 0, the LLVM backend will use V_MFMA_*_F8F6F4
    // instructions instead of V_MFMA_SCALE_*_F8F6F4 to reduce memory access.
    loweredOp.addOperands({valA, valB, valC, b.i32_val(cbsz), b.i32_val(blgp),
                           zeroFlag, zeroFlag, zeroFlag, zeroFlag});
    return rewriter.create(loweredOp)->getResult(0);
  }

  Value generateScaledMFMAOp(StringRef intrinsicName, Value valA, Value valB,
                             Value valC, Value valScaleA, Value valScaleB,
                             Type elemTypeA, Type elemTypeB) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto resType = valC.getType();
    Value zeroFlag = b.i32_val(0);
    OperationState loweredOp(loc, intrinsicName);
    int32_t cbsz = getMfmaF8F6F4MatrixFormat(elemTypeA);
    int32_t blgp = getMfmaF8F6F4MatrixFormat(elemTypeB);
    assert((cbsz != -1) && (blgp != -1));
    loweredOp.addTypes(resType);
    loweredOp.addOperands({valA, valB, valC, b.i32_val(cbsz), b.i32_val(blgp),
                           zeroFlag, valScaleA, zeroFlag, valScaleB});
    return rewriter.create(loweredOp)->getResult(0);
  }

  LogicalResult convertScaledDot(DotScaledOp op,
                                 DotScaledOpAdaptor adaptor) const {
    // Check if this dot has come with priority set by setprio.
    auto setPrioOp = dyn_cast_or_null<ROCDL::SetPrioOp>(op->getPrevNode());

    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto mDim = mfmaLayout.getMDim();
    auto nDim = mfmaLayout.getNDim();
    auto mfmaVersion = mfmaLayout.getVersionMajor();
    assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
           (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));

    Value a = op.getA();
    Value b = op.getB();
    Value aScale = op.getAScale();
    Value bScale = op.getBScale();
    if ((aScale && !bScale) || (!aScale && bScale)) {
      llvm::report_fatal_error("Single scale is not supported\n");
    }

    bool existBothScales = aScale && bScale;
    bool isAScaleConstant = aScale && aScale.getDefiningOp<arith::ConstantOp>();
    bool isBScaleConstant = bScale && bScale.getDefiningOp<arith::ConstantOp>();
    Value d = op.getD();
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto dTensorTy = cast<RankedTensorType>(d.getType());
    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();
    ScaleDotElemType aElemType = op.getAElemType();
    ScaleDotElemType bElemType = op.getBElemType();

    auto supportsTypes = [](ScaleDotElemType elemType) {
      return elemType == ScaleDotElemType::E2M1 ||
             elemType == ScaleDotElemType::E4M3 ||
             elemType == ScaleDotElemType::E5M2;
    };

    if (!supportsTypes(aElemType) || !supportsTypes(bElemType)) {
      llvm::report_fatal_error("NYI: mxfp6\n");
    }

    int64_t kDimOperandSize = aTensorTy.getShape().back();

    auto ctx = op.getContext();
    constexpr bool allowXF32 = false;
    FailureOr<MfmaIntrinsic> maybeMfmaIntrinsic = MfmaIntrinsic::selectFor(
        mfmaVersion, mDim, nDim,
        aElemType == ScaleDotElemType::E2M1 ? kDimOperandSize * 2
                                            : kDimOperandSize,
        scaleDotElemTypeToMLIRType(ctx, aElemType),
        scaleDotElemTypeToMLIRType(ctx, bElemType),
        /*withScale=*/true, allowXF32);
    if (failed(maybeMfmaIntrinsic))
      llvm::report_fatal_error("No match found in MFMA database\n");

    StringRef intrinsicName = maybeMfmaIntrinsic->name;
    unsigned kBase = maybeMfmaIntrinsic->kBase;
    // Two fp4 are packed into an uint8.
    unsigned aKBase = aElemType == ScaleDotElemType::E2M1 ? kBase / 2 : kBase;
    unsigned bKBase = bElemType == ScaleDotElemType::E2M1 ? kBase / 2 : kBase;

    int aKWidth = aKBase;
    int bKWidth = bKBase;

    const auto kDimInstrSize = mfmaLayout.getInstrShapeForOperand(aKBase, 0)[1];

    auto repA = mfmaLayout.getRepForOperand(aTensorTy.getShape(), aKWidth, 0);
    auto repB = mfmaLayout.getRepForOperand(bTensorTy.getShape(), bKWidth, 1);
    assert(repA[2] == repB[1]);

    // For fp4 scaled mfma, each thread takes 1 element from scale. Will have
    // better way to get it when adapting other data types. Similar to
    // scaleKBase
    constexpr int scaleKWidth = 1;
    constexpr int scaleKBase = 1;

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedAScale = adaptor.getAScale();
    Value loadedBScale = adaptor.getBScale();
    Value loadedC = adaptor.getC();

    auto numRepM = repA[1];
    auto numRepN = repB[2];
    auto numRepK = repA[2];
    auto numRepB = repA[0];
    assert(repA[0] == repB[0]);

    auto operandA = getValuesFromDotOperandLayoutStruct(
        loadedA, numRepB, numRepM, numRepK, aKWidth, aKBase,
        aTensorTy.getElementType(), allowXF32, /*preserveBF16=*/false);
    auto operandB = getValuesFromDotOperandLayoutStruct(
        loadedB, numRepB, numRepN, numRepK, bKWidth, bKBase,
        bTensorTy.getElementType(), allowXF32, /*preserveBF16=*/false);

    // Scales have the same replica distributions as their corresponding
    // operands.
    SmallVector<ValueTable> operandAScale;
    SmallVector<ValueTable> operandBScale;
    if (existBothScales) {
      auto aScaleTensorTy = cast<RankedTensorType>(aScale.getType());
      operandAScale = getValuesFromDotOperandLayoutStruct(
          loadedAScale, numRepB, numRepM, numRepK, scaleKWidth, scaleKBase,
          aScaleTensorTy.getElementType(), allowXF32, /*preserveBF16=*/false,
          isAScaleConstant);

      auto bScaleTensorTy = cast<RankedTensorType>(bScale.getType());
      operandBScale = getValuesFromDotOperandLayoutStruct(
          loadedBScale, numRepB, numRepN, numRepK, scaleKWidth, scaleKBase,
          bScaleTensorTy.getElementType(), allowXF32, /*preserveBF16=*/false,
          isBScaleConstant);
    }

    auto dstElemTy = dTensorTy.getElementType();
    auto fc = unpackLLElements(loc, loadedC, rewriter);

    unsigned warpSize = triton::gpu::lookupThreadsPerWarp(rewriter);
    // compute number of output elements that each thread holds for one MFMA
    // instruction. subBlocks
    const int subBlocks =
        getNumSubmatrices(aTensorTy.getElementType(), mDim, nDim);
    auto elemsPerVec = mDim * nDim * subBlocks / warpSize;

    Value firstMfma;
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    auto vecTy = vec_ty(dstElemTy, elemsPerVec);
    for (int b = 0; b < numRepB; ++b) {
      for (int m = 0; m < numRepM; ++m) {
        for (int n = 0; n < numRepN; ++n) {
          Value acc = tb.undef(vecTy);
          for (unsigned v = 0; v < elemsPerVec; ++v) {
            acc = tb.insert_element(
                vecTy, acc,
                fc[b * numRepM * numRepN * elemsPerVec +
                   m * numRepN * elemsPerVec + n * elemsPerVec + v],
                tb.i32_val(v));
          }
          acc = zeroAuxiliarBlocks(subBlocks, acc);
          for (int k = 0; k < numRepK; k++) {
            for (int kPack = 0; kPack < aKWidth / aKBase; ++kPack) {
              if (existBothScales) {
                if (mfmaLayout.getIsTransposed()) {
                  acc = generateScaledMFMAOp(intrinsicName,
                                             operandB[kPack][{b, n, k}],
                                             operandA[kPack][{b, m, k}], acc,
                                             operandBScale[kPack][{b, n, k}],
                                             operandAScale[kPack][{b, m, k}],
                                             maybeMfmaIntrinsic->bElementType,
                                             maybeMfmaIntrinsic->aElementType);
                } else {
                  acc = generateScaledMFMAOp(intrinsicName,
                                             operandA[kPack][{b, m, k}],
                                             operandB[kPack][{b, n, k}], acc,
                                             operandAScale[kPack][{b, m, k}],
                                             operandBScale[kPack][{b, n, k}],
                                             maybeMfmaIntrinsic->aElementType,
                                             maybeMfmaIntrinsic->bElementType);
                }
              } else {
                if (mfmaLayout.getIsTransposed()) {
                  acc = generateScaledMFMAOp(intrinsicName,
                                             operandB[kPack][{b, n, k}],
                                             operandA[kPack][{b, m, k}], acc,
                                             maybeMfmaIntrinsic->bElementType,
                                             maybeMfmaIntrinsic->aElementType);
                } else {
                  acc = generateScaledMFMAOp(intrinsicName,
                                             operandA[kPack][{b, m, k}],
                                             operandB[kPack][{b, n, k}], acc,
                                             maybeMfmaIntrinsic->aElementType,
                                             maybeMfmaIntrinsic->bElementType);
                }
              }
              if (!firstMfma)
                firstMfma = acc;
            }
          }
          acc = reduceSubBlocks(subBlocks, acc);
          adjustAccForSmallKDim(fc, acc, dstElemTy, b, m, n, numRepM, numRepN,
                                kDimInstrSize, kDimOperandSize, elemsPerVec);
        }
      }
    }

    // Originally, setprio (high) is set to the high-level dot op. After dot is
    // being lowered to the series of mfma operations, it should be moved next
    // to the first mfma leaving the first mfma staying at the low priority. In
    // this way, incoming warp can be effectively waiting on the first mfma
    // instruction (low priority) while the other warp is executing mfma with
    // high priority. Otherwise, incoming warp can break the cluster.
    if (setPrioOp && firstMfma)
      setPrioOp->moveAfter(firstMfma.getDefiningOp());

    const size_t mmaCount =
        numRepB * numRepM * numRepN * numRepK * aKWidth / aKBase;
    packAndReplaceResult(op, fc, maybeMfmaIntrinsic, dstElemTy, elemTyA,
                         mmaCount);

    return success();
  }
};

} // namespace

namespace mlir::triton::AMD {
LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return cast<RankedTensorType>(tensor.getType());
  };

  assert(isa<DotOperandEncodingAttr>(rankedTType(op.getA()).getEncoding()) &&
         isa<DotOperandEncodingAttr>(rankedTType(op.getB()).getEncoding()) &&
         "Both A and B should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(isa<AMDMfmaEncodingAttr>(cTensorTy.getEncoding()) &&
         "Currently, we only support C with a mfma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's C operand should pass the same number of values as D.");

  auto loc = op.getLoc();
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  DotOpMFMAConversionHelper helper(mfmaLayout, rewriter, typeConverter, loc);

  return helper.convertDot(op, adaptor);
}

LogicalResult convertScaledMFMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter) {
  assert(isa<DotOperandEncodingAttr>(op.getA().getType().getEncoding()) &&
         isa<DotOperandEncodingAttr>(op.getB().getType().getEncoding()) &&
         "Both lhs and rhs should be in DotOperand layout.");

  auto aScale = op.getAScale();
  auto bScale = op.getBScale();

  // If the tt.dot_scaled is transformed from a tt.dot, both scales are None. In
  // this case, both scales remain None in this method and we will generate a
  // mfma instruction with the scale operand to be 0. Then there's an
  // optimization pass in the LLVM backend to convert such V_MFMA_SCALE_*_F8F6F4
  // instruction to V_MFMA_*_F8F6F4 to avoid LD_SCALE.
  //
  // If the tt.dot_scaled is not from a tt.dot but native, we support 0, 1, 2
  // scales and treat them in different ways:
  //
  // 1. #scales = 0: Just like those transformed from tt.dot, both scales remain
  // None.
  // 2. #scales = 1: The upstream transform guarantees to create constant
  // scales for the absent.
  // 2. #scales = 2: Both scales should exist.

  // Thus in this pass, there shouldn't be a single scale present.
  assert(((aScale && bScale) || (!aScale && !bScale)) &&
         "Single scale is not supported");

  if (aScale && bScale) {
    assert(
        isa<LinearEncodingAttr>(aScale.getType().getEncoding()) &&
        isa<LinearEncodingAttr>(bScale.getType().getEncoding()) &&
        "If scales exist, both LhsScale and RhsScale should be linear layout.");
  }

  auto cTensorTy = op.getC().getType();
  auto dTensorTy = op.getD().getType();
  assert(isa<AMDMfmaEncodingAttr>(cTensorTy.getEncoding()) &&
         "Currently, we only support C with a mfma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's C operand should pass the same number of values as D.");

  auto loc = op.getLoc();
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  ScaledDotOpMFMAConversionHelper helper(mfmaLayout, rewriter, typeConverter,
                                         loc);

  return helper.convertScaledDot(op, adaptor);
}
} // namespace mlir::triton::AMD
