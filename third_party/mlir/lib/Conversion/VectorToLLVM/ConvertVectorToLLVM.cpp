//===- VectorToLLVM.cpp - Conversion from Vector to the LLVM dialect ------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/VectorOps/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

template <typename T>
static LLVM::LLVMType getPtrToElementType(T containerType,
                                          LLVMTypeConverter &lowering) {
  return lowering.convertType(containerType.getElementType())
      .template cast<LLVM::LLVMType>()
      .getPointerTo();
}

// Helper to reduce vector type by one rank at front.
static VectorType reducedVectorTypeFront(VectorType tp) {
  assert((tp.getRank() > 1) && "unlowerable vector type");
  return VectorType::get(tp.getShape().drop_front(), tp.getElementType());
}

// Helper to reduce vector type by *all* but one rank at back.
static VectorType reducedVectorTypeBack(VectorType tp) {
  assert((tp.getRank() > 1) && "unlowerable vector type");
  return VectorType::get(tp.getShape().take_back(), tp.getElementType());
}

// Helper that picks the proper sequence for inserting.
static ValuePtr insertOne(ConversionPatternRewriter &rewriter,
                          LLVMTypeConverter &lowering, Location loc,
                          ValuePtr val1, ValuePtr val2, Type llvmType,
                          int64_t rank, int64_t pos) {
  if (rank == 1) {
    auto idxType = rewriter.getIndexType();
    auto constant = rewriter.create<LLVM::ConstantOp>(
        loc, lowering.convertType(idxType),
        rewriter.getIntegerAttr(idxType, pos));
    return rewriter.create<LLVM::InsertElementOp>(loc, llvmType, val1, val2,
                                                  constant);
  }
  return rewriter.create<LLVM::InsertValueOp>(loc, llvmType, val1, val2,
                                              rewriter.getI64ArrayAttr(pos));
}

// Helper that picks the proper sequence for extracting.
static ValuePtr extractOne(ConversionPatternRewriter &rewriter,
                           LLVMTypeConverter &lowering, Location loc,
                           ValuePtr val, Type llvmType, int64_t rank,
                           int64_t pos) {
  if (rank == 1) {
    auto idxType = rewriter.getIndexType();
    auto constant = rewriter.create<LLVM::ConstantOp>(
        loc, lowering.convertType(idxType),
        rewriter.getIntegerAttr(idxType, pos));
    return rewriter.create<LLVM::ExtractElementOp>(loc, llvmType, val,
                                                   constant);
  }
  return rewriter.create<LLVM::ExtractValueOp>(loc, llvmType, val,
                                               rewriter.getI64ArrayAttr(pos));
}

class VectorBroadcastOpConversion : public LLVMOpLowering {
public:
  explicit VectorBroadcastOpConversion(MLIRContext *context,
                                       LLVMTypeConverter &typeConverter)
      : LLVMOpLowering(vector::BroadcastOp::getOperationName(), context,
                       typeConverter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<ValuePtr> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto broadcastOp = cast<vector::BroadcastOp>(op);
    VectorType dstVectorType = broadcastOp.getVectorType();
    if (lowering.convertType(dstVectorType) == nullptr)
      return matchFailure();
    // Rewrite when the full vector type can be lowered (which
    // implies all 'reduced' types can be lowered too).
    auto adaptor = vector::BroadcastOpOperandAdaptor(operands);
    VectorType srcVectorType =
        broadcastOp.getSourceType().dyn_cast<VectorType>();
    rewriter.replaceOp(
        op, expandRanks(adaptor.source(), // source value to be expanded
                        op->getLoc(),     // location of original broadcast
                        srcVectorType, dstVectorType, rewriter));
    return matchSuccess();
  }

private:
  // Expands the given source value over all the ranks, as defined
  // by the source and destination type (a null source type denotes
  // expansion from a scalar value into a vector).
  //
  // TODO(ajcbik): consider replacing this one-pattern lowering
  //               with a two-pattern lowering using other vector
  //               ops once all insert/extract/shuffle operations
  //               are available with lowering implemention.
  //
  ValuePtr expandRanks(ValuePtr value, Location loc, VectorType srcVectorType,
                       VectorType dstVectorType,
                       ConversionPatternRewriter &rewriter) const {
    assert((dstVectorType != nullptr) && "invalid result type in broadcast");
    // Determine rank of source and destination.
    int64_t srcRank = srcVectorType ? srcVectorType.getRank() : 0;
    int64_t dstRank = dstVectorType.getRank();
    int64_t curDim = dstVectorType.getDimSize(0);
    if (srcRank < dstRank)
      // Duplicate this rank.
      return duplicateOneRank(value, loc, srcVectorType, dstVectorType, dstRank,
                              curDim, rewriter);
    // If all trailing dimensions are the same, the broadcast consists of
    // simply passing through the source value and we are done. Otherwise,
    // any non-matching dimension forces a stretch along this rank.
    assert((srcVectorType != nullptr) && (srcRank > 0) &&
           (srcRank == dstRank) && "invalid rank in broadcast");
    for (int64_t r = 0; r < dstRank; r++) {
      if (srcVectorType.getDimSize(r) != dstVectorType.getDimSize(r)) {
        return stretchOneRank(value, loc, srcVectorType, dstVectorType, dstRank,
                              curDim, rewriter);
      }
    }
    return value;
  }

  // Picks the best way to duplicate a single rank. For the 1-D case, a
  // single insert-elt/shuffle is the most efficient expansion. For higher
  // dimensions, however, we need dim x insert-values on a new broadcast
  // with one less leading dimension, which will be lowered "recursively"
  // to matching LLVM IR.
  // For example:
  //   v = broadcast s : f32 to vector<4x2xf32>
  // becomes:
  //   x = broadcast s : f32 to vector<2xf32>
  //   v = [x,x,x,x]
  // becomes:
  //   x = [s,s]
  //   v = [x,x,x,x]
  ValuePtr duplicateOneRank(ValuePtr value, Location loc,
                            VectorType srcVectorType, VectorType dstVectorType,
                            int64_t rank, int64_t dim,
                            ConversionPatternRewriter &rewriter) const {
    Type llvmType = lowering.convertType(dstVectorType);
    assert((llvmType != nullptr) && "unlowerable vector type");
    if (rank == 1) {
      ValuePtr undef = rewriter.create<LLVM::UndefOp>(loc, llvmType);
      ValuePtr expand =
          insertOne(rewriter, lowering, loc, undef, value, llvmType, rank, 0);
      SmallVector<int32_t, 4> zeroValues(dim, 0);
      return rewriter.create<LLVM::ShuffleVectorOp>(
          loc, expand, undef, rewriter.getI32ArrayAttr(zeroValues));
    }
    ValuePtr expand =
        expandRanks(value, loc, srcVectorType,
                    reducedVectorTypeFront(dstVectorType), rewriter);
    ValuePtr result = rewriter.create<LLVM::UndefOp>(loc, llvmType);
    for (int64_t d = 0; d < dim; ++d) {
      result =
          insertOne(rewriter, lowering, loc, result, expand, llvmType, rank, d);
    }
    return result;
  }

  // Picks the best way to stretch a single rank. For the 1-D case, a
  // single insert-elt/shuffle is the most efficient expansion when at
  // a stretch. Otherwise, every dimension needs to be expanded
  // individually and individually inserted in the resulting vector.
  // For example:
  //   v = broadcast w : vector<4x1x2xf32> to vector<4x2x2xf32>
  // becomes:
  //   a = broadcast w[0] : vector<1x2xf32> to vector<2x2xf32>
  //   b = broadcast w[1] : vector<1x2xf32> to vector<2x2xf32>
  //   c = broadcast w[2] : vector<1x2xf32> to vector<2x2xf32>
  //   d = broadcast w[3] : vector<1x2xf32> to vector<2x2xf32>
  //   v = [a,b,c,d]
  // becomes:
  //   x = broadcast w[0][0] : vector<2xf32> to vector <2x2xf32>
  //   y = broadcast w[1][0] : vector<2xf32> to vector <2x2xf32>
  //   a = [x, y]
  //   etc.
  ValuePtr stretchOneRank(ValuePtr value, Location loc,
                          VectorType srcVectorType, VectorType dstVectorType,
                          int64_t rank, int64_t dim,
                          ConversionPatternRewriter &rewriter) const {
    Type llvmType = lowering.convertType(dstVectorType);
    assert((llvmType != nullptr) && "unlowerable vector type");
    ValuePtr result = rewriter.create<LLVM::UndefOp>(loc, llvmType);
    bool atStretch = dim != srcVectorType.getDimSize(0);
    if (rank == 1) {
      assert(atStretch);
      Type redLlvmType = lowering.convertType(dstVectorType.getElementType());
      ValuePtr one =
          extractOne(rewriter, lowering, loc, value, redLlvmType, rank, 0);
      ValuePtr expand =
          insertOne(rewriter, lowering, loc, result, one, llvmType, rank, 0);
      SmallVector<int32_t, 4> zeroValues(dim, 0);
      return rewriter.create<LLVM::ShuffleVectorOp>(
          loc, expand, result, rewriter.getI32ArrayAttr(zeroValues));
    }
    VectorType redSrcType = reducedVectorTypeFront(srcVectorType);
    VectorType redDstType = reducedVectorTypeFront(dstVectorType);
    Type redLlvmType = lowering.convertType(redSrcType);
    for (int64_t d = 0; d < dim; ++d) {
      int64_t pos = atStretch ? 0 : d;
      ValuePtr one =
          extractOne(rewriter, lowering, loc, value, redLlvmType, rank, pos);
      ValuePtr expand = expandRanks(one, loc, redSrcType, redDstType, rewriter);
      result =
          insertOne(rewriter, lowering, loc, result, expand, llvmType, rank, d);
    }
    return result;
  }
};

class VectorShuffleOpConversion : public LLVMOpLowering {
public:
  explicit VectorShuffleOpConversion(MLIRContext *context,
                                     LLVMTypeConverter &typeConverter)
      : LLVMOpLowering(vector::ShuffleOp::getOperationName(), context,
                       typeConverter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<ValuePtr> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto adaptor = vector::ShuffleOpOperandAdaptor(operands);
    auto shuffleOp = cast<vector::ShuffleOp>(op);
    auto v1Type = shuffleOp.getV1VectorType();
    auto v2Type = shuffleOp.getV2VectorType();
    auto vectorType = shuffleOp.getVectorType();
    Type llvmType = lowering.convertType(vectorType);
    auto maskArrayAttr = shuffleOp.mask();

    // Bail if result type cannot be lowered.
    if (!llvmType)
      return matchFailure();

    // Get rank and dimension sizes.
    int64_t rank = vectorType.getRank();
    assert(v1Type.getRank() == rank);
    assert(v2Type.getRank() == rank);
    int64_t v1Dim = v1Type.getDimSize(0);

    // For rank 1, where both operands have *exactly* the same vector type,
    // there is direct shuffle support in LLVM. Use it!
    if (rank == 1 && v1Type == v2Type) {
      ValuePtr shuffle = rewriter.create<LLVM::ShuffleVectorOp>(
          loc, adaptor.v1(), adaptor.v2(), maskArrayAttr);
      rewriter.replaceOp(op, shuffle);
      return matchSuccess();
    }

    // For all other cases, insert the individual values individually.
    ValuePtr insert = rewriter.create<LLVM::UndefOp>(loc, llvmType);
    int64_t insPos = 0;
    for (auto en : llvm::enumerate(maskArrayAttr)) {
      int64_t extPos = en.value().cast<IntegerAttr>().getInt();
      ValuePtr value = adaptor.v1();
      if (extPos >= v1Dim) {
        extPos -= v1Dim;
        value = adaptor.v2();
      }
      ValuePtr extract =
          extractOne(rewriter, lowering, loc, value, llvmType, rank, extPos);
      insert = insertOne(rewriter, lowering, loc, insert, extract, llvmType,
                         rank, insPos++);
    }
    rewriter.replaceOp(op, insert);
    return matchSuccess();
  }
};

class VectorExtractElementOpConversion : public LLVMOpLowering {
public:
  explicit VectorExtractElementOpConversion(MLIRContext *context,
                                            LLVMTypeConverter &typeConverter)
      : LLVMOpLowering(vector::ExtractElementOp::getOperationName(), context,
                       typeConverter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<ValuePtr> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto adaptor = vector::ExtractElementOpOperandAdaptor(operands);
    auto extractEltOp = cast<vector::ExtractElementOp>(op);
    auto vectorType = extractEltOp.getVectorType();
    auto llvmType = lowering.convertType(vectorType.getElementType());

    // Bail if result type cannot be lowered.
    if (!llvmType)
      return matchFailure();

    rewriter.replaceOpWithNewOp<LLVM::ExtractElementOp>(
        op, llvmType, adaptor.vector(), adaptor.position());
    return matchSuccess();
  }
};

class VectorExtractOpConversion : public LLVMOpLowering {
public:
  explicit VectorExtractOpConversion(MLIRContext *context,
                                     LLVMTypeConverter &typeConverter)
      : LLVMOpLowering(vector::ExtractOp::getOperationName(), context,
                       typeConverter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<ValuePtr> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto adaptor = vector::ExtractOpOperandAdaptor(operands);
    auto extractOp = cast<vector::ExtractOp>(op);
    auto vectorType = extractOp.getVectorType();
    auto resultType = extractOp.getResult()->getType();
    auto llvmResultType = lowering.convertType(resultType);
    auto positionArrayAttr = extractOp.position();

    // Bail if result type cannot be lowered.
    if (!llvmResultType)
      return matchFailure();

    // One-shot extraction of vector from array (only requires extractvalue).
    if (resultType.isa<VectorType>()) {
      ValuePtr extracted = rewriter.create<LLVM::ExtractValueOp>(
          loc, llvmResultType, adaptor.vector(), positionArrayAttr);
      rewriter.replaceOp(op, extracted);
      return matchSuccess();
    }

    // Potential extraction of 1-D vector from array.
    auto *context = op->getContext();
    ValuePtr extracted = adaptor.vector();
    auto positionAttrs = positionArrayAttr.getValue();
    if (positionAttrs.size() > 1) {
      auto oneDVectorType = reducedVectorTypeBack(vectorType);
      auto nMinusOnePositionAttrs =
          ArrayAttr::get(positionAttrs.drop_back(), context);
      extracted = rewriter.create<LLVM::ExtractValueOp>(
          loc, lowering.convertType(oneDVectorType), extracted,
          nMinusOnePositionAttrs);
    }

    // Remaining extraction of element from 1-D LLVM vector
    auto position = positionAttrs.back().cast<IntegerAttr>();
    auto i64Type = LLVM::LLVMType::getInt64Ty(lowering.getDialect());
    auto constant = rewriter.create<LLVM::ConstantOp>(loc, i64Type, position);
    extracted =
        rewriter.create<LLVM::ExtractElementOp>(loc, extracted, constant);
    rewriter.replaceOp(op, extracted);

    return matchSuccess();
  }
};

class VectorInsertElementOpConversion : public LLVMOpLowering {
public:
  explicit VectorInsertElementOpConversion(MLIRContext *context,
                                           LLVMTypeConverter &typeConverter)
      : LLVMOpLowering(vector::InsertElementOp::getOperationName(), context,
                       typeConverter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<ValuePtr> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto adaptor = vector::InsertElementOpOperandAdaptor(operands);
    auto insertEltOp = cast<vector::InsertElementOp>(op);
    auto vectorType = insertEltOp.getDestVectorType();
    auto llvmType = lowering.convertType(vectorType);

    // Bail if result type cannot be lowered.
    if (!llvmType)
      return matchFailure();

    rewriter.replaceOpWithNewOp<LLVM::InsertElementOp>(
        op, llvmType, adaptor.dest(), adaptor.source(), adaptor.position());
    return matchSuccess();
  }
};

class VectorInsertOpConversion : public LLVMOpLowering {
public:
  explicit VectorInsertOpConversion(MLIRContext *context,
                                    LLVMTypeConverter &typeConverter)
      : LLVMOpLowering(vector::InsertOp::getOperationName(), context,
                       typeConverter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<ValuePtr> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto adaptor = vector::InsertOpOperandAdaptor(operands);
    auto insertOp = cast<vector::InsertOp>(op);
    auto sourceType = insertOp.getSourceType();
    auto destVectorType = insertOp.getDestVectorType();
    auto llvmResultType = lowering.convertType(destVectorType);
    auto positionArrayAttr = insertOp.position();

    // Bail if result type cannot be lowered.
    if (!llvmResultType)
      return matchFailure();

    // One-shot insertion of a vector into an array (only requires insertvalue).
    if (sourceType.isa<VectorType>()) {
      ValuePtr inserted = rewriter.create<LLVM::InsertValueOp>(
          loc, llvmResultType, adaptor.dest(), adaptor.source(),
          positionArrayAttr);
      rewriter.replaceOp(op, inserted);
      return matchSuccess();
    }

    // Potential extraction of 1-D vector from array.
    auto *context = op->getContext();
    ValuePtr extracted = adaptor.dest();
    auto positionAttrs = positionArrayAttr.getValue();
    auto position = positionAttrs.back().cast<IntegerAttr>();
    auto oneDVectorType = destVectorType;
    if (positionAttrs.size() > 1) {
      oneDVectorType = reducedVectorTypeBack(destVectorType);
      auto nMinusOnePositionAttrs =
          ArrayAttr::get(positionAttrs.drop_back(), context);
      extracted = rewriter.create<LLVM::ExtractValueOp>(
          loc, lowering.convertType(oneDVectorType), extracted,
          nMinusOnePositionAttrs);
    }

    // Insertion of an element into a 1-D LLVM vector.
    auto i64Type = LLVM::LLVMType::getInt64Ty(lowering.getDialect());
    auto constant = rewriter.create<LLVM::ConstantOp>(loc, i64Type, position);
    ValuePtr inserted = rewriter.create<LLVM::InsertElementOp>(
        loc, lowering.convertType(oneDVectorType), extracted, adaptor.source(),
        constant);

    // Potential insertion of resulting 1-D vector into array.
    if (positionAttrs.size() > 1) {
      auto nMinusOnePositionAttrs =
          ArrayAttr::get(positionAttrs.drop_back(), context);
      inserted = rewriter.create<LLVM::InsertValueOp>(loc, llvmResultType,
                                                      adaptor.dest(), inserted,
                                                      nMinusOnePositionAttrs);
    }

    rewriter.replaceOp(op, inserted);
    return matchSuccess();
  }
};

class VectorOuterProductOpConversion : public LLVMOpLowering {
public:
  explicit VectorOuterProductOpConversion(MLIRContext *context,
                                          LLVMTypeConverter &typeConverter)
      : LLVMOpLowering(vector::OuterProductOp::getOperationName(), context,
                       typeConverter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<ValuePtr> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto adaptor = vector::OuterProductOpOperandAdaptor(operands);
    auto *ctx = op->getContext();
    auto vLHS = adaptor.lhs()->getType().cast<LLVM::LLVMType>();
    auto vRHS = adaptor.rhs()->getType().cast<LLVM::LLVMType>();
    auto rankLHS = vLHS.getUnderlyingType()->getVectorNumElements();
    auto rankRHS = vRHS.getUnderlyingType()->getVectorNumElements();
    auto llvmArrayOfVectType = lowering.convertType(
        cast<vector::OuterProductOp>(op).getResult()->getType());
    ValuePtr desc = rewriter.create<LLVM::UndefOp>(loc, llvmArrayOfVectType);
    ValuePtr a = adaptor.lhs(), b = adaptor.rhs();
    ValuePtr acc = adaptor.acc().empty() ? nullptr : adaptor.acc().front();
    SmallVector<ValuePtr, 8> lhs, accs;
    lhs.reserve(rankLHS);
    accs.reserve(rankLHS);
    for (unsigned d = 0, e = rankLHS; d < e; ++d) {
      // shufflevector explicitly requires i32.
      auto attr = rewriter.getI32IntegerAttr(d);
      SmallVector<Attribute, 4> bcastAttr(rankRHS, attr);
      auto bcastArrayAttr = ArrayAttr::get(bcastAttr, ctx);
      ValuePtr aD = nullptr, accD = nullptr;
      // 1. Broadcast the element a[d] into vector aD.
      aD = rewriter.create<LLVM::ShuffleVectorOp>(loc, a, a, bcastArrayAttr);
      // 2. If acc is present, extract 1-d vector acc[d] into accD.
      if (acc)
        accD = rewriter.create<LLVM::ExtractValueOp>(
            loc, vRHS, acc, rewriter.getI64ArrayAttr(d));
      // 3. Compute aD outer b (plus accD, if relevant).
      ValuePtr aOuterbD =
          accD ? rewriter.create<LLVM::FMulAddOp>(loc, vRHS, aD, b, accD)
                     .getResult()
               : rewriter.create<LLVM::FMulOp>(loc, aD, b).getResult();
      // 4. Insert as value `d` in the descriptor.
      desc = rewriter.create<LLVM::InsertValueOp>(loc, llvmArrayOfVectType,
                                                  desc, aOuterbD,
                                                  rewriter.getI64ArrayAttr(d));
    }
    rewriter.replaceOp(op, desc);
    return matchSuccess();
  }
};

class VectorTypeCastOpConversion : public LLVMOpLowering {
public:
  explicit VectorTypeCastOpConversion(MLIRContext *context,
                                      LLVMTypeConverter &typeConverter)
      : LLVMOpLowering(vector::TypeCastOp::getOperationName(), context,
                       typeConverter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<ValuePtr> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    vector::TypeCastOp castOp = cast<vector::TypeCastOp>(op);
    MemRefType sourceMemRefType =
        castOp.getOperand()->getType().cast<MemRefType>();
    MemRefType targetMemRefType =
        castOp.getResult()->getType().cast<MemRefType>();

    // Only static shape casts supported atm.
    if (!sourceMemRefType.hasStaticShape() ||
        !targetMemRefType.hasStaticShape())
      return matchFailure();

    auto llvmSourceDescriptorTy =
        operands[0]->getType().dyn_cast<LLVM::LLVMType>();
    if (!llvmSourceDescriptorTy || !llvmSourceDescriptorTy.isStructTy())
      return matchFailure();
    MemRefDescriptor sourceMemRef(operands[0]);

    auto llvmTargetDescriptorTy = lowering.convertType(targetMemRefType)
                                      .dyn_cast_or_null<LLVM::LLVMType>();
    if (!llvmTargetDescriptorTy || !llvmTargetDescriptorTy.isStructTy())
      return matchFailure();

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides =
        getStridesAndOffset(sourceMemRefType, strides, offset);
    bool isContiguous = (strides.back() == 1);
    if (isContiguous) {
      auto sizes = sourceMemRefType.getShape();
      for (int index = 0, e = strides.size() - 2; index < e; ++index) {
        if (strides[index] != strides[index + 1] * sizes[index + 1]) {
          isContiguous = false;
          break;
        }
      }
    }
    // Only contiguous source tensors supported atm.
    if (failed(successStrides) || !isContiguous)
      return matchFailure();

    auto int64Ty = LLVM::LLVMType::getInt64Ty(lowering.getDialect());

    // Create descriptor.
    auto desc = MemRefDescriptor::undef(rewriter, loc, llvmTargetDescriptorTy);
    Type llvmTargetElementTy = desc.getElementType();
    // Set allocated ptr.
    ValuePtr allocated = sourceMemRef.allocatedPtr(rewriter, loc);
    allocated =
        rewriter.create<LLVM::BitcastOp>(loc, llvmTargetElementTy, allocated);
    desc.setAllocatedPtr(rewriter, loc, allocated);
    // Set aligned ptr.
    ValuePtr ptr = sourceMemRef.alignedPtr(rewriter, loc);
    ptr = rewriter.create<LLVM::BitcastOp>(loc, llvmTargetElementTy, ptr);
    desc.setAlignedPtr(rewriter, loc, ptr);
    // Fill offset 0.
    auto attr = rewriter.getIntegerAttr(rewriter.getIndexType(), 0);
    auto zero = rewriter.create<LLVM::ConstantOp>(loc, int64Ty, attr);
    desc.setOffset(rewriter, loc, zero);

    // Fill size and stride descriptors in memref.
    for (auto indexedSize : llvm::enumerate(targetMemRefType.getShape())) {
      int64_t index = indexedSize.index();
      auto sizeAttr =
          rewriter.getIntegerAttr(rewriter.getIndexType(), indexedSize.value());
      auto size = rewriter.create<LLVM::ConstantOp>(loc, int64Ty, sizeAttr);
      desc.setSize(rewriter, loc, index, size);
      auto strideAttr =
          rewriter.getIntegerAttr(rewriter.getIndexType(), strides[index]);
      auto stride = rewriter.create<LLVM::ConstantOp>(loc, int64Ty, strideAttr);
      desc.setStride(rewriter, loc, index, stride);
    }

    rewriter.replaceOp(op, {desc});
    return matchSuccess();
  }
};

class VectorPrintOpConversion : public LLVMOpLowering {
public:
  explicit VectorPrintOpConversion(MLIRContext *context,
                                   LLVMTypeConverter &typeConverter)
      : LLVMOpLowering(vector::PrintOp::getOperationName(), context,
                       typeConverter) {}

  // Proof-of-concept lowering implementation that relies on a small
  // runtime support library, which only needs to provide a few
  // printing methods (single value for all data types, opening/closing
  // bracket, comma, newline). The lowering fully unrolls a vector
  // in terms of these elementary printing operations. The advantage
  // of this approach is that the library can remain unaware of all
  // low-level implementation details of vectors while still supporting
  // output of any shaped and dimensioned vector. Due to full unrolling,
  // this approach is less suited for very large vectors though.
  //
  // TODO(ajcbik): rely solely on libc in future? something else?
  //
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<ValuePtr> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto printOp = cast<vector::PrintOp>(op);
    auto adaptor = vector::PrintOpOperandAdaptor(operands);
    Type printType = printOp.getPrintType();

    if (lowering.convertType(printType) == nullptr)
      return matchFailure();

    // Make sure element type has runtime support (currently just Float/Double).
    VectorType vectorType = printType.dyn_cast<VectorType>();
    Type eltType = vectorType ? vectorType.getElementType() : printType;
    int64_t rank = vectorType ? vectorType.getRank() : 0;
    Operation *printer;
    if (eltType.isF32())
      printer = getPrintFloat(op);
    else if (eltType.isF64())
      printer = getPrintDouble(op);
    else
      return matchFailure();

    // Unroll vector into elementary print calls.
    emitRanks(rewriter, op, adaptor.source(), vectorType, printer, rank);
    emitCall(rewriter, op->getLoc(), getPrintNewline(op));
    rewriter.eraseOp(op);
    return matchSuccess();
  }

private:
  void emitRanks(ConversionPatternRewriter &rewriter, Operation *op,
                 ValuePtr value, VectorType vectorType, Operation *printer,
                 int64_t rank) const {
    Location loc = op->getLoc();
    if (rank == 0) {
      emitCall(rewriter, loc, printer, value);
      return;
    }

    emitCall(rewriter, loc, getPrintOpen(op));
    Operation *printComma = getPrintComma(op);
    int64_t dim = vectorType.getDimSize(0);
    for (int64_t d = 0; d < dim; ++d) {
      auto reducedType =
          rank > 1 ? reducedVectorTypeFront(vectorType) : nullptr;
      auto llvmType = lowering.convertType(
          rank > 1 ? reducedType : vectorType.getElementType());
      ValuePtr nestedVal =
          extractOne(rewriter, lowering, loc, value, llvmType, rank, d);
      emitRanks(rewriter, op, nestedVal, reducedType, printer, rank - 1);
      if (d != dim - 1)
        emitCall(rewriter, loc, printComma);
    }
    emitCall(rewriter, loc, getPrintClose(op));
  }

  // Helper to emit a call.
  static void emitCall(ConversionPatternRewriter &rewriter, Location loc,
                       Operation *ref, ValueRange params = ValueRange()) {
    rewriter.create<LLVM::CallOp>(loc, ArrayRef<Type>{},
                                  rewriter.getSymbolRefAttr(ref), params);
  }

  // Helper for printer method declaration (first hit) and lookup.
  static Operation *getPrint(Operation *op, LLVM::LLVMDialect *dialect,
                             StringRef name, ArrayRef<LLVM::LLVMType> params) {
    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (func)
      return func;
    OpBuilder moduleBuilder(module.getBodyRegion());
    return moduleBuilder.create<LLVM::LLVMFuncOp>(
        op->getLoc(), name,
        LLVM::LLVMType::getFunctionTy(LLVM::LLVMType::getVoidTy(dialect),
                                      params, /*isVarArg=*/false));
  }

  // Helpers for method names.
  Operation *getPrintFloat(Operation *op) const {
    LLVM::LLVMDialect *dialect = lowering.getDialect();
    return getPrint(op, dialect, "print_f32",
                    LLVM::LLVMType::getFloatTy(dialect));
  }
  Operation *getPrintDouble(Operation *op) const {
    LLVM::LLVMDialect *dialect = lowering.getDialect();
    return getPrint(op, dialect, "print_f64",
                    LLVM::LLVMType::getDoubleTy(dialect));
  }
  Operation *getPrintOpen(Operation *op) const {
    return getPrint(op, lowering.getDialect(), "print_open", {});
  }
  Operation *getPrintClose(Operation *op) const {
    return getPrint(op, lowering.getDialect(), "print_close", {});
  }
  Operation *getPrintComma(Operation *op) const {
    return getPrint(op, lowering.getDialect(), "print_comma", {});
  }
  Operation *getPrintNewline(Operation *op) const {
    return getPrint(op, lowering.getDialect(), "print_newline", {});
  }
};

/// Populate the given list with patterns that convert from Vector to LLVM.
void mlir::populateVectorToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  patterns.insert<VectorBroadcastOpConversion, VectorShuffleOpConversion,
                  VectorExtractElementOpConversion, VectorExtractOpConversion,
                  VectorInsertElementOpConversion, VectorInsertOpConversion,
                  VectorOuterProductOpConversion, VectorTypeCastOpConversion,
                  VectorPrintOpConversion>(converter.getDialect()->getContext(),
                                           converter);
}

namespace {
struct LowerVectorToLLVMPass : public ModulePass<LowerVectorToLLVMPass> {
  void runOnModule() override;
};
} // namespace

void LowerVectorToLLVMPass::runOnModule() {
  // Convert to the LLVM IR dialect using the converter defined above.
  OwningRewritePatternList patterns;
  LLVMTypeConverter converter(&getContext());
  populateVectorToLLVMConversionPatterns(converter, patterns);
  populateStdToLLVMConversionPatterns(converter, patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
  if (failed(
          applyPartialConversion(getModule(), target, patterns, &converter))) {
    signalPassFailure();
  }
}

OpPassBase<ModuleOp> *mlir::createLowerVectorToLLVMPass() {
  return new LowerVectorToLLVMPass();
}

static PassRegistration<LowerVectorToLLVMPass>
    pass("convert-vector-to-llvm",
         "Lower the operations from the vector dialect into the LLVM dialect");
