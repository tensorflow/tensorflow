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

class VectorBroadcastOpConversion : public LLVMOpLowering {
public:
  explicit VectorBroadcastOpConversion(MLIRContext *context,
                                       LLVMTypeConverter &typeConverter)
      : LLVMOpLowering(vector::BroadcastOp::getOperationName(), context,
                       typeConverter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto broadcastOp = cast<vector::BroadcastOp>(op);
    VectorType dstVectorType = broadcastOp.getVectorType();
    if (lowering.convertType(dstVectorType) == nullptr)
      return matchFailure();
    // Rewrite when the full vector type can be lowered (which
    // implies all 'reduced' types can be lowered too).
    VectorType srcVectorType =
        broadcastOp.getSourceType().dyn_cast<VectorType>();
    rewriter.replaceOp(
        op, expandRanks(operands[0],  // source value to be expanded
                        op->getLoc(), // location of original broadcast
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
  Value *expandRanks(Value *value, Location loc, VectorType srcVectorType,
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
  Value *duplicateOneRank(Value *value, Location loc, VectorType srcVectorType,
                          VectorType dstVectorType, int64_t rank, int64_t dim,
                          ConversionPatternRewriter &rewriter) const {
    Type llvmType = lowering.convertType(dstVectorType);
    assert((llvmType != nullptr) && "unlowerable vector type");
    if (rank == 1) {
      Value *undef = rewriter.create<LLVM::UndefOp>(loc, llvmType);
      Value *expand = insertOne(undef, value, loc, llvmType, rank, 0, rewriter);
      SmallVector<int32_t, 4> zeroValues(dim, 0);
      return rewriter.create<LLVM::ShuffleVectorOp>(
          loc, expand, undef, rewriter.getI32ArrayAttr(zeroValues));
    }
    Value *expand = expandRanks(value, loc, srcVectorType,
                                reducedVectorType(dstVectorType), rewriter);
    Value *result = rewriter.create<LLVM::UndefOp>(loc, llvmType);
    for (int64_t d = 0; d < dim; ++d) {
      result = insertOne(result, expand, loc, llvmType, rank, d, rewriter);
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
  Value *stretchOneRank(Value *value, Location loc, VectorType srcVectorType,
                        VectorType dstVectorType, int64_t rank, int64_t dim,
                        ConversionPatternRewriter &rewriter) const {
    Type llvmType = lowering.convertType(dstVectorType);
    assert((llvmType != nullptr) && "unlowerable vector type");
    Value *result = rewriter.create<LLVM::UndefOp>(loc, llvmType);
    bool atStretch = dim != srcVectorType.getDimSize(0);
    if (rank == 1) {
      Type redLlvmType = lowering.convertType(dstVectorType.getElementType());
      if (atStretch) {
        Value *one = extractOne(value, loc, redLlvmType, rank, 0, rewriter);
        Value *expand =
            insertOne(result, one, loc, llvmType, rank, 0, rewriter);
        SmallVector<int32_t, 4> zeroValues(dim, 0);
        return rewriter.create<LLVM::ShuffleVectorOp>(
            loc, expand, result, rewriter.getI32ArrayAttr(zeroValues));
      }
      for (int64_t d = 0; d < dim; ++d) {
        Value *one = extractOne(value, loc, redLlvmType, rank, d, rewriter);
        result = insertOne(result, one, loc, llvmType, rank, d, rewriter);
      }
    } else {
      VectorType redSrcType = reducedVectorType(srcVectorType);
      VectorType redDstType = reducedVectorType(dstVectorType);
      Type redLlvmType = lowering.convertType(redSrcType);
      for (int64_t d = 0; d < dim; ++d) {
        int64_t pos = atStretch ? 0 : d;
        Value *one = extractOne(value, loc, redLlvmType, rank, pos, rewriter);
        Value *expand = expandRanks(one, loc, redSrcType, redDstType, rewriter);
        result = insertOne(result, expand, loc, llvmType, rank, d, rewriter);
      }
    }
    return result;
  }

  // Picks the proper sequence for inserting.
  Value *insertOne(Value *val1, Value *val2, Location loc, Type llvmType,
                   int64_t rank, int64_t pos,
                   ConversionPatternRewriter &rewriter) const {
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

  // Picks the proper sequence for extracting.
  Value *extractOne(Value *value, Location loc, Type llvmType, int64_t rank,
                    int64_t pos, ConversionPatternRewriter &rewriter) const {
    if (rank == 1) {
      auto idxType = rewriter.getIndexType();
      auto constant = rewriter.create<LLVM::ConstantOp>(
          loc, lowering.convertType(idxType),
          rewriter.getIntegerAttr(idxType, pos));
      return rewriter.create<LLVM::ExtractElementOp>(loc, llvmType, value,
                                                     constant);
    }
    return rewriter.create<LLVM::ExtractValueOp>(loc, llvmType, value,
                                                 rewriter.getI64ArrayAttr(pos));
  }

  // Helper to reduce vector type by one rank.
  static VectorType reducedVectorType(VectorType tp) {
    assert((tp.getRank() > 1) && "unlowerable vector type");
    return VectorType::get(tp.getShape().drop_front(), tp.getElementType());
  }
};

class VectorExtractElementOpConversion : public LLVMOpLowering {
public:
  explicit VectorExtractElementOpConversion(MLIRContext *context,
                                            LLVMTypeConverter &typeConverter)
      : LLVMOpLowering(vector::ExtractElementOp::getOperationName(), context,
                       typeConverter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto adaptor = vector::ExtractElementOpOperandAdaptor(operands);
    auto extractOp = cast<vector::ExtractElementOp>(op);
    auto vectorType = extractOp.vector()->getType().cast<VectorType>();
    auto resultType = extractOp.getResult()->getType();
    auto llvmResultType = lowering.convertType(resultType);

    auto positionArrayAttr = extractOp.position();
    // One-shot extraction of vector from array (only requires extractvalue).
    if (resultType.isa<VectorType>()) {
      Value *extracted = rewriter.create<LLVM::ExtractValueOp>(
          loc, llvmResultType, adaptor.vector(), positionArrayAttr);
      rewriter.replaceOp(op, extracted);
      return matchSuccess();
    }

    // Potential extraction of 1-D vector from struct.
    auto *context = op->getContext();
    Value *extracted = adaptor.vector();
    auto positionAttrs = positionArrayAttr.getValue();
    auto i32Type = rewriter.getIntegerType(32);
    if (positionAttrs.size() > 1) {
      auto nDVectorType = vectorType;
      auto oneDVectorType = VectorType::get(nDVectorType.getShape().take_back(),
                                            nDVectorType.getElementType());
      auto nMinusOnePositionAttrs =
          ArrayAttr::get(positionAttrs.drop_back(), context);
      extracted = rewriter.create<LLVM::ExtractValueOp>(
          loc, lowering.convertType(oneDVectorType), extracted,
          nMinusOnePositionAttrs);
    }

    // Remaining extraction of element from 1-D LLVM vector
    auto position = positionAttrs.back().cast<IntegerAttr>();
    auto constant = rewriter.create<LLVM::ConstantOp>(
        loc, lowering.convertType(i32Type), position);
    extracted =
        rewriter.create<LLVM::ExtractElementOp>(loc, extracted, constant);
    rewriter.replaceOp(op, extracted);

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
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
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
    Value *desc = rewriter.create<LLVM::UndefOp>(loc, llvmArrayOfVectType);
    Value *a = adaptor.lhs(), *b = adaptor.rhs();
    Value *acc = adaptor.acc().empty() ? nullptr : adaptor.acc().front();
    SmallVector<Value *, 8> lhs, accs;
    lhs.reserve(rankLHS);
    accs.reserve(rankLHS);
    for (unsigned d = 0, e = rankLHS; d < e; ++d) {
      // shufflevector explicitly requires i32.
      auto attr = rewriter.getI32IntegerAttr(d);
      SmallVector<Attribute, 4> bcastAttr(rankRHS, attr);
      auto bcastArrayAttr = ArrayAttr::get(bcastAttr, ctx);
      Value *aD = nullptr, *accD = nullptr;
      // 1. Broadcast the element a[d] into vector aD.
      aD = rewriter.create<LLVM::ShuffleVectorOp>(loc, a, a, bcastArrayAttr);
      // 2. If acc is present, extract 1-d vector acc[d] into accD.
      if (acc)
        accD = rewriter.create<LLVM::ExtractValueOp>(
            loc, vRHS, acc, rewriter.getI64ArrayAttr(d));
      // 3. Compute aD outer b (plus accD, if relevant).
      Value *aOuterbD =
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
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
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
    Value *allocated = sourceMemRef.allocatedPtr(rewriter, loc);
    allocated =
        rewriter.create<LLVM::BitcastOp>(loc, llvmTargetElementTy, allocated);
    desc.setAllocatedPtr(rewriter, loc, allocated);
    // Set aligned ptr.
    Value *ptr = sourceMemRef.alignedPtr(rewriter, loc);
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

/// Populate the given list with patterns that convert from Vector to LLVM.
void mlir::populateVectorToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  patterns.insert<VectorBroadcastOpConversion, VectorExtractElementOpConversion,
                  VectorOuterProductOpConversion, VectorTypeCastOpConversion>(
      converter.getDialect()->getContext(), converter);
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
