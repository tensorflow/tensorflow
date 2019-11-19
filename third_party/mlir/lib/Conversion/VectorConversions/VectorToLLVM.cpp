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
#include "mlir/Conversion/VectorConversions/VectorConversions.h"
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

class VectorExtractElementOpConversion : public LLVMOpLowering {
public:
  explicit VectorExtractElementOpConversion(MLIRContext *context,
                                            LLVMTypeConverter &typeConverter)
      : LLVMOpLowering(vector::VectorExtractElementOp::getOperationName(),
                       context, typeConverter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto adaptor = vector::VectorExtractElementOpOperandAdaptor(operands);
    auto extractOp = cast<vector::VectorExtractElementOp>(op);
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
      : LLVMOpLowering(vector::VectorOuterProductOp::getOperationName(),
                       context, typeConverter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto adaptor = vector::VectorOuterProductOpOperandAdaptor(operands);
    auto *ctx = op->getContext();
    auto vLHS = adaptor.lhs()->getType().cast<LLVM::LLVMType>();
    auto vRHS = adaptor.rhs()->getType().cast<LLVM::LLVMType>();
    auto rankLHS = vLHS.getUnderlyingType()->getVectorNumElements();
    auto rankRHS = vRHS.getUnderlyingType()->getVectorNumElements();
    auto llvmArrayOfVectType = lowering.convertType(
        cast<vector::VectorOuterProductOp>(op).getResult()->getType());
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
      : LLVMOpLowering(vector::VectorTypeCastOp::getOperationName(), context,
                       typeConverter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    vector::VectorTypeCastOp castOp = cast<vector::VectorTypeCastOp>(op);
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
  patterns.insert<VectorExtractElementOpConversion,
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
