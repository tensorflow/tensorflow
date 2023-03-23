/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <optional>
#include <utility>

#include "deallocation/IR/deallocation_ops.h"
#include "deallocation/transforms/passes.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace deallocation {
namespace {

struct NullOpLowering : public ConvertOpToLLVMPattern<NullOp> {
  using ConvertOpToLLVMPattern<NullOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      NullOp nullOp, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = nullOp->getLoc();
    LLVMTypeConverter typeConverter = *getTypeConverter();

    auto baseMemRefType = nullOp.getType().cast<BaseMemRefType>();

    FailureOr<unsigned> addressSpaceOr =
        typeConverter.getMemRefAddressSpace(baseMemRefType);
    if (failed(addressSpaceOr)) return failure();
    unsigned addressSpace = addressSpaceOr.value();  // NOLINT

    Value zero = createIndexConstant(rewriter, loc, 0);
    if (auto resultType = nullOp.getType().dyn_cast<MemRefType>()) {
      // Set all dynamic sizes to 1 and compute fake strides.
      SmallVector<Value> dynSizes(resultType.getNumDynamicDims(),
                                  createIndexConstant(rewriter, loc, 1));
      SmallVector<Value> sizes, strides;
      Value sizeBytes;
      getMemRefDescriptorSizes(loc, resultType, dynSizes, rewriter, sizes,
                               strides, sizeBytes);

      // Prepare packed args [allocatedPtr, alignedPtr, offset, sizes, strides]
      // to create a memref descriptor.
      Value null = rewriter.create<LLVM::NullOp>(
          loc, LLVM::LLVMPointerType::get(rewriter.getContext(), addressSpace));
      SmallVector<Value> packedValues{null, null, zero};
      packedValues.append(sizes);
      packedValues.append(strides);

      rewriter.replaceOp(nullOp,
                         MemRefDescriptor::pack(rewriter, loc, typeConverter,
                                                resultType, packedValues));
      return success();
    }

    auto resultType = nullOp.getType().cast<UnrankedMemRefType>();
    Type llvmResultType = typeConverter.convertType(resultType);

    auto desc = UnrankedMemRefDescriptor::undef(rewriter, loc, llvmResultType);
    desc.setRank(rewriter, loc, zero);

    // The allocated pointer is stored in the underlying ranked memref
    // descriptor.
    SmallVector<Value, 1> sizes;
    UnrankedMemRefDescriptor::computeSizes(rewriter, loc, *getTypeConverter(),
                                           desc, addressSpace, sizes);
    Value underlyingDestPtr = rewriter.create<LLVM::AllocaOp>(
        loc, getVoidPtrType(), rewriter.getI8Type(), sizes.front());

    // Populate underlying ranked descriptor.
    LLVM::LLVMPointerType elemPtrType =
        LLVM::LLVMPointerType::get(rewriter.getContext(), addressSpace);

    Value null = rewriter.create<LLVM::NullOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext(), addressSpace));
    UnrankedMemRefDescriptor::setAllocatedPtr(rewriter, loc, underlyingDestPtr,
                                              elemPtrType, null);
    UnrankedMemRefDescriptor::setAlignedPtr(rewriter, loc, *getTypeConverter(),
                                            underlyingDestPtr, elemPtrType,
                                            null);
    UnrankedMemRefDescriptor::setOffset(rewriter, loc, *getTypeConverter(),
                                        underlyingDestPtr, elemPtrType, zero);

    desc.setMemRefDescPtr(rewriter, loc, underlyingDestPtr);
    rewriter.replaceOp(nullOp, {desc});
    return success();
  }
};

struct GetBufferOpLowering : public ConvertOpToLLVMPattern<GetBufferOp> {
  using ConvertOpToLLVMPattern<GetBufferOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      GetBufferOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto memref = adaptor.getMemref();
    Value ptr;
    if (auto unrankedTy =
            llvm::dyn_cast<UnrankedMemRefType>(op.getMemref().getType())) {
      LLVM::LLVMPointerType elementPtrTy = LLVM::LLVMPointerType::get(
          rewriter.getContext(), unrankedTy.getMemorySpaceAsInt());
      memref = UnrankedMemRefDescriptor(memref).memRefDescPtr(rewriter, loc);
      ptr = UnrankedMemRefDescriptor::allocatedPtr(rewriter, loc, memref,
                                                   elementPtrTy);
    } else {
      ptr = MemRefDescriptor(memref).allocatedPtr(rewriter, loc);
    }
    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(
        op, getTypeConverter()->getIndexType(), ptr);
    return success();
  }
};

#define GEN_PASS_DEF_CONVERTDEALLOCATIONOPSTOLLVMPASS
#include "deallocation/transforms/passes.h.inc"

struct ConvertDeallocationOpsToLLVMPass
    : public impl::ConvertDeallocationOpsToLLVMPassBase<
          ConvertDeallocationOpsToLLVMPass> {
  ConvertDeallocationOpsToLLVMPass() = default;

  void runOnOperation() override {
    Operation *func = getOperation();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(func));

    LLVMTypeConverter typeConverter(&getContext(), options,
                                    &dataLayoutAnalysis);
    RewritePatternSet patterns(&getContext());
    patterns.add<GetBufferOpLowering, NullOpLowering>(typeConverter);

    LLVMConversionTarget target(getContext());
    target.addLegalOp<func::FuncOp>();
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createConvertDeallocationOpsToLLVM() {
  return std::make_unique<ConvertDeallocationOpsToLLVMPass>();
}

}  // namespace deallocation
}  // namespace mlir
