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
#include <utility>

#include "deallocation/IR/deallocation_ops.h"
#include "deallocation/transforms/passes.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace deallocation {
namespace {

struct NullOpLowering : public ConvertOpToLLVMPattern<NullOp> {
  using ConvertOpToLLVMPattern<NullOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      NullOp nullOp, OpAdaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::NullOp>(
        nullOp, LLVM::LLVMPointerType::get(rewriter.getContext(), 0));
    return success();
  }
};

struct OwnOpLowering : public ConvertOpToLLVMPattern<OwnOp> {
  using ConvertOpToLLVMPattern<OwnOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      OwnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, MemRefDescriptor(adaptor.getMemref())
                               .allocatedPtr(rewriter, op->getLoc()));
    return success();
  }
};

struct GetBufferOpLowering : public ConvertOpToLLVMPattern<GetBufferOp> {
  using ConvertOpToLLVMPattern<GetBufferOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      GetBufferOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (op.getAlloc().getType().isa<OwnershipIndicatorType>()) {
      rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(
          op, getTypeConverter()->getIndexType(), adaptor.getAlloc());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(
          op, getTypeConverter()->getIndexType(),
          MemRefDescriptor(adaptor.getAlloc())
              .allocatedPtr(rewriter, op->getLoc()));
    }
    return success();
  }
};

struct FreeOpLowering : public ConvertOpToLLVMPattern<FreeOp> {
  using ConvertOpToLLVMPattern<FreeOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      FreeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto freeFn =
        LLVM::lookupOrCreateFreeFn(op->getParentOfType<ModuleOp>(),
                                   getTypeConverter()->useOpaquePointers());

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, freeFn, adaptor.getAlloc());
    return success();
  }
};

struct RetainOpLowering : public ConvertOpToLLVMPattern<RetainOp> {
  using ConvertOpToLLVMPattern<RetainOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      RetainOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(op.getContext());
    rewriter.setInsertionPoint(op);
    auto alloca = rewriter.create<memref::AllocaScopeOp>(
        loc, SmallVector<Type>(op->getNumResults(), ptrTy));
    auto& body = alloca.getBodyRegion().emplaceBlock();
    rewriter.setInsertionPoint(&body, body.begin());

    auto i64Ty = rewriter.getI64Type();
    auto ptrPtrTy = getTypeConverter()->getPointerType(ptrTy);
    auto getBuffers = [&](ValueRange values) {
      auto ret = rewriter.create<LLVM::AllocaOp>(
          loc, ptrPtrTy, ptrTy,
          createIndexConstant(rewriter, loc,
                              values.size() *
                                  getTypeConverter()->getPointerBitwidth() /
                                  CHAR_BIT));
      for (auto [index, value] : llvm::enumerate(values)) {
        auto ptr = rewriter.create<LLVM::GEPOp>(
            loc, ptrPtrTy, ptrTy, ret,
            createIndexConstant(rewriter, loc, index));
        rewriter.create<LLVM::StoreOp>(loc, value, ptr);
      }
      return ret;
    };

    Value numAllocs = createIndexConstant(rewriter, loc, op.getAllocs().size());
    Value allocBuffers = getBuffers(adaptor.getAllocs());
    Value numRetained =
        createIndexConstant(rewriter, loc, op.getRetained().size());
    Value retainedBuffers = getBuffers(adaptor.getRetained());

    auto retainFn =
        LLVM::lookupOrCreateFn(op->getParentOfType<ModuleOp>(), "retainBuffers",
                               {i64Ty, ptrPtrTy, i64Ty, ptrPtrTy},
                               LLVM::LLVMVoidType::get(op->getContext()));
    rewriter.create<LLVM::CallOp>(
        loc, retainFn,
        ValueRange{numAllocs, allocBuffers, numRetained, retainedBuffers});

    SmallVector<Value> results;
    for (auto index : llvm::seq<size_t>(0, op.getRetained().size())) {
      auto ptr = rewriter.create<LLVM::GEPOp>(
          loc, ptrPtrTy, ptrTy, retainedBuffers,
          createIndexConstant(rewriter, loc, index));
      results.push_back(rewriter.create<LLVM::LoadOp>(loc, ptrTy, ptr));
    }
    rewriter.create<memref::AllocaScopeReturnOp>(loc, results);
    llvm::errs() << "\n\n" << alloca << "\n";

    rewriter.replaceOp(op, alloca->getResults());
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
    Operation* func = getOperation();
    const auto& dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(func));

    LLVMTypeConverter typeConverter(&getContext(), options,
                                    &dataLayoutAnalysis);
    RewritePatternSet patterns(&getContext());
    populateDeallocationToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(getContext());
    target.addLegalOp<func::FuncOp>();
    target.addLegalOp<memref::AllocaScopeOp, memref::AllocaScopeReturnOp>();
    target.addIllegalOp<OwnOp, FreeOp, GetBufferOp, NullOp, RetainOp>();
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

void populateDeallocationToLLVMConversionPatterns(LLVMTypeConverter& converter,
                                                  RewritePatternSet& patterns) {
  converter.addConversion([&](OwnershipIndicatorType) {
    return LLVM::LLVMPointerType::get(&converter.getContext());
  });
  patterns.add<OwnOpLowering, FreeOpLowering, GetBufferOpLowering,
               NullOpLowering, RetainOpLowering>(converter);
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createConvertDeallocationOpsToLLVM() {
  return std::make_unique<ConvertDeallocationOpsToLLVMPass>();
}

}  // namespace deallocation
}  // namespace mlir
