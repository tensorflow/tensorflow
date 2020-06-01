/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace xla_lhlo {
namespace {

struct StaticMemRefCastOpConverter
    : public ConvertOpToLLVMPattern<StaticMemRefCastOp> {
  using ConvertOpToLLVMPattern<StaticMemRefCastOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto cast_op = cast<StaticMemRefCastOp>(op);

    StaticMemRefCastOpOperandAdaptor operands_adaptor(operands);
    MemRefDescriptor sourceMemRef(operands_adaptor.operand());

    MemRefType targetMemRefType =
        cast_op.getResult().getType().cast<MemRefType>();
    auto llvmTargetDescriptorTy = typeConverter.convertType(targetMemRefType)
                                      .dyn_cast_or_null<LLVM::LLVMType>();
    if (!llvmTargetDescriptorTy || !llvmTargetDescriptorTy.isStructTy())
      return failure();
    // Create descriptor.
    auto desc = MemRefDescriptor::undef(rewriter, loc, llvmTargetDescriptorTy);
    Type llvmTargetElementTy = desc.getElementType();
    // Set allocated ptr.
    Value allocated = sourceMemRef.allocatedPtr(rewriter, loc);
    allocated =
        rewriter.create<LLVM::BitcastOp>(loc, llvmTargetElementTy, allocated);
    desc.setAllocatedPtr(rewriter, loc, allocated);
    // Set aligned ptr.
    Value ptr = sourceMemRef.alignedPtr(rewriter, loc);
    ptr = rewriter.create<LLVM::BitcastOp>(loc, llvmTargetElementTy, ptr);
    desc.setAlignedPtr(rewriter, loc, ptr);

    // Fill size and stride descriptors in memref.
    auto target_sizes = targetMemRefType.getShape();
    int64_t target_offset;
    llvm::SmallVector<int64_t, 4> target_strides;
    if (failed((getStridesAndOffset(targetMemRefType, target_strides,
                                    target_offset))))
      return failure();

    // Copy offset of `targetMemRef`.
    desc.setConstantOffset(rewriter, loc, target_offset);
    for (int i = 0, e = targetMemRefType.getRank(); i < e; ++i) {
      desc.setConstantSize(rewriter, loc, i, target_sizes[i]);
      desc.setConstantStride(rewriter, loc, i, target_strides[i]);
    }
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

struct DynamicMemRefCastOpConverter
    : public ConvertOpToLLVMPattern<DynamicMemRefCastOp> {
  using ConvertOpToLLVMPattern<DynamicMemRefCastOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto cast_op = cast<DynamicMemRefCastOp>(op);

    DynamicMemRefCastOpOperandAdaptor operands_adaptor(operands);
    MemRefDescriptor sourceMemRef(operands_adaptor.operand());

    MemRefType targetMemRefType =
        cast_op.getResult().getType().cast<MemRefType>();
    auto llvmTargetDescriptorTy = typeConverter.convertType(targetMemRefType)
                                      .dyn_cast_or_null<LLVM::LLVMType>();
    if (!llvmTargetDescriptorTy || !llvmTargetDescriptorTy.isStructTy())
      return failure();
    // Create descriptor.
    auto desc = MemRefDescriptor::undef(rewriter, loc, llvmTargetDescriptorTy);
    Type llvmTargetElementTy = desc.getElementType();
    // Set allocated ptr.
    Value allocated = sourceMemRef.allocatedPtr(rewriter, loc);
    allocated =
        rewriter.create<LLVM::BitcastOp>(loc, llvmTargetElementTy, allocated);
    desc.setAllocatedPtr(rewriter, loc, allocated);
    // Set aligned ptr.
    Value ptr = sourceMemRef.alignedPtr(rewriter, loc);
    ptr = rewriter.create<LLVM::BitcastOp>(loc, llvmTargetElementTy, ptr);
    desc.setAlignedPtr(rewriter, loc, ptr);
    // Copy offset of `sourceMemRef`.
    desc.setOffset(rewriter, loc, sourceMemRef.offset(rewriter, loc));

    // Fill size and stride descriptors in memref.
    if (!cast_op.sizes().empty()) {
      auto sizes = operands_adaptor.sizes();
      auto strides = operands_adaptor.strides();
      for (int i = 0, e = targetMemRefType.getRank(); i < e; ++i) {
        desc.setSize(rewriter, loc, i, sizes[i]);
        desc.setStride(rewriter, loc, i, strides[i]);
      }
    }
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

}  // namespace

void PopulateLhloToLLVMConversionPatterns(LLVMTypeConverter *converter,
                                          OwningRewritePatternList *patterns) {
  patterns->insert<DynamicMemRefCastOpConverter, StaticMemRefCastOpConverter>(
      *converter);
}

}  // namespace xla_lhlo
}  // namespace mlir
