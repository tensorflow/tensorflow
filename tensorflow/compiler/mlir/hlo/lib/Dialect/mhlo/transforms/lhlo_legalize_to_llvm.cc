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
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"

namespace mlir {
namespace lmhlo {
namespace {

struct StaticMemRefCastOpConverter
    : public ConvertOpToLLVMPattern<StaticMemRefCastOp> {
  using ConvertOpToLLVMPattern<StaticMemRefCastOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto cast_op = cast<StaticMemRefCastOp>(op);

    StaticMemRefCastOp::Adaptor operands_adaptor(operands);
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

    DynamicMemRefCastOp::Adaptor operands_adaptor(operands);
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

struct ReshapeMemRefCastOpConverter
    : public ConvertOpToLLVMPattern<ReshapeMemRefCastOp> {
  using ConvertOpToLLVMPattern<ReshapeMemRefCastOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto reshape_op = cast<ReshapeMemRefCastOp>(op);
    Type dst_type = reshape_op.getResult().getType();
    auto element_type = dst_type.cast<ShapedType>().getElementType();

    auto shape = reshape_op.shape();

    ReshapeMemRefCastOp::Adaptor operands_adaptor(operands);
    PtrsAndOffset ptrs_n_offset = ExtractMemRefPtrsAndOffset(
        loc, reshape_op.operand(), operands_adaptor.operand(), &rewriter);

    MemRefDescriptor shape_desc(operands_adaptor.shape());

    auto shape_memref_type = shape.getType().cast<MemRefType>();

    if (shape_memref_type.hasStaticShape()) {
      auto shape_length = shape_memref_type.getDimSize(0);

      MemRefType targetMemRefType = MemRefType::get(
          SmallVector<int64_t, 1>(shape_length, 1), element_type);
      auto llvmTargetDescriptorTy = typeConverter.convertType(targetMemRefType)
                                        .dyn_cast_or_null<LLVM::LLVMType>();
      if (!llvmTargetDescriptorTy || !llvmTargetDescriptorTy.isStructTy())
        return failure();
      // Create descriptor.
      auto desc =
          MemRefDescriptor::undef(rewriter, loc, llvmTargetDescriptorTy);
      desc.setAllocatedPtr(rewriter, loc, ptrs_n_offset.allocated_ptr);
      desc.setAlignedPtr(rewriter, loc, ptrs_n_offset.aligned_ptr);
      desc.setOffset(rewriter, loc, ptrs_n_offset.offset);

      auto llvmIndexTy = typeConverter.convertType(rewriter.getIndexType())
                             .cast<LLVM::LLVMType>();
      auto llvmIndexTyPtr = llvmIndexTy.getPointerTo();
      Value stride_carried = rewriter.create<LLVM::ConstantOp>(
          loc, llvmIndexTy,
          rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
      for (int i = shape_length - 1; i >= 0; --i) {
        Value pos = rewriter.create<LLVM::ConstantOp>(
            loc, llvmIndexTy,
            rewriter.getIntegerAttr(rewriter.getIndexType(), i));
        Value ptr = rewriter.create<LLVM::GEPOp>(
            loc, llvmIndexTyPtr, shape_desc.alignedPtr(rewriter, loc),
            ValueRange{pos});
        Value extracted_size = rewriter.create<LLVM::LoadOp>(loc, ptr);
        desc.setSize(rewriter, loc, i, extracted_size);
        desc.setStride(rewriter, loc, i, stride_carried);
        // Update stride
        if (i > 0) {
          stride_carried =
              rewriter.create<LLVM::MulOp>(loc, stride_carried, extracted_size);
        }
      }
      if (dst_type.isa<MemRefType>()) {
        rewriter.replaceOp(op, {desc});
      } else {
        Value rank = rewriter.create<LLVM::ConstantOp>(
            loc, llvmIndexTy,
            rewriter.getIntegerAttr(rewriter.getIndexType(), shape_length));
        Value alloca =
            typeConverter.promoteOneMemRefDescriptor(loc, desc, rewriter);
        Value void_ptr =
            rewriter.create<LLVM::BitcastOp>(loc, getVoidPtrType(), alloca);
        auto unranked_desc = UnrankedMemRefDescriptor::pack(
            rewriter, loc, typeConverter, dst_type.cast<UnrankedMemRefType>(),
            {rank, void_ptr});
        rewriter.replaceOp(op, {unranked_desc});
      }
    } else {
      /*
       * TODO(pifon, herhut):
       *   Compute strides with llvm.loop;
       *   Use UnrankedMemrefDescr::ComputeSize with Alloca;
       *   Set all the fields using getelementptr.
       */
      return failure();
    }
    return success();
  }

 private:
  struct PtrsAndOffset {
    Value allocated_ptr;
    Value aligned_ptr;
    Value offset;
  };

  PtrsAndOffset ExtractMemRefPtrsAndOffset(
      Location loc, Value originalOperand, Value convertedOperand,
      ConversionPatternRewriter *rewriter) const {
    Type operandType = originalOperand.getType();
    Value descriptor_ptr;
    if (operandType.isa<MemRefType>()) {
      descriptor_ptr = convertedOperand;
    } else {
      UnrankedMemRefDescriptor unranked_descriptor(convertedOperand);
      Value underlying_desc_ptr =
          unranked_descriptor.memRefDescPtr(*rewriter, loc);

      Type element_type =
          operandType.cast<UnrankedMemRefType>().getElementType();
      LLVM::LLVMType memref_type_0d =
          typeConverter.convertType(MemRefType::get(/*shape=*/{}, element_type))
              .cast<LLVM::LLVMType>();
      descriptor_ptr = rewriter->create<LLVM::BitcastOp>(
          loc, memref_type_0d.getPointerTo(), underlying_desc_ptr);
      descriptor_ptr = rewriter->create<LLVM::LoadOp>(loc, descriptor_ptr);
    }
    MemRefDescriptor descriptor(descriptor_ptr);
    PtrsAndOffset result;
    result.allocated_ptr = descriptor.allocatedPtr(*rewriter, loc);
    result.aligned_ptr = descriptor.alignedPtr(*rewriter, loc);
    result.offset = descriptor.offset(*rewriter, loc);
    return result;
  }
};

}  // namespace

void PopulateLhloToLLVMConversionPatterns(const LowerToLLVMOptions &options,
                                          LLVMTypeConverter *converter,
                                          OwningRewritePatternList *patterns) {
  patterns->insert<DynamicMemRefCastOpConverter, ReshapeMemRefCastOpConverter,
                   StaticMemRefCastOpConverter>(*converter, options);
}

}  // namespace lmhlo
}  // namespace mlir
