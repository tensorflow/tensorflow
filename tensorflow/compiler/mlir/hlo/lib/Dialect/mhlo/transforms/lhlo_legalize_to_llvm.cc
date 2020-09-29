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

#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/DialectConversion.h"

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
    Type llvmTargetElementTy = desc.getElementPtrType();
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
    Type llvmTargetElementTy = desc.getElementPtrType();
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
    auto dst_type = reshape_op.getResult().getType().cast<BaseMemRefType>();
    auto element_type = dst_type.getElementType();

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

      auto llvm_index_type = typeConverter.getIndexType();
      auto llvm_index_ptr_type = llvm_index_type.getPointerTo();
      Value stride_carried = rewriter.create<LLVM::ConstantOp>(
          loc, llvm_index_type,
          rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
      for (int i = shape_length - 1; i >= 0; --i) {
        Value pos = rewriter.create<LLVM::ConstantOp>(
            loc, llvm_index_type,
            rewriter.getIntegerAttr(rewriter.getIndexType(), i));
        Value ptr = rewriter.create<LLVM::GEPOp>(
            loc, llvm_index_ptr_type, shape_desc.alignedPtr(rewriter, loc),
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
            loc, llvm_index_type,
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
      return success();
    }

    // The shape is a rank-1 tensor with unknown length.
    Value result_rank = shape_desc.size(rewriter, loc, 0);
    // TODO(herhut): Propely handle address spaces.
    unsigned address_space = 0;
    auto target_type =
        typeConverter
            .convertType(UnrankedMemRefType::get(element_type, address_space))
            .cast<LLVM::LLVMType>();
    // Create the unranked memref descriptor that holds the ranked one. The
    // inner descriptor is allocated on stack.
    UnrankedMemRefDescriptor target_desc =
        UnrankedMemRefDescriptor::undef(rewriter, loc, target_type);
    target_desc.setRank(rewriter, loc, result_rank);
    SmallVector<Value, 1> sizes;
    UnrankedMemRefDescriptor::computeSizes(rewriter, loc, typeConverter,
                                           {target_desc}, sizes);
    auto void_ptr_type = LLVM::LLVMType::getInt8PtrTy(rewriter.getContext());
    Value ranked_desc_mem = rewriter.create<LLVM::AllocaOp>(
        loc, void_ptr_type, sizes.front(), llvm::None);
    target_desc.setMemRefDescPtr(rewriter, loc, ranked_desc_mem);

    // Fill the fixed parts. For this, we cast to a 0-D memref.
    auto zero_d_memref_type = MemRefType::get({}, element_type);
    Value as_zero_d = rewriter.create<LLVM::BitcastOp>(
        loc,
        typeConverter.convertType(zero_d_memref_type)
            .cast<LLVM::LLVMType>()
            .getPointerTo(address_space),
        ranked_desc_mem);
    // Some common constants. Use 32 bit where required by gep struct indexes.
    auto int32_type = typeConverter.convertType(rewriter.getI32Type());
    Value zero_index = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter.getIndexType(), rewriter.getIndexAttr(0));
    Value zero = rewriter.create<LLVM::ConstantOp>(
        loc, int32_type, rewriter.getI32IntegerAttr(0));
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, int32_type, rewriter.getI32IntegerAttr(1));
    Value two = rewriter.create<LLVM::ConstantOp>(
        loc, int32_type, rewriter.getI32IntegerAttr(2));
    // Set base_pointer and aligned pointer.
    auto element_ptr_ptr_type = typeConverter.convertType(element_type)
                                    .cast<LLVM::LLVMType>()
                                    .getPointerTo(address_space)
                                    .getPointerTo(address_space);
    auto base_gep = rewriter.create<LLVM::GEPOp>(
        loc, element_ptr_ptr_type, as_zero_d, ValueRange({zero_index, zero}));
    rewriter.create<LLVM::StoreOp>(loc, ptrs_n_offset.allocated_ptr, base_gep);
    auto aligned_gep = rewriter.create<LLVM::GEPOp>(
        loc, element_ptr_ptr_type, as_zero_d, ValueRange({zero_index, one}));
    rewriter.create<LLVM::StoreOp>(loc, ptrs_n_offset.aligned_ptr, aligned_gep);
    // Set offset.
    auto index_ptr_type =
        typeConverter.getIndexType().getPointerTo(address_space);
    auto offset_gep = rewriter.create<LLVM::GEPOp>(
        loc, index_ptr_type, as_zero_d, ValueRange({zero_index, two}));
    rewriter.create<LLVM::StoreOp>(loc, ptrs_n_offset.offset, offset_gep);

    // Use the offset pointer as base for further addressing. Copy over the
    // new shape and compute strides. For this, we need to create a loop from
    // rank - 1 to 0.
    Value one_index = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter.getIndexType(), rewriter.getIndexAttr(1));
    auto target_shape_base = rewriter.create<LLVM::GEPOp>(
        loc, index_ptr_type, offset_gep, ValueRange({one}));
    auto target_strides_base = rewriter.create<LLVM::GEPOp>(
        loc, index_ptr_type, target_shape_base, ValueRange({result_rank}));
    auto shape_ptr = shape_desc.alignedPtr(rewriter, loc);
    auto result_rank_minus_one =
        rewriter.create<LLVM::SubOp>(loc, result_rank, one_index);

    Block *init_block = rewriter.getInsertionBlock();
    Block *cond_block =
        rewriter.splitBlock(init_block, rewriter.getInsertionPoint());
    rewriter.setInsertionPointToEnd(init_block);
    rewriter.create<LLVM::BrOp>(
        loc, ValueRange({result_rank_minus_one, one_index}), cond_block);
    rewriter.setInsertionPointToStart(cond_block);
    auto index_arg = cond_block->addArgument(typeConverter.getIndexType());
    auto stride_arg = cond_block->addArgument(typeConverter.getIndexType());
    auto pred = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::LLVMType::getInt1Ty(rewriter.getContext()),
        LLVM::ICmpPredicate::sge, index_arg, zero_index);

    Block *body_block =
        rewriter.splitBlock(cond_block, rewriter.getInsertionPoint());
    rewriter.setInsertionPointToStart(body_block);

    // Copy size from shape to descriptor.
    auto size_load_gep = rewriter.create<LLVM::GEPOp>(
        loc, index_ptr_type, shape_ptr, ValueRange{index_arg});
    auto extracted_size = rewriter.create<LLVM::LoadOp>(loc, size_load_gep);
    auto size_store_gep = rewriter.create<LLVM::GEPOp>(
        loc, index_ptr_type, target_shape_base, ValueRange({index_arg}));
    rewriter.create<LLVM::StoreOp>(loc, extracted_size, size_store_gep);
    // Write stride value and compute next one.
    auto stride_store_gep = rewriter.create<LLVM::GEPOp>(
        loc, index_ptr_type, target_strides_base, ValueRange({index_arg}));
    rewriter.create<LLVM::StoreOp>(loc, stride_arg, stride_store_gep);
    auto next_stride =
        rewriter.create<LLVM::MulOp>(loc, stride_arg, extracted_size);

    // Decrement loop counter and branch back.
    auto decrement = rewriter.create<LLVM::SubOp>(loc, index_arg, one_index);
    rewriter.create<LLVM::BrOp>(loc, ValueRange({decrement, next_stride}),
                                cond_block);

    Block *remainder =
        rewriter.splitBlock(body_block, rewriter.getInsertionPoint());

    // Hook up the cond exit to the remainder.
    rewriter.setInsertionPointToEnd(cond_block);
    rewriter.create<LLVM::CondBrOp>(loc, pred, body_block, ValueRange(),
                                    remainder, ValueRange());

    // Reset position to beginning of new remainder block.
    rewriter.setInsertionPointToStart(remainder);
    rewriter.replaceOp(op, {target_desc});
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

void PopulateLhloToLLVMConversionPatterns(LLVMTypeConverter *converter,
                                          OwningRewritePatternList *patterns) {
  patterns->insert<DynamicMemRefCastOpConverter, ReshapeMemRefCastOpConverter,
                   StaticMemRefCastOpConverter>(*converter);
}

}  // namespace lmhlo
}  // namespace mlir
