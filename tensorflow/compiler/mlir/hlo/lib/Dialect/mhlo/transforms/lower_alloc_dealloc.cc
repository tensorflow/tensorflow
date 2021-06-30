/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"             // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"               // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"                   // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"             // from @llvm-project
#include "mlir/IR/Attributes.h"                          // from @llvm-project
#include "mlir/IR/BuiltinOps.h"                          // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                        // from @llvm-project
#include "mlir/IR/Operation.h"                           // from @llvm-project

// This file implements the logic to convert alloc/dealloc ops to disc op

namespace mlir {
namespace disc_ral {

using LLVM::LLVMFuncOp;

static constexpr const char* kMalloc = "alloc";
static constexpr const char* kFree = "free";

// A rewrite pattern to convert memref.alloc operations into corresponding
// runtime wrapper calls (modeled by ral.dispatch ops)
class ConvertMemRefAllocOpToDispatchOpPattern
    : public ConvertOpToLLVMPattern<memref::AllocOp> {
 public:
  ConvertMemRefAllocOpToDispatchOpPattern(LLVMTypeConverter& type_converter)
      : ConvertOpToLLVMPattern<memref::AllocOp>(type_converter) {}

 private:
  // TODO(disc): Remove strides computation.
  MemRefDescriptor CreateMemRefDescriptor(Location loc,
                                          ConversionPatternRewriter& rewriter,
                                          MemRefType memref_type,
                                          Value allocated_byte_ptr,
                                          ArrayRef<Value> sizes) const {
    auto memref_desc = MemRefDescriptor::undef(
        rewriter, loc, typeConverter->convertType(memref_type));

    // TF AllocateRaw returns aligned pointer => AllocatedPtr == AlignedPtr.
    Value allocated_type_ptr = rewriter.create<LLVM::BitcastOp>(
        loc, getElementPtrType(memref_type), allocated_byte_ptr);
    memref_desc.setAllocatedPtr(rewriter, loc, allocated_type_ptr);
    memref_desc.setAlignedPtr(rewriter, loc, allocated_type_ptr);
    memref_desc.setConstantOffset(rewriter, loc, 0);

    if (memref_type.getRank() == 0) {
      return memref_desc;
    }

    // Compute strides and populate descriptor `size` and `stride` fields.
    Value stride_carried = createIndexConstant(rewriter, loc, 1);
    for (int pos = sizes.size() - 1; pos >= 0; --pos) {
      Value size = sizes[pos];
      memref_desc.setSize(rewriter, loc, pos, size);
      memref_desc.setStride(rewriter, loc, pos, stride_carried);
      // Update stride
      if (pos > 0) {
        stride_carried =
            rewriter.create<LLVM::MulOp>(loc, stride_carried, size);
      }
    }
    return memref_desc;
  }
  LogicalResult matchAndRewrite(
      memref::AllocOp alloc_op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override;
};

// Emits LLVM IR to malloc a device memory.
LogicalResult ConvertMemRefAllocOpToDispatchOpPattern::matchAndRewrite(
    memref::AllocOp alloc_op, ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const {
  mlir::Operation* op = alloc_op.getOperation();
  ModuleOp module = op->getParentOfType<ModuleOp>();

  Location loc = op->getLoc();

  // check address space
  auto memref = alloc_op.getResult();
  MemRefType memref_type = memref.getType().cast<MemRefType>();
  Attribute memorySpace = memref_type.getMemorySpace();
  if (!memorySpace) {
    return failure();
  }
  if (!memorySpace.isa<StringAttr>()) {
    return failure();
  }

  // get ral context
  FuncOp std_func_op = alloc_op->getParentOfType<FuncOp>();
  LLVM::LLVMFuncOp llvm_func_op = alloc_op->getParentOfType<LLVM::LLVMFuncOp>();
  if (!std_func_op && !llvm_func_op) {
    return failure();
  }
  Value context_arg;
  if (std_func_op) {
    context_arg = std_func_op.getArgument(0);
  } else {
    context_arg = llvm_func_op.getArgument(0);
  }

  // Set all dynamic sizes to 1 and compute fake strides.
  SmallVector<Value, 4> dyn_sizes(memref_type.getNumDynamicDims(),
                                  createIndexConstant(rewriter, loc, 1));
  // Get memref descriptor sizes.
  SmallVector<Value, 4> sizes;
  SmallVector<Value, 4> strides;
  Value sizeBytes;
  getMemRefDescriptorSizes(loc, memref_type, dyn_sizes, rewriter, sizes,
                           strides, sizeBytes);

  // create dispatch op
  auto dispatch_op = rewriter.create<disc_ral::DispatchOp>(
      loc, getVoidPtrType(), context_arg, sizeBytes, kMalloc, false, "device");
  Value allocated_byte_ptr = dispatch_op.getResult(0);

  // Create the MemRef descriptor.
  MemRefDescriptor memRefDescriptor = CreateMemRefDescriptor(
      loc, rewriter, memref_type, allocated_byte_ptr, sizes);

  rewriter.replaceOp(alloc_op, {memRefDescriptor});

  return success();
}

// A rewrite pattern to convert memref.dealloc operations into corresponding
// runtime wrapper calls (modeled by ral.dispatch ops)
class ConvertMemRefDeallocOpToDispatchOpPattern
    : public ConvertOpToLLVMPattern<memref::DeallocOp> {
 public:
  ConvertMemRefDeallocOpToDispatchOpPattern(LLVMTypeConverter& type_converter)
      : ConvertOpToLLVMPattern<memref::DeallocOp>(type_converter) {}

 private:
  LogicalResult matchAndRewrite(
      memref::DeallocOp dealloc_op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override;
};

// Emits LLVM IR to malloc a device memory.
LogicalResult ConvertMemRefDeallocOpToDispatchOpPattern::matchAndRewrite(
    memref::DeallocOp dealloc_op, ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const {
  mlir::Operation* op = dealloc_op.getOperation();
  ModuleOp module = op->getParentOfType<ModuleOp>();
  Location loc = op->getLoc();

  // check address space
  MemRefType memref_type = dealloc_op.memref().getType().cast<MemRefType>();
  Attribute memorySpace = memref_type.getMemorySpace();
  if (!memorySpace) {
    return failure();
  }
  if (!memorySpace.isa<StringAttr>()) {
    return failure();
  }

  // get ral context
  FuncOp std_func_op = dealloc_op->getParentOfType<FuncOp>();
  LLVM::LLVMFuncOp llvm_func_op =
      dealloc_op->getParentOfType<LLVM::LLVMFuncOp>();
  if (!std_func_op && !llvm_func_op) {
    return failure();
  }
  Value context_arg;
  if (std_func_op) {
    context_arg = std_func_op.getArgument(0);
  } else {
    context_arg = llvm_func_op.getArgument(0);
  }

  // create dispatch op
  memref::DeallocOp::Adaptor transformed(operands);
  MemRefDescriptor memref(transformed.memref());
  Value allocated_bytes_ptr = rewriter.create<LLVM::BitcastOp>(
      loc, getVoidPtrType(), memref.allocatedPtr(rewriter, loc));
  rewriter.replaceOpWithNewOp<disc_ral::DispatchOp>(
      op, llvm::None, context_arg, allocated_bytes_ptr, kFree, false, "device");

  return success();
}

}  // namespace disc_ral
}  // namespace mlir