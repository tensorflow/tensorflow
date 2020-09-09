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
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {
namespace {

using LLVM::LLVMFuncOp;
using LLVM::LLVMType;

static constexpr StringRef kCInterfaceAlloc = "_mlir_ciface_tf_alloc_raw";
static constexpr StringRef kCInterfaceDealloc = "_mlir_ciface_tf_dealloc_raw";

/// Base class for patterns converting TF Framework ops to function calls.
template <typename OpTy>
class ConvertToLLVMCallOpPattern : public ConvertOpToLLVMPattern<OpTy> {
 public:
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  // Attempts to find function symbol in the module, adds it if not found.
  FlatSymbolRefAttr getOrInsertTFFunction(PatternRewriter &rewriter,
                                          Operation *op) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    StringRef tf_func_name = GetFuncName();
    auto tf_func = module.lookupSymbol<LLVMFuncOp>(tf_func_name);
    if (!tf_func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto func_type = GetFuncType();
      tf_func = rewriter.create<LLVMFuncOp>(rewriter.getUnknownLoc(),
                                            tf_func_name, func_type);
    }
    return SymbolRefAttr::get(tf_func_name, rewriter.getContext());
  }

 protected:
  virtual StringRef GetFuncName() const = 0;
  virtual LLVMType GetFuncType() const = 0;
};

class AllocRawOpConverter : public ConvertToLLVMCallOpPattern<AllocRawOp> {
 public:
  using ConvertToLLVMCallOpPattern<AllocRawOp>::ConvertToLLVMCallOpPattern;

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    AllocRawOp alloc_raw_op = cast<AllocRawOp>(op);
    AllocRawOp::Adaptor transformed(operands);

    MemRefType memref_type = alloc_raw_op.getType();

    // Get memref descriptor sizes.
    SmallVector<Value, 4> sizes;
    getMemRefDescriptorSizes(loc, memref_type,
                             llvm::to_vector<4>(transformed.dyn_sizes()),
                             rewriter, sizes);
    // Get memory block size in bytes.
    Value num_bytes = getCumulativeSizeInBytes(
        loc, memref_type.getElementType(), sizes, rewriter);

    // Insert function call.
    FlatSymbolRefAttr tf_func_ref = getOrInsertTFFunction(rewriter, op);
    Value allocated_byte_ptr =
        rewriter
            .create<LLVM::CallOp>(
                loc, getVoidPtrType(), tf_func_ref,
                llvm::makeArrayRef({transformed.ctx(), num_bytes}))
            .getResult(0);

    MemRefDescriptor memRefDescriptor = CreateMemRefDescriptor(
        loc, rewriter, memref_type, allocated_byte_ptr, sizes);

    // Return the final value of the descriptor.
    rewriter.replaceOp(op, {memRefDescriptor});
    return success();
  }

 protected:
  StringRef GetFuncName() const override { return kCInterfaceAlloc; }

  LLVMType GetFuncType() const override {
    LLVMType llvm_void_ptr_type = getVoidPtrType();
    return LLVM::LLVMType::getFunctionTy(
        llvm_void_ptr_type,
        llvm::makeArrayRef({llvm_void_ptr_type, getIndexType()}),
        /*isVarArg=*/false);
  }

 private:
  MemRefDescriptor CreateMemRefDescriptor(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          MemRefType memref_type,
                                          Value allocated_byte_ptr,
                                          ArrayRef<Value> sizes) const {
    auto memref_desc = MemRefDescriptor::undef(
        rewriter, loc, typeConverter.convertType(memref_type));

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
};

class DeallocRawOpConverter : public ConvertToLLVMCallOpPattern<DeallocRawOp> {
 public:
  using ConvertToLLVMCallOpPattern<DeallocRawOp>::ConvertToLLVMCallOpPattern;

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    DeallocRawOp::Adaptor transformed(operands);
    MemRefDescriptor memref(transformed.memref());

    Value allocated_bytes_ptr = rewriter.create<LLVM::BitcastOp>(
        op->getLoc(), getVoidPtrType(),
        memref.allocatedPtr(rewriter, op->getLoc()));

    // Insert function call.
    FlatSymbolRefAttr tf_func_ref = getOrInsertTFFunction(rewriter, op);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, llvm::None, tf_func_ref,
        llvm::makeArrayRef({transformed.ctx(), allocated_bytes_ptr}));
    return success();
  }

 protected:
  StringRef GetFuncName() const override { return kCInterfaceDealloc; }
  LLVMType GetFuncType() const override {
    return LLVM::LLVMType::getFunctionTy(getVoidType(), getVoidPtrType(),
                                         /*isVarArg=*/false);
  }
};

class NullContextOpConverter : public ConvertOpToLLVMPattern<NullContextOp> {
 public:
  using ConvertOpToLLVMPattern<NullContextOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::NullOp>(op, getVoidPtrType());
    return success();
  }
};

}  // namespace

void PopulateTFFrameworkToLLVMConversionPatterns(
    LLVMTypeConverter *converter, OwningRewritePatternList *patterns) {
  patterns->insert<NullContextOpConverter>(*converter);
  patterns->insert<AllocRawOpConverter, DeallocRawOpConverter>(*converter);
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
