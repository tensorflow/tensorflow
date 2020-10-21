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
#include "mlir/IR/Attributes.h"  // from @llvm-project
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

static constexpr StringRef kCInterfaceAlloc = "_mlir_ciface_tf_alloc";
static constexpr StringRef kCInterfaceDealloc = "_mlir_ciface_tf_dealloc";

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

class TFAllocOpConverter : public ConvertToLLVMCallOpPattern<TFAllocOp> {
 public:
  using ConvertToLLVMCallOpPattern<TFAllocOp>::ConvertToLLVMCallOpPattern;

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    TFAllocOp tf_alloc_op = cast<TFAllocOp>(op);
    TFAllocOp::Adaptor transformed(operands);

    MemRefType memref_type = tf_alloc_op.getType();

    // Get memref descriptor sizes.
    SmallVector<Value, 4> sizes;
    getMemRefDescriptorSizes(loc, memref_type,
                             llvm::to_vector<4>(transformed.dyn_sizes()),
                             rewriter, sizes);
    // Get memory block size in bytes.
    Value num_bytes = getCumulativeSizeInBytes(
        loc, memref_type.getElementType(), sizes, rewriter);

    // Convert `output_index` or set it to -1 if the attribute is missing.
    LLVM::LLVMType llvmInt32Type =
        LLVM::LLVMType::getInt32Ty(rewriter.getContext());
    Value output_index = rewriter.create<LLVM::ConstantOp>(
        loc, llvmInt32Type,
        rewriter.getI32IntegerAttr(tf_alloc_op.output_index().hasValue()
                                       ? tf_alloc_op.output_index().getValue()
                                       : -1));

    // Convert `candidate_input_indices`.
    auto candidates_count_and_ptr = ConvertI32ArrayAttrToStackAllocatedArray(
        loc, tf_alloc_op.input_indices(), &rewriter);

    // Insert function call.
    FlatSymbolRefAttr tf_func_ref = getOrInsertTFFunction(rewriter, op);
    Value allocated_byte_ptr =
        rewriter
            .create<LLVM::CallOp>(
                loc, getVoidPtrType(), tf_func_ref,
                llvm::makeArrayRef({transformed.ctx(), num_bytes, output_index,
                                    candidates_count_and_ptr.first,
                                    candidates_count_and_ptr.second}))
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
    LLVMType llvm_i32_type =
        LLVM::LLVMType::getInt32Ty(getDialect().getContext());
    LLVMType llvm_i32_ptr_type = llvm_i32_type.getPointerTo();
    LLVMType llvm_void_ptr_type = getVoidPtrType();
    return LLVMType::getFunctionTy(
        llvm_void_ptr_type,
        llvm::makeArrayRef(
            {/*void* op_kernel_ctx*/ llvm_void_ptr_type,
             /*size_t num_bytes*/ getIndexType(),
             /*int32_t output_index*/ llvm_i32_type,
             /*int32_t num_candidates*/ llvm_i32_type,
             /*int32_t* candidate_input_indices*/ llvm_i32_ptr_type}),
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

  std::pair<Value, Value> ConvertI32ArrayAttrToStackAllocatedArray(
      Location loc, llvm::Optional<ArrayAttr> attr,
      ConversionPatternRewriter *rewriter) const {
    LLVMType llvm_i32_type =
        LLVM::LLVMType::getInt32Ty(getDialect().getContext());
    LLVMType llvm_i32_ptr_type = llvm_i32_type.getPointerTo();

    // If the attribute is missing or empty, set the element count to 0 and
    // return NULL.
    if (!attr.hasValue() || attr.getValue().empty()) {
      Value zero = rewriter->create<LLVM::ConstantOp>(
          loc, llvm_i32_type, rewriter->getI32IntegerAttr(0));
      Value null_ptr = rewriter->create<LLVM::NullOp>(loc, llvm_i32_ptr_type);
      return std::make_pair(zero, null_ptr);
    }

    // Allocate array to store the elements.
    auto &array_attr = attr.getValue();
    Value array_size = rewriter->create<LLVM::ConstantOp>(
        loc, llvm_i32_type, rewriter->getI32IntegerAttr(array_attr.size()));
    Value array_ptr = rewriter->create<LLVM::AllocaOp>(
        loc, llvm_i32_ptr_type, array_size, /*alignment=*/0);

    for (auto &dim : llvm::enumerate(array_attr)) {
      Value index = rewriter->create<LLVM::ConstantOp>(
          loc, llvm_i32_type, rewriter->getI32IntegerAttr(dim.index()));
      Value elem_ptr = rewriter->create<LLVM::GEPOp>(loc, llvm_i32_ptr_type,
                                                     array_ptr, index);
      Value elem = rewriter->create<LLVM::ConstantOp>(
          loc, llvm_i32_type,
          rewriter->getI32IntegerAttr(
              dim.value().cast<IntegerAttr>().getInt()));
      rewriter->create<LLVM::StoreOp>(loc, elem, elem_ptr);
    }
    return std::make_pair(array_size, array_ptr);
  }
};

class TFDeallocOpConverter : public ConvertToLLVMCallOpPattern<TFDeallocOp> {
 public:
  using ConvertToLLVMCallOpPattern<TFDeallocOp>::ConvertToLLVMCallOpPattern;

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    TFDeallocOp::Adaptor transformed(operands);
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
    return LLVM::LLVMType::getFunctionTy(getVoidType(),
                                         {getVoidPtrType(), getVoidPtrType()},
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
  patterns->insert<TFAllocOpConverter, TFDeallocOpConverter>(*converter);
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
