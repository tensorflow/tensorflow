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

#include "llvm/Support/FormatVariadic.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {
namespace {

using LLVM::LLVMFuncOp;

static constexpr StringRef kCInterfaceAlloc = "_mlir_ciface_tf_alloc";
static constexpr StringRef kCInterfaceDealloc = "_mlir_ciface_tf_dealloc";
static constexpr StringRef kCInterfaceReportError =
    "_mlir_ciface_tf_report_error";

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
    return SymbolRefAttr::get(rewriter.getContext(), tf_func_name);
  }

 protected:
  virtual StringRef GetFuncName() const = 0;
  virtual Type GetFuncType() const = 0;
};

class TFAllocOpConverter : public ConvertToLLVMCallOpPattern<TFAllocOp> {
 public:
  using ConvertToLLVMCallOpPattern<TFAllocOp>::ConvertToLLVMCallOpPattern;

  LogicalResult matchAndRewrite(
      TFAllocOp tf_alloc_op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    mlir::Operation *op = tf_alloc_op.getOperation();
    Location loc = op->getLoc();
    TFAllocOp::Adaptor transformed(operands);

    MemRefType memref_type = tf_alloc_op.getType();

    // Get memref descriptor sizes.
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memref_type,
                             llvm::to_vector<4>(transformed.dyn_sizes()),
                             rewriter, sizes, strides, sizeBytes);
    // Get number of elements.
    Value num_elements = getNumElements(loc, sizes, rewriter);
    // Get element size.
    Value element_size =
        getSizeInBytes(loc, memref_type.getElementType(), rewriter);

    // Convert `output_index` or set it to -1 if the attribute is missing.
    Type llvmInt32Type = IntegerType::get(rewriter.getContext(), 32);
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
                llvm::makeArrayRef({transformed.ctx(), num_elements,
                                    element_size, output_index,
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

  Type GetFuncType() const override {
    Type llvm_i32_type = IntegerType::get(getDialect().getContext(), 32);
    Type llvm_i32_ptr_type = LLVM::LLVMPointerType::get(llvm_i32_type);
    Type llvm_void_ptr_type = getVoidPtrType();
    return LLVM::LLVMFunctionType::get(
        llvm_void_ptr_type,
        llvm::makeArrayRef(
            {/*void* op_kernel_ctx*/ llvm_void_ptr_type,
             /*size_t num_elements*/ getIndexType(),
             /*size_t element_size*/ getIndexType(),
             /*int32_t output_index*/ llvm_i32_type,
             /*int32_t num_candidates*/ llvm_i32_type,
             /*int32_t* candidate_input_indices*/ llvm_i32_ptr_type}));
  }

 private:
  MemRefDescriptor CreateMemRefDescriptor(Location loc,
                                          ConversionPatternRewriter &rewriter,
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

  std::pair<Value, Value> ConvertI32ArrayAttrToStackAllocatedArray(
      Location loc, llvm::Optional<ArrayAttr> attr,
      ConversionPatternRewriter *rewriter) const {
    Type llvm_i32_type = IntegerType::get(getDialect().getContext(), 32);
    Type llvm_i32_ptr_type = LLVM::LLVMPointerType::get(llvm_i32_type);

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
      TFDeallocOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    TFDeallocOp::Adaptor transformed(operands);
    MemRefDescriptor memref(transformed.memref());

    Value allocated_bytes_ptr = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(), getVoidPtrType(),
        memref.allocatedPtr(rewriter, op.getLoc()));

    // Insert function call.
    FlatSymbolRefAttr tf_func_ref = getOrInsertTFFunction(rewriter, op);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, llvm::None, tf_func_ref,
        llvm::makeArrayRef({transformed.ctx(), allocated_bytes_ptr}));
    return success();
  }

 protected:
  StringRef GetFuncName() const override { return kCInterfaceDealloc; }
  Type GetFuncType() const override {
    return LLVM::LLVMFunctionType::get(getVoidType(),
                                       {getVoidPtrType(), getVoidPtrType()});
  }
};

class ReportErrorOpConverter
    : public ConvertToLLVMCallOpPattern<ReportErrorOp> {
 public:
  using ConvertToLLVMCallOpPattern<ReportErrorOp>::ConvertToLLVMCallOpPattern;

  LogicalResult matchAndRewrite(
      ReportErrorOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ReportErrorOp::Adaptor transformed(operands,
                                       op.getOperation()->getAttrDictionary());

    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    Value message_constant = GenerateErrorMessageConstant(
        loc, module, transformed.msg().getValue(), rewriter);

    // Insert function call.
    FlatSymbolRefAttr tf_func_ref = getOrInsertTFFunction(rewriter, op);
    Value error_code = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter->convertType(rewriter.getI32Type()),
        transformed.error_code());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(

        op, llvm::None, tf_func_ref,
        llvm::makeArrayRef({transformed.ctx(), error_code, message_constant}));
    return success();
  }

 protected:
  StringRef GetFuncName() const override { return kCInterfaceReportError; }
  Type GetFuncType() const override {
    MLIRContext *ctx = &getTypeConverter()->getContext();
    auto i8_ptr_type = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
    auto i32_type = IntegerType::get(ctx, 32);
    return LLVM::LLVMFunctionType::get(
        getVoidType(), {getVoidPtrType(), i32_type, i8_ptr_type});
  }

 private:
  // Generates an LLVM IR dialect global that contains the name of the given
  // kernel function as a C string, and returns a pointer to its beginning.
  Value GenerateErrorMessageConstant(Location loc, Operation *module,
                                     StringRef message,
                                     OpBuilder &builder) const {
    std::string loc_str;
    llvm::raw_string_ostream loc_stream(loc_str);
    loc_stream << message << " at ";
    loc.print(loc_stream);

    StringRef generated_error(loc_stream.str().c_str());

    std::string global_name =
        llvm::formatv("error_message_{0}", llvm::hash_value(generated_error));

    Operation *global_constant =
        SymbolTable::lookupNearestSymbolFrom(module, global_name);

    if (global_constant) {
      Value globalPtr = builder.create<LLVM::AddressOfOp>(
          loc, cast<LLVM::GlobalOp>(global_constant));

      MLIRContext *ctx = &getTypeConverter()->getContext();
      Value c0 = builder.create<LLVM::ConstantOp>(
          loc, IntegerType::get(ctx, 64),
          builder.getIntegerAttr(builder.getIndexType(), 0));
      return builder.create<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8)), globalPtr,
          ValueRange{c0, c0});
    }
    return LLVM::createGlobalString(loc, builder, global_name, generated_error,
                                    LLVM::Linkage::Internal);
  }
};

class NullContextOpConverter : public ConvertOpToLLVMPattern<NullContextOp> {
 public:
  using ConvertOpToLLVMPattern<NullContextOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      NullContextOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::NullOp>(op, getVoidPtrType());
    return success();
  }
};

class NullMemRefOpConverter : public ConvertOpToLLVMPattern<NullMemRefOp> {
 public:
  using ConvertOpToLLVMPattern<NullMemRefOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      NullMemRefOp null_memref_op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    mlir::Operation *op = null_memref_op.getOperation();

    Location loc = op->getLoc();
    auto result_type = null_memref_op.getType().cast<UnrankedMemRefType>();
    Type llvm_result_type = typeConverter->convertType(result_type);

    auto desc =
        UnrankedMemRefDescriptor::undef(rewriter, loc, llvm_result_type);
    Value zero = createIndexConstant(rewriter, loc, 0);
    desc.setRank(rewriter, loc, zero);

    // Due to the current way of handling unranked memref results escaping, we
    // have to actually construct a ranked underlying descriptor instead of just
    // setting its pointer to NULL.
    SmallVector<Value, 4> sizes;
    UnrankedMemRefDescriptor::computeSizes(rewriter, loc, *getTypeConverter(),
                                           desc, sizes);
    Value underlying_desc_ptr = rewriter.create<LLVM::AllocaOp>(
        loc, getVoidPtrType(), sizes.front(), llvm::None);

    // Populate underlying ranked descriptor.
    unsigned address_space = result_type.getMemorySpaceAsInt();
    Type elem_type = result_type.getElementType();
    Type llvm_elem_type = typeConverter->convertType(elem_type);
    Type elem_ptr_ptr_type = LLVM::LLVMPointerType::get(
        LLVM::LLVMPointerType::get(llvm_elem_type, address_space));

    auto nullPtr = rewriter.create<LLVM::NullOp>(
        loc, LLVM::LLVMPointerType::get(llvm_elem_type, address_space));
    UnrankedMemRefDescriptor::setAllocatedPtr(
        rewriter, loc, underlying_desc_ptr, elem_ptr_ptr_type, nullPtr);
    UnrankedMemRefDescriptor::setAlignedPtr(rewriter, loc, *getTypeConverter(),
                                            underlying_desc_ptr,
                                            elem_ptr_ptr_type, nullPtr);
    UnrankedMemRefDescriptor::setOffset(rewriter, loc, *getTypeConverter(),
                                        underlying_desc_ptr, elem_ptr_ptr_type,
                                        zero);

    desc.setMemRefDescPtr(rewriter, loc, underlying_desc_ptr);
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

}  // namespace

void PopulateTFFrameworkToLLVMConversionPatterns(
    LLVMTypeConverter *converter, OwningRewritePatternList *patterns) {
  // clang-format off
  patterns->insert<
      NullContextOpConverter,
      NullMemRefOpConverter,
      ReportErrorOpConverter,
      TFAllocOpConverter,
      TFDeallocOpConverter
    >(*converter);
  // clang-format on
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
