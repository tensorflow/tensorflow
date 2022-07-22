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

#include <string>

#include "mlir/Conversion/LLVMCommon/Pattern.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/utils.h"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {
namespace {

using transforms::CreateOrFindGlobalStringConstant;
using transforms::GetGlobalName;
using transforms::GetOrInsertLLVMFunction;

static constexpr StringRef kCInterfaceAlloc = "_mlir_ciface_tf_alloc";
static constexpr StringRef kCInterfaceDealloc = "_mlir_ciface_tf_dealloc";
static constexpr StringRef kCInterfaceReportError =
    "_mlir_ciface_tf_report_error";
static constexpr StringRef kCInterfaceJITCompile =
    "_mlir_ciface_tf_jit_compile";
static constexpr StringRef kCInterfaceJITExecute =
    "_mlir_ciface_tf_jit_execute";
static constexpr StringRef kJITCodeGlobalBaseName = "jit_module_code";
static constexpr StringRef kErrorMessageGlobalBaseName = "error_message";

/// Base class for patterns converting TF Framework ops to function calls.
template <typename OpTy>
class ConvertToLLVMCallOpPattern : public ConvertOpToLLVMPattern<OpTy> {
 public:
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

 protected:
  virtual StringRef GetFuncName() const = 0;
  virtual Type GetFuncType() const = 0;

  std::pair<Value, Value> ConvertArrayAttrToStackAllocatedArray(
      Location loc, Type size_ty, Type element_ty,
      llvm::Optional<ArrayAttr> attr, ConversionPatternRewriter *rewriter,
      std::function<Value(Attribute)> create_element) const {
    Type element_ptr_ty = LLVM::LLVMPointerType::get(element_ty);

    // If the attribute is missing or empty, set the element count to 0 and
    // return NULL.
    if (!attr.has_value() || attr.getValue().empty()) {
      Value zero = rewriter->create<LLVM::ConstantOp>(
          loc, size_ty, rewriter->getIntegerAttr(size_ty, 0));
      Value null_ptr = rewriter->create<LLVM::NullOp>(loc, element_ptr_ty);
      return std::make_pair(zero, null_ptr);
    }

    // Allocate array to store the elements.
    auto &array_attr = attr.getValue();
    Value array_size = rewriter->create<LLVM::ConstantOp>(
        loc, size_ty, rewriter->getIntegerAttr(size_ty, array_attr.size()));
    Value array_ptr = rewriter->create<LLVM::AllocaOp>(
        loc, element_ptr_ty, array_size, /*alignment=*/0);
    for (auto &e : llvm::enumerate(array_attr)) {
      Value index = rewriter->create<LLVM::ConstantOp>(
          loc, size_ty, rewriter->getIntegerAttr(size_ty, e.index()));
      Value element_ptr =
          rewriter->create<LLVM::GEPOp>(loc, element_ptr_ty, array_ptr, index);
      Value element = create_element(e.value());
      rewriter->create<LLVM::StoreOp>(loc, element, element_ptr);
    }
    return std::make_pair(array_size, array_ptr);
  }

  std::pair<Value, Value> ConvertIntegerArrayAttrToStackAllocatedArray(
      Location loc, Type size_ty, Type element_ty,
      llvm::Optional<ArrayAttr> attr,
      ConversionPatternRewriter *rewriter) const {
    assert(size_ty.isa<IntegerType>() && "expect integer size type");
    assert(element_ty.isa<IntegerType>() && "expect integer element type");
    return ConvertArrayAttrToStackAllocatedArray(
        loc, size_ty, element_ty, attr, rewriter, [&](Attribute attr) {
          return rewriter->create<LLVM::ConstantOp>(
              loc, element_ty,
              rewriter->getIntegerAttr(element_ty,
                                       attr.cast<IntegerAttr>().getInt()));
        });
  }
};

class TFAllocOpConverter : public ConvertToLLVMCallOpPattern<TFAllocOp> {
 public:
  using ConvertToLLVMCallOpPattern<TFAllocOp>::ConvertToLLVMCallOpPattern;

  LogicalResult matchAndRewrite(
      TFAllocOp tf_alloc_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    mlir::Operation *op = tf_alloc_op.getOperation();
    Location loc = op->getLoc();
    MemRefType memref_type = tf_alloc_op.getType();

    // Get memref descriptor sizes.
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memref_type,
                             llvm::to_vector<4>(adaptor.dyn_sizes()), rewriter,
                             sizes, strides, sizeBytes);
    // Get number of elements.
    Value num_elements = getNumElements(loc, sizes, rewriter);
    // Get element size.
    Value element_size =
        getSizeInBytes(loc, memref_type.getElementType(), rewriter);

    // Convert `output_index` or set it to -1 if the attribute is missing.
    Type llvmInt32Type = IntegerType::get(rewriter.getContext(), 32);
    Value output_index = rewriter.create<LLVM::ConstantOp>(
        loc, llvmInt32Type,
        rewriter.getI32IntegerAttr(tf_alloc_op.output_index().has_value()
                                       ? tf_alloc_op.output_index().getValue()
                                       : -1));

    // Convert `candidate_input_indices`.
    auto candidates_count_and_ptr =
        ConvertIntegerArrayAttrToStackAllocatedArray(
            loc, rewriter.getI32Type(), rewriter.getI32Type(),
            tf_alloc_op.input_indices(), &rewriter);

    // Insert function call.
    FlatSymbolRefAttr tf_func_ref =
        GetOrInsertLLVMFunction(GetFuncName(), GetFuncType(), op, &rewriter);
    Value allocated_byte_ptr =
        rewriter
            .create<LLVM::CallOp>(
                loc, getVoidPtrType(), tf_func_ref,
                llvm::makeArrayRef({adaptor.ctx(), num_elements, element_size,
                                    output_index,
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
  // TODO(pifon): Remove strides computation.
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
};

class TFDeallocOpConverter : public ConvertToLLVMCallOpPattern<TFDeallocOp> {
 public:
  using ConvertToLLVMCallOpPattern<TFDeallocOp>::ConvertToLLVMCallOpPattern;

  LogicalResult matchAndRewrite(
      TFDeallocOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(herhut) Support unranked memrefs.
    if (!op.memref().getType().isa<MemRefType>()) return failure();
    MemRefDescriptor memref(adaptor.memref());

    Value allocated_bytes_ptr = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(), getVoidPtrType(),
        memref.allocatedPtr(rewriter, op.getLoc()));

    // Insert function call.
    FlatSymbolRefAttr tf_func_ref =
        GetOrInsertLLVMFunction(GetFuncName(), GetFuncType(), op, &rewriter);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, llvm::None, tf_func_ref,
        llvm::makeArrayRef({adaptor.ctx(), allocated_bytes_ptr}));
    return success();
  }

 protected:
  StringRef GetFuncName() const override { return kCInterfaceDealloc; }
  Type GetFuncType() const override {
    return LLVM::LLVMFunctionType::get(getVoidType(),
                                       {getVoidPtrType(), getVoidPtrType()});
  }
};

class JITCompileFromStrOpConverter
    : public ConvertToLLVMCallOpPattern<JITCompileFromStrOp> {
  using ConvertToLLVMCallOpPattern<
      JITCompileFromStrOp>::ConvertToLLVMCallOpPattern;

  LogicalResult matchAndRewrite(
      JITCompileFromStrOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (adaptor.ctx() == nullptr) return failure();
    auto loc = op.getLoc();
    std::string zero_terminated_code = op.code().str() + '\00';
    Value jit_module_code = CreateOrFindGlobalStringConstant(
        loc, GetGlobalName(kJITCodeGlobalBaseName, zero_terminated_code),
        zero_terminated_code, &rewriter);
    std::pair<Value, Value> tile_sizes =
        ConvertIntegerArrayAttrToStackAllocatedArray(loc, rewriter.getI64Type(),
                                                     rewriter.getI64Type(),
                                                     op.tileSizes(), &rewriter);
    std::pair<Value, Value> unroll_factors =
        ConvertIntegerArrayAttrToStackAllocatedArray(
            loc, rewriter.getI64Type(), rewriter.getI64Type(),
            op.unrollFactors(), &rewriter);
    Value max_supported_rank = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), op.maxSupportedRankAttr());
    Value enable_ftz = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), op.enableFtzAttr());
    Value index_64bit = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), op.index64BitAttr());
    Value cpu_codegen = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), op.cpuCodegenAttr());
    FlatSymbolRefAttr tf_func_ref =
        GetOrInsertLLVMFunction(GetFuncName(), GetFuncType(), op, &rewriter);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, getVoidPtrType(), tf_func_ref,
        llvm::makeArrayRef({adaptor.ctx(), jit_module_code, tile_sizes.first,
                            tile_sizes.second, unroll_factors.first,
                            unroll_factors.second, max_supported_rank,
                            enable_ftz, index_64bit, cpu_codegen}));
    return success();
  }

 protected:
  StringRef GetFuncName() const override { return kCInterfaceJITCompile; }

  Type GetFuncType() const override {
    auto i8_ptr_ty =
        LLVM::LLVMPointerType::get(IntegerType::get(getContext(), 8));
    auto i64_ty = IntegerType::get(getContext(), 64);
    Type i64_ptr_ty = LLVM::LLVMPointerType::get(i64_ty);
    auto i1_ty = IntegerType::get(getContext(), 1);
    return LLVM::LLVMFunctionType::get(
        getVoidPtrType(), {/*void* op_kernel_ctx*/ getVoidPtrType(),
                           /*char* code*/ i8_ptr_ty,
                           /*int64_t num_tile_sizes*/ i64_ty,
                           /*int64_t* tile_sizes_ptr*/ i64_ptr_ty,
                           /*int64_t num_unroll_factors*/ i64_ty,
                           /*int64_t* unroll_factors_ptr*/ i64_ptr_ty,
                           /*int64_t max_supported_rank*/ i64_ty,
                           /*bool enable_ftz*/ i1_ty,
                           /*bool index_64bit*/ i1_ty,
                           /*bool cpu_codegen*/ i1_ty});
  }
};

class JITExecuteOpConverter : public ConvertToLLVMCallOpPattern<JITExecuteOp> {
 public:
  using ConvertToLLVMCallOpPattern<JITExecuteOp>::ConvertToLLVMCallOpPattern;

  LogicalResult matchAndRewrite(
      JITExecuteOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // The TF context must be known for a successful lowering.
    if (adaptor.ctx() == nullptr || op.operands().empty()) {
      return failure();
    }

    // Allocate result on stack.
    auto loc = op.getLoc();
    Type result_ty =
        getTypeConverter()->convertType(op->getResultTypes().front());
    Type result_ptr_ty = LLVM::LLVMPointerType::get(result_ty);
    Type i64_ty = rewriter.getI64Type();
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, i64_ty, rewriter.getI64IntegerAttr(1));
    auto result_ptr =
        rewriter.create<LLVM::AllocaOp>(loc, result_ptr_ty, one, llvm::None);
    Type void_ptr_ty = getVoidPtrType();
    auto result_void_ptr =
        rewriter.create<LLVM::BitcastOp>(loc, void_ptr_ty, result_ptr);

    // Pass the buffer arguments as a stack-allocated array.
    Type arg_ptr_ty =
        LLVM::LLVMPointerType::get(adaptor.operands().front().getType());
    Value num_args = rewriter.create<LLVM::ConstantOp>(
        loc, i64_ty,
        rewriter.getI64IntegerAttr(
            static_cast<int64_t>(adaptor.operands().size())));
    Value args_ptr = rewriter.create<LLVM::AllocaOp>(loc, arg_ptr_ty, num_args,
                                                     /*alignment=*/0);
    for (const auto &it : llvm::enumerate(adaptor.operands())) {
      Value index = rewriter.create<LLVM::ConstantOp>(
          loc, i64_ty, rewriter.getI64IntegerAttr(it.index()));
      Value element_ptr =
          rewriter.create<LLVM::GEPOp>(loc, arg_ptr_ty, args_ptr, index);
      rewriter.create<LLVM::StoreOp>(loc, it.value(), element_ptr);
    }
    auto args_void_ptr =
        rewriter.create<LLVM::BitcastOp>(loc, void_ptr_ty, args_ptr);

    // Materialize runtime call.
    FlatSymbolRefAttr tf_func_ref =
        GetOrInsertLLVMFunction(GetFuncName(), GetFuncType(), op, &rewriter);
    rewriter.create<LLVM::CallOp>(
        loc, llvm::None, tf_func_ref,
        ValueRange{adaptor.ctx(), adaptor.callable(), result_void_ptr, num_args,
                   args_void_ptr});

    // Copy result (including the descriptor) to a stack-allocated buffer and
    // free the old descriptor.
    llvm::SmallVector<Value, 1> final_result = {
        rewriter.create<LLVM::LoadOp>(loc, result_ptr)};
    if (failed(copyUnrankedDescriptors(rewriter, loc, op->getResultTypes(),
                                       final_result,
                                       /*toDynamic=*/false))) {
      return failure();
    }

    rewriter.replaceOp(op, final_result.front());
    return success();
  }

 protected:
  StringRef GetFuncName() const override { return kCInterfaceJITExecute; }

  Type GetFuncType() const override {
    auto i64_ty = IntegerType::get(getContext(), 64);
    auto void_ptr_ty = getVoidPtrType();
    return LLVM::LLVMFunctionType::get(getVoidType(),
                                       {/*void* op_kernel_ctx*/ void_ptr_ty,
                                        /*void* callable*/ void_ptr_ty,
                                        /*void* result*/ void_ptr_ty,
                                        /*int64_t num_args*/ i64_ty,
                                        /*void* args_ptr*/ void_ptr_ty});
  }
};

class ReportErrorOpConverter
    : public ConvertToLLVMCallOpPattern<ReportErrorOp> {
 public:
  using ConvertToLLVMCallOpPattern<ReportErrorOp>::ConvertToLLVMCallOpPattern;

  LogicalResult matchAndRewrite(
      ReportErrorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    Value message_constant =
        GenerateErrorMessageConstant(loc, module, adaptor.msg(), rewriter);

    // Insert function call.
    FlatSymbolRefAttr tf_func_ref =
        GetOrInsertLLVMFunction(GetFuncName(), GetFuncType(), op, &rewriter);
    Value error_code = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter->convertType(rewriter.getI32Type()),
        adaptor.error_codeAttr());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, llvm::None, tf_func_ref,
        llvm::makeArrayRef({adaptor.ctx(), error_code, message_constant}));
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
    std::string err_str;
    llvm::raw_string_ostream err_stream(err_str);
    err_stream << message;
    if (!loc.isa<UnknownLoc>()) {
      err_stream << " at ";
      loc.print(err_stream);
    }
    err_stream << '\00';
    StringRef generated_error(err_stream.str());
    return CreateOrFindGlobalStringConstant(
        loc, GetGlobalName(kErrorMessageGlobalBaseName, generated_error),
        generated_error, &builder);
  }
};

class NullContextOpConverter : public ConvertOpToLLVMPattern<NullContextOp> {
 public:
  using ConvertOpToLLVMPattern<NullContextOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      NullContextOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::NullOp>(op, getVoidPtrType());
    return success();
  }
};

class NullMemRefOpConverter : public ConvertOpToLLVMPattern<NullMemRefOp> {
 public:
  using ConvertOpToLLVMPattern<NullMemRefOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      NullMemRefOp null_memref_op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = null_memref_op->getLoc();
    LLVMTypeConverter type_converter = *getTypeConverter();
    mlir::Operation *op = null_memref_op.getOperation();

    auto shaped_result_type = null_memref_op.getType().cast<BaseMemRefType>();
    unsigned address_space = shaped_result_type.getMemorySpaceAsInt();

    Type elem_type = shaped_result_type.getElementType();
    Type llvm_elem_type = type_converter.convertType(elem_type);

    Value zero = createIndexConstant(rewriter, loc, 0);
    if (auto result_type = null_memref_op.getType().dyn_cast<MemRefType>()) {
      // Set all dynamic sizes to 1 and compute fake strides.
      SmallVector<Value, 4> dyn_sizes(result_type.getNumDynamicDims(),
                                      createIndexConstant(rewriter, loc, 1));
      SmallVector<Value, 4> sizes, strides;
      Value sizeBytes;
      getMemRefDescriptorSizes(loc, result_type, dyn_sizes, rewriter, sizes,
                               strides, sizeBytes);

      // Prepare packed args [allocatedPtr, alignedPtr, offset, sizes, strides]
      // to create a memref descriptor.
      Value null = rewriter.create<LLVM::NullOp>(
          loc, LLVM::LLVMPointerType::get(llvm_elem_type, address_space));
      SmallVector<Value, 12> packed_values{null, null, zero};
      packed_values.append(sizes);
      packed_values.append(strides);

      rewriter.replaceOp(
          op, MemRefDescriptor::pack(rewriter, loc, type_converter, result_type,
                                     packed_values));
      return success();
    }

    auto result_type = null_memref_op.getType().cast<UnrankedMemRefType>();
    Type llvm_result_type = type_converter.convertType(result_type);

    auto desc =
        UnrankedMemRefDescriptor::undef(rewriter, loc, llvm_result_type);
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
    Type elem_ptr_ptr_type = LLVM::LLVMPointerType::get(
        LLVM::LLVMPointerType::get(llvm_elem_type, address_space));

    Value null = rewriter.create<LLVM::NullOp>(
        loc, LLVM::LLVMPointerType::get(llvm_elem_type, address_space));
    UnrankedMemRefDescriptor::setAllocatedPtr(
        rewriter, loc, underlying_desc_ptr, elem_ptr_ptr_type, null);
    UnrankedMemRefDescriptor::setAlignedPtr(rewriter, loc, *getTypeConverter(),
                                            underlying_desc_ptr,
                                            elem_ptr_ptr_type, null);
    UnrankedMemRefDescriptor::setOffset(rewriter, loc, *getTypeConverter(),
                                        underlying_desc_ptr, elem_ptr_ptr_type,
                                        zero);

    desc.setMemRefDescPtr(rewriter, loc, underlying_desc_ptr);
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

class IsValidMemRefOpConverter
    : public ConvertOpToLLVMPattern<IsValidMemRefOp> {
 public:
  using ConvertOpToLLVMPattern<IsValidMemRefOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      IsValidMemRefOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MemRefDescriptor desc(adaptor.arg());

    // Compare every size in the descriptor to 0 to check num_elements == 0.
    int64_t rank = op.arg().getType().cast<MemRefType>().getRank();
    Value is_empty_shape = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));
    Value zero = createIndexConstant(rewriter, loc, 0);
    for (int i = 0; i < rank; ++i) {
      Value size = desc.size(rewriter, loc, i);
      Value is_zero_size = rewriter.create<LLVM::ICmpOp>(
          loc, rewriter.getI1Type(), LLVM::ICmpPredicate::eq, size, zero);
      is_empty_shape =
          rewriter.create<LLVM::OrOp>(loc, is_empty_shape, is_zero_size);
    }

    Value ptr = rewriter.create<LLVM::BitcastOp>(
        loc, getVoidPtrType(), desc.allocatedPtr(rewriter, loc));
    Value null = rewriter.create<LLVM::NullOp>(loc, getVoidPtrType());
    Value is_not_nullptr = rewriter.create<LLVM::ICmpOp>(
        loc, rewriter.getI1Type(), LLVM::ICmpPredicate::ne, ptr, null);

    // Valid memref = ptr != NULL || num_elements == 0;
    rewriter.replaceOpWithNewOp<LLVM::OrOp>(op, is_not_nullptr, is_empty_shape);
    return success();
  }
};

}  // namespace

void PopulateTFFrameworkToLLVMConversionPatterns(LLVMTypeConverter *converter,
                                                 RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
      IsValidMemRefOpConverter,
      JITCompileFromStrOpConverter,
      JITExecuteOpConverter,
      NullContextOpConverter,
      NullMemRefOpConverter,
      ReportErrorOpConverter,
      TFAllocOpConverter,
      TFDeallocOpConverter>(*converter);
  // clang-format on
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
