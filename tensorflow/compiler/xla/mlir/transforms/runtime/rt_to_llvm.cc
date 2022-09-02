/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Async/IR/Async.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_ops.h"
#include "tensorflow/compiler/xla/mlir/transforms/runtime/custom_call_encoding.h"
#include "tensorflow/compiler/xla/mlir/transforms/runtime/passes.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"

namespace xla {
namespace runtime {
namespace {

using namespace mlir;  // NOLINT
using mlir::arith::ConstantOp;
using mlir::func::CallOp;
using mlir::func::FuncOp;

using llvm::DenseMap;

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/xla/mlir/transforms/runtime/passes.h.inc"

//===----------------------------------------------------------------------===//
// Runtime C API declaration (see runtime.h header file).
//===----------------------------------------------------------------------===//

static constexpr const char *kGetResultStorage = "runtimeGetResultStorage";
static constexpr const char *kSetError = "runtimeSetError";
static constexpr const char *kCustomCall = "runtimeCustomCall";

struct RuntimeAPI {
  static LLVM::LLVMPointerType OpaquePointerType(MLIRContext *ctx) {
    return LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  }

  static LLVM::LLVMPointerType CustomCallArgumentsType(MLIRContext *ctx) {
    return LLVM::LLVMPointerType::get(RuntimeAPI::OpaquePointerType(ctx));
  }

  static LLVM::LLVMPointerType CustomCallAttributesType(MLIRContext *ctx) {
    return LLVM::LLVMPointerType::get(RuntimeAPI::OpaquePointerType(ctx));
  }

  static LLVM::LLVMPointerType CustomCallResultsType(MLIRContext *ctx) {
    return LLVM::LLVMPointerType::get(RuntimeAPI::OpaquePointerType(ctx));
  }

  static FunctionType GetResultStorageFunctionType(MLIRContext *ctx) {
    auto kernel_context = OpaquePointerType(ctx);
    auto i64 = IntegerType::get(ctx, 64);
    auto storage = OpaquePointerType(ctx);
    return FunctionType::get(ctx, {kernel_context, i64}, {storage});
  }

  static FunctionType SetErrorFunctionType(MLIRContext *ctx) {
    auto kernel_context = OpaquePointerType(ctx);
    auto error_msg = OpaquePointerType(ctx);
    return FunctionType::get(ctx, {kernel_context, error_msg}, {});
  }

  static FunctionType CustomCallFunctionType(MLIRContext *ctx) {
    auto kernel_context = OpaquePointerType(ctx);
    auto callee = OpaquePointerType(ctx);
    auto args = CustomCallArgumentsType(ctx);
    auto attrs = CustomCallAttributesType(ctx);
    auto rets = CustomCallResultsType(ctx);
    auto i1 = IntegerType::get(ctx, 1);
    return FunctionType::get(ctx, {kernel_context, callee, args, attrs, rets},
                             {i1});
  }

  static FunctionType DirectCustomCallFunctionType(MLIRContext *ctx) {
    auto kernel_context = OpaquePointerType(ctx);
    auto args = CustomCallArgumentsType(ctx);
    auto attrs = CustomCallAttributesType(ctx);
    auto rets = CustomCallResultsType(ctx);
    auto i1 = IntegerType::get(ctx, 1);
    return FunctionType::get(ctx, {kernel_context, args, attrs, rets}, {i1});
  }
};

// Adds function declaration if it doesn't already exist.
static void AddDeclaration(ModuleOp module, std::string_view name,
                           FunctionType type) {
  auto b = ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());
  if (module.lookupSymbol(name)) return;

  MLIRContext *ctx = module.getContext();
  FuncOp func = b.create<FuncOp>(name, type);
  func.setPrivate();

  // TODO(ezhulenev): Add per-argument nocapture attributes?
  func->setAttr("passthrough",
                ArrayAttr::get(ctx, {StringAttr::get(ctx, "nounwind")}));
}

// Adds Runtime C API declarations to the module.
static void AddRuntimeApiDeclarations(ModuleOp module) {
  auto add = [&](std::string_view name, FunctionType type) {
    AddDeclaration(module, name, type);
  };

  MLIRContext *ctx = module.getContext();
  add(kGetResultStorage, RuntimeAPI::GetResultStorageFunctionType(ctx));
  add(kSetError, RuntimeAPI::SetErrorFunctionType(ctx));
  add(kCustomCall, RuntimeAPI::CustomCallFunctionType(ctx));
}

//===----------------------------------------------------------------------===//

class RuntimeTypeConverter : public TypeConverter {
 public:
  RuntimeTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(ConvertKernelContextType);
    addConversion(ConvertStatusType);
  }

  static llvm::Optional<Type> ConvertKernelContextType(KernelContextType type) {
    return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
  }

  static llvm::Optional<Type> ConvertStatusType(StatusType type) {
    return IntegerType::get(type.getContext(), 1);
  }
};

//===----------------------------------------------------------------------===//
// Convert rt.set_output to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

class SetOutputOpLowering : public OpConversionPattern<SetOutputOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SetOutputOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto kernel_context = adaptor.ctx();
    auto index = rewriter.create<ConstantOp>(loc, adaptor.indexAttr());

    // Get a pointer to the result value storage from the runtime.
    auto result_ptr_ty = RuntimeAPI::OpaquePointerType(rewriter.getContext());
    auto result_ptr = rewriter.create<CallOp>(
        loc, kGetResultStorage, TypeRange(result_ptr_ty),
        ValueRange({kernel_context, index}));

    // Cast from i8* to the LLVM pointer type to store the result.
    auto stored_type = getTypeConverter()->convertType(op.value().getType());
    if (!stored_type)
      return rewriter.notifyMatchFailure(
          op, "failed to convert output type to LLVM type");

    auto casted_result_ptr = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(stored_type), result_ptr.getResult(0));

    // Store the output value into the result value storage.
    auto value = adaptor.value();
    rewriter.create<LLVM::StoreOp>(loc, value, casted_result_ptr.getResult());

    // Erase the original runtime operation.
    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Convert rt.is_ok to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

class IsOkOpLowering : public OpConversionPattern<IsOkOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IsOkOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Just pass through the converted operand.
    rewriter.replaceOp(op, adaptor.status());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Convert rt.custom_call to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

static FailureOr<Value> EncodeArguments(
    CustomCallOp op, CustomCallArgEncodingSet &encodings, Globals &g,
    DenseMap<Value, CustomCallArgEncoding::Encoded> &encoded_args,
    ImplicitLocOpBuilder &b, ValueRange operands, ValueRange converted) {
  llvm::SmallVector<CustomCallArgEncoding::Encoded> encoded;

  // Encode all arguments as a set of pointers (skip the kernel context).
  for (auto tuple : llvm::drop_begin(llvm::zip(operands, converted))) {
    // Check if the value was already encoded.
    auto it = encoded_args.find(std::get<0>(tuple));
    if (it != encoded_args.end()) {
      encoded.push_back(it->second);
      continue;
    }

    // Otherwise encode it right after the converted value definition.
    OpBuilder::InsertionGuard guard(b);
    if (auto *defining_op = std::get<1>(tuple).getDefiningOp()) {
      b.setInsertionPointAfter(defining_op);
    } else {
      b.setInsertionPointToStart(std::get<1>(tuple).getParentBlock());
    }

    auto encoded_arg =
        encodings.Encode(g, b, std::get<0>(tuple), std::get<1>(tuple));
    if (failed(encoded_arg)) return failure();
    encoded.push_back(*encoded_arg);
    encoded_args.try_emplace(std::get<0>(tuple), *encoded_arg);
  }

  // We store encoded arguments as `!llvm.array<ptr<i8> x len>`.
  Type ptr = LLVM::LLVMPointerType::get(b.getI8Type());
  Type type = LLVM::LLVMArrayType::get(ptr, 1 + encoded.size() * 2);

  // Prepare an array for encoding arguments.
  Value arr = b.create<LLVM::UndefOp>(type);
  auto insert_value = [&](Value value, int64_t offset) {
    Value bcasted = b.createOrFold<LLVM::BitcastOp>(ptr, value);
    arr = b.create<LLVM::InsertValueOp>(arr, bcasted, offset);
  };

  // Insert the number of encoded arguments.
  Attribute num_args = b.getI64IntegerAttr(encoded.size());
  insert_value(PackScalarAttribute(g, b, num_args, "__rt_num_args"), 0);

  // Store encoded arguments into the allocated storage.
  for (auto &pair : llvm::enumerate(encoded)) {
    CustomCallArgEncoding::Encoded encoded = pair.value();
    int64_t offset = 1 + pair.index() * 2;

    insert_value(encoded.type_id, offset + 0);
    insert_value(encoded.value, offset + 1);
  }

  // Always create an `alloca` in the parent function entry block.
  // See: https://llvm.org/docs/Frontend/PerformanceTips.html#use-of-allocas
  Value mem = [&]() -> Value {
    Block &block = op->getParentOfType<FuncOp>().getBody().front();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&block);
    Value c1 = b.create<ConstantOp>(b.getI32IntegerAttr(1));
    return b.create<LLVM::AllocaOp>(LLVM::LLVMPointerType::get(type), c1, 0);
  }();

  // Store constructed arguments array on the stack and return a pointer to it.
  b.create<LLVM::StoreOp>(arr, mem);

  // Return a pointer to the first element of the arguments array.
  Type ptr_ptr = mlir::LLVM::LLVMPointerType::get(ptr);
  Value c0 = b.create<ConstantOp>(b.getI64IntegerAttr(0));
  Value gep = b.create<LLVM::GEPOp>(ptr_ptr, mem, ValueRange({c0, c0}));
  return gep;
}

// Encodes attributes into the global constant (array of pointers to the
// attributes data, which are also stored as global constants).
static FailureOr<Value> EncodeAttributes(CustomCallAttrEncodingSet &encodings,
                                         Globals &g, ImplicitLocOpBuilder &b,
                                         ArrayRef<NamedAttribute> attrs) {
  // Skip attributes passed explicitly as a custom call argument.
  auto skip = [](NamedAttribute attr) {
    return attr.getName() == "callee" || attr.getName() == "direct";
  };

  llvm::SmallVector<NamedAttribute> custom_call_attrs =
      llvm::to_vector(llvm::make_filter_range(attrs, std::not_fn(skip)));

  // Sort encoded attributes in lexicographical order so that when decoding we
  // can efficiently find attributes by name.
  llvm::sort(custom_call_attrs, [](NamedAttribute &a, NamedAttribute &b) {
    return a.getName().strref() < b.getName().strref();
  });

  return EncodeAttributes(g, b, encodings, "__rt_custom_call_attrs",
                          custom_call_attrs);
}

struct EncodedResults {
  Value result_array_ptr;  // passed as 'rets' argument to custom call
  SmallVector<LLVM::AllocaOp> allocas;  // storage for values of results
};

static FailureOr<EncodedResults> EncodeResults(
    CustomCallOp op, CustomCallRetEncodingSet &encodings, Globals &g,
    ImplicitLocOpBuilder &b, TypeRange ret_types, TypeRange converted_types) {
  llvm::SmallVector<CustomCallRetEncoding::Encoded> encoded;
  EncodedResults results;

  // Encode all returns as a set of pointers (skip the status type).
  for (auto tuple : llvm::drop_begin(llvm::zip(ret_types, converted_types))) {
    Block &block = op->getParentOfType<FuncOp>().getBody().front();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&block);
    auto encoded_ret =
        encodings.Encode(g, b, std::get<0>(tuple), std::get<1>(tuple));
    if (failed(encoded_ret)) return failure();
    encoded.push_back(*encoded_ret);
  }

  // We store encoded results as `!llvm.array<ptr<i8> x len>`.
  Type ptr = LLVM::LLVMPointerType::get(b.getI8Type());
  Type type = LLVM::LLVMArrayType::get(ptr, 1 + encoded.size() * 2);

  // Prepare an array for encoding results.
  Value arr = b.create<LLVM::UndefOp>(type);
  auto insert_value = [&](Value value, int64_t offset) {
    Value bcasted = b.createOrFold<LLVM::BitcastOp>(ptr, value);
    arr = b.create<LLVM::InsertValueOp>(arr, bcasted, offset);
  };

  // Insert the number of encoded results.
  Attribute num_rets = b.getI64IntegerAttr(encoded.size());
  insert_value(PackScalarAttribute(g, b, num_rets, "__rt_num_rets"), 0);

  // Store encoded results into the allocated storage.
  for (auto &pair : llvm::enumerate(encoded)) {
    CustomCallRetEncoding::Encoded encoded_pair = pair.value();
    int64_t offset = 1 + pair.index() * 2;

    insert_value(encoded_pair.type_id, offset + 0);
    insert_value(encoded_pair.value, offset + 1);

    results.allocas.push_back(encoded_pair.value);
  }

  // Always create an `alloca` in the parent function entry block.
  // See: https://llvm.org/docs/Frontend/PerformanceTips.html#use-of-allocas
  Value mem = [&]() -> Value {
    Block &block = op->getParentOfType<FuncOp>().getBody().front();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&block);
    Value c1 = b.create<ConstantOp>(b.getI32IntegerAttr(1));
    return b.create<LLVM::AllocaOp>(LLVM::LLVMPointerType::get(type), c1, 0);
  }();

  // Store constructed results array on the stack
  b.create<LLVM::StoreOp>(arr, mem);

  // Return a pointer to the first element of the results array.
  Type ptr_ptr = mlir::LLVM::LLVMPointerType::get(ptr);
  Value c0 = b.create<ConstantOp>(b.getI64IntegerAttr(0));
  Value gep = b.create<LLVM::GEPOp>(ptr_ptr, mem, ValueRange({c0, c0}));
  results.result_array_ptr = gep;
  return results;
}

// TODO(yijiagu): Add memref support
static SmallVector<Value> GenResult(CallOp op, ImplicitLocOpBuilder b,
                                    ArrayRef<LLVM::AllocaOp> allocas) {
  SmallVector<Value> load_results;
  load_results.push_back(op.getResult(0));
  for (auto v : allocas) {
    auto load_value = b.create<LLVM::LoadOp>(v);
    load_results.push_back(load_value);
  }
  return load_results;
}

class CustomCallOpLowering : public OpConversionPattern<CustomCallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  CustomCallOpLowering(
      TypeConverter &converter, MLIRContext *ctx, Globals &globals,
      CustomCallArgEncodingSet &arg_encoding,
      CustomCallAttrEncodingSet &attr_encoding,
      CustomCallRetEncodingSet &ret_encoding,
      DenseMap<Value, CustomCallArgEncoding::Encoded> &encoded_args)
      : OpConversionPattern(converter, ctx),
        globals_(globals),
        arg_encoding_(arg_encoding),
        attr_encoding_(attr_encoding),
        ret_encoding_(ret_encoding),
        encoded_args_(encoded_args) {}

  LogicalResult matchAndRewrite(
      CustomCallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Encode operation arguments as a runtime API arguments.
    auto args = EncodeArguments(op, arg_encoding_, globals_, encoded_args_, b,
                                op->getOperands(), adaptor.getOperands());
    if (failed(args)) return op.emitOpError() << "failed to encode arguments";

    // Encode operation attributes as a runtime API argument.
    auto attrs = EncodeAttributes(attr_encoding_, globals_, b, op->getAttrs());
    if (failed(attrs)) return op.emitOpError() << "failed to encode attributes";

    // Encode operation results as a runtime API arguments.
    auto ret_types = op->getResultTypes();
    std::vector<Type> converted_ret_types(ret_types.size());
    std::transform(
        ret_types.begin(), ret_types.end(), converted_ret_types.begin(),
        [&](Type type) { return getTypeConverter()->convertType(type); });
    auto rets = EncodeResults(op, ret_encoding_, globals_, b, ret_types,
                              converted_ret_types);
    if (failed(rets)) return op.emitOpError() << "failed to encode results";

    if (op.direct()) {
      // Call custom call target directly.
      auto type = RuntimeAPI::DirectCustomCallFunctionType(op.getContext());
      AddDeclaration(op->getParentOfType<ModuleOp>(), op.callee(), type);

      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointAfter(op);
      auto call_op = b.create<CallOp>(
          op.callee(), TypeRange(rewriter.getI1Type()),
          ValueRange({adaptor.ctx(), *args, *attrs, rets->result_array_ptr}));
      auto load_rets = GenResult(call_op, b, rets->allocas);
      rewriter.replaceOp(op, ValueRange(load_rets));
    } else {
      // Otherwise pass the custom call callee to the generic custom call API.
      auto callee = Globals::OpaqueAddrOf(
          b, globals_.GetOrCreate(b, op.callee(), "__rt_custom_call_callee"));

      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointAfter(op);
      auto call_op =
          b.create<CallOp>(kCustomCall, TypeRange(rewriter.getI1Type()),
                           ValueRange({adaptor.ctx(), callee, *args, *attrs,
                                       rets->result_array_ptr}));

      auto load_rets = GenResult(call_op, b, rets->allocas);
      rewriter.replaceOp(op, ValueRange(load_rets));
    }

    return success();
  }

 private:
  Globals &globals_;
  CustomCallArgEncodingSet &arg_encoding_;
  CustomCallAttrEncodingSet &attr_encoding_;
  CustomCallRetEncodingSet &ret_encoding_;
  DenseMap<Value, CustomCallArgEncoding::Encoded> &encoded_args_;
};

//===----------------------------------------------------------------------===//
// Convert rt.set_error to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

class SetErrorOpLowering : public OpConversionPattern<SetErrorOp> {
 public:
  SetErrorOpLowering(TypeConverter &converter, MLIRContext *ctx,
                     Globals &globals)
      : OpConversionPattern(converter, ctx), globals_(globals) {}

  LogicalResult matchAndRewrite(
      SetErrorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Get the error message (pointer to a null terminated string).
    auto err = Globals::OpaqueAddrOf(
        b, globals_.GetOrCreate(b, op.error(), "__assert_failed"));

    // Call runtime API to report the error.
    auto kernel_context = adaptor.ctx();
    rewriter.replaceOpWithNewOp<CallOp>(op, kSetError, TypeRange(),
                                        ValueRange({kernel_context, err}));

    return success();
  }

 private:
  Globals &globals_;
};

//===----------------------------------------------------------------------===//

class ConvertRuntimeToLLVMPass
    : public ConvertRuntimeToLLVMPassBase<ConvertRuntimeToLLVMPass> {
 public:
  explicit ConvertRuntimeToLLVMPass(ConvertRuntimeToLLvmOpts opts)
      : opts_(std::move(opts)) {}

  void runOnOperation() override;

 private:
  ConvertRuntimeToLLvmOpts opts_;
};

void ConvertRuntimeToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();

  // Add declarations for the runtime API functions.
  AddRuntimeApiDeclarations(module);

  RuntimeTypeConverter converter;
  RewritePatternSet patterns(ctx);

  // We use conversion to LLVM type to lower all runtime operands to LLVM types.
  LLVMTypeConverter llvm_converter(ctx);
  llvm_converter.addConversion(RuntimeTypeConverter::ConvertKernelContextType);
  llvm_converter.addConversion(RuntimeTypeConverter::ConvertStatusType);

  // TODO(ezhulenev): We should combine AsyncToLLVM and RtToLLVM into a single
  // pass that composed from `rt` and `async` patterns, because they both
  // rewriter function into the CFG and they interact badly.

  // Convert all async types to opaque pointers.
  llvm_converter.addConversion([](Type type) -> Optional<Type> {
    if (type.isa<async::TokenType, async::GroupType, async::ValueType>())
      return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));

    return llvm::None;
  });

  // Use UnrealizedConversionCast as the bridge so that we don't need to pull
  // in patterns for other dialects.
  auto add_unrealized_cast = [](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) {
    auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return Optional<Value>(cast.getResult(0));
  };
  converter.addSourceMaterialization(add_unrealized_cast);

  // Add type conversions for user-defined types so that we can properly convert
  // all function signatures in the module and prepare values for custom calls.
  if (opts_.populate_type_conversions) {
    opts_.populate_type_conversions(converter);
    opts_.populate_type_conversions(llvm_converter);
  }

  // Register mappings from the TypeID to type names.
  TypeIDNameRegistry type_id_names;
  PopulateCustomCallTypeIdNames(type_id_names);
  if (opts_.populate_type_id_names) opts_.populate_type_id_names(type_id_names);

  // A helper class to create unique global constants.
  Globals globals(module, type_id_names);

  // Keep a cache of encoded values to encode each unique value just once.
  DenseMap<Value, CustomCallArgEncoding::Encoded> encoded_args;

  // Lower from the runtime operations to the runtime API function calls.
  patterns.add<SetOutputOpLowering, IsOkOpLowering>(llvm_converter, ctx);
  patterns.add<SetErrorOpLowering>(llvm_converter, ctx, globals);

  // Use default custom call encoding for canonical types.
  CustomCallArgEncodingSet args = DefaultArgEncodings();
  CustomCallAttrEncodingSet attrs = DefaultAttrEncodings();
  CustomCallRetEncodingSet rets = DefaultRetEncodings();

  // Add user-defined arg and attr encodings.
  if (opts_.populate_arg_encodings) opts_.populate_arg_encodings(args);
  if (opts_.populate_attr_encodings) opts_.populate_attr_encodings(attrs);
  if (opts_.populate_ret_encodings) opts_.populate_ret_encodings(rets);

  patterns.add<CustomCallOpLowering>(llvm_converter, ctx, globals, args, attrs,
                                     rets, encoded_args);

  // Convert function signatures and call sites.
  mlir::populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                                 converter);
  populateCallOpTypeConversionPattern(patterns, converter);

  // Set up conversion target to rewrite all runtime operations.
  ConversionTarget target(*ctx);
  target.addIllegalDialect<RuntimeDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ConstantOp, UnrealizedConversionCastOp, CallOp>();

  // Add dynamic legality constraints to apply conversions defined above.
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType());
  });

  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    return converter.isSignatureLegal(op.getCalleeType());
  });

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateConvertRuntimeToLLVMPass(
    ConvertRuntimeToLLvmOpts opts) {
  return std::make_unique<ConvertRuntimeToLLVMPass>(std::move(opts));
}

}  // namespace runtime
}  // namespace xla
