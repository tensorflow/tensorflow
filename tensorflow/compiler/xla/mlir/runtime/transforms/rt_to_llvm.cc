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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Async/IR/AsyncTypes.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_ops.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/passes.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/tracing.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"

namespace xla {
namespace runtime {
namespace {

using namespace mlir;  // NOLINT
using mlir::arith::ConstantOp;

using llvm::DenseMap;

#define GEN_PASS_DEF_CONVERTRUNTIMETOLLVMPASS
#include "tensorflow/compiler/xla/mlir/runtime/transforms/passes.h.inc"

//===----------------------------------------------------------------------===//
// Runtime C API declaration (see runtime.h header file).
//===----------------------------------------------------------------------===//

static constexpr const char *kGetResultStorage = "runtimeGetResultStorage";
static constexpr const char *kSetError = "runtimeSetError";
static constexpr const char *kCustomCall = "runtimeCustomCall";

struct RuntimeAPI {
  static FunctionType GetResultStorageFunctionType(MLIRContext *ctx) {
    auto ptr = LLVM::LLVMPointerType::get(ctx);
    auto i64 = IntegerType::get(ctx, 64);
    return FunctionType::get(ctx, {/*execution_ctx=*/ptr, i64},
                             {/*storage=*/ptr});
  }

  static FunctionType SetErrorFunctionType(MLIRContext *ctx) {
    auto ptr = LLVM::LLVMPointerType::get(ctx);
    return FunctionType::get(ctx, {/*execution_ctx=*/ptr, /*error_msg=*/ptr},
                             {});
  }

  static FunctionType CustomCallFunctionType(MLIRContext *ctx) {
    auto ptr = LLVM::LLVMPointerType::get(ctx);
    auto i1 = IntegerType::get(ctx, 1);
    return FunctionType::get(ctx,
                             {/*execution_ctx=*/ptr, /*callee=*/ptr,
                              /*args=*/ptr, /*attrs=*/ptr, /*rets=*/ptr},
                             {i1});
  }

  static FunctionType DirectCustomCallFunctionType(MLIRContext *ctx) {
    auto ptr = LLVM::LLVMPointerType::get(ctx);
    auto i1 = IntegerType::get(ctx, 1);
    return FunctionType::get(
        ctx, {/*execution_ctx=*/ptr, /*args=*/ptr, /*attrs=*/ptr, /*rets=*/ptr},
        {i1});
  }
};

// Adds function declaration if it doesn't already exist.
static void AddDeclaration(SymbolTable &sym_table, ModuleOp module,
                           std::string_view name, FunctionType type) {
  assert(sym_table.getOp() == module && "incorrect symbol table");
  if (sym_table.lookup(name)) return;

  auto b = ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  MLIRContext *ctx = module.getContext();
  func::FuncOp func = b.create<func::FuncOp>(name, type);
  func.setPrivate();

  // TODO(ezhulenev): Add per-argument nocapture attributes?
  func->setAttr("passthrough",
                ArrayAttr::get(ctx, {StringAttr::get(ctx, "nounwind")}));

  sym_table.insert(func);
}

// Adds Runtime C API declarations to the module.
static void AddRuntimeApiDeclarations(SymbolTable &sym_table, ModuleOp module) {
  auto add = [&](std::string_view name, FunctionType type) {
    AddDeclaration(sym_table, module, name, type);
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
    addConversion(ConvertExecutionContextType);
    addConversion(ConvertStatusType);
    addConversion(ConvertOpaqueType);
  }

  static llvm::Optional<Type> ConvertExecutionContextType(
      ExecutionContextType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  }

  static llvm::Optional<Type> ConvertStatusType(StatusType type) {
    return IntegerType::get(type.getContext(), 1);
  }

  static llvm::Optional<Type> ConvertOpaqueType(OpaqueType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
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

    auto execution_ctx = adaptor.getCtx();
    auto index = rewriter.create<ConstantOp>(loc, adaptor.getIndexAttr());

    // Get a pointer to the result value storage from the runtime.
    auto result_ptr_ty = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto result_ptr = rewriter.create<func::CallOp>(
        loc, kGetResultStorage, TypeRange(result_ptr_ty),
        ValueRange({execution_ctx, index}));

    // Cast from i8* to the LLVM pointer type to store the result.
    auto stored_type = getTypeConverter()->convertType(op.getValue().getType());
    if (!stored_type)
      return rewriter.notifyMatchFailure(
          op, "failed to convert output type to LLVM type");

    // Store the output value into the result value storage.
    auto value = adaptor.getValue();
    rewriter.create<LLVM::StoreOp>(loc, value, result_ptr.getResult(0));

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
    rewriter.replaceOp(op, adaptor.getStatus());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Convert rt.custom_call to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

static Value AsPtr(ImplicitLocOpBuilder &b,
                   std::variant<LLVM::AllocaOp, LLVM::GlobalOp> &v) {
  if (auto *alloca = std::get_if<LLVM::AllocaOp>(&v))
    return alloca->getResult();
  return Globals::AddrOf(b, std::get<LLVM::GlobalOp>(v));
}

static LLVM::GlobalOp EncodeEmptyArgsRets(Globals &g, ImplicitLocOpBuilder &b,
                                          std::string_view symbol_base) {
  // Empty args/rets is just an array with a single pointer to size (zero).
  Type ptr = LLVM::LLVMPointerType::get(b.getContext());
  Type type = LLVM::LLVMArrayType::get(ptr, 1);

  auto init = [&](ImplicitLocOpBuilder &ib, Attribute) {
    LLVM::GlobalOp zero =
        EncodeScalar(g, b, b.getI64IntegerAttr(0), "__rt_zero");

    Value arr = b.create<LLVM::UndefOp>(type);
    arr = b.create<LLVM::InsertValueOp>(arr, Globals::AddrOf(b, zero), 0);
    b.create<LLVM::ReturnOp>(arr);
  };

  return g.GetOrCreate(b, b.getArrayAttr({}), type, symbol_base, init);
}

static LLVM::GlobalOp EncodeTypeTable(Globals &g, ImplicitLocOpBuilder &b,
                                      ArrayRef<LLVM::GlobalOp> type_ids,
                                      std::string_view symbol_base) {
  // We store type table as `!llvm.array<ptr x len>`.
  Type ptr = LLVM::LLVMPointerType::get(b.getContext());
  Type type = LLVM::LLVMArrayType::get(ptr, type_ids.size());

  // Global initializer that encodes type ids as pointers.
  auto init = [&](ImplicitLocOpBuilder &ib, Attribute) -> LogicalResult {
    Value arr = b.create<LLVM::UndefOp>(type);
    for (auto &pair : llvm::enumerate(type_ids)) {
      arr = b.create<LLVM::InsertValueOp>(arr, Globals::AddrOf(b, pair.value()),
                                          pair.index());
    }
    b.create<LLVM::ReturnOp>(arr);
    return success();
  };

  // Put all type ids into an array attribute, so we can use it as a globals
  // cache key, so we do not encode the same type table multiple times.
  llvm::SmallVector<llvm::StringRef> type_id_syms;
  for (auto type_id : type_ids) type_id_syms.push_back(type_id.getSymName());
  auto arr_attr = b.getStrArrayAttr(type_id_syms);

  return g.GetOrCreate(b, arr_attr, type, symbol_base, init);
}

struct EncodedArguments {
  std::variant<LLVM::AllocaOp, LLVM::GlobalOp> encoded;  // `args` argument
  SmallVector<std::variant<LLVM::AllocaOp, LLVM::GlobalOp>> values;
};

static FailureOr<EncodedArguments> EncodeArguments(
    CallOp op, CustomCallArgEncodingSet &encodings, Globals &g, Allocas &a,
    DenseMap<Value, CustomCallArgEncoding::Encoded> &encoded_args,
    ImplicitLocOpBuilder &b, ValueRange operands, ValueRange converted) {
  llvm::SmallVector<CustomCallArgEncoding::Encoded> encoded;

  // Encode empty arguments as a global array (skip the status type).
  if (operands.drop_front().empty()) {
    return EncodedArguments{EncodeEmptyArgsRets(g, b, "__rt_empty_args"), {}};
  }

  EncodedArguments arguments;

  // Encode all arguments as a set of pointers (skip the execution context).
  for (auto tuple : llvm::drop_begin(llvm::zip(operands, converted))) {
    auto encoded_arg =
        encodings.Encode(g, a, b, std::get<0>(tuple), std::get<1>(tuple));
    if (failed(encoded_arg)) return failure();
    encoded.push_back(*encoded_arg);
    encoded_args.try_emplace(std::get<0>(tuple), *encoded_arg);
  }

  // We store encoded arguments as `!llvm.array<ptr x len>`.
  size_t len = encoded.empty() ? 1 : 2 + encoded.size();
  Type ptr = LLVM::LLVMPointerType::get(b.getContext());
  Type type = LLVM::LLVMArrayType::get(ptr, len);

  // Prepare an array for encoded arguments.
  Value arr = b.create<LLVM::UndefOp>(type);
  auto insert_value = [&](Value value, int64_t offset) {
    arr = b.create<LLVM::InsertValueOp>(arr, value, offset);
  };

  // Insert the number of encoded arguments.
  LLVM::GlobalOp num_args =
      EncodeScalar(g, b, b.getI64IntegerAttr(encoded.size()), "__rt_num_args");
  insert_value(Globals::AddrOf(b, num_args), 0);

  // Package arguments type ids into a type table global value.
  llvm::SmallVector<LLVM::GlobalOp> type_ids;
  for (auto &arg : encoded) type_ids.push_back(arg.type_id);
  LLVM::GlobalOp type_table =
      EncodeTypeTable(g, b, type_ids, "__rt_args_type_table");
  if (!encoded.empty()) insert_value(Globals::AddrOf(b, type_table), 1);

  // Store pointer to encoded arguments into the allocated storage.
  for (auto &pair : llvm::enumerate(encoded)) {
    CustomCallArgEncoding::Encoded encoded = pair.value();
    int64_t offset = 2 + pair.index();
    insert_value(AsPtr(b, encoded.value), offset);
    arguments.values.push_back(encoded.value);
  }

  // Get allocation for packed arguments pointers.
  LLVM::AllocaOp alloca = a.GetOrCreate(b, type);

  // Start the lifetime of the encoded arguments pointers.
  b.create<LLVM::LifetimeStartOp>(b.getI64IntegerAttr(-1), alloca);

  // Store constructed arguments pointers array into the alloca.
  b.create<LLVM::StoreOp>(arr, alloca.getRes());

  // Alloca that encodes the custom call arguments.
  arguments.encoded = alloca;

  return arguments;
}

// Encodes attributes into the global constant (array of pointers to the
// attributes data, which are also stored as global constants).
static FailureOr<LLVM::GlobalOp> EncodeAttributes(
    CustomCallAttrEncodingSet &encodings, SymbolTable &sym_table, Globals &g,
    ImplicitLocOpBuilder &b, ArrayRef<NamedAttribute> attrs) {
  // Forward attributes that are not part of the custom call operation itself.
  auto forward_attr = [](NamedAttribute attr) -> bool {
    return attr.getName() != "callee" && attr.getName() != "dynamic";
  };

  llvm::SmallVector<NamedAttribute> custom_call_attrs =
      llvm::to_vector(llvm::make_filter_range(attrs, forward_attr));

  // Sort encoded attributes in lexicographical order so that when decoding we
  // can efficiently find attributes by name.
  llvm::sort(custom_call_attrs, [](NamedAttribute &a, NamedAttribute &b) {
    return a.getName().strref() < b.getName().strref();
  });

  return EncodeAttributes(sym_table, g, b, encodings, "__rt_custom_call_attrs",
                          custom_call_attrs);
}

struct EncodedResults {
  std::variant<LLVM::AllocaOp, LLVM::GlobalOp> encoded;  // `rets` argument
  SmallVector<LLVM::AllocaOp> allocas;                   // encoded returns
};

static FailureOr<EncodedResults> EncodeResults(
    CallOp op, CustomCallRetEncodingSet &encodings, Globals &g, Allocas &a,
    ImplicitLocOpBuilder &b, TypeRange ret_types, TypeRange converted_types) {
  llvm::SmallVector<CustomCallRetEncoding::Encoded> encoded;

  // Encode empty returns as a global array (skip the status type).
  if (ret_types.drop_front().empty()) {
    return EncodedResults{EncodeEmptyArgsRets(g, b, "__rt_empty_rets"), {}};
  }

  EncodedResults results;

  // Encode all returns as a set of pointers (skip the status type).
  for (auto tuple : llvm::drop_begin(llvm::zip(ret_types, converted_types))) {
    auto encoded_ret =
        encodings.Encode(g, a, b, std::get<0>(tuple), std::get<1>(tuple));
    if (failed(encoded_ret)) return failure();
    encoded.push_back(*encoded_ret);
  }

  // We store encoded results as `!llvm.array<ptr x len>`.
  size_t len = encoded.empty() ? 1 : 2 + encoded.size();
  Type ptr = LLVM::LLVMPointerType::get(b.getContext());
  Type type = LLVM::LLVMArrayType::get(ptr, len);

  // Prepare an array for encoding results.
  Value arr = b.create<LLVM::UndefOp>(type);
  auto insert_value = [&](Value value, int64_t offset) {
    arr = b.create<LLVM::InsertValueOp>(arr, value, offset);
  };

  // Insert the number of encoded results.
  LLVM::GlobalOp num_rets =
      EncodeScalar(g, b, b.getI64IntegerAttr(encoded.size()), "__rt_num_rets");
  insert_value(Globals::AddrOf(b, num_rets), 0);

  // Package results type ids into a type table global value.
  llvm::SmallVector<LLVM::GlobalOp> type_ids;
  for (auto &arg : encoded) type_ids.push_back(arg.type_id);
  LLVM::GlobalOp type_table =
      EncodeTypeTable(g, b, type_ids, "__rt_rets_type_table");
  if (!encoded.empty()) insert_value(Globals::AddrOf(b, type_table), 1);

  // Store encoded results into the allocated storage.
  for (auto &pair : llvm::enumerate(encoded)) {
    CustomCallRetEncoding::Encoded encoded_pair = pair.value();
    int64_t offset = 2 + pair.index();
    insert_value(encoded_pair.value, offset);
    results.allocas.push_back(encoded_pair.value);
  }

  // Get allocation for packed results pointers.
  LLVM::AllocaOp alloca = a.GetOrCreate(b, type);

  // Start the lifetime of the encoded results pointers allocation.
  b.create<LLVM::LifetimeStartOp>(b.getI64IntegerAttr(-1), alloca);

  // Store constructed results pointers array on the stack
  b.create<LLVM::StoreOp>(arr, alloca);

  // Alloca that encodes the custom call returns.
  results.encoded = alloca;

  return results;
}

static FailureOr<SmallVector<Value>> DecodeResults(
    func::CallOp op, ImplicitLocOpBuilder b,
    CustomCallRetEncodingSet &encodings, TypeRange ret_types,
    TypeRange converted_types, SmallVector<LLVM::AllocaOp> &allocas) {
  SmallVector<Value> load_results;
  load_results.push_back(op.getResult(0));

  for (auto tuple : llvm::zip(llvm::drop_begin(ret_types),
                              llvm::drop_begin(converted_types), allocas)) {
    auto decoded_ret = encodings.Decode(b, std::get<0>(tuple),
                                        std::get<1>(tuple), std::get<2>(tuple));
    if (failed(decoded_ret)) return failure();
    load_results.push_back(*decoded_ret);
  }

  return load_results;
}

class CallOpLowering : public OpConversionPattern<CallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  CallOpLowering(TypeConverter &converter, MLIRContext *ctx,
                 SymbolTable &sym_table, Globals &globals,
                 EncodingAllocas &allocas,
                 CustomCallArgEncodingSet &arg_encoding,
                 CustomCallAttrEncodingSet &attr_encoding,
                 CustomCallRetEncodingSet &ret_encoding,
                 DenseMap<Value, CustomCallArgEncoding::Encoded> &encoded_args)
      : OpConversionPattern(converter, ctx),
        sym_table_(sym_table),
        globals_(globals),
        allocas_(allocas),
        arg_encoding_(arg_encoding),
        attr_encoding_(attr_encoding),
        ret_encoding_(ret_encoding),
        encoded_args_(encoded_args) {}

  LogicalResult matchAndRewrite(
      CallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Reuse allocas for encoding custom call arguments.
    Allocas allocas = allocas_.GetForOperation(op);

    // Encode operation arguments as a runtime API arguments.
    auto args =
        EncodeArguments(op, arg_encoding_, globals_, allocas, encoded_args_, b,
                        op->getOperands(), adaptor.getOperands());
    if (failed(args)) return op.emitOpError() << "failed to encode arguments";

    // Encode operation attributes as a runtime API argument.
    auto attrs = EncodeAttributes(attr_encoding_, sym_table_, globals_, b,
                                  op->getAttrs());
    if (failed(attrs)) return op.emitOpError() << "failed to encode attributes";

    // Encode operation results as a runtime API arguments.
    auto ret_types = op->getResultTypes();
    std::vector<Type> converted_ret_types(ret_types.size());
    std::transform(
        ret_types.begin(), ret_types.end(), converted_ret_types.begin(),
        [&](Type type) { return getTypeConverter()->convertType(type); });
    auto rets = EncodeResults(op, ret_encoding_, globals_, allocas, b,
                              ret_types, converted_ret_types);
    if (failed(rets)) return op.emitOpError() << "failed to encode results";

    // Creates a dynamic custom call resolved by name at run time.
    auto call_dynamic = [&]() -> func::CallOp {
      auto callee = Globals::AddrOf(
          b, globals_.GetOrCreate(b, op.getCallee(), "__rt_custom_call_name"));

      return b.create<func::CallOp>(
          kCustomCall, TypeRange(rewriter.getI1Type()),
          ValueRange({adaptor.getCtx(), callee, AsPtr(b, args->encoded),
                      Globals::AddrOf(b, *attrs), AsPtr(b, rets->encoded)}));
    };

    // Creates a direct custom call resolved at link time.
    auto call_direct = [&]() -> func::CallOp {
      auto type = RuntimeAPI::DirectCustomCallFunctionType(op.getContext());
      AddDeclaration(sym_table_, op->getParentOfType<ModuleOp>(),
                     op.getCallee(), type);

      return b.create<func::CallOp>(
          op.getCallee(), TypeRange(rewriter.getI1Type()),
          ValueRange({adaptor.getCtx(), AsPtr(b, args->encoded),
                      Globals::AddrOf(b, *attrs), AsPtr(b, rets->encoded)}));
    };

    // Build a call operation and result decoding right after the original op.
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfter(op);

    func::CallOp call = op.getDynamic() ? call_dynamic() : call_direct();

    // Load results written by custom call into the allocated storage and decode
    // them back to the expected type (e.g. convert memref descriptor type).
    FailureOr<SmallVector<Value>> decoded_results = DecodeResults(
        call, b, ret_encoding_, ret_types, converted_ret_types, rets->allocas);
    if (failed(decoded_results))
      return op.emitOpError() << "failed to decode results";

    auto size = b.getI64IntegerAttr(-1);

    // End the lifetime of encoded arguments and results pointers.
    if (auto *alloca = std::get_if<LLVM::AllocaOp>(&args->encoded))
      b.create<LLVM::LifetimeEndOp>(size, *alloca);
    if (auto *alloca = std::get_if<LLVM::AllocaOp>(&rets->encoded))
      b.create<LLVM::LifetimeEndOp>(size, *alloca);

    // End the lifetime of arguments encoded on a stack.
    for (auto &arg : args->values)
      if (auto *alloca = std::get_if<LLVM::AllocaOp>(&arg))
        b.create<LLVM::LifetimeEndOp>(size, *alloca);

    // End the lifetime of results encoded on a stack.
    for (LLVM::AllocaOp alloca : rets->allocas)
      b.create<LLVM::LifetimeEndOp>(size, alloca);

    rewriter.replaceOp(op, ValueRange(*decoded_results));
    return success();
  }

 private:
  SymbolTable &sym_table_;
  Globals &globals_;
  EncodingAllocas &allocas_;
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
    auto err = Globals::AddrOf(
        b, globals_.GetOrCreate(b, op.getError(), "__assert_failed"));

    // Call runtime API to report the error.
    auto execution_ctx = adaptor.getCtx();
    rewriter.replaceOpWithNewOp<func::CallOp>(op, kSetError, TypeRange(),
                                              ValueRange({execution_ctx, err}));

    return success();
  }

 private:
  Globals &globals_;
};

//===----------------------------------------------------------------------===//
// Convert rt.trace to a pair of custom calls (start and end trace activity).
//===----------------------------------------------------------------------===//

class TraceOpLowering : public OpConversionPattern<TraceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TraceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type status = StatusType::get(getContext());
    Type activity_id = rewriter.getI64Type();

    // Start the trace activity with the given annotation.
    b.setInsertionPoint(op);
    auto start = b.create<CallOp>(TypeRange({status, activity_id}), op.getCtx(),
                                  "xla.trace.activity_start",
                                  /*dynamic=*/false, ValueRange());
    start->setAttr("annotation", op.getAnnotation());

    // End activity after executing the attached region.
    b.setInsertionPointAfter(op);
    b.create<CallOp>(status, op.getCtx(), "xla.trace.activity_end",
                     /*dynamic=*/false, start.getResults());

    // Replace trace operation with inlined region.
    b.setInsertionPointAfter(op);
    auto terminator = cast<YieldOp>(op.getBody().front().getTerminator());
    rewriter.mergeBlockBefore(terminator->getBlock(), op);
    rewriter.replaceOp(op, terminator->getOperands());
    rewriter.eraseOp(terminator);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Convert rt.unsigned_cast to no-op.
//===----------------------------------------------------------------------===//

class UnsignedCastOpLowering : public OpConversionPattern<UnsignedCastOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      UnsignedCastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Just pass through the argument value.
    rewriter.replaceOp(op, adaptor.getValue());
    return success();
  }
};

//===----------------------------------------------------------------------===//

class ConvertRuntimeToLLVMPass
    : public impl::ConvertRuntimeToLLVMPassBase<ConvertRuntimeToLLVMPass> {
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

  // A symbol table for resolving symbol references attributes.
  SymbolTable sym_table(module);

  // Add declarations for the runtime API functions.
  AddRuntimeApiDeclarations(sym_table, module);

  RuntimeTypeConverter converter;
  RewritePatternSet patterns(ctx);

  // We use conversion to LLVM type to lower all runtime operands to LLVM types.
  LLVMTypeConverter llvm_converter(ctx);
  llvm_converter.addConversion(
      RuntimeTypeConverter::ConvertExecutionContextType);
  llvm_converter.addConversion(RuntimeTypeConverter::ConvertStatusType);
  llvm_converter.addConversion(RuntimeTypeConverter::ConvertOpaqueType);

  // TODO(ezhulenev): We should combine AsyncToLLVM and RtToLLVM into a single
  // pass that composed from `rt` and `async` patterns, because they both
  // rewriter function into the CFG and they interact badly.

  // Convert all async types to opaque pointers.
  llvm_converter.addConversion([](Type type) -> Optional<Type> {
    if (type.isa<async::TokenType, async::GroupType, async::ValueType>())
      // TODO(yijiagu): We should change the asyncRuntime function type with
      // opaque pointer
      return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));

    return std::nullopt;
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
  PopulateTraceTypeIdNames(type_id_names);
  if (opts_.populate_type_id_names) opts_.populate_type_id_names(type_id_names);

  // A helper class to create unique global constants.
  Globals globals(module, type_id_names);

  // A helper class to create allocas for values encoded on a stack.
  EncodingAllocas allocas;

  // Keep a cache of encoded values to encode each unique value just once.
  DenseMap<Value, CustomCallArgEncoding::Encoded> encoded_args;

  // Lower from the runtime operations to the runtime API function calls.
  patterns.add<SetOutputOpLowering, IsOkOpLowering>(llvm_converter, ctx);
  patterns.add<SetErrorOpLowering>(llvm_converter, ctx, globals);

  // Lower tracing operation to a pair of push/pop custom calls.
  patterns.add<TraceOpLowering>(llvm_converter, ctx);

  // Erase special signless-unsigned casting operation that we added to work
  // around the unsigned constants limitation.
  patterns.add<UnsignedCastOpLowering>(llvm_converter, ctx);

  // Use default custom call encoding for canonical types.
  CustomCallArgEncodingSet args = DefaultArgEncodings();
  CustomCallAttrEncodingSet attrs = DefaultAttrEncodings();
  CustomCallRetEncodingSet rets = DefaultRetEncodings();

  // Add user-defined arg and attr encodings.
  if (opts_.populate_arg_encodings) opts_.populate_arg_encodings(args);
  if (opts_.populate_attr_encodings) opts_.populate_attr_encodings(attrs);
  if (opts_.populate_ret_encodings) opts_.populate_ret_encodings(rets);

  patterns.add<CallOpLowering>(llvm_converter, ctx, sym_table, globals, allocas,
                               args, attrs, rets, encoded_args);

  // Convert function signatures and call sites.
  mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, converter);
  populateCallOpTypeConversionPattern(patterns, converter);

  // Set up conversion target to rewrite all runtime operations.
  ConversionTarget target(*ctx);
  target.addIllegalDialect<RuntimeDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ConstantOp, UnrealizedConversionCastOp, func::CallOp>();

  // Add dynamic legality constraints to apply conversions defined above.
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType());
  });

  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    return converter.isSignatureLegal(op.getCalleeType());
  });

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    return signalPassFailure();

  // Remove all rt.exported attributes once we are done with conversion to LLVM.
  module.walk([](Operation *op) { op->removeAttr("rt.exported"); });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateConvertRuntimeToLLVMPass(
    ConvertRuntimeToLLvmOpts opts) {
  return std::make_unique<ConvertRuntimeToLLVMPass>(std::move(opts));
}

}  // namespace runtime
}  // namespace xla
