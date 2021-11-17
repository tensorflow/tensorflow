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
#include "mlir-hlo/Dialect/disc-ral/IR/disc_ral_ops.h"
#include "mlir-hlo/Dialect/disc-ral/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/disc-ral/transforms/rewriters.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

// This file implements the logic to convert disc ral ops to llvm dialect

namespace mlir {
namespace disc_ral {

using LLVM::GlobalOp;
using LLVM::LLVMFuncOp;
using StrT = SmallString<128>;

namespace {

constexpr const char* kRalDispatchFunctionName = "disc_ral_call";
constexpr const char* kGpuBinaryAttrName = "gpu.binary_blob";
constexpr const char* kRalGpuLaunch = "ral_kernel_launch";

// Encodes a mlir type and appends the encoding to the string buffer `out`.
LogicalResult getTypeEncoding(MLIRContext* ctx, Type t, StrT& out) {
  Type llvm_pointer_type = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  Type llvm_pointer_pointer_type =
      LLVM::LLVMPointerType::get(llvm_pointer_type);
  if (auto memref_type = t.dyn_cast<MemRefType>()) {
    out.append(
        Twine("m").concat(Twine(memref_type.getRank()).concat("d")).str());
    return getTypeEncoding(ctx, memref_type.getElementType(), out);
  } else if (auto int_type = t.dyn_cast<IntegerType>()) {
    out.append(Twine("i").concat(Twine(int_type.getWidth())).str());
  } else if (auto fp_type = t.dyn_cast<FloatType>()) {
    out.append(Twine("f").concat(Twine(fp_type.getWidth())).str());
  } else if (auto ctx_type = t.dyn_cast<RalExecutionContextType>() ||
                             t == llvm_pointer_type) {
    out.append("pvoid");
  } else if (t == llvm_pointer_pointer_type) {
    out.append("ppvoid");
  } else if (t.isIndex()) {
    // index is mapping to int64_t a.t.m. Re-visit this in case necessary.
    out.append("i64");
  } else {
    // unknown type
    return failure();
  }
  return success();
}

// Encodes a ral_dispatch op and appends the encoding to the string buffer
// `out`. The format:
//   encoding = separator.join(target_name, device, inputs_encode,
//   outputs_encode)
//
//   separator = '___'
//
//   target_name: name of the external function to dispatch.
//
//   device: user defined string (e.g. cpu or gpu)
//
//   inputs_encode = type_separator.join([type_encoding for type in
//   input_types])
//
//   outputs_encode = type_separator.join([type_encoding for type
//   in output_types])
//
//   type_separator = '_'
LogicalResult getDispatchOpSignatureEncoding(DispatchOp dispatch_op,
                                             StrT& out) {
  const char* separator = "___";
  // append signature prefix
  out.append(dispatch_op.call_target_name());
  out.append(separator);

  // encode backend (device) info
  out.append(dispatch_op.backend_config());
  out.append(separator);

  // encode input types
  Operation* op = dispatch_op.getOperation();
  for (auto& en : llvm::enumerate(op->getOperandTypes())) {
    if (en.index() != 0) out.append("_");
    if (failed(getTypeEncoding(op->getContext(), en.value(), out)))
      return failure();
  }
  out.append(separator);

  // encode output types
  for (auto& en : llvm::enumerate(op->getResultTypes())) {
    if (en.index() != 0) out.append("_");
    if (failed(getTypeEncoding(op->getContext(), en.value(), out)))
      return failure();
  }
  if (!op->getNumResults()) out.append("void");
  return success();
}

// Loads a global op at the current insertion point and returns the loaded
// value.
Value loadGlobalString(OpBuilder& builder, const Location& loc,
                       GlobalOp globalOp) {
  MLIRContext* ctx = builder.getContext();
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, globalOp);
  Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, IntegerType::get(ctx, 64),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8)), globalPtr,
      ValueRange{cst0, cst0});
}

// Returns true if the globalOp has the same value as `value`.
bool checkGlobalOpContent(GlobalOp globalOp, StringRef value) {
  Optional<Attribute> optValue = globalOp.getValue();
  if (!optValue) return false;

  StringAttr attr = (*optValue).cast<StringAttr>();
  if (!attr) return false;

  return attr.getValue() == value;
}

// Creates a global const string op named `name` using the value if not exists
// and returns the Loaded value of this global op.
Value loadOrCreateGlobalString(PatternRewriter& rewriter,
                               SymbolTable& symbol_table, Operation* op,
                               StringRef name, StringRef value) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  GlobalOp globalOp = symbol_table.lookup<GlobalOp>(name);
  if (!globalOp) {
    OpBuilder::InsertionGuard guard(rewriter);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(module.getBody());

    auto type = LLVM::LLVMArrayType::get(IntegerType::get(op->getContext(), 8),
                                         value.size());
    globalOp = rewriter.create<LLVM::GlobalOp>(
        op->getLoc(), type, /*isConstant=*/true, LLVM::Linkage::Internal, name,
        rewriter.getStringAttr(value), /*alignment=*/0);

    // Update the symbol table
    symbol_table.insert(globalOp);

    rewriter.restoreInsertionPoint(ip);
  } else {
    assert(checkGlobalOpContent(globalOp, value));
  }

  return loadGlobalString(rewriter, op->getLoc(), globalOp);
}

// Converts a ral.dispatch_op to its llvm format.
class DispatchOpToLLVMPattern : public ConvertOpToLLVMPattern<DispatchOp> {
 public:
  DispatchOpToLLVMPattern(LLVMTypeConverter& type_converter,
                          SymbolTable& symbol_table)
      : ConvertOpToLLVMPattern<DispatchOp>(type_converter),
        symbol_table_(symbol_table) {}

  // Returns the ral dispatch function and inserts the declaration if not found.
  LLVMFuncOp getOrInsertDispatchFunction(PatternRewriter& rewriter,
                                         Operation* op) const;

  // Packs the inputs and outputs into a type-erased pointer array.
  // For example, `int func(int)` -> `void func(void* args[]) where args =
  // {in_ptr, out_ptr}`
  Value rewriteInsOutsOfDispatchOp(DispatchOp dispatch_op, ValueRange operands,
                                   ConversionPatternRewriter& rewriter,
                                   SmallVectorImpl<Value>& resultPtrs) const;

  LogicalResult matchAndRewrite(
      DispatchOp dispatch_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;

 private:
  SymbolTable& symbol_table_;
};

// Returns the llvm function definition of ral dispatch op and creates it first
// if not exists.
LLVMFuncOp DispatchOpToLLVMPattern::getOrInsertDispatchFunction(
    PatternRewriter& rewriter, Operation* op) const {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  LLVMFuncOp func = symbol_table_.lookup<LLVMFuncOp>(kRalDispatchFunctionName);

  if (func) return func;

  // Try to insert the function since it's not found.
  OpBuilder::InsertionGuard guard(rewriter);
  OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(module.getBody());
  Type llvm_pointer_type =
      LLVM::LLVMPointerType::get(IntegerType::get(op->getContext(), 8));
  Type llvm_pointer_pointer_type =
      LLVM::LLVMPointerType::get(llvm_pointer_type);
  func = rewriter.create<LLVMFuncOp>(
      op->getLoc(), kRalDispatchFunctionName,
      LLVM::LLVMFunctionType::get(
          getVoidType(),
          {
              llvm_pointer_type,        /* ral_context_t */
              llvm_pointer_type,        /* void* call_target_name */
              llvm_pointer_pointer_type /* void** args */
          },
          /*isVarArg=*/false));

  symbol_table_.insert(func);

  rewriter.restoreInsertionPoint(ip);

  return func;
}

// Packs the original inputs and outputs of the ral dispatch op to a uniform
// format.
//
// %struct = alloca(sizeof(struct { Parameters..., Results..., }))
// %array = alloca((NumParameters + NumResult) * sizeof(void *))
// for (i : [0, NumParameters))
//   %fieldPtr = llvm.getelementptr %struct[0, i]
//   llvm.store parameters[i], %fieldPtr
//   %elementPtr = llvm.getelementptr %array[i]
//   llvm.store %fieldPtr, %elementPtr
// for (i : [NumParameters, NumParameters + NumResult))
//   %fieldPtr = llvm.getelementptr %struct[0, i]
//   %elementPtr = llvm.getelementptr %array[i]
//   llvm.store %fieldPtr, %elementPtr
// return %array
Value DispatchOpToLLVMPattern::rewriteInsOutsOfDispatchOp(
    DispatchOp dispatch_op, ValueRange operands,
    ConversionPatternRewriter& rewriter,
    SmallVectorImpl<Value>& resultPtrs) const {
  MLIRContext* ctx = rewriter.getContext();
  Location loc = dispatch_op.getLoc();

  Type llvm_pointer_type = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  Type llvm_pointer_pointer_type =
      LLVM::LLVMPointerType::get(llvm_pointer_type);
  Type llvm_int32_type = IntegerType::get(ctx, 32);

  Value zero = rewriter.create<LLVM::ConstantOp>(loc, llvm_int32_type,
                                                 rewriter.getI32IntegerAttr(0));
  Value one = rewriter.create<LLVM::ConstantOp>(loc, llvm_int32_type,
                                                rewriter.getI32IntegerAttr(1));

  SmallVector<Value, 4> arguments = getTypeConverter()->promoteOperands(
      loc, dispatch_op.getOperands(), operands, rewriter);
  SmallVector<Type, 4> argument_types;
  for (auto argument : arguments) argument_types.push_back(argument.getType());
  for (auto resultType : dispatch_op.getResultTypes())
    argument_types.push_back(getTypeConverter()->convertType(resultType));

  auto struct_type =
      LLVM::LLVMStructType::getNewIdentified(ctx, StringRef(), argument_types);
  Value struct_ptr = rewriter.create<LLVM::AllocaOp>(
      loc, LLVM::LLVMPointerType::get(struct_type), one, /*alignment=*/0);
  Value array_size = rewriter.create<LLVM::ConstantOp>(
      loc, llvm_int32_type, rewriter.getI32IntegerAttr(argument_types.size()));
  Value array_ptr = rewriter.create<LLVM::AllocaOp>(
      loc, llvm_pointer_pointer_type, array_size, /*alignment=*/0);

  for (auto en : llvm::enumerate(argument_types)) {
    Value index = rewriter.create<LLVM::ConstantOp>(
        loc, llvm_int32_type, rewriter.getI32IntegerAttr(en.index()));
    Value field_ptr = rewriter.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(en.value()), struct_ptr,
        ArrayRef<Value>{zero, index});
    if (en.index() < arguments.size()) {
      rewriter.create<LLVM::StoreOp>(loc, arguments[en.index()], field_ptr);
    } else {
      resultPtrs.push_back(field_ptr);
    }

    Value element_ptr = rewriter.create<LLVM::GEPOp>(
        loc, llvm_pointer_pointer_type, array_ptr, index);
    Value casted =
        rewriter.create<LLVM::BitcastOp>(loc, llvm_pointer_type, field_ptr);
    rewriter.create<LLVM::StoreOp>(loc, casted, element_ptr);
  }

  return array_ptr;
}

LogicalResult DispatchOpToLLVMPattern::matchAndRewrite(
    DispatchOp dispatch_op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  StrT target_name;
  if (failed(getDispatchOpSignatureEncoding(dispatch_op, target_name))) {
    dispatch_op->emitError("unknown types in the dispatch op");
    return failure();
  }

  // Make sure the trailing zero is included in the constant.
  target_name.push_back('\0');

  Operation* op = dispatch_op.getOperation();
  Location loc = op->getLoc();
  SmallVector<Value, 3> callOpOperands;
  LLVMFuncOp dispatch_func = getOrInsertDispatchFunction(rewriter, op);

  SmallVector<Value, 1> resultPtrs;
  Value packedArgs = rewriteInsOutsOfDispatchOp(
      dispatch_op, adaptor.getOperands(), rewriter, resultPtrs);

  // the first argument is ral_context
  callOpOperands.push_back(adaptor.ctx());
  // the second argument is the target name
  callOpOperands.push_back(loadOrCreateGlobalString(
      rewriter, symbol_table_, op, target_name.str().drop_back(),
      target_name.str()));
  // the third argument is the args for target function
  callOpOperands.push_back(packedArgs);

  rewriter.create<LLVM::CallOp>(
      loc, llvm::None, mlir::SymbolRefAttr::get(dispatch_func), callOpOperands);

  SmallVector<Value, 1> results;
  llvm::transform(resultPtrs, std::back_inserter(results), [&](Value v) {
    return rewriter.create<LLVM::LoadOp>(loc, v);
  });

  rewriter.replaceOp(op, results);

  return success();
}

// A rewrite pattern to convert gpu.launch_func operations into corresponding
// runtime wrapper calls (modeled by ral.dispatch ops)
class ConvertLaunchFuncOpToRalCallPattern
    : public ConvertOpToLLVMPattern<gpu::LaunchFuncOp> {
 public:
  ConvertLaunchFuncOpToRalCallPattern(LLVMTypeConverter& type_converter,
                                      SymbolTable& symbol_table)
      : ConvertOpToLLVMPattern<gpu::LaunchFuncOp>(type_converter),
        symbol_table_(symbol_table) {}

 private:
  Value generateParamsArray(gpu::LaunchFuncOp launch_op, ValueRange operands,
                            OpBuilder& builder) const;
  Value generateKernelNameConstant(StringRef moduleName, StringRef name,
                                   Location loc, OpBuilder& builder) const;

  LogicalResult matchAndRewrite(
      gpu::LaunchFuncOp launch_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;

  SymbolTable& symbol_table_;
};

// Creates a struct containing all kernel parameters on the stack and returns
// an array of type-erased pointers to the fields of the struct. The array can
// then be passed to the CUDA / ROCm (HIP) kernel launch calls.
// The generated code is essentially as follows:
//
// %struct = alloca(sizeof(struct { Parameters... }))
// %array = alloca(NumParameters * sizeof(void *))
// for (i : [0, NumParameters))
//   %fieldPtr = llvm.getelementptr %struct[0, i]
//   llvm.store parameters[i], %fieldPtr
//   %elementPtr = llvm.getelementptr %array[i]
//   llvm.store %fieldPtr, %elementPtr
// return %array
Value ConvertLaunchFuncOpToRalCallPattern::generateParamsArray(
    gpu::LaunchFuncOp launch_op, ValueRange operands,
    OpBuilder& builder) const {
  MLIRContext* ctx = builder.getContext();
  Type llvm_pointer_type = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  Type llvm_pointer_pointer_type =
      LLVM::LLVMPointerType::get(llvm_pointer_type);
  Type llvm_int32_type = IntegerType::get(ctx, 32);

  Location loc = launch_op.getLoc();
  int num_kernel_operands = launch_op.getNumKernelOperands();
  auto arguments = getTypeConverter()->promoteOperands(
      loc, launch_op.getOperands().take_back(num_kernel_operands),
      operands.take_back(num_kernel_operands), builder);
  int num_arguments = static_cast<int>(arguments.size());
  SmallVector<Type, 4> argument_types;
  argument_types.reserve(num_arguments);
  for (auto argument : arguments) argument_types.push_back(argument.getType());
  auto struct_type =
      LLVM::LLVMStructType::getNewIdentified(ctx, StringRef(), argument_types);
  Value one = builder.create<LLVM::ConstantOp>(loc, llvm_int32_type,
                                               builder.getI32IntegerAttr(1));
  Value struct_ptr = builder.create<LLVM::AllocaOp>(
      loc, LLVM::LLVMPointerType::get(struct_type), one, /*alignment=*/0);
  Value array_size = builder.create<LLVM::ConstantOp>(
      loc, llvm_int32_type, builder.getI32IntegerAttr(num_arguments));
  Value array_ptr = builder.create<LLVM::AllocaOp>(
      loc, llvm_pointer_pointer_type, array_size, /*alignment=*/0);
  Value zero = builder.create<LLVM::ConstantOp>(loc, llvm_int32_type,
                                                builder.getI32IntegerAttr(0));
  for (auto en : llvm::enumerate(arguments)) {
    Value index = builder.create<LLVM::ConstantOp>(
        loc, llvm_int32_type, builder.getI32IntegerAttr(en.index()));
    Value field_ptr = builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(argument_types[en.index()]), struct_ptr,
        ArrayRef<Value>{zero, index});
    builder.create<LLVM::StoreOp>(loc, en.value(), field_ptr);
    Value element_ptr = builder.create<LLVM::GEPOp>(
        loc, llvm_pointer_pointer_type, array_ptr, index);
    Value casted =
        builder.create<LLVM::BitcastOp>(loc, llvm_pointer_type, field_ptr);
    builder.create<LLVM::StoreOp>(loc, casted, element_ptr);
  }
  return array_ptr;
}

// Emits LLVM IR to launch a kernel function. Expects the module that contains
// the compiled kernel function as a cubin in the `kRalGpuLaunch` attribute.
LogicalResult ConvertLaunchFuncOpToRalCallPattern::matchAndRewrite(
    gpu::LaunchFuncOp launch_op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  if (!launch_op.asyncDependencies().empty() || launch_op.asyncToken()) {
    return rewriter.notifyMatchFailure(
        launch_op, "Cannot convert with async dependency or result.");
  }

  // Create an LLVM global with CUBIN extracted from the kernel annotation and
  // obtain a pointer to the first byte in it.
  auto kernel_module = SymbolTable::lookupNearestSymbolFrom<gpu::GPUModuleOp>(
      launch_op, launch_op.getKernelModuleName());
  if (!kernel_module) {
    launch_op.emitOpError() << "cannot find corresponding kernel module.";
    return failure();
  }

  auto binary_attr =
      kernel_module->getAttrOfType<StringAttr>(kGpuBinaryAttrName);
  if (!binary_attr) {
    kernel_module.emitOpError()
        << "missing " << kGpuBinaryAttrName << " attribute";
    return failure();
  }

  Operation* op = launch_op.getOperation();
  Location loc = launch_op.getLoc();

  // Create a global for the module blob.
  StrT name_buffer(kernel_module.getName());
  name_buffer.append("_blob");

  Value module_blob = loadOrCreateGlobalString(
      rewriter, symbol_table_, op, name_buffer.str(), binary_attr.getValue());

  // Make sure the trailing zero is included in the constant.
  auto kernel_name = launch_op.getKernelName().getValue();
  SmallString<128> kernel_name_buffer(kernel_name);
  kernel_name_buffer.push_back('\0');

  // Create a global for the kernel name.
  SmallString<128> kernel_name_global_name_buffer;
  auto kernel_name_global_name =
      (kernel_module.getName() + "_" + kernel_name + "_kernel_name")
          .toStringRef(kernel_name_global_name_buffer);
  Value kernel_name_global = loadOrCreateGlobalString(
      rewriter, symbol_table_, op, kernel_name_global_name,
      kernel_name_buffer.str());

  // The Ral Context is the first argument of the surrounding LLVMFunc.
  Value context_arg =
      launch_op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);
  auto kernel_params =
      generateParamsArray(launch_op, adaptor.getOperands(), rewriter);

  Type llvm_int32_type = IntegerType::get(rewriter.getContext(), 32);
  Value zero = rewriter.create<LLVM::ConstantOp>(loc, llvm_int32_type,
                                                 rewriter.getI32IntegerAttr(0));
  // clang-format off
  // TODO(disc): we use the default stream a.t.m. Implement a stream assignment
  // algo in case necessary.
  SmallVector<Value, 12> newOperands{
      module_blob, /* gpu module string */
      kernel_name_global, /* name of the kernel to launch */
      adaptor.gridSizeX(), adaptor.gridSizeY(), adaptor.gridSizeZ(),
      adaptor.blockSizeX(), adaptor.blockSizeY(), adaptor.blockSizeZ(),
      zero, /* sharedMemBytes */
      zero, /* gpu stream index */
      kernel_params /* params for the kernel to launch */
  };
  // clang-format on

  rewriter.replaceOpWithNewOp<disc_ral::DispatchOp>(
      launch_op, llvm::None, context_arg, newOperands, kRalGpuLaunch, false,
      "cpu");
  return success();
}

class RalToLLVMPass : public RalToLLVMPassBase<RalToLLVMPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

 public:
  void runOnOperation() override {
    ModuleOp m = getOperation();
    SymbolTable symbol_table(m);

    // Populate type conversions.
    MLIRContext* ctx = m.getContext();
    LLVMTypeConverter type_converter(ctx);
    type_converter.addConversion([&](RalExecutionContextType type) {
      return LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
    });

    // Populate patterns.
    RewritePatternSet patterns(&getContext());
    arith::populateArithmeticExpandOpsPatterns(patterns);
    populateStdExpandOpsPatterns(patterns);
    arith::populateArithmeticToLLVMConversionPatterns(type_converter, patterns);
    populateMathToLLVMConversionPatterns(type_converter, patterns);
    populateStdToLLVMConversionPatterns(type_converter, patterns);
    populateDiscRalToLLVMConversionPatterns(&type_converter, &symbol_table,
                                            &patterns);

    // Set target.
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<arith::ArithmeticDialect, StandardOpsDialect,
                             gpu::GPUDialect, disc_ral::RalDialect,
                             math::MathDialect>();
    target.addIllegalOp<UnrealizedConversionCastOp>();
    // Mark modules as legal.
    target.addLegalOp<ModuleOp, gpu::GPUModuleOp>();
    // Do not look into gpu modules, only consider host-side.
    target.markOpRecursivelyLegal<gpu::GPUModuleOp>();

    if (failed(applyFullConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // Finally, strip the GPU modules, as they are no longer needed.
    for (auto op : llvm::make_early_inc_range(m.getOps<gpu::GPUModuleOp>())) {
      op.erase();
    }
  }
};

}  // namespace

void populateDiscRalToLLVMConversionPatterns(LLVMTypeConverter* converter,
                                             SymbolTable* symbol_table,
                                             RewritePatternSet* patterns) {
  // clang-format off
  patterns->insert<
      ConvertLaunchFuncOpToRalCallPattern,
      DispatchOpToLLVMPattern
    >(*converter, *symbol_table);
  // clang-format on
}

std::unique_ptr<OperationPass<ModuleOp>> createRalToLLVMPass() {
  return std::make_unique<RalToLLVMPass>();
}

}  // namespace disc_ral
}  // namespace mlir
