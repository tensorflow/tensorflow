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

#include <optional>

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {
namespace {

// Prepends argument type list of the function with an OpKernelContextType arg.
class FuncOpConverter : public OpConversionPattern<func::FuncOp> {
 public:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp func, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const override {
    // Convert function arguments using the provided TypeConverter.
    auto func_type = func.getFunctionType();
    TypeConverter::SignatureConversion conversion(func_type.getNumInputs());

    conversion.addInputs(OpKernelContextType::get(rewriter.getContext()));
    for (auto arg_type : llvm::enumerate(func_type.getInputs())) {
      conversion.addInputs(arg_type.index(), arg_type.value());
    }

    rewriter.applySignatureConversion(&func.getBody(), conversion);

    // Update the signature of the function.
    rewriter.updateRootInPlace(func, [&] {
      func.setType(rewriter.getFunctionType(conversion.getConvertedTypes(),
                                            func_type.getResults()));
    });
    return success();
  }
};

llvm::Optional<Value> FindOpKernelContext(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (func.getNumArguments() == 0) {
    return std::nullopt;
  }
  Value ctx = func.getArgument(0);
  if (!ctx.getType().isa<OpKernelContextType>()) {
    return std::nullopt;
  }
  return ctx;
}

// Converts std.alloc to tf_framework.alloc_raw using OpKernelContextType arg of
// the parent function.
struct AllocOpConverter : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::AllocOp alloc, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    llvm::Optional<Value> ctx = FindOpKernelContext(alloc);
    if (!ctx) return failure();

    // Symbolic operands that bind to the symbols of the memref's layout map are
    // not supported by TFAllocOp.
    if (!alloc.getSymbolOperands().empty()) {
      return failure();
    }
    auto reuse_input_candidates = alloc->getAttrOfType<ArrayAttr>(
        TFAllocOp::kReuseInputCandidatesAttrName);
    auto reuse_output_index =
        alloc->getAttrOfType<IntegerAttr>(TFAllocOp::kReuseOutputAttrName);
    Value buffer = rewriter.replaceOpWithNewOp<TFAllocOp>(
        alloc, alloc.getType(), *ctx, adaptor.getOperands(),
        reuse_input_candidates, reuse_output_index);
    Location loc = buffer.getLoc();
    Value cond = rewriter.create<IsValidMemRefOp>(
        loc, rewriter.getIntegerType(1), buffer);
    rewriter.create<TFAssertOp>(loc, *ctx, cond, ErrorCode::RESOURCE_EXHAUSTED,
                                "failed to allocate memory");
    return success();
  }
};

// Converts std.dealloc to tf_framework.dealloc_raw using OpKernelContextType
// arg of the parent function.
struct DeallocOpConverter : public OpConversionPattern<memref::DeallocOp> {
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::DeallocOp dealloc, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    llvm::Optional<Value> ctx = FindOpKernelContext(dealloc);
    if (!ctx) return failure();

    // Operand with no layout is expected.
    auto operand_memref_type = dealloc.getMemref().getType().cast<MemRefType>();
    if (!operand_memref_type.getLayout().isIdentity()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<TFDeallocOp>(dealloc, *ctx,
                                             adaptor.getMemref());
    return success();
  }
};

// Converts std.assert to tf_framework.assert with using OpKernelContextType
// arg of the parent function.
struct AssertOpConverter : public OpConversionPattern<cf::AssertOp> {
 public:
  using OpConversionPattern<cf::AssertOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cf::AssertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    llvm::Optional<Value> ctx = FindOpKernelContext(op);
    if (!ctx) return failure();
    rewriter.replaceOpWithNewOp<TFAssertOp>(op, *ctx, adaptor.getArg(),
                                            ErrorCode::INVALID_ARGUMENT,
                                            adaptor.getMsg());
    return success();
  }
};

// Amends `tf_framework.jit_execute` with the newly introduced OpKernelContext.
struct JITExecuteOpConverter : public OpConversionPattern<JITExecuteOp> {
  using OpConversionPattern<JITExecuteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      JITExecuteOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const override {
    llvm::Optional<Value> ctx = FindOpKernelContext(op);
    if (!ctx) return failure();
    rewriter.replaceOpWithNewOp<JITExecuteOp>(
        op, op.getResult().getType(), *ctx, op.getCallable(), op.getInputs());
    return success();
  }
};

// Amends `tf_framework.jit_compile_from_str` with the newly introduced
// OpKernelContext.
struct JITCompileFromStrOpConverter
    : public OpConversionPattern<JITCompileFromStrOp> {
  using OpConversionPattern<JITCompileFromStrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      JITCompileFromStrOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const override {
    llvm::Optional<Value> ctx = FindOpKernelContext(op);
    if (!ctx) return failure();
    rewriter.replaceOpWithNewOp<JITCompileFromStrOp>(
        op, rewriter.getType<JITCallableType>(), *ctx, op->getAttrs());
    return success();
  }
};

}  // namespace

void PopulateEmbedTFFrameworkAssertPattern(RewritePatternSet *patterns) {
  patterns->add<AssertOpConverter>(patterns->getContext());
}

void PopulateEmbedTFFrameworkPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
      AllocOpConverter,
      AssertOpConverter,
      DeallocOpConverter,
      FuncOpConverter,
      JITCompileFromStrOpConverter,
      JITExecuteOpConverter>(patterns->getContext());
  // clang-format on
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
