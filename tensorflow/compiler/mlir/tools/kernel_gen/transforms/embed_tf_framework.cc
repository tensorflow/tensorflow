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

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {
namespace {

// Prepends argument type list of the function with an OpKernelContextType arg.
class FuncOpConverter : public OpConversionPattern<FuncOp> {
 public:
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FuncOp func, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Convert function arguments using the provided TypeConverter.
    auto func_type = func.getType();
    TypeConverter::SignatureConversion conversion(func_type.getNumInputs());

    conversion.addInputs(OpKernelContextType::get(rewriter.getContext()));
    for (auto arg_type : llvm::enumerate(func_type.getInputs())) {
      conversion.addInputs(arg_type.index(), arg_type.value());
    }

    TypeConverter type_converter;
    if (failed(rewriter.convertRegionTypes(&func.getBody(), type_converter,
                                           &conversion))) {
      return failure();
    }

    // Update the signature of the function.
    rewriter.updateRootInPlace(func, [&] {
      func.setType(rewriter.getFunctionType(conversion.getConvertedTypes(),
                                            func_type.getResults()));
    });
    return success();
  }
};

// Converts std.alloc to tf_framework.alloc_raw using OpKernelContextType arg of
// the parent function.
class AllocOpConverter : public OpConversionPattern<AllocOp> {
 public:
  using OpConversionPattern<AllocOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AllocOp alloc, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto func = alloc.getParentOfType<FuncOp>();
    if (func.getNumArguments() == 0) {
      return failure();
    }
    Value ctx = func.getArgument(0);
    if (!ctx.getType().isa<OpKernelContextType>()) {
      return failure();
    }
    // Symbolic operands that bind to the symbols of the memref's layout map are
    // not supported by AllocRawOp.
    if (alloc.getNumSymbolicOperands() != 0) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<AllocRawOp>(alloc, alloc.getType(), ctx,
                                            operands);
    return success();
  }
};

// Converts std.dealloc to tf_framework.dealloc_raw using OpKernelContextType
// arg of the parent function.
class DeallocOpConverter : public OpConversionPattern<DeallocOp> {
 public:
  using OpConversionPattern<DeallocOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DeallocOp dealloc, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    FuncOp func = dealloc.getParentOfType<FuncOp>();
    if (func.getNumArguments() == 0) {
      return failure();
    }
    Value ctx = func.getArgument(0);
    if (!ctx.getType().isa<OpKernelContextType>()) {
      return failure();
    }
    // Operand with no layout is expected.
    auto operand_memref_type = dealloc.memref().getType().cast<MemRefType>();
    if (!operand_memref_type.getAffineMaps().empty()) {
      return failure();
    }
    DeallocOp::Adaptor transformed(operands);
    rewriter.replaceOpWithNewOp<DeallocRawOp>(dealloc, ctx,
                                              transformed.memref());
    return success();
  }
};

}  // namespace

void PopulateEmbedTFFrameworkConversionPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  patterns->insert<AllocOpConverter, DeallocOpConverter, FuncOpConverter>(
      context);
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
