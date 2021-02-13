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

    rewriter.applySignatureConversion(&func.getBody(), conversion);

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
class TFAllocOpConverter : public OpConversionPattern<AllocOp> {
 public:
  using OpConversionPattern<AllocOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AllocOp alloc, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto func = alloc->getParentOfType<FuncOp>();
    if (func.getNumArguments() == 0) {
      return failure();
    }
    Value ctx = func.getArgument(0);
    if (!ctx.getType().isa<OpKernelContextType>()) {
      return failure();
    }
    // Symbolic operands that bind to the symbols of the memref's layout map are
    // not supported by TFAllocOp.
    if (!alloc.symbolOperands().empty()) {
      return failure();
    }
    auto reuse_input_candidates = alloc->getAttrOfType<ArrayAttr>(
        TFAllocOp::kReuseInputCandidatesAttrName);
    auto reuse_output_index =
        alloc->getAttrOfType<IntegerAttr>(TFAllocOp::kReuseOutputAttrName);
    rewriter.replaceOpWithNewOp<TFAllocOp>(alloc, alloc.getType(), ctx,
                                           operands, reuse_input_candidates,
                                           reuse_output_index);
    return success();
  }
};

// Converts std.dealloc to tf_framework.dealloc_raw using OpKernelContextType
// arg of the parent function.
class TFDeallocOpConverter : public OpConversionPattern<DeallocOp> {
 public:
  using OpConversionPattern<DeallocOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DeallocOp dealloc, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto func = dealloc->getParentOfType<FuncOp>();
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
    rewriter.replaceOpWithNewOp<TFDeallocOp>(dealloc, ctx,
                                             transformed.memref());
    return success();
  }
};

// Converts std.assert to tf_framework.assert with using OpKernelContextType
// arg of the parent function.
class TFAssertOpConverter : public OpConversionPattern<AssertOp> {
 public:
  using OpConversionPattern<AssertOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AssertOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto func = op->getParentOfType<FuncOp>();
    if (func.getNumArguments() == 0) {
      return failure();
    }
    Value ctx = func.getArgument(0);
    if (!ctx.getType().isa<OpKernelContextType>()) {
      return failure();
    }
    Location loc = op.getLoc();
    AssertOp::Adaptor transformed(operands, op->getAttrDictionary());

    // Split the block to insert CondBr.
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    Block *split_block = rewriter.splitBlock(
        rewriter.getInsertionBlock(), std::next(rewriter.getInsertionPoint()));

    Block *error_reporting_block =
        rewriter.createBlock(&func.getRegion(), {}, {});
    rewriter.create<ReportErrorOp>(loc, ctx, ErrorCode::INVALID_ARGUMENT,
                                   transformed.msg().getValue());

    SmallVector<Value, 2> null_memrefs;
    for (auto type : func.getType().getResults()) {
      // This can be extended to support various result types if necessary.
      if (!type.isa<UnrankedMemRefType>()) {
        op.emitError("only UnrankedMemRefType results are supported");
        return failure();
      }
      null_memrefs.push_back(rewriter.create<NullMemRefOp>(loc, type));
    }
    rewriter.create<ReturnOp>(loc, null_memrefs);

    rewriter.restoreInsertionPoint(ip);
    rewriter.replaceOpWithNewOp<CondBranchOp>(
        op, transformed.arg(), split_block, llvm::None, error_reporting_block,
        llvm::None);
    return success();
  }
};

}  // namespace

void PopulateEmbedTFFrameworkFunctionAndAllocConversionPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  patterns->insert<TFAllocOpConverter, TFDeallocOpConverter, FuncOpConverter>(
      context);
}

void PopulateEmbedTFFrameworkAssertConversionPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  patterns->insert<TFAssertOpConverter, FuncOpConverter>(context);
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
