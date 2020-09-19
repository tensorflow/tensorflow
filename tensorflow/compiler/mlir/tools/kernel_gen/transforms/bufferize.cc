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

// This file implements logic for translating mixed IR to buffer form.

#include <cstddef>
#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/BufferPlacement.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace kernel_gen {
namespace transforms {

namespace {

class TensorFromElementsOpConverter
    : public BufferAssignmentOpConversionPattern<TensorFromElementsOp> {
 public:
  using BufferAssignmentOpConversionPattern<
      TensorFromElementsOp>::BufferAssignmentOpConversionPattern;

  LogicalResult matchAndRewrite(
      TensorFromElementsOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    ShapedType result_type = op.getType().cast<ShapedType>();
    int number_of_elements = op.elements().size();
    MemRefType memref_type =
        MemRefType::get({number_of_elements}, result_type.getElementType());
    Value result = rewriter.create<AllocaOp>(loc, memref_type);
    for (auto operand : llvm::enumerate(operands)) {
      Value index = rewriter.create<ConstantIndexOp>(loc, operand.index());
      rewriter.create<StoreOp>(loc, operand.value(), result, index);
    }
    rewriter.replaceOp(op, {result});
    return success();
  }
};

class DynamicTensorFromElementsOpConverter
    : public BufferAssignmentOpConversionPattern<DynamicTensorFromElementsOp> {
 public:
  using BufferAssignmentOpConversionPattern<
      DynamicTensorFromElementsOp>::BufferAssignmentOpConversionPattern;

  LogicalResult matchAndRewrite(
      DynamicTensorFromElementsOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Allocate memory on stack.
    Location loc = op.getLoc();
    DynamicTensorFromElementsOp::Adaptor transformed(operands);
    RankedTensorType tensor_ty = op.getType().cast<RankedTensorType>();
    MemRefType memref_type =
        MemRefType::get(tensor_ty.getShape(), tensor_ty.getElementType());
    Value result = rewriter.create<AllocaOp>(loc, memref_type,
                                             transformed.dynamicExtents());

    // Collect loop bounds.
    int64_t rank = tensor_ty.getRank();
    Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<ConstantIndexOp>(loc, 1);
    SmallVector<Value, 4> lower_bounds(rank, zero);
    SmallVector<Value, 4> steps(rank, one);
    SmallVector<Value, 4> upper_bounds;
    int next_dynamic_index = 0;
    for (int i = 0; i < rank; i++) {
      Value ub = tensor_ty.isDynamicDim(i)
                     ? transformed.dynamicExtents()[next_dynamic_index++]
                     : rewriter.create<ConstantIndexOp>(
                           loc, memref_type.getDimSize(i));
      upper_bounds.push_back(ub);
    }

    // Generate tensor elements.
    rewriter.create<scf::ParallelOp>(
        loc, lower_bounds, upper_bounds, steps,
        [&](OpBuilder &b, Location loc, ValueRange ivs) {
          BlockAndValueMapping mapping;
          mapping.map(op.body().getArguments(), ivs);
          for (auto &nested_op : op.getBody()->without_terminator())
            b.clone(nested_op, mapping);
          auto yield_op = llvm::cast<YieldOp>(op.getBody()->getTerminator());
          b.create<StoreOp>(loc, mapping.lookup(yield_op.value()), result, ivs);
          b.create<scf::YieldOp>(loc);
        });

    rewriter.replaceOp(op, {result});
    return success();
  }
};

class TensorLoadOpConversion
    : public BufferAssignmentOpConversionPattern<TensorLoadOp> {
 public:
  using BufferAssignmentOpConversionPattern<
      TensorLoadOp>::BufferAssignmentOpConversionPattern;

  LogicalResult matchAndRewrite(
      TensorLoadOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    TensorLoadOpAdaptor adaptor(operands);
    rewriter.replaceOp(op, {adaptor.memref()});
    return success();
  }
};

class ExtractElementOpConversion
    : public BufferAssignmentOpConversionPattern<ExtractElementOp> {
 public:
  using BufferAssignmentOpConversionPattern<
      ExtractElementOp>::BufferAssignmentOpConversionPattern;

  LogicalResult matchAndRewrite(
      ExtractElementOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ExtractElementOpAdaptor adaptor(operands);

    if (!adaptor.aggregate().getType().isa<MemRefType>()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<LoadOp>(op, adaptor.aggregate(),
                                        adaptor.indices());
    return success();
  }
};

class TensorCastOpConverter
    : public BufferAssignmentOpConversionPattern<TensorCastOp> {
 public:
  using BufferAssignmentOpConversionPattern<
      TensorCastOp>::BufferAssignmentOpConversionPattern;

  LogicalResult matchAndRewrite(
      TensorCastOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto tensor_ty = op.getType().dyn_cast<RankedTensorType>();
    if (!tensor_ty) return failure();

    Value arg = operands.front();
    auto arg_ty = arg.getType().dyn_cast<MemRefType>();
    if (!arg_ty) return failure();

    auto result_ty = converter->convertType(tensor_ty);
    rewriter.replaceOpWithNewOp<MemRefCastOp>(op, arg, result_ty);

    return success();
  }
};

}  // namespace

void populateStandardBufferizePattern(MLIRContext *context,
                                      BufferAssignmentTypeConverter *converter,
                                      OwningRewritePatternList *patterns) {
  patterns->insert<ExtractElementOpConversion, TensorFromElementsOpConverter,
                   DynamicTensorFromElementsOpConverter, TensorLoadOpConversion,
                   TensorCastOpConverter>(context, converter);
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
