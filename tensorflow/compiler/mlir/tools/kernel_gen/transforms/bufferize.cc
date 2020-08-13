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
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
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

}  // namespace

void populateStandardBufferizePattern(MLIRContext *context,
                                      BufferAssignmentPlacer *bufferAssignment,
                                      TypeConverter *converter,
                                      OwningRewritePatternList *patterns) {
  patterns->insert<ExtractElementOpConversion, TensorFromElementsOpConverter,
                   TensorLoadOpConversion>(context, bufferAssignment,
                                           converter);
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
