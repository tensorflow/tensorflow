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

#include "mlir/Transforms/Bufferize.h"  // from @llvm-project

#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

class BufferizeConstantOp : public OpConversionPattern<ConstantOp> {
 public:
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // We only need to bufferize tensor constants.
    Location loc = op.getLoc();
    auto result_type = op.getType().dyn_cast<RankedTensorType>();
    if (!result_type || !result_type.hasStaticShape() ||
        result_type.getRank() != 1)
      return failure();

    auto memref_type = MemRefType::get({result_type.getNumElements()},
                                       result_type.getElementType());
    Value buffer = rewriter.create<AllocaOp>(loc, memref_type);

    auto elements_attr = op.getValue().dyn_cast<DenseElementsAttr>();
    bool all_same_elems = elements_attr.isSplat();
    Value value;
    if (all_same_elems)
      value = rewriter.create<ConstantOp>(loc, elements_attr.getSplatValue());
    for (auto en : llvm::enumerate(elements_attr.getAttributeValues())) {
      if (!all_same_elems) value = rewriter.create<ConstantOp>(loc, en.value());
      Value index = rewriter.create<ConstantIndexOp>(loc, en.index());
      rewriter.create<StoreOp>(loc, value, buffer, index);
    }
    rewriter.replaceOp(op, {buffer});
    return success();
  }
};

class BufferizeDimOp : public OpConversionPattern<DimOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      DimOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    DimOp::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<DimOp>(op, adaptor.memrefOrTensor(),
                                       adaptor.index());
    return success();
  }
};

class BufferizeRankOp : public OpConversionPattern<RankOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      RankOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    RankOp::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<RankOp>(op, adaptor.memrefOrTensor());
    return success();
  }
};
}  // namespace

void populateExtraStdBufferizePattern(MLIRContext *context,
                                      BufferizeTypeConverter *converter,
                                      OwningRewritePatternList *patterns) {
  patterns->insert<BufferizeConstantOp, BufferizeDimOp,
                   BufferizeRankOp>(*converter, context);
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
