/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct MakeTensorPtrOpConversion
    : public ConvertOpToLLVMPattern<triton::MakeTensorPtrOp> {
  using ConvertOpToLLVMPattern<triton::MakeTensorPtrOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // struct { offset0, offset1, shape0, shape1, stride0,
    // stride1, base_ptr};
    auto offsets = adaptor.getOffsets();
    auto shapes = adaptor.getShape();
    auto strides = adaptor.getStrides();
    auto base = adaptor.getBase();
    auto result = op.getResult();

    SmallVector<Value> elems;
    for (auto offset : offsets)
      elems.push_back(offset);
    for (auto shape : shapes)
      elems.push_back(shape);
    for (auto stride : strides)
      elems.push_back(stride);

    elems.push_back(base);

    auto newValue = packLLElements(op.getLoc(), getTypeConverter(), elems,
                                   rewriter, result.getType());
    rewriter.replaceOp(op, newValue);
    return success();
  }
};

struct AdvanceOpConversion : public ConvertOpToLLVMPattern<triton::AdvanceOp> {
  using ConvertOpToLLVMPattern<triton::AdvanceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // struct { offset0, offset1, shape0, shape1, stride0,
    // stride1, base_ptr};
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto ptrType = op.getPtr().getType();
    auto tensorPtr = adaptor.getPtr();

    auto offsets = adaptor.getOffsets();
    auto elems = unpackLLElements(loc, tensorPtr, rewriter);

    SmallVector<Value, 2> newOffsets;

    for (auto [offset, oldOffset] : llvm::zip_first(offsets, elems)) {
      newOffsets.push_back((b.add(offset, oldOffset)));
    }

    for (size_t i = 0; i < newOffsets.size(); ++i) {
      elems[i] = newOffsets[i];
    }

    auto newValue = packLLElements(op.getLoc(), getTypeConverter(), elems,
                                   rewriter, ptrType);
    rewriter.replaceOp(op, newValue);
    return success();
  }
};
} // namespace

void mlir::triton::NVIDIA::populateTensorPtrOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MakeTensorPtrOpConversion>(typeConverter, benefit);
  patterns.add<AdvanceOpConversion>(typeConverter, benefit);
  return;
}
