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

#include "Dialect/NVGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct ClusterArriveOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ClusterArriveOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::ClusterArriveOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ClusterArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<triton::nvgpu::ClusterArriveOp>(
        op, op.getRelaxed());
    return success();
  }
};

struct ClusterWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::ClusterWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::ClusterWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::ClusterWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<triton::nvgpu::ClusterWaitOp>(op);
    return success();
  }
};
} // namespace

void mlir::triton::NVIDIA::populateClusterOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ClusterArriveOpConversion>(typeConverter, benefit);
  patterns.add<ClusterWaitOpConversion>(typeConverter, benefit);
  return;
}
