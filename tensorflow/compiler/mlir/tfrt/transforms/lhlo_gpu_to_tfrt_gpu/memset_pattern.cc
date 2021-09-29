// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Pattern to lower mlir::gpu::memset Ops to tfrt cuda dialect.
#include "tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu/memset_pattern.h"

#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime

namespace tensorflow {
namespace {

// Creates tfrt::gpu::MemsetOp from mlir::gpu::MemSetOp.
struct MemsetRewritePattern
    : tfrt::gpu::GpuAsyncOpConversionPattern<mlir::gpu::MemsetOp> {
  using tfrt::gpu::GpuAsyncOpConversionPattern<
      mlir::gpu::MemsetOp>::GpuAsyncOpConversionPattern;

  FailureOr<Value> matchAndRewriteOp(
      mlir::gpu::MemsetOp op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    if (adaptor.value().getType().getIntOrFloatBitWidth() != 32)
      return rewriter.notifyMatchFailure(op, "expected value to be 32bit.");
    if (!adaptor.dst().getType().isa<tfrt::gpu::BufferType>())
      return rewriter.notifyMatchFailure(op, "expected dst to be gpu buffer.");

    rewriter.eraseOp(op);
    auto memset_op = rewriter.create<tfrt::gpu::MemSetOp>(
        op.getLoc(), adaptor.dst(), adaptor.value(), stream, chain);
    return memset_op.getResult();
  }
};

}  // namespace

void populateMemsetConversionPattern(RewritePatternSet& patterns) {
  patterns.add<MemsetRewritePattern>(patterns.getContext());
}

}  // namespace tensorflow
