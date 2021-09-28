// Copyright 2021 The TensorFlow Runtime Authors
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

// Pattern to lower gpu.memcpy ops to tfrt_gpu.mem.copy.
#include "tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu/memcpy_pattern.h"

#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime

namespace tensorflow {
namespace {

// Creates tfrt::gpu::MemCopyOp from mlir::gpu::MemcpyOp.
struct MemcpyRewritePattern
    : tfrt::gpu::GpuAsyncOpConversionPattern<mlir::gpu::MemcpyOp> {
  using tfrt::gpu::GpuAsyncOpConversionPattern<
      mlir::gpu::MemcpyOp>::GpuAsyncOpConversionPattern;

  FailureOr<Value> matchAndRewriteOp(
      mlir::gpu::MemcpyOp op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    if (!adaptor.src().getType().isa<tfrt::gpu::BufferType>() ||
        !adaptor.dst().getType().isa<tfrt::gpu::BufferType>()) {
      return rewriter.notifyMatchFailure(op, "expected buffer operands");
    }
    rewriter.eraseOp(op);
    auto copy_op = rewriter.create<tfrt::gpu::MemCopyOp>(
        op.getLoc(), adaptor.dst(), adaptor.src(), stream, chain);
    return copy_op.getResult();
  }
};

}  // namespace

void populateMemcpyConversionPattern(RewritePatternSet& patterns) {
  patterns.add<MemcpyRewritePattern>(patterns.getContext());
}

}  // namespace tensorflow
