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

// Pattern to lower lmhlo.replica_id and lmhlo.partition_id ops to xlir dialect.
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/gpu/xlir_ops.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {
namespace {

struct ReplicaIdRewritePattern
    : tfrt::gpu::StreamifyOpConversionPattern<lmhlo::ReplicaIdOp> {
  using typename tfrt::gpu::StreamifyOpConversionPattern<
      lmhlo::ReplicaIdOp>::OpAdaptor;
  using tfrt::gpu::StreamifyOpConversionPattern<
      lmhlo::ReplicaIdOp>::StreamifyOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      lmhlo::ReplicaIdOp op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    chain = rewriter.create<xla::gpu::ReplicaIdOp>(
        op.getLoc(), stream, adaptor.getOperands().front(), chain);
    rewriter.eraseOp(op);
    return chain;
  }
};

struct PartitionIdRewritePattern
    : tfrt::gpu::StreamifyOpConversionPattern<lmhlo::PartitionIdOp> {
  using typename tfrt::gpu::StreamifyOpConversionPattern<
      lmhlo::PartitionIdOp>::OpAdaptor;
  using tfrt::gpu::StreamifyOpConversionPattern<
      lmhlo::PartitionIdOp>::StreamifyOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      lmhlo::PartitionIdOp op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    chain = rewriter.create<xla::gpu::PartitionIdOp>(
        op.getLoc(), stream, adaptor.getOperands().front(), chain);
    rewriter.eraseOp(op);
    return chain;
  }
};

}  // namespace

void populateReplicaAndPartitionConversionPattern(RewritePatternSet& patterns,
                                                  TypeConverter& converter) {
  patterns.add<ReplicaIdRewritePattern, PartitionIdRewritePattern>(
      converter, patterns.getContext());
}

}  // namespace tensorflow
