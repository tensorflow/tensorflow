/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/convert_graph_region_op.h"

#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/de_bufferization.h"
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_dialect.h"
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_ops.h"

namespace xla {
namespace gpu {
namespace {

using namespace mlir;  // NOLINT

//===----------------------------------------------------------------------===//
// Converts xla_gpu.graph.region op to a xla_gpu.graph.dispatch
//===----------------------------------------------------------------------===//

struct ConvertGraphRegionOp : public OpConversionPattern<GraphRegionOp> {
  ConvertGraphRegionOp(TypeConverter &converter, MLIRContext *ctx,
                       DeBufferization &state)
      : OpConversionPattern(converter, ctx), state(state) {}

  LogicalResult matchAndRewrite(
      GraphRegionOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto dispatch = b.create<GraphDispatchOp>();
    Block *body = &dispatch.getBody().emplaceBlock();
    body->addArgument(rewriter.getType<GraphType>(), op.getLoc());
    rewriter.mergeBlocks(&op.getBody().front(), body);

    // Set up buffer to tensor remapping inside nested region.
    UsedBuffers bufs = getUsedBuffers(body);
    for (auto r : bufs.read)
      state.remapped[body][r] = state.remapped[op->getBlock()][r];
    for (auto w : bufs.write)
      state.remapped[body][w] = state.remapped[op->getBlock()][w];

    return success();
  }

  DeBufferization &state;
};

}  // namespace

void populateGraphRegionConversionPatterns(RewritePatternSet &patterns,
                                           TypeConverter &converter,
                                           DeBufferization &state) {
  auto *ctx = patterns.getContext();
  patterns.insert<ConvertGraphRegionOp>(converter, ctx, state);
}

}  // namespace gpu
}  // namespace xla
