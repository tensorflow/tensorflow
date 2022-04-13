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

// Pattern to lower lmhlo_gpu.cholesky op to tfrt_gpu dialect.
#include <functional>
#include <string>

#include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/cublas_wrapper.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {
namespace {

struct CholeskyRewritePattern
    : tfrt::gpu::StreamifyOpConversionPattern<lmhlo_gpu::CholeskyOp> {
  using tfrt::gpu::StreamifyOpConversionPattern<
      lmhlo_gpu::CholeskyOp>::StreamifyOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      lmhlo_gpu::CholeskyOp op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op->getLoc();
    chain = rewriter.create<tfrt::gpu::MemCopyOp>(
        loc, adaptor.output(), adaptor.input(), stream, chain);

    Value context = rewriter.create<tfrt::gpu::StreamGetContextOp>(loc, stream);
    auto handle = rewriter.create<tfrt::gpu::SolverCreateOp>(loc, context);

    auto fill_mode = op.is_lower() ? kBlasFillModeLower : kBlasFillModeUpper;

    mlir::Type element_type =
        op.input().getType().cast<mlir::MemRefType>().getElementType();
    auto data_type = MlirTypeToBlasDataType(element_type);

    const xla::Shape shape = xla::gpu::GetShape(op.input());
    int rank = shape.dimensions_size();
    assert(rank >= 2);
    auto n = rewriter.create<tfrt::compiler::ConstantI32Op>(
        loc, shape.dimensions(rank - 1));

    const auto& dims = shape.dimensions();
    int64_t batch_size = std::accumulate(
        dims.begin(), dims.end() - 2, int64_t{1}, std::multiplies<int64_t>());
    auto batch =
        rewriter.create<tfrt::compiler::ConstantI32Op>(loc, batch_size);

    chain = rewriter.create<tfrt::gpu::SolverPotrfBatchOp>(
        loc, handle, stream, fill_mode, n, data_type, adaptor.output(), n,
        adaptor.info(), batch, chain);
    rewriter.eraseOp(op);
    return chain;
  }
};

}  // namespace

void populateCholeskyConversionPattern(RewritePatternSet& patterns,
                                       TypeConverter& converter) {
  patterns.add<CholeskyRewritePattern>(converter, patterns.getContext());
}

}  // namespace tensorflow
