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

// Pattern to lower lmhlo.triangular_solve op to tfrt_gpu dialect.
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.h"
#include "tensorflow/compiler/mlir/xla/attribute_exporter.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/cublas_wrapper.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {
namespace {

struct TriangularSolveRewritePattern
    : tfrt::gpu::GpuAsyncOpConversionPattern<lmhlo::TriangularSolveOp> {
  using tfrt::gpu::GpuAsyncOpConversionPattern<
      lmhlo::TriangularSolveOp>::GpuAsyncOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      lmhlo::TriangularSolveOp op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    auto has_fortran_layout = [](mlir::DenseIntElementsAttr layout_attr) {
      int64_t n = layout_attr.getNumElements();
      return layout_attr.getValues<int64_t>()[0] == n - 2 &&
             layout_attr.getValues<int64_t>()[1] == n - 1;
    };
    if (!has_fortran_layout(op.layout_a()) ||
        !has_fortran_layout(op.layout_b()) ||
        !has_fortran_layout(op.layout_output()))
      return rewriter.notifyMatchFailure(op, "expected fortran layout");

    auto transpose_or = xla::ConvertTranspose(op.transpose_a());
    if (!transpose_or.ok()) {
      return rewriter.notifyMatchFailure(op,
                                         transpose_or.status().error_message());
    }
    cublasOperation_t trans = [&] {
      switch (transpose_or.ValueOrDie()) {
        case xla::TriangularSolveOptions::NO_TRANSPOSE:
          return CUBLAS_OP_N;
        case xla::TriangularSolveOptions::TRANSPOSE:
          return CUBLAS_OP_T;
        case xla::TriangularSolveOptions::ADJOINT:
          return CUBLAS_OP_C;
        default:
          LOG(ERROR) << "Invalid triangular solve transpose value "
                     << transpose_or.ValueOrDie();
          return CUBLAS_OP_N;
      }
    }();

    chain = rewriter.create<tfrt::gpu::MemCopyOp>(op.getLoc(), adaptor.output(),
                                                  adaptor.b(), stream, chain);

    auto handle = rewriter.create<tfrt::gpu::BlasCreateOp>(op.getLoc(), stream);

    cublasSideMode_t side_mode =
        op.left_side() ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t fill_mode =
        op.lower() ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cublasDiagType_t diag_type =
        op.unit_diagonal() ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

    const xla::Shape b_shape = xla::gpu::GetShape(op.b());
    int64_t m_value = b_shape.dimensions(b_shape.rank() - 2);
    int64_t n_value = b_shape.dimensions(b_shape.rank() - 1);
    auto m =
        rewriter.create<tfrt::compiler::ConstantI32Op>(op.getLoc(), m_value);
    auto n =
        rewriter.create<tfrt::compiler::ConstantI32Op>(op.getLoc(), n_value);

    mlir::Type element_type =
        op.output().getType().cast<mlir::MemRefType>().getElementType();
    auto data_type = MlirTypeToCudaDataType(element_type);

    auto alpha =
        MakeScalingFactorConstant(rewriter, op.getLoc(), element_type,
                                  llvm::APFloat(1.0), llvm::APFloat(0.0));

    // If side_mode == LEFT, the triangular linear system to be solved is
    // op(A).X = alpha*B. Since X is an m-by-n matrix, the minimum height of A
    // is m (it is m here). OTOH, if side_mode == RIGHT, we're solving
    // X.op(A) = alpha*B, and the minimum height of A is n (it is n here).
    auto height_a = rewriter.create<tfrt::compiler::ConstantI32Op>(
        op.getLoc(), side_mode == CUBLAS_SIDE_LEFT ? m_value : n_value);
    auto height_b =
        rewriter.create<tfrt::compiler::ConstantI32Op>(op.getLoc(), m_value);

    int64_t batch_count = std::accumulate(
        b_shape.dimensions().begin(), b_shape.dimensions().end() - 2,
        int64_t{1}, std::multiplies<int64_t>());
    auto batch = rewriter.create<tfrt::compiler::ConstantI32Op>(op.getLoc(),
                                                                batch_count);

    chain = rewriter.create<tfrt::gpu::BlasTrsmBatchOp>(
        op.getLoc(), handle, side_mode, fill_mode, trans, diag_type, m, n,
        data_type, alpha, adaptor.a(), height_a, adaptor.output(), height_b,
        batch, chain);
    rewriter.eraseOp(op);
    return chain;
  }
};

}  // namespace

void populateTriangularSolveConversionPattern(RewritePatternSet& patterns,
                                              TypeConverter& converter) {
  patterns.add<TriangularSolveRewritePattern>(converter, patterns.getContext());
}

}  // namespace tensorflow
