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
    : tfrt::gpu::StreamifyOpConversionPattern<lmhlo::TriangularSolveOp> {
  using tfrt::gpu::StreamifyOpConversionPattern<
      lmhlo::TriangularSolveOp>::StreamifyOpConversionPattern;
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

    auto transpose_or =
        xla::ConvertTranspose(mlir::mhlo::stringifyTranspose(op.transpose_a()));
    if (!transpose_or.ok()) {
      return rewriter.notifyMatchFailure(op,
                                         transpose_or.status().error_message());
    }
    tfrt::gpu::wrapper::BlasOperation trans = [&] {
      switch (transpose_or.ValueOrDie()) {
        case xla::TriangularSolveOptions::NO_TRANSPOSE:
          return kBlasOperationNone;
        case xla::TriangularSolveOptions::TRANSPOSE:
          return kBlasOperationTranspose;
        case xla::TriangularSolveOptions::ADJOINT:
          return kBlasOperationConjTranspose;
        default:
          LOG(ERROR) << "Invalid triangular solve transpose value "
                     << transpose_or.ValueOrDie();
          return kBlasOperationNone;
      }
    }();

    Location loc = op->getLoc();
    chain = rewriter.create<tfrt::gpu::MemCopyOp>(loc, adaptor.output(),
                                                  adaptor.b(), stream, chain);

    Value context = rewriter.create<tfrt::gpu::StreamGetContextOp>(loc, stream);
    auto handle = rewriter.create<tfrt::gpu::BlasCreateOp>(loc, context);

    auto side_mode = op.left_side() ? kBlasSideLeft : kBlasSideRight;
    auto fill_mode = op.lower() ? kBlasFillModeLower : kBlasFillModeUpper;
    auto diag_type = op.unit_diagonal() ? kBlasDiagUnit : kBlasDiagNonUnit;

    const xla::Shape b_shape = xla::gpu::GetShape(op.b());
    int64_t m_value = b_shape.dimensions(b_shape.rank() - 2);
    int64_t n_value = b_shape.dimensions(b_shape.rank() - 1);
    auto m = rewriter.create<tfrt::compiler::ConstantI32Op>(loc, m_value);
    auto n = rewriter.create<tfrt::compiler::ConstantI32Op>(loc, n_value);

    mlir::Type element_type =
        op.output().getType().cast<mlir::MemRefType>().getElementType();
    auto data_type = MlirTypeToBlasDataType(element_type);

    auto alpha = MakeScalingFactorConstant(
        rewriter, loc, element_type, llvm::APFloat(1.0), llvm::APFloat(0.0));

    // If side_mode == LEFT, the triangular linear system to be solved is
    // op(A).X = alpha*B. Since X is an m-by-n matrix, the minimum height of A
    // is m (it is m here). OTOH, if side_mode == RIGHT, we're solving
    // X.op(A) = alpha*B, and the minimum height of A is n (it is n here).
    auto height_a = rewriter.create<tfrt::compiler::ConstantI32Op>(
        loc, side_mode == kBlasSideLeft ? m_value : n_value);
    auto height_b =
        rewriter.create<tfrt::compiler::ConstantI32Op>(loc, m_value);

    int64_t batch_count = std::accumulate(
        b_shape.dimensions().begin(), b_shape.dimensions().end() - 2,
        int64_t{1}, std::multiplies<int64_t>());
    auto batch =
        rewriter.create<tfrt::compiler::ConstantI32Op>(loc, batch_count);

    chain = rewriter.create<tfrt::gpu::BlasTrsmBatchOp>(
        loc, handle, stream, side_mode, fill_mode, trans, diag_type, m, n,
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
