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

//===- gemm_pattern.cc
//---------------------------------------------------------===//
//
// Pattern to lower lhlogpu_gemm Ops to tfrt cuda dialect.
//
//===----------------------------------------------------------------------===//
#include <assert.h>
#include <stdint.h>

#include <type_traits>
#include <utility>

#include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/cublas_wrapper.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/rocblas_wrapper.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {
namespace {

// This struct contains the metadata of a matrix, e.g., its base address and
// dimensions.
struct MatrixDescriptor {
  Value data;
  bool transpose;  // Whether this matrix needs to be transposed.
  int64_t num_rows;
  int64_t num_cols;
  int64_t stride;
};

MatrixDescriptor GetMatrixDesc(const xla::gpu::MatrixLayout& layout,
                               Value data) {
  // TODO(cjfj): Add support for batch not in most major physical dimension.
  CHECK((layout.batch_stride == 0) ||
        (layout.batch_stride == layout.num_rows * layout.num_cols));
  bool transpose = layout.order != xla::gpu::MatrixLayout::Order::kColumnMajor;
  return {
      data,
      transpose,
      transpose ? layout.num_cols : layout.num_rows,
      transpose ? layout.num_rows : layout.num_cols,
      layout.batch_stride,
  };
}

void TransposeMatrixDesc(MatrixDescriptor& matrix_desc) {
  matrix_desc.transpose = !matrix_desc.transpose;
}

void MakeBlasGemmCompatible(MatrixDescriptor& lhs, MatrixDescriptor& rhs,
                            MatrixDescriptor& output) {
  // BLAS GeMM doesn't support transposed output, but we can use the identity:
  // C^T = (A @ B)^T = B^T @ A^T.
  if (output.transpose) {
    std::swap(lhs, rhs);
    lhs.transpose = !lhs.transpose;
    rhs.transpose = !rhs.transpose;
    output.transpose = !output.transpose;
  }
}

Value GetBias(lmhlo_gpu::GEMMOpAdaptor op) { return nullptr; }
Value GetBias(lmhlo_gpu::GEMM_BiasOpAdaptor op) { return op.bias(); }

// Match GEMM auto-tuning, see ComputationTypeFromPrimitive()
Type MlirComputationType(Type element_type,
                         ConversionPatternRewriter& rewriter) {
  if (element_type.isF16() || element_type.isBF16())
    return rewriter.getF32Type();

#if !TENSORFLOW_USE_ROCM
  if (auto complex_type = element_type.dyn_cast<mlir::ComplexType>())
    return complex_type.getElementType();
#endif

  return element_type;
}

// Gets the platform specific Gemm algorithm value.
template <class GemmOp>
tfrt::gpu::wrapper::BlasGemmAlgo GetBlasGemmAlgoOrDefault(GemmOp op) {
  if (!op.algorithm().hasValue()) return kBlasGemmDefaultAlgo;
  return {static_cast<int>(op.algorithm().getValue()), kGpuTargetPlatform};
}

// Returns the platform specific matrix transpose operation value.
tfrt::gpu::wrapper::BlasOperation MatrixTransposeToBlasOperation(
    bool transpose) {
  return transpose ? kBlasOperationTranspose : kBlasOperationNone;
}

// Create all the Ops necessary for the GEMM operation, including the GEMM
// operation itself.
template <class GemmOp>
Value CreateTfrtOps(GemmOp op, typename GemmOp::Adaptor adaptor, Value chain,
                    Value stream, mlir::Type input_type, mlir::Type output_type,
                    int64_t batch_size, const MatrixDescriptor& lhs,
                    const MatrixDescriptor& rhs, const MatrixDescriptor& output,
                    xla::complex128 alpha, double beta,
                    ConversionPatternRewriter& rewriter) {
  auto loc = op.getLoc();
  if (auto bias = GetBias(adaptor)) {
    chain = rewriter.create<tfrt::gpu::MemCopyOp>(loc, adaptor.output(), bias,
                                                  stream, chain);
  }

  auto k_val = lhs.transpose ? lhs.num_rows : lhs.num_cols;

  const Type mlir_compute_type = MlirComputationType(output_type, rewriter);

  auto m = rewriter.create<tfrt::compiler::ConstantI32Op>(loc, output.num_rows);
  auto n = rewriter.create<tfrt::compiler::ConstantI32Op>(loc, output.num_cols);
  auto k = rewriter.create<tfrt::compiler::ConstantI32Op>(loc, k_val);

  // Scale type must match compute type, except for complex types, where
  // it must match the output type
  const Type mlir_scale_type =
      output_type.isa<mlir::ComplexType>() ? output_type : mlir_compute_type;

  auto const_alpha = MakeScalingFactorConstant(rewriter, loc, mlir_scale_type,
                                               llvm::APFloat(alpha.real()),
                                               llvm::APFloat(alpha.imag()));
  auto const_beta = MakeScalingFactorConstant(
      rewriter, loc, mlir_scale_type, llvm::APFloat(beta), llvm::APFloat(0.));

  auto lda = rewriter.create<tfrt::compiler::ConstantI32Op>(loc, lhs.num_rows);
  auto ldb = rewriter.create<tfrt::compiler::ConstantI32Op>(loc, rhs.num_rows);
  auto ldc =
      rewriter.create<tfrt::compiler::ConstantI32Op>(loc, output.num_rows);

  tfrt::gpu::wrapper::BlasGemmAlgo algorithm = GetBlasGemmAlgoOrDefault(op);
  auto algo = rewriter.create<tfrt::gpu::BlasGemmAlgoOp>(loc, algorithm);

  Value context = rewriter.create<tfrt::gpu::StreamGetContextOp>(loc, stream);
  auto handle = rewriter.create<tfrt::gpu::BlasCreateOp>(loc, context);

  auto lhs_op = MatrixTransposeToBlasOperation(lhs.transpose);
  auto rhs_op = MatrixTransposeToBlasOperation(rhs.transpose);

  const auto input_data_type = MlirTypeToBlasDataType(input_type);
  const auto output_data_type = MlirTypeToBlasDataType(output_type);
  const auto compute_type = MlirTypeToBlasComputeType(mlir_compute_type);
  if (batch_size != 1) {
    auto lhs_stride =
        rewriter.create<tfrt::compiler::ConstantI64Op>(loc, lhs.stride);
    auto rhs_stride =
        rewriter.create<tfrt::compiler::ConstantI64Op>(loc, rhs.stride);
    auto output_stride =
        rewriter.create<tfrt::compiler::ConstantI64Op>(loc, output.stride);
    auto batch =
        rewriter.create<tfrt::compiler::ConstantI32Op>(loc, batch_size);
    return rewriter
        .create<tfrt::gpu::BlasGemmBatchExOp>(
            loc, chain.getType(), handle, stream, lhs_op, rhs_op, m, n, k,
            const_alpha, lhs.data, input_data_type, lda, lhs_stride, rhs.data,
            input_data_type, ldb, rhs_stride, const_beta, output.data,
            output_data_type, ldc, output_stride, batch, compute_type, algo,
            chain)
        .getResult();
  }

  return rewriter
      .create<tfrt::gpu::BlasGemmOp>(
          loc, chain.getType(), handle, stream, lhs_op, rhs_op, m, n, k,
          const_alpha, lhs.data, input_data_type, lda, rhs.data,
          input_data_type, ldb, const_beta, output.data, output_data_type, ldc,
          compute_type, algo, chain)
      .getResult();
}

template <class GemmOp>
FailureOr<Value> GemmOpConversionRewrite(GemmOp op,
                                         typename GemmOp::Adaptor adaptor,
                                         Value chain, Value stream,
                                         ConversionPatternRewriter& rewriter) {
  auto get_element_type = [](Value value) {
    return value.getType().cast<mlir::MemRefType>().getElementType();
  };

  if (get_element_type(op.lhs()) != get_element_type(op.rhs())) {
    return rewriter.notifyMatchFailure(op, "Input element type mismatch.");
  }

  bool use_cublaslt = false;  // FIXME(cjfj).
  StatusOr<xla::gpu::GemmConfig> config =
      xla::gpu::GemmConfig::For(op, use_cublaslt);

  if (!config.ok())
    return rewriter.notifyMatchFailure(op, config.status().ToString());

  MatrixDescriptor lhs = GetMatrixDesc(config->lhs_layout, adaptor.lhs());
  MatrixDescriptor rhs = GetMatrixDesc(config->rhs_layout, adaptor.rhs());
  MatrixDescriptor output =
      GetMatrixDesc(config->output_layout, adaptor.output());
  int64_t batch_size = config->output_layout.batch_size;

  MakeBlasGemmCompatible(lhs, rhs, output);

  return CreateTfrtOps(op, adaptor, chain, stream, get_element_type(op.lhs()),
                       get_element_type(op.output()), batch_size, lhs, rhs,
                       output, config->alpha, config->beta, rewriter);
}

template <class GemmOpType>
struct GemmRewritePattern
    : tfrt::gpu::StreamifyOpConversionPattern<GemmOpType> {
  using typename tfrt::gpu::StreamifyOpConversionPattern<GemmOpType>::OpAdaptor;
  using tfrt::gpu::StreamifyOpConversionPattern<
      GemmOpType>::StreamifyOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      GemmOpType op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    auto result = GemmOpConversionRewrite(op, adaptor, chain, stream, rewriter);
    if (succeeded(result)) rewriter.eraseOp(op);
    return result;
  }
};

}  // namespace

void populateGemmConversionPattern(RewritePatternSet& patterns,
                                   TypeConverter& converter) {
  patterns.add<GemmRewritePattern<lmhlo_gpu::GEMMOp>,
               GemmRewritePattern<lmhlo_gpu::GEMM_BiasOp>>(
      converter, patterns.getContext());
}

}  // namespace tensorflow
