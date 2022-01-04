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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/util.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/cublas_wrapper.h"  // from @tf_runtime
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

FloatAttr GetBeta(lmhlo_gpu::GEMMOp op) { return nullptr; }
Value GetBias(lmhlo_gpu::GEMMOpAdaptor op) { return nullptr; }

FloatAttr GetBeta(lmhlo_gpu::GEMM_BiasOp op) { return op.betaAttr(); }
Value GetBias(lmhlo_gpu::GEMM_BiasOpAdaptor op) { return op.bias(); }

// Match GEMM auto-tuning, see ComputationTypeFromPrimitive()
Type MlirComputationType(Type element_type,
                         ConversionPatternRewriter& rewriter) {
  if (element_type.isF16()) {
    return rewriter.getF32Type();
  } else if (auto complex_type = element_type.dyn_cast<mlir::ComplexType>()) {
    return complex_type.getElementType();
  } else {
    return element_type;
  }
}

// Create all the Ops necessary for the GEMM operation, including the GEMM
// operation itself.
template <class GemmOp>
Value CreateTfrtOps(GemmOp op, typename GemmOp::Adaptor adaptor, Value chain,
                    Value stream, int64_t batch_size, mlir::Type element_type,
                    MatrixDescriptor lhs_matrix, MatrixDescriptor rhs_matrix,
                    MatrixDescriptor output_matrix, llvm::APFloat alpha_real,
                    llvm::APFloat alpha_imaginary, llvm::APFloat beta_real,
                    cublasGemmAlgo_t algorithm,
                    ConversionPatternRewriter& rewriter) {
  auto loc = op.getLoc();
  if (auto bias = GetBias(adaptor)) {
    chain = rewriter.create<tfrt::gpu::MemCopyOp>(loc, adaptor.output(), bias,
                                                  stream, chain);
  }

  auto k_val = lhs_matrix.transpose ? lhs_matrix.num_rows : lhs_matrix.num_cols;

  const Type mlir_compute_type = MlirComputationType(element_type, rewriter);

  auto m = rewriter.create<tfrt::compiler::ConstantI32Op>(
      loc, output_matrix.num_rows);
  auto n = rewriter.create<tfrt::compiler::ConstantI32Op>(
      loc, output_matrix.num_cols);
  auto k = rewriter.create<tfrt::compiler::ConstantI32Op>(loc, k_val);

  // Scale type must match compute type, except for complex types, where
  // it must match the element type
  const Type mlir_scale_type =
      element_type.isa<mlir::ComplexType>() ? element_type : mlir_compute_type;

  auto const_alpha = MakeScalingFactorConstant(rewriter, loc, mlir_scale_type,
                                               alpha_real, alpha_imaginary);

  auto lda =
      rewriter.create<tfrt::compiler::ConstantI32Op>(loc, lhs_matrix.num_rows);
  auto ldb =
      rewriter.create<tfrt::compiler::ConstantI32Op>(loc, rhs_matrix.num_rows);

  llvm::APFloat fp_zero = APFloat::getZero(alpha_imaginary.getSemantics());
  auto const_beta = MakeScalingFactorConstant(rewriter, loc, mlir_scale_type,
                                              beta_real, fp_zero);

  auto ldc = rewriter.create<tfrt::compiler::ConstantI32Op>(
      loc, output_matrix.num_rows);

  auto algo = rewriter.create<tfrt::gpu::BlasGemmAlgoOp>(loc, algorithm);

  auto blas_handle = rewriter.create<tfrt::gpu::BlasCreateOp>(loc, stream);

  auto lhs_op = lhs_matrix.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto rhs_op = rhs_matrix.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

  const auto data_type = MlirTypeToCudaDataType(element_type);
  const auto compute_type = MlirTypeToCublasComputeType(mlir_compute_type);

  if (batch_size != 1) {
    auto lhs_stride =
        rewriter.create<tfrt::compiler::ConstantI64Op>(loc, lhs_matrix.stride);
    auto rhs_stride =
        rewriter.create<tfrt::compiler::ConstantI64Op>(loc, rhs_matrix.stride);
    auto output_stride = rewriter.create<tfrt::compiler::ConstantI64Op>(
        loc, output_matrix.stride);
    auto batch =
        rewriter.create<tfrt::compiler::ConstantI32Op>(loc, batch_size);
    return rewriter
        .create<tfrt::gpu::BlasGemmBatchExOp>(
            loc, chain.getType(), blas_handle, lhs_op, rhs_op, m, n, k,
            const_alpha, lhs_matrix.data, data_type, lda, lhs_stride,
            rhs_matrix.data, data_type, ldb, rhs_stride, const_beta,
            output_matrix.data, data_type, ldc, output_stride, batch,
            compute_type, algo, chain)
        .getResult();
  }

  return rewriter
      .create<tfrt::gpu::BlasGemmOp>(
          loc, chain.getType(), blas_handle, lhs_op, rhs_op, m, n, k,
          const_alpha, lhs_matrix.data, data_type, lda, rhs_matrix.data,
          data_type, ldb, const_beta, output_matrix.data, data_type, ldc,
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
  mlir::Type element_type = get_element_type(op.output());
  if (element_type != get_element_type(op.lhs()) ||
      element_type != get_element_type(op.rhs())) {
    return rewriter.notifyMatchFailure(op, "Element type mismatch.");
  }

  const xla::Shape output_shape = xla::gpu::GetShape(op.output());
  const xla::Shape lhs_shape = xla::gpu::GetShape(op.lhs());
  const xla::Shape rhs_shape = xla::gpu::GetShape(op.rhs());
  const mlir::mhlo::DotDimensionNumbersAttr dim_nums =
      op.dot_dimension_numbers();
  absl::Span<const int64_t> output_batch_dims =
      (dim_nums.getLhsBatchingDimensions().size() >
       dim_nums.getRhsBatchingDimensions().size())
          ? dim_nums.getLhsBatchingDimensions()
          : dim_nums.getRhsBatchingDimensions();

  int64_t batch_size = op.batch_size();
  int64_t output_row_dim = output_batch_dims.size();
  int64_t output_col_dim = output_row_dim + 1;

  if (op.rhs_stride() && op.lhs_stride()) {
    if (dim_nums.getLhsBatchingDimensions().size() !=
        dim_nums.getRhsBatchingDimensions().size()) {
      return rewriter.notifyMatchFailure(
          op, "Batching dimension size mismatch for nonzero strides.");
    }
  }

  int64_t output_num_rows = output_shape.dimensions(output_row_dim);
  int64_t output_num_cols = output_shape.dimensions(output_col_dim);

  auto validate_matrix = [&](const xla::Shape& shape,
                             auto batch_dimensions) -> LogicalResult {
    int64_t row_dim = batch_dimensions.size();
    int64_t col_dim = row_dim + 1;
    if (row_dim + 2 != shape.rank()) {
      return rewriter.notifyMatchFailure(op, "Invalid dimensions.");
    }

    for (int64_t batch_dim : batch_dimensions) {
      if (row_dim == batch_dim || col_dim == batch_dim) {
        return rewriter.notifyMatchFailure(
            op, "Batch dimensions overlap the last two dimensions.");
      }
    }

    // Verify that the non-batch dimensions are minor-most. This is required for
    // efficient access.
    if (shape.layout().minor_to_major(row_dim) >= 2 ||
        shape.layout().minor_to_major(col_dim) >= 2) {
      return rewriter.notifyMatchFailure(
          op, "Non-batch dimensions are not minor-most.");
    }
    return success();
  };

  auto valid_lhs =
      validate_matrix(lhs_shape, dim_nums.getLhsBatchingDimensions());
  if (failed(valid_lhs)) return valid_lhs;
  auto valid_rhs =
      validate_matrix(rhs_shape, dim_nums.getRhsBatchingDimensions());
  if (failed(valid_rhs)) return valid_rhs;
  auto valid_output = validate_matrix(output_shape, output_batch_dims);
  if (failed(valid_output)) return valid_output;

  // BLAS gemm expects the inputs and the output are in column-major order.
  // Therefore, we need to convert dot between row-major matrices to that
  // between column-major matrices. The key insight for the conversion is that,
  // in linear storage, matrix M in column-major order is identical to the
  // transpose of M in row-major order. In other words,
  //
  //   column-major(M) = row-major(M^T).
  //
  // Leveraging this insight, we can perform dot between row-major matrices as
  // follows.
  //
  // row-major(C)
  //   = row-major(A x B) = column-major((A x B)^T) = column-major(B^T x A^T)
  //   = gemm(column-major(B^T), column-major(A^T))
  //   = gemm(row-major(B), row-major(A))
  //
  // Although we do not modify the content of A and B in linear memory, we
  // should use the dimensions of B^T and A^T when calling gemm. For example,
  // the leading dimension of the LHS matrix of gemm is the number of rows in
  // B^T and thus the number of columns in B.
  auto make_descriptor = [&](Value data, const xla::Shape& shape,
                             int64_t row_dim, bool transpose,
                             int64_t stride) -> MatrixDescriptor {
    bool is_row_major = xla::LayoutUtil::Minor(shape.layout(), row_dim) != 0;
    bool layout_mismatch =
        xla::LayoutUtil::Minor(shape.layout(), row_dim) !=
        xla::LayoutUtil::Minor(output_shape.layout(), output_row_dim);
    int64_t rows =
        shape.dimensions(row_dim + static_cast<int64_t>(is_row_major));
    int64_t cols =
        shape.dimensions(row_dim + static_cast<int64_t>(!is_row_major));
    return MatrixDescriptor{data, transpose != layout_mismatch, rows, cols,
                            stride};
  };

  bool lhs_transpose = dim_nums.getLhsContractingDimensions()[0] ==
                       dim_nums.getLhsBatchingDimensions().size();
  bool rhs_transpose = dim_nums.getRhsContractingDimensions()[0] ==
                       dim_nums.getRhsBatchingDimensions().size() + 1;

  MatrixDescriptor lhs_matrix = make_descriptor(
      adaptor.lhs(), lhs_shape, dim_nums.getLhsBatchingDimensions().size(),
      lhs_transpose, op.lhs_stride());
  MatrixDescriptor rhs_matrix = make_descriptor(
      adaptor.rhs(), rhs_shape, dim_nums.getRhsBatchingDimensions().size(),
      rhs_transpose, op.rhs_stride());

  if (xla::LayoutUtil::Minor(output_shape.layout(), output_row_dim) != 0) {
    std::swap(lhs_matrix, rhs_matrix);
    std::swap(output_num_cols, output_num_rows);
  }

  const MatrixDescriptor output_matrix{adaptor.output(), /*transpose=*/false,
                                       output_num_rows, output_num_cols,
                                       output_num_rows * output_num_cols};

  auto valid_stride = [](const MatrixDescriptor& matrix) {
    if (matrix.stride != 0) {
      if (matrix.stride != matrix.num_rows * matrix.num_cols) return false;
    }
    return true;
  };
  if (!valid_stride(lhs_matrix) || !valid_stride(rhs_matrix) ||
      !valid_stride(output_matrix))
    return rewriter.notifyMatchFailure(op, "Invalid nonzero stride.");

  // Use zero with alpha's semantic if no beta_arg is supplied.
  llvm::APFloat beta_real = APFloat::getZero(op.alpha_real().getSemantics());
  if (auto attr = GetBeta(op)) beta_real = attr.getValue();

  auto algorithm = static_cast<cublasGemmAlgo_t>(
      op.algorithm().getValueOr(CUBLAS_GEMM_DEFAULT));

  return CreateTfrtOps(op, adaptor, chain, stream, batch_size, element_type,
                       lhs_matrix, rhs_matrix, output_matrix, op.alpha_real(),
                       op.alpha_imag(), beta_real, algorithm, rewriter);
}

template <class GemmOpType>
struct GemmRewritePattern : tfrt::gpu::GpuAsyncOpConversionPattern<GemmOpType> {
  using typename tfrt::gpu::GpuAsyncOpConversionPattern<GemmOpType>::OpAdaptor;
  using tfrt::gpu::GpuAsyncOpConversionPattern<
      GemmOpType>::GpuAsyncOpConversionPattern;
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
