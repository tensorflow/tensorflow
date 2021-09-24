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
#include "tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu/gemm_pattern.h"

#include <assert.h>
#include <stdint.h>

#include <type_traits>
#include <utility>

#include "mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/pass/pass.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/cublas_wrapper.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {
namespace {

// This struct contains the metadata of a matrix, e.g., its base address and
// dimensions.
struct MatrixDescriptor {
  mlir::Value converted_value;
  bool transpose;  // Whether this matrix needs to be transposed.
  int64_t num_rows;
  int64_t num_cols;
};

static cudaDataType_t MlirTypeToCudaDataType(mlir::Type type) {
  if (type.isF16()) return CUDA_R_16F;
  if (type.isF32()) return CUDA_R_32F;
  if (type.isF64()) return CUDA_R_64F;
  if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    auto element_type = complex_type.getElementType();
    if (element_type.isF32()) return CUDA_C_32F;
    if (element_type.isF64()) return CUDA_C_64F;
  }
  llvm_unreachable("unsupported type");
}

static cublasComputeType_t MlirTypeToBlasComputeType(mlir::Type type) {
  if (auto complexType = type.dyn_cast<mlir::ComplexType>())
    return MlirTypeToBlasComputeType(complexType.getElementType());
  if (type.isF16()) return CUBLAS_COMPUTE_16F;
  if (type.isF32()) return CUBLAS_COMPUTE_32F;
  if (type.isF64()) return CUBLAS_COMPUTE_64F;
  llvm_unreachable("unsupported type");
}

// TODO(b/176561997): remove this once lhlo_gpu ops have properly typed alpha
// and beta attributes. We can't use std::complex here because the effect of
// instantiating it for anything other than float, double, or long double is
// unspecified. We need it for APFloat.
template <class T>
struct Complex {
  T real;
  T imag;
};

// TODO(b/176561997): remove this once lhlo_gpu ops have properly typed alpha
// and beta attributes.
mlir::Value MakeScalingFactorConstant(mlir::OpBuilder& builder,
                                      mlir::Location loc, mlir::Type type,
                                      Complex<llvm::APFloat> value) {
  // Dummy boolean we need to pass to convert functions. Since this whole
  // funciton will go away when the scaling factors are properly typed
  // (b/176561997), we won't worry about possible losses during conversions for
  // now.
  bool losesInfo = false;
  if (type.isF32()) {
    value.real.convert(llvm::APFloat::IEEEsingle(),
                       llvm::RoundingMode::NearestTiesToEven, &losesInfo);
    return builder.create<tfrt::compiler::ConstantF32Op>(loc, value.real);
  }
  if (type.isF64()) {
    value.real.convert(llvm::APFloat::IEEEdouble(),
                       llvm::RoundingMode::NearestTiesToEven, &losesInfo);
    return builder.create<tfrt::compiler::ConstantF64Op>(loc, value.real);
  }
  if (type == mlir::ComplexType::get(builder.getF32Type())) {
    value.real.convert(llvm::APFloat::IEEEsingle(),
                       llvm::RoundingMode::NearestTiesToEven, &losesInfo);
    value.imag.convert(llvm::APFloat::IEEEsingle(),
                       llvm::RoundingMode::NearestTiesToEven, &losesInfo);
    return builder.create<tfrt::compiler::ConstantComplexF32Op>(loc, value.real,
                                                                value.imag);
  }
  if (type == mlir::ComplexType::get(builder.getF64Type())) {
    value.real.convert(llvm::APFloat::IEEEdouble(),
                       llvm::RoundingMode::NearestTiesToEven, &losesInfo);
    value.imag.convert(llvm::APFloat::IEEEdouble(),
                       llvm::RoundingMode::NearestTiesToEven, &losesInfo);
    return builder.create<tfrt::compiler::ConstantComplexF64Op>(loc, value.real,
                                                                value.imag);
  }

  llvm_unreachable("unsupported type");
}

// Create all the Ops necessary for the GEMM operation, including the GEMM
// operation itself.
FailureOr<Value> CreateTfrtOps(
    mlir::Location loc, mlir::Value chain, mlir::Value stream,
    int64_t batch_size, mlir::Type element_type, MatrixDescriptor lhs_matrix,
    MatrixDescriptor rhs_matrix, MatrixDescriptor output_matrix,
    Complex<llvm::APFloat> alpha, Complex<llvm::APFloat> beta,
    cublasGemmAlgo_t algorithm, mlir::OpBuilder& builder) {
  auto k_val = lhs_matrix.transpose ? lhs_matrix.num_rows : lhs_matrix.num_cols;

  // Use mixed precision for fp16 to match GEMM auto-tuning, see
  // ComputationTypeFromPrimitive().
  Type mlir_compute_type =
      element_type.isF16() ? builder.getF32Type() : element_type;

  auto m = builder.create<tfrt::compiler::ConstantI32Op>(
      loc, output_matrix.num_rows);
  auto n = builder.create<tfrt::compiler::ConstantI32Op>(
      loc, output_matrix.num_cols);
  auto k = builder.create<tfrt::compiler::ConstantI32Op>(loc, k_val);

  auto const_alpha =
      MakeScalingFactorConstant(builder, loc, mlir_compute_type, alpha);

  auto lda =
      builder.create<tfrt::compiler::ConstantI32Op>(loc, lhs_matrix.num_rows);
  auto ldb =
      builder.create<tfrt::compiler::ConstantI32Op>(loc, rhs_matrix.num_rows);

  auto const_beta =
      MakeScalingFactorConstant(builder, loc, mlir_compute_type, beta);

  auto ldc = builder.create<tfrt::compiler::ConstantI32Op>(
      loc, output_matrix.num_rows);

  auto algo = builder.create<tfrt::gpu::BlasGemmAlgoOp>(loc, algorithm);

  auto blas_handle_type = builder.getType<tfrt::gpu::BlasHandleType>();
  auto blas_handle =
      builder.create<tfrt::gpu::BlasCreateOp>(loc, blas_handle_type, stream);

  auto lhs_op = lhs_matrix.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto rhs_op = rhs_matrix.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

  const auto data_type = MlirTypeToCudaDataType(element_type);
  const auto compute_type = MlirTypeToBlasComputeType(mlir_compute_type);

  if (batch_size != 1) {
    int64_t lhs_stride_val = lhs_matrix.num_rows * lhs_matrix.num_cols;
    int64_t rhs_stride_val = rhs_matrix.num_rows * rhs_matrix.num_cols;
    int64_t output_stride_val = output_matrix.num_rows * output_matrix.num_cols;
    auto lhs_stride =
        builder.create<tfrt::compiler::ConstantI64Op>(loc, lhs_stride_val);
    auto rhs_stride =
        builder.create<tfrt::compiler::ConstantI64Op>(loc, rhs_stride_val);
    auto output_stride =
        builder.create<tfrt::compiler::ConstantI64Op>(loc, output_stride_val);
    auto batch = builder.create<tfrt::compiler::ConstantI32Op>(loc, batch_size);
    return builder
        .create<tfrt::gpu::BlasGemmBatchExOp>(
            loc, chain.getType(), blas_handle, lhs_op, rhs_op, m, n, k,
            const_alpha, lhs_matrix.converted_value, data_type, lda, lhs_stride,
            rhs_matrix.converted_value, data_type, ldb, rhs_stride, const_beta,
            output_matrix.converted_value, data_type, ldc, output_stride, batch,
            compute_type, algo, chain)
        .getResult();
  }

  return builder
      .create<tfrt::gpu::BlasGemmOp>(loc, chain.getType(), blas_handle, lhs_op,
                                     rhs_op, m, n, k, const_alpha,
                                     lhs_matrix.converted_value, data_type, lda,
                                     rhs_matrix.converted_value, data_type, ldb,
                                     const_beta, output_matrix.converted_value,
                                     data_type, ldc, compute_type, algo, chain)
      .getResult();
}

template <class GemmOpType>
FailureOr<Value> GemmOpConversionRewrite(
    GemmOpType srcOp, Value chain, Value stream,
    mlir::BlockAndValueMapping& mapping, mlir::OpBuilder& builder,
    absl::optional<llvm::APFloat> beta_arg = absl::nullopt) {
  mlir::Type element_type = srcOp.output()
                                .getType()
                                .template cast<mlir::MemRefType>()
                                .getElementType();
  // Ensure the types of all elements are the same.
  if (element_type !=
      srcOp.lhs().getType().template cast<mlir::MemRefType>().getElementType())
    return mlir::failure();
  if (element_type !=
      srcOp.rhs().getType().template cast<mlir::MemRefType>().getElementType())
    return mlir::failure();
  const mlir::mhlo::DotDimensionNumbers dim_nums =
      srcOp.dot_dimension_numbers();

  // The row and column dimensions are the last two dimensions. All the
  // dimensions before them are batching dimensions.
  int64_t row_dim = dim_nums.lhs_batching_dimensions().size();
  int64_t col_dim = dim_nums.lhs_batching_dimensions().size() + 1;

  int64_t batch_size = srcOp.batch_size();

  // Check that the batch dims don't cover the last two dims.
  for (auto batch_dim : dim_nums.lhs_batching_dimensions()) {
    if (row_dim == batch_dim) return mlir::failure();
    if (col_dim == batch_dim) return mlir::failure();
  }

  // Verify that the non-batch dimensions are minor-most. This is required for
  // efficient access.
  const xla::Shape& lhs_shape = xla::TypeToShape(srcOp.lhs().getType());
  const xla::Shape& rhs_shape = xla::TypeToShape(srcOp.rhs().getType());
  const xla::Shape& output_shape = xla::TypeToShape(srcOp.output().getType());
  for (const auto* shape : {&lhs_shape, &rhs_shape, &output_shape}) {
    if (shape->layout().minor_to_major(row_dim) >= 2) return mlir::failure();
    if (shape->layout().minor_to_major(col_dim) >= 2) return mlir::failure();
  }

  // BLAS gemm expects the inputs and the output are in column-major order.
  // Therefore, we need to convert multiplication between row-major matrices to
  // that between column-major matrices. The key insight for the conversion is
  // that, in linear storage, matrix M in column-major order is identical to the
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
  auto make_descriptor = [&](const xla::Shape& shape,
                             mlir::Value replaced_value,
                             bool transpose) -> MatrixDescriptor {
    bool is_row_major = xla::LayoutUtil::Minor(shape.layout(), row_dim) != 0;
    bool layout_mismatch =
        xla::LayoutUtil::Minor(shape.layout(), row_dim) !=
        xla::LayoutUtil::Minor(output_shape.layout(), row_dim);
    return MatrixDescriptor{
        replaced_value, static_cast<bool>(transpose ^ layout_mismatch),
        shape.dimensions(row_dim + static_cast<int64_t>(is_row_major)),
        shape.dimensions(row_dim + static_cast<int64_t>(!is_row_major))};
  };

  MatrixDescriptor lhs_matrix = make_descriptor(
      lhs_shape, mapping.lookup(srcOp.lhs()),
      dim_nums.lhs_contracting_dimensions().getValue<int64_t>({0}) == row_dim);
  MatrixDescriptor rhs_matrix = make_descriptor(
      rhs_shape, mapping.lookup(srcOp.rhs()),
      dim_nums.rhs_contracting_dimensions().getValue<int64_t>({0}) == col_dim);
  MatrixDescriptor output_matrix = MatrixDescriptor{
      mapping.lookup(srcOp.output()), /*transpose=*/false,
      output_shape.dimensions(row_dim), output_shape.dimensions(col_dim)};

  Complex<llvm::APFloat> alpha{srcOp.alpha_real(), srcOp.alpha_imag()};
  // If no beta_arg is supplied, we copy alpha and then zero it out to ensure
  // beta has the same float semantics (IEEE single, IEEE double, ...) as alpha.
  llvm::APFloat beta_real = beta_arg.has_value()
                                ? beta_arg.value()
                                : APFloat::getZero(alpha.real.getSemantics());
  Complex<llvm::APFloat> beta{beta_real,
                              APFloat::getZero(alpha.imag.getSemantics())};

  if (xla::LayoutUtil::Minor(output_shape.layout(), row_dim) != 0) {
    std::swap(lhs_matrix, rhs_matrix);
    std::swap(output_matrix.num_cols, output_matrix.num_rows);
  }

  auto algorithm = static_cast<cublasGemmAlgo_t>(
      srcOp.algorithm().getValueOr(CUBLAS_GEMM_DEFAULT));

  return CreateTfrtOps(srcOp.getLoc(), chain, stream, batch_size, element_type,
                       lhs_matrix, rhs_matrix, output_matrix, alpha, beta,
                       algorithm, builder);
}

absl::optional<llvm::APFloat> GetBeta(lmhlo_gpu::GEMMOp op) {
  return absl::nullopt;
}

absl::optional<llvm::APFloat> GetBeta(lmhlo_gpu::GEMM_BiasOp op) {
  return op.beta();
}

template <class GemmOpType>
struct GemmRewritePattern : tfrt::gpu::GpuAsyncOpConversionPattern<GemmOpType> {
  using tfrt::gpu::GpuAsyncOpConversionPattern<
      GemmOpType>::GpuAsyncOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      GemmOpType op, Value chain, Value stream, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    if (!all_of(operands, [](Value operand) {
          return operand.getType().isa<tfrt::gpu::BufferType>();
        }))
      return rewriter.notifyMatchFailure(op, "expected buffer operands");

    BlockAndValueMapping mapping;
    for (auto pair : llvm::zip_first(op->getOperands(), operands))
      mapping.map(std::get<0>(pair), std::get<1>(pair));

    rewriter.eraseOp(op);

    return GemmOpConversionRewrite(op, chain, stream, mapping, rewriter,
                                   GetBeta(op));
  }
};

}  // namespace

void populateGemmConversionPattern(RewritePatternSet& patterns) {
  patterns.add<GemmRewritePattern<lmhlo_gpu::GEMMOp>,
               GemmRewritePattern<lmhlo_gpu::GEMM_BiasOp>>(
      patterns.getContext());
}

}  // namespace tensorflow
