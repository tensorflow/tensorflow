/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/stream_executor/blas.h"

namespace xla {
namespace gpu {
namespace {

void TransposeMatrixDesc(se::blas::MatrixDescriptor& matrix_desc) {
  matrix_desc.transpose =
      (matrix_desc.transpose == se::blas::Transpose::kNoTranspose)
          ? se::blas::Transpose::kTranspose
          : se::blas::Transpose::kNoTranspose;
}

}  // namespace

StatusOr<std::vector<int64_t>> GetNonContractingDims(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> contracting_dims) {
  std::vector<int64_t> non_contracting_dims;
  // This is O(rank**2), but we expect rank to be small.
  for (int64_t dim = 0; dim < shape.rank(); ++dim) {
    bool is_batch = absl::c_count(batch_dims, dim) != 0;
    bool is_contracting = absl::c_count(contracting_dims, dim) != 0;
    TF_RET_CHECK(!(is_batch && is_contracting));
    if (!(is_batch || is_contracting)) non_contracting_dims.push_back(dim);
  }

  TF_RET_CHECK(batch_dims.size() + contracting_dims.size() +
                   non_contracting_dims.size() ==
               shape.rank());
  return non_contracting_dims;
}

StatusOr<Shape> GetBatchRowColumnShape(const Shape& shape,
                                       absl::Span<const int64_t> batch_dims,
                                       absl::Span<const int64_t> row_dims,
                                       absl::Span<const int64_t> col_dims) {
  TF_RET_CHECK(shape.has_layout());
  TF_RET_CHECK(!row_dims.empty());
  TF_RET_CHECK(!col_dims.empty());

  std::vector<int64_t> minor_to_major;
  for (size_t i = 0; i < shape.rank();) {
    // The GeMM output always has its layout set such that the batch, row, and
    // col dim groups are each laid out physically sequentially. GeMM operands
    // must, therefore, be laid out similarly.
    auto check_physically_sequential = [&](absl::Span<const int64_t> dims) {
      for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
        // NOTE: `i` is incremented as we check the dimensions.
        if (*it != shape.layout().minor_to_major()[i++])
          return InvalidArgument("dims not physically sequential");
      }
      return OkStatus();
    };

    int64_t dim = shape.layout().minor_to_major()[i];
    if (dim == row_dims.back()) {
      minor_to_major.push_back(1);
      TF_RETURN_IF_ERROR(check_physically_sequential(row_dims));
    } else if (dim == col_dims.back()) {
      minor_to_major.push_back(2);
      TF_RETURN_IF_ERROR(check_physically_sequential(col_dims));
    } else if (!batch_dims.empty() && (dim == batch_dims.back())) {
      minor_to_major.push_back(0);
      TF_RETURN_IF_ERROR(check_physically_sequential(batch_dims));
    } else {
      return InvalidArgument("dims not physically sequential");
    }
  }

  if (batch_dims.empty()) minor_to_major.push_back(0);

  auto dim_size = [&](absl::Span<const int64_t> dims) {
    return absl::c_accumulate(dims, 1, [&](int64_t size, int64_t dim) {
      return size * shape.dimensions(dim);
    });
  };

  return ShapeUtil::MakeShapeWithLayout(
      shape.element_type(),
      {dim_size(batch_dims), dim_size(row_dims), dim_size(col_dims)},
      minor_to_major);
}

// Returns the matrix layout for a logical shape (batch, rows, columns).
/*static*/ StatusOr<MatrixLayout> MatrixLayout::For(const Shape& shape) {
  TF_RET_CHECK(shape.rank() == 3);
  TF_RET_CHECK(shape.has_layout());

  int64_t batch_size = shape.dimensions(0);
  int64_t num_rows = shape.dimensions(1);
  int64_t num_cols = shape.dimensions(2);

  MatrixLayout::Order order = MatrixLayout::Order::kRowMajor;
  int64_t leading_dim_stride = num_cols;
  int64_t batch_stride = num_rows * num_cols;

  // `MatrixLayout`, like BLAS, uses only two strides, so either the row or
  // column must be contiguous in memory (i.e. most minor physical dimension).
  absl::Span<const int64_t> minor_to_major = shape.layout().minor_to_major();
  switch (64 * minor_to_major[2] + 8 * minor_to_major[1] + minor_to_major[0]) {
    case 012:  // (B,R,C) (major-to-minor)
      break;
    case 021:  // (B,C,R)
      order = MatrixLayout::Order::kColumnMajor;
      leading_dim_stride = num_rows;
      break;
    case 0102:  // (R,B,C)
      leading_dim_stride = batch_size * num_cols;
      batch_stride = num_cols;
      break;
    case 0201:  // (C,B,R)
      order = MatrixLayout::Order::kColumnMajor;
      leading_dim_stride = batch_size * num_rows;
      batch_stride = num_rows;
      break;
    default:
      return Unimplemented("batch in most minor dimension");
  }

  if (batch_size == 1) batch_stride = 0;
  return MatrixLayout{
      shape.element_type(), num_rows,   num_cols,     order,
      leading_dim_stride,   batch_size, batch_stride,
  };
}

/*static*/ StatusOr<MatrixLayout> MatrixLayout::For(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> row_dims, absl::Span<const int64_t> col_dims) {
  TF_ASSIGN_OR_RETURN(
      Shape batch_row_col_shape,
      GetBatchRowColumnShape(shape, batch_dims, row_dims, col_dims));
  return MatrixLayout::For(batch_row_col_shape);
}

/*static*/ StatusOr<MatrixLayout> MatrixLayout::For(const Shape& shape,
                                                    size_t lhs_num_batch_dims,
                                                    size_t lhs_num_row_dims,
                                                    size_t rhs_num_batch_dims,
                                                    size_t rhs_num_col_dims) {
  size_t num_batch_dims = std::max(lhs_num_batch_dims, rhs_num_batch_dims);

  TF_RET_CHECK(shape.rank() ==
               num_batch_dims + lhs_num_row_dims + rhs_num_col_dims);

  std::vector<int64_t> dims(shape.rank());
  absl::c_iota(dims, 0);

  auto batch_dims = absl::Span<const int64_t>(dims).first(num_batch_dims);
  auto row_dims =
      absl::Span<const int64_t>(dims).subspan(num_batch_dims, lhs_num_row_dims);
  auto col_dims = absl::Span<const int64_t>(dims).last(rhs_num_col_dims);

  return MatrixLayout::For(shape, batch_dims, row_dims, col_dims);
}

StatusOr<bool> CanFoldTransposeOperandIntoDot(const HloInstruction& dot,
                                              int64_t operand_idx) {
  TF_RET_CHECK(dot.opcode() == HloOpcode::kDot);
  TF_RET_CHECK(dot.operand_count() > operand_idx);

  const HloInstruction& transpose = *dot.operand(operand_idx);
  TF_RET_CHECK(transpose.opcode() == HloOpcode::kTranspose);

  const DotDimensionNumbers& dot_dims = dot.dot_dimension_numbers();

  auto transposed = [&](const auto& dims) {
    std::vector<int64_t> transposed_dims;
    transposed_dims.reserve(dims.size());
    for (int64_t dim : dims) {
      transposed_dims.push_back(transpose.dimensions(dim));
    }
    return transposed_dims;
  };

  auto batch_dims = (operand_idx == 0) ? dot_dims.lhs_batch_dimensions()
                                       : dot_dims.rhs_batch_dimensions();
  auto contracting_dims = (operand_idx == 0)
                              ? dot_dims.lhs_contracting_dimensions()
                              : dot_dims.rhs_contracting_dimensions();
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> non_contracting_dims,
      GetNonContractingDims(transpose.shape(), batch_dims, contracting_dims));

  // If we're able to construct a valid `MatrixLayout` for the transposed
  // dimensions, then GeMM can support folding the transpose.
  return MatrixLayout::For(transpose.operand(0)->shape(),
                           transposed(batch_dims), transposed(contracting_dims),
                           transposed(non_contracting_dims))
      .ok();
}

namespace {

bool IsBlasPlansCompatibleType(PrimitiveType type) {
  switch (type) {
    case F16:
    case F32:
    case F64:
    case C64:
    case C128:
      return true;
    default:
      return false;
  }
}

}  // namespace

/*static*/ StatusOr<GemmConfig> GemmConfig::For(
    const Shape& lhs_shape, absl::Span<const int64_t> lhs_batch_dims,
    absl::Span<const int64_t> lhs_contracting_dims, const Shape& rhs_shape,
    absl::Span<const int64_t> rhs_batch_dims,
    absl::Span<const int64_t> rhs_contracting_dims, const Shape& output_shape,
    double alpha_real, double alpha_imag, double beta,
    std::optional<int64_t> algorithm, int64_t compute_precision,
    bool use_cublaslt) {
  absl::Span<const int64_t> lhs_col_dims = lhs_contracting_dims;
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> lhs_row_dims,
      GetNonContractingDims(lhs_shape, lhs_batch_dims, lhs_col_dims));

  TF_ASSIGN_OR_RETURN(
      MatrixLayout lhs_layout,
      MatrixLayout::For(lhs_shape, lhs_batch_dims, lhs_row_dims, lhs_col_dims));

  absl::Span<const int64_t> rhs_row_dims = rhs_contracting_dims;
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> rhs_col_dims,
      GetNonContractingDims(rhs_shape, rhs_batch_dims, rhs_row_dims));

  TF_ASSIGN_OR_RETURN(
      MatrixLayout rhs_layout,
      MatrixLayout::For(rhs_shape, rhs_batch_dims, rhs_row_dims, rhs_col_dims));

  int64_t num_batch_dims =
      std::max(lhs_batch_dims.size(), rhs_batch_dims.size());

  TF_RET_CHECK(output_shape.rank() ==
               num_batch_dims + lhs_row_dims.size() + rhs_col_dims.size());

  std::vector<int64_t> output_dims(output_shape.rank());
  absl::c_iota(output_dims, 0);

  auto output_batch_dims =
      absl::Span<const int64_t>(output_dims).first(num_batch_dims);
  auto output_row_dims = absl::Span<const int64_t>(output_dims)
                             .subspan(num_batch_dims, lhs_row_dims.size());
  auto output_col_dims =
      absl::Span<const int64_t>(output_dims).last(rhs_col_dims.size());

  TF_ASSIGN_OR_RETURN(MatrixLayout output_layout,
                      MatrixLayout::For(output_shape, output_batch_dims,
                                        output_row_dims, output_col_dims));

  // TODO(cjfj): We should also check that the batch, contracting and
  // non-contracting dimensions match in size and relative physical location.
  TF_RET_CHECK(lhs_layout.num_cols == rhs_layout.num_rows);
  TF_RET_CHECK(output_layout.num_rows == lhs_layout.num_rows);
  TF_RET_CHECK(output_layout.num_cols == rhs_layout.num_cols);
  TF_RET_CHECK((lhs_layout.batch_size == output_layout.batch_size) ||
               (lhs_layout.batch_size == 1));
  TF_RET_CHECK((rhs_layout.batch_size == output_layout.batch_size) ||
               (rhs_layout.batch_size == 1));

  use_cublaslt &= IsBlasPlansCompatibleType(output_shape.element_type());

  switch (output_shape.element_type()) {
    case F16:
    case BF16:
    case F32:
    case F64:
      TF_RET_CHECK(alpha_imag == 0);
      break;
    case C64:
    case C128:
      break;
    case S32:
      TF_RET_CHECK(alpha_imag == 0);
      if (lhs_layout.dtype != PrimitiveType::S8 ||
          rhs_layout.dtype != PrimitiveType::S8) {
        return InternalError(
            "For int32 gemm output only int8 input is supported, got input: "
            "%s, %s",
            primitive_util::LowercasePrimitiveTypeName(lhs_layout.dtype),
            primitive_util::LowercasePrimitiveTypeName(rhs_layout.dtype));
      }
      break;
    default:
      return InternalError("Unexpected GEMM datatype: %s",
                           primitive_util::LowercasePrimitiveTypeName(
                               output_shape.element_type()));
  }

  return GemmConfig{
      lhs_layout, rhs_layout, output_layout,     {alpha_real, alpha_imag},
      beta,       algorithm,  compute_precision, use_cublaslt,
  };
}

/*static*/ StatusOr<GemmConfig> GemmConfig::For(const HloInstruction* gemm) {
  TF_ASSIGN_OR_RETURN(GemmBackendConfig config,
                      gemm->backend_config<GemmBackendConfig>());

  std::optional<int64_t> algorithm;
  if (config.algorithm_case() != GemmBackendConfig::ALGORITHM_NOT_SET) {
    algorithm = config.selected_algorithm();
  }

  const Shape& lhs_shape = gemm->operand(0)->shape();
  const Shape& rhs_shape = gemm->operand(1)->shape();
  const DotDimensionNumbers& dot_dims = config.dot_dimension_numbers();
  bool use_cublaslt =
      gemm->GetModule()->config().debug_options().xla_gpu_enable_cublaslt();

  return GemmConfig::For(lhs_shape, dot_dims.lhs_batch_dimensions(),
                         dot_dims.lhs_contracting_dimensions(), rhs_shape,
                         dot_dims.rhs_batch_dimensions(),
                         dot_dims.rhs_contracting_dimensions(),
                         /*output_shape=*/gemm->shape(), config.alpha_real(),
                         config.alpha_imag(), config.beta(), algorithm,
                         se::blas::kDefaultComputePrecision, use_cublaslt);
}

/*static*/ StatusOr<GemmConfig> GemmConfig::For(mlir::Operation* op,
                                                bool use_cublaslt) {
  auto get_config = [&](auto op, llvm::APFloat beta) {
    mlir::mhlo::DotDimensionNumbersAttr dot_dims = op.getDotDimensionNumbers();

    std::optional<int64_t> algorithm;
    if (op.getAlgorithm()) algorithm = *op.getAlgorithm();

    int64_t compute_precision = 0;  // Default
    if (op.getPrecisionConfig().hasValue()) {
      auto precision_config = op.getPrecisionConfig();
      for (auto attr : precision_config.getValue()) {
        int64_t value = static_cast<int64_t>(
            attr.template cast<mlir::mhlo::PrecisionAttr>().getValue());
        if (value > compute_precision) {
          compute_precision = value;
        }
      }
    }

    return GemmConfig::For(
        GetShape(op.getLhs()), dot_dims.getLhsBatchingDimensions(),
        dot_dims.getLhsContractingDimensions(), GetShape(op.getRhs()),
        dot_dims.getRhsBatchingDimensions(),
        dot_dims.getRhsContractingDimensions(), GetShape(op.getOutput()),
        op.getAlphaReal().convertToDouble(),
        op.getAlphaImag().convertToDouble(), beta.convertToDouble(), algorithm,
        compute_precision, use_cublaslt);
  };

  if (auto gemm = mlir::dyn_cast<mlir::lmhlo_gpu::GEMMOp>(op))
    return get_config(gemm, llvm::APFloat(0.));

  auto gemm = mlir::dyn_cast<mlir::lmhlo_gpu::GEMM_BiasOp>(op);
  TF_RET_CHECK(gemm != nullptr);
  return get_config(gemm, gemm.getBeta());
}

se::blas::MatrixDescriptor GetMatrixDesc(const MatrixLayout& layout,
                                         se::DeviceMemoryBase data) {
  bool transpose = layout.order != MatrixLayout::Order::kColumnMajor;
  return {
      data,
      layout.leading_dim_stride,
      layout.batch_stride,
      transpose ? se::blas::Transpose::kTranspose
                : se::blas::Transpose::kNoTranspose,
  };
}

void MakeBlasGemmCompatible(int64_t& m, int64_t& n,
                            se::blas::MatrixDescriptor& lhs,
                            se::blas::MatrixDescriptor& rhs,
                            se::blas::MatrixDescriptor& output) {
  // BLAS GeMM doesn't support transposed output, but we can use the identity:
  // C^T = (A @ B)^T = B^T @ A^T.
  if (output.transpose == se::blas::Transpose::kTranspose) {
    std::swap(m, n);
    std::swap(lhs, rhs);
    TransposeMatrixDesc(lhs);
    TransposeMatrixDesc(rhs);
    TransposeMatrixDesc(output);
  }
}

namespace {

// Converts from an XLA PrimitiveType to a blas::ComputationType, which is
// used to specify the precision with which matmul computations should be
// performed, separately from the precision of the inputs and result.
std::optional<se::blas::ComputationType> ComputationTypeFromPrimitive(
    PrimitiveType type) {
  switch (type) {
    case F16:  // Use F32 computation for higher precision.
    case BF16:
    case F32:
      return se::blas::ComputationType::kF32;
    case F64:
      return se::blas::ComputationType::kF64;
    case C64:
      return se::blas::ComputationType::kComplexF32;
    case C128:
      return se::blas::ComputationType::kComplexF64;
    case S32:
      return se::blas::ComputationType::kI32;
    default:
      return std::nullopt;
  }
}

template <typename Input, typename Output>
Status DoGemmWithAlgorithm(int64_t batch_size, int64_t m, int64_t n, int64_t k,
                           const se::blas::MatrixDescriptor& lhs,
                           const se::blas::MatrixDescriptor& rhs,
                           const se::blas::MatrixDescriptor& output,
                           Output alpha, Output beta, se::Stream* stream,
                           se::blas::AlgorithmType algorithm,
                           se::blas::ProfileResult* profile_result) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  PrimitiveType output_type = primitive_util::NativeToPrimitiveType<Output>();
  se::blas::ComputationType computation_type =
      *ComputationTypeFromPrimitive(output_type);
  se::DeviceMemory<Output> output_data(output.data);

  if (batch_size != 1) {
    return stream->ThenBlasGemmStridedBatchedWithAlgorithm(
        lhs.transpose, rhs.transpose, m, n, k, alpha, lhs.cast<Input>(),
        lhs.leading_dim_stride, lhs.batch_stride, rhs.cast<Input>(),
        rhs.leading_dim_stride, rhs.batch_stride, beta, &output_data,
        output.leading_dim_stride, output.batch_stride, batch_size,
        computation_type, algorithm, profile_result);
  } else {
    return stream->ThenBlasGemmWithAlgorithm(
        lhs.transpose, rhs.transpose, m, n, k, alpha, lhs.cast<Input>(),
        lhs.leading_dim_stride, rhs.cast<Input>(), rhs.leading_dim_stride, beta,
        &output_data, output.leading_dim_stride, computation_type, algorithm,
        profile_result);
  }
}

template <typename Input>
Status DoGemm(int64_t batch_size, int64_t m, int64_t n, int64_t k,
              const se::blas::MatrixDescriptor& lhs,
              const se::blas::MatrixDescriptor& rhs,
              const se::blas::MatrixDescriptor& output, Input alpha, Input beta,
              se::Stream* stream,
              std::optional<se::blas::AlgorithmType> algorithm,
              se::blas::ComputePrecision compute_precision,
              se::blas::ProfileResult* profile_result) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  se::DeviceMemory<Input> output_data(output.data);

  if (algorithm) {
    return DoGemmWithAlgorithm<Input, Input>(batch_size, m, n, k, lhs, rhs,
                                             output, alpha, beta, stream,
                                             *algorithm, profile_result);
  }

  if (batch_size != 1) {
    return stream->ThenBlasGemmStridedBatched(
        lhs.transpose, rhs.transpose, m, n, k, alpha, lhs.cast<Input>(),
        lhs.leading_dim_stride, lhs.batch_stride, rhs.cast<Input>(),
        rhs.leading_dim_stride, rhs.batch_stride, beta, &output_data,
        output.leading_dim_stride, output.batch_stride, batch_size);
  }

  return stream->ThenBlasGemm(
      lhs.transpose, rhs.transpose, m, n, k, alpha, lhs.cast<Input>(),
      lhs.leading_dim_stride, rhs.cast<Input>(), rhs.leading_dim_stride, beta,
      &output_data, output.leading_dim_stride, compute_precision);
}

}  // namespace

Status RunGemm(const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
               se::DeviceMemoryBase rhs_buffer,
               se::DeviceMemoryBase output_buffer, se::Stream* stream,
               std::optional<se::blas::AlgorithmType> algorithm,
               se::blas::ProfileResult* profile_result) {
  VLOG(2) << "Executing a GemmThunk";
  int64_t m = config.output_layout.num_rows;
  int64_t n = config.output_layout.num_cols;
  int64_t k = config.lhs_layout.num_cols;
  se::blas::MatrixDescriptor lhs = GetMatrixDesc(config.lhs_layout, lhs_buffer);
  se::blas::MatrixDescriptor rhs = GetMatrixDesc(config.rhs_layout, rhs_buffer);
  se::blas::MatrixDescriptor output =
      GetMatrixDesc(config.output_layout, output_buffer);
  int64_t batch_size = config.output_layout.batch_size;

  // TODO(cjfj): Support transposed output when using cuBLASLt.
  MakeBlasGemmCompatible(m, n, lhs, rhs, output);

  if (!algorithm) algorithm = config.algorithm;

  switch (config.output_layout.dtype) {
    case S32:
      if (!algorithm) algorithm = se::blas::kDefaultGemmAlgo;
      return DoGemmWithAlgorithm<int8_t, int32_t>(
          batch_size, m, n, k, lhs, rhs, output,
          static_cast<int32_t>(config.alpha.real()),
          static_cast<int32_t>(config.beta), stream, *algorithm,
          profile_result);
    case F16:
      return DoGemm<Eigen::half>(batch_size, m, n, k, lhs, rhs, output,
                                 static_cast<Eigen::half>(config.alpha.real()),
                                 static_cast<Eigen::half>(config.beta), stream,
                                 algorithm, config.compute_precision,
                                 profile_result);
    case BF16:
      return DoGemm<Eigen::bfloat16>(
          batch_size, m, n, k, lhs, rhs, output,
          static_cast<Eigen::bfloat16>(config.alpha.real()),
          static_cast<Eigen::bfloat16>(config.beta), stream, algorithm,
          config.compute_precision, profile_result);
    case F32:
      return DoGemm<float>(batch_size, m, n, k, lhs, rhs, output,
                           config.alpha.real(), config.beta, stream, algorithm,
                           config.compute_precision, profile_result);
    case F64:
      return DoGemm<double>(batch_size, m, n, k, lhs, rhs, output,
                            config.alpha.real(), config.beta, stream, algorithm,
                            config.compute_precision, profile_result);
    case C64:
      return DoGemm<complex64>(batch_size, m, n, k, lhs, rhs, output,
                               static_cast<complex64>(config.alpha),
                               static_cast<complex64>(config.beta), stream,
                               algorithm, config.compute_precision,
                               profile_result);
    case C128:
      return DoGemm<complex128>(
          batch_size, m, n, k, lhs, rhs, output, config.alpha,
          static_cast<complex128>(config.beta), stream, algorithm,
          config.compute_precision, profile_result);
    default:
      return InternalError("Unexpected GEMM dtype: %s",
                           primitive_util::LowercasePrimitiveTypeName(
                               config.output_layout.dtype));
  }
}

namespace {

template <typename Input>
Status DoGemmLt(int64_t batch_size, int64_t m, int64_t n, int64_t k,
                const se::blas::MatrixDescriptor& lhs,
                const se::blas::MatrixDescriptor& rhs,
                const se::blas::MatrixDescriptor& output, se::Stream* stream,
                Input alpha, Input beta,
                se::ScratchAllocator& scratch_allocator,
                const se::blas::IBlasLtMatmulAlgorithm* algorithm,
                se::blas::ProfileResult* profile_result) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  tensorflow::DataType dtype = tensorflow::DataTypeToEnum<Input>::value;

  int device_id = stream->parent()->device_ordinal();

  bool trans_x = lhs.transpose == se::blas::Transpose::kTranspose;
  bool trans_y = rhs.transpose == se::blas::Transpose::kTranspose;
  bool broadcast_lhs = lhs.batch_stride == 0;
  bool broadcast_rhs = rhs.batch_stride == 0;
  VLOG(2) << "matmul params: trans_x " << trans_x << " trans_y " << trans_y
          << " adj_x " << false << " adj_y " << false << " m " << m << " n "
          << n << " k " << k << " batch_size " << batch_size
          << " broadcast_lhs " << broadcast_lhs << " broadcast_rhs "
          << broadcast_rhs << " dtype " << dtype << " device_id " << device_id;
  se::BatchMatmulParameters matmul_parameters(
      trans_x, trans_y, false, false, m, n, k, batch_size, broadcast_lhs,
      broadcast_rhs, dtype, dtype, device_id);

  TF_ASSIGN_OR_RETURN(
      const se::blas::PlanAndAlgorithms* plan_and_algorithms,
      GetPlanAndAlgorithms(stream, matmul_parameters, batch_size, m, n, k,
                           dtype, lhs, rhs, output));

  const std::unique_ptr<se::blas::IBlasLtMatmulPlan>& plan =
      plan_and_algorithms->plan;
  const std::vector<std::unique_ptr<se::blas::IBlasLtMatmulAlgorithm>>&
      algorithms = plan_and_algorithms->algorithms;

  if (algorithm == nullptr) {
    BlasPlansAutotuneCache& cache = GetBlasPlansAutotuneCache();
    std::optional<se::blas::AlgorithmConfig> algorithm_config =
        cache.Find(matmul_parameters);
    if (algorithm_config) {
      algorithm = algorithms[algorithm_config->algorithm()].get();
    } else {
      VLOG(4) << "Autotuner disabled: Inserting algorithm id 0"
              << " for " << trans_x << " " << trans_y << " " << m << " " << n
              << " " << k << " " << batch_size << " " << broadcast_lhs << " "
              << broadcast_rhs << " " << dtype << " " << device_id;
      cache.Insert(matmul_parameters, se::blas::AlgorithmConfig(0));
      algorithm = algorithms[0].get();
    }
  }

  se::DeviceMemory<Input> output_data(output.data);
  // NOLINTBEGIN: (b/223663260) ClangTidy mistakenly reports .get() as a
  // redundant call
  if (stream
          ->ThenBlasLtMatmul(plan.get(), alpha, lhs.cast<Input>(),
                             rhs.cast<Input>(), beta, &output_data,
                             &scratch_allocator, algorithm, {}, profile_result)
          .ok()) {
    return Status::OK();
  }
  // NOLINTEND
  return InternalError("BlasLtMatmul failed.");
}

}  // namespace

Status RunBlasLtMatmul(const GemmConfig& config,
                       se::DeviceMemoryBase lhs_buffer,
                       se::DeviceMemoryBase rhs_buffer,
                       se::DeviceMemoryBase output_buffer, se::Stream* stream,
                       se::ScratchAllocator& scratch_allocator,
                       const se::blas::IBlasLtMatmulAlgorithm* algorithm,
                       se::blas::ProfileResult* profile_result) {
  int64_t m = config.output_layout.num_rows;
  int64_t n = config.output_layout.num_cols;
  int64_t k = config.lhs_layout.num_cols;
  se::blas::MatrixDescriptor lhs = GetMatrixDesc(config.lhs_layout, lhs_buffer);
  se::blas::MatrixDescriptor rhs = GetMatrixDesc(config.rhs_layout, rhs_buffer);
  se::blas::MatrixDescriptor output =
      GetMatrixDesc(config.output_layout, output_buffer);
  int64_t batch_size = config.output_layout.batch_size;

  // TODO(cjfj): Support transposed output when using cuBLASLt.
  MakeBlasGemmCompatible(m, n, lhs, rhs, output);

  switch (config.output_layout.dtype) {
    case F16:
      return DoGemmLt<Eigen::half>(
          batch_size, m, n, k, lhs, rhs, output, stream,
          static_cast<Eigen::half>(config.alpha.real()),
          static_cast<Eigen::half>(config.beta), scratch_allocator, algorithm,
          profile_result);
    case F32:
      return DoGemmLt<float>(batch_size, m, n, k, lhs, rhs, output, stream,
                             static_cast<float>(config.alpha.real()),
                             static_cast<float>(config.beta), scratch_allocator,
                             algorithm, profile_result);
    case F64:
      return DoGemmLt<double>(batch_size, m, n, k, lhs, rhs, output, stream,
                              static_cast<double>(config.alpha.real()),
                              config.beta, scratch_allocator, algorithm,
                              profile_result);
    case C64:
      return DoGemmLt<complex64>(batch_size, m, n, k, lhs, rhs, output, stream,
                                 static_cast<complex64>(config.alpha),
                                 static_cast<complex64>(config.beta),
                                 scratch_allocator, algorithm, profile_result);
    case C128:
      return DoGemmLt<complex128>(batch_size, m, n, k, lhs, rhs, output, stream,
                                  config.alpha,
                                  static_cast<complex64>(config.beta),
                                  scratch_allocator, algorithm, profile_result);
    default:
      return InternalError("Unexpected GEMMLt dtype: %s",
                           primitive_util::LowercasePrimitiveTypeName(
                               config.output_layout.dtype));
  }
}

std::optional<se::blas::AlgorithmConfig> BlasPlansAutotuneCache::Find(
    const se::BatchMatmulParameters& params) const {
  absl::MutexLock lock(&mu_);
  auto it = blas_plans_algorithms_map_.find(params);
  if (it == blas_plans_algorithms_map_.end()) {
    return std::nullopt;
  }
  return it->second;
}

void BlasPlansAutotuneCache::Insert(se::BatchMatmulParameters params,
                                    se::blas::AlgorithmConfig config) {
  absl::MutexLock lock(&mu_);
  blas_plans_algorithms_map_.insert({std::move(params), std::move(config)});
}

BlasPlansAutotuneCache& GetBlasPlansAutotuneCache() {
  static auto& instance = *new BlasPlansAutotuneCache();
  return instance;
}

}  // namespace gpu
}  // namespace xla
