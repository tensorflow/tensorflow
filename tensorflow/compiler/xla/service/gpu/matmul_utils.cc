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
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/statusor.h"

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_blas_lt.h"
#include "tensorflow/compiler/xla/stream_executor/host_or_device_scalar.h"
#include "tensorflow/tsl/platform/tensor_float_32_utils.h"
#endif  // GOOGLE_CUDA

namespace xla {
namespace gpu {

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

  std::vector<int64_t> minor_to_major;
  for (size_t i = 0; i < shape.rank();) {
    // The GeMM output always has its layout set such that the batch, row, and
    // col dim groups are each laid out physically sequentially. GeMM operands
    // must, therefore, be laid out similarly.
    auto check_physically_sequential = [&](absl::Span<const int64_t> dims) {
      for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
        // NOTE: `i` is incremented as we check the dimensions.
        if (*it != shape.layout().minor_to_major()[i++])
          return InvalidArgument("dims not physically_sequential");
      }
      return OkStatus();
    };

    int64_t dim = shape.layout().minor_to_major()[i];
    if (!row_dims.empty() && dim == row_dims.back()) {
      minor_to_major.push_back(1);
      TF_RETURN_IF_ERROR(check_physically_sequential(row_dims));
    } else if (!col_dims.empty() && dim == col_dims.back()) {
      minor_to_major.push_back(2);
      TF_RETURN_IF_ERROR(check_physically_sequential(col_dims));
    } else if (!batch_dims.empty() && (dim == batch_dims.back())) {
      minor_to_major.push_back(0);
      TF_RETURN_IF_ERROR(check_physically_sequential(batch_dims));
    } else {
      return InvalidArgument("dims not physically sequential");
    }
  }

  if (col_dims.empty()) minor_to_major.push_back(2);
  if (row_dims.empty()) minor_to_major.push_back(1);
  if (batch_dims.empty()) minor_to_major.push_back(0);

  auto dim_size = [&](absl::Span<const int64_t> dims) {
    return absl::c_accumulate(dims, 1, [&](int64_t size, int64_t dim) {
      return size * shape.dimensions(dim);
    });
  };

  return ShapeUtil::MakeShapeWithDenseLayout(
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

void MatrixLayout::Transpose() {
  std::swap(num_rows, num_cols);
  order = (order == Order::kRowMajor) ? Order::kColumnMajor : Order::kRowMajor;
}

namespace {
// Returns the relative order of 'dims' as indices from 0 to dims.size() - 1.
// Let 'indices' be the returned vector, then it holds that
// dims[indices[i - 1]] < dims[indices[i]] for 0 < i < dims.size()
std::vector<int64_t> NormalizedRelativeOrder(absl::Span<const int64_t> dims) {
  // Remap the dimensions to values between 0 and dims.size() - 1, keeping their
  // relative order the same.
  std::vector<int64_t> indices(dims.size());
  absl::c_iota(indices, 0);
  absl::c_sort(indices,
               [&](int64_t a, int64_t b) { return dims[a] < dims[b]; });
  return indices;
}
}  // namespace

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

  // TransposeFolding assumes that folding the transpose into the dot operand
  // doesn't change the dot shape. This means that the non-contracting
  // dimensions of the dot operand need to keep their relative order.
  auto transposed_non_contracting_dims = transposed(non_contracting_dims);
  if (NormalizedRelativeOrder(non_contracting_dims) !=
      NormalizedRelativeOrder(transposed_non_contracting_dims)) {
    return false;
  }

  // If we're able to construct a valid `MatrixLayout` for the transposed
  // dimensions, then GeMM can support folding the transpose.
  return MatrixLayout::For(transpose.operand(0)->shape(),
                           transposed(batch_dims), transposed(contracting_dims),
                           transposed_non_contracting_dims)
      .ok();
}

/*static*/ StatusOr<GemmConfig> GemmConfig::For(
    const Shape& lhs_shape, absl::Span<const int64_t> lhs_batch_dims,
    absl::Span<const int64_t> lhs_contracting_dims, const Shape& rhs_shape,
    absl::Span<const int64_t> rhs_batch_dims,
    absl::Span<const int64_t> rhs_contracting_dims, const Shape& output_shape,
    double alpha_real, double alpha_imag, double beta,
    std::optional<int64_t> algorithm, int64_t compute_precision) {
  return GemmConfig::For(lhs_shape, lhs_batch_dims, lhs_contracting_dims,
                         rhs_shape, rhs_batch_dims, rhs_contracting_dims,
                         output_shape, output_shape, alpha_real, alpha_imag,
                         beta, algorithm, compute_precision);
}

/*static*/ StatusOr<GemmConfig> GemmConfig::For(
    const Shape& lhs_shape, absl::Span<const int64_t> lhs_batch_dims,
    absl::Span<const int64_t> lhs_contracting_dims, const Shape& rhs_shape,
    absl::Span<const int64_t> rhs_batch_dims,
    absl::Span<const int64_t> rhs_contracting_dims, const Shape& c_shape,
    const Shape& output_shape, double alpha_real, double alpha_imag,
    double beta, std::optional<int64_t> algorithm, int64_t compute_precision,
    const Shape* bias_shape_ptr) {
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
  Shape c_matrix_shape = c_shape;
#if CUDA_VERSION >= 12000
  if (lhs_shape.element_type() == F8E4M3FN ||
      lhs_shape.element_type() == F8E5M2) {
    if (beta == 0.0) {
      c_matrix_shape = output_shape;
      if (output_shape.element_type() == F32) {
        c_matrix_shape.set_element_type(F32);
      } else if (output_shape.element_type() == BF16) {
        c_matrix_shape.set_element_type(BF16);
      } else if (output_shape.element_type() == F16) {
        c_matrix_shape.set_element_type(F16);
      } else {
        // The C type has to be compatible with bias type. See
        // https://docs.nvidia.com/cuda/cublas/index.html#id10 for details.
        c_matrix_shape.set_element_type(
            bias_shape_ptr != nullptr ? bias_shape_ptr->element_type() : BF16);
      }
    } else {
      c_matrix_shape.set_element_type(c_shape.element_type());
    }
  }
#endif

  TF_ASSIGN_OR_RETURN(MatrixLayout c_layout,
                      MatrixLayout::For(c_matrix_shape, output_batch_dims,
                                        output_row_dims, output_col_dims));

  // TODO(cjfj): We should also check that the batch, contracting and
  // non-contracting dimensions match in size and relative physical location.
  // TODO(philipphack): Check the remaining dimensions in the FP8 case once
  // cuBLASLt supports the NN configuration.
  if (lhs_shape.element_type() != F8E4M3FN &&
      lhs_shape.element_type() != F8E5M2) {
    TF_RET_CHECK(lhs_layout.num_cols == rhs_layout.num_rows);
    TF_RET_CHECK(output_layout.num_rows == lhs_layout.num_rows);
    TF_RET_CHECK(output_layout.num_cols == rhs_layout.num_cols);
  }
  TF_RET_CHECK(c_layout.num_rows == output_layout.num_rows);
  TF_RET_CHECK(c_layout.num_cols == output_layout.num_cols);
  TF_RET_CHECK((lhs_layout.batch_size == output_layout.batch_size) ||
               (lhs_layout.batch_size == 1));
  TF_RET_CHECK((rhs_layout.batch_size == output_layout.batch_size) ||
               (rhs_layout.batch_size == 1));

  switch (output_shape.element_type()) {
    case F8E4M3FN:
    case F8E5M2:
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
      lhs_layout,
      rhs_layout,
      c_layout,
      output_layout,
      {alpha_real, alpha_imag},
      beta,
      algorithm,
      compute_precision,
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
  const Shape& output_shape =
      gemm->shape().IsTuple() ? gemm->shape().tuple_shapes(0) : gemm->shape();

  return GemmConfig::For(
      lhs_shape, dot_dims.lhs_batch_dimensions(),
      dot_dims.lhs_contracting_dimensions(), rhs_shape,
      dot_dims.rhs_batch_dimensions(), dot_dims.rhs_contracting_dimensions(),
      output_shape, config.alpha_real(), config.alpha_imag(), config.beta(),
      algorithm, se::blas::kDefaultComputePrecision);
}

/*static*/ StatusOr<GemmConfig> GemmConfig::For(mlir::lmhlo_gpu::GEMMOp op) {
  mlir::mhlo::DotDimensionNumbersAttr dot_dims = op.getDotDimensionNumbers();

  std::optional<int64_t> algorithm;
  if (op.getAlgorithm()) algorithm = *op.getAlgorithm();

  int64_t compute_precision = 0;  // Default
  if (op.getPrecisionConfig().has_value()) {
    auto precision_config = op.getPrecisionConfig();
    for (auto attr : precision_config.value()) {
      int64_t value = static_cast<int64_t>(
          attr.template cast<mlir::mhlo::PrecisionAttr>().getValue());
      if (value > compute_precision) {
        compute_precision = value;
      }
    }
  }

  return GemmConfig::For(
      GetShape(op.getA()), dot_dims.getLhsBatchingDimensions(),
      dot_dims.getLhsContractingDimensions(), GetShape(op.getB()),
      dot_dims.getRhsBatchingDimensions(),
      dot_dims.getRhsContractingDimensions(), GetShape(op.getC()),
      op.getAlphaReal().convertToDouble(), op.getAlphaImag().convertToDouble(),
      op.getBeta().convertToDouble(), algorithm, compute_precision);
}

StatusOr<se::blas::ComputationType> GetBlasComputationType(
    PrimitiveType lhs_dtype, PrimitiveType output_dtype,
    int64_t compute_precision) {
  switch (output_dtype) {
    case F8E5M2:    // fall-through
    case F8E4M3FN:  // fall-through
    case F16:       // fall-through
    case BF16:
      // Accumulate in f32 precision.
      return se::blas::ComputationType::kF32;
    case F32:  // fall-through
    case C64:
#if GOOGLE_CUDA
      if (tsl::tensor_float_32_execution_enabled() && compute_precision <= 1 &&
          lhs_dtype != F8E4M3FN && lhs_dtype != F8E5M2) {
        // CublasLt requires compute type to be F32 for F8 matmul.
        return se::blas::ComputationType::kTF32AsF32;
      }
#endif
      return se::blas::ComputationType::kF32;
    case F64:  // fall-through
    case C128:
      return se::blas::ComputationType::kF64;
    case S32:
      return se::blas::ComputationType::kI32;
    default:
      return InternalError("GetBlasComputationType: unsupported type");
  }
}

namespace cublas_lt {

se::blas::DataType GetScaleType(se::blas::DataType c_type,
                                se::blas::ComputationType computation_type) {
  return ((computation_type == se::blas::ComputationType::kF32) &&
          (c_type != se::blas::DataType::kComplexFloat))
             ? se::blas::DataType::kFloat
             : c_type;
}

}  // namespace cublas_lt

namespace {

// This struct contains the metadata of a matrix, e.g., its base address and
// dimensions.
struct MatrixDescriptor {
  se::DeviceMemoryBase data;
  int64_t leading_dim_stride;
  int64_t batch_stride;
  se::blas::Transpose transpose;

  template <typename T>
  se::DeviceMemory<T> cast() const {
    return se::DeviceMemory<T>(data);
  }
};

// BLAS GeMM's output is column-major. If we require row-major, use identity:
// C^T = (A @ B)^T = B^T @ A^T.
bool MakeOutputColumnMajor(MatrixLayout& lhs, MatrixLayout& rhs,
                           MatrixLayout& output) {
  bool swap_operands = output.order != MatrixLayout::Order::kColumnMajor;
  if (swap_operands) {
    std::swap(lhs, rhs);
    lhs.Transpose();
    rhs.Transpose();
    output.Transpose();
  }
  return swap_operands;
}

bool MakeOutputColumnMajor(MatrixLayout& lhs, MatrixLayout& rhs,
                           MatrixLayout& output, MatrixLayout& c) {
  bool swap_operands = output.order != MatrixLayout::Order::kColumnMajor;
  if (swap_operands) {
    std::swap(lhs, rhs);
    rhs.Transpose();
    lhs.Transpose();
    c.Transpose();
    output.Transpose();
  }
  return swap_operands;
}

se::blas::Transpose AsBlasTranspose(MatrixLayout::Order order) {
  // BLAS is column-major by default.
  return (order == MatrixLayout::Order::kColumnMajor)
             ? se::blas::Transpose::kNoTranspose
             : se::blas::Transpose::kTranspose;
}

MatrixDescriptor GetMatrixDesc(const MatrixLayout& layout,
                               se::DeviceMemoryBase data) {
  return {
      data,
      layout.leading_dim_stride,
      layout.batch_stride,
      AsBlasTranspose(layout.order),
  };
}

template <typename Input, typename Output>
Status DoGemmWithAlgorithm(int64_t batch_size, int64_t m, int64_t n, int64_t k,
                           const MatrixDescriptor& lhs,
                           const MatrixDescriptor& rhs,
                           const MatrixDescriptor& output, Output alpha,
                           Output beta, se::Stream* stream,
                           se::blas::AlgorithmType algorithm,
                           se::blas::ComputePrecision compute_precision,
                           se::blas::ProfileResult* profile_result) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  PrimitiveType lhs_type = primitive_util::NativeToPrimitiveType<Input>();
  PrimitiveType output_type = primitive_util::NativeToPrimitiveType<Output>();
  TF_ASSIGN_OR_RETURN(
      se::blas::ComputationType computation_type,
      GetBlasComputationType(lhs_type, output_type, compute_precision));
  se::DeviceMemory<Output> output_data(output.data);

  if (batch_size != 1) {
    return stream->ThenBlasGemmStridedBatchedWithAlgorithm(
        lhs.transpose, rhs.transpose, m, n, k, alpha, lhs.cast<Input>(),
        lhs.leading_dim_stride, lhs.batch_stride, rhs.cast<Input>(),
        rhs.leading_dim_stride, rhs.batch_stride, beta, &output_data,
        output.leading_dim_stride, output.batch_stride, batch_size,
        computation_type, algorithm, compute_precision, profile_result);
  } else {
    return stream->ThenBlasGemmWithAlgorithm(
        lhs.transpose, rhs.transpose, m, n, k, alpha, lhs.cast<Input>(),
        lhs.leading_dim_stride, rhs.cast<Input>(), rhs.leading_dim_stride, beta,
        &output_data, output.leading_dim_stride, computation_type, algorithm,
        compute_precision, profile_result);
  }
}

template <typename Input>
Status DoGemm(int64_t batch_size, int64_t m, int64_t n, int64_t k,
              const MatrixDescriptor& lhs, const MatrixDescriptor& rhs,
              const MatrixDescriptor& output, Input alpha, Input beta,
              se::Stream* stream,
              std::optional<se::blas::AlgorithmType> algorithm,
              se::blas::ComputePrecision compute_precision,
              se::blas::ProfileResult* profile_result) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  se::DeviceMemory<Input> output_data(output.data);

  if (algorithm) {
    return DoGemmWithAlgorithm<Input, Input>(
        batch_size, m, n, k, lhs, rhs, output, alpha, beta, stream, *algorithm,
        compute_precision, profile_result);
  }

  if (batch_size != 1) {
    return stream->ThenBlasGemmStridedBatched(
        lhs.transpose, rhs.transpose, m, n, k, alpha, lhs.cast<Input>(),
        lhs.leading_dim_stride, lhs.batch_stride, rhs.cast<Input>(),
        rhs.leading_dim_stride, rhs.batch_stride, beta, &output_data,
        output.leading_dim_stride, output.batch_stride, batch_size,
        compute_precision);
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

  MatrixLayout lhs_layout = config.lhs_layout;
  MatrixLayout rhs_layout = config.rhs_layout;
  MatrixLayout output_layout = config.output_layout;
  bool must_swap_operands =
      MakeOutputColumnMajor(lhs_layout, rhs_layout, output_layout);
  if (must_swap_operands) {
    std::swap(lhs_buffer, rhs_buffer);
  }

  int64_t m = output_layout.num_rows;
  int64_t n = output_layout.num_cols;
  int64_t k = lhs_layout.num_cols;
  MatrixDescriptor lhs = GetMatrixDesc(lhs_layout, lhs_buffer);
  MatrixDescriptor rhs = GetMatrixDesc(rhs_layout, rhs_buffer);
  MatrixDescriptor output = GetMatrixDesc(output_layout, output_buffer);
  int64_t batch_size = output_layout.batch_size;

  if (!algorithm) algorithm = config.algorithm;

  if ((output_layout.dtype == F16 || output_layout.dtype == BF16 ||
       output_layout.dtype == F32 || output_layout.dtype == F64 ||
       output_layout.dtype == C64 || output_layout.dtype == C128) &&
      (lhs_layout.dtype != output_layout.dtype ||
       rhs_layout.dtype != output_layout.dtype)) {
    return InternalError(
        "GEMM lhs type(%s) and rhs type(%s) must match output type(%s)",
        primitive_util::LowercasePrimitiveTypeName(lhs_layout.dtype),
        primitive_util::LowercasePrimitiveTypeName(rhs_layout.dtype),
        primitive_util::LowercasePrimitiveTypeName(output_layout.dtype));
  }

  switch (output_layout.dtype) {
    case S32:
      if (!algorithm) algorithm = se::blas::kDefaultGemmAlgo;
      return DoGemmWithAlgorithm<int8_t, int32_t>(
          batch_size, m, n, k, lhs, rhs, output,
          static_cast<int32_t>(config.alpha.real()),
          static_cast<int32_t>(config.beta), stream, *algorithm,
          se::blas::kDefaultComputePrecision, profile_result);
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
      return InternalError(
          "Unexpected GEMM dtype: %s",
          primitive_util::LowercasePrimitiveTypeName(output_layout.dtype));
  }
}

namespace cublas_lt {

StatusOr<bool> EpilogueAddsVectorBias(GemmBackendConfig_Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
    case GemmBackendConfig::RELU:
    case GemmBackendConfig::GELU:
    case GemmBackendConfig::GELU_AUX:
      return false;
    case GemmBackendConfig::BIAS:
    case GemmBackendConfig::BIAS_RELU:
    case GemmBackendConfig::BIAS_GELU:
    case GemmBackendConfig::BIAS_GELU_AUX:
      return true;
    default:
      return InternalError("Unknown Epilogue.");
  }
}

StatusOr<bool> EpilogueHasAuxiliaryOutput(GemmBackendConfig_Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
    case GemmBackendConfig::RELU:
    case GemmBackendConfig::GELU:
    case GemmBackendConfig::BIAS:
    case GemmBackendConfig::BIAS_RELU:
    case GemmBackendConfig::BIAS_GELU:
      return false;
    case GemmBackendConfig::GELU_AUX:
    case GemmBackendConfig::BIAS_GELU_AUX:
      return true;
    default:
      return InternalError("Unknown Epilogue.");
  }
}

}  // namespace cublas_lt

StatusOr<se::blas::DataType> AsBlasDataType(PrimitiveType dtype) {
  switch (dtype) {
    case F8E5M2:
      return se::blas::DataType::kF8E5M2;
    case F8E4M3FN:
      return se::blas::DataType::kF8E4M3FN;
    case S8:
      return se::blas::DataType::kInt8;
    case F16:
      return se::blas::DataType::kHalf;
    case BF16:
      return se::blas::DataType::kBF16;
    case F32:
      return se::blas::DataType::kFloat;
    case S32:
      return se::blas::DataType::kInt32;
    case F64:
      return se::blas::DataType::kDouble;
    case C64:
      return se::blas::DataType::kComplexFloat;
    case C128:
      return se::blas::DataType::kComplexDouble;
    default:
      return InternalError("AsBlasDataType: unsupported type");
  }
}

#if GOOGLE_CUDA

namespace {

StatusOr<se::cuda::BlasLt::MatrixLayout> AsBlasLtMatrixLayout(
    const MatrixLayout& layout) {
  TF_ASSIGN_OR_RETURN(se::blas::DataType dtype, AsBlasDataType(layout.dtype));

  auto order = (layout.order == MatrixLayout::Order::kColumnMajor)
                   ? se::cuda::BlasLt::MatrixLayout::Order::kColumnMajor
                   : se::cuda::BlasLt::MatrixLayout::Order::kRowMajor;

  return se::cuda::BlasLt::MatrixLayout::Create(
      dtype, layout.num_rows, layout.num_cols, order, layout.batch_size,
      layout.leading_dim_stride, layout.batch_stride);
}

template <cudaDataType_t CudaT>
struct CudaToNativeT;

#if CUDA_VERSION >= 11080
template <>
struct CudaToNativeT<CUDA_R_8F_E4M3> {
  using type = tsl::float8_e4m3fn;
};
template <>
struct CudaToNativeT<CUDA_R_8F_E5M2> {
  using type = tsl::float8_e5m2;
};
#endif

template <>
struct CudaToNativeT<CUDA_R_16BF> {
  using type = Eigen::bfloat16;
};
template <>
struct CudaToNativeT<CUDA_R_16F> {
  using type = Eigen::half;
};
template <>
struct CudaToNativeT<CUDA_R_32F> {
  using type = float;
};
template <>
struct CudaToNativeT<CUDA_R_64F> {
  using type = double;
};
template <>
struct CudaToNativeT<CUDA_C_32F> {
  using type = complex64;
};
template <>
struct CudaToNativeT<CUDA_C_64F> {
  using type = complex128;
};

}  // namespace

namespace cublas_lt {

StatusOr<se::cuda::BlasLt::Epilogue> AsBlasLtEpilogue(
    mlir::lmhlo_gpu::CublasLtMatmulEpilogue epilogue) {
  switch (epilogue) {
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::Default:
      return se::cuda::BlasLt::Epilogue::kDefault;
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::Relu:
      return se::cuda::BlasLt::Epilogue::kReLU;
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::Gelu:
      return se::cuda::BlasLt::Epilogue::kGELU;
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::GeluAux:
      return se::cuda::BlasLt::Epilogue::kGELUWithAux;
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::Bias:
      return se::cuda::BlasLt::Epilogue::kBias;
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::BiasRelu:
      return se::cuda::BlasLt::Epilogue::kBiasThenReLU;
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::BiasGelu:
      return se::cuda::BlasLt::Epilogue::kBiasThenGELU;
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::BiasGeluAux:
      return se::cuda::BlasLt::Epilogue::kBiasThenGELUWithAux;
  }
  return InternalError("unexpected epilogue value");
}

/*static*/ StatusOr<MatmulPlan> MatmulPlan::From(
    const GemmConfig& config, se::cuda::BlasLt::Epilogue epilogue) {
  MatrixLayout lhs_layout = config.lhs_layout;
  MatrixLayout rhs_layout = config.rhs_layout;
  MatrixLayout output_layout = config.output_layout;
  MatrixLayout c_layout = config.c_layout;

  // cublasLt matmul requires batch sizes to be equal. If only one operand has a
  // batch, the other will be broadcast (as its batch_stride == 0).
  size_t batch_size = std::max(lhs_layout.batch_size, rhs_layout.batch_size);
  lhs_layout.batch_size = batch_size;
  rhs_layout.batch_size = batch_size;

  bool must_swap_operands =
      MakeOutputColumnMajor(lhs_layout, rhs_layout, c_layout, output_layout);

  // Do not transopse either input. Note the cuBLASLt documentation somewhat
  // incorrectly claims "A must be transposed and B non-transposed" when A and B
  // are FP8 (https://docs.nvidia.com/cuda/cublas/#cublasltmatmul). In reality,
  // this is only true if A and B are column-major. If A is row-major, A must
  // *not* be transposed, and if B is row-major, B must be transposed. We never
  // transpose A or B, and expect the caller to ensure A is row-major and B is
  // column when A and B are FP8.
  const se::blas::Transpose trans_a = se::blas::Transpose::kNoTranspose;
  const se::blas::Transpose trans_b = se::blas::Transpose::kNoTranspose;
  if (primitive_util::IsF8Type(lhs_layout.dtype) &&
      lhs_layout.order == MatrixLayout::Order::kColumnMajor) {
    return InternalError("The F8 LHS must be column-major");
  }
  if (primitive_util::IsF8Type(rhs_layout.dtype) &&
      rhs_layout.order == MatrixLayout::Order::kRowMajor) {
    return InternalError("The F8 RHS must be row-major");
  }

  TF_ASSIGN_OR_RETURN(se::blas::DataType output_dtype,
                      AsBlasDataType(output_layout.dtype));
  TF_ASSIGN_OR_RETURN(
      se::blas::ComputationType computation_type,
      GetBlasComputationType(lhs_layout.dtype, output_layout.dtype,
                             config.compute_precision));

  TF_ASSIGN_OR_RETURN(
      se::cuda::BlasLt::MatmulDesc op_desc,
      se::cuda::BlasLt::MatmulDesc::Create(
          computation_type, GetScaleType(output_dtype, computation_type),
          trans_a, trans_b, epilogue));

  TF_ASSIGN_OR_RETURN(se::cuda::BlasLt::MatrixLayout a_desc,
                      AsBlasLtMatrixLayout(lhs_layout));
  TF_ASSIGN_OR_RETURN(se::cuda::BlasLt::MatrixLayout b_desc,
                      AsBlasLtMatrixLayout(rhs_layout));
  TF_ASSIGN_OR_RETURN(se::cuda::BlasLt::MatrixLayout c_desc,
                      AsBlasLtMatrixLayout(c_layout));
  TF_ASSIGN_OR_RETURN(se::cuda::BlasLt::MatrixLayout d_desc,
                      AsBlasLtMatrixLayout(output_layout));

  return MatmulPlan{
      se::cuda::BlasLt::MatmulPlan{std::move(op_desc), std::move(a_desc),
                                   std::move(b_desc), std::move(c_desc),
                                   std::move(d_desc)},
      config.alpha, config.beta, must_swap_operands};
}

template <typename Scale, typename A, typename B, typename C, typename D>
Status MatmulPlan::DoMatmul(
    se::Stream* stream, se::DeviceMemoryBase a_buffer,
    se::DeviceMemoryBase b_buffer, se::DeviceMemoryBase c_buffer,
    se::DeviceMemoryBase d_buffer, se::DeviceMemoryBase bias_buffer,
    se::DeviceMemoryBase aux_buffer, se::DeviceMemoryBase a_scale_buffer,
    se::DeviceMemoryBase b_scale_buffer, se::DeviceMemoryBase c_scale_buffer,
    se::DeviceMemoryBase d_scale_buffer, se::DeviceMemoryBase d_amax_buffer,
    const se::cuda::BlasLt::MatmulAlgorithm& algorithm,
    se::ScratchAllocator& scratch_allocator,
    se::blas::ProfileResult* profile_result) const {
  se::cuda::BlasLt* blas_lt = se::cuda::GetBlasLt(stream);
  TF_RET_CHECK(blas_lt != nullptr);

  Scale alpha;
  if constexpr (std::is_same_v<Scale, complex64> ||
                std::is_same_v<Scale, complex128>) {
    alpha = static_cast<Scale>(alpha_);
  } else {
    alpha = static_cast<Scale>(alpha_.real());
  }

  Scale beta = static_cast<Scale>(beta_);

  se::DeviceMemory<D> output(d_buffer);
  return blas_lt->DoMatmul(
      stream, plan_, se::HostOrDeviceScalar<Scale>(alpha),
      se::DeviceMemory<A>(a_buffer), se::DeviceMemory<B>(b_buffer),
      se::HostOrDeviceScalar<Scale>(beta), se::DeviceMemory<C>(c_buffer),
      output, algorithm, scratch_allocator, se::DeviceMemory<C>(bias_buffer),
      aux_buffer, se::DeviceMemory<Scale>(a_scale_buffer),
      se::DeviceMemory<Scale>(b_scale_buffer),
      se::DeviceMemory<Scale>(c_scale_buffer),
      se::DeviceMemory<Scale>(d_scale_buffer),
      se::DeviceMemory<Scale>(d_amax_buffer), profile_result);
}

Status MatmulPlan::ExecuteOnStream(
    se::Stream* stream, se::DeviceMemoryBase a_buffer,
    se::DeviceMemoryBase b_buffer, se::DeviceMemoryBase c_buffer,
    se::DeviceMemoryBase d_buffer, se::DeviceMemoryBase bias_buffer,
    se::DeviceMemoryBase aux_buffer, se::DeviceMemoryBase a_scale_buffer,
    se::DeviceMemoryBase b_scale_buffer, se::DeviceMemoryBase c_scale_buffer,
    se::DeviceMemoryBase d_scale_buffer, se::DeviceMemoryBase d_amax_buffer,
    const se::cuda::BlasLt::MatmulAlgorithm& algorithm,
    se::ScratchAllocator& scratch_allocator,
    se::blas::ProfileResult* profile_result) const {
  if (must_swap_operands_) {
    std::swap(a_buffer, b_buffer);
  }

  std::tuple<cudaDataType_t, cudaDataType_t, cudaDataType_t, cudaDataType_t>
      operand_types{plan_.a_desc.type(), plan_.b_desc.type(),
                    plan_.c_desc.type(), plan_.d_desc.type()};

#define TYPED_MATMUL(SCALENTYPE, ATYPE, BTYPE, CTYPE, DTYPE)                \
  if (operand_types == std::make_tuple(ATYPE, BTYPE, CTYPE, DTYPE)) {       \
    return DoMatmul<SCALENTYPE, CudaToNativeT<ATYPE>::type,                 \
                    CudaToNativeT<BTYPE>::type, CudaToNativeT<CTYPE>::type, \
                    CudaToNativeT<DTYPE>::type>(                            \
        stream, a_buffer, b_buffer, c_buffer, d_buffer, bias_buffer,        \
        aux_buffer, a_scale_buffer, b_scale_buffer, c_scale_buffer,         \
        d_scale_buffer, d_amax_buffer, algorithm, scratch_allocator,        \
        profile_result);                                                    \
  }

#if CUDA_VERSION >= 11080
  // FP8 compatible type combinations (see cuBLASLt documentation):
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, CUDA_R_16BF)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF,
               CUDA_R_8F_E4M3)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16F,
               CUDA_R_8F_E4M3)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16F, CUDA_R_16F)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_32F, CUDA_R_32F)

  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16BF, CUDA_R_16BF)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16BF,
               CUDA_R_8F_E4M3)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16BF,
               CUDA_R_8F_E5M2)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16F,
               CUDA_R_8F_E4M3)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16F,
               CUDA_R_8F_E5M2)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16F, CUDA_R_16F)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_32F, CUDA_R_32F)

  TYPED_MATMUL(float, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16BF, CUDA_R_16BF)
  TYPED_MATMUL(float, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16BF,
               CUDA_R_8F_E4M3)
  TYPED_MATMUL(float, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16BF,
               CUDA_R_8F_E5M2)
  TYPED_MATMUL(float, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16F,
               CUDA_R_8F_E4M3)
  TYPED_MATMUL(float, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16F,
               CUDA_R_8F_E5M2)
  TYPED_MATMUL(float, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16F, CUDA_R_16F)
  TYPED_MATMUL(float, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_32F, CUDA_R_32F)
#endif

  // Other data types:
  TYPED_MATMUL(float, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F)
  TYPED_MATMUL(float, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF)
  TYPED_MATMUL(float, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F)
  TYPED_MATMUL(double, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F)
  TYPED_MATMUL(complex64, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F)
  TYPED_MATMUL(complex128, CUDA_C_64F, CUDA_C_64F, CUDA_C_64F, CUDA_C_64F)

#undef TYPED_MATMUL

  return InternalError("Unexpected dtype");
}

StatusOr<std::vector<se::cuda::BlasLt::MatmulAlgorithm>>
MatmulPlan::GetAlgorithms(se::Stream* stream) const {
  se::cuda::BlasLt* blas_lt = se::cuda::GetBlasLt(stream);
  TF_RET_CHECK(blas_lt != nullptr);
  TF_ASSIGN_OR_RETURN(auto preference,
                      se::cuda::BlasLt::MatmulPreference::Create(
                          /*max_workspace_size=*/1ll << 32));  // 4GB
  return blas_lt->GetMatmulAlgorithms(plan_, preference);
}

bool MatmulPlan::IsF8MatmulTrivialMatrixBias() const {
  return (plan_.a_desc.type() == CUDA_R_8F_E4M3 ||
          plan_.a_desc.type() == CUDA_R_8F_E5M2) &&
         (beta_ == 0.0);
}

}  // namespace cublas_lt

#endif  // GOOGLE_CUDA

}  // namespace gpu
}  // namespace xla
