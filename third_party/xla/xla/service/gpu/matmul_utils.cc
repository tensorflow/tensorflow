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

#include "xla/service/gpu/matmul_utils.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

#if GOOGLE_CUDA
#include "xla/stream_executor/host_or_device_scalar.h"
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

const tsl::protobuf::RepeatedField<int64_t>& BatchDimensionsForOperand(
    const HloInstruction& dot, const int operand_number) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  if (operand_number == 0) {
    return dimension_numbers.lhs_batch_dimensions();
  }
  return dimension_numbers.rhs_batch_dimensions();
}

int64_t ContractingDimensionIndex(const HloInstruction& dot,
                                  const int operand_number) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  if (operand_number == 0) {
    CHECK_EQ(dimension_numbers.lhs_contracting_dimensions().size(), 1);
    return dimension_numbers.lhs_contracting_dimensions(0);
  }
  CHECK_EQ(dimension_numbers.rhs_contracting_dimensions().size(), 1);
  return dimension_numbers.rhs_contracting_dimensions(0);
}

int64_t NonContractingDimensionIndex(const HloInstruction& dot,
                                     const int operand_number) {
  StatusOr<std::vector<int64_t>> non_contracting_dims =
      GetNonContractingDims(dot.operand(operand_number)->shape(),
                            BatchDimensionsForOperand(dot, operand_number),
                            {ContractingDimensionIndex(dot, operand_number)});
  TF_CHECK_OK(non_contracting_dims.status());
  CHECK_EQ(non_contracting_dims->size(), 1);
  return non_contracting_dims->front();
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
      shape.element_type(), num_rows,           num_cols,     order,
      batch_size,           leading_dim_stride, batch_stride,
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
                         /*c_shape=*/output_shape, /*bias_shape_ptr=*/nullptr,
                         output_shape, alpha_real, alpha_imag, beta, algorithm,
                         compute_precision);
}

/*static*/ StatusOr<GemmConfig> GemmConfig::For(
    const Shape& lhs_shape, absl::Span<const int64_t> lhs_batch_dims,
    absl::Span<const int64_t> lhs_contracting_dims, const Shape& rhs_shape,
    absl::Span<const int64_t> rhs_batch_dims,
    absl::Span<const int64_t> rhs_contracting_dims, const Shape& c_shape,
    const Shape* bias_shape_ptr, const Shape& output_shape, double alpha_real,
    double alpha_imag, double beta, std::optional<int64_t> algorithm,
    int64_t compute_precision) {
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
  if (primitive_util::IsF8Type(lhs_shape.element_type()) &&
      primitive_util::IsF8Type(output_shape.element_type()) && beta == 0.0) {
    // By default, if c is not present (i.e., beta is 0), c_shape will be the
    // output shape. cublasLT requires a valid c_shape to be passed, even if c
    // is not present, and normally setting it to the output shape is fine. But
    // for matmuls with FP8 inputs and outputs, C must instead have the same
    // dtype as the vector bias if present, and either BF16 or F16 otherwise. So
    // we set the dtype of C here.
    c_matrix_shape.set_element_type(
        bias_shape_ptr != nullptr ? bias_shape_ptr->element_type() : BF16);
  }

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
      compute_precision,
      algorithm,
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

se::blas::Transpose AsBlasTranspose(MatrixLayout::Order order) {
  // BLAS is column-major by default.
  return (order == MatrixLayout::Order::kColumnMajor)
             ? se::blas::Transpose::kNoTranspose
             : se::blas::Transpose::kTranspose;
}

MatrixDescriptor GetMatrixDesc(const MatrixLayout& layout,
                               se::DeviceMemoryBase data) {
  return MatrixDescriptor{
      data,
      *layout.leading_dim_stride,
      *layout.batch_stride,
      AsBlasTranspose(layout.order),
  };
}

template <typename Scale, typename Input, typename Output>
Status DoGemmWithAlgorithm(int64_t batch_size, int64_t m, int64_t n, int64_t k,
                           const MatrixDescriptor& lhs,
                           const MatrixDescriptor& rhs,
                           const MatrixDescriptor& output, Scale alpha,
                           Scale beta, se::Stream* stream,
                           se::blas::AlgorithmType algorithm,
                           se::blas::ComputePrecision compute_precision,
                           const se::NumericOptions& numeric_options,
                           se::blas::ProfileResult* profile_result) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  PrimitiveType lhs_type = primitive_util::NativeToPrimitiveType<Input>();
  PrimitiveType output_type = primitive_util::NativeToPrimitiveType<Output>();
  TF_ASSIGN_OR_RETURN(se::blas::ComputationType computation_type,
                      se::gpu::GetBlasComputationType(lhs_type, output_type,
                                                      compute_precision));
  se::DeviceMemory<Output> output_data(output.data);

  if (batch_size != 1) {
    return stream->ThenBlasGemmStridedBatchedWithAlgorithm(
        lhs.transpose, rhs.transpose, m, n, k, alpha, lhs.cast<Input>(),
        lhs.leading_dim_stride, lhs.batch_stride, rhs.cast<Input>(),
        rhs.leading_dim_stride, rhs.batch_stride, beta, &output_data,
        output.leading_dim_stride, output.batch_stride, batch_size,
        computation_type, algorithm, numeric_options, profile_result);
  } else {
    return stream->ThenBlasGemmWithAlgorithm(
        lhs.transpose, rhs.transpose, m, n, k, alpha, lhs.cast<Input>(),
        lhs.leading_dim_stride, rhs.cast<Input>(), rhs.leading_dim_stride, beta,
        &output_data, output.leading_dim_stride, computation_type, algorithm,
        numeric_options, profile_result);
  }
}

template <typename Scale, typename Input, typename Output>
Status DoGemm(int64_t batch_size, int64_t m, int64_t n, int64_t k,
              const MatrixDescriptor& lhs, const MatrixDescriptor& rhs,
              const MatrixDescriptor& output, Scale alpha, Scale beta,
              se::Stream* stream,
              std::optional<se::blas::AlgorithmType> algorithm,
              se::blas::ComputePrecision compute_precision,
              const se::NumericOptions& numeric_options,
              se::blas::ProfileResult* profile_result) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  se::DeviceMemory<Output> output_data(output.data);

#if GOOGLE_CUDA
  if (algorithm) {
    return DoGemmWithAlgorithm<Scale, Input, Output>(
        batch_size, m, n, k, lhs, rhs, output, alpha, beta, stream, *algorithm,
        compute_precision, numeric_options, profile_result);
  }
#endif

  if (batch_size != 1) {
    return stream->ThenBlasGemmStridedBatched(
        lhs.transpose, rhs.transpose, m, n, k, alpha, lhs.cast<Input>(),
        lhs.leading_dim_stride, lhs.batch_stride, rhs.cast<Input>(),
        rhs.leading_dim_stride, rhs.batch_stride, beta, &output_data,
        output.leading_dim_stride, output.batch_stride, batch_size,
        numeric_options);
  }

  return stream->ThenBlasGemm(
      lhs.transpose, rhs.transpose, m, n, k, alpha, lhs.cast<Input>(),
      lhs.leading_dim_stride, rhs.cast<Input>(), rhs.leading_dim_stride, beta,
      &output_data, output.leading_dim_stride, numeric_options);
}

}  // namespace

Status RunGemm(const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
               se::DeviceMemoryBase rhs_buffer,
               se::DeviceMemoryBase output_buffer, bool deterministic_ops,
               se::Stream* stream,
               std::optional<se::blas::AlgorithmType> algorithm,
               se::blas::ProfileResult* profile_result) {
  VLOG(2) << "Executing a GemmThunk";

  auto lhs_layout = MatrixLayout{config.lhs_layout},
       rhs_layout = MatrixLayout{config.rhs_layout},
       output_layout = MatrixLayout{config.output_layout};
  bool must_swap_operands =
      se::gpu::MakeOutputColumnMajor(lhs_layout, rhs_layout, output_layout);
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
  se::NumericOptions numeric_options{
      deterministic_ops,
      /*allow_tf32=*/config.compute_precision <= 1};

  if (!algorithm) algorithm = config.algorithm;

  std::tuple<PrimitiveType, PrimitiveType, PrimitiveType> operand_types{
      lhs_layout.dtype, rhs_layout.dtype, output_layout.dtype};

#define TYPED_GEMM(SCALENTYPE, ATYPE, BTYPE, CTYPE)                         \
  if (operand_types == std::make_tuple(ATYPE, BTYPE, CTYPE)) {              \
    using NativeScaleType =                                                 \
        primitive_util::PrimitiveTypeToNative<SCALENTYPE>::type;            \
    using NativeAType = primitive_util::PrimitiveTypeToNative<ATYPE>::type; \
    using NativeCType = primitive_util::PrimitiveTypeToNative<CTYPE>::type; \
    return DoGemm<NativeScaleType, NativeAType, NativeCType>(               \
        batch_size, m, n, k, lhs, rhs, output,                              \
        static_cast<NativeScaleType>(config.alpha.real()),                  \
        static_cast<NativeScaleType>(config.beta), stream, algorithm,       \
        config.compute_precision, numeric_options, profile_result);         \
  }

#define TYPED_GEMM_COMPLEX(SCALENTYPE, ATYPE, BTYPE, CTYPE)                 \
  if (operand_types == std::make_tuple(ATYPE, BTYPE, CTYPE)) {              \
    using NativeScaleType =                                                 \
        primitive_util::PrimitiveTypeToNative<SCALENTYPE>::type;            \
    using NativeAType = primitive_util::PrimitiveTypeToNative<ATYPE>::type; \
    using NativeCType = primitive_util::PrimitiveTypeToNative<CTYPE>::type; \
    return DoGemm<NativeScaleType, NativeAType, NativeCType>(               \
        batch_size, m, n, k, lhs, rhs, output,                              \
        static_cast<NativeScaleType>(config.alpha),                         \
        static_cast<NativeScaleType>(config.beta), stream, algorithm,       \
        config.compute_precision, numeric_options, profile_result);         \
  }

  if (output_layout.dtype == S32) {
    if (!algorithm) algorithm = se::blas::kDefaultGemmAlgo;
    return DoGemmWithAlgorithm<int32_t, int8_t, int32_t>(
        batch_size, m, n, k, lhs, rhs, output,
        static_cast<int32_t>(config.alpha.real()),
        static_cast<int32_t>(config.beta), stream, *algorithm,
        se::blas::kDefaultComputePrecision, numeric_options, profile_result);
  }

  TYPED_GEMM(F32, BF16, BF16, BF16)
  TYPED_GEMM(F32, F16, F16, F16)
  TYPED_GEMM(F32, S8, S8, F32)
  TYPED_GEMM(F32, BF16, BF16, F32)
  TYPED_GEMM(F32, F16, F16, F32)
  TYPED_GEMM(F32, F32, F32, F32)
  TYPED_GEMM(F64, F64, F64, F64)
  TYPED_GEMM_COMPLEX(C64, C64, C64, C64)
  TYPED_GEMM_COMPLEX(C128, C128, C128, C128)

#undef TYPED_GEMM
#undef TYPED_GEMM_COMPLEX
  return InternalError(
      "Unexpected GEMM dtype: %s %s %s",
      primitive_util::LowercasePrimitiveTypeName(lhs_layout.dtype),
      primitive_util::LowercasePrimitiveTypeName(rhs_layout.dtype),
      primitive_util::LowercasePrimitiveTypeName(output_layout.dtype));
}

namespace gpublas_lt {

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

StatusOr<se::gpu::BlasLt::Epilogue> AsBlasLtEpilogue(
    mlir::lmhlo_gpu::CublasLtMatmulEpilogue epilogue) {
  switch (epilogue) {
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::Default:
      return se::gpu::BlasLt::Epilogue::kDefault;
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::Relu:
      return se::gpu::BlasLt::Epilogue::kReLU;
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::Gelu:
      return se::gpu::BlasLt::Epilogue::kGELU;
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::GeluAux:
      return se::gpu::BlasLt::Epilogue::kGELUWithAux;
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::Bias:
      return se::gpu::BlasLt::Epilogue::kBias;
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::BiasRelu:
      return se::gpu::BlasLt::Epilogue::kBiasThenReLU;
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::BiasGelu:
      return se::gpu::BlasLt::Epilogue::kBiasThenGELU;
    case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::BiasGeluAux:
      return se::gpu::BlasLt::Epilogue::kBiasThenGELUWithAux;
  }
  return InternalError("unexpected epilogue value");
}

}  // namespace gpublas_lt

}  // namespace gpu
}  // namespace xla
