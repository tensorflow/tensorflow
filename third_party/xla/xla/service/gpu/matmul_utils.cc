/* Copyright 2022 The OpenXLA Authors.

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
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/matmul_indexing_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::StatusOr<Shape> GetBatchRowColumnShape(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> row_dims, absl::Span<const int64_t> col_dims) {
  TF_RET_CHECK(shape.has_layout());

  std::vector<int64_t> minor_to_major;
  for (size_t i = 0; i < shape.rank();) {
    // The GeMM output always has its layout set such that the batch, row, and
    // col dim groups are each laid out physically sequentially. GeMM operands
    // must, therefore, be laid out similarly.
    auto check_physically_sequential =
        [&](absl::Span<const int64_t> dims) -> absl::Status {
      for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
        // NOTE: `i` is incremented as we check the dimensions.
        if (*it != shape.layout().minor_to_major()[i++])
          return InvalidArgument("dims not physically_sequential");
      }
      return absl::OkStatus();
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
/*static*/ absl::StatusOr<MatrixLayout> MatrixLayout::For(const Shape& shape) {
  TF_RET_CHECK(shape.rank() == 3);
  TF_RET_CHECK(shape.has_layout());

  int64_t batch_size = shape.dimensions(0);
  int64_t num_rows = shape.dimensions(1);
  int64_t num_cols = shape.dimensions(2);

  Order order{Order::kRowMajor};
  int64_t leading_dim_stride = num_cols;
  int64_t batch_stride = num_rows * num_cols;

  // `MatrixLayout`, like BLAS, uses only two strides, so either the row or
  // column must be contiguous in memory (i.e. most minor physical dimension).
  absl::Span<const int64_t> minor_to_major = shape.layout().minor_to_major();
  switch (64 * minor_to_major[2] + 8 * minor_to_major[1] + minor_to_major[0]) {
    case 012:  // (B,R,C) (major-to-minor)
      break;
    case 021:  // (B,C,R)
      if (num_cols == 1) {
        // If rhs operand has no non-contracting dims, guarantee bias vector
        // length will still match matrix D rows with HIPBLASLT_EPILOGUE_BIAS
        // epilogue
        // (https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/datatypes.html).
        break;
      }
      order = Order::kColumnMajor;
      leading_dim_stride = num_rows;
      break;
    case 0102:  // (R,B,C)
      leading_dim_stride = batch_size * num_cols;
      batch_stride = num_cols;
      break;
    case 0201:  // (C,B,R)
      order = Order::kColumnMajor;
      leading_dim_stride = batch_size * num_rows;
      batch_stride = num_rows;
      break;
    default:
      return Unimplemented("batch in most minor dimension");
  }

  if (batch_size == 1) {
    batch_stride = 0;
  }
  return MatrixLayout{se::gpu::MatrixLayout{shape.element_type(), num_rows,
                                            num_cols, order, batch_size,
                                            leading_dim_stride, batch_stride}};
}

/*static*/ absl::StatusOr<MatrixLayout> MatrixLayout::For(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> row_dims, absl::Span<const int64_t> col_dims) {
  TF_ASSIGN_OR_RETURN(
      Shape batch_row_col_shape,
      GetBatchRowColumnShape(shape, batch_dims, row_dims, col_dims));
  return MatrixLayout::For(batch_row_col_shape);
}

/*static*/ absl::StatusOr<MatrixLayout> MatrixLayout::For(
    const Shape& shape, size_t lhs_num_batch_dims, size_t lhs_num_row_dims,
    size_t rhs_num_batch_dims, size_t rhs_num_col_dims) {
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

absl::StatusOr<bool> CanFoldTransposeOperandIntoDot(const HloInstruction& dot,
                                                    int64_t operand_idx) {
  if (Cast<HloDotInstruction>(&dot)->sparse_operands()) {
    return false;
  }
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

/*static*/ absl::StatusOr<GemmConfig> GemmConfig::For(
    const Shape& lhs_shape, absl::Span<const int64_t> lhs_batch_dims,
    absl::Span<const int64_t> lhs_contracting_dims, const Shape& rhs_shape,
    absl::Span<const int64_t> rhs_batch_dims,
    absl::Span<const int64_t> rhs_contracting_dims, const Shape& output_shape,
    double alpha_real, double alpha_imag, double beta,
    PrecisionConfig::Algorithm precision_algorithm,
    std::optional<int64_t> algorithm, int64_t compute_precision, bool grad_x,
    bool grad_y, const se::GpuComputeCapability& gpu_version) {
  return GemmConfig::For(lhs_shape, lhs_batch_dims, lhs_contracting_dims,
                         rhs_shape, rhs_batch_dims, rhs_contracting_dims,
                         /*c_shape=*/output_shape, /*bias_shape_ptr=*/nullptr,
                         output_shape, alpha_real, alpha_imag, beta,
                         precision_algorithm, algorithm, compute_precision,
                         grad_x, grad_y, gpu_version);
}

/*static*/ absl::StatusOr<GemmConfig> GemmConfig::For(
    const Shape& lhs_shape, absl::Span<const int64_t> lhs_batch_dims,
    absl::Span<const int64_t> lhs_contracting_dims, const Shape& rhs_shape,
    absl::Span<const int64_t> rhs_batch_dims,
    absl::Span<const int64_t> rhs_contracting_dims, const Shape& c_shape,
    const Shape* bias_shape_ptr, const Shape& output_shape, double alpha_real,
    double alpha_imag, double beta,
    PrecisionConfig::Algorithm precision_algorithm,
    std::optional<int64_t> algorithm, int64_t compute_precision, bool grad_x,
    bool grad_y, const se::GpuComputeCapability& gpu_version) {
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
  // hipBlasLt does not yet support the C matrix to be BF16 for fp8 matmul
  // with fp8 output. Thus only do this for CUDA side.
  if (std::holds_alternative<se::CudaComputeCapability>(gpu_version) &&
      primitive_util::IsF8Type(lhs_shape.element_type()) &&
      primitive_util::IsF8Type(output_shape.element_type()) && (beta == 0.0)) {
    // By default, if c is not present (i.e., beta is 0), c_shape will be the
    // output shape. cublasLT requires a valid c_shape to be passed, even if c
    // is not present, and normally setting it to the output shape is fine.
    // But for matmuls with FP8 inputs and outputs, C must instead have the
    // same dtype as the vector bias if present, and either BF16 or F16
    // otherwise. So we set the dtype of C here. hipBlasLt does not yet
    // support the C matrix to be BF16 for fp8 matmul with fp8 output. Thus
    // only do this for CUDA side.
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
    case F8E4M3FNUZ:
    case F8E5M2FNUZ:
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
        return Internal(
            "For int32 gemm output only int8 input is supported, got input: "
            "%s, %s",
            primitive_util::LowercasePrimitiveTypeName(lhs_layout.dtype),
            primitive_util::LowercasePrimitiveTypeName(rhs_layout.dtype));
      }
      break;
    default:
      return Internal("Unexpected GEMM datatype: %s",
                      primitive_util::LowercasePrimitiveTypeName(
                          output_shape.element_type()));
  }

  return GemmConfig{lhs_layout,
                    rhs_layout,
                    c_layout,
                    output_layout,
                    {alpha_real, alpha_imag},
                    beta,
                    compute_precision,
                    precision_algorithm,
                    algorithm,
                    grad_x,
                    grad_y};
}

namespace {

bool IsTf32Allowed(PrecisionConfig::Algorithm algorithm,
                   int64_t compute_precision) {
  if (algorithm == PrecisionConfig::ALG_UNSET) {
    return compute_precision <= 1;
  }

  return algorithm_util::HasTf32InputType(algorithm);
}

}  // namespace

/*static*/ absl::StatusOr<GemmConfig> GemmConfig::For(
    const HloInstruction* gemm, const se::GpuComputeCapability& gpu_version) {
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      gemm->backend_config<GpuBackendConfig>());
  return For(gemm, gpu_config.gemm_backend_config(), gpu_version);
}

/*static*/ absl::StatusOr<GemmConfig> GemmConfig::For(
    const HloInstruction* gemm, const GemmBackendConfig& config,
    const se::GpuComputeCapability& gpu_version) {
  std::optional<int64_t> algorithm;
  if (config.algorithm_case() != GemmBackendConfig::ALGORITHM_NOT_SET) {
    algorithm = config.selected_algorithm();
  } else {
    algorithm = se::blas::kDefaultAlgorithm;
  }

  const Shape& lhs_shape = gemm->operand(0)->shape();
  const Shape& rhs_shape = gemm->operand(1)->shape();
  const DotDimensionNumbers& dot_dims = config.dot_dimension_numbers();
  const Shape& output_shape =
      gemm->shape().IsTuple() ? gemm->shape().tuple_shapes(0) : gemm->shape();

  bool has_matrix_bias = config.beta() != 0.;
  Shape c_shape = has_matrix_bias ? gemm->operand(2)->shape() : output_shape;

  std::optional<Shape> vector_bias_shape;
  TF_ASSIGN_OR_RETURN(
      bool has_vector_bias,
      xla::gpu::gpublas_lt::EpilogueAddsVectorBias(config.epilogue()));
  if (has_vector_bias) {
    int vector_bias_index = has_matrix_bias ? 3 : 2;
    if (primitive_util::IsF8Type(lhs_shape.element_type())) {
      // FP8 gemms have 2 scales as inputs which come before the vector bias.
      vector_bias_index += 2;
    }
    vector_bias_shape = gemm->operand(vector_bias_index)->shape();
  }

  auto attributes = gemm->frontend_attributes().map();
  bool grad_x = (attributes["grad_x"] == "true");
  bool grad_y = (attributes["grad_y"] == "true");

  int64_t precision = se::blas::kDefaultComputePrecision;
  for (auto operand_precision : config.precision_config().operand_precision()) {
    precision = std::max(precision, static_cast<int64_t>(operand_precision));
  }
  const PrecisionConfig::Algorithm precision_algorithm =
      config.precision_config().algorithm();

  return GemmConfig::For(
      lhs_shape, dot_dims.lhs_batch_dimensions(),
      dot_dims.lhs_contracting_dimensions(), rhs_shape,
      dot_dims.rhs_batch_dimensions(), dot_dims.rhs_contracting_dimensions(),
      /*c_shape=*/c_shape,
      /*bias_shape_ptr=*/
      vector_bias_shape ? &vector_bias_shape.value() : nullptr, output_shape,
      config.alpha_real(), config.alpha_imag(), config.beta(),
      precision_algorithm, algorithm, precision, grad_x, grad_y, gpu_version);
}

absl::StatusOr<GemmConfig::DescriptorsTuple> GemmConfig::GetMatrixDescriptors(
    se::DeviceMemoryBase lhs_buf, se::DeviceMemoryBase rhs_buf,
    se::DeviceMemoryBase out_buf) const {
  auto create_matrix_desc = [](const se::gpu::MatrixLayout& layout,
                               se::DeviceMemoryBase data)
      -> absl::StatusOr<se::gpu::MatrixDescriptor> {
    TF_ASSIGN_OR_RETURN(se::blas::DataType type,
                        se::gpu::AsBlasDataType(layout.dtype));
    return se::gpu::MatrixDescriptor{
        data, layout.leading_dim_stride, layout.batch_stride, type,
        // BLAS is column-major by default.
        (layout.order == se::gpu::MatrixLayout::Order::kColumnMajor
             ? se::blas::Transpose::kNoTranspose
             : se::blas::Transpose::kTranspose)};
  };
  // TODO: make a local copy to prevent modification of layouts,
  // but maybe we can modify them once instead during creation ?
  se::gpu::MatrixLayout lhs = lhs_layout, rhs = rhs_layout, out = output_layout;

  bool must_swap_operands = MakeOutputColumnMajor(lhs, rhs, out);
  if (must_swap_operands) {
    std::swap(lhs_buf, rhs_buf);
  }

  TF_ASSIGN_OR_RETURN(se::gpu::OutputMatrixDescriptor out_desc,
                      create_matrix_desc(out, out_buf));
  out_desc.batch_size = out.batch_size;
  out_desc.m = out.num_rows;
  out_desc.n = out.num_cols;
  out_desc.k = lhs.num_cols;
  // TODO(tdanyluk): Investigate why don't we use the actual precision (and
  // algorithm) here? Why do we use the default?
  TF_ASSIGN_OR_RETURN(out_desc.compute_type,
                      se::gpu::GetBlasComputationType(
                          PrecisionConfig::ALG_UNSET, lhs.dtype, out.dtype,
                          se::blas::kDefaultComputePrecision));

  TF_ASSIGN_OR_RETURN(se::gpu::MatrixDescriptor lhs_desc,
                      create_matrix_desc(lhs, lhs_buf));
  TF_ASSIGN_OR_RETURN(se::gpu::MatrixDescriptor rhs_desc,
                      create_matrix_desc(rhs, rhs_buf));

  return DescriptorsTuple{lhs_desc, rhs_desc, out_desc, must_swap_operands};
}

namespace {

template <typename Scale, typename Input, typename Output>
absl::Status DoGemmWithAlgorithm(const se::gpu::MatrixDescriptor& lhs,
                                 const se::gpu::MatrixDescriptor& rhs,
                                 const se::gpu::OutputMatrixDescriptor& output,
                                 se::DeviceMemoryBase workspace, Scale alpha,
                                 Scale beta, se::Stream* stream,
                                 PrecisionConfig::Algorithm precision_algorithm,
                                 se::blas::AlgorithmType algorithm,
                                 se::blas::ComputePrecision compute_precision,
                                 const se::NumericOptions& numeric_options,
                                 se::blas::ProfileResult* profile_result,
                                 se::blas::CallContext context) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  PrimitiveType lhs_type = primitive_util::NativeToPrimitiveType<Input>();
  PrimitiveType output_type = primitive_util::NativeToPrimitiveType<Output>();
  TF_ASSIGN_OR_RETURN(
      se::blas::ComputationType computation_type,
      se::gpu::GetBlasComputationType(precision_algorithm, lhs_type,
                                      output_type, compute_precision));
  se::DeviceMemory<Output> output_data(output.data);

  // Set a workspace for all Blas operations launched below.
  auto* blas = stream->parent()->AsBlas();
  if (blas == nullptr) {
    return absl::InternalError("No Blas support for stream");
  }

  se::blas::BlasSupport::ScopedWorkspace scoped_workspace(blas, &workspace);

  if (output.batch_size != 1) {
    return blas->BlasGemmStridedBatchedWithAlgorithm(
        stream, lhs.transpose, rhs.transpose, output.m, output.n, output.k,
        alpha, lhs.cast<Input>(), lhs.leading_dim_stride, lhs.batch_stride,
        rhs.cast<Input>(), rhs.leading_dim_stride, rhs.batch_stride, beta,
        &output_data, output.leading_dim_stride, output.batch_stride,
        output.batch_size, computation_type, algorithm, numeric_options,
        profile_result, context);
  } else {
    return blas->BlasGemmWithAlgorithm(
        stream, lhs.transpose, rhs.transpose, output.m, output.n, output.k,
        alpha, lhs.cast<Input>(), lhs.leading_dim_stride, rhs.cast<Input>(),
        rhs.leading_dim_stride, beta, &output_data, output.leading_dim_stride,
        computation_type, algorithm, numeric_options, profile_result, context);
  }
}

template <typename Scale, typename Input, typename Output>
absl::Status DoGemm(const se::gpu::MatrixDescriptor& lhs,
                    const se::gpu::MatrixDescriptor& rhs,
                    const se::gpu::OutputMatrixDescriptor& output,
                    se::DeviceMemoryBase workspace, Scale alpha, Scale beta,
                    se::Stream* stream,
                    PrecisionConfig::Algorithm precision_algorithm,
                    std::optional<se::blas::AlgorithmType> algorithm,
                    se::blas::ComputePrecision compute_precision,
                    const se::NumericOptions& numeric_options,
                    se::blas::ProfileResult* profile_result,
                    se::blas::CallContext context) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  se::DeviceMemory<Output> output_data(output.data);
  auto* blas = stream->parent()->AsBlas();
  if (blas == nullptr) {
    return absl::InternalError("No Blas support for stream");
  }

  if (algorithm) {
    return DoGemmWithAlgorithm<Scale, Input, Output>(
        lhs, rhs, output, workspace, alpha, beta, stream, precision_algorithm,
        *algorithm, compute_precision, numeric_options, profile_result,
        context);
  }

  // Set a workspace for all Blas operations launched below.
  se::blas::BlasSupport::ScopedWorkspace scoped_workspace(blas, &workspace);

  if (output.batch_size != 1) {
    return blas->BlasGemmStridedBatched(
        stream, lhs.transpose, rhs.transpose, output.m, output.n, output.k,
        alpha, lhs.cast<Input>(), lhs.leading_dim_stride, lhs.batch_stride,
        rhs.cast<Input>(), rhs.leading_dim_stride, rhs.batch_stride, beta,
        &output_data, output.leading_dim_stride, output.batch_stride,
        output.batch_size, numeric_options, context);
  }

  return blas->BlasGemm(stream, lhs.transpose, rhs.transpose, output.m,
                        output.n, output.k, alpha, lhs.cast<Input>(),
                        lhs.leading_dim_stride, rhs.cast<Input>(),
                        rhs.leading_dim_stride, beta, &output_data,
                        output.leading_dim_stride, numeric_options, context);
}

}  // namespace

absl::Status RunGemm(const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
                     se::DeviceMemoryBase rhs_buffer,
                     se::DeviceMemoryBase output_buffer,
                     se::DeviceMemoryBase workspace_buffer,
                     bool deterministic_ops, se::Stream* stream,
                     std::optional<se::blas::AlgorithmType> algorithm,
                     se::blas::ProfileResult* profile_result) {
  VLOG(2) << "Executing a GemmThunk";

  TF_ASSIGN_OR_RETURN(
      GemmConfig::DescriptorsTuple desc,
      config.GetMatrixDescriptors(lhs_buffer, rhs_buffer, output_buffer));

  se::NumericOptions numeric_options{
      deterministic_ops,
      /*allow_tf32=*/IsTf32Allowed(config.precision_algorithm,
                                   config.compute_precision)};

  if (!algorithm) algorithm = config.algorithm;

  se::blas::CallContext context = se::blas::CallContext::kNone;
  if (config.grad_x) {
    context = desc.operands_swapped ? se::blas::CallContext::kBackpropInput2
                                    : se::blas::CallContext::kBackpropInput1;
  }
  if (config.grad_y) {
    context = desc.operands_swapped ? se::blas::CallContext::kBackpropInput1
                                    : se::blas::CallContext::kBackpropInput2;
  }

  std::tuple operand_types{config.lhs_layout.dtype, config.rhs_layout.dtype,
                           config.output_layout.dtype};

  // Skip degenerate gemm with memzero. In general this is not safe, because it
  // will suppress NaN propagation, however cuBLAS internally has exactly the
  // same optimization for compatibility with NETLIB implementation, so we are
  // not making things worse (and cuBLAS optimization is incompatible with CUDA
  // graphs, so we are making sure we do not trigger it).
  if (config.alpha.real() == 0.0 && config.alpha.imag() == 0.0 &&
      config.beta == 0.0) {
    return stream->MemZero(&output_buffer, output_buffer.size());
  }

#define TYPED_GEMM(SCALENTYPE, ATYPE, BTYPE, CTYPE)                         \
  if (operand_types == std::make_tuple(ATYPE, BTYPE, CTYPE)) {              \
    using NativeScaleType =                                                 \
        primitive_util::PrimitiveTypeToNative<SCALENTYPE>::type;            \
    using NativeAType = primitive_util::PrimitiveTypeToNative<ATYPE>::type; \
    using NativeCType = primitive_util::PrimitiveTypeToNative<CTYPE>::type; \
    return DoGemm<NativeScaleType, NativeAType, NativeCType>(               \
        desc.lhs, desc.rhs, desc.output, workspace_buffer,                  \
        static_cast<NativeScaleType>(config.alpha.real()),                  \
        static_cast<NativeScaleType>(config.beta), stream,                  \
        config.precision_algorithm, algorithm, config.compute_precision,    \
        numeric_options, profile_result, context);                          \
  }

#define TYPED_GEMM_COMPLEX(SCALENTYPE, ATYPE, BTYPE, CTYPE)                 \
  if (operand_types == std::make_tuple(ATYPE, BTYPE, CTYPE)) {              \
    using NativeScaleType =                                                 \
        primitive_util::PrimitiveTypeToNative<SCALENTYPE>::type;            \
    using NativeAType = primitive_util::PrimitiveTypeToNative<ATYPE>::type; \
    using NativeCType = primitive_util::PrimitiveTypeToNative<CTYPE>::type; \
    return DoGemm<NativeScaleType, NativeAType, NativeCType>(               \
        desc.lhs, desc.rhs, desc.output, workspace_buffer,                  \
        static_cast<NativeScaleType>(config.alpha),                         \
        static_cast<NativeScaleType>(config.beta), stream,                  \
        config.precision_algorithm, algorithm, config.compute_precision,    \
        numeric_options, profile_result, context);                          \
  }

  if (config.output_layout.dtype == S32) {
    if (!algorithm) algorithm = se::blas::kDefaultGemmAlgo;
    // TODO(tdanyluk): Investigate why don't we use the actual precision (and
    // algorithm) here? Why do we use the default?
    return DoGemmWithAlgorithm<int32_t, int8_t, int32_t>(
        desc.lhs, desc.rhs, desc.output, workspace_buffer,
        static_cast<int32_t>(config.alpha.real()),
        static_cast<int32_t>(config.beta), stream, PrecisionConfig::ALG_UNSET,
        *algorithm, se::blas::kDefaultComputePrecision, numeric_options,
        profile_result, context);
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
  return Internal(
      "Unexpected GEMM dtype: %s %s %s",
      primitive_util::LowercasePrimitiveTypeName(config.lhs_layout.dtype),
      primitive_util::LowercasePrimitiveTypeName(config.rhs_layout.dtype),
      primitive_util::LowercasePrimitiveTypeName(config.output_layout.dtype));
}  // namespace gpu

namespace gpublas_lt {

absl::StatusOr<bool> EpilogueAddsVectorBias(
    GemmBackendConfig_Epilogue epilogue) {
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
      return Internal("Unknown Epilogue.");
  }
}

absl::StatusOr<bool> EpilogueHasAuxiliaryOutput(
    GemmBackendConfig_Epilogue epilogue) {
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
      return Internal("Unknown Epilogue.");
  }
}

absl::StatusOr<se::gpu::BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig_Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
      return se::gpu::BlasLt::Epilogue::kDefault;
    case GemmBackendConfig::RELU:
      return se::gpu::BlasLt::Epilogue::kReLU;
    case GemmBackendConfig::GELU:
      return se::gpu::BlasLt::Epilogue::kGELU;
    case GemmBackendConfig::GELU_AUX:
      return se::gpu::BlasLt::Epilogue::kGELUWithAux;
    case GemmBackendConfig::BIAS:
      return se::gpu::BlasLt::Epilogue::kBias;
    case GemmBackendConfig::BIAS_RELU:
      return se::gpu::BlasLt::Epilogue::kBiasThenReLU;
    case GemmBackendConfig::BIAS_GELU:
      return se::gpu::BlasLt::Epilogue::kBiasThenGELU;
    case GemmBackendConfig::BIAS_GELU_AUX:
      return se::gpu::BlasLt::Epilogue::kBiasThenGELUWithAux;
    default:
      return Internal("unexpected epilogue value");
  }
}

}  // namespace gpublas_lt

/*static*/ absl::StatusOr<TritonGemmConfig> TritonGemmConfig::FromProto(
    const AutotuneResult::TritonGemmKey& proto) {
  // Sanity check to avoid loading incomplete data.
  TF_RET_CHECK(proto.block_m() > 0);
  TF_RET_CHECK(proto.block_n() > 0);
  TF_RET_CHECK(proto.block_k() > 0);
  TF_RET_CHECK(proto.split_k() > 0);
  TF_RET_CHECK(proto.num_stages() > 0);
  TF_RET_CHECK(proto.num_warps() > 0);
  TF_RET_CHECK(proto.num_ctas() > 0);

  return TritonGemmConfig(proto.block_m(), proto.block_n(), proto.block_k(),
                          proto.split_k(), proto.num_stages(),
                          proto.num_warps(), proto.num_ctas());
}

AutotuneResult::TritonGemmKey TritonGemmConfig::ToProto() const {
  AutotuneResult::TritonGemmKey key;
  key.set_block_m(block_m);
  key.set_block_n(block_n);
  key.set_block_k(block_k);
  key.set_split_k(split_k);
  key.set_num_stages(num_stages);
  key.set_num_warps(num_warps);
  key.set_num_ctas(num_ctas);
  return key;
}

std::string TritonGemmConfig::ToString() const {
  return absl::StrCat("{block_m:", block_m, ",block_n:", block_n,
                      ",block_k:", block_k, ",split_k:", split_k,
                      ",num_stages:", num_stages, ",num_warps:", num_warps,
                      ",num_ctas:", num_ctas, "}");
}

absl::StatusOr<bool> IsMatrixMultiplicationTooSmallForRewriting(
    const HloInstruction& dot, int64_t threshold) {
  CHECK_EQ(dot.opcode(), HloOpcode::kDot);

  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  const DotDimensionNumbers& dot_dims = dot.dot_dimension_numbers();

  int64_t contracting_size = 1;
  for (int64_t dim : dot_dims.lhs_contracting_dimensions()) {
    contracting_size *= lhs_shape.dimensions(dim);
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> lhs_non_contracting_dims,
      GetNonContractingDims(lhs_shape, dot_dims.lhs_batch_dimensions(),
                            dot_dims.lhs_contracting_dimensions()));
  int64_t lhs_non_contracting_size = 1;
  for (int64_t dim : lhs_non_contracting_dims) {
    lhs_non_contracting_size *= lhs_shape.dimensions(dim);
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> rhs_non_contracting_dims,
      GetNonContractingDims(rhs_shape, dot_dims.rhs_batch_dimensions(),
                            dot_dims.rhs_contracting_dimensions()));
  int64_t rhs_non_contracting_size = 1;
  for (int64_t dim : rhs_non_contracting_dims) {
    rhs_non_contracting_size *= rhs_shape.dimensions(dim);
  }

  return (rhs_non_contracting_size + lhs_non_contracting_size) *
             contracting_size <
         threshold;
}

bool IsDotSupportedByClassicalEmitters(const HloInstruction& dot) {
  if (!algorithm_util::IsSupportedByElementalIrEmitter(
          dot.precision_config().algorithm())) {
    return false;
  }

  // Let us be conservative and only throw float dots at the emitters.
  switch (dot.shape().element_type()) {
    case F16:
    case F32:
    case BF16:
      return true;
    default:
      return false;
  }
}

}  // namespace gpu
}  // namespace xla
