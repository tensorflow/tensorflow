/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/sycl/sycl_matmul_utils.h"

#include <cmath>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace sycl {
namespace sycl_gemm {

constexpr absl::string_view kValidEpilogueNames =
    "DEFAULT, RELU, GELU, BIAS, BIAS_RELU, BIAS_GELU";

absl::StatusOr<GemmBackendEpilogue> EpilogueCast(absl::string_view epilogue) {
  if (epilogue == "DEFAULT") {
    return GemmBackendEpilogue::DEFAULT;
  } else if (epilogue == "RELU") {
    return GemmBackendEpilogue::RELU;
  } else if (epilogue == "GELU") {
    return GemmBackendEpilogue::GELU;
  } else if (epilogue == "BIAS") {
    return GemmBackendEpilogue::BIAS;
  } else if (epilogue == "BIAS_RELU") {
    return GemmBackendEpilogue::BIAS_RELU;
  } else if (epilogue == "BIAS_GELU") {
    return GemmBackendEpilogue::BIAS_GELU;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown epilogue name \"", epilogue,
                     "\"; expected one of {", kValidEpilogueNames, "}"));
  }
}

absl::StatusOr<std::string> EpilogueCast(GemmBackendEpilogue epilogue) {
  switch (epilogue) {
    case GemmBackendEpilogue::DEFAULT:
      return "DEFAULT";
    case GemmBackendEpilogue::RELU:
      return "RELU";
    case GemmBackendEpilogue::GELU:
      return "GELU";
    case GemmBackendEpilogue::BIAS:
      return "BIAS";
    case GemmBackendEpilogue::BIAS_RELU:
      return "BIAS_RELU";
    case GemmBackendEpilogue::BIAS_GELU:
      return "BIAS_GELU";
    default:
      return absl::InternalError(absl::StrCat(
          "Unknown GemmBackendEpilogue value ", static_cast<int>(epilogue),
          "; expected one of {", kValidEpilogueNames, "}"));
  }
}

absl::StatusOr<bool> EpilogueAddsVectorBias(GemmBackendEpilogue epilogue) {
  switch (epilogue) {
    case GemmBackendEpilogue::DEFAULT:
    case GemmBackendEpilogue::RELU:
    case GemmBackendEpilogue::GELU:
      return false;
    case GemmBackendEpilogue::BIAS:
    case GemmBackendEpilogue::BIAS_RELU:
    case GemmBackendEpilogue::BIAS_GELU:
      return true;
    default:
      return absl::InternalError(absl::StrCat(
          "Unknown GemmBackendEpilogue value ", static_cast<int>(epilogue),
          "; expected one of {", kValidEpilogueNames, "}"));
  }
}

absl::StatusOr<bool> EpilogueHasAuxiliaryOutput(GemmBackendEpilogue epilogue) {
  switch (epilogue) {
    case GemmBackendEpilogue::DEFAULT:
    case GemmBackendEpilogue::RELU:
    case GemmBackendEpilogue::GELU:
    case GemmBackendEpilogue::BIAS:
    case GemmBackendEpilogue::BIAS_RELU:
    case GemmBackendEpilogue::BIAS_GELU:
      return false;
    default:
      return absl::InternalError(absl::StrCat(
          "Unknown GemmBackendEpilogue value ", static_cast<int>(epilogue),
          "; expected one of {", kValidEpilogueNames, "}"));
  }
}

absl::StatusOr<GemmBackendEpilogue> AsSYCLEpilogue(
    xla::gpu::GemmBackendConfig_Epilogue epilogue) {
  switch (epilogue) {
    case xla::gpu::GemmBackendConfig::DEFAULT:
      return GemmBackendEpilogue::DEFAULT;
    case xla::gpu::GemmBackendConfig::RELU:
      return GemmBackendEpilogue::RELU;
    case xla::gpu::GemmBackendConfig::GELU:
      return GemmBackendEpilogue::GELU;
    case xla::gpu::GemmBackendConfig::BIAS:
      return GemmBackendEpilogue::BIAS;
    case xla::gpu::GemmBackendConfig::BIAS_RELU:
      return GemmBackendEpilogue::BIAS_RELU;
    case xla::gpu::GemmBackendConfig::BIAS_GELU:
      return GemmBackendEpilogue::BIAS_GELU;
    default:
      return absl::InternalError(
          absl::StrCat("Unsupported epilogue \"",
                       xla::gpu::GemmBackendConfig::Epilogue_Name(epilogue),
                       "\" (value ", static_cast<int>(epilogue),
                       "); SYCL backend supports {", kValidEpilogueNames, "}"));
  }
}

absl::StatusOr<GemmBackendEpilogue> AsSYCLEpilogue(
    se::gpu::BlasLt::Epilogue epilogue) {
  switch (epilogue) {
    case se::gpu::BlasLt::Epilogue::kDefault:
      return GemmBackendEpilogue::DEFAULT;
    case se::gpu::BlasLt::Epilogue::kReLU:
      return GemmBackendEpilogue::RELU;
    case se::gpu::BlasLt::Epilogue::kGELU:
      return GemmBackendEpilogue::GELU;
    case se::gpu::BlasLt::Epilogue::kBias:
      return GemmBackendEpilogue::BIAS;
    case se::gpu::BlasLt::Epilogue::kBiasThenReLU:
      return GemmBackendEpilogue::BIAS_RELU;
    case se::gpu::BlasLt::Epilogue::kBiasThenGELU:
      return GemmBackendEpilogue::BIAS_GELU;
    default:
      return absl::InternalError(absl::StrCat(
          "Unsupported BlasLt::Epilogue value ", static_cast<int>(epilogue),
          "; SYCL backend supports {kDefault, kReLU, kGELU, kBias, "
          "kBiasThenReLU, kBiasThenGELU}"));
  }
}

}  // namespace sycl_gemm

// Returns the native type (eg, float) corresponding to the given
// template parameter XLA primitive type (eg, F32).
template <xla::PrimitiveType>
struct PrimitiveTypeToNative;

template <>
struct PrimitiveTypeToNative<xla::F32> {
  using type = float;
};
template <>
struct PrimitiveTypeToNative<xla::F16> {
  using type = ::sycl::half;
};
template <>
struct PrimitiveTypeToNative<xla::BF16> {
  using type = ::sycl::ext::oneapi::bfloat16;
};
template <>
struct PrimitiveTypeToNative<xla::S8> {
  using type = int8_t;
};
template <>
struct PrimitiveTypeToNative<xla::S32> {
  using type = int32_t;
};

/// Variable template for oneDNN data type mapping
/// @tparam T C++ type to map to dnnl::memory::data_type
template <typename T>
inline constexpr dnnl::memory::data_type OneDnnType =
    dnnl::memory::data_type::undef;

template <>
inline constexpr dnnl::memory::data_type OneDnnType<float> =
    dnnl::memory::data_type::f32;

template <>
inline constexpr dnnl::memory::data_type OneDnnType<double> =
    dnnl::memory::data_type::f64;

template <>
inline constexpr dnnl::memory::data_type OneDnnType<::sycl::half> =
    dnnl::memory::data_type::f16;

template <>
inline constexpr dnnl::memory::data_type OneDnnType<int8_t> =
    dnnl::memory::data_type::s8;

template <>
inline constexpr dnnl::memory::data_type OneDnnType<int32_t> =
    dnnl::memory::data_type::s32;

template <>
inline constexpr dnnl::memory::data_type
    OneDnnType<::sycl::ext::oneapi::bfloat16> = dnnl::memory::data_type::bf16;

MatrixDescriptor GetMatrixDesc(const se::gpu::MatrixLayout& layout,
                               se::DeviceMemoryBase data) {
  bool transpose = layout.order == se::gpu::MatrixLayout::Order::kColumnMajor;
  return MatrixDescriptor{
      data,
      transpose ? se::blas::Transpose::kTranspose
                : se::blas::Transpose::kNoTranspose,
      transpose ? layout.num_cols : layout.num_rows,
      transpose ? layout.num_rows : layout.num_cols,
      layout.batch_stride,
      layout.leading_dim_stride,
  };
}

struct OneDnnMatMulParams {
  dnnl::memory::dims a_dims;
  dnnl::memory::dims b_dims;
  dnnl::memory::dims c_dims;
  dnnl::memory::dims bias_dims;
  dnnl::memory::dims a_strides;
  dnnl::memory::dims b_strides;
  dnnl::memory::dims c_strides;
  dnnl::memory::dims bias_strides;

  OneDnnMatMulParams(dnnl::memory::dims a_dims, dnnl::memory::dims b_dims,
                     dnnl::memory::dims c_dims, dnnl::memory::dims bias_dims,
                     dnnl::memory::dims a_strides, dnnl::memory::dims b_strides,
                     dnnl::memory::dims c_strides,
                     dnnl::memory::dims bias_strides)
      : a_dims(std::move(a_dims)),
        b_dims(std::move(b_dims)),
        c_dims(std::move(c_dims)),
        bias_dims(std::move(bias_dims)),
        a_strides(std::move(a_strides)),
        b_strides(std::move(b_strides)),
        c_strides(std::move(c_strides)),
        bias_strides(std::move(bias_strides)) {}
};

std::unique_ptr<OneDnnMatMulParams> CreateMatMulParams(
    int64_t batch_size, const MatrixDescriptor& lhs,
    const MatrixDescriptor& rhs, const MatrixDescriptor& out) {
  dnnl::memory::dims lhs_dims{batch_size, lhs.num_rows, lhs.num_cols};
  dnnl::memory::dims rhs_dims{batch_size, rhs.num_rows, rhs.num_cols};
  dnnl::memory::dims out_dims{batch_size, out.num_rows, out.num_cols};

  auto lhs_strides =
      dnnl::memory::dims{lhs.batch_stride, lhs.leading_dim_stride, 1};
  auto rhs_strides =
      dnnl::memory::dims{rhs.batch_stride, rhs.leading_dim_stride, 1};
  auto out_strides =
      dnnl::memory::dims{out.batch_stride, out.leading_dim_stride, 1};
  // Positions of the cols (innermost) and rows dims within the
  // {batch, rows, cols} tuples built above. Derived from the tuple rank so
  // the transpose swaps below read as "exchange the last two dims".
  const int idx_last = static_cast<int>(lhs_dims.size()) - 1;
  const int idx_2nd_last = idx_last - 1;

  // dst(m,n) = \sigma{src(m,k) * weights(k, n)}
  // lhs_strides holds the strides for each dim, say {24, 12, 4, 1} for
  // src_tensor {1, 2, 3, 4} if adj_x_ is false.
  // If adj_x_ is true, swap the innermost two dims of lhs_strides
  // to {24, 12, 1, 4}, just like set memory::format_tag::abdc
  if (lhs.transpose == se::blas::Transpose::kTranspose) {
    std::swap(lhs_dims[idx_last], lhs_dims[idx_2nd_last]);
    std::swap(lhs_strides[idx_last], lhs_strides[idx_2nd_last]);
  }
  if (rhs.transpose == se::blas::Transpose::kTranspose) {
    std::swap(rhs_dims[idx_last], rhs_dims[idx_2nd_last]);
    std::swap(rhs_strides[idx_last], rhs_strides[idx_2nd_last]);
  }

  // Build a per-column bias shape {1, ..., 1, N} matching the matmul output
  // rank. The trailing N lets oneDNN apply one bias value per output column
  // and broadcast across the batch and row dimensions; leaving it at 1 would
  // collapse the bias to a single scalar applied uniformly, which is incorrect
  // for the BIAS / BIAS_RELU / BIAS_GELU epilogues.
  dnnl::memory::dims bias_dims(rhs_dims.size(), 1);
  bias_dims[rhs_dims.size() - 1] = rhs_dims[rhs_dims.size() - 1];
  auto bias_strides = ComputeRowMajorStrides(bias_dims);

  return std::make_unique<OneDnnMatMulParams>(
      lhs_dims, rhs_dims, out_dims, bias_dims, lhs_strides, rhs_strides,
      out_strides, bias_strides);
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> GetDimsStrides(
    const xla::Shape& shape) {
  // oneDNN handles scalar as a vector of size 1.
  const bool is_scalar = shape.dimensions_size() == 0;
  int64_t rank = is_scalar ? 1 : shape.dimensions_size();
  std::vector<int64_t> strides(rank);
  std::vector<int64_t> scalar_shape(1, 1);
  absl::Span<const int64_t> dimensions =
      is_scalar ? scalar_shape : shape.dimensions();
  std::vector<int64_t> dims(dimensions.begin(), dimensions.end());
  if (is_scalar) {
    strides[0] = 1;
  } else {
    int64_t stride = 1;
    for (int i : shape.layout().minor_to_major()) {
      strides.at(i) = stride;
      stride *= dims.at(i);
    }
  }
  return std::make_pair(dims, strides);
}

absl::StatusOr<dnnl::memory::desc> ShapeToMemDesc(const xla::Shape& shape) {
  auto [dims, strides] = GetDimsStrides(shape);
  if (dims.empty()) {
    return dnnl::memory::desc{};
  }
  dnnl::memory::data_type dtype;
  switch (shape.element_type()) {
    case xla::PrimitiveType::F16:
      dtype = dnnl::memory::data_type::f16;
      break;
    case xla::PrimitiveType::BF16:
      dtype = dnnl::memory::data_type::bf16;
      break;
    case xla::PrimitiveType::F32:
      dtype = dnnl::memory::data_type::f32;
      break;
    case xla::PrimitiveType::S8:
      dtype = dnnl::memory::data_type::s8;
      break;
    case xla::PrimitiveType::U8:
      dtype = dnnl::memory::data_type::u8;
      break;
    case xla::PrimitiveType::S32:
      dtype = dnnl::memory::data_type::s32;
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported element type: %s",
                          xla::primitive_util::LowercasePrimitiveTypeName(
                              shape.element_type())));
  }
  return dnnl::memory::desc(dims, dtype, strides);
}

void TransposeMatrixDesc(MatrixDescriptor& matrix_desc) {
  matrix_desc.transpose =
      (matrix_desc.transpose == se::blas::Transpose::kNoTranspose)
          ? se::blas::Transpose::kTranspose
          : se::blas::Transpose::kNoTranspose;
}

void MakeBlasGemmCompatible(MatrixDescriptor& lhs, MatrixDescriptor& rhs,
                            MatrixDescriptor& output) {
  // BLAS GeMM doesn't support transposed output, but we can use the identity:
  // C^T = (A @ B)^T = B^T @ A^T.
  if (output.transpose == se::blas::Transpose::kTranspose) {
    std::swap(lhs, rhs);
    TransposeMatrixDesc(lhs);
    TransposeMatrixDesc(rhs);
    TransposeMatrixDesc(output);
  }
}

absl::StatusOr<std::unique_ptr<dnnl::matmul::primitive_desc>>
CreateMatMulPrimDescFromGemmConfig(
    const xla::gpu::GemmConfig& config,
    xla::gpu::GemmBackendConfig_Epilogue epilogue) {
  // Create default GPU engine for OneDNN operations
  dnnl::engine dnnl_engine(dnnl::engine::kind::gpu, 0);

  // Extract matrix layouts from GemmConfig
  auto lhs_layout = se::gpu::MatrixLayout{config.lhs_layout};
  auto rhs_layout = se::gpu::MatrixLayout{config.rhs_layout};
  auto output_layout = se::gpu::MatrixLayout{config.output_layout};
  int64_t batch_size = output_layout.batch_size;

  // Create matrix descriptors (without memory buffers)
  MatrixDescriptor lhs = GetMatrixDesc(lhs_layout, se::DeviceMemoryBase());
  MatrixDescriptor rhs = GetMatrixDesc(rhs_layout, se::DeviceMemoryBase());
  MatrixDescriptor output =
      GetMatrixDesc(output_layout, se::DeviceMemoryBase());

  // Make BLAS GEMM compatible (handles output transpose)
  MakeBlasGemmCompatible(lhs, rhs, output);

  // Create OneDNN memory descriptors using dimensions and strides
  dnnl::memory::dims lhs_dims{batch_size, lhs.num_rows, lhs.num_cols};
  dnnl::memory::dims rhs_dims{batch_size, rhs.num_rows, rhs.num_cols};
  dnnl::memory::dims out_dims{batch_size, output.num_rows, output.num_cols};

  auto lhs_strides =
      dnnl::memory::dims{lhs.batch_stride, lhs.leading_dim_stride, 1};
  auto rhs_strides =
      dnnl::memory::dims{rhs.batch_stride, rhs.leading_dim_stride, 1};
  auto out_strides =
      dnnl::memory::dims{output.batch_stride, output.leading_dim_stride, 1};

  // Positions of the cols (innermost) and rows dims within the
  // {batch, rows, cols} tuples built above. Derived from the tuple rank so
  // the transpose swaps below read as "exchange the last two dims".
  const int idx_last = static_cast<int>(lhs_dims.size()) - 1;
  const int idx_2nd_last = idx_last - 1;

  // Handle transpose by swapping dimensions and strides
  if (lhs.transpose == se::blas::Transpose::kTranspose) {
    std::swap(lhs_dims[idx_last], lhs_dims[idx_2nd_last]);
    std::swap(lhs_strides[idx_last], lhs_strides[idx_2nd_last]);
  }
  if (rhs.transpose == se::blas::Transpose::kTranspose) {
    std::swap(rhs_dims[idx_last], rhs_dims[idx_2nd_last]);
    std::swap(rhs_strides[idx_last], rhs_strides[idx_2nd_last]);
  }

  // Get OneDNN data type from layout
  dnnl::memory::data_type lhs_dtype, rhs_dtype, output_dtype;
  switch (lhs_layout.dtype) {
    case xla::PrimitiveType::F16:
      lhs_dtype = dnnl::memory::data_type::f16;
      break;
    case xla::PrimitiveType::BF16:
      lhs_dtype = dnnl::memory::data_type::bf16;
      break;
    case xla::PrimitiveType::F32:
      lhs_dtype = dnnl::memory::data_type::f32;
      break;
    case xla::PrimitiveType::S8:
      lhs_dtype = dnnl::memory::data_type::s8;
      break;
    case xla::PrimitiveType::S32:
      lhs_dtype = dnnl::memory::data_type::s32;
      break;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported LHS element type: %s",
          xla::primitive_util::LowercasePrimitiveTypeName(lhs_layout.dtype)));
  }

  switch (rhs_layout.dtype) {
    case xla::PrimitiveType::F16:
      rhs_dtype = dnnl::memory::data_type::f16;
      break;
    case xla::PrimitiveType::BF16:
      rhs_dtype = dnnl::memory::data_type::bf16;
      break;
    case xla::PrimitiveType::F32:
      rhs_dtype = dnnl::memory::data_type::f32;
      break;
    case xla::PrimitiveType::S8:
      rhs_dtype = dnnl::memory::data_type::s8;
      break;
    case xla::PrimitiveType::S32:
      rhs_dtype = dnnl::memory::data_type::s32;
      break;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported RHS element type: %s",
          xla::primitive_util::LowercasePrimitiveTypeName(rhs_layout.dtype)));
  }

  switch (output_layout.dtype) {
    case xla::PrimitiveType::F16:
      output_dtype = dnnl::memory::data_type::f16;
      break;
    case xla::PrimitiveType::BF16:
      output_dtype = dnnl::memory::data_type::bf16;
      break;
    case xla::PrimitiveType::F32:
      output_dtype = dnnl::memory::data_type::f32;
      break;
    case xla::PrimitiveType::S32:
      output_dtype = dnnl::memory::data_type::s32;
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported output element type: %s",
                          xla::primitive_util::LowercasePrimitiveTypeName(
                              output_layout.dtype)));
  }

  auto lhs_md = dnnl::memory::desc(lhs_dims, lhs_dtype, lhs_strides);
  auto rhs_md = dnnl::memory::desc(rhs_dims, rhs_dtype, rhs_strides);
  auto output_md = dnnl::memory::desc(out_dims, output_dtype, out_strides);

  // Bias is optional - create empty descriptor for now
  auto bias_md = dnnl::memory::desc();

  // Set up post-ops based on epilogue
  dnnl::post_ops post_ops;
  ABSL_ASSIGN_OR_RETURN(sycl_gemm::GemmBackendEpilogue sycl_epilogue,
                   sycl_gemm::AsSYCLEpilogue(epilogue));

  switch (sycl_epilogue) {
    case sycl_gemm::GemmBackendEpilogue::RELU:
      post_ops.append_eltwise(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
      break;
    case sycl_gemm::GemmBackendEpilogue::GELU:
      post_ops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.0f, 0.0f);
      break;
    case sycl_gemm::GemmBackendEpilogue::BIAS_RELU:
      post_ops.append_eltwise(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
      break;
    case sycl_gemm::GemmBackendEpilogue::BIAS_GELU:
      post_ops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.0f, 0.0f);
      break;
    default:
      // No post-ops for DEFAULT and BIAS epilogues
      break;
  }

  dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  attr.set_post_ops(post_ops);

  return std::make_unique<dnnl::matmul::primitive_desc>(
      dnnl_engine, lhs_md, rhs_md, bias_md, output_md, attr);
}

template <typename InputT, typename OutputT>
absl::Status DoOnednnGemm(int64_t batch_size, const MatrixDescriptor& lhs,
                          const MatrixDescriptor& rhs, se::DeviceMemoryBase c,
                          const MatrixDescriptor& output,
                          se::DeviceMemoryBase bias, float alpha, float beta,
                          sycl_gemm::GemmBackendEpilogue epilogue,
                          se::Stream* stream, se::DeviceMemoryBase workspace,
                          se::ScratchAllocator* scratch_allocator) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  ::sycl::queue* stream_handle =
      absl::bit_cast<::sycl::queue*>(stream->platform_specific_handle().stream);
  void* lhs_data = lhs.data.opaque();
  void* rhs_data = rhs.data.opaque();
  void* c_data = c.opaque();
  void* out_data = output.data.opaque();
  void* bias_data = bias.opaque();
  auto params = CreateMatMulParams(batch_size, lhs, rhs, output);

  auto src_md =
      dnnl::memory::desc(params->a_dims, OneDnnType<InputT>, params->a_strides);
  auto weights_md =
      dnnl::memory::desc(params->b_dims, OneDnnType<InputT>, params->b_strides);
  auto dst_md = dnnl::memory::desc(params->c_dims, OneDnnType<OutputT>,
                                   params->c_strides);
  auto bias_md = bias_data
                     ? dnnl::memory::desc(params->bias_dims, OneDnnType<InputT>,
                                          params->bias_strides)
                     : dnnl::memory::desc();
  auto dnnl_engine = FindOrCreateEngine(stream_handle);
  dnnl::primitive_attr post_ops_attr;
  post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // Set fp32 mode.
  dnnl::fpmath_mode fp32_math_mode = GetFP32MathMode();
  if (std::is_same<InputT, float>::value) {
    post_ops_attr.set_fpmath_mode(fp32_math_mode);
  }
  dnnl::post_ops post_ops = dnnl::post_ops();
  // C = activation(alpha * MatMul(x, w, bias) + beta * C)
  // Post-ops are applied in order:
  //   1. Scale by alpha (if alpha != 1.0) using eltwise_linear
  //   2. Add beta * C (if beta != 0.0) using sum
  //   3. Apply activation (if any)

  // Step 1: Scale by alpha using eltwise_linear(alpha, 0.0) -> output = alpha *
  // input + 0.0
  if (std::fabs(alpha - 1.0f) >= 1e-6) {
    post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
  }

  // Step 2: Add beta * C (sum post-op reads initial destination buffer)
  if (c_data && std::fabs(beta - 0.0f) > 1e-6) {
    post_ops.append_sum(beta);
  }

  // Step 3: Apply activation
  switch (epilogue) {
    case sycl_gemm::GemmBackendEpilogue::RELU:
    case sycl_gemm::GemmBackendEpilogue::BIAS_RELU:
      post_ops.append_eltwise(dnnl::algorithm::eltwise_relu, 0, 0);
      break;
    case sycl_gemm::GemmBackendEpilogue::GELU:
    case sycl_gemm::GemmBackendEpilogue::BIAS_GELU:
      post_ops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0, 0);
      break;
    case sycl_gemm::GemmBackendEpilogue::DEFAULT:
    case sycl_gemm::GemmBackendEpilogue::BIAS:
      break;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported activation mode: ", static_cast<int>(epilogue)));
  }
  post_ops_attr.set_post_ops(post_ops);
  auto matmul_pd =
      bias_data
          ? std::make_shared<dnnl::matmul::primitive_desc>(
                dnnl_engine, src_md, weights_md, bias_md, dst_md, post_ops_attr)
          : std::make_shared<dnnl::matmul::primitive_desc>(
                dnnl_engine, src_md, weights_md, dst_md, post_ops_attr);
  std::unordered_map<int, dnnl::memory> fwd_primitive_args;
  size_t scratchpad_size = matmul_pd->scratchpad_desc().get_size();

  void* workspace_addr = workspace.opaque();
  if (scratchpad_size > 0) {
    if (scratch_allocator != nullptr) {
      ABSL_ASSIGN_OR_RETURN(stream_executor::DeviceMemory<uint8_t> alloc,
                       scratch_allocator->AllocateBytes(scratchpad_size));
      workspace_addr = alloc.opaque();
    } else {
      workspace_addr = workspace.opaque();
    }
  }

  auto scratchpad_mem = dnnl::sycl_interop::make_memory(
      matmul_pd->scratchpad_desc(), dnnl_engine,
      dnnl::sycl_interop::memory_kind::usm, workspace_addr);

  auto matmul_primitive = dnnl::matmul(*matmul_pd);
  auto dnnl_stream =
      dnnl::sycl_interop::make_stream(dnnl_engine, *stream_handle);
  auto src_mem = CreateDnnlMemory(src_md, dnnl_engine, lhs_data);
  auto wei_mem = CreateDnnlMemory(weights_md, dnnl_engine, rhs_data);
  auto dst_mem = CreateDnnlMemory(dst_md, dnnl_engine, out_data);
  fwd_primitive_args.emplace(DNNL_ARG_SRC, src_mem);
  fwd_primitive_args.emplace(DNNL_ARG_WEIGHTS, wei_mem);
  fwd_primitive_args.emplace(DNNL_ARG_DST, dst_mem);
  fwd_primitive_args.emplace(DNNL_ARG_SCRATCHPAD, scratchpad_mem);
  if (bias_data) {
    auto bias_mem = CreateDnnlMemory(bias_md, dnnl_engine, bias_data);
    fwd_primitive_args.emplace(DNNL_ARG_BIAS, bias_mem);
  }

  // OneDNN's post_ops.append_sum(beta) reads the initial contents of the
  // destination buffer and adds beta * dst_initial to the matmul result.
  //
  // If c_data and out_data point to the same buffer, we can skip the copy.
  // Otherwise, we must copy c_data to out_data before executing matmul.
  if (c_data && std::fabs(beta - 0.0f) > 1e-6) {
    bool buffers_aliased = (c_data == out_data);
    if (!buffers_aliased) {
      // Buffers are different - need to copy c_data into destination
      size_t output_size =
          batch_size * output.num_rows * output.num_cols * sizeof(OutputT);
      se::DeviceMemoryBase dst_mem(out_data, output_size);
      if (absl::Status status = stream->MemcpyD2D(&dst_mem, c, output_size);
          !status.ok()) {
        return status;
      }
    }
    // else: buffers already aliased, SUM post-op will read directly from
    // the destination buffer - true in-place operation with zero copy!
  }

  matmul_primitive.execute(dnnl_stream, fwd_primitive_args);
  return absl::OkStatus();
}

template <typename InputT, typename OutputT>
absl::Status DoGemm(int64_t batch_size, const MatrixDescriptor& lhs,
                    const MatrixDescriptor& rhs, se::DeviceMemoryBase c,
                    const MatrixDescriptor& output, se::DeviceMemoryBase bias,
                    se::DeviceMemoryBase workspace, float alpha, float beta,
                    sycl_gemm::GemmBackendEpilogue epilogue, se::Stream* stream,
                    se::blas::AlgorithmType algorithm,
                    se::ScratchAllocator* scratch_allocator) {
  if (algorithm == 1 /* OneDNN */) {
    return DoOnednnGemm<InputT, OutputT>(batch_size, lhs, rhs, c, output, bias,
                                         alpha, beta, epilogue, stream,
                                         workspace, scratch_allocator);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported GEMM algorithm: ", algorithm));
  }
}

absl::Status RunGemm(const gpu::GemmConfig& config,
                     se::DeviceMemoryBase lhs_buffer,
                     se::DeviceMemoryBase rhs_buffer,
                     se::DeviceMemoryBase c_buffer,
                     se::DeviceMemoryBase output_buffer,
                     se::DeviceMemoryBase bias_buffer,
                     se::DeviceMemoryBase workspace, se::Stream* stream,
                     sycl_gemm::GemmBackendEpilogue epilogue, int64_t algorithm,
                     se::ScratchAllocator* scratch_allocator) {
  auto lhs_layout = se::gpu::MatrixLayout{config.lhs_layout};
  auto rhs_layout = se::gpu::MatrixLayout{config.rhs_layout};
  auto output_layout = se::gpu::MatrixLayout{config.output_layout};

  MatrixDescriptor lhs = GetMatrixDesc(lhs_layout, lhs_buffer);
  MatrixDescriptor rhs = GetMatrixDesc(rhs_layout, rhs_buffer);
  MatrixDescriptor output = GetMatrixDesc(output_layout, output_buffer);
  int64_t batch_size = output_layout.batch_size;
  MakeBlasGemmCompatible(lhs, rhs, output);

  std::tuple operand_types{lhs_layout.dtype, rhs_layout.dtype,
                           output_layout.dtype};

#define TYPED_GEMM(ATYPE, BTYPE, CTYPE)                                 \
  if (operand_types == std::make_tuple(ATYPE, BTYPE, CTYPE)) {          \
    using NativeAType = PrimitiveTypeToNative<ATYPE>::type;             \
    using NativeCType = PrimitiveTypeToNative<CTYPE>::type;             \
    return DoGemm<NativeAType, NativeCType>(                            \
        batch_size, lhs, rhs, c_buffer, output, bias_buffer, workspace, \
        config.alpha.real(), config.beta, epilogue, stream, algorithm,  \
        scratch_allocator);                                             \
  }

  TYPED_GEMM(xla::F32, xla::F32, xla::F32)
  TYPED_GEMM(xla::BF16, xla::BF16, xla::BF16)
  TYPED_GEMM(xla::BF16, xla::BF16, xla::F32)
  TYPED_GEMM(xla::F16, xla::F16, xla::F16)
  TYPED_GEMM(xla::F16, xla::F16, xla::F32)
  TYPED_GEMM(xla::S8, xla::S8, xla::S32)
  // TODO (intel-tf): Add support for more combinations of input/output types

#undef TYPED_GEMM
  return absl::InternalError(absl::StrCat(
      "Unexpected GEMM lhs type ",
      xla::primitive_util::LowercasePrimitiveTypeName(lhs_layout.dtype),
      ", rhs type ",
      xla::primitive_util::LowercasePrimitiveTypeName(rhs_layout.dtype),
      " and output type ",
      xla::primitive_util::LowercasePrimitiveTypeName(output_layout.dtype)));
}
}  // namespace sycl
}  // namespace stream_executor
