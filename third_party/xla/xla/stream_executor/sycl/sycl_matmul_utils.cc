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
#include "xla/xla_data.pb.h"

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

MatrixDescriptor GetMatrixDesc(const se::gpu::MatrixLayout& layout,
                               se::DeviceAddressBase data) {
  bool transpose = layout.order == se::gpu::MatrixLayout::Order::kColumnMajor;
  return MatrixDescriptor{
      data,
      layout.dtype,
      transpose ? se::blas::Transpose::kTranspose
                : se::blas::Transpose::kNoTranspose,
      transpose ? layout.num_cols : layout.num_rows,
      transpose ? layout.num_rows : layout.num_cols,
      layout.batch_size,
      layout.batch_stride,
      layout.leading_dim_stride,
  };
}

// Maps a BLAS-style MatrixDescriptor onto a oneDNN memory desc. For a
// transposed operand, the last two dims and strides are swapped so the desc
// has the logical shape; oneDNN infers the transpose from the strides.
absl::StatusOr<dnnl::memory::desc> ToOneDnnMemDesc(const MatrixDescriptor& m) {
  ABSL_ASSIGN_OR_RETURN(dnnl::memory::data_type dtype,
                   sycl::ToOneDnnDataType(m.dtype));
  dnnl::memory::dims dims{m.batch_size, m.num_rows, m.num_cols};
  dnnl::memory::dims strides{m.batch_stride, m.leading_dim_stride, 1};
  if (m.transpose == se::blas::Transpose::kTranspose) {
    std::swap(dims[dims.size() - 1], dims[dims.size() - 2]);
    std::swap(strides[strides.size() - 1], strides[strides.size() - 2]);
  }
  return dnnl::memory::desc(dims, dtype, strides);
}

// Builds the bias memory desc for the matmul output. The bias has shape
// {1, ..., 1, N}, where N is the output's column dim, so oneDNN broadcasts
// one bias value per output column across the batch and row dims. The bias
// dtype matches the output dtype.
dnnl::memory::desc MakeBiasMemDesc(const dnnl::memory::desc& output_md) {
  const auto out_dims = output_md.get_dims();
  dnnl::memory::dims bias_dims(out_dims.size(), 1);
  bias_dims.back() = out_dims.back();
  return dnnl::memory::desc(bias_dims, output_md.get_data_type(),
                            ComputeRowMajorStrides(bias_dims));
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
  ABSL_ASSIGN_OR_RETURN(dnnl::memory::data_type dtype,
                   sycl::ToOneDnnDataType(shape.element_type()));
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

namespace {

// Element types of the lhs, rhs and output that the oneDNN matmul supports.
struct GemmElementTypes {
  xla::PrimitiveType lhs;
  xla::PrimitiveType rhs;
  xla::PrimitiveType output;
};
constexpr GemmElementTypes kSupportedGemmElementTypes[] = {
    {xla::F32, xla::F32, xla::F32},    {xla::F64, xla::F64, xla::F64},
    {xla::BF16, xla::BF16, xla::BF16}, {xla::BF16, xla::BF16, xla::F32},
    {xla::F16, xla::F16, xla::F16},    {xla::F16, xla::F16, xla::F32},
    {xla::S8, xla::S8, xla::S32},
};

bool IsSupportedGemmElementTypes(xla::PrimitiveType lhs, xla::PrimitiveType rhs,
                                 xla::PrimitiveType output) {
  for (const auto& t : kSupportedGemmElementTypes) {
    if (t.lhs == lhs && t.rhs == rhs && t.output == output) {
      return true;
    }
  }
  return false;
}

// Returns whether the gemm adds beta times C to the result.
bool HasBetaTerm(float beta) { return std::fabs(beta) > 1e-6; }

// Everything a oneDNN matmul call needs, derived from a GemmConfig, its
// epilogue and (optional) buffers. Populated by CreateMatMulParams and
// consumed by both the HLO-pass and execution paths. Everything except the
// buffers comes from the config and epilogue, so both paths build the same
// primitive desc.
struct OneDnnMatMulParams {
  dnnl::memory::desc lhs_md;
  dnnl::memory::desc rhs_md;
  dnnl::memory::desc output_md;
  dnnl::memory::desc bias_md;  // empty when the epilogue has no bias
  float alpha;
  float beta;
  sycl_gemm::GemmBackendEpilogue epilogue;
  // Device buffers. Null at HLO-pass time; non-null during execution.
  // bias_data is also null when the epilogue has no bias.
  void* lhs_data;
  void* rhs_data;
  void* output_data;
  void* bias_data;
};

absl::StatusOr<OneDnnMatMulParams> CreateMatMulParams(
    const se::gpu::GemmConfig& config, sycl_gemm::GemmBackendEpilogue epilogue,
    se::DeviceAddressBase lhs_buffer = se::DeviceAddressBase(),
    se::DeviceAddressBase rhs_buffer = se::DeviceAddressBase(),
    se::DeviceAddressBase output_buffer = se::DeviceAddressBase(),
    se::DeviceAddressBase bias_buffer = se::DeviceAddressBase()) {
  ABSL_ASSIGN_OR_RETURN(const bool has_bias,
                   sycl_gemm::EpilogueAddsVectorBias(epilogue));
  auto lhs_layout = se::gpu::MatrixLayout{config.lhs_layout};
  auto rhs_layout = se::gpu::MatrixLayout{config.rhs_layout};
  auto output_layout = se::gpu::MatrixLayout{config.output_layout};
  if (!IsSupportedGemmElementTypes(lhs_layout.dtype, rhs_layout.dtype,
                                   output_layout.dtype)) {
    return absl::InternalError(absl::StrCat(
        "Unexpected GEMM lhs type ",
        xla::primitive_util::LowercasePrimitiveTypeName(lhs_layout.dtype),
        ", rhs type ",
        xla::primitive_util::LowercasePrimitiveTypeName(rhs_layout.dtype),
        " and output type ",
        xla::primitive_util::LowercasePrimitiveTypeName(output_layout.dtype)));
  }

  MatrixDescriptor lhs = GetMatrixDesc(lhs_layout, lhs_buffer);
  MatrixDescriptor rhs = GetMatrixDesc(rhs_layout, rhs_buffer);
  MatrixDescriptor output = GetMatrixDesc(output_layout, output_buffer);
  MakeBlasGemmCompatible(lhs, rhs, output);

  ABSL_ASSIGN_OR_RETURN(dnnl::memory::desc lhs_md, ToOneDnnMemDesc(lhs));
  ABSL_ASSIGN_OR_RETURN(dnnl::memory::desc rhs_md, ToOneDnnMemDesc(rhs));
  ABSL_ASSIGN_OR_RETURN(dnnl::memory::desc output_md, ToOneDnnMemDesc(output));
  return OneDnnMatMulParams{
      .lhs_md = lhs_md,
      .rhs_md = rhs_md,
      .output_md = output_md,
      .bias_md = has_bias ? MakeBiasMemDesc(output_md) : dnnl::memory::desc(),
      .alpha = static_cast<float>(config.alpha.real()),
      .beta = static_cast<float>(config.beta),
      .epilogue = epilogue,
      .lhs_data = lhs.data.opaque(),
      .rhs_data = rhs.data.opaque(),
      .output_data = output.data.opaque(),
      .bias_data = has_bias ? bias_buffer.opaque() : nullptr,
  };
}

// Builds the oneDNN matmul primitive descriptor. The workspace pass and gemm
// execution both use it, so the scratchpad size reserved at compile time
// matches what the primitive needs at run time.
absl::StatusOr<std::unique_ptr<dnnl::matmul::primitive_desc>>
CreateOneDnnMatMulPrimDesc(const dnnl::engine& dnnl_engine,
                           const OneDnnMatMulParams& params) {
  try {
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // fpmath mode is only meaningful when the accumulator input is f32.
    if (params.lhs_md.get_data_type() == dnnl::memory::data_type::f32) {
      attr.set_fpmath_mode(GetFP32MathMode());
    }

    // Post-ops compute activation(alpha * (lhs @ rhs + bias) + beta * output),
    // where output on the right is its value before the matmul. In order:
    //   1. Scale by alpha (if alpha != 1.0) using eltwise_linear
    //   2. Add beta * output (if beta != 0.0) using sum
    //   3. Apply activation (if any)
    dnnl::post_ops post_ops;
    if (std::fabs(params.alpha - 1.0f) >= 1e-6) {
      post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, params.alpha,
                              0.0f);
    }
    if (HasBetaTerm(params.beta)) {
      post_ops.append_sum(params.beta);
    }
    switch (params.epilogue) {
      case sycl_gemm::GemmBackendEpilogue::RELU:
      case sycl_gemm::GemmBackendEpilogue::BIAS_RELU:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
        break;
      case sycl_gemm::GemmBackendEpilogue::GELU:
      case sycl_gemm::GemmBackendEpilogue::BIAS_GELU:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.0f, 0.0f);
        break;
      case sycl_gemm::GemmBackendEpilogue::DEFAULT:
      case sycl_gemm::GemmBackendEpilogue::BIAS:
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported activation mode: ",
                         static_cast<int>(params.epilogue)));
    }
    attr.set_post_ops(post_ops);

    return std::make_unique<dnnl::matmul::primitive_desc>(
        dnnl_engine, params.lhs_md, params.rhs_md, params.bias_md,
        params.output_md, attr);
  } catch (const dnnl::error& e) {
    return absl::InternalError(absl::StrCat("oneDNN error: ", e.message));
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<dnnl::matmul::primitive_desc>>
CreateMatMulPrimDescFromGemmConfig(const xla::gpu::GemmConfig& config,
                                   sycl_gemm::GemmBackendEpilogue epilogue) {
  ABSL_ASSIGN_OR_RETURN(OneDnnMatMulParams params,
                   CreateMatMulParams(config, epilogue));

  dnnl::engine dnnl_engine(dnnl::engine::kind::gpu, 0);
  return CreateOneDnnMatMulPrimDesc(dnnl_engine, params);
}

namespace {

absl::Status DoOnednnGemm(const OneDnnMatMulParams& params,
                          se::DeviceAddressBase c, se::Stream* stream,
                          se::DeviceAddressBase workspace,
                          se::ScratchAllocator* scratch_allocator) {
  ::sycl::queue* stream_handle =
      absl::bit_cast<::sycl::queue*>(stream->platform_specific_handle().stream);
  auto dnnl_engine = FindOrCreateEngine(stream_handle);

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<dnnl::matmul::primitive_desc> matmul_pd,
                   CreateOneDnnMatMulPrimDesc(dnnl_engine, params));

  // Pass a workspace buffer only if oneDNN needs scratchpad memory.
  const size_t scratchpad_size = matmul_pd->scratchpad_desc().get_size();
  void* workspace_addr = DNNL_MEMORY_NONE;
  if (scratchpad_size > 0) {
    if (scratch_allocator != nullptr) {
      ABSL_ASSIGN_OR_RETURN(stream_executor::DeviceAddress<uint8_t> alloc,
                       scratch_allocator->AllocateBytes(scratchpad_size));
      workspace_addr = alloc.opaque();
    } else {
      // The workspace was sized at compile time. Check that it covers the
      // scratchpad oneDNN needs at run time.
      if (workspace.size() < scratchpad_size) {
        return absl::InternalError(absl::StrCat(
            "oneDNN matmul needs ", scratchpad_size,
            " bytes of scratchpad but the workspace has ", workspace.size()));
      }
      workspace_addr = workspace.opaque();
    }
  }

  auto scratchpad_mem = dnnl::sycl_interop::make_memory(
      matmul_pd->scratchpad_desc(), dnnl_engine,
      dnnl::sycl_interop::memory_kind::usm, workspace_addr);

  auto matmul_primitive = dnnl::matmul(*matmul_pd);
  auto dnnl_stream =
      dnnl::sycl_interop::make_stream(dnnl_engine, *stream_handle);
  std::unordered_map<int, dnnl::memory> fwd_primitive_args{
      {DNNL_ARG_SRC,
       CreateDnnlMemory(params.lhs_md, dnnl_engine, params.lhs_data)},
      {DNNL_ARG_WEIGHTS,
       CreateDnnlMemory(params.rhs_md, dnnl_engine, params.rhs_data)},
      {DNNL_ARG_DST,
       CreateDnnlMemory(params.output_md, dnnl_engine, params.output_data)},
      {DNNL_ARG_SCRATCHPAD, scratchpad_mem},
  };
  if (params.bias_data != nullptr) {
    fwd_primitive_args.emplace(
        DNNL_ARG_BIAS,
        CreateDnnlMemory(params.bias_md, dnnl_engine, params.bias_data));
  }

  // The sum post-op adds beta times the output buffer's existing contents, so
  // C must be in the output before the matmul runs. Copy it there unless C
  // already shares the output buffer.
  if (HasBetaTerm(params.beta) && c.opaque() != params.output_data) {
    const size_t output_size = params.output_md.get_size();
    se::DeviceAddressBase dst_addr(params.output_data, output_size);
    ABSL_RETURN_IF_ERROR(stream->MemcpyD2D(&dst_addr, c, output_size));
  }

  matmul_primitive.execute(dnnl_stream, fwd_primitive_args);
  return absl::OkStatus();
}

}  // namespace

absl::Status RunGemm(const gpu::GemmConfig& config,
                     se::DeviceAddressBase lhs_buffer,
                     se::DeviceAddressBase rhs_buffer,
                     se::DeviceAddressBase c_buffer,
                     se::DeviceAddressBase output_buffer,
                     se::DeviceAddressBase bias_buffer,
                     se::DeviceAddressBase workspace, se::Stream* stream,
                     sycl_gemm::GemmBackendEpilogue epilogue, int64_t algorithm,
                     se::ScratchAllocator* scratch_allocator) {
  if (algorithm != kOneDnnGemm) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported GEMM algorithm: ", algorithm));
  }

  // The workspace was sized from the config and epilogue alone, so the
  // buffers must match them.
  ABSL_ASSIGN_OR_RETURN(const bool has_bias,
                   sycl_gemm::EpilogueAddsVectorBias(epilogue));
  if (has_bias != (bias_buffer.opaque() != nullptr)) {
    return absl::InternalError(absl::StrCat(
        "Gemm epilogue ", sycl_gemm::EpilogueCast(epilogue).value_or("unknown"),
        has_bias ? " requires a bias buffer" : " does not take a bias buffer"));
  }
  if (HasBetaTerm(config.beta) && c_buffer.opaque() == nullptr) {
    return absl::InternalError(absl::StrCat("Gemm beta is ", config.beta,
                                            " but there is no C buffer"));
  }

  ABSL_ASSIGN_OR_RETURN(OneDnnMatMulParams params,
                   CreateMatMulParams(config, epilogue, lhs_buffer, rhs_buffer,
                                      output_buffer, bias_buffer));
  return DoOnednnGemm(params, c_buffer, stream, workspace, scratch_allocator);
}
}  // namespace sycl
}  // namespace stream_executor
