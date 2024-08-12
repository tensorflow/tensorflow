/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/transforms/cudnn_custom_call_compiler.h"

#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/stream_executor/cuda/cuda_dnn.h"
#include "xla/stream_executor/cuda/cudnn_frontend_helpers.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

inline absl::StatusOr<CudnnfMHAMaskKind> AsCudnnFmhaMaskKind(
    CudnnfMHABackendConfig_MaskType mask_type) {
  switch (mask_type) {
    case CudnnfMHABackendConfig::NO_MASK:
      return CudnnfMHAMaskKind::kNoMask;
    case CudnnfMHABackendConfig::PADDING:
      return CudnnfMHAMaskKind::kPadding;
    case CudnnfMHABackendConfig::CAUSAL:
      return CudnnfMHAMaskKind::kCausal;
    case CudnnfMHABackendConfig::PADDING_CAUSAL:
      return CudnnfMHAMaskKind::kPaddingCausal;
    case CudnnfMHABackendConfig::ALIBI:
      return CudnnfMHAMaskKind::kAlibi;
    default:
      return xla::Internal("Unknown fmha mask kind.");
  }
}

// This is an interim structure to hold the parameters to construct a
// GpufMHAConfig.
// Struct to describe properties of a FMHA without being tied to specific
// IR. Will be used to help build FMHA thunks from either XLA HLO or
// LHLO GPU dialect in MLIR.
struct GpufMHADescriptor {
  CudnnfMHAKind kind;
  CudnnfMHABackendConfig backend_config;
  CudnnfMHAMaskKind mask_type;
  Shape lhs_bmm1_shape;
  Shape rhs_bmm1_shape;
  Shape rhs_bmm2_shape;
  Shape intermediate_lhs_bmm2_shape;
  // This will contain both output shape and activation shape
  absl::InlinedVector<Shape, 2> output_shapes;
  DotDimensionNumbers bmm1_dnums;
  DotDimensionNumbers bmm2_dnums;

  std::optional<Shape> mask_shape;
  std::optional<Shape> bias_shape;
};

struct GpufMHABackwardDescriptor {
  CudnnfMHAKind kind;
  CudnnfMHABackendConfig backend_config;
  CudnnfMHAMaskKind mask_type;
  Shape bmm1_grad_gemm1_rhs_shape;
  Shape bmm1_grad_gemm2_rhs_shape;
  Shape bmm2_grad_gemm1_lhs_shape;
  Shape bmm2_grad_gemm2_rhs_shape;
  Shape d_output_shape;
  Shape d_bmm1_lhs_shape;
  Shape d_bmm1_rhs_shape;
  Shape d_bmm2_rhs_shape;
  DotDimensionNumbers bmm1_grad_gemm1_dnums;
  DotDimensionNumbers bmm1_grad_gemm2_dnums;
  DotDimensionNumbers bmm2_grad_gemm1_dnums;
  DotDimensionNumbers bmm2_grad_gemm2_dnums;

  std::optional<Shape> d_s_shape;
  std::optional<Shape> fwd_output_shape;
  std::optional<Shape> mask_shape;
  std::optional<Shape> d_bias_shape;
  std::optional<Shape> bias_shape;
  bool force_deterministic;
};

// Structure to describe static properties of a GPU fused Multi-Headed
// Attention.
struct GpufMHAConfig {
  static absl::StatusOr<GpufMHAConfig> For(const GpufMHADescriptor &fmha_desc);
  PrimitiveType
      input_type;  // Capture the primitive type of one of the inputs of BMM1
  PrimitiveType output_type;
  CudnnfMHAKind kind;
  std::optional<double> fmha_scale;
  std::optional<double> dropout_rate;
  std::optional<int64_t> seed;

  se::dnn::AlgorithmDesc algorithm;
  CudnnfMHAMaskKind mask_type;
  // bias -> [1, num_attn_heads, q_seq_len, kv_seq_len]
  // mask -> [batch_size, 1, q_seq_len, kv_seq_len]
  se::dnn::MatmulTensorDescriptor lhs_bmm1;
  se::dnn::MatmulTensorDescriptor rhs_bmm1;
  se::dnn::MatmulTensorDescriptor rhs_bmm2;
  se::dnn::MatmulTensorDescriptor intermediate_lhs_bmm2;
  se::dnn::TensorDescriptor output;

  std::optional<se::dnn::TensorDescriptor> activation;
  std::optional<se::dnn::TensorDescriptor> mask;
  std::optional<se::dnn::TensorDescriptor> bias;
};

// Structure to describe static properties of a GPU fused Multi-Headed
// Attention backward.
struct GpufMHABackwardConfig {
  static absl::StatusOr<GpufMHABackwardConfig> For(
      const GpufMHABackwardDescriptor &fmha_desc);
  PrimitiveType
      input_type;  // Capture the primitive type of one of the inputs of BMM1
  PrimitiveType output_type;
  CudnnfMHAKind kind;
  std::optional<double> fmha_scale;
  std::optional<double> dropout_rate;
  std::optional<int64_t> seed;

  se::dnn::AlgorithmDesc algorithm;
  CudnnfMHAMaskKind mask_type;
  // mask -> [batch_size, 1, q_seq_len, kv_seq_len]
  // d_bias -> [1, num_heads, q_seq_len, kv_seq_len]
  se::dnn::MatmulTensorDescriptor bmm1_grad_gemm1_rhs;
  se::dnn::MatmulTensorDescriptor bmm1_grad_gemm2_rhs;
  se::dnn::MatmulTensorDescriptor bmm2_grad_gemm1_lhs;
  se::dnn::MatmulTensorDescriptor bmm2_grad_gemm2_rhs;
  se::dnn::MatmulTensorDescriptor d_output;
  se::dnn::TensorDescriptor d_bmm1_lhs;
  se::dnn::TensorDescriptor d_bmm1_rhs;
  se::dnn::TensorDescriptor d_bmm2_rhs;
  std::optional<se::dnn::TensorDescriptor> d_s;
  std::optional<se::dnn::TensorDescriptor> mask;
  std::optional<se::dnn::TensorDescriptor> d_bias;
  std::optional<se::dnn::TensorDescriptor> fwd_output;
  std::optional<se::dnn::TensorDescriptor> bias;
};

using se::DeviceMemory;
using se::DeviceMemoryBase;
using se::dnn::DataType;
using se::dnn::MatmulTensorDescriptor;
using se::dnn::TensorDescriptor;

/*static*/ absl::StatusOr<GpufMHAConfig> GpufMHAConfig::For(
    const GpufMHADescriptor &desc) {
  // Get shapes from desc.
  const Shape &lhs_bmm1_shape = desc.lhs_bmm1_shape;
  const Shape &rhs_bmm1_shape = desc.rhs_bmm1_shape;
  const Shape &rhs_bmm2_shape = desc.rhs_bmm2_shape;
  const Shape &intermediate_lhs_bmm2_shape = desc.intermediate_lhs_bmm2_shape;
  const Shape &output_shape = desc.output_shapes[0];

  // Get DNN dtype from primtive types
  TF_ASSIGN_OR_RETURN(
      DataType lhs_bmm1_type,
      GetDNNDataTypeFromPrimitiveType(lhs_bmm1_shape.element_type()));
  TF_ASSIGN_OR_RETURN(
      DataType rhs_bmm1_type,
      GetDNNDataTypeFromPrimitiveType(rhs_bmm1_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType rhs_bmm2_type,
      GetDNNDataTypeFromPrimitiveType(rhs_bmm2_shape.element_type()));
  TF_ASSIGN_OR_RETURN(DataType lhs_bmm2_type,
                      GetDNNDataTypeFromPrimitiveType(
                          intermediate_lhs_bmm2_shape.element_type()));
  TF_ASSIGN_OR_RETURN(DataType output_type, GetDNNDataTypeFromPrimitiveType(
                                                output_shape.element_type()));
  GpufMHAConfig config;
  config.input_type = lhs_bmm1_shape.element_type();
  config.output_type = output_shape.element_type();

  // Get MatmulTensorDescriptors for BMM1
  config.lhs_bmm1 =
      MatmulTensorDescriptor::For(lhs_bmm1_type, lhs_bmm1_shape.dimensions(),
                                  desc.lhs_bmm1_shape.layout().minor_to_major(),
                                  desc.bmm1_dnums.lhs_batch_dimensions(),
                                  desc.bmm1_dnums.lhs_contracting_dimensions());
  config.rhs_bmm1 =
      MatmulTensorDescriptor::For(rhs_bmm1_type, rhs_bmm1_shape.dimensions(),
                                  desc.rhs_bmm1_shape.layout().minor_to_major(),
                                  desc.bmm1_dnums.rhs_batch_dimensions(),
                                  desc.bmm1_dnums.rhs_contracting_dimensions());

  // Get MatmulTensorDescriptors for BMM2
  config.rhs_bmm2 =
      MatmulTensorDescriptor::For(rhs_bmm2_type, rhs_bmm2_shape.dimensions(),
                                  desc.rhs_bmm2_shape.layout().minor_to_major(),
                                  desc.bmm2_dnums.rhs_batch_dimensions(),
                                  desc.bmm2_dnums.rhs_contracting_dimensions());

  config.intermediate_lhs_bmm2 = MatmulTensorDescriptor::For(
      lhs_bmm2_type, intermediate_lhs_bmm2_shape.dimensions(),
      desc.intermediate_lhs_bmm2_shape.layout().minor_to_major(),
      desc.bmm2_dnums.lhs_batch_dimensions(),
      desc.bmm2_dnums.lhs_contracting_dimensions());

  config.output = TensorDescriptor::For(output_type, output_shape.dimensions(),
                                        output_shape.layout().minor_to_major());

  if (desc.output_shapes.size() > 1) {
    const Shape &activation_shape = desc.output_shapes.back();
    // Generally, activation should have same type as output, but set it
    // explicityly just to be safe.
    TF_ASSIGN_OR_RETURN(
        DataType activation_type,
        GetDNNDataTypeFromPrimitiveType(activation_shape.element_type()));
    config.activation =
        TensorDescriptor::For(activation_type, activation_shape.dimensions(),
                              activation_shape.layout().minor_to_major());
  }

  if (desc.mask_shape) {
    const Shape &mask_shape = *desc.mask_shape;
    TF_ASSIGN_OR_RETURN(DataType mask_type, GetDNNDataTypeFromPrimitiveType(
                                                mask_shape.element_type()));
    config.mask = TensorDescriptor::For(mask_type, mask_shape.dimensions(),
                                        mask_shape.layout().minor_to_major());
  }

  if (desc.bias_shape) {
    const Shape &bias_shape = *desc.bias_shape;
    TF_ASSIGN_OR_RETURN(DataType bias_type, GetDNNDataTypeFromPrimitiveType(
                                                bias_shape.element_type()));
    config.bias = TensorDescriptor::For(bias_type, bias_shape.dimensions(),
                                        bias_shape.layout().minor_to_major());
  }
  config.kind = desc.kind;
  config.mask_type = desc.mask_type;
  const CudnnfMHABackendConfig &backend_config = desc.backend_config;
  config.algorithm = se::dnn::AlgorithmDesc(backend_config.algorithm());
  config.fmha_scale.emplace(backend_config.fmha_scale());
  config.dropout_rate.emplace(backend_config.dropout_rate());
  config.seed.emplace(backend_config.seed());
  return config;
}

/*static*/ absl::StatusOr<GpufMHABackwardConfig> GpufMHABackwardConfig::For(
    const GpufMHABackwardDescriptor &desc) {
  // Get shapes from desc.
  const Shape &bmm1_grad_gemm1_rhs_shape = desc.bmm1_grad_gemm1_rhs_shape;
  const Shape &bmm1_grad_gemm2_rhs_shape = desc.bmm1_grad_gemm2_rhs_shape;
  const Shape &bmm2_grad_gemm1_lhs_shape = desc.bmm2_grad_gemm1_lhs_shape;
  const Shape &bmm2_grad_gemm2_rhs_shape = desc.bmm2_grad_gemm2_rhs_shape;
  const Shape &d_output_shape = desc.d_output_shape;
  const Shape &d_bmm1_lhs_shape = desc.d_bmm1_lhs_shape;
  const Shape &d_bmm1_rhs_shape = desc.d_bmm1_rhs_shape;
  const Shape &d_bmm2_rhs_shape = desc.d_bmm2_rhs_shape;
  // Get DNN dtype from primtive types
  TF_ASSIGN_OR_RETURN(DataType bmm1_grad_gemm1_rhs_type,
                      GetDNNDataTypeFromPrimitiveType(
                          bmm1_grad_gemm1_rhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(DataType bmm1_grad_gemm2_rhs_type,
                      GetDNNDataTypeFromPrimitiveType(
                          bmm1_grad_gemm2_rhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(DataType bmm2_grad_gemm1_lhs_type,
                      GetDNNDataTypeFromPrimitiveType(
                          bmm2_grad_gemm1_lhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(DataType bmm2_grad_gemm2_rhs_type,
                      GetDNNDataTypeFromPrimitiveType(
                          bmm2_grad_gemm2_rhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType d_output_type,
      GetDNNDataTypeFromPrimitiveType(d_output_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType d_bmm1_lhs_type,
      GetDNNDataTypeFromPrimitiveType(d_bmm1_lhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType d_bmm1_rhs_type,
      GetDNNDataTypeFromPrimitiveType(d_bmm1_rhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType d_bmm2_rhs_type,
      GetDNNDataTypeFromPrimitiveType(d_bmm2_rhs_shape.element_type()));

  GpufMHABackwardConfig config;
  config.input_type = bmm1_grad_gemm1_rhs_shape.element_type();
  config.output_type = d_bmm1_lhs_shape.element_type();

  // Get MatmulTensorDescriptors for lhs of BMM1 grad GEMM 1
  config.bmm1_grad_gemm1_rhs = MatmulTensorDescriptor::For(
      bmm1_grad_gemm1_rhs_type, bmm1_grad_gemm1_rhs_shape.dimensions(),
      desc.bmm1_grad_gemm1_rhs_shape.layout().minor_to_major(),
      desc.bmm1_grad_gemm1_dnums.rhs_batch_dimensions(),
      desc.bmm1_grad_gemm1_dnums.rhs_contracting_dimensions());

  // Get MatmulTensorDescriptors for rhs of BMM1 grad GEMM 2
  config.bmm1_grad_gemm2_rhs = MatmulTensorDescriptor::For(
      bmm1_grad_gemm2_rhs_type, bmm1_grad_gemm2_rhs_shape.dimensions(),
      desc.bmm1_grad_gemm2_rhs_shape.layout().minor_to_major(),
      desc.bmm1_grad_gemm2_dnums.rhs_batch_dimensions(),
      desc.bmm1_grad_gemm2_dnums.rhs_contracting_dimensions());

  // Get MatmulTensorDescriptors for BMM2 grad GEMM 1
  config.bmm2_grad_gemm1_lhs = MatmulTensorDescriptor::For(
      bmm2_grad_gemm1_lhs_type, bmm2_grad_gemm1_lhs_shape.dimensions(),
      desc.bmm2_grad_gemm1_lhs_shape.layout().minor_to_major(),
      desc.bmm2_grad_gemm1_dnums.lhs_batch_dimensions(),
      desc.bmm2_grad_gemm1_dnums.lhs_contracting_dimensions());

  config.d_output = MatmulTensorDescriptor::For(
      d_output_type, d_output_shape.dimensions(),
      desc.d_output_shape.layout().minor_to_major(),
      desc.bmm2_grad_gemm1_dnums.rhs_batch_dimensions(),
      desc.bmm2_grad_gemm1_dnums.rhs_contracting_dimensions());

  // Get MatmulTensorDescriptors for BMM2 grad GEMM 2
  config.bmm2_grad_gemm2_rhs = MatmulTensorDescriptor::For(
      bmm2_grad_gemm2_rhs_type, bmm2_grad_gemm2_rhs_shape.dimensions(),
      desc.bmm2_grad_gemm2_rhs_shape.layout().minor_to_major(),
      desc.bmm2_grad_gemm2_dnums.rhs_batch_dimensions(),
      desc.bmm2_grad_gemm2_dnums
          .rhs_contracting_dimensions());  // FMHA TODO: transpose here?

  config.d_bmm1_lhs =
      TensorDescriptor::For(d_bmm1_lhs_type, d_bmm1_lhs_shape.dimensions(),
                            d_bmm1_lhs_shape.layout().minor_to_major());
  config.d_bmm1_rhs =
      TensorDescriptor::For(d_bmm1_rhs_type, d_bmm1_rhs_shape.dimensions(),
                            d_bmm1_rhs_shape.layout().minor_to_major());
  config.d_bmm2_rhs =
      TensorDescriptor::For(d_bmm2_rhs_type, d_bmm2_rhs_shape.dimensions(),
                            d_bmm2_rhs_shape.layout().minor_to_major());
  config.d_s = TensorDescriptor::For(
      bmm2_grad_gemm1_lhs_type, bmm2_grad_gemm1_lhs_shape.dimensions(),
      bmm2_grad_gemm1_lhs_shape.layout().minor_to_major());

  if (desc.d_bias_shape) {
    const Shape &d_bias_shape = *desc.d_bias_shape;
    // Get DNN dtype from primtive types
    TF_ASSIGN_OR_RETURN(DataType d_bias_type, GetDNNDataTypeFromPrimitiveType(
                                                  d_bias_shape.element_type()));
    config.d_bias =
        TensorDescriptor::For(d_bias_type, d_bias_shape.dimensions(),
                              d_bias_shape.layout().minor_to_major());
  }

  if (desc.mask_shape) {
    const Shape &mask_shape = *desc.mask_shape;
    TF_ASSIGN_OR_RETURN(DataType mask_type, GetDNNDataTypeFromPrimitiveType(
                                                mask_shape.element_type()));
    config.mask = TensorDescriptor::For(mask_type, mask_shape.dimensions(),
                                        mask_shape.layout().minor_to_major());
  }
  if (desc.fwd_output_shape) {
    const Shape &fwd_output_shape = *desc.fwd_output_shape;
    TF_ASSIGN_OR_RETURN(
        DataType fwd_output_type,
        GetDNNDataTypeFromPrimitiveType(fwd_output_shape.element_type()));
    config.fwd_output =
        TensorDescriptor::For(fwd_output_type, fwd_output_shape.dimensions(),
                              fwd_output_shape.layout().minor_to_major());
  }

  if (desc.bias_shape) {
    const Shape &bias_shape = *desc.bias_shape;
    TF_ASSIGN_OR_RETURN(DataType bias_type, GetDNNDataTypeFromPrimitiveType(
                                                bias_shape.element_type()));
    config.bias = TensorDescriptor::For(bias_type, bias_shape.dimensions(),
                                        bias_shape.layout().minor_to_major());
  }

  config.kind = desc.kind;
  config.mask_type = desc.mask_type;
  const CudnnfMHABackendConfig &backend_config = desc.backend_config;
  config.algorithm = se::dnn::AlgorithmDesc(backend_config.algorithm());
  config.fmha_scale.emplace(backend_config.fmha_scale());
  config.dropout_rate.emplace(backend_config.dropout_rate());
  config.seed.emplace(backend_config.seed());
  return config;
}

absl::StatusOr<se::gpu::CudnnGraph> HloCustomCallToCuDnnGraph(
    se::dnn::DnnSupport &dnn_support, HloCustomCallInstruction *custom_call) {
  if (IsFwdCustomCallTofMHA(*custom_call)) {
    TF_ASSIGN_OR_RETURN(const xla::gpu::CudnnfMHAKind kind,
                        xla::gpu::GetCudnnfMHAKind(custom_call));
    std::optional<Shape> mask_shape, bias_shape;
    {
      bool has_bias = kind == CudnnfMHAKind::kScaleBiasSoftmax ||
                      kind == CudnnfMHAKind::kScaleBiasSoftmaxDropout;

      if (has_bias) {
        const HloInstruction *bias = custom_call->operand(3);
        bias_shape = bias->shape();
      }
    }

    TF_ASSIGN_OR_RETURN(
        const auto gpu_config,
        custom_call->backend_config<xla::gpu::GpuBackendConfig>());
    const xla::gpu::CudnnfMHABackendConfig &config =
        gpu_config.cudnn_fmha_backend_config();
    Shape intermediate_tensor_shape(config.intermediate_tensor_shape());
    absl::InlinedVector<Shape, 2> output_shapes = {
        ShapeUtil::GetSubshape(custom_call->shape(), {0})};

    bool has_activation =
        xla::ShapeUtil::TupleElementCount(custom_call->shape()) == 3;
    if (has_activation) {
      output_shapes.push_back(
          ShapeUtil::GetSubshape(custom_call->shape(), {1}));
    }

    Shape q_shape = custom_call->operand(0)->shape();
    Shape k_shape = custom_call->operand(1)->shape();
    Shape v_shape = custom_call->operand(2)->shape();
    TF_ASSIGN_OR_RETURN(CudnnfMHAMaskKind cudnn_mask_type,
                        AsCudnnFmhaMaskKind(config.mask_type()));
    GpufMHADescriptor descriptor = {kind,
                                    config,
                                    cudnn_mask_type,
                                    q_shape,
                                    k_shape,
                                    v_shape,
                                    intermediate_tensor_shape,
                                    output_shapes,
                                    config.bmm1_dot_dimension_numbers(),
                                    config.bmm2_dot_dimension_numbers(),
                                    mask_shape,
                                    bias_shape};

    TF_ASSIGN_OR_RETURN(GpufMHAConfig fmha_config,
                        GpufMHAConfig::For(descriptor));
    TF_ASSIGN_OR_RETURN(
        se::dnn::FMHAMaskKind dnn_mask_type,
        GetDNNFmhaMaskKindFromCudnnFmhaMaskKind(fmha_config.mask_type));
    TF_ASSIGN_OR_RETURN(
        se::gpu::CudnnGraph graph,
        se::gpu::GetCudnnFlashAttentionOperationGraph(
            dnn_support, fmha_config.lhs_bmm1, fmha_config.rhs_bmm1,
            fmha_config.rhs_bmm2, fmha_config.output, fmha_config.bias,
            fmha_config.activation, static_cast<float>(*fmha_config.fmha_scale),
            fmha_config.dropout_rate && *fmha_config.dropout_rate > 0.0,
            fmha_config.dropout_rate, dnn_mask_type));
    return std::move(graph);
  } else {
    TF_ASSIGN_OR_RETURN(
        auto gpu_config,
        custom_call->backend_config<xla::gpu::GpuBackendConfig>());
    xla::gpu::CudnnfMHABackendConfig &config =
        *gpu_config.mutable_cudnn_fmha_backend_config();

    int input_index = 0;
    Shape bmm1_grad_gemm1_rhs_shape =
        custom_call->operand(input_index++)->shape();
    Shape bmm1_grad_gemm2_rhs_shape =
        custom_call->operand(input_index++)->shape();
    Shape bmm2_grad_gemm2_rhs_shape =
        custom_call->operand(input_index++)->shape();
    Shape bmm2_grad_gemm1_lhs_shape(config.intermediate_tensor_shape());
    input_index++;
    Shape d_output_shape = custom_call->operand(input_index++)->shape();

    TF_ASSIGN_OR_RETURN(const CudnnfMHAKind kind,
                        GetCudnnfMHAKind(custom_call));
    std::optional<Shape> mask_shape;

    bool has_bias = (kind == CudnnfMHAKind::kBackwardScaleBiasSoftmax ||
                     kind == CudnnfMHAKind::kBackwardScaleBiasSoftmaxDropout);
    std::optional<Shape> bias_shape;
    if (has_bias) {
      bias_shape = custom_call->operand(input_index++)->shape();
    }

    std::optional<Shape> fwd_output_shape =
        custom_call->operand(input_index++)->shape();
    if (config.mask_type() == xla::gpu::CudnnfMHABackendConfig::PADDING ||
        config.mask_type() ==
            xla::gpu::CudnnfMHABackendConfig::PADDING_CAUSAL) {
      // skip q_seqlen and kv_seqlen
      input_index += 2;
    }
    TF_RET_CHECK(input_index == custom_call->operand_count());

    int output_index = 0;
    Shape d_bmm1_lhs_shape =
        ShapeUtil::GetSubshape(custom_call->shape(), {output_index++});
    Shape d_bmm1_rhs_shape =
        ShapeUtil::GetSubshape(custom_call->shape(), {output_index++});
    Shape d_bmm2_rhs_shape =
        ShapeUtil::GetSubshape(custom_call->shape(), {output_index++});
    std::optional<Shape> d_s_shape;
    std::optional<Shape> d_bias_shape;
    bool has_dbias = custom_call->shape().tuple_shapes().size() == 5;
    if (has_dbias) {
      d_bias_shape =
          ShapeUtil::GetSubshape(custom_call->shape(), {output_index++});
    }
    // The last one is the workspace.
    TF_RET_CHECK(output_index ==
                 custom_call->shape().tuple_shapes().size() - 1);
    TF_ASSIGN_OR_RETURN(CudnnfMHAMaskKind cudnn_mask_type,
                        AsCudnnFmhaMaskKind(config.mask_type()));

    const DebugOptions &debug_options =
        custom_call->GetModule()->config().debug_options();
    bool force_deterministic =
        debug_options.xla_gpu_deterministic_ops() ||
        debug_options.xla_gpu_exclude_nondeterministic_ops();
    config.set_force_deterministic(force_deterministic);
    TF_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_config));

    GpufMHABackwardDescriptor descriptor = {
        kind,
        config,
        cudnn_mask_type,
        bmm1_grad_gemm1_rhs_shape,
        bmm1_grad_gemm2_rhs_shape,
        bmm2_grad_gemm1_lhs_shape,
        bmm2_grad_gemm2_rhs_shape,
        d_output_shape,
        d_bmm1_lhs_shape,
        d_bmm1_rhs_shape,
        d_bmm2_rhs_shape,
        config.bmm1_grad_gemm1_dot_dimension_numbers(),
        config.bmm1_grad_gemm2_dot_dimension_numbers(),
        config.bmm2_grad_gemm1_dot_dimension_numbers(),
        config.bmm2_grad_gemm2_dot_dimension_numbers(),
        d_s_shape,
        fwd_output_shape,
        mask_shape,
        d_bias_shape,
        bias_shape,
        force_deterministic};

    TF_ASSIGN_OR_RETURN(GpufMHABackwardConfig fmha_config,
                        GpufMHABackwardConfig::For(descriptor));
    TF_ASSIGN_OR_RETURN(
        se::dnn::FMHAMaskKind dnn_mask_type,
        GetDNNFmhaMaskKindFromCudnnFmhaMaskKind(fmha_config.mask_type));

    TF_ASSIGN_OR_RETURN(
        se::gpu::CudnnGraph graph,
        se::gpu::GetCudnnFlashAttentionBackwardOperationGraph(
            dnn_support, fmha_config.bmm1_grad_gemm1_rhs,
            fmha_config.bmm1_grad_gemm2_rhs, fmha_config.bmm2_grad_gemm1_lhs,
            fmha_config.bmm2_grad_gemm2_rhs, fmha_config.d_output,
            fmha_config.d_bmm1_lhs, fmha_config.d_bmm1_rhs,
            fmha_config.d_bmm2_rhs, fmha_config.bias, fmha_config.dropout_rate,
            fmha_config.seed, *fmha_config.fmha_scale,
            fmha_config.dropout_rate && *fmha_config.dropout_rate > 0.0,
            fmha_config.bias != std::nullopt, dnn_mask_type,
            force_deterministic));
    return std::move(graph);
  }
}

class CuDnnCustomCallVisitor : public DfsHloRewriteVisitor {
 public:
  explicit CuDnnCustomCallVisitor(se::dnn::DnnSupport &dnn_support,
                                  BinaryMap &compilation_results)
      : dnn_support_(dnn_support), compilation_results_(compilation_results) {}

  void AddWorkspace(HloInstruction &hlo, int64_t workspace_size) {
    if (workspace_size == 0) {
      return;
    }
    VLOG(4) << "Applying workspace size " << workspace_size << " to "
            << hlo.ToString();
    Shape *shape = hlo.mutable_shape();
    shape->mutable_tuple_shapes()->back().set_dimensions(0, workspace_size);
  }

  absl::Status HandleCustomCall(HloInstruction *hlo) override {
    if (!IsCustomCallTofMHA(*hlo)) {
      return absl::OkStatus();
    }

    TF_ASSIGN_OR_RETURN(const std::string fingerprint_without_workspace,
                        FingerprintWithBackendConfig<GpuBackendConfig>(*hlo));
    auto workspace_size_it =
        workspace_sizes_.find(fingerprint_without_workspace);
    if (workspace_size_it == workspace_sizes_.cend()) {
      TF_ASSIGN_OR_RETURN(
          se::gpu::CudnnGraph graph,
          HloCustomCallToCuDnnGraph(dnn_support_,
                                    DynCast<HloCustomCallInstruction>(hlo)));

      const int64_t workspace_size = graph.Graph().get_workspace_size();
      workspace_sizes_.insert(workspace_size_it,
                              {fingerprint_without_workspace, workspace_size});
      AddWorkspace(*hlo, workspace_size);

      std::vector<uint8_t> serialized_graph;
      RETURN_IF_CUDNN_FRONTEND_ERROR(graph.Graph().serialize(serialized_graph));
      // Compute a new fingerprint with a potential workspace for the
      // compilation results to match a fingerprint computed by the emitter.
      TF_ASSIGN_OR_RETURN(const std::string fingerprint_with_workspace,
                          FingerprintWithBackendConfig<GpuBackendConfig>(*hlo));
      compilation_results_[fingerprint_with_workspace] =
          std::string(reinterpret_cast<char *>(serialized_graph.data()),
                      serialized_graph.size());
    } else {
      VLOG(4) << "Cache hit.";
      AddWorkspace(*hlo, workspace_size_it->second);
    }

    MarkAsChanged();
    return absl::OkStatus();
  }

 private:
  se::dnn::DnnSupport &dnn_support_;
  BinaryMap &compilation_results_;
  absl::flat_hash_map<std::string, int64_t> workspace_sizes_;
};

}  // namespace

absl::StatusOr<bool> CuDnnCustomCallCompiler::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  XLA_SCOPED_LOGGING_TIMER_LEVEL("cuDNN custom call compiler", 8);
  return CuDnnCustomCallVisitor(dnn_support_, compilation_results_)
      .RunOnModule(module, execution_threads);
}

}  // namespace gpu
}  // namespace xla
