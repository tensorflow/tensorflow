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

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
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
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_dnn.h"
#include "xla/stream_executor/cuda/cudnn_frontend_helpers.h"
#include "xla/stream_executor/dnn.h"
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

using se::dnn::DataType;
using se::dnn::MatmulTensorDescriptor;
using se::dnn::TensorDescriptor;

absl::StatusOr<TensorDescriptor> TensorDescriptorFor(const Shape &shape) {
  TF_ASSIGN_OR_RETURN(const DataType type,
                      GetDNNDataTypeFromPrimitiveType(shape.element_type()));
  return TensorDescriptor::For(type, shape.dimensions(),
                               shape.layout().minor_to_major());
}

enum Side { LHS, RHS };

absl::StatusOr<MatmulTensorDescriptor> MatmulTensorDescriptorFor(
    const Shape &shape, const DotDimensionNumbers &dnums, const Side side) {
  TF_ASSIGN_OR_RETURN(const DataType type,
                      GetDNNDataTypeFromPrimitiveType(shape.element_type()));
  return MatmulTensorDescriptor::For(
      type, shape.dimensions(), shape.layout().minor_to_major(),
      (side == LHS) ? dnums.lhs_batch_dimensions()
                    : dnums.rhs_batch_dimensions(),
      (side == LHS) ? dnums.lhs_contracting_dimensions()
                    : dnums.rhs_contracting_dimensions());
}

absl::StatusOr<se::gpu::CudnnGraph> HloCustomCallToCuDnnGraph(
    se::dnn::DnnSupport &dnn_support, HloCustomCallInstruction *custom_call) {
  if (IsFwdCustomCallTofMHA(*custom_call)) {
    TF_ASSIGN_OR_RETURN(const xla::gpu::CudnnfMHAKind kind,
                        xla::gpu::GetCudnnfMHAKind(custom_call));
    TF_ASSIGN_OR_RETURN(
        const auto gpu_config,
        custom_call->backend_config<xla::gpu::GpuBackendConfig>());
    const xla::gpu::CudnnfMHABackendConfig &config =
        gpu_config.cudnn_fmha_backend_config();

    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor lhs_bmm1,
        MatmulTensorDescriptorFor(custom_call->operand(0)->shape(),
                                  config.bmm1_dot_dimension_numbers(), LHS));
    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor rhs_bmm1,
        MatmulTensorDescriptorFor(custom_call->operand(1)->shape(),
                                  config.bmm1_dot_dimension_numbers(), RHS));
    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor rhs_bmm2,
        MatmulTensorDescriptorFor(custom_call->operand(2)->shape(),
                                  config.bmm2_dot_dimension_numbers(), RHS));
    TF_ASSIGN_OR_RETURN(
        TensorDescriptor output,
        TensorDescriptorFor(ShapeUtil::GetSubshape(custom_call->shape(), {0})));

    std::optional<se::dnn::TensorDescriptor> activation;
    const bool has_activation =
        xla::ShapeUtil::TupleElementCount(custom_call->shape()) == 3;
    if (has_activation) {
      TF_ASSIGN_OR_RETURN(
          activation, TensorDescriptorFor(
                          ShapeUtil::GetSubshape(custom_call->shape(), {1})));
    }

    std::optional<se::dnn::TensorDescriptor> bias;
    if (kind == CudnnfMHAKind::kScaleBiasSoftmax ||
        kind == CudnnfMHAKind::kScaleBiasSoftmaxDropout) {
      const HloInstruction &bias_hlo = *custom_call->operand(3);
      TF_ASSIGN_OR_RETURN(bias, TensorDescriptorFor(bias_hlo.shape()));
    }

    const double dropout_rate = config.dropout_rate();

    TF_ASSIGN_OR_RETURN(CudnnfMHAMaskKind cudnn_mask_type,
                        AsCudnnFmhaMaskKind(config.mask_type()));
    TF_ASSIGN_OR_RETURN(
        se::dnn::FMHAMaskKind dnn_mask_type,
        GetDNNFmhaMaskKindFromCudnnFmhaMaskKind(cudnn_mask_type));

    const int sliding_window_length = config.sliding_window_length();
    TF_ASSIGN_OR_RETURN(
        se::gpu::CudnnGraph graph,
        se::gpu::GetCudnnFlashAttentionOperationGraph(
            dnn_support, lhs_bmm1, rhs_bmm1, rhs_bmm2, output, bias, activation,
            static_cast<float>(config.fmha_scale()), dropout_rate > 0.0,
            dropout_rate, dnn_mask_type, sliding_window_length));
    return graph;
  } else if (IsFwdCustomCallTofMHAF8(*custom_call)) {
    TF_ASSIGN_OR_RETURN(
        const auto gpu_config,
        custom_call->backend_config<xla::gpu::GpuBackendConfig>());
    const xla::gpu::CudnnfMHABackendConfig &config =
        gpu_config.cudnn_fmha_backend_config();
    Shape intermediate_tensor_shape(config.intermediate_tensor_shape());

    TF_ASSIGN_OR_RETURN(CudnnfMHAMaskKind cudnn_mask_type,
                        AsCudnnFmhaMaskKind(config.mask_type()));
    TF_ASSIGN_OR_RETURN(
        se::dnn::FMHAMaskKind dnn_mask_type,
        GetDNNFmhaMaskKindFromCudnnFmhaMaskKind(cudnn_mask_type));
    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor lhs_bmm1,
        MatmulTensorDescriptorFor(custom_call->operand(0)->shape(),
                                  config.bmm1_dot_dimension_numbers(), LHS));
    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor rhs_bmm1,
        MatmulTensorDescriptorFor(custom_call->operand(1)->shape(),
                                  config.bmm1_dot_dimension_numbers(), RHS));
    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor rhs_bmm2,
        MatmulTensorDescriptorFor(custom_call->operand(2)->shape(),
                                  config.bmm2_dot_dimension_numbers(), RHS));
    TF_ASSIGN_OR_RETURN(
        TensorDescriptor output,
        TensorDescriptorFor(ShapeUtil::GetSubshape(custom_call->shape(), {0})));

    std::optional<se::dnn::TensorDescriptor> activation;
    bool has_activation =
        xla::ShapeUtil::TupleElementCount(custom_call->shape()) == 5;
    if (has_activation) {
      TF_ASSIGN_OR_RETURN(
          activation, TensorDescriptorFor(
                          ShapeUtil::GetSubshape(custom_call->shape(), {3})));
    }
    TF_ASSIGN_OR_RETURN(
        se::gpu::CudnnGraph graph,
        se::gpu::GetCudnnFlashAttentionF8OperationGraph(
            dnn_support, lhs_bmm1, rhs_bmm1, rhs_bmm2, output, activation,
            static_cast<float>(config.fmha_scale()), dnn_mask_type));
    return graph;
  } else if (IsBwdCustomCallTofMHA(*custom_call)) {
    TF_ASSIGN_OR_RETURN(
        auto gpu_config,
        custom_call->backend_config<xla::gpu::GpuBackendConfig>());
    xla::gpu::CudnnfMHABackendConfig &config =
        *gpu_config.mutable_cudnn_fmha_backend_config();

    int input_index = 0;
    const Shape &bmm1_grad_gemm1_rhs_shape =
        custom_call->operand(input_index++)->shape();
    const Shape &bmm1_grad_gemm2_rhs_shape =
        custom_call->operand(input_index++)->shape();
    const Shape &bmm2_grad_gemm2_rhs_shape =
        custom_call->operand(input_index++)->shape();
    const Shape bmm2_grad_gemm1_lhs_shape(config.intermediate_tensor_shape());
    ++input_index;
    const Shape &d_output_shape = custom_call->operand(input_index++)->shape();

    TF_ASSIGN_OR_RETURN(const CudnnfMHAKind kind,
                        GetCudnnfMHAKind(custom_call));

    bool has_bias = (kind == CudnnfMHAKind::kBackwardScaleBiasSoftmax ||
                     kind == CudnnfMHAKind::kBackwardScaleBiasSoftmaxDropout);
    std::optional<Shape> bias_shape;
    if (has_bias) {
      bias_shape = custom_call->operand(input_index++)->shape();
    }

    // Unused fwd_output_shape
    ++input_index;

    if (config.mask_type() == xla::gpu::CudnnfMHABackendConfig::PADDING ||
        config.mask_type() ==
            xla::gpu::CudnnfMHABackendConfig::PADDING_CAUSAL) {
      // skip q_seqlen and kv_seqlen
      input_index += 2;
    }
    TF_RET_CHECK(input_index == custom_call->operand_count());

    int output_index = 0;
    const Shape &d_bmm1_lhs_shape =
        ShapeUtil::GetSubshape(custom_call->shape(), {output_index++});
    const Shape &d_bmm1_rhs_shape =
        ShapeUtil::GetSubshape(custom_call->shape(), {output_index++});
    const Shape &d_bmm2_rhs_shape =
        ShapeUtil::GetSubshape(custom_call->shape(), {output_index++});
    bool has_dbias = custom_call->shape().tuple_shapes().size() == 5;
    if (has_dbias) {
      ++output_index;
    }
    // The last one is the workspace.
    TF_RET_CHECK(output_index ==
                 custom_call->shape().tuple_shapes().size() - 1);

    const bool force_deterministic =
        RequireDeterminism(custom_call->GetModule()->config());
    config.set_force_deterministic(force_deterministic);
    TF_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_config));

    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor bmm1_grad_gemm1_rhs,
        MatmulTensorDescriptorFor(
            bmm1_grad_gemm1_rhs_shape,
            config.bmm1_grad_gemm1_dot_dimension_numbers(), RHS));
    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor bmm1_grad_gemm2_rhs,
        MatmulTensorDescriptorFor(
            bmm1_grad_gemm2_rhs_shape,
            config.bmm1_grad_gemm2_dot_dimension_numbers(), RHS));
    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor bmm2_grad_gemm1_lhs,
        MatmulTensorDescriptorFor(
            bmm2_grad_gemm1_lhs_shape,
            config.bmm2_grad_gemm1_dot_dimension_numbers(), LHS));
    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor bmm2_grad_gemm2_rhs,
        MatmulTensorDescriptorFor(
            bmm2_grad_gemm2_rhs_shape,
            config.bmm2_grad_gemm2_dot_dimension_numbers(), RHS));
    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor d_output,
        MatmulTensorDescriptorFor(
            d_output_shape, config.bmm2_grad_gemm1_dot_dimension_numbers(),
            RHS));

    TF_ASSIGN_OR_RETURN(TensorDescriptor d_bmm1_lhs,
                        TensorDescriptorFor(d_bmm1_lhs_shape));
    TF_ASSIGN_OR_RETURN(TensorDescriptor d_bmm1_rhs,
                        TensorDescriptorFor(d_bmm1_rhs_shape));
    TF_ASSIGN_OR_RETURN(TensorDescriptor d_bmm2_rhs,
                        TensorDescriptorFor(d_bmm2_rhs_shape));

    std::optional<se::dnn::TensorDescriptor> bias;
    if (bias_shape.has_value()) {
      TF_ASSIGN_OR_RETURN(bias, TensorDescriptorFor(*bias_shape));
    }

    const double dropout_rate = config.dropout_rate();

    TF_ASSIGN_OR_RETURN(CudnnfMHAMaskKind cudnn_mask_type,
                        AsCudnnFmhaMaskKind(config.mask_type()));
    TF_ASSIGN_OR_RETURN(
        se::dnn::FMHAMaskKind dnn_mask_type,
        GetDNNFmhaMaskKindFromCudnnFmhaMaskKind(cudnn_mask_type));

    const int sliding_window_length = config.sliding_window_length();
    TF_ASSIGN_OR_RETURN(
        se::gpu::CudnnGraph graph,
        se::gpu::GetCudnnFlashAttentionBackwardOperationGraph(
            dnn_support, bmm1_grad_gemm1_rhs, bmm1_grad_gemm2_rhs,
            bmm2_grad_gemm1_lhs, bmm2_grad_gemm2_rhs, d_output, d_bmm1_lhs,
            d_bmm1_rhs, d_bmm2_rhs, bias, dropout_rate, config.seed(),
            config.fmha_scale(), dropout_rate > 0.0, bias != std::nullopt,
            dnn_mask_type, force_deterministic, sliding_window_length));
    return graph;
  } else {
    TF_ASSIGN_OR_RETURN(
        auto gpu_config,
        custom_call->backend_config<xla::gpu::GpuBackendConfig>());
    xla::gpu::CudnnfMHABackendConfig &config =
        *gpu_config.mutable_cudnn_fmha_backend_config();

    Shape bmm1_grad_gemm1_rhs_shape = custom_call->operand(0)->shape();
    Shape bmm1_grad_gemm2_rhs_shape = custom_call->operand(1)->shape();
    Shape bmm2_grad_gemm2_rhs_shape = custom_call->operand(2)->shape();

    Shape fwd_output_shape = custom_call->operand(3)->shape();
    Shape d_output_shape = custom_call->operand(4)->shape();

    Shape bmm2_grad_gemm1_lhs_shape(config.intermediate_tensor_shape());

    Shape d_bmm1_lhs_shape = ShapeUtil::GetSubshape(custom_call->shape(), {0});
    Shape d_bmm1_rhs_shape = ShapeUtil::GetSubshape(custom_call->shape(), {1});
    Shape d_bmm2_rhs_shape = ShapeUtil::GetSubshape(custom_call->shape(), {2});

    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor bmm1_grad_gemm1_rhs,
        MatmulTensorDescriptorFor(
            bmm1_grad_gemm1_rhs_shape,
            config.bmm1_grad_gemm1_dot_dimension_numbers(), RHS));
    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor bmm1_grad_gemm2_rhs,
        MatmulTensorDescriptorFor(
            bmm1_grad_gemm2_rhs_shape,
            config.bmm1_grad_gemm2_dot_dimension_numbers(), RHS));
    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor bmm2_grad_gemm1_lhs,
        MatmulTensorDescriptorFor(
            bmm2_grad_gemm1_lhs_shape,
            config.bmm2_grad_gemm1_dot_dimension_numbers(), LHS));
    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor bmm2_grad_gemm2_rhs,
        MatmulTensorDescriptorFor(
            bmm2_grad_gemm2_rhs_shape,
            config.bmm2_grad_gemm2_dot_dimension_numbers(), RHS));
    TF_ASSIGN_OR_RETURN(
        MatmulTensorDescriptor d_output,
        MatmulTensorDescriptorFor(
            d_output_shape, config.bmm2_grad_gemm1_dot_dimension_numbers(),
            RHS));

    TF_ASSIGN_OR_RETURN(TensorDescriptor d_bmm1_lhs,
                        TensorDescriptorFor(d_bmm1_lhs_shape));
    TF_ASSIGN_OR_RETURN(TensorDescriptor d_bmm1_rhs,
                        TensorDescriptorFor(d_bmm1_rhs_shape));
    TF_ASSIGN_OR_RETURN(TensorDescriptor d_bmm2_rhs,
                        TensorDescriptorFor(d_bmm2_rhs_shape));
    // 3 gradients, 4 amaxs and one workspace
    TF_RET_CHECK(8 == custom_call->shape().tuple_shapes().size());

    TF_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_config));

    TF_ASSIGN_OR_RETURN(CudnnfMHAMaskKind cudnn_mask_type,
                        AsCudnnFmhaMaskKind(config.mask_type()));
    TF_ASSIGN_OR_RETURN(
        se::dnn::FMHAMaskKind dnn_mask_type,
        GetDNNFmhaMaskKindFromCudnnFmhaMaskKind(cudnn_mask_type));
    TF_ASSIGN_OR_RETURN(
        se::gpu::CudnnGraph graph,
        se::gpu::GetCudnnFlashAttentionBackwardF8OperationGraph(
            dnn_support, bmm1_grad_gemm1_rhs, bmm1_grad_gemm2_rhs,
            bmm2_grad_gemm1_lhs, bmm2_grad_gemm2_rhs, d_output, d_bmm1_lhs,
            d_bmm1_rhs, d_bmm2_rhs, config.fmha_scale(), dnn_mask_type));
    return graph;
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
    if (!IsCustomCallTofMHA(*hlo) && !IsCustomCallTofMHAF8(*hlo)) {
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
