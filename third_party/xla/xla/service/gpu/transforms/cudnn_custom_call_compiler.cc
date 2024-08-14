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

using se::dnn::DataType;
using se::dnn::MatmulTensorDescriptor;
using se::dnn::TensorDescriptor;

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

    const Shape &lhs_bmm1_shape = custom_call->operand(0)->shape();
    const Shape &rhs_bmm1_shape = custom_call->operand(1)->shape();
    const Shape &rhs_bmm2_shape = custom_call->operand(2)->shape();
    const Shape &intermediate_lhs_bmm2_shape = intermediate_tensor_shape;
    const Shape &output_shape = output_shapes[0];
    const DotDimensionNumbers &bmm1_dnums = config.bmm1_dot_dimension_numbers();
    const DotDimensionNumbers &bmm2_dnums = config.bmm2_dot_dimension_numbers();

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

    MatmulTensorDescriptor lhs_bmm1 =
        MatmulTensorDescriptor::For(lhs_bmm1_type, lhs_bmm1_shape.dimensions(),
                                    lhs_bmm1_shape.layout().minor_to_major(),
                                    bmm1_dnums.lhs_batch_dimensions(),
                                    bmm1_dnums.lhs_contracting_dimensions());
    MatmulTensorDescriptor rhs_bmm1 =
        MatmulTensorDescriptor::For(rhs_bmm1_type, rhs_bmm1_shape.dimensions(),
                                    rhs_bmm1_shape.layout().minor_to_major(),
                                    bmm1_dnums.rhs_batch_dimensions(),
                                    bmm1_dnums.rhs_contracting_dimensions());
    MatmulTensorDescriptor rhs_bmm2 =
        MatmulTensorDescriptor::For(rhs_bmm2_type, rhs_bmm2_shape.dimensions(),
                                    rhs_bmm2_shape.layout().minor_to_major(),
                                    bmm2_dnums.rhs_batch_dimensions(),
                                    bmm2_dnums.rhs_contracting_dimensions());
    MatmulTensorDescriptor intermediate_lhs_bmm2 = MatmulTensorDescriptor::For(
        lhs_bmm2_type, intermediate_lhs_bmm2_shape.dimensions(),
        intermediate_lhs_bmm2_shape.layout().minor_to_major(),
        bmm2_dnums.lhs_batch_dimensions(),
        bmm2_dnums.lhs_contracting_dimensions());
    TensorDescriptor output =
        TensorDescriptor::For(output_type, output_shape.dimensions(),
                              output_shape.layout().minor_to_major());

    std::optional<se::dnn::TensorDescriptor> activation;
    if (output_shapes.size() > 1) {
      const Shape &activation_shape = output_shapes.back();
      // Generally, activation should have same type as output, but set it
      // explicityly just to be safe.
      TF_ASSIGN_OR_RETURN(
          DataType activation_type,
          GetDNNDataTypeFromPrimitiveType(activation_shape.element_type()));
      activation =
          TensorDescriptor::For(activation_type, activation_shape.dimensions(),
                                activation_shape.layout().minor_to_major());
    }

    std::optional<se::dnn::TensorDescriptor> bias;
    if (bias_shape.has_value()) {
      TF_ASSIGN_OR_RETURN(DataType bias_type, GetDNNDataTypeFromPrimitiveType(
                                                  bias_shape->element_type()));
      bias = TensorDescriptor::For(bias_type, bias_shape->dimensions(),
                                   bias_shape->layout().minor_to_major());
    }

    const double dropout_rate = config.dropout_rate();

    TF_ASSIGN_OR_RETURN(CudnnfMHAMaskKind cudnn_mask_type,
                        AsCudnnFmhaMaskKind(config.mask_type()));
    TF_ASSIGN_OR_RETURN(
        se::dnn::FMHAMaskKind dnn_mask_type,
        GetDNNFmhaMaskKindFromCudnnFmhaMaskKind(cudnn_mask_type));

    TF_ASSIGN_OR_RETURN(
        se::gpu::CudnnGraph graph,
        se::gpu::GetCudnnFlashAttentionOperationGraph(
            dnn_support, lhs_bmm1, rhs_bmm1, rhs_bmm2, output, bias, activation,
            static_cast<float>(config.fmha_scale()), dropout_rate > 0.0,
            dropout_rate, dnn_mask_type));
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
    std::optional<Shape> d_bias_shape;
    bool has_dbias = custom_call->shape().tuple_shapes().size() == 5;
    if (has_dbias) {
      d_bias_shape =
          ShapeUtil::GetSubshape(custom_call->shape(), {output_index++});
    }
    // The last one is the workspace.
    TF_RET_CHECK(output_index ==
                 custom_call->shape().tuple_shapes().size() - 1);

    const DebugOptions &debug_options =
        custom_call->GetModule()->config().debug_options();
    bool force_deterministic =
        debug_options.xla_gpu_deterministic_ops() ||
        debug_options.xla_gpu_exclude_nondeterministic_ops();
    config.set_force_deterministic(force_deterministic);
    TF_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_config));

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

    const DotDimensionNumbers &bmm1_grad_gemm1_dnums =
        config.bmm1_grad_gemm1_dot_dimension_numbers();
    const DotDimensionNumbers &bmm1_grad_gemm2_dnums =
        config.bmm1_grad_gemm2_dot_dimension_numbers();
    const DotDimensionNumbers &bmm2_grad_gemm1_dnums =
        config.bmm2_grad_gemm1_dot_dimension_numbers();
    const DotDimensionNumbers &bmm2_grad_gemm2_dnums =
        config.bmm2_grad_gemm2_dot_dimension_numbers();

    MatmulTensorDescriptor bmm1_grad_gemm1_rhs = MatmulTensorDescriptor::For(
        bmm1_grad_gemm1_rhs_type, bmm1_grad_gemm1_rhs_shape.dimensions(),
        bmm1_grad_gemm1_rhs_shape.layout().minor_to_major(),
        bmm1_grad_gemm1_dnums.rhs_batch_dimensions(),
        bmm1_grad_gemm1_dnums.rhs_contracting_dimensions());

    MatmulTensorDescriptor bmm1_grad_gemm2_rhs = MatmulTensorDescriptor::For(
        bmm1_grad_gemm2_rhs_type, bmm1_grad_gemm2_rhs_shape.dimensions(),
        bmm1_grad_gemm2_rhs_shape.layout().minor_to_major(),
        bmm1_grad_gemm2_dnums.rhs_batch_dimensions(),
        bmm1_grad_gemm2_dnums.rhs_contracting_dimensions());

    MatmulTensorDescriptor bmm2_grad_gemm1_lhs = MatmulTensorDescriptor::For(
        bmm2_grad_gemm1_lhs_type, bmm2_grad_gemm1_lhs_shape.dimensions(),
        bmm2_grad_gemm1_lhs_shape.layout().minor_to_major(),
        bmm2_grad_gemm1_dnums.lhs_batch_dimensions(),
        bmm2_grad_gemm1_dnums.lhs_contracting_dimensions());

    MatmulTensorDescriptor d_output = MatmulTensorDescriptor::For(
        d_output_type, d_output_shape.dimensions(),
        d_output_shape.layout().minor_to_major(),
        bmm2_grad_gemm1_dnums.rhs_batch_dimensions(),
        bmm2_grad_gemm1_dnums.rhs_contracting_dimensions());

    MatmulTensorDescriptor bmm2_grad_gemm2_rhs = MatmulTensorDescriptor::For(
        bmm2_grad_gemm2_rhs_type, bmm2_grad_gemm2_rhs_shape.dimensions(),
        bmm2_grad_gemm2_rhs_shape.layout().minor_to_major(),
        bmm2_grad_gemm2_dnums.rhs_batch_dimensions(),
        bmm2_grad_gemm2_dnums
            .rhs_contracting_dimensions());  // FMHA TODO: transpose here?

    TensorDescriptor d_bmm1_lhs =
        TensorDescriptor::For(d_bmm1_lhs_type, d_bmm1_lhs_shape.dimensions(),
                              d_bmm1_lhs_shape.layout().minor_to_major());
    TensorDescriptor d_bmm1_rhs =
        TensorDescriptor::For(d_bmm1_rhs_type, d_bmm1_rhs_shape.dimensions(),
                              d_bmm1_rhs_shape.layout().minor_to_major());
    TensorDescriptor d_bmm2_rhs =
        TensorDescriptor::For(d_bmm2_rhs_type, d_bmm2_rhs_shape.dimensions(),
                              d_bmm2_rhs_shape.layout().minor_to_major());

    std::optional<se::dnn::TensorDescriptor> d_bias;
    if (d_bias_shape.has_value()) {
      // Get DNN dtype from primtive types
      TF_ASSIGN_OR_RETURN(
          DataType d_bias_type,
          GetDNNDataTypeFromPrimitiveType(d_bias_shape->element_type()));
      d_bias = TensorDescriptor::For(d_bias_type, d_bias_shape->dimensions(),
                                     d_bias_shape->layout().minor_to_major());
    }

    std::optional<se::dnn::TensorDescriptor> mask;
    if (mask_shape.has_value()) {
      TF_ASSIGN_OR_RETURN(DataType mask_type, GetDNNDataTypeFromPrimitiveType(
                                                  mask_shape->element_type()));
      mask = TensorDescriptor::For(mask_type, mask_shape->dimensions(),
                                   mask_shape->layout().minor_to_major());
    }

    std::optional<se::dnn::TensorDescriptor> fwd_output;
    if (fwd_output_shape.has_value()) {
      TF_ASSIGN_OR_RETURN(
          DataType fwd_output_type,
          GetDNNDataTypeFromPrimitiveType(fwd_output_shape->element_type()));
      fwd_output =
          TensorDescriptor::For(fwd_output_type, fwd_output_shape->dimensions(),
                                fwd_output_shape->layout().minor_to_major());
    }

    std::optional<se::dnn::TensorDescriptor> bias;
    if (bias_shape.has_value()) {
      TF_ASSIGN_OR_RETURN(DataType bias_type, GetDNNDataTypeFromPrimitiveType(
                                                  bias_shape->element_type()));
      bias = TensorDescriptor::For(bias_type, bias_shape->dimensions(),
                                   bias_shape->layout().minor_to_major());
    }

    const CudnnfMHABackendConfig &backend_config = config;
    const double dropout_rate = backend_config.dropout_rate();

    TF_ASSIGN_OR_RETURN(CudnnfMHAMaskKind cudnn_mask_type,
                        AsCudnnFmhaMaskKind(config.mask_type()));
    TF_ASSIGN_OR_RETURN(
        se::dnn::FMHAMaskKind dnn_mask_type,
        GetDNNFmhaMaskKindFromCudnnFmhaMaskKind(cudnn_mask_type));

    TF_ASSIGN_OR_RETURN(
        se::gpu::CudnnGraph graph,
        se::gpu::GetCudnnFlashAttentionBackwardOperationGraph(
            dnn_support, bmm1_grad_gemm1_rhs, bmm1_grad_gemm2_rhs,
            bmm2_grad_gemm1_lhs, bmm2_grad_gemm2_rhs, d_output, d_bmm1_lhs,
            d_bmm1_rhs, d_bmm2_rhs, bias, dropout_rate, backend_config.seed(),
            backend_config.fmha_scale(), dropout_rate > 0.0,
            bias != std::nullopt, dnn_mask_type, force_deterministic));
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
