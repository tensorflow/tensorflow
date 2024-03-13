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

#include "xla/service/gpu/cudnn_fusion_compiler.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/stream_executor/cuda/cuda_dnn.h"
#include "xla/stream_executor/cuda/cudnn_frontend_helpers.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

namespace {

namespace fe = cudnn_frontend;
namespace graph = fe::graph;

inline std::optional<fe::PointwiseMode_t> GetElementwiseMode(
    const HloInstruction& instruction) {
  const HloOpcode opcode = instruction.opcode();
  using m = fe::PointwiseMode_t;
  switch (opcode) {
    case HloOpcode::kAdd:
      return m::ADD;
    case HloOpcode::kConvert:
      return m::IDENTITY;
    case HloOpcode::kDivide:
      return m::DIV;
    case HloOpcode::kMultiply:
      return m::MUL;
    case HloOpcode::kNegate:
      return m::NEG;
    case HloOpcode::kSubtract:
      return m::SUB;
    default:
      return std::nullopt;
  }
}

inline std::optional<fe::DataType_t> ToCudnnDataType(const PrimitiveType type) {
  using t = fe::DataType_t;
  switch (type) {
    case PrimitiveType::F32:
      return t::FLOAT;
    case PrimitiveType::F16:
      return t::HALF;
    case PrimitiveType::BF16:
      return t::BFLOAT16;
    case PrimitiveType::S32:
      return t::INT32;
    case PrimitiveType::S8:
      return t::INT8;
    default:
      return std::nullopt;
  }
}

// Extracts dimensions and strides from HLO tensors in the format expected by
// cuDNN.
class GemmDimensionAdapter {
  explicit GemmDimensionAdapter(const HloDotInstruction& dot,
                                TritonFusionAnalysis analysis)
      : analysis_(std::move(analysis)), dot_(dot){};

 public:
  const TritonFusionAnalysis analysis_;

  static absl::StatusOr<std::optional<GemmDimensionAdapter>> Create(
      const HloComputation& computation) {
    const HloInstruction* maybe_dot =
        hlo_query::GetFirstInstructionWithOpcode(computation, HloOpcode::kDot);
    if (maybe_dot == nullptr) {
      VLOG(3) << "Not a GEMM fusion.";
      return std::nullopt;
    }
    const HloDotInstruction* dot = DynCast<HloDotInstruction>(
        hlo_query::GetFirstInstructionWithOpcode(computation, HloOpcode::kDot));
    if (absl::c_any_of(dot->precision_config().operand_precision(),
                       [](int x) { return x != PrecisionConfig::DEFAULT; })) {
      VLOG(3) << "Non-default precision is not supported.";
      return std::nullopt;
    }
    TF_ASSIGN_OR_RETURN(auto analysis,
                        TritonFusionAnalysis::Execute(computation));
    return GemmDimensionAdapter{*dot, std::move(analysis)};
  }

  bool DimensionsAndStrides(const HloInstruction& hlo,
                            const TritonFusionAnalysis::Scope scope,
                            std::vector<int64_t>& dimensions,
                            std::vector<int64_t>& strides) {
    const DotDimensionNumbers& dims = dot_.dot_dimension_numbers();
    // GEMM fusions require a specific canonical order of dimensions.
    std::vector<int64_t> dim_indices;
    switch (scope) {
      case TritonFusionAnalysis::Scope::LHS:
        dim_indices = {dims.lhs_batch_dimensions().empty()
                           ? -1
                           : dims.lhs_batch_dimensions(0),
                       GetNonContractingDims(dot_.operand(0)->shape(),
                                             dims.lhs_batch_dimensions(),
                                             dims.lhs_contracting_dimensions())
                           .value()[0],
                       dims.lhs_contracting_dimensions(0)};
        break;
      case TritonFusionAnalysis::Scope::RHS:
        dim_indices = {dims.rhs_batch_dimensions().empty()
                           ? -1
                           : dims.rhs_batch_dimensions(0),
                       dims.rhs_contracting_dimensions(0),
                       GetNonContractingDims(dot_.operand(1)->shape(),
                                             dims.rhs_batch_dimensions(),
                                             dims.rhs_contracting_dimensions())
                           .value()[0]};
        break;
      case TritonFusionAnalysis::Scope::OUTPUT:
        dim_indices = {dims.lhs_batch_dimensions().empty() ? -1 : 0,
                       dot_.shape().rank() - 2, dot_.shape().rank() - 1};
        break;
      case TritonFusionAnalysis::Scope::META:
        LOG(FATAL) << "Unsupported scope.";
    }
    dimensions.reserve(dim_indices.size());
    strides.reserve(dim_indices.size());
    for (const int index : dim_indices) {
      const auto* spec = analysis_.IterSpec(scope, &hlo, index);
      if (spec == nullptr) {
        dimensions.push_back(1);
        strides.push_back(strides.empty() ? 1 : strides.back());
        continue;
      } else {
        if (spec->size() != 1) {
          return false;
        }
        dimensions.push_back(spec->front().count);
        strides.push_back(spec->front().stride);
      }
    }
    return true;
  }

 private:
  const HloDotInstruction& dot_;
};

// Traverses fusion computations and creates cuDNN graphs out of them.
absl::StatusOr<std::optional<se::gpu::CudnnGraph>> HloFusionToCuDnnGraph(
    const HloFusionInstruction& fusion) {
  const HloComputation& computation = *fusion.fused_instructions_computation();
  VLOG(5) << fusion.ToString();
  VLOG(5) << computation.ToString();
  graph::Graph graph;
  std::vector<HloInstruction*> instructions =
      computation.MakeInstructionPostOrder();
  absl::flat_hash_map<const HloInstruction*,
                      std::shared_ptr<graph::Tensor_attributes>>
      hlo_to_cudnn;
  TF_ASSIGN_OR_RETURN(std::optional<GemmDimensionAdapter> adapter,
                      GemmDimensionAdapter::Create(computation));
  if (!adapter.has_value()) {
    return std::nullopt;
  }
  auto add_parameter = [&](const HloInstruction& parameter,
                           std::vector<int64_t>& dimensions,
                           std::vector<int64_t> strides) {
    const std::optional<fe::DataType_t> data_type =
        ToCudnnDataType(parameter.shape().element_type());
    if (!data_type.has_value()) {
      VLOG(3) << "Unsupported data type.";
      return false;
    }
    hlo_to_cudnn[&parameter] = graph.tensor(
        graph::Tensor_attributes()
            .set_dim(dimensions)
            .set_stride(strides)
            .set_data_type(*data_type)
            .set_uid(se::gpu::CuDnnTensorUID(parameter.parameter_number())));
    return true;
  };
  for (const TritonFusionAnalysis::Scope scope :
       {TritonFusionAnalysis::Scope::LHS, TritonFusionAnalysis::Scope::RHS,
        TritonFusionAnalysis::Scope::OUTPUT}) {
    for (const HloInstruction* parameter :
         adapter->analysis_.ScopeParameters(scope)) {
      std::vector<int64_t> dimensions;
      std::vector<int64_t> strides;
      if (!adapter->DimensionsAndStrides(*parameter, scope, dimensions,
                                         strides)) {
        VLOG(3) << "Unsupported dimensions.";
        return std::nullopt;
      }
      if (!add_parameter(*parameter, dimensions, strides)) {
        return std::nullopt;
      }
    }
  }

  for (const HloInstruction* hlo : instructions) {
    VLOG(5) << hlo->ToShortString();
    auto operand = [&hlo_to_cudnn, &hlo](int i) {
      return hlo_to_cudnn[hlo->operand(i)];
    };
    if (hlo->opcode() == HloOpcode::kParameter) {
      CHECK(hlo_to_cudnn.contains(hlo));
      continue;
    } else if (hlo->opcode() == HloOpcode::kReshape ||
               hlo->opcode() == HloOpcode::kBitcast ||
               hlo->opcode() == HloOpcode::kTranspose ||
               hlo->opcode() == HloOpcode::kCopy) {
      // All these are accounted for separately as transformations of strides.
      hlo_to_cudnn[hlo] = operand(0);
    } else if (hlo->IsElementwise()) {
      const auto mode = GetElementwiseMode(*hlo);
      if (!mode.has_value()) {
        VLOG(3) << "Unsupported elementwise operation.";
        return std::nullopt;
      }
      const auto compute_dtype =
          (primitive_util::IsIntegralType(hlo->shape().element_type()))
              ? fe::DataType_t::INT32
              : fe::DataType_t::FLOAT;
      const auto attrs = graph::Pointwise_attributes()
                             .set_mode(mode.value())
                             .set_compute_data_type(compute_dtype);
      if (hlo->operand_count() == 1) {
        hlo_to_cudnn[hlo] = graph.pointwise(operand(0), attrs);
      } else if (hlo->operand_count() == 2) {
        hlo_to_cudnn[hlo] = graph.pointwise(operand(0), operand(1), attrs);
      } else if (hlo->operand_count() == 3) {
        hlo_to_cudnn[hlo] =
            graph.pointwise(operand(0), operand(1), operand(2), attrs);
      } else {
        VLOG(3) << "Unimplemented elementwise operation.";
        return std::nullopt;
      }
    } else if (hlo->opcode() == HloOpcode::kDot) {
      const auto compute_dtype =
          (primitive_util::IsIntegralType(hlo->shape().element_type()))
              ? fe::DataType_t::INT32
              : fe::DataType_t::FLOAT;
      hlo_to_cudnn[hlo] = graph.matmul(
          operand(0), operand(1),
          graph::Matmul_attributes().set_compute_data_type(compute_dtype));
    } else {
      VLOG(3) << "Unimplemented operation.";
      return std::nullopt;
    }
    if (hlo_to_cudnn[hlo] == nullptr) {
      VLOG(3) << "Creation of the operation failed.";
      return std::nullopt;
    }
    const auto data_type = ToCudnnDataType(hlo->shape().element_type());
    if (!data_type.has_value()) {
      VLOG(3) << "Unimplemented data type: " << hlo->shape().element_type();
      return std::nullopt;
    }
    hlo_to_cudnn[hlo]->set_data_type(data_type.value());
  }
  const HloInstruction* output = instructions.back();
  if (instructions.back()->shape().IsTuple()) {
    output = instructions.back()->operand(0);
  }
  std::vector<int64_t> dimensions;
  std::vector<int64_t> strides;
  if (!adapter->DimensionsAndStrides(
          *output, TritonFusionAnalysis::Scope::OUTPUT, dimensions, strides)) {
    VLOG(3) << "Unsupported dimensions.";
    return std::nullopt;
  }
  hlo_to_cudnn[output]
      ->set_output(true)
      .set_dim(dimensions)
      .set_stride(strides)
      .set_uid(se::gpu::CuDnnTensorUID(fusion.operand_count()));
  if (cudnn_frontend::error_t result = graph.validate(); result.is_bad()) {
    VLOG(3) << result.get_message();
    return std::nullopt;
  }

  return se::gpu::CudnnGraph(std::move(graph));
}

// Creates a cuDNN graph, queries cuDNN whether it is supported.
absl::StatusOr<se::gpu::CudnnGraph> PrepareGraph(
    const HloFusionInstruction& hlo, se::Stream& stream) {
  TF_ASSIGN_OR_RETURN(std::optional<se::gpu::CudnnGraph> graph,
                      HloFusionToCuDnnGraph(hlo));
  if (!graph.has_value()) {
    return absl::InternalError("Construction of cuDNN graph failed.");
  }
  TF_ASSIGN_OR_RETURN(bool supported, graph->Prepare());
  if (!supported) {
    return absl::InternalError("cuDNN graph is not supported.");
  }
  return *graph;
}

class CuDnnFusionVisitor : public DfsHloRewriteVisitor {
 public:
  explicit CuDnnFusionVisitor(const AutotuneConfig& config) : config_(config) {}

  absl::Status HandleFusion(HloInstruction* hlo) override {
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        hlo->backend_config<GpuBackendConfig>());
    const auto& fusion_backend_config = gpu_config.fusion_backend_config();
    if (fusion_backend_config.kind() != kCuDnnFusionKind) {
      return absl::OkStatus();
    }
    int64_t plan_id = -1;
    if (fusion_backend_config.has_cudnn_fusion_config()) {
      if (fusion_backend_config.cudnn_fusion_config().has_serialized_graph()) {
        VLOG(4) << "Skipping already serialized " << hlo->ToShortString();
        return absl::OkStatus();
      }
      plan_id = fusion_backend_config.cudnn_fusion_config().plan_id();
    }

    VLOG(4) << "Processing " << hlo->ToString();
    VLOG(4) << "Plan ID: " << plan_id;

    const std::string cache_key =
        GetComputationFingerprint(hlo->fused_instructions_computation(), {});
    std::string& cache_entry = compilation_cache_[cache_key];
    if (cache_entry.empty()) {
      TF_ASSIGN_OR_RETURN(se::Stream * stream, config_.GetStream());

      TF_ASSIGN_OR_RETURN(
          se::gpu::CudnnGraph graph,
          PrepareGraph(*DynCast<HloFusionInstruction>(hlo), *stream));

      if (plan_id >= 0) {
        // Build single plan with given ID.
        if (plan_id >= graph.Graph().get_execution_plan_count()) {
          return absl::InternalError("cuDNN graph plan does not exist.");
        }
        TF_RETURN_IF_ERROR(graph.Build(plan_id));
      } else {
        // Build plans one by one till first successful when no plan_id was
        // provided.
        for (plan_id = 0; plan_id < graph.Graph().get_execution_plan_count();
             ++plan_id) {
          VLOG(7) << "Trying plan ID " << plan_id;
          if (graph.Build(plan_id).ok()) {
            VLOG(7) << "Successfully built plan ID " << plan_id;
            break;
          }
        }
        if (plan_id == graph.Graph().get_execution_plan_count()) {
          return absl::InternalError("No cuDNN plans can be built.");
        }
      }

      if (graph.Graph().get_workspace_size() != 0) {
        return absl::UnimplementedError(
            "Support of workspace allocation is not added yet.");
      }

      std::vector<uint8_t> serialized_graph;
      RETURN_IF_CUDNN_FRONTEND_ERROR(graph.Graph().serialize(serialized_graph));
      cache_entry = absl::CEscape(
          absl::string_view(reinterpret_cast<char*>(serialized_graph.data()),
                            serialized_graph.size()));
    } else {
      VLOG(4) << "Cache hit.";
    }
    auto cudnn_config = gpu_config.mutable_fusion_backend_config()
                            ->mutable_cudnn_fusion_config();
    cudnn_config->set_plan_id(plan_id);
    cudnn_config->set_serialized_graph(cache_entry);
    TF_RETURN_IF_ERROR(hlo->set_backend_config(gpu_config));

    MarkAsChanged();
    return absl::OkStatus();
  }

 private:
  AutotuneConfig config_;
  // <HLO computation fingerprint, serialized compiled cuDNN graph>.
  absl::flat_hash_map<std::string, std::string> compilation_cache_;
};

}  // namespace

absl::StatusOr<bool> CuDnnFusionCompiler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER("cuDNN fusion compiler");
  return CuDnnFusionVisitor(config_).RunOnModule(module, execution_threads);
}

int CuDnnFusionCompiler::GetAvailablePlanCount(
    const HloFusionInstruction& hlo) const {
  se::Stream& stream = *config_.GetStream().value();
  auto graph = PrepareGraph(hlo, stream);
  if (!graph.ok()) {
    return 0;
  }
  constexpr int64_t kMaxPlans = 10;
  return std::min(graph->Graph().get_execution_plan_count(), kMaxPlans);
}

}  // namespace gpu
}  // namespace xla
