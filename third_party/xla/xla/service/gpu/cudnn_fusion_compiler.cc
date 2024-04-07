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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cudnn/cudnn_version.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
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
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

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
    case HloOpcode::kAbs:
      return m::ABS;
    case HloOpcode::kAdd:
      return m::ADD;
    case HloOpcode::kCompare:
      switch (instruction.comparison_direction()) {
        case Comparison::Direction::kEq:
          return m::CMP_EQ;
        case Comparison::Direction::kNe:
          return m::CMP_NEQ;
        case Comparison::Direction::kGe:
          return m::CMP_GE;
        case Comparison::Direction::kGt:
          return m::CMP_GT;
        case Comparison::Direction::kLe:
          return m::CMP_LE;
        case Comparison::Direction::kLt:
          return m::CMP_LT;
      }
      break;
    case HloOpcode::kConvert:
      return m::IDENTITY;
    case HloOpcode::kCos:
      return m::COS;
    case HloOpcode::kDivide:
      return m::DIV;
    case HloOpcode::kExp:
      return m::EXP;
    case HloOpcode::kLog:
      return m::LOG;
    case HloOpcode::kMaximum:
      return m::MAX;
    case HloOpcode::kMinimum:
      return m::MIN;
    case HloOpcode::kMultiply:
      return m::MUL;
    case HloOpcode::kNegate:
      return m::NEG;
    case HloOpcode::kPower:
      return m::POW;
    case HloOpcode::kRsqrt:
      return m::RSQRT;
#if CUDNN_VERSION >= 90100
    case HloOpcode::kSelect:
      return m::BINARY_SELECT;
#endif  // CUDNN_VERSION
    case HloOpcode::kSin:
      return m::SIN;
    case HloOpcode::kSqrt:
      return m::SQRT;
    case HloOpcode::kSubtract:
      return m::SUB;
    case HloOpcode::kTan:
      return m::TAN;
    case HloOpcode::kTanh:
      return m::TANH_FWD;
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
    case PrimitiveType::PRED:
      return t::INT8;
    default:
      return std::nullopt;
  }
}

inline std::optional<fe::DataType_t> GetComputeDataType(
    const PrimitiveType type) {
  fe::DataType_t compute_dtype = fe::DataType_t::FLOAT;
  if (primitive_util::IsIntegralType(type)) {
#if CUDNN_VERSION >= 90100
    compute_dtype = fe::DataType_t::INT32;
#else
    VLOG(3) << "Integer math requires cuDNN 9.1+.";
    return std::nullopt;
#endif  // CUDNN_VERSION
  }
  return compute_dtype;
}

int FusionLevel(const HloInstruction& hlo) {
  return hlo.GetModule()
      ->config()
      .debug_options()
      .xla_gpu_cudnn_gemm_fusion_level();
};

// Extracts dimensions and strides from HLO tensors in the format expected by
// cuDNN.
class GemmDimensionAdapter {
  explicit GemmDimensionAdapter(const HloDotInstruction& dot,
                                TritonFusionAnalysis analysis)
      : analysis_(std::move(analysis)), dot_(dot) {};

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
    constexpr int kBatchDimensionIndex = 0;
    constexpr int kOutputLHSNonContractingDimensionIndex = 1;
    std::vector<int64_t> dim_indices;
    int lhs_noncontracting_index = -1;
    switch (scope) {
      case TritonFusionAnalysis::Scope::LHS:
        lhs_noncontracting_index =
            GetNonContractingDims(dot_.operand(0)->shape(),
                                  dims.lhs_batch_dimensions(),
                                  dims.lhs_contracting_dimensions())
                .value()[0];
        dim_indices = {
            dims.lhs_batch_dimensions().empty() ? -1
                                                : dims.lhs_batch_dimensions(0),
            lhs_noncontracting_index, dims.lhs_contracting_dimensions(0)};
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
        lhs_noncontracting_index = dot_.shape().rank() - 2;
        dim_indices = {dims.lhs_batch_dimensions().empty() ? -1 : 0,
                       lhs_noncontracting_index, dot_.shape().rank() - 1};
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
        if (spec->size() == 1) {
          // The dimension is not split, nothing to do.
        } else if (spec->size() == 2) {
          if (FusionLevel(hlo) < 3) {
            return false;
          }
          if (!dims.lhs_batch_dimensions().empty()) {
            VLOG(8) << "Noncontracting dimension split is not compatible with "
                       "batch dimensions.";
            return false;
          }
          if (index != lhs_noncontracting_index) {
            VLOG(8) << "Only LHS noncontracting dimension can be split.";
            return false;
          }
          switch (scope) {
            case TritonFusionAnalysis::Scope::LHS:
              lhs_noncontracting_split = spec->back().count;
              break;
            case TritonFusionAnalysis::Scope::OUTPUT:
              if (lhs_noncontracting_split != spec->back().count) {
                VLOG(8) << "Output non-contracting dimension has to be split "
                           "the same way as the LHS input one if it is split.";
                return false;
              }
              break;
            default:
              VLOG(8) << "Only LHS noncontracting dimension can be split.";
              return false;
          }
          // Assign the major part of the noncontracting dimension to the
          // unused batch one.
          CHECK_EQ(dimensions[kBatchDimensionIndex], 1);
          dimensions[kBatchDimensionIndex] = spec->back().count;
          strides[kBatchDimensionIndex] = spec->back().stride;
        } else {
          VLOG(8) << "The dimension is split multiple times.";
          return false;
        }
        dimensions.push_back(spec->front().count);
        strides.push_back(spec->front().stride);
      }
    }
    if (lhs_noncontracting_split > 1 &&
        scope == TritonFusionAnalysis::Scope::OUTPUT &&
        dimensions[kBatchDimensionIndex] == 1) {
      // LHS input noncontracting dimension is split but the corresponding
      // output one is not. Assign part of the output one to the unused batch
      // dimension.
      dimensions[kBatchDimensionIndex] = lhs_noncontracting_split;
      dimensions[kOutputLHSNonContractingDimensionIndex] /=
          lhs_noncontracting_split;
      strides[kBatchDimensionIndex] =
          strides[kOutputLHSNonContractingDimensionIndex] *
          dimensions[kOutputLHSNonContractingDimensionIndex];
    }
    return true;
  }

 private:
  int64_t lhs_noncontracting_split = 1;
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
               hlo->opcode() == HloOpcode::kCopy ||
               (FusionLevel(fusion) >= 2 &&
                hlo->opcode() == HloOpcode::kBroadcast)) {
      // All these are accounted for separately as transformations of strides.
      hlo_to_cudnn[hlo] = operand(0);
    } else if (hlo->IsElementwise()) {
      const auto mode = GetElementwiseMode(*hlo);
      if (!mode.has_value()) {
        VLOG(3) << "Unsupported elementwise operation.";
        return std::nullopt;
      }
      const auto compute_dtype =
          GetComputeDataType(hlo->shape().element_type());
      if (!compute_dtype.has_value()) {
        return std::nullopt;
      }
      const auto attrs = graph::Pointwise_attributes()
                             .set_mode(mode.value())
                             .set_compute_data_type(compute_dtype.value());
      if (hlo->operand_count() == 1) {
        hlo_to_cudnn[hlo] = graph.pointwise(operand(0), attrs);
      } else if (hlo->operand_count() == 2) {
        hlo_to_cudnn[hlo] = graph.pointwise(operand(0), operand(1), attrs);
      } else if (hlo->operand_count() == 3) {
        if (hlo->opcode() != HloOpcode::kSelect) {
          VLOG(3) << "Unexpected ternary operation: " << hlo->ToString();
          return std::nullopt;
        }
        // Operand order for select differs between HLO and cuDNN.
        hlo_to_cudnn[hlo] =
            graph.pointwise(operand(1), operand(2), operand(0), attrs);
      } else {
        VLOG(3) << "Unimplemented elementwise operation.";
        return std::nullopt;
      }
    } else if (hlo->opcode() == HloOpcode::kDot) {
      const auto compute_dtype =
          GetComputeDataType(hlo->shape().element_type());
      if (!compute_dtype.has_value()) {
        return std::nullopt;
      }
      hlo_to_cudnn[hlo] =
          graph.matmul(operand(0), operand(1),
                       graph::Matmul_attributes().set_compute_data_type(
                           compute_dtype.value()));
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
    se::dnn::DnnSupport& dnn_support, const HloFusionInstruction& hlo) {
  TF_ASSIGN_OR_RETURN(std::optional<se::gpu::CudnnGraph> graph,
                      HloFusionToCuDnnGraph(hlo));
  if (!graph.has_value()) {
    return absl::InternalError("Construction of cuDNN graph failed.");
  }
  TF_ASSIGN_OR_RETURN(bool supported, graph->Prepare(dnn_support));
  if (!supported) {
    return absl::InternalError("cuDNN graph is not supported.");
  }
  return *graph;
}

class CuDnnFusionVisitor : public DfsHloRewriteVisitor {
 public:
  explicit CuDnnFusionVisitor(
      se::dnn::DnnSupport& dnn_support,
      CuDnnFusionCompiler::BinaryMap& compilation_results)
      : dnn_support_(dnn_support), compilation_results_(compilation_results) {}

  absl::Status HandleFusion(HloInstruction* hlo) override {
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        hlo->backend_config<GpuBackendConfig>());
    const auto& fusion_backend_config = gpu_config.fusion_backend_config();
    if (fusion_backend_config.kind() != kCuDnnFusionKind) {
      return absl::OkStatus();
    }
    int64_t plan_id = -1;
    if (fusion_backend_config.has_cudnn_fusion_config()) {
      plan_id = fusion_backend_config.cudnn_fusion_config().plan_id();
    }

    VLOG(4) << "Processing " << hlo->ToString();
    VLOG(4) << "Plan ID: " << plan_id;

    const std::string cache_key =
        GetComputationFingerprint(hlo->fused_instructions_computation(), {});
    std::string& cache_entry = compilation_results_[cache_key];
    if (cache_entry.empty()) {
      TF_ASSIGN_OR_RETURN(
          se::gpu::CudnnGraph graph,
          PrepareGraph(dnn_support_, *DynCast<HloFusionInstruction>(hlo)));

      if (plan_id >= 0) {
        // Build single plan with given ID.
        if (plan_id >= graph.Graph().get_execution_plan_count()) {
          return absl::InternalError("cuDNN graph plan does not exist.");
        }
        TF_RETURN_IF_ERROR(graph.Build(dnn_support_, plan_id));
      } else {
        // Build plans one by one till first successful when no plan_id was
        // provided.
        for (plan_id = 0; plan_id < graph.Graph().get_execution_plan_count();
             ++plan_id) {
          VLOG(7) << "Trying plan ID " << plan_id;
          if (graph.Build(dnn_support_, plan_id).ok()) {
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
      cache_entry =
          std::string(reinterpret_cast<char*>(serialized_graph.data()),
                      serialized_graph.size());
    } else {
      VLOG(4) << "Cache hit.";
    }
    auto cudnn_config = gpu_config.mutable_fusion_backend_config()
                            ->mutable_cudnn_fusion_config();
    cudnn_config->set_plan_id(plan_id);
    TF_RETURN_IF_ERROR(hlo->set_backend_config(gpu_config));

    MarkAsChanged();
    return absl::OkStatus();
  }

 private:
  se::dnn::DnnSupport& dnn_support_;
  // <HLO computation fingerprint, serialized compiled cuDNN graph>.
  CuDnnFusionCompiler::BinaryMap& compilation_results_;
};

}  // namespace

absl::StatusOr<bool> CuDnnFusionCompiler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER("cuDNN fusion compiler");
  return CuDnnFusionVisitor(dnn_support_, compilation_results_)
      .RunOnModule(module, execution_threads);
}

int CuDnnFusionCompiler::GetAvailablePlanCount(
    se::StreamExecutor& stream_exec, const HloFusionInstruction& hlo) {
  auto graph = PrepareGraph(*stream_exec.AsDnn(), hlo);
  if (!graph.ok()) {
    return 0;
  }
  constexpr int64_t kMaxPlans = 10;
  return std::min(graph->Graph().get_execution_plan_count(), kMaxPlans);
}

}  // namespace gpu
}  // namespace xla
