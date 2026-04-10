/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/autotuner/hipblaslt.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;
using se::gpu::BlasLt;

using HipblasLtBackendConfig = AutotuneResult::GemmKey;

namespace {

absl::StatusOr<BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig_Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
      return BlasLt::Epilogue::kDefault;
    case GemmBackendConfig::RELU:
      return BlasLt::Epilogue::kReLU;
    case GemmBackendConfig::GELU:
      return BlasLt::Epilogue::kGELU;
    case GemmBackendConfig::GELU_AUX:
      return BlasLt::Epilogue::kGELUWithAux;
    case GemmBackendConfig::SILU:
      return BlasLt::Epilogue::kSILU;
    case GemmBackendConfig::BIAS:
      return BlasLt::Epilogue::kBias;
    case GemmBackendConfig::BIAS_RELU:
      return BlasLt::Epilogue::kBiasThenReLU;
    case GemmBackendConfig::BIAS_GELU:
      return BlasLt::Epilogue::kBiasThenGELU;
    case GemmBackendConfig::BIAS_GELU_AUX:
      return BlasLt::Epilogue::kBiasThenGELUWithAux;
    case GemmBackendConfig::BIAS_SILU:
      return BlasLt::Epilogue::kBiasThenSILU;
    default:
      return Internal("Unsupported Epilogue.");
  }
}

bool IsValidMxScaledDot(const HloInstruction* scaled_dot) {
  const Shape& lhs_shape = scaled_dot->operand(0)->shape();
  const Shape& rhs_shape = scaled_dot->operand(1)->shape();
  const Shape& lhs_scale_shape = scaled_dot->operand(2)->shape();
  const Shape& rhs_scale_shape = scaled_dot->operand(3)->shape();
  const Shape& output_shape = scaled_dot->shape();
  const DotDimensionNumbers& dot_dims = scaled_dot->dot_dimension_numbers();

  auto IsValidInputType = [](PrimitiveType type) {
    return type == F8E4M3FN || type == F8E5M2 || type == F4E2M1FN;
  };

  PrimitiveType lhs_type = lhs_shape.element_type();
  PrimitiveType rhs_type = rhs_shape.element_type();
  if (!IsValidInputType(lhs_type) || !IsValidInputType(rhs_type)) {
    VLOG(2) << "hipBLASLt MX: unsupported data type, lhs="
            << PrimitiveType_Name(lhs_type)
            << " rhs=" << PrimitiveType_Name(rhs_type);
    return false;
  }

  if (lhs_scale_shape.element_type() != F8E8M0FNU ||
      rhs_scale_shape.element_type() != F8E8M0FNU) {
    VLOG(2) << "hipBLASLt MX: scale type must be F8E8M0FNU";
    return false;
  }

  PrimitiveType out_type = output_shape.element_type();
  if (out_type != F32 && out_type != F16 && out_type != BF16) {
    VLOG(2) << "hipBLASLt MX: output type must be F32/F16/BF16, got "
            << PrimitiveType_Name(out_type);
    return false;
  }

  int64_t batch_size = 1;
  for (int64_t dim : dot_dims.lhs_batch_dimensions()) {
    batch_size *= lhs_shape.dimensions(dim);
  }
  if (batch_size != 1) {
    VLOG(2) << "hipBLASLt MX: batch_size > 1 not supported, got " << batch_size;
    return false;
  }

  int64_t m = 1;
  for (int64_t i = 0; i < lhs_shape.dimensions().size(); ++i) {
    if (!absl::c_linear_search(dot_dims.lhs_batch_dimensions(), i) &&
        !absl::c_linear_search(dot_dims.lhs_contracting_dimensions(), i)) {
      m *= lhs_shape.dimensions(i);
    }
  }

  int64_t n = 1;
  for (int64_t i = 0; i < rhs_shape.dimensions().size(); ++i) {
    if (!absl::c_linear_search(dot_dims.rhs_batch_dimensions(), i) &&
        !absl::c_linear_search(dot_dims.rhs_contracting_dimensions(), i)) {
      n *= rhs_shape.dimensions(i);
    }
  }

  int64_t k = 1;
  for (int64_t dim : dot_dims.lhs_contracting_dimensions()) {
    k *= lhs_shape.dimensions(dim);
  }

  if (m % 16 != 0 || n % 16 != 0) {
    VLOG(2) << "hipBLASLt MX: M and N must be divisible by 16, got M=" << m
            << " N=" << n;
    return false;
  }

  if (k % 32 != 0) {
    VLOG(2) << "hipBLASLt MX: K must be divisible by 32, got K=" << k;
    return false;
  }

  int64_t lhs_scale_k = 1;
  for (int64_t dim : dot_dims.lhs_contracting_dimensions()) {
    lhs_scale_k *= lhs_scale_shape.dimensions(dim);
  }
  int64_t rhs_scale_k = 1;
  for (int64_t dim : dot_dims.rhs_contracting_dimensions()) {
    rhs_scale_k *= rhs_scale_shape.dimensions(dim);
  }
  if (lhs_scale_k == 0 || k / lhs_scale_k != 32 || rhs_scale_k == 0 ||
      k / rhs_scale_k != 32) {
    VLOG(2) << "hipBLASLt MX: block size must be 32, got lhs="
            << (lhs_scale_k > 0 ? k / lhs_scale_k : 0)
            << " rhs=" << (rhs_scale_k > 0 ? k / rhs_scale_k : 0);
    return false;
  }

  return true;
}

bool IsScaledDotFusion(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) return false;
  auto gpu_config = instr.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) return false;
  if (gpu_config->fusion_backend_config().kind() != kTritonGemmFusionKind) {
    return false;
  }
  return hlo_query::GetFirstInstructionWithOpcode(
             *instr.fused_instructions_computation(), HloOpcode::kScaledDot) !=
         nullptr;
}

}  // namespace

bool HipblasLtBackend::IsSupported(const HloInstruction& instr) {
  if (IsCublasLtMatmul(instr) || IsCublasLtMatmulF8(instr) ||
      IsCublasLtGroupedMatmul(instr)) {
    return true;
  }
  if (IsScaledDotFusion(instr)) {
    const auto& gpu_cc =
        target_config().device_description.gpu_compute_capability();
    const auto* rocm_cc = gpu_cc.rocm_compute_capability();
    return rocm_cc != nullptr && rocm_cc->has_mx_type_support();
  }
  return false;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
HipblasLtBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  } else if (IsCublasLtMatmul(instr) || IsCublasLtMatmulF8(instr)) {
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        instr.backend_config<GpuBackendConfig>());
    const GemmBackendConfig& backend_config = gpu_config.gemm_backend_config();

    TF_ASSIGN_OR_RETURN(
        GemmConfig gemm_config,
        GemmConfig::For(
            &instr,
            target_config().device_description.gpu_compute_capability()));

    TF_ASSIGN_OR_RETURN(BlasLt::Epilogue epilogue,
                        AsBlasLtEpilogue(backend_config.epilogue()));

    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                        stream_executor()->CreateStream());

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<BlasLt::MatmulPlan> plan,
        se::gpu::BlasLt::GetMatmulPlan(stream.get(), gemm_config, epilogue));

    const Shape& output_shape = instr.shape();
    if (!output_shape.IsTuple() || output_shape.tuple_shapes().empty()) {
      return Internal(
          "Invalid shape for HipblasLt matmul: output is not a non-empty "
          "tuple.");
    }
    const int64_t workspace_size =
        ShapeUtil::ByteSizeOf(output_shape.tuple_shapes().back());

    TF_ASSIGN_OR_RETURN(
        std::vector<BlasLt::MatmulAlgorithm> algorithms,
        plan->GetAlgorithms(stream.get(), GemmConfig::kNumAlgorithms,
                            workspace_size));
    int num_algorithms = algorithms.size();
    std::vector<std::unique_ptr<BackendConfig>> configs;
    configs.reserve(num_algorithms);
    for (int i = 0; i < num_algorithms; ++i) {
      HipblasLtBackendConfig gemm_key;
      gemm_key.set_algorithm(i);
      gemm_key.set_autotune_workspace_size(workspace_size);
      auto any = std::make_unique<google::protobuf::Any>();
      any->PackFrom(gemm_key);
      configs.push_back(std::move(any));
    }
    return configs;
  } else if (IsScaledDotFusion(instr)) {
    const HloInstruction* scaled_dot = hlo_query::GetFirstInstructionWithOpcode(
        *instr.fused_instructions_computation(), HloOpcode::kScaledDot);
    TF_RET_CHECK(scaled_dot != nullptr);

    if (!IsValidMxScaledDot(scaled_dot)) {
      return std::vector<std::unique_ptr<BackendConfig>>();
    }

    const Shape& lhs_shape = scaled_dot->operand(0)->shape();
    const Shape& rhs_shape = scaled_dot->operand(1)->shape();
    const DotDimensionNumbers& dot_dims = scaled_dot->dot_dimension_numbers();
    const Shape& output_shape = scaled_dot->shape();

    auto gemm_config_or = GemmConfig::For(
        lhs_shape, dot_dims.lhs_batch_dimensions(),
        dot_dims.lhs_contracting_dimensions(), rhs_shape,
        dot_dims.rhs_batch_dimensions(), dot_dims.rhs_contracting_dimensions(),
        output_shape,
        /*alpha_real=*/1.0, /*alpha_imag=*/0.0, /*beta=*/0.0,
        PrecisionConfig::ALG_UNSET, /*algorithm=*/std::nullopt,
        se::blas::kDefaultComputePrecision, /*grad_x=*/false,
        /*grad_y=*/false, /*scale_mode=*/se::gpu::ScaleMode::kBlockScaling,
        target_config().device_description.gpu_compute_capability());
    if (!gemm_config_or.ok()) {
      VLOG(2) << "hipBLASLt MX: GemmConfig::For failed: "
              << gemm_config_or.status();
      return std::vector<std::unique_ptr<BackendConfig>>();
    }

    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                        stream_executor()->CreateStream());
    auto plan_or = se::gpu::BlasLt::GetMatmulPlan(stream.get(), *gemm_config_or,
                                                  BlasLt::Epilogue::kDefault);
    if (!plan_or.ok()) {
      VLOG(2) << "hipBLASLt MX: GetMatmulPlan failed: " << plan_or.status();
      return std::vector<std::unique_ptr<BackendConfig>>();
    }

    int64_t workspace_size = GemmConfig::kGFX950Workspace;
    TF_ASSIGN_OR_RETURN(
        std::vector<BlasLt::MatmulAlgorithm> algorithms,
        (*plan_or)->GetAlgorithms(stream.get(), GemmConfig::kNumAlgorithms,
                                  workspace_size));
    if (algorithms.empty()) {
      VLOG(2) << "hipBLASLt MX: no algorithms found for scaled dot.";
      return std::vector<std::unique_ptr<BackendConfig>>();
    }

    std::vector<std::unique_ptr<BackendConfig>> configs;
    configs.reserve(algorithms.size());
    for (int64_t i = 0; i < static_cast<int64_t>(algorithms.size()); ++i) {
      HipblasLtBackendConfig gemm_key;
      gemm_key.set_algorithm(i);
      gemm_key.set_autotune_workspace_size(workspace_size);
      auto any = std::make_unique<google::protobuf::Any>();
      any->PackFrom(gemm_key);
      configs.push_back(std::move(any));
    }
    return configs;
  } else if (IsCublasLtGroupedMatmul(instr)) {
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        instr.backend_config<GpuBackendConfig>());

    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                        stream_executor()->CreateStream());

    std::unique_ptr<BlasLt::MatmulPlan> plan;
    int64_t workspace_size;
    const GroupedGemmBackendConfig& grouped_config =
        gpu_config.grouped_gemm_backend_config();
    const GemmBackendConfig& backend_config =
        grouped_config.gemm_backend_config();

    TF_ASSIGN_OR_RETURN(
        GroupedGemmConfig grouped_gemm_config,
        GroupedGemmConfig::For(
            &instr,
            target_config().device_description.gpu_compute_capability()));

    TF_ASSIGN_OR_RETURN(BlasLt::Epilogue epilogue,
                        AsBlasLtEpilogue(backend_config.epilogue()));

    std::vector<BlasLt::Epilogue> epilogues = {epilogue};
    TF_ASSIGN_OR_RETURN(plan,
                        se::gpu::BlasLt::GetGroupedMatmulPlan(
                            stream.get(), grouped_gemm_config, epilogues));

    const Shape& output_shape = instr.shape();
    if (!output_shape.IsTuple() || output_shape.tuple_shapes().empty()) {
      return Internal(
          "Invalid shape for HipblasLt grouped matmul: output is not a "
          "non-empty tuple.");
    }
    workspace_size = ShapeUtil::ByteSizeOf(output_shape.tuple_shapes().back());

    TF_ASSIGN_OR_RETURN(
        std::vector<BlasLt::MatmulAlgorithm> algorithms,
        plan->GetAlgorithms(stream.get(), GemmConfig::kNumAlgorithms,
                            workspace_size));
    int num_algorithms = algorithms.size();
    std::vector<std::unique_ptr<BackendConfig>> configs;
    configs.reserve(num_algorithms);
    for (int i = 0; i < num_algorithms; ++i) {
      HipblasLtBackendConfig gemm_key;
      gemm_key.set_algorithm(i);
      gemm_key.set_autotune_workspace_size(workspace_size);
      auto any = std::make_unique<google::protobuf::Any>();
      any->PackFrom(gemm_key);
      configs.push_back(std::move(any));
    }

    return configs;
  }

  return std::vector<std::unique_ptr<BackendConfig>>();
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
HipblasLtBackend::GetDefaultConfig(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError("Not a supported HipblasLt instruction.");
  }

  AutotuneResult::GemmKey gemm_key;
  gemm_key.set_algorithm(0);
  auto any = std::make_unique<google::protobuf::Any>();
  any->PackFrom(gemm_key);
  return any;
}

absl::Status HipblasLtBackend::ApplyConfig(HloInstruction& instr,
                                           const BackendConfig& config) {
  HipblasLtBackendConfig gemm_key;
  if (!config.UnpackTo(&gemm_key)) {
    return absl::InvalidArgumentError(
        "Failed to unpack HipblasLtBackendConfig from Any.");
  }

  if (IsCublasLtMatmul(instr) || IsCublasLtMatmulF8(instr)) {
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        instr.backend_config<GpuBackendConfig>());
    GemmBackendConfig& backend_config =
        *gpu_config.mutable_gemm_backend_config();
    backend_config.set_selected_algorithm(gemm_key.algorithm());
    backend_config.set_autotune_workspace_size(
        gemm_key.autotune_workspace_size());
    TF_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));

    if (instr.shape().IsTuple() && !instr.shape().tuple_shapes().empty()) {
      Shape* workspace_shape = instr.mutable_shape()->mutable_tuple_shapes(
          instr.shape().tuple_shapes().size() - 1);
      if (workspace_shape->element_type() == S8 &&
          workspace_shape->dimensions().size() == 1) {
        workspace_shape->set_dimensions(0, gemm_key.autotune_workspace_size());
        if (HloModule* module = instr.GetModule()) {
          if (module->entry_computation() &&
              module->entry_computation()->root_instruction() == &instr) {
            *module->mutable_entry_computation_layout()
                 ->mutable_result_layout() = ShapeLayout(instr.shape());
          }
        }
      }
    }
    return absl::OkStatus();
  } else if (IsScaledDotFusion(instr)) {
    const HloInstruction* scaled_dot = hlo_query::GetFirstInstructionWithOpcode(
        *instr.fused_instructions_computation(), HloOpcode::kScaledDot);
    TF_RET_CHECK(scaled_dot != nullptr);
    HloComputation* parent = instr.parent();

    TF_RET_CHECK(instr.operand_count() == 4);
    HloInstruction* lhs = instr.mutable_operand(0);
    HloInstruction* rhs = instr.mutable_operand(1);
    HloInstruction* lhs_scale = instr.mutable_operand(2);
    HloInstruction* rhs_scale = instr.mutable_operand(3);

    const Shape& result_shape = scaled_dot->shape();
    int64_t workspace_size = gemm_key.autotune_workspace_size();
    Shape workspace_shape = ShapeUtil::MakeShape(S8, {workspace_size});
    Shape output_shape =
        ShapeUtil::MakeTupleShape({result_shape, workspace_shape});

    GpuBackendConfig gpu_backend_config;
    GemmBackendConfig& gemm_config =
        *gpu_backend_config.mutable_gemm_backend_config();
    *gemm_config.mutable_dot_dimension_numbers() =
        scaled_dot->dot_dimension_numbers();
    gemm_config.set_alpha_real(1.0);
    gemm_config.set_alpha_imag(0.0);
    gemm_config.set_beta(0.0);
    gemm_config.set_scale_mode(
        static_cast<int32_t>(se::gpu::ScaleMode::kBlockScaling));
    gemm_config.set_selected_algorithm(gemm_key.algorithm());
    gemm_config.set_autotune_workspace_size(workspace_size);

    HloInstruction* custom_call =
        parent->AddInstruction(HloInstruction::CreateCustomCall(
            output_shape, {lhs, rhs, lhs_scale, rhs_scale},
            kCublasLtMatmulMxCallTarget));
    TF_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_backend_config));
    HloInstruction* gte = parent->AddInstruction(
        HloInstruction::CreateGetTupleElement(result_shape, custom_call, 0));
    return parent->ReplaceInstruction(&instr, gte);
  } else if (IsCublasLtGroupedMatmul(instr)) {
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        instr.backend_config<GpuBackendConfig>());
    GemmBackendConfig* backend_config =
        gpu_config.mutable_grouped_gemm_backend_config()
            ->mutable_gemm_backend_config();

    backend_config->set_selected_algorithm(gemm_key.algorithm());
    backend_config->set_autotune_workspace_size(
        gemm_key.autotune_workspace_size());
    TF_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError(
      "ApplyConfig called on unsupported instruction.");
}

}  // namespace gpu
}  // namespace xla
