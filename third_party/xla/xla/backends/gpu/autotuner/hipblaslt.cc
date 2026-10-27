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
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/compiler.h"
#include "xla/service/decision.h"
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
#include "xla/tools/hlo_decomposer.h"
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

// Returns the scaled-dot instruction inside a Triton GEMM fusion, or nullptr
// if `instr` is not such a fusion.
const HloInstruction* GetScaledDotFromFusion(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return nullptr;
  }
  auto gpu_config = instr.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return nullptr;
  }
  if (gpu_config->fusion_backend_config().kind() != kTritonGemmFusionKind) {
    return nullptr;
  }
  return hlo_query::GetFirstInstructionWithOpcode(
      *instr.fused_instructions_computation(), HloOpcode::kScaledDot);
}

// Decides whether `gemm` is an FP8-output __cublas$lt$matmul$f8, without amax
// or auxiliary output, that can run as a GEMM with a wider output followed by
// the D scale, the saturation and the conversion to FP8. A forbidding decision
// explains what prevents the rewrite.
Decision CanUnfuseFp8Output(const HloInstruction& gemm) {
  if (!IsCublasLtMatmulF8(gemm)) {
    return Decision::Forbid("not a __cublas$lt$matmul$f8 call");
  }
  if (!gemm.shape().IsTuple() || gemm.shape().tuple_shapes().size() != 2) {
    return Decision::Forbid(absl::StrCat(
        "expected an output and a workspace, got ", gemm.shape().ToString()));
  }
  PrimitiveType d_type = gemm.shape().tuple_shapes(0).element_type();
  if (d_type != F8E4M3FN && d_type != F8E5M2 && d_type != F8E4M3FNUZ &&
      d_type != F8E5M2FNUZ) {
    return Decision::Forbid(absl::StrCat(
        "the output is ", PrimitiveType_Name(d_type), ", not FP8"));
  }
  absl::StatusOr<GpuBackendConfig> gpu_config =
      gemm.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return Decision(gpu_config.status());
  }
  const GemmBackendConfig& config = gpu_config->gemm_backend_config();
  if (config.beta() != 0) {
    return Decision::Forbid(
        absl::StrCat("beta is ", config.beta(), ", so the output is read"));
  }
  absl::StatusOr<bool> has_vector_bias =
      gpublas_lt::EpilogueAddsVectorBias(config.epilogue());
  if (!has_vector_bias.ok()) {
    return Decision(has_vector_bias.status());
  }
  // An amax or auxiliary output takes an operand the unfused GEMM cannot keep.
  int64_t expected_operands =
      4 + int{*has_vector_bias} + int{config.has_d_scale()};
  if (gemm.operand_count() != expected_operands) {
    return Decision::Forbid(absl::StrCat("expected ", expected_operands,
                                         " operands, got ",
                                         gemm.operand_count()));
  }
  // The vector bias follows the A and B scales.
  if (*has_vector_bias) {
    PrimitiveType bias_type = gemm.operand(4)->shape().element_type();
    if (bias_type != F16 && bias_type != BF16) {
      return Decision::Forbid(absl::StrCat("the vector bias is ",
                                           PrimitiveType_Name(bias_type),
                                           ", not F16 or BF16"));
    }
  }
  return Decision::Allow();
}

// Adds a copy of `gemm` without the D scale whose output is F32, or has the
// type of the vector bias, since hipBLASLt reads the bias with the output type
// for FP8 inputs.
absl::StatusOr<HloInstruction*> AddWideOutputGemm(HloInstruction* gemm) {
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   gemm->backend_config<GpuBackendConfig>());
  GemmBackendConfig& config = *gpu_config.mutable_gemm_backend_config();
  std::vector<HloInstruction*> operands(gemm->operands().begin(),
                                        gemm->operands().end());
  if (config.has_d_scale()) {
    operands.pop_back();
    // Cleared, not set to false, to print like a GEMM without a D scale.
    config.clear_has_d_scale();
  }
  ABSL_ASSIGN_OR_RETURN(bool has_vector_bias,
                   gpublas_lt::EpilogueAddsVectorBias(config.epilogue()));
  Shape shape = gemm->shape();
  *shape.mutable_tuple_shapes(0) = ShapeUtil::ChangeElementType(
      shape.tuple_shapes(0),
      has_vector_bias ? operands[4]->shape().element_type() : F32);
  HloInstruction* wide = gemm->parent()->AddInstruction(
      gemm->CloneWithNewOperands(shape, operands));
  ABSL_RETURN_IF_ERROR(wide->set_backend_config(gpu_config));
  return wide;
}

// Rewrites `gemm` into `wide`, made by AddWideOutputGemm, followed by a loop
// fusion that applies the D scale, saturates like hipBLASLt and converts the
// result to the FP8 output type of `gemm`.
absl::Status RewriteWithWideOutputGemm(HloInstruction* gemm,
                                       HloInstruction* wide) {
  HloComputation* computation = gemm->parent();
  const Shape& d_shape = gemm->shape().tuple_shapes(0);
  Shape f32_shape = ShapeUtil::ChangeElementType(d_shape, F32);
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   gemm->backend_config<GpuBackendConfig>());

  // The instructions to fuse, in topological order.
  std::vector<HloInstruction*> fused;
  auto add = [&](std::unique_ptr<HloInstruction> instr) {
    fused.push_back(computation->AddInstruction(std::move(instr)));
    return fused.back();
  };
  auto add_broadcast_constant = [&](float value) {
    HloInstruction* constant =
        add(HloInstruction::CreateConstant(LiteralUtil::CreateR0(value)));
    return add(HloInstruction::CreateBroadcast(f32_shape, constant, {}));
  };
  HloInstruction* result = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(wide, 0));
  if (result->shape().element_type() != F32) {
    result = add(HloInstruction::CreateConvert(f32_shape, result));
  }
  if (gpu_config.gemm_backend_config().has_d_scale()) {
    HloInstruction* d_scale = add(HloInstruction::CreateBroadcast(
        f32_shape, gemm->operands().back(), {}));
    result = add(HloInstruction::CreateBinary(f32_shape, HloOpcode::kMultiply,
                                              result, d_scale));
  }
  ABSL_ASSIGN_OR_RETURN(
      Literal max,
      LiteralUtil::MaxFiniteValue(d_shape.element_type()).Convert(F32));
  HloInstruction* lower = add_broadcast_constant(-max.GetFirstElement<float>());
  HloInstruction* upper = add_broadcast_constant(max.GetFirstElement<float>());
  result = add(HloInstruction::CreateTernary(f32_shape, HloOpcode::kClamp,
                                             lower, result, upper));
  add(HloInstruction::CreateConvert(d_shape, result));
  absl::c_reverse(fused);
  HloInstruction* fusion = computation->CreateFusionInstruction(
      fused, HloInstruction::FusionKind::kLoop);
  fusion->fused_instructions_computation()->SetExecutionThread(
      computation->execution_thread());

  ABSL_RETURN_IF_ERROR(gemm->CopyAllControlDepsTo(wide, fusion));
  ABSL_RETURN_IF_ERROR(gemm->DropAllControlDeps());
  HloInstruction* workspace = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(wide, 1));
  for (HloInstruction* user : std::vector<HloInstruction*>(gemm->users())) {
    TF_RET_CHECK(user->opcode() == HloOpcode::kGetTupleElement)
        << user->ToString();
    ABSL_RETURN_IF_ERROR(computation
                        ->ReplaceInstructionWithDifferentShape(
                            user, user->tuple_index() == 0 ? fusion : workspace,
                            /*preserve_sharding=*/false,
                            /*relay_control_dependency=*/true,
                            /*remove_unused_operands=*/false)
                        .status());
  }
  if (computation->root_instruction() == gemm) {
    computation->set_root_instruction(computation->AddInstruction(
        HloInstruction::CreateTuple({fusion, workspace})));
  }
  ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(gemm));
  return workspace->IsDead() ? computation->RemoveInstruction(workspace)
                             : absl::OkStatus();
}

}  // namespace

bool HipblasLtBackend::IsSupported(const HloInstruction& instr) {
  if (IsCublasLtMatmul(instr) || IsCublasLtMatmulF8(instr) ||
      IsCublasLtGroupedMatmul(instr)) {
    return true;
  }
  if (GetScaledDotFromFusion(instr) != nullptr) {
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
  }

  if (stream_executor() == nullptr) {
    return absl::InvalidArgumentError(
        "HipblasLtBackend cannot enumerate configs in deviceless mode.");
  }

  if (IsCublasLtMatmul(instr) || IsCublasLtMatmulF8(instr)) {
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                     instr.backend_config<GpuBackendConfig>());
    const GemmBackendConfig& backend_config = gpu_config.gemm_backend_config();

    ABSL_ASSIGN_OR_RETURN(
        GemmConfig gemm_config,
        GemmConfig::For(
            &instr,
            target_config().device_description.gpu_compute_capability()));

    ABSL_ASSIGN_OR_RETURN(BlasLt::Epilogue epilogue,
                     AsBlasLtEpilogue(backend_config.epilogue()));

    ABSL_ASSIGN_OR_RETURN(BlasLt * blas_lt, se::gpu::BlasLt::Get(stream_executor()));

    ABSL_ASSIGN_OR_RETURN(BlasLt::MatmulPlanPtr plan,
                     blas_lt->GetMatmulPlan(gemm_config, epilogue));

    const Shape& output_shape = instr.shape();
    if (!output_shape.IsTuple() || output_shape.tuple_shapes().empty()) {
      return Internal(
          "Invalid shape for HipblasLt matmul: output is not a non-empty "
          "tuple.");
    }
    const int64_t workspace_size =
        ShapeUtil::ByteSizeOf(output_shape.tuple_shapes().back());

    int max_algorithms = debug_options().xla_gpu_blas_max_algorithms();
    if (max_algorithms <= 0) {
      max_algorithms = GemmConfig::kNumAlgorithms;
    }

    ABSL_ASSIGN_OR_RETURN(std::vector<BlasLt::MatmulAlgorithm> algorithms,
                     plan->GetAlgorithms(max_algorithms, workspace_size));
    int num_algorithms = algorithms.size();
    std::vector<std::unique_ptr<BackendConfig>> configs;
    configs.reserve(num_algorithms);
    for (int i = 0; i < num_algorithms; ++i) {
      auto config = std::make_unique<BackendConfig>();
      auto* gemm_key = config->mutable_gemm();
      gemm_key->set_algorithm(i);
      gemm_key->set_autotune_workspace_size(workspace_size);
      configs.push_back(std::move(config));
    }
    if (configs.empty()) {
      Decision can_unfuse = CanUnfuseFp8Output(instr);
      if (can_unfuse.IsForbidden()) {
        VLOG(2) << "hipBLASLt: not unfusing the output of " << instr.name()
                << ": " << can_unfuse.Explain();
      } else {
        // Query the wider GEMM in a scratch module, leaving `instr` unchanged.
        std::unique_ptr<HloModule> module =
            ExtractInstructionIntoNewModule(instr);
        ABSL_ASSIGN_OR_RETURN(
            HloInstruction * wide,
            AddWideOutputGemm(module->entry_computation()->root_instruction()));
        ABSL_ASSIGN_OR_RETURN(configs, GetSupportedConfigs(*wide));
        for (std::unique_ptr<BackendConfig>& config : configs) {
          config->mutable_gemm()->set_unfuse_fp8_output(true);
        }
        if (!configs.empty()) {
          LOG(WARNING)
              << "hipBLASLt: no algorithms found for the FP8 output of "
              << instr.name()
              << "; using a wider output and a separate FP8 conversion.";
        }
      }
    }
    return configs;
  }

  if (const auto* scaled_dot = GetScaledDotFromFusion(instr)) {
    if (!IsValidMxScaledDot(scaled_dot)) {
      LOG(WARNING) << "hipBLASLt MX: IsValidMxScaledDot failed";
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
      LOG(WARNING) << "hipBLASLt MX: GemmConfig::For failed: "
                   << gemm_config_or.status();
      return std::vector<std::unique_ptr<BackendConfig>>();
    }

    ABSL_ASSIGN_OR_RETURN(BlasLt * blas_lt, se::gpu::BlasLt::Get(stream_executor()));
    auto plan_or =
        blas_lt->GetMatmulPlan(*gemm_config_or, BlasLt::Epilogue::kDefault);
    if (!plan_or.ok()) {
      LOG(WARNING) << "hipBLASLt MX: GetMatmulPlan failed: "
                   << plan_or.status();
      return std::vector<std::unique_ptr<BackendConfig>>();
    }

    int64_t workspace_size = GemmConfig::kGFX950Workspace;
    int max_algorithms = debug_options().xla_gpu_blas_max_algorithms();
    if (max_algorithms <= 0) {
      max_algorithms = GemmConfig::kNumAlgorithms;
    }
    ABSL_ASSIGN_OR_RETURN(std::vector<BlasLt::MatmulAlgorithm> algorithms,
                     (*plan_or)->GetAlgorithms(max_algorithms, workspace_size));
    if (algorithms.empty()) {
      LOG(WARNING) << "hipBLASLt MX: no algorithms found for scaled dot.";
      return std::vector<std::unique_ptr<BackendConfig>>();
    }

    std::vector<std::unique_ptr<BackendConfig>> configs;
    configs.reserve(algorithms.size());
    for (int64_t i = 0; i < static_cast<int64_t>(algorithms.size()); ++i) {
      auto config = std::make_unique<BackendConfig>();
      auto* gemm_key = config->mutable_gemm();
      gemm_key->set_algorithm(i);
      gemm_key->set_autotune_workspace_size(workspace_size);
      configs.push_back(std::move(config));
    }
    return configs;
  }
  if (IsCublasLtGroupedMatmul(instr)) {
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                     instr.backend_config<GpuBackendConfig>());

    ABSL_ASSIGN_OR_RETURN(BlasLt * blas_lt, se::gpu::BlasLt::Get(stream_executor()));

    int64_t workspace_size;
    const GroupedGemmBackendConfig& grouped_config =
        gpu_config.grouped_gemm_backend_config();
    const GemmBackendConfig& backend_config =
        grouped_config.gemm_backend_config();

    ABSL_ASSIGN_OR_RETURN(
        GroupedGemmConfig grouped_gemm_config,
        GroupedGemmConfig::For(
            &instr,
            target_config().device_description.gpu_compute_capability()));

    ABSL_ASSIGN_OR_RETURN(BlasLt::Epilogue epilogue,
                     AsBlasLtEpilogue(backend_config.epilogue()));

    ABSL_ASSIGN_OR_RETURN(BlasLt::MatmulPlanPtr plan,
                     blas_lt->GetMatmulPlan(grouped_gemm_config, epilogue));

    const Shape& output_shape = instr.shape();
    if (!output_shape.IsTuple() || output_shape.tuple_shapes().empty()) {
      return Internal(
          "Invalid shape for HipblasLt grouped matmul: output is not a "
          "non-empty tuple.");
    }
    workspace_size = ShapeUtil::ByteSizeOf(output_shape.tuple_shapes().back());

    int max_algorithms = debug_options().xla_gpu_blas_max_algorithms();
    if (max_algorithms <= 0) {
      max_algorithms = GemmConfig::kNumAlgorithms;
    }

    ABSL_ASSIGN_OR_RETURN(std::vector<BlasLt::MatmulAlgorithm> algorithms,
                     plan->GetAlgorithms(max_algorithms, workspace_size));
    int num_algorithms = algorithms.size();
    std::vector<std::unique_ptr<BackendConfig>> configs;
    configs.reserve(num_algorithms);
    for (int i = 0; i < num_algorithms; ++i) {
      auto config = std::make_unique<BackendConfig>();
      auto* gemm_key = config->mutable_gemm();
      gemm_key->set_algorithm(i);
      gemm_key->set_autotune_workspace_size(workspace_size);
      configs.push_back(std::move(config));
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

  auto config = std::make_unique<BackendConfig>();
  auto* gemm_key = config->mutable_gemm();
  gemm_key->set_algorithm(0);
  // Standard GEMMs were already lowered into custom calls with a workspace
  // buffer tuple by GemmRewriter, so we can preserve that allocated workspace
  // size. In contrast, scaled-dot GEMMs enter this pass as a fusion and are
  // only lowered to a custom call in ApplyConfig(), so they do not yet have a
  // workspace buffer in their shape. MX scaled dots on ROCm are currently
  // supported on gfx950, so we use kGFX950Workspace (matching
  // GetSupportedConfigs).
  if (GetScaledDotFromFusion(instr) != nullptr) {
    gemm_key->set_autotune_workspace_size(GemmConfig::kGFX950Workspace);
  } else if (instr.shape().IsTuple() && !instr.shape().tuple_shapes().empty()) {
    gemm_key->set_autotune_workspace_size(
        ShapeUtil::ByteSizeOf(instr.shape().tuple_shapes().back()));
  }
  return config;
}

absl::Status HipblasLtBackend::ApplyConfig(HloInstruction& instr,
                                           const BackendConfig& config) {
  if (!config.has_gemm()) {
    return absl::InvalidArgumentError(
        "Expected GemmKey config for HipblasLtBackend.");
  }
  const AutotuneResult::GemmKey& gemm_key = config.gemm();

  if (gemm_key.unfuse_fp8_output()) {
    Decision can_unfuse = CanUnfuseFp8Output(instr);
    if (can_unfuse.IsForbidden()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "unfuse_fp8_output is set for an instruction whose FP8 output "
          "cannot be unfused: ",
          can_unfuse.Explain()));
    }
    BackendConfig gemm_config = config;
    gemm_config.mutable_gemm()->clear_unfuse_fp8_output();
    ABSL_RETURN_IF_ERROR(ApplyConfig(instr, gemm_config));
    ABSL_ASSIGN_OR_RETURN(HloInstruction * wide, AddWideOutputGemm(&instr));
    return RewriteWithWideOutputGemm(&instr, wide);
  }

  if (IsCublasLtMatmul(instr) || IsCublasLtMatmulF8(instr)) {
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                     instr.backend_config<GpuBackendConfig>());
    GemmBackendConfig& backend_config =
        *gpu_config.mutable_gemm_backend_config();
    backend_config.set_selected_algorithm(gemm_key.algorithm());
    backend_config.set_autotune_workspace_size(
        gemm_key.autotune_workspace_size());
    ABSL_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));

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
  }

  if (const auto* scaled_dot = GetScaledDotFromFusion(instr)) {
    HloComputation* fused = instr.fused_instructions_computation();
    HloComputation* parent = instr.parent();

    TF_RET_CHECK(instr.operand_count() == 4);
    // The scaled-dot must be the root of the fusion, otherwise we would drop
    // any instructions that consume the scaled-dot result.
    TF_RET_CHECK(scaled_dot == fused->root_instruction())
        << "Scaled-dot is not the root of the fusion; lowering to hipBLASLt "
           "matmul would drop instructions: "
        << fused->root_instruction()->ToString();

    // Walk forward from each fusion parameter to the scaled-dot operand it
    // feeds. The fusion's external operands may have a different rank than the
    // scaled-dot operands: when quantization happens in-graph the operand is in
    // block-scaled [..., K/block, block] form and is only flattened to [..., K]
    // by a bitcast/reshape inside the fusion. Only such trivial rank-changing
    // ops are allowed on the path.
    absl::InlinedVector<HloInstruction*, 4> operands(4, nullptr);
    for (HloInstruction* param : fused->parameter_instructions()) {
      const HloInstruction* cur = param;
      while (cur->user_count() == 1 && cur->users()[0] != scaled_dot) {
        HloInstruction* user = cur->users()[0];
        TF_RET_CHECK(user->opcode() == HloOpcode::kBitcast ||
                     user->opcode() == HloOpcode::kReshape)
            << "Unexpected op between fusion parameter and scaled-dot: "
            << user->ToString();
        cur = user;
      }
      TF_RET_CHECK(cur->user_count() == 1 && cur->users()[0] == scaled_dot)
          << "Fusion parameter " << param->parameter_number()
          << " does not feed directly into the scaled-dot.";

      int64_t dot_index = scaled_dot->operand_index(cur);
      const Shape& inner_shape = scaled_dot->operand(dot_index)->shape();
      HloInstruction* ext = instr.mutable_operand(param->parameter_number());
      if (!ShapeUtil::Equal(ext->shape(), inner_shape)) {
        ext = parent->AddInstruction(
            HloInstruction::CreateBitcast(inner_shape, ext));
      }
      TF_RET_CHECK(operands[dot_index] == nullptr)
          << "Multiple fusion parameters feed scaled-dot operand " << dot_index;
      operands[dot_index] = ext;
    }  // for
    for (HloInstruction* op : operands) {
      TF_RET_CHECK(op != nullptr)
          << "Not all scaled-dot operands are fed by a fusion parameter.";
    }
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
            output_shape, operands, kCublasLtMatmulMxCallTarget));
    ABSL_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_backend_config));
    HloInstruction* gte = parent->AddInstruction(
        HloInstruction::CreateGetTupleElement(result_shape, custom_call, 0));
    return parent->ReplaceInstruction(&instr, gte);
  }
  if (IsCublasLtGroupedMatmul(instr)) {
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                     instr.backend_config<GpuBackendConfig>());
    GemmBackendConfig* backend_config =
        gpu_config.mutable_grouped_gemm_backend_config()
            ->mutable_gemm_backend_config();

    backend_config->set_selected_algorithm(gemm_key.algorithm());
    backend_config->set_autotune_workspace_size(
        gemm_key.autotune_workspace_size());
    ABSL_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError(
      "ApplyConfig called on unsupported instruction.");
}

}  // namespace gpu
}  // namespace xla
