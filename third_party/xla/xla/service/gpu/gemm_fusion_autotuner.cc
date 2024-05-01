/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/gemm_fusion_autotuner.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/float_normalization.h"
#include "xla/service/gpu/autotuner_compile_util.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_comparator.h"
#include "xla/service/gpu/cudnn_fusion_compiler.h"
#include "xla/service/gpu/gemm_rewriter.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/instruction_fusion.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/split_k_gemm_rewriter.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/util/proto/proto_utils.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/bits.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/profiler/lib/scoped_annotation.h"

// Log levels used in this file:
// VLOG(1): Overview
// VLOG(2): Autotuning progress
// VLOG(3): Autotuning progress - more frequent
// VLOG(4): Print all fusions
// VLOG(5): Profiling information for every tiling

// TODO(b/317016172): Update usages of TritonGemmConfig to use newly exposed
// parameters.

namespace xla {
namespace gpu {

using Config = GemmFusionAutotunerImpl::Config;
using ProfilingOutput = AutotunerCompileUtil::ProfilingOutput;

namespace {

// Minimum tile size.
constexpr int kMinTileSize = 16;
constexpr int kDimKMinTileSize = 32;
constexpr std::array<int, 6> kDimKBlockSizes = {32, 64, 128, 256, 512};

// Default tiling when autotuning is disabled.
constexpr TritonGemmConfig kDefaultGemmTiling = {32, 32, 32, 1, 1, 4};

// Split-K is enabled when the estimate number of waves is lower than the limit.
constexpr int kMaxWavesForSplitK = 5;

// Search space for exhaustive matmul autotuning.
constexpr std::array<int, 6> kBlockSizes = {16, 32, 64, 128, 256, 512};
constexpr std::array<int, 4> kNumStages = {1, 2, 3, 4};
constexpr std::array<int, 4> kNumWarps = {2, 4, 8, 16};
constexpr std::array<int, 5> kSplitK = {1, 2, 4, 8, 16};
constexpr std::array<int, 5> kNumCtas = {1, 2, 4, 8, 16};

class GemmFusionAutotunerVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmFusionAutotunerVisitor(const AutotuneConfig& config)
      : config_(config) {}

  absl::Status HandleFusion(HloInstruction* hlo) override {
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        hlo->backend_config<GpuBackendConfig>());
    FusionBackendConfig& backend_config =
        *gpu_config.mutable_fusion_backend_config();
    if (backend_config.kind() != kTritonGemmFusionKind &&
        backend_config.kind() != kCuDnnFusionKind) {
      return absl::OkStatus();
    }

    VLOG(4) << "Processing " << hlo->ToString();
    if (!backend_config.has_triton_gemm_config() &&
        !backend_config.has_cudnn_fusion_config()) {
      TF_ASSIGN_OR_RETURN(
          AutotuneResult autotune_result,
          AutotunerUtil::Autotune(
              hlo, config_, [&]() -> absl::StatusOr<AutotuneResult> {
                if (config_.IsDeviceless()) {
                  return absl::InternalError(absl::StrCat(
                      "Expect autotune result cache hit for deviceless "
                      "compilation (HLO: ",
                      hlo->ToString(), ")"));
                }
                return absl::InternalError("Expect autotune result cache hit.");
              }));
      VLOG(4) << "Result: " << autotune_result.ShortDebugString();

      if (autotune_result.has_triton()) {
        *backend_config.mutable_triton_gemm_config() = autotune_result.triton();
        TF_RETURN_IF_ERROR(hlo->set_backend_config(gpu_config));
      } else if (autotune_result.has_gemm()) {
        // Falling back to cuBLAS: Converting the fusion to a Call, so that it
        // can be inlined back again.
        HloComputation* const computation = hlo->parent();
        HloInstruction* const call = computation->AddInstruction(
            HloInstruction::CreateCall(hlo->shape(), hlo->operands(),
                                       hlo->fused_instructions_computation()));
        TF_RETURN_IF_ERROR(computation->ReplaceInstruction(hlo, call));
        hlo = call;
      } else {
        CHECK(autotune_result.has_algorithm());
        backend_config.set_kind(std::string(kCuDnnFusionKind));
        backend_config.mutable_cudnn_fusion_config()->set_plan_id(
            autotune_result.algorithm().algo_id());
        TF_RETURN_IF_ERROR(hlo->set_backend_config(gpu_config));
      }
    }

    if (backend_config.has_triton_gemm_config()) {
      TF_ASSIGN_OR_RETURN(
          const TritonGemmConfig config,
          TritonGemmConfig::FromProto(backend_config.triton_gemm_config()));
      if (config.split_k > 1) {
        TF_RETURN_IF_ERROR(MakeDotSplitKBatch(hlo, config));
      }
    }

    MarkAsChanged();
    return absl::OkStatus();
  }

 private:
  AutotuneConfig config_;
};

class GemmConfigSetCollector : public ConstDfsHloVisitorWithDefault {
 public:
  explicit GemmConfigSetCollector(GemmFusionAutotunerImpl* impl)
      : impl_(impl) {}

  absl::StatusOr<
      absl::flat_hash_map<const HloFusionInstruction*, std::vector<Config>>>
  CollectGemmConfigSets(
      const HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {}) {
    gemm_config_sets_.clear();
    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      TF_RETURN_IF_ERROR(computation->Accept(this));
    }
    return std::move(gemm_config_sets_);
  }

  absl::Status HandleFusion(const HloInstruction* hlo) override {
    const HloFusionInstruction* fusion = Cast<HloFusionInstruction>(hlo);

    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        hlo->backend_config<GpuBackendConfig>());
    const FusionBackendConfig& backend_config =
        gpu_config.fusion_backend_config();

    AutotuneCacheKey key = AutotunerUtil::GetKey(hlo, impl_->GetConfig());
    if (AutotunerUtil::IsInCache(key) || handled_fusions_.contains(key)) {
      return absl::OkStatus();
    }

    bool missing_config = (backend_config.kind() == kTritonGemmFusionKind &&
                           !backend_config.has_triton_gemm_config()) ||
                          (backend_config.kind() == kCuDnnFusionKind &&
                           !backend_config.has_cudnn_fusion_config());
    if (missing_config) {
      TF_ASSIGN_OR_RETURN(std::vector<Config> configs,
                          impl_->GenerateConfigs(*fusion));
      TF_RET_CHECK(
          gemm_config_sets_.insert({fusion, std::move(configs)}).second);
    }

    handled_fusions_.insert(key);
    return absl::OkStatus();
  }

  absl::Status DefaultAction(const HloInstruction* hlo) override {
    return absl::OkStatus();
  }

 private:
  GemmFusionAutotunerImpl* impl_;
  absl::flat_hash_map<const HloFusionInstruction*, std::vector<Config>>
      gemm_config_sets_;
  absl::flat_hash_set<AutotuneCacheKey> handled_fusions_;
};

struct TileSizeLimit {
  int block_m = 0;
  int block_n = 0;
  int block_k = 0;
};

absl::StatusOr<TileSizeLimit> GetLimits(const HloDotInstruction& dot,
                                        bool has_8_bit_operand) {
  TF_ASSIGN_OR_RETURN(int64_t non_contracting_index_lhs,
                      NonContractingDimensionIndex(dot, /*operand_number=*/0));
  TF_ASSIGN_OR_RETURN(int64_t non_contracting_index_rhs,
                      NonContractingDimensionIndex(dot, /*operand_number=*/1));
  TF_ASSIGN_OR_RETURN(int64_t contracting_index,
                      ContractingDimensionIndex(dot, /*operand_number=*/1));
  // This is not a sharp upper limit, the actual m value can be much smaller
  // based on how much of the m dimension is physically contiguous.
  // TODO(tdanyluk): Get the exact m value by running a TritonFusionAnalysis.
  const int max_m = tsl::NextPowerOfTwoS64(
      dot.operand(0)->shape().dimensions(non_contracting_index_lhs));
  // Theoretically the same is true as for m, but that is not possible in
  // practice with the current implementation.
  const int max_n = tsl::NextPowerOfTwoS64(
      dot.operand(1)->shape().dimensions(non_contracting_index_rhs));
  // This is before doing the split-k transform.
  const int max_k = tsl::NextPowerOfTwoS64(
      dot.operand(1)->shape().dimensions(contracting_index));

  // TODO(b/337839570): block_k = 16 is bugged in Triton for dots with 8-bit
  // input. Setting minimum to 32 instead of 16 for these cases.
  // TODO(b/337838200): Write the restriction on the minimum tile size to be
  // generic. Currently we only handle the 8-bit case as this was the bug we
  // ran into.
  return TileSizeLimit{
      /*block_m=*/std::max(max_m, kMinTileSize),
      /*block_n=*/std::max(max_n, kMinTileSize),
      /*block_k=*/
      std::max(max_k, has_8_bit_operand ? kDimKMinTileSize : kMinTileSize),
  };
}

int GetLogEveryN() { return VLOG_IS_ON(3) ? 100 : 1000; }

absl::StatusOr<std::unique_ptr<HloModule>> TritonGemmAutotuneExtractor(
    const TritonGemmConfig& config,
    const se::DeviceDescription& gpu_device_info,
    const HloFusionInstruction* fusion, DebugOptions debug_opts,
    bool allow_filtering_kernels_spilling_registers) {
  std::unique_ptr<HloModule> new_module =
      ExtractInstructionIntoNewModule(*fusion);
  // Reduce memory usage during compilation by disabling GPU runtime.
  debug_opts.set_xla_gpu_enable_xla_runtime_executable(false);
  // TODO(anlunx): Disable command buffers for now because it breaks triton
  // autotuner test. Enable this when the function of command buffers is stable.
  debug_opts.clear_xla_gpu_enable_command_buffer();
  if (!allow_filtering_kernels_spilling_registers) {
    debug_opts.set_xla_gpu_filter_kernels_spilling_registers_on_autotuning(
        false);
  }
  new_module->mutable_config().set_debug_options(debug_opts);

  HloComputation* entry_computation = new_module->entry_computation();
  HloInstruction* cloned_dot_fusion = entry_computation->root_instruction();

  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      cloned_dot_fusion->backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();

  *backend_config.mutable_triton_gemm_config() = config.ToProto();
  TF_RETURN_IF_ERROR(cloned_dot_fusion->set_backend_config(gpu_config));

  if (config.split_k > 1) {
    TF_RETURN_IF_ERROR(MakeDotSplitKBatch(cloned_dot_fusion, config));
    GpuFloatSupport bf16_support(gpu_device_info.cuda_compute_capability(),
                                 BF16);
    FloatNormalization float_normalization(&bf16_support);
    TF_RETURN_IF_ERROR(float_normalization.Run(new_module.get()).status());
    GpuInstructionFusion instruction_fusion(/*may_duplicate=*/false,
                                            gpu_device_info);
    TF_RETURN_IF_ERROR(instruction_fusion.Run(new_module.get()).status());
    HloInstruction* root = entry_computation->root_instruction();
    // If the instruction fusion pass above skipped the reduction, turn it
    // into a fusion for a universal set of arguments for execution.
    if (root->opcode() == HloOpcode::kReduce) {
      HloInstruction* fusion_instruction =
          entry_computation->AddInstruction(HloInstruction::CreateFusion(
              root->shape(), ChooseFusionKind(*root, *root), root));
      HloInstruction* init_value = root->mutable_operand(1);
      TF_CHECK_OK(
          entry_computation->ReplaceInstruction(root, fusion_instruction));
      fusion_instruction->FuseInstruction(init_value);
      TF_CHECK_OK(entry_computation->RemoveInstruction(init_value));
    }
  }
  return new_module;
}

absl::StatusOr<std::unique_ptr<HloModule>> CublasGemmAutotuneExtractor(
    const AutotuneConfig& config, const HloFusionInstruction* fusion,
    const DebugOptions& debug_opts) {
  const HloComputation* fusion_computation =
      fusion->called_computations().at(0);
  std::unique_ptr<HloModule> new_module =
      ExtractComputationIntoNewModule(*fusion_computation);
  new_module->mutable_config().set_debug_options(debug_opts);

  auto* dot = hlo_query::GetFirstInstructionWithOpcode(
      *new_module->entry_computation(), HloOpcode::kDot);
  // Substitute algorithms, which are not supported by cuBLAS for the check, but
  // don't use cuBlas in the end. This assumes that the substituting algorithm
  // has result which are close enough for the check in this file.
  if (dot->precision_config().algorithm() ==
          PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3 ||
      dot->precision_config().algorithm() ==
          PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6) {
    dot->mutable_precision_config()->set_algorithm(
        PrecisionConfig::ALG_DOT_F32_F32_F32);
  }

  GemmRewriter rewriter(config.GetGpuComputeCapability());
  GpuInstructionFusion fusion_pass(
      /*may_duplicate=*/false, config.GetExecutor()->GetDeviceDescription());
  TF_RETURN_IF_ERROR(rewriter.Run(new_module.get()).status());
  TF_RETURN_IF_ERROR(fusion_pass.Run(new_module.get()).status());
  // TODO(tdanyluk): Consider running GemmAlgorithmPicker here for better cuBLAS
  // performance. It is probably not needed on Ampere and later because cuBLAS
  // ignores the algorithm parameter for those targets. If we run
  // GemmAlgorithmPicker, we probably should not run this in parallel with other
  // compilations.
  return new_module;
}

absl::StatusOr<std::unique_ptr<HloModule>> CudnnGemmAutotuneExtractor(
    const AutotuneConfig& autotune_config, const HloFusionInstruction* fusion,
    const DebugOptions& debug_opts, const int plan_id) {
  std::unique_ptr<HloModule> new_module =
      ExtractInstructionIntoNewModule(*fusion);
  new_module->mutable_config().set_debug_options(debug_opts);

  GpuBackendConfig gpu_config;
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind(std::string(kCuDnnFusionKind));
  // Provided a plan ID the autotuner just compiles one plan.
  backend_config.mutable_cudnn_fusion_config()->set_plan_id(plan_id);
  TF_RETURN_IF_ERROR(
      new_module->entry_computation()->root_instruction()->set_backend_config(
          gpu_config));

  return new_module;
}

bool IsFusionKind(const HloInstruction& hlo, absl::string_view kind) {
  auto gpu_config = hlo.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return false;
  }
  return gpu_config->fusion_backend_config().kind() == kind;
}

int GetCuDnnPlanCount(const HloInstruction& hlo,
                      const AutotuneConfig& autotune_config) {
  if (auto gpu_config = hlo.backend_config<GpuBackendConfig>();
      !gpu_config.ok() ||
      gpu_config->fusion_backend_config().has_cudnn_fusion_config()) {
    return {};
  }
  return CuDnnFusionCompiler::GetAvailablePlanCount(
      *autotune_config.GetExecutor(), *DynCast<HloFusionInstruction>(&hlo));
}

AutotuneResult FromConfig(const Config& config) {
  AutotuneResult res;
  if (std::holds_alternative<GemmFusionAutotunerImpl::CuBlasConfig>(config)) {
    res.mutable_gemm()->set_algorithm(CUBLAS_GEMM_DEFAULT);
  } else if (std::holds_alternative<GemmFusionAutotunerImpl::CuDnnConfig>(
                 config)) {
    res.mutable_algorithm()->set_algo_id(
        std::get<GemmFusionAutotunerImpl::CuDnnConfig>(config).plan_id);
  } else if (std::holds_alternative<TritonGemmConfig>(config)) {
    *res.mutable_triton() = std::get<TritonGemmConfig>(config).ToProto();
  } else {
    LOG(FATAL) << "Unsupported config type: " << config.index();
  }
  return res;
}

absl::Status DumpAutotunedFusion(const AutotuneConfig& autotune_config,
                                 AutotunerCompileUtil& util,
                                 const AutotuneResult result,
                                 const HloFusionInstruction* fusion,
                                 int fusion_id) {
  TritonGemmConfig triton_gemm_config;
  if (!result.has_triton()) {
    LOG(WARNING) << "Using empty triton GEMM config for op " << fusion->name();
    // Empty TritonGemmConfig has all zero values which is good enough to keep
    // fused computation in the dump but illustrate that Triton is not used for
    // it after autotuning.
  } else {
    TF_ASSIGN_OR_RETURN(triton_gemm_config,
                        TritonGemmConfig::FromProto(result.triton()));
  }
  const se::DeviceDescription& device_desc =
      autotune_config.GetExecutor()->GetDeviceDescription();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      util.ExtractModule([&](const DebugOptions& debug_opts) {
        if (result.has_algorithm()) {
          return CudnnGemmAutotuneExtractor(autotune_config, fusion, debug_opts,
                                            result.algorithm().algo_id());
        } else if (result.has_triton()) {
          return TritonGemmAutotuneExtractor(
              triton_gemm_config, device_desc, fusion, debug_opts,
              /*allow_filtering_kernels_spilling_registers=*/true);
        } else if (result.has_gemm()) {
          return CublasGemmAutotuneExtractor(autotune_config, fusion,
                                             debug_opts);
        } else {
          LOG(FATAL) << "Unknown result type: " << result.DebugString();
        }
      }));
  module->set_name(std::string(fusion->name()));
  // Using the original module for its debug info and name in the first
  // parameter. It's better to include the name of both the original module
  // and the extracted module, to avoid name clashes.
  DumpToFileInDirOrStdout(
      /*module=*/*fusion->GetModule(),
      /*file_prefix=*/"",
      /*file_suffix=*/
      absl::StrCat("triton_fusion_", fusion_id, ".", module->name(),
                   ".optimized.txt"),
      /*contents=*/module->ToString());
  return absl::OkStatus();
}

}  // anonymous namespace

// Methods required for sorting the configs.
bool GemmFusionAutotunerImpl::CuBlasConfig::operator<(
    const CuBlasConfig& other) const {
  return false;
}
bool GemmFusionAutotunerImpl::CuDnnConfig::operator<(
    const CuDnnConfig& other) const {
  return plan_id < other.plan_id;
}

bool GemmFusionAutotunerImpl::IsAutotuningEnabled() const {
  return debug_options_.xla_gpu_autotune_level() > 0 &&
         !debug_options_.xla_gpu_deterministic_ops();
}

/*static*/ std::string GemmFusionAutotunerImpl::ToString(const Config& config) {
  if (std::holds_alternative<TritonGemmConfig>(config)) {
    return std::get<TritonGemmConfig>(config).ToString();
  } else if (std::holds_alternative<CuDnnConfig>(config)) {
    return absl::StrFormat("cuDNN plan %d",
                           std::get<CuDnnConfig>(config).plan_id);
  } else if (std::holds_alternative<CuBlasConfig>(config)) {
    return "reference (cublas)";
  } else {
    LOG(FATAL) << "Unsupported config type: " << config.index();
  }
}

absl::StatusOr<std::vector<Config>> GemmFusionAutotunerImpl::GenerateConfigs(
    const HloFusionInstruction& fusion) {
  const HloDotInstruction* dot =
      Cast<HloDotInstruction>(hlo_query::GetFirstInstructionWithOpcode(
          *fusion.called_computations().at(0), HloOpcode::kDot));

  // Add cuBLAS reference config, if available.
  std::vector<Config> configs;
  if (algorithm_util::IsSupportedByCublasOrCublasLt(
          dot->precision_config().algorithm()) &&
      !dot->sparse_operands() && IsAutotuningEnabled()) {
    configs.push_back(CuBlasConfig{});
  }

  // Add cuDNN plans, if available.
  bool is_hopper =
      !config_.IsDeviceless() && GetComputeCapability().IsAtLeastHopper();
  bool is_cudnn_enabled =
      debug_options_.xla_gpu_cudnn_gemm_fusion_level() > 0 && is_hopper &&
      GetDnnVersionInfo(config_.GetExecutor()).major_version() >= 9;
  if ((IsFusionKind(fusion, kCuDnnFusionKind) && IsAutotuningEnabled()) ||
      (IsFusionKind(fusion, kTritonGemmFusionKind) && is_cudnn_enabled &&
       algorithm_util::IsSupportedByCudnn(
           dot->precision_config().algorithm()) &&
       !dot->sparse_operands() && IsAutotuningEnabled())) {
    const int plan_count = GetCuDnnPlanCount(fusion, config_);
    for (int plan_id = 0; plan_id < plan_count; ++plan_id) {
      configs.push_back(CuDnnConfig{plan_id});
    }
  }
  if (IsFusionKind(fusion, kCuDnnFusionKind)) {
    if (!IsAutotuningEnabled()) {
      configs.push_back(CuDnnConfig{-1});
    }
    return configs;
  }

  // Add triton configs.
  TF_ASSIGN_OR_RETURN(std::vector<TritonGemmConfig> triton_configs,
                      GenerateTritonConfigs(*dot));
  for (TritonGemmConfig& config : triton_configs) {
    configs.push_back(std::move(config));
  }
  return configs;
}

absl::StatusOr<std::vector<TritonGemmConfig>>
GemmFusionAutotunerImpl::GenerateTritonConfigs(const HloDotInstruction& dot) {
  bool has_8_bit_operand = HloAnyOf({&dot}, [&](const HloInstruction* node) {
    if (node->opcode() != HloOpcode::kConvert) {
      return false;
    }
    auto in_type = node->operand(0)->shape().element_type();
    return (in_type == PrimitiveType::S8) || (in_type == PrimitiveType::U8);
  });

  std::vector<TritonGemmConfig> result_configs;
  TF_ASSIGN_OR_RETURN(TileSizeLimit limits, GetLimits(dot, has_8_bit_operand));

  // Generate the list of configurations (once).
  if (triton_configs_.empty()) {
    triton_configs_ = !IsAutotuningEnabled()
                          ? std::vector(1, kDefaultGemmTiling)
                      : debug_options_.xla_gpu_exhaustive_tiling_search()
                          ? GetExhaustiveTritonConfigs(has_8_bit_operand)
                          : GetDefaultTritonConfigs();
  }

  // Avoid autotuning tiny fusions.
  constexpr int kMinGemmElements = 32 * 32;
  bool small_dot =
      ShapeUtil::ElementsIn(dot.operand(0)->shape()) <= kMinGemmElements &&
      ShapeUtil::ElementsIn(dot.operand(1)->shape()) <= kMinGemmElements;
  std::vector<TritonGemmConfig> triton_configs =
      small_dot ? std::vector(1, kDefaultGemmTiling) : triton_configs_;

  // Split-K optimization enables more even utilization of a GPU in cases
  // where tiling just the non-contracting dimensions of a GEMM does not create
  // a sufficient number of thread block programs to occupy all available cores.
  // Around 5 full waves completely avoid the need for split-K.
  // n_tiles = split_k * (M * N) / (block_m * block_n)
  const int kCoreCount =
      !config_.IsDeviceless()
          ? config_.GetExecutor()->GetDeviceDescription().core_count()
          : 100;  // some sensible default
  const int64_t kSufficientNumberOfTiles = kMaxWavesForSplitK * kCoreCount;
  const int64_t result_size = ShapeUtil::ElementsIn(dot.shape());

  // Triton configurations are adjusted and deduplicated.
  absl::flat_hash_set<TritonGemmConfig> added;
  bool is_hopper =
      !config_.IsDeviceless() && GetComputeCapability().IsAtLeastHopper();
  for (TritonGemmConfig& config : triton_configs) {
    config.block_m = std::min(config.block_m, limits.block_m);
    config.block_n = std::min(config.block_n, limits.block_n);
    config.block_k = std::min(config.block_k, limits.block_k);
    int max_split_k = 1;
    if (debug_options_.xla_gpu_enable_split_k_autotuning()) {
      int64_t ratio = kSufficientNumberOfTiles * config.block_m *
                      config.block_n / result_size;
      max_split_k = 1 << std::max<int>(tsl::Log2Floor64(ratio), 0);
    }
    config.split_k = std::min(config.split_k, max_split_k);

    // Hopper `wgmma` instruction requires at least 4 warps and 64 elements
    // for LHS tile height.
    if (is_hopper) {
      config.block_m = std::max(config.block_m, 64);
      config.num_warps = std::max(config.num_warps, 4);
    }

    // Sparse meta should have at least one element per thread.
    // Note: only 2:4 structured sparsity is currently supported.
    if (dot.sparse_operands()) {
      config.block_k = std::max(config.block_k, 4 * kMinTileSize);
      int meta_elements = config.block_m * config.block_k / 16;
      config.num_warps =
          std::min<int>(config.num_warps, meta_elements / WarpSize());
    }

    if (added.insert(config).second) {
      result_configs.push_back(config);
    }
  }
  return result_configs;
}

absl::StatusOr<absl::flat_hash_map<
    const HloFusionInstruction*,
    std::vector<GemmFusionAutotunerImpl::ExecutableCandidate>>>
GemmFusionAutotunerImpl::CompileAll(
    AutotunerCompileUtil& compile_util,
    const absl::flat_hash_map<const HloFusionInstruction*, std::vector<Config>>&
        task) {
  tsl::profiler::ScopedAnnotation annotation("XlaAutotunerCompilation");
  absl::Mutex results_mu;
  absl::flat_hash_map<const HloFusionInstruction*,
                      std::vector<ExecutableCandidate>>
      results;
  if (task.empty()) {
    return results;
  }

  const int log_every_n = GetLogEveryN();
  int64_t config_count = 0;
  for (const auto& key_value : task) {
    config_count += key_value.second.size();
  }

  std::atomic<int> done_count = 0;
  std::atomic<int> good_count = 0;
  auto log = [&](bool success) {
    const int done_so_far = done_count.fetch_add(1) + 1;
    const int good_so_far =
        success ? good_count.fetch_add(1) + 1 : good_count.load();
    if (done_so_far % log_every_n == 0) {
      VLOG(2) << "Compiled " << done_so_far << " of " << config_count
              << " configs (successful: " << good_so_far << ")";
    }
  };

  auto compile = [&](const HloFusionInstruction* fusion, const Config& config,
                     bool allow_filtering_kernels_spilling_registers)
      -> absl::StatusOr<bool> {
    std::unique_ptr<Executable> executable;
    if (std::holds_alternative<TritonGemmConfig>(config)) {
      TF_ASSIGN_OR_RETURN(
          executable, compile_util.Compile([&](const DebugOptions& opts) {
            return TritonGemmAutotuneExtractor(
                std::get<TritonGemmConfig>(config),
                config_.GetExecutor()->GetDeviceDescription(), fusion, opts,
                allow_filtering_kernels_spilling_registers);
          }));
    } else if (std::holds_alternative<CuDnnConfig>(config)) {
      executable = compile_util
                       .Compile([&](const DebugOptions& opts) {
                         return CudnnGemmAutotuneExtractor(
                             config_, fusion, opts,
                             std::get<CuDnnConfig>(config).plan_id);
                       })
                       .value_or(nullptr);
    } else if (std::holds_alternative<CuBlasConfig>(config)) {
      TF_ASSIGN_OR_RETURN(
          executable, compile_util.Compile([&](const DebugOptions& opts) {
            return CublasGemmAutotuneExtractor(config_, fusion, opts);
          }));
    } else {
      LOG(FATAL) << "Unsupported config type: " << config.index();
    }
    if (executable != nullptr) {
      absl::MutexLock lock(&results_mu);
      results[fusion].push_back({config, std::move(executable)});
      return true;
    }
    return false;
  };

  // If the thread pool has only one thread, then it is actually slower to
  // offload the tasks there.
  if (thread_pool_ && thread_pool_->NumThreads() > 1 &&
      debug_options_.xla_gpu_force_compilation_parallelism() != 1) {
    if (task.size() == 1) {
      absl::string_view fusion_name = task.begin()->first->name();
      VLOG(1) << "Compiling " << config_count << " configs for " << fusion_name
              << " on " << thread_pool_->NumThreads() << " threads.";
    } else {
      VLOG(1) << "Compiling " << config_count << " configs for " << task.size()
              << " fusions on " << thread_pool_->NumThreads() << " threads.";
    }

    tsl::BlockingCounter counter(config_count);
    for (const auto& key_value : task) {
      const HloFusionInstruction* fusion = key_value.first;
      const std::vector<Config>& gemm_config_set = key_value.second;

      for (const Config& config : gemm_config_set) {
        thread_pool_->Schedule([&, fusion] {
          absl::StatusOr<bool> has_executable =
              compile(fusion, config, gemm_config_set.size() > 1);
          TF_CHECK_OK(has_executable.status())
              << "Failure occured when compiling fusion " << fusion->name()
              << " with config '" << ToString(config)
              << "'\nFused HLO computation:\n"
              << fusion->fused_instructions_computation()->ToString();
          log(has_executable.value());
          counter.DecrementCount();
        });
      }
    }
    counter.Wait();
  } else {
    if (task.size() == 1) {
      absl::string_view fusion_name = task.begin()->first->name();
      LOG(WARNING) << "Compiling " << config_count << " configs for "
                   << fusion_name << " on a single thread.";
    } else {
      LOG(WARNING) << "Compiling " << config_count << " configs for "
                   << task.size() << " fusions on a single thread.";
    }

    for (const auto& key_value : task) {
      const HloFusionInstruction* fusion = key_value.first;
      const auto& gemm_config_set = key_value.second;

      VLOG(2) << "Compiling the fusion: " << fusion->name();
      for (const Config& config : gemm_config_set) {
        VLOG(5) << "Trying configuration: " << ToString(config);
        TF_ASSIGN_OR_RETURN(
            bool has_executable,
            compile(fusion, config, gemm_config_set.size() > 1));
        log(has_executable);
      }
    }
  }

  VLOG(1) << "Done compiling (successful: " << good_count.load() << ").";
  return results;
}

absl::StatusOr<std::vector<AutotuneResult>> GemmFusionAutotunerImpl::Profile(
    AutotunerCompileUtil& compile_util, const HloFusionInstruction& fusion,
    absl::Span<const ExecutableCandidate> candidates) {
  const HloComputation* fusion_computation = fusion.called_computations().at(0);

  se::StreamExecutor* stream_exec = config_.GetExecutor();
  if (!stream_exec->SynchronizeAllActivity()) {
    return Internal("Failed to synchronize GPU for autotuning.");
  }
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaAutotunerMeasurement:#hlo_op=%s#",
                           fusion.name());
  });
  se::DeviceMemoryAllocator* allocator = config_.GetAllocator();
  std::unique_ptr<se::DeviceMemoryAllocator> owned_allocator;
  if (allocator == nullptr) {
    owned_allocator =
        std::make_unique<se::StreamExecutorMemoryAllocator>(stream_exec);
    allocator = owned_allocator.get();
  }
  TF_ASSIGN_OR_RETURN(se::Stream* const stream, config_.GetStream());

  const HloInstruction& root = *fusion_computation->root_instruction();
  BufferComparator comparator(root.shape(),
                              fusion_computation->parent()->config());

  TF_ASSIGN_OR_RETURN(auto rz_buffers,
                      RedzoneBuffers::FromInstruction(
                          *fusion_computation->FusionInstruction(), config_,
                          debug_options_, RedzoneBuffers::kAllInputs));

  const int log_every_n = GetLogEveryN();
  std::vector<AutotuneResult> results;
  std::optional<ScopedShapedBuffer> reference_buffer;
  for (const ExecutableCandidate& candidate : candidates) {
    VLOG(5) << "Trying : " << ToString(candidate.config);
    AutotuneResult res = FromConfig(candidate.config);

    std::optional<ProfilingOutput> profiling_output;
    if (IsAutotuningEnabled()) {
      TF_ASSIGN_OR_RETURN(
          profiling_output,
          compile_util.ProfileExecutable(candidate.executable.get(), stream,
                                         rz_buffers.input_buffers(),
                                         rz_buffers.input_shapes()));
      if (std::holds_alternative<CuBlasConfig>(candidate.config) &&
          config_.should_check_correctness()) {
        reference_buffer = std::move(profiling_output->output);
      }

      int ran_so_far = results.size() + 1;
      if (ran_so_far % log_every_n == 0) {
        VLOG(2) << "Ran " << ran_so_far << " configs of " << candidates.size()
                << ".";
      }
      if (!profiling_output) {
        VLOG(5) << "Skipping this tiling.";
        continue;
      }

      VLOG(5) << "Running the kernel took: " << profiling_output->duration;
      if (profiling_output->duration >= absl::Seconds(1)) {
        LOG(WARNING) << "Slow kernel for " << fusion.name()
                     << " took: " << profiling_output->duration << ". "
                     << ToString(candidate.config);
      }
      *res.mutable_run_time() =
          tsl::proto_utils::ToDurationProto(profiling_output->duration);
    }

    // Reference buffer is available when `config.should_check_correctness()`
    // is set and reference executable was compiled.
    if (reference_buffer.has_value() &&
        !std::holds_alternative<CuBlasConfig>(candidate.config)) {
      TF_ASSIGN_OR_RETURN(
          se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
          rz_buffers.RedzoneAllocator().CheckRedzones());
      if (!rz_check_status.ok()) {
        LOG(ERROR) << "Red zone modified";
        res.mutable_failure()->set_kind(AutotuneResult::REDZONE_MODIFIED);
        res.mutable_failure()->set_msg(rz_check_status.RedzoneFailureMsg());
        CHECK(!config_.should_crash_on_check_failure());
        continue;
      }

      TF_ASSIGN_OR_RETURN(
          bool outputs_match,
          comparator.CompareEqual(
              stream, /*current=*/profiling_output->output.root_buffer(),
              /*expected=*/reference_buffer->root_buffer()));
      if (!outputs_match) {
        const char kMessage[] =
            "Results do not match the reference. This is likely a "
            "bug/unexpected loss of precision.";
        LOG(ERROR) << kMessage;
        CHECK(!config_.should_crash_on_check_failure());
        // WRONG_RESULT is not taken seriously by PickBestResult(), so
        // use DISQUALIFIED.
        res.mutable_failure()->set_kind(AutotuneResult::DISQUALIFIED);
        res.mutable_failure()->set_msg(kMessage);
      }
    }
    results.push_back(std::move(res));
  }
  VLOG(2) << "Done running.";
  return results;
}

std::vector<TritonGemmConfig>
GemmFusionAutotunerImpl::GetExhaustiveTritonConfigs(
    bool has_8_bit_operand) const {
  std::vector<TritonGemmConfig> configs;
  se::CudaComputeCapability cc = GetComputeCapability();
  bool tune_ctas =
      debug_options_.xla_gpu_enable_triton_hopper() && cc.IsAtLeastHopper();

  // TODO(b/337839570): block_k = 16 is bugged in Triton for dots with 8-bit
  // input. Setting minimum to 32 instead of 16 for these cases.
  // TODO(b/337838200): Write the restriction on the minimum tile size to be
  // generic. Currently we only handle the 8-bit case as this was the bug we
  // ran into.
  auto kDimBlockSizes = has_8_bit_operand ? kDimKBlockSizes : kBlockSizes;
  for (int num_stages : kNumStages) {
    // Volta doesn't support num_stages > 2.
    if (!cc.IsAtLeastAmpere() && num_stages > 2) {
      break;
    }
    for (int tile_m : kBlockSizes) {
      for (int tile_n : kBlockSizes) {
        for (int tile_k : kDimBlockSizes) {
          const int tile_lhs = tile_m * tile_k;
          const int tile_rhs = tile_k * tile_n;
          for (int num_warps : kNumWarps) {
            // Each thread should read at least one input element.
            if (num_warps * WarpSize() > std::min(tile_lhs, tile_rhs)) {
              break;
            }
            for (int split_k : kSplitK) {
              // Split-K autotuning may be disabled by a flag.
              if (!debug_options_.xla_gpu_enable_split_k_autotuning() &&
                  split_k > 1) {
                break;
              }
              for (int num_ctas : kNumCtas) {
                // Clusters are only supported on Hopper.
                // Autotuning this parameter is enabled by a flag.
                if (!tune_ctas && num_ctas > 1) {
                  break;
                }
                if (num_ctas > num_warps) {
                  break;
                }
                configs.push_back(TritonGemmConfig(tile_m, tile_n, tile_k,
                                                   split_k, num_stages,
                                                   num_warps, num_ctas));
              }
            }
          }
        }
      }
    }
  }
  return configs;
}

std::vector<TritonGemmConfig> GemmFusionAutotunerImpl::GetDefaultTritonConfigs()
    const {
  using Config = TritonGemmConfig;
  std::vector<Config> configs = {
      Config(32, 32, 256, 1, 1, 4), Config(64, 32, 32, 16, 1, 4),
      Config(32, 64, 64, 4, 1, 4),  Config(128, 128, 64, 4, 1, 4),
      Config(16, 16, 256, 1, 1, 4), Config(16, 128, 32, 16, 1, 4),
      Config(16, 64, 128, 1, 1, 4), Config(16, 128, 32, 8, 1, 4),
      Config(16, 16, 512, 1, 1, 4), Config(32, 16, 512, 1, 1, 4),
      Config(64, 32, 64, 1, 2, 8)};
  if (GetComputeCapability().IsAtLeastAmpere()) {
    absl::c_copy(
        std::vector<Config>{
            Config(128, 256, 32, 1, 3, 8),  Config(256, 128, 32, 1, 3, 8),
            Config(256, 64, 32, 1, 4, 4),   Config(64, 256, 32, 1, 4, 4),
            Config(128, 64, 32, 1, 4, 4),   Config(64, 128, 32, 1, 4, 4),
            Config(256, 128, 128, 1, 3, 8), Config(256, 64, 128, 1, 4, 4),
            Config(64, 256, 128, 1, 4, 4),  Config(128, 128, 128, 1, 4, 4),
            Config(128, 64, 64, 1, 4, 4),   Config(64, 128, 64, 1, 4, 4),
            Config(128, 32, 64, 1, 4, 4),   Config(64, 32, 64, 1, 4, 4),
            Config(32, 128, 32, 1, 4, 4),   Config(128, 128, 32, 1, 4, 4),
            Config(16, 16, 256, 1, 3, 4),   Config(128, 128, 64, 2, 1, 8),
            Config(64, 64, 64, 1, 2, 4),    Config(16, 64, 256, 8, 1, 4),
            Config(256, 256, 128, 1, 3, 8)},
        std::back_inserter(configs));
  }
  if (GetComputeCapability().IsAtLeastHopper()) {
    absl::c_copy(
        std::vector<Config>{
            Config(16, 32, 32, 8, 1, 2),
            Config(16, 64, 128, 8, 1, 4),
            Config(16, 64, 128, 16, 3, 4),
        },
        std::back_inserter(configs));
  }
  return configs;
}

absl::Status GemmFusionAutotunerImpl::Autotune(
    AutotunerCompileUtil& compile_util,
    const absl::flat_hash_map<const HloFusionInstruction*, std::vector<Config>>&
        gemm_config_sets) {
  TF_ASSIGN_OR_RETURN(auto executable_sets,
                      CompileAll(compile_util, gemm_config_sets));

  // Sort the candidates to make their execution order well-defined for each
  // fusion.
  for (auto& key_value : executable_sets) {
    absl::c_sort(key_value.second, [](const auto& a, const auto& b) {
      return a.config < b.config;
    });
  }

  int fusion_id = 0;
  for (const auto& key_value : executable_sets) {
    const HloFusionInstruction* fusion = key_value.first;
    TF_ASSIGN_OR_RETURN(
        std::vector<AutotuneResult> results,
        Profile(compile_util, *fusion, /*candidates=*/key_value.second));

    // The reference config (if it exists) will be the first in the results,
    // due to how sorting the variants work.
    if (!debug_options_.xla_gpu_cublas_fallback() &&
        results.front().has_gemm()) {
      results.erase(results.begin());
    }

    const HloInstruction* root =
        fusion->called_computations().at(0)->root_instruction();
    TF_ASSIGN_OR_RETURN(
        AutotuneResult best,
        PickBestResult(results, root->ToString(), root->GetModule()->config()));
    VLOG(2) << "Best time: "
            << tsl::proto_utils::FromDurationProto(best.run_time());

    if (debug_options_.xla_gpu_dump_autotuned_gemm_fusions()) {
      TF_RETURN_IF_ERROR(DumpAutotunedFusion(config_, compile_util, best,
                                             fusion, fusion_id++));
    }

    const AutotuneCacheKey key = AutotunerUtil::GetKey(fusion, config_);
    if (!AutotunerUtil::AddResult(key, std::move(best))) {
      // In the context of model server, concurrent autotuning is expected and
      // insertion of identical autotuning keys is accepted.
      LOG(WARNING) << "AutotunerUtil::AddResult already existed: "
                   << key.ToString();
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> GemmFusionAutotuner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER("GEMM fusion autotuner");

  const DebugOptions& debug_options = module->config().debug_options();
  GemmFusionAutotunerImpl autotuner(config_, debug_options, thread_pool_);
  GemmConfigSetCollector gemm_config_set_collector(&autotuner);
  absl::flat_hash_map<const HloFusionInstruction*, std::vector<Config>>
      gemm_config_sets;
  TF_ASSIGN_OR_RETURN(gemm_config_sets,
                      gemm_config_set_collector.CollectGemmConfigSets(
                          module, execution_threads));

  if (!autotuner.IsAutotuningEnabled()) {
    // Pick the first option for each gemm instead of autotuning.
    for (const auto& [fusion, tilings] : gemm_config_sets) {
      const AutotuneCacheKey key = AutotunerUtil::GetKey(fusion, config_);
      AutotuneResult res = FromConfig(tilings[0]);
      *res.mutable_run_time() =
          tsl::proto_utils::ToDurationProto(absl::ZeroDuration());
      AutotunerUtil::AddResult(key, res);
    }
  } else if (!config_.IsDeviceless()) {
    TF_ASSIGN_OR_RETURN(std::optional<AutotunerCompileUtil> opt_compile_util,
                        AutotunerCompileUtil::Create(config_, debug_options));
    TF_RET_CHECK(opt_compile_util.has_value());
    std::string correctness_check_str = config_.should_check_correctness()
                                            ? "(with correctness check)"
                                            : "(without correctness check)";

    VLOG(1) << "Autotuning " << gemm_config_sets.size() << " fusions "
            << correctness_check_str << ".";
    TF_RETURN_IF_ERROR(autotuner.Autotune(*opt_compile_util, gemm_config_sets));
    VLOG(1) << "Done autotuning.";
  }

  return GemmFusionAutotunerVisitor(config_).RunOnModule(module,
                                                         execution_threads);
}

}  // namespace gpu
}  // namespace xla
