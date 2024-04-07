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
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
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

using ProfilingOutput = AutotunerCompileUtil::ProfilingOutput;

namespace {

// Currently supported minimum tile size.
constexpr int kMinTileSize = 16;
// Not a hard limit, just an assumption that should stay valid.
constexpr int kMaxTileSize = 512;

// Default tiling when autotuning is disabled.
constexpr TritonGemmConfig kDefaultGemmTiling = {32, 32, 32, 1, 1, 4};

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

// This contains all alternative Triton GEMM configs related to one fusion.
struct GemmConfigSet {
  std::vector<TritonGemmConfig> configs;
  // Setting this to true disallows verification and fallback to cuBLAS, and
  // the usage of cuDNN.
  bool has_sparsity = false;
};

using CuDnnPlanId = int64_t;

struct ExecutableCandidate {
  std::variant<TritonGemmConfig, CuDnnPlanId> config;
  // Not nullptr.
  std::unique_ptr<Executable> executable;
};

// This contains all alternative executables related to one fusion.
struct ExecutableSet {
  std::vector<ExecutableCandidate> candidates;
  // Not nullptr.
  std::unique_ptr<Executable> reference;
};

class GemmConfigSetCollector : public ConstDfsHloVisitorWithDefault {
 public:
  explicit GemmConfigSetCollector(const AutotuneConfig& config)
      : config_(config) {}

  absl::StatusOr<
      absl::flat_hash_map<const HloFusionInstruction*, GemmConfigSet>>
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

    AutotuneCacheKey key = AutotunerUtil::GetKey(hlo, config_);
    if (AutotunerUtil::IsInCache(key) || handled_fusions_.contains(key)) {
      return absl::OkStatus();
    }

    if (backend_config.kind() == kTritonGemmFusionKind &&
        !backend_config.has_triton_gemm_config()) {
      TF_ASSIGN_OR_RETURN(GemmConfigSet gemm_config_set,
                          GetGemmConfigSet(fusion));
      TF_RET_CHECK(
          gemm_config_sets_.insert({fusion, std::move(gemm_config_set)})
              .second);
    } else if (backend_config.kind() == kCuDnnFusionKind &&
               !backend_config.has_cudnn_fusion_config()) {
      TF_RET_CHECK(gemm_config_sets_.insert({fusion, {}}).second);
    }

    handled_fusions_.insert(key);
    return absl::OkStatus();
  }

  absl::Status DefaultAction(const HloInstruction* hlo) override {
    return absl::OkStatus();
  }

 private:
  absl::StatusOr<GemmConfigSet> GetGemmConfigSet(
      const HloFusionInstruction* fusion) {
    const DebugOptions& debug_options =
        fusion->GetModule()->config().debug_options();
    auto cuda_comp =
        std::get<se::CudaComputeCapability>(config_.GetGpuComputeCapability());
    const HloDotInstruction* dot_instr =
        Cast<HloDotInstruction>(hlo_query::GetFirstInstructionWithOpcode(
            *fusion->called_computations().at(0), HloOpcode::kDot));
    TF_ASSIGN_OR_RETURN(auto configs, GetPossibleMatmulAutotuneConfigs(
                                          *dot_instr, cuda_comp, debug_options,
                                          config_.ExhaustiveTilingSearch()));
    return GemmConfigSet{std::move(configs),
                         /*has_sparsity=*/dot_instr->sparse_operands() > 0};
  }

  AutotuneConfig config_;
  absl::flat_hash_map<const HloFusionInstruction*, GemmConfigSet>
      gemm_config_sets_;
  absl::flat_hash_set<AutotuneCacheKey> handled_fusions_;
};

struct TileSizeLimit {
  int64_t block_m = 0;
  int64_t block_n = 0;
  int64_t block_k = 0;
};

absl::StatusOr<TileSizeLimit> GetUpperLimit(const HloDotInstruction& dot) {
  TF_ASSIGN_OR_RETURN(int64_t non_contracting_index0,
                      NonContractingDimensionIndex(dot, /*operand_number=*/0));
  TF_ASSIGN_OR_RETURN(int64_t non_contracting_index1,
                      NonContractingDimensionIndex(dot, /*operand_number=*/1));
  TF_ASSIGN_OR_RETURN(int64_t contracting_index0,
                      ContractingDimensionIndex(dot, /*operand_number=*/0));
  // This is not a sharp upper limit, the actual m value can be much smaller
  // based on how much of the m dimension is physically contiguous.
  // TODO(tdanyluk): Get the exact m value by running a TritonFusionAnalysis.
  const int64_t m = dot.operand(0)->shape().dimensions(non_contracting_index0);
  // Theoretically the same is true as for m, but that is not possible in
  // practice with the current implementation.
  const int64_t n = dot.operand(1)->shape().dimensions(non_contracting_index1);
  // This is before doing the split-k transform.
  const int64_t k = dot.operand(0)->shape().dimensions(contracting_index0);
  const int64_t block_m_limit =
      std::max<int64_t>(tsl::NextPowerOfTwoS64(m), kMinTileSize);
  const int64_t block_n_limit =
      std::max<int64_t>(tsl::NextPowerOfTwoS64(n), kMinTileSize);
  // Increase minimum tile size for the contracting dimension proportionally
  // to the sparsity multiplier (assume 2:4 structured sparsity).
  const int64_t block_k_limit =
      std::max<int64_t>(tsl::NextPowerOfTwoS64(k),
                        kMinTileSize * (dot.sparse_operands() ? 2 : 1));
  return TileSizeLimit{block_m_limit, block_n_limit, block_k_limit};
}

int64_t GetSplitKLimit(int64_t block_k, int64_t block_k_limit) {
  return std::max<int64_t>(block_k_limit / block_k, 1);
}

// Search space for exhaustive matmul autotuning.
constexpr std::array<int, 6> BLOCK_SIZES = {16, 32, 64, 128, 256, 512};
constexpr std::array<int, 4> NUM_STAGES = {1, 2, 3, 4};
constexpr std::array<int, 4> NUM_WARPS = {2, 4, 8, 16};
constexpr std::array<int, 5> SPLIT_K = {1, 2, 4, 8, 16};
// This is the number of blocks per cluster.
//
// Clusters have 3 dimensions (x,y,z) and only 1 <= x*y*z <= 16 are supported.
// Triton doesn't support (3,3,1) and possibly other non-"power of 2" values.
// It's possible that some other values may be(come) supported.
constexpr std::array<int, 5> NUM_CTAS = {1, 2, 4, 8, 16};

absl::StatusOr<std::vector<TritonGemmConfig>>
GetExhaustiveMatmulAutotuneConfigs(
    const HloDotInstruction& dot,
    const se::CudaComputeCapability compute_capability, const int max_split_k,
    const DebugOptions& debug_options) {
  TF_ASSIGN_OR_RETURN(const TileSizeLimit limit, GetUpperLimit(dot));
  std::vector<TritonGemmConfig> configs;
  bool mma_layout_v2 =
      compute_capability.IsAtLeast(se::CudaComputeCapability::AMPERE);
  bool enable_hopper_optimizations =
      debug_options.xla_gpu_enable_triton_hopper() &&
      compute_capability.IsAtLeast(se::CudaComputeCapability::HOPPER);

  for (int num_warps : NUM_WARPS) {
    for (int num_stages : NUM_STAGES) {
      // Volta doesn't support num_stages > 2.
      if (!mma_layout_v2 && num_stages > 2) {
        continue;
      }
      for (int block_m : BLOCK_SIZES) {
        if (block_m > limit.block_m) {
          continue;
        }
        for (int block_n : BLOCK_SIZES) {
          if (block_n > limit.block_n) {
            continue;
          }
          for (int block_k : BLOCK_SIZES) {
            if (block_k > limit.block_k) {
              continue;
            }
            // Sparse meta should have at least one element per thread.
            // Note: only 2:4 structured sparsity is currently supported.
            if (dot.sparse_operands() &&
                block_m * block_k / 16 < num_warps * WarpSize()) {
              continue;
            }
            for (int split_k : SPLIT_K) {
              if (split_k >
                  std::min<int64_t>(max_split_k,
                                    GetSplitKLimit(block_k, limit.block_k))) {
                continue;
              }
              if (!enable_hopper_optimizations) {
                configs.push_back(TritonGemmConfig(
                    block_m, block_n, block_k, split_k, num_stages, num_warps));
                continue;
              }
              // Arch >= Hopper autotuning.
              // We only want to autotune this if it provides any speedup. So
              // please think about that before adding it to the default
              // autotuning parameters.
              for (int num_ctas : NUM_CTAS) {
                configs.push_back(TritonGemmConfig(block_m, block_n, block_k,
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

std::vector<TritonGemmConfig> GetFixedMatmulAutotuneConfigs(
    const se::CudaComputeCapability compute_capability, const int max_split_k) {
  // Shorter name for better formatting.
  using Config = TritonGemmConfig;
  std::vector<Config> configs = {
      Config(32, 32, 256, 1, 1, 4), Config(64, 32, 32, 16, 1, 4),
      Config(32, 64, 64, 4, 1, 4),  Config(128, 128, 64, 4, 1, 4),
      Config(16, 16, 256, 1, 1, 4), Config(16, 128, 32, 16, 1, 4),
      Config(16, 64, 128, 1, 1, 4), Config(16, 128, 32, 8, 1, 4),
      Config(16, 16, 512, 1, 1, 4), Config(32, 16, 512, 1, 1, 4),
      Config(64, 32, 64, 1, 2, 8)};
  if (compute_capability.IsAtLeast(se::CudaComputeCapability::AMPERE)) {
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
  if (compute_capability.IsAtLeast(se::CudaComputeCapability::HOPPER)) {
    absl::c_copy(
        std::vector<Config>{
            Config(16, 32, 32, 8, 1, 2),
            Config(16, 64, 128, 8, 1, 4),
            Config(16, 64, 128, 16, 3, 4),
        },
        std::back_inserter(configs));
  }
  configs.erase(std::remove_if(configs.begin(), configs.end(),
                               [&](const Config& config) {
                                 return config.split_k > max_split_k;
                               }),
                configs.end());
  return configs;
}

// This prefers to take the parameter by moving it.
absl::StatusOr<std::vector<TritonGemmConfig>> ReduceTileSizes(
    const HloDotInstruction& dot, std::vector<TritonGemmConfig> configs) {
  TF_ASSIGN_OR_RETURN(const TileSizeLimit limit, GetUpperLimit(dot));
  // Decrease the block sizes and split_k if they are unnecessarily big.
  for (TritonGemmConfig& config : configs) {
    config.block_m = std::min<int64_t>(config.block_m, limit.block_m);
    config.block_n = std::min<int64_t>(config.block_n, limit.block_n);
    config.block_k = std::min<int64_t>(config.block_k, limit.block_k);
    config.split_k = std::min<int64_t>(
        config.split_k, GetSplitKLimit(config.block_k, limit.block_k));
    // Sparse meta should have at least one element per thread.
    // Note: only 2:4 structured sparsity is currently supported.
    if (dot.sparse_operands()) {
      int meta_elements = config.block_m * config.block_k / 16;
      config.num_warps =
          std::min<int64_t>(config.num_warps, meta_elements / WarpSize());
    }
  }

  // Remove duplicates.
  absl::flat_hash_set<TritonGemmConfig> configs_so_far;
  configs.erase(std::remove_if(configs.begin(), configs.end(),
                               [&](const TritonGemmConfig& config) {
                                 return !configs_so_far.insert(config).second;
                               }),
                configs.end());
  TF_RET_CHECK(!configs.empty());
  return configs;
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

bool ShouldAllowFilteringKernelsSpillingRegisters(
    const GemmConfigSet& gemm_config_set) {
  return gemm_config_set.configs.size() > 1;
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

bool IsCuDnnEnabled(const AutotuneConfig& config,
                    const DebugOptions& debug_opts) {
  return !config.IsDeviceless() &&
         std::get<se::CudaComputeCapability>(config.GetGpuComputeCapability())
             .IsAtLeastHopper() &&
         debug_opts.xla_gpu_cudnn_gemm_fusion_level() > 0 &&
         GetDnnVersionInfo(config.GetExecutor()).major_version() >= 9;
}

bool HasAlgorithmSupportedByCublasOrCublasLt(
    const HloFusionInstruction& fusion) {
  const PrecisionConfig::Algorithm algorithm =
      hlo_query::GetFirstInstructionWithOpcode(*fusion.called_computation(),
                                               HloOpcode::kDot)
          ->precision_config()
          .algorithm();
  return algorithm_util::IsSupportedByCublasOrCublasLt(algorithm);
}

bool HasAlgorithmSupportedByCudnn(const HloFusionInstruction& fusion) {
  const PrecisionConfig::Algorithm algorithm =
      hlo_query::GetFirstInstructionWithOpcode(*fusion.called_computation(),
                                               HloOpcode::kDot)
          ->precision_config()
          .algorithm();
  return algorithm_util::IsSupportedByCudnn(algorithm);
}

absl::StatusOr<absl::flat_hash_map<const HloFusionInstruction*, ExecutableSet>>
CompileMany(const AutotuneConfig& config, AutotunerCompileUtil& util,
            tsl::thread::ThreadPool* thread_pool,
            const DebugOptions& debug_opts,
            const absl::flat_hash_map<const HloFusionInstruction*,
                                      GemmConfigSet>& gemm_config_sets) {
  absl::Mutex executable_sets_mu;
  absl::flat_hash_map<const HloFusionInstruction*, ExecutableSet>
      executable_sets;

  if (gemm_config_sets.empty()) {
    return executable_sets;
  }

  const se::DeviceDescription& gpu_device_info =
      config.GetExecutor()->GetDeviceDescription();

  const int log_every_n = GetLogEveryN();
  int64_t config_count = 0;
  for (const auto& key_value : gemm_config_sets) {
    const HloFusionInstruction& hlo = *key_value.first;
    const GemmConfigSet& gemm_config_set = key_value.second;

    if (IsFusionKind(hlo, kTritonGemmFusionKind)) {
      config_count += gemm_config_set.configs.size();
      if (!gemm_config_set.has_sparsity && IsCuDnnEnabled(config, debug_opts) &&
          HasAlgorithmSupportedByCudnn(hlo)) {
        config_count += GetCuDnnPlanCount(hlo, config);
      }
    } else if (IsFusionKind(hlo, kCuDnnFusionKind)) {
      config_count += GetCuDnnPlanCount(hlo, config);
    }
    // Reference config for verification (uses cuBLAS).
    config_count += !gemm_config_set.has_sparsity;
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

  // Returns true on success.
  auto compile = [&](const HloFusionInstruction* fusion,
                     const TritonGemmConfig& conf,
                     bool allow_filtering_kernels_spilling_registers)
      -> absl::StatusOr<bool> {
    CHECK_LE(conf.block_m, kMaxTileSize);
    CHECK_LE(conf.block_n, kMaxTileSize);
    CHECK_LE(conf.block_k, kMaxTileSize);
    // TODO(b/296884861): Reenable GPU runtime, when it will have much smaller
    // memory overhead (regarding the size of the executables).
    // We can also remove the force_disable_gpu_runtime argument at that
    // point.
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                        util.Compile([&](const DebugOptions& opts) {
                          return TritonGemmAutotuneExtractor(
                              conf, gpu_device_info, fusion, opts,
                              allow_filtering_kernels_spilling_registers);
                        }));

    if (executable != nullptr) {
      absl::MutexLock lock(&executable_sets_mu);
      ExecutableSet& executable_set = executable_sets[fusion];
      executable_set.candidates.push_back(
          ExecutableCandidate{conf, std::move(executable)});
      return true;
    }

    return false;
  };

  // Returns true on success.
  auto compile_reference_executable =
      [&](const HloFusionInstruction* fusion) -> absl::StatusOr<bool> {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                        util.Compile([&](const DebugOptions& opts) {
                          return CublasGemmAutotuneExtractor(config, fusion,
                                                             opts);
                        }));

    if (executable != nullptr) {
      absl::MutexLock lock(&executable_sets_mu);
      ExecutableSet& executable_set = executable_sets[fusion];
      TF_RET_CHECK(executable_set.reference == nullptr);
      executable_set.reference = std::move(executable);
      return true;
    }

    return false;
  };

  auto compile_cudnn_executable = [&](const HloFusionInstruction* fusion,
                                      const int plan_id) {
    std::unique_ptr<Executable> executable =
        util.Compile([&](const DebugOptions& opts) {
              return CudnnGemmAutotuneExtractor(config, fusion, opts, plan_id);
            })
            .value_or(nullptr);
    if (executable != nullptr) {
      absl::MutexLock lock(&executable_sets_mu);
      ExecutableSet& executable_set = executable_sets[fusion];
      executable_set.candidates.push_back(
          ExecutableCandidate{plan_id, std::move(executable)});
      return true;
    }
    return false;
  };

  // If the thread pool has only one thread, then it is actually slower to
  // offload the tasks there.
  if (thread_pool && thread_pool->NumThreads() > 1 &&
      debug_opts.xla_gpu_force_compilation_parallelism() != 1) {
    if (gemm_config_sets.size() == 1) {
      absl::string_view fusion_name = gemm_config_sets.begin()->first->name();
      VLOG(1) << "Compiling " << config_count << " configs for " << fusion_name
              << " on " << thread_pool->NumThreads() << " threads.";
    } else {
      VLOG(1) << "Compiling " << config_count << " configs for "
              << gemm_config_sets.size() << " fusions on "
              << thread_pool->NumThreads() << " threads.";
    }

    tsl::BlockingCounter counter(config_count);
    for (const auto& key_value : gemm_config_sets) {
      const HloFusionInstruction* fusion = key_value.first;
      const GemmConfigSet& gemm_config_set = key_value.second;

      for (const TritonGemmConfig& conf : gemm_config_set.configs) {
        thread_pool->Schedule([&, fusion] {
          absl::StatusOr<bool> has_executable = compile(
              fusion, conf,
              ShouldAllowFilteringKernelsSpillingRegisters(gemm_config_set));
          TF_CHECK_OK(has_executable.status())
              << "Failure occured when compiling fusion " << fusion->name()
              << " with config '" << conf.ToString()
              << "'\nFused HLO computation:\n"
              << fusion->fused_instructions_computation()->ToString();
          log(has_executable.value());
          counter.DecrementCount();
        });
      }

      if (!gemm_config_set.has_sparsity) {
        thread_pool->Schedule([&, fusion] {
          absl::StatusOr<bool> has_executable =
              compile_reference_executable(fusion);
          TF_CHECK_OK(has_executable.status());
          log(has_executable.value());
          counter.DecrementCount();
        });
      }

      if (IsFusionKind(*fusion, kCuDnnFusionKind) ||
          (IsFusionKind(*fusion, kTritonGemmFusionKind) &&
           !gemm_config_set.has_sparsity &&
           IsCuDnnEnabled(config, debug_opts) &&
           HasAlgorithmSupportedByCudnn(*fusion))) {
        const int plan_count = GetCuDnnPlanCount(*fusion, config);
        for (int plan_id = 0; plan_id < plan_count; ++plan_id) {
          thread_pool->Schedule([&, fusion, plan_id] {
            log(compile_cudnn_executable(fusion, plan_id));
            counter.DecrementCount();
          });
        }
      }
    }
    counter.Wait();
  } else {
    if (gemm_config_sets.size() == 1) {
      absl::string_view fusion_name = gemm_config_sets.begin()->first->name();
      LOG(WARNING) << "Compiling " << config_count << " configs for "
                   << fusion_name << " on a single thread.";

    } else {
      LOG(WARNING) << "Compiling " << config_count << " configs for "
                   << gemm_config_sets.size() << " fusions on a single thread.";
    }

    for (const auto& key_value : gemm_config_sets) {
      const HloFusionInstruction* fusion = key_value.first;
      const GemmConfigSet& gemm_config_set = key_value.second;

      for (const TritonGemmConfig& gemm_config : gemm_config_set.configs) {
        VLOG(5) << "Compiling " << gemm_config.ToString();
        TF_ASSIGN_OR_RETURN(
            bool has_executable,
            compile(
                fusion, gemm_config,
                ShouldAllowFilteringKernelsSpillingRegisters(gemm_config_set)));
        log(has_executable);
      }

      if (!gemm_config_set.has_sparsity) {
        TF_ASSIGN_OR_RETURN(bool has_executable,
                            compile_reference_executable(fusion));
        log(has_executable);
      }

      if (IsFusionKind(*fusion, kCuDnnFusionKind) ||
          (IsFusionKind(*fusion, kTritonGemmFusionKind) &&
           !gemm_config_set.has_sparsity &&
           IsCuDnnEnabled(config, debug_opts) &&
           HasAlgorithmSupportedByCudnn(*fusion))) {
        const int plan_count = GetCuDnnPlanCount(*fusion, config);
        for (int plan_id = 0; plan_id < plan_count; ++plan_id) {
          log(compile_cudnn_executable(fusion, plan_id));
        }
      }
    }
  }

  VLOG(1) << "Done compiling (successful: " << good_count.load() << ").";

  return executable_sets;
}

absl::StatusOr<AutotuneResult> Execute(const AutotuneConfig& config,
                                       AutotunerCompileUtil& util,
                                       const DebugOptions& debug_opts,
                                       const HloFusionInstruction* fusion,
                                       const ExecutableSet& executable_set) {
  const HloComputation* fusion_computation =
      fusion->called_computations().at(0);

  se::StreamExecutor* stream_exec = config.GetExecutor();
  if (!stream_exec->SynchronizeAllActivity()) {
    return Internal("Failed to synchronize GPU for autotuning.");
  }
  se::DeviceMemoryAllocator* allocator = config.GetAllocator();
  if (allocator == nullptr) {
    allocator = stream_exec->GetAllocator();
  }
  TF_ASSIGN_OR_RETURN(se::Stream* const stream,
                      allocator->GetStream(stream_exec->device_ordinal()));
  TF_ASSIGN_OR_RETURN(
      se::RedzoneAllocator rz_allocator,
      AutotunerUtil::CreateRedzoneAllocator(config, debug_opts));

  const HloInstruction& root = *fusion_computation->root_instruction();
  BufferComparator comparator(root.shape(),
                              fusion_computation->parent()->config());

  std::vector<se::DeviceMemoryBase> inputs;
  inputs.reserve(fusion_computation->parameter_instructions().size());
  std::vector<Shape> input_shapes;
  input_shapes.reserve(fusion_computation->parameter_instructions().size());
  int64_t rng_state = 0;
  for (const HloInstruction* param :
       fusion_computation->parameter_instructions()) {
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase param_buffer,
                        AutotunerUtil::CreateBuffer(
                            rz_allocator, param->shape(), config, rng_state));
    inputs.push_back(param_buffer);
    input_shapes.push_back(param->shape());
  }

  // Run with cuBLAS (optional).
  std::optional<ScopedShapedBuffer> reference_buffer;
  absl::Duration cublas_duration = absl::InfiniteDuration();
  if (executable_set.reference != nullptr) {
    TF_ASSIGN_OR_RETURN(std::optional<ProfilingOutput> output,
                        util.ProfileExecutable(&*executable_set.reference,
                                               stream, inputs, input_shapes));
    TF_RET_CHECK(output.has_value());
    if (config.should_check_correctness()) {
      reference_buffer = std::move(output->output);
    }
    cublas_duration = output->duration;
  }

  const int log_every_n = GetLogEveryN();
  const int64_t executable_count = executable_set.candidates.size();
  int ran_so_far = 0;
  std::vector<AutotuneResult> triton_results, cudnn_results;
  VLOG(2) << "Running " << executable_count << " configs for " << fusion->name()
          << ".";
  for (const ExecutableCandidate& candidate : executable_set.candidates) {
    AutotuneResult res;

    std::string candidate_description;
    if (std::holds_alternative<TritonGemmConfig>(candidate.config)) {
      candidate_description = absl::StrFormat(
          "triton tiling %s",
          std::get<TritonGemmConfig>(candidate.config).ToString());
      *res.mutable_triton() =
          std::get<TritonGemmConfig>(candidate.config).ToProto();
    } else {
      const int64_t plan_id = std::get<CuDnnPlanId>(candidate.config);
      candidate_description = absl::StrFormat("cuDNN plan %d", plan_id);
      res.mutable_algorithm()->set_algo_id(plan_id);
    }
    VLOG(5) << "Trying : " << candidate_description;

    TF_ASSIGN_OR_RETURN(std::optional<ProfilingOutput> profiling_output,
                        util.ProfileExecutable(candidate.executable.get(),
                                               stream, inputs, input_shapes));
    ran_so_far += 1;
    if (ran_so_far % log_every_n == 0) {
      VLOG(2) << "Ran " << ran_so_far << " configs of " << executable_count
              << ".";
    }

    if (!profiling_output) {
      VLOG(5) << "Skipping this tiling.";
      continue;
    }

    VLOG(5) << "Running the kernel took: " << profiling_output->duration;
    if (profiling_output->duration >= absl::Seconds(1)) {
      LOG(WARNING) << "Slow kernel for " << fusion->name()
                   << " took: " << profiling_output->duration << ". "
                   << candidate_description;
    }
    *res.mutable_run_time() =
        tsl::proto_utils::ToDurationProto(profiling_output->duration);

    // Reference buffer is available when `config.should_check_correctness()`
    // is set and reference executable was compiled.
    if (reference_buffer.has_value()) {
      TF_ASSIGN_OR_RETURN(
          se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
          rz_allocator.CheckRedzones());
      if (!rz_check_status.ok()) {
        LOG(ERROR) << "Red zone modified";
        res.mutable_failure()->set_kind(AutotuneResult::REDZONE_MODIFIED);
        res.mutable_failure()->set_msg(rz_check_status.RedzoneFailureMsg());
        CHECK(!config.should_crash_on_check_failure());
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
        CHECK(!config.should_crash_on_check_failure());
        // WRONG_RESULT is not taken seriously by PickBestResult(), so
        // use DISQUALIFIED.
        res.mutable_failure()->set_kind(AutotuneResult::DISQUALIFIED);
        res.mutable_failure()->set_msg(kMessage);
      }
    }
    if (std::holds_alternative<TritonGemmConfig>(candidate.config)) {
      triton_results.push_back(res);
    } else {
      cudnn_results.push_back(res);
    }
  }

  VLOG(2) << "Done running.";

  VLOG(2) << fusion->name() << ": time with cuBLAS: " << cublas_duration;
  AutotuneResult best;
  best.mutable_failure()->set_kind(AutotuneResult::DISQUALIFIED);
  if (!triton_results.empty()) {
    TF_ASSIGN_OR_RETURN(const AutotuneResult triton_best,
                        PickBestResult(triton_results, root.ToString(),
                                       root.GetModule()->config()));
    VLOG(2) << "Best time with Triton: "
            << tsl::proto_utils::FromDurationProto(triton_best.run_time());
    best = triton_best;
  }
  if (!cudnn_results.empty()) {
    TF_ASSIGN_OR_RETURN(const AutotuneResult cudnn_best,
                        PickBestResult(cudnn_results, root.ToString(),
                                       root.GetModule()->config()));
    VLOG(2) << "Best time with cuDNN: "
            << tsl::proto_utils::FromDurationProto(cudnn_best.run_time());
    TF_ASSIGN_OR_RETURN(best,
                        PickBestResult({best, cudnn_best}, root.ToString(),
                                       root.GetModule()->config()));
  }

  if (debug_opts.xla_gpu_cublas_fallback() &&
      !debug_opts.xla_gpu_deterministic_ops() &&
      HasAlgorithmSupportedByCublasOrCublasLt(*fusion)) {
    if (cublas_duration <
        tsl::proto_utils::FromDurationProto(best.run_time())) {
      VLOG(2) << "Falling back to cuBLAS for " << fusion->name();

      AutotuneResult cublas;
      *cublas.mutable_run_time() =
          tsl::proto_utils::ToDurationProto(cublas_duration);
      // We will ignore this value anyway.
      cublas.mutable_gemm()->set_algorithm(CUBLAS_GEMM_DEFAULT);

      return cublas;
    }
  }
  if (!best.has_triton()) {
    VLOG(2) << "Using cuDNN plan " << best.algorithm().algo_id() << " for "
            << fusion->name();
  }
  return best;
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
        } else {
          return TritonGemmAutotuneExtractor(
              triton_gemm_config, device_desc, fusion, debug_opts,
              /*allow_filtering_kernels_spilling_registers=*/true);
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

absl::Status Autotune(
    const AutotuneConfig& config, AutotunerCompileUtil& util,
    tsl::thread::ThreadPool* thread_pool, const DebugOptions& debug_opts,
    const absl::flat_hash_map<const HloFusionInstruction*, GemmConfigSet>&
        gemm_config_sets) {
  absl::flat_hash_map<const HloFusionInstruction*, ExecutableSet>
      executable_sets;
  TF_ASSIGN_OR_RETURN(
      executable_sets,
      CompileMany(config, util, thread_pool, debug_opts, gemm_config_sets));

  // Sort the candidates to make their execution order well-defined for each
  // fusion.
  for (auto& key_value : executable_sets) {
    ExecutableSet& executable_set = key_value.second;
    std::vector<ExecutableCandidate>& candidates = executable_set.candidates;
    absl::c_sort(candidates, [](const ExecutableCandidate& a,
                                const ExecutableCandidate& b) {
      return a.config < b.config;
    });
  }

  int fusion_id = 0;
  for (const auto& key_value : executable_sets) {
    const HloFusionInstruction* fusion = key_value.first;
    const ExecutableSet& executable_set = key_value.second;

    TF_ASSIGN_OR_RETURN(AutotuneResult result, Execute(config, util, debug_opts,
                                                       fusion, executable_set));

    if (debug_opts.xla_gpu_dump_autotuned_triton_fusions()) {
      TF_RETURN_IF_ERROR(
          DumpAutotunedFusion(config, util, result, fusion, fusion_id++));
    }

    const AutotuneCacheKey key = AutotunerUtil::GetKey(fusion, config);
    if (!AutotunerUtil::AddResult(key, std::move(result))) {
      // In the context of model server, concurrent autotuning is expected and
      // insertion of identical autotuning keys is accepted.
      LOG(WARNING) << "AutotunerUtil::AddResult already existed: "
                   << key.ToString();
    }
  }

  return absl::OkStatus();
}

}  // anonymous namespace

absl::StatusOr<std::vector<TritonGemmConfig>> GetPossibleMatmulAutotuneConfigs(
    const HloDotInstruction& dot,
    const se::CudaComputeCapability compute_capability,
    const DebugOptions& debug_options, bool exhaustive_tiling_search) {
  // Avoid autotuning tiny fusions.
  constexpr int kMinGemmElements = 32 * 32;
  if (ShapeUtil::ElementsIn(dot.operand(0)->shape()) <= kMinGemmElements &&
      ShapeUtil::ElementsIn(dot.operand(1)->shape()) <= kMinGemmElements) {
    return ReduceTileSizes(dot, {kDefaultGemmTiling});
  }
  // Split-K optimization enables more even utilization of a GPU in cases
  // where tiling just the non-contracting dimensions of a GEMM does not create
  // a sufficient number of thread block programs to occupy all available cores.
  // Given the typical ~100 cores per GPU 500 tiles make around 5 full
  // waves that completely avoid the need for split-K. The formula below is
  // n_tiles = split_k * (M * N) / (block_m * block_n)
  // with pessimistically assumed maximum block_m and block_n.
  // Most likely there is no need for split-K already at much smaller output
  // tensor sizes.
  constexpr int kSufficientNumberOfTiles = 500;
  const int max_split_k =
      debug_options.xla_gpu_enable_split_k_autotuning()
          ? std::max<int64_t>(1L, kSufficientNumberOfTiles * kMaxTileSize *
                                      kMaxTileSize /
                                      ShapeUtil::ElementsIn(dot.shape()))
          : 1;
  return exhaustive_tiling_search
             ? GetExhaustiveMatmulAutotuneConfigs(dot, compute_capability,
                                                  max_split_k, debug_options)
             : ReduceTileSizes(dot, GetFixedMatmulAutotuneConfigs(
                                        compute_capability, max_split_k));
}

absl::StatusOr<bool> GemmFusionAutotuner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER("GEMM fusion autotuner");
  const DebugOptions& debug_options = module->config().debug_options();
  TF_ASSIGN_OR_RETURN(std::optional<AutotunerCompileUtil> opt_compile_util,
                      AutotunerCompileUtil::Create(config_, debug_options));

  GemmConfigSetCollector gemm_config_set_collector(config_);
  absl::flat_hash_map<const HloFusionInstruction*, GemmConfigSet>
      gemm_config_sets;
  TF_ASSIGN_OR_RETURN(gemm_config_sets,
                      gemm_config_set_collector.CollectGemmConfigSets(
                          module, execution_threads));

  if (debug_options.xla_gpu_autotune_level() == 0 ||
      debug_options.xla_gpu_deterministic_ops()) {
    // Pick the first option for each gemm instead of autotuning.
    for (const auto& [fusion, tilings] : gemm_config_sets) {
      const AutotuneCacheKey key = AutotunerUtil::GetKey(fusion, config_);
      AutotuneResult res;
      if (IsFusionKind(*fusion, kCuDnnFusionKind)) {
        res.mutable_algorithm()->set_algo_id(-1);
      } else {
        const HloDotInstruction* dot_instr =
            Cast<HloDotInstruction>(hlo_query::GetFirstInstructionWithOpcode(
                *fusion->called_computations().at(0), HloOpcode::kDot));
        TF_ASSIGN_OR_RETURN(auto configs,
                            ReduceTileSizes(*dot_instr, {kDefaultGemmTiling}));
        auto config = configs.front();
        *res.mutable_triton() = config.ToProto();
      }
      *res.mutable_run_time() =
          tsl::proto_utils::ToDurationProto(absl::ZeroDuration());
      AutotunerUtil::AddResult(key, res);
    }
  } else if (!config_.IsDeviceless()) {
    TF_RET_CHECK(opt_compile_util.has_value());
    if (!gemm_config_sets.empty()) {
      std::string correctness_check_str = config_.should_check_correctness()
                                              ? "(with correctness check)"
                                              : "(without correctness check)";

      VLOG(1) << "Autotuning " << gemm_config_sets.size() << " fusions "
              << correctness_check_str << ".";
      TF_RETURN_IF_ERROR(Autotune(config_, *opt_compile_util, thread_pool_,
                                  debug_options, gemm_config_sets));
      VLOG(1) << "Done autotuning.";
    }
  }

  return GemmFusionAutotunerVisitor(config_).RunOnModule(module,
                                                         execution_threads);
}

}  // namespace gpu
}  // namespace xla
