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

#include "xla/service/gpu/autotuning/gemm_fusion_autotuner.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
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
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/runtime/buffer_comparator.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/primitive_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/call_inliner.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/autotuner_compile_util.h"
#include "xla/service/gpu/autotuning/autotuner_status_key.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/autotuning/dot_search_space.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion_pattern.h"
#include "xla/service/gpu/matmul_indexing_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/split_k_gemm_rewriter.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/transforms/custom_kernel_fusion_rewriter.h"
#include "xla/service/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/service/gpu/transforms/fusion_wrapper.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/service/gpu/transforms/nest_gemm_fusion.h"
#include "xla/service/gpu/transforms/priority_fusion.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/ptx_compiler_helpers.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/lib/core/bits.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/proto/proto_utils.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/scoped_annotation.h"

// Log levels used in this file:
// VLOG(1): Overview
// VLOG(2): Autotuning progress
// VLOG(3): Autotuning progress - more frequent
// VLOG(4): Print all fusions
// VLOG(5): Profiling information for every tiling
// VLOG(10): Print fusion computations and each configuration

namespace xla {
namespace gpu {

using BackendConfig = GemmFusionAutotunerImpl::BackendConfig;
using BackendConfigs = GemmFusionAutotunerImpl::BackendConfigs;
using ProfilingOutput = AutotunerCompileUtil::ProfilingOutput;

namespace {

// Minimum tile size.
constexpr int kMinTileSize = 16;

// Split-K is enabled when the estimate number of waves is lower than the limit.
constexpr int kMaxWavesForSplitK = 5;

// Search space for exhaustive matmul autotuning.
constexpr std::array<int, 6> kBlockSizes = {16, 32, 64, 128, 256, 512};
constexpr std::array<int, 4> kNumStages = {1, 2, 3, 4};
constexpr std::array<int, 4> kNumWarps = {2, 4, 8, 16};
constexpr std::array<int, 5> kSplitK = {1, 2, 4, 8, 16};
constexpr std::array<int, 5> kNumCtas = {1, 2, 4, 8, 16};

using AutoTuneCacheKeyCount = absl::flat_hash_map<AutotuneCacheKey, uint64_t>;

using KeysAndInstructions =
    std::vector<std::pair<AutotuneCacheKey, const HloFusionInstruction*>>;

struct GemmFusionCollectorResult {
  KeysAndInstructions keys_and_instructions;
  // Counts unique fusions in the module.
  AutoTuneCacheKeyCount fusion_count_map;
  // Fingerprints the module by all its unique GEMM fusions.
  std::string fingerprint;
};

class GemmFusionCollector : public ConstDfsHloVisitorWithDefault {
 public:
  explicit GemmFusionCollector(GemmFusionAutotunerImpl* impl) : impl_(impl) {}

  // Find fusions to tune.
  absl::StatusOr<GemmFusionCollectorResult> CollectGemmFusions(
      const HloModule& module,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {}) {
    error_out_on_cache_miss_ =
        module.config()
            .debug_options()
            .xla_gpu_require_complete_aot_autotune_results();
    result_ = {};
    handled_fusions_.clear();
    for (HloComputation* computation :
         module.MakeNonfusionComputations(execution_threads)) {
      TF_RETURN_IF_ERROR(computation->Accept(this));
    }
    TF_ASSIGN_OR_RETURN(result_.fingerprint,
                        GetBase64EncodedSha256Hash(result_.fingerprint));
    return std::move(result_);
  }

  absl::Status HandleFusion(const HloInstruction* hlo) override {
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        hlo->backend_config<GpuBackendConfig>());
    const FusionBackendConfig& backend_config =
        gpu_config.fusion_backend_config();
    if (backend_config.kind() != kTritonGemmFusionKind &&
        backend_config.kind() != kCuDnnFusionKind &&
        backend_config.kind() != kCustomFusionKind) {
      return absl::OkStatus();
    }

    AutotuneCacheKey key = AutotunerUtil::GetKey(hlo, impl_->GetConfig());
    auto [iterator, inserted] = result_.fusion_count_map.insert({key, 1});
    if (inserted) {
      result_.fingerprint += key.GetHlo();
    } else {
      ++(iterator->second);
    }

    bool missing_config = !backend_config.has_triton_gemm_config() &&
                          !backend_config.has_cudnn_fusion_config() &&
                          !backend_config.has_custom_fusion_config();
    if (missing_config && handled_fusions_.insert(key).second) {
      result_.keys_and_instructions.push_back(
          {key, Cast<HloFusionInstruction>(hlo)});
    }

    return absl::OkStatus();
  }

  absl::StatusOr<KeysAndInstructions> RemoveCached(
      const KeysAndInstructions& entries) const {
    KeysAndInstructions result;
    for (const auto& [key, fusion] : entries) {
      TF_ASSIGN_OR_RETURN(bool is_in_cache,
                          AutotunerUtil::IsInCache(key, impl_->GetConfig()));
      if (is_in_cache) {
        continue;
      }
      if (error_out_on_cache_miss_) {
        return absl::NotFoundError(absl::StrCat(
            "Complete autotuning results are required, but no cache result "
            "found for key: ",
            key.ToString()));
      }
      result.push_back({key, fusion});
    }
    return result;
  }

  // Trim the set of entries to what one rank has to run.
  static KeysAndInstructions Shard(const KeysAndInstructions& entries,
                                   const int shard_index,
                                   const int shard_count) {
    const uint64_t bucket_size =
        (entries.size() + shard_count - 1) / shard_count;
    const uint64_t start = bucket_size * shard_index;
    const uint64_t end = std::min(start + bucket_size, entries.size());
    if (start >= end) {
      return {};
    }
    return KeysAndInstructions(entries.cbegin() + start,
                               entries.cbegin() + end);
  }

  absl::StatusOr<BackendConfigs> GenerateConfigs(
      const KeysAndInstructions& keys_and_instructions) const {
    BackendConfigs result;
    result.reserve(keys_and_instructions.size());
    for (const auto& [_, fusion] : keys_and_instructions) {
      TF_ASSIGN_OR_RETURN(std::vector<BackendConfig> configs,
                          impl_->GenerateConfigs(*fusion));
      result.push_back({fusion, std::move(configs)});
    }
    return result;
  }

  absl::Status DefaultAction(const HloInstruction* hlo) override {
    return absl::OkStatus();
  }

 private:
  bool error_out_on_cache_miss_;
  GemmFusionAutotunerImpl* impl_;
  GemmFusionCollectorResult result_;
  AutotuneCacheKeySet handled_fusions_;
};

struct TileSizeLimit {
  int block_m = 0;
  int block_n = 0;
  int block_k = 0;
};

absl::StatusOr<TileSizeLimit> GetLimits(const HloDotInstruction& dot) {
  TF_ASSIGN_OR_RETURN(int64_t non_contracting_index_lhs,
                      NonContractingDimensionIndex(dot, /*operand_number=*/0));
  TF_ASSIGN_OR_RETURN(int64_t non_contracting_index_rhs,
                      NonContractingDimensionIndex(dot, /*operand_number=*/1));
  TF_ASSIGN_OR_RETURN(int64_t contracting_index,
                      ContractingDimensionIndex(dot, /*operand_number=*/1));
  // This is not a sharp upper limit, the actual m value can be much smaller
  // based on how much of the m dimension is physically contiguous.
  const int max_m = tsl::NextPowerOfTwoS64(
      dot.operand(0)->shape().dimensions(non_contracting_index_lhs));
  // Theoretically the same is true as for m, but that is not possible in
  // practice with the current implementation.
  const int max_n = tsl::NextPowerOfTwoS64(
      dot.operand(1)->shape().dimensions(non_contracting_index_rhs));
  // This is before doing the split-k transform.
  const int max_k = tsl::NextPowerOfTwoS64(
      dot.operand(1)->shape().dimensions(contracting_index));

  return TileSizeLimit{
      /*block_m=*/std::max(max_m, kMinTileSize),
      /*block_n=*/std::max(max_n, kMinTileSize),
      /*block_k=*/std::max(max_k, kMinTileSize),
  };
}

int GetLogEveryN() { return VLOG_IS_ON(3) ? 100 : 1000; }

HloCostAnalysis::Options PriorityFusionOptions() {
  // The real pointer size is set in GpuCompiler. In HloCostAnalysis, the
  // pointer size is used only to determine the size of tuple types. We
  // shouldn't have any tuples in the autotuned module, so it's safe to use
  // the default value here, instead of piping the real value.
  return {.count_multiple_input_accesses = true};
}

absl::StatusOr<std::unique_ptr<HloModule>> TritonGemmAutotuneExtractor(
    const TritonGemmConfig& config,
    const se::DeviceDescription& gpu_device_info,
    const HloFusionInstruction* fusion, DebugOptions debug_opts,
    bool allow_filtering_kernels_spilling_registers) {
  std::unique_ptr<HloModule> new_module =
      ExtractInstructionIntoNewModule(*fusion);
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
    for (PrimitiveType type :
         {BF16, F8E5M2, F8E4M3FN, F8E4M3B11FNUZ, F8E5M2FNUZ, F8E4M3FNUZ}) {
      GpuFloatSupport float_support(gpu_device_info.cuda_compute_capability(),
                                    type);
      FloatNormalization float_normalization(&float_support);
      TF_RETURN_IF_ERROR(float_normalization.Run(new_module.get()).status());
    }

    PriorityFusion priority_fusion(
        /*thread_pool=*/nullptr, gpu_device_info, PriorityFusionOptions());
    TF_RETURN_IF_ERROR(priority_fusion.Run(new_module.get()).status());

    // If the priority fusion pass above skipped some instructions, turn them
    // into fusions.
    FusionWrapper fusion_wrapper(gpu_device_info);
    TF_RETURN_IF_ERROR(fusion_wrapper.Run(new_module.get()).status());
  }

  if (debug_opts
          .xla_gpu_unsupported_enable_generic_triton_emitter_for_gemms()) {
    NestGemmFusion nest_gemm_fusion(gpu_device_info.gpu_compute_capability());
    TF_RETURN_IF_ERROR(nest_gemm_fusion.Run(new_module.get()).status());
  }

  return new_module;
}

absl::StatusOr<std::unique_ptr<HloModule>> CublasGemmAutotuneExtractor(
    const AutotuneConfig& config, const se::DeviceDescription& gpu_device_info,
    const se::SemanticVersion& toolkit_version,
    const HloFusionInstruction* fusion, const DebugOptions& debug_opts) {
  const HloComputation* fusion_computation = fusion->called_computation();
  std::unique_ptr<HloModule> new_module =
      ExtractComputationIntoNewModule(*fusion_computation);
  new_module->mutable_config().set_debug_options(debug_opts);

  auto* dot = hlo_query::GetFirstInstructionWithOpcode(
      *new_module->entry_computation(), HloOpcode::kDot);
  // Substitute algorithms, which are not supported by cuBLAS for the check, but
  // don't use cuBlas in the end. This assumes that the substituting algorithm
  // has result which are close enough for the check in this file.
  if (dot->precision_config().algorithm() ==
      PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3) {
    dot->mutable_precision_config()->set_algorithm(
        PrecisionConfig::ALG_DOT_F32_F32_F32);
  }

  for (GemmRewriterOptions::DType dtype :
       {GemmRewriterOptions::DType::kFp8Only,
        GemmRewriterOptions::DType::kNonFp8Only}) {
    GemmRewriter gemm_rewriter(config.GetGpuComputeCapability(),
                               toolkit_version, GemmRewriterOptions{dtype});
    DotAlgorithmRewriter dot_algorithm_rewriter;
    PriorityFusion fusion_pass(
        /*thread_pool=*/nullptr, gpu_device_info, PriorityFusionOptions());
    TF_RETURN_IF_ERROR(dot_algorithm_rewriter.Run(new_module.get()).status());
    TF_RETURN_IF_ERROR(gemm_rewriter.Run(new_module.get()).status());
    TF_RETURN_IF_ERROR(fusion_pass.Run(new_module.get()).status());
  }
  return new_module;
}

absl::Status UpdateFusionInstructionKernelIndex(
    HloInstruction* fusion_instruction, int kernel_index) {
  GpuBackendConfig gpu_config =
      fusion_instruction->backend_config<GpuBackendConfig>().value();
  gpu_config.mutable_fusion_backend_config()
      ->mutable_custom_fusion_config()
      ->set_kernel_index(kernel_index);
  TF_RETURN_IF_ERROR(fusion_instruction->set_backend_config(gpu_config));

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<HloModule>> CustomFusionKernelAutotuneExtractor(
    const GemmFusionAutotunerImpl::CustomKernelFusionConfig& cutlass_config,
    const AutotuneConfig& config, const se::SemanticVersion& toolkit_version,
    const HloFusionInstruction* fusion, const DebugOptions& debug_opts) {
  const HloComputation* fusion_computation = fusion->called_computation();
  std::unique_ptr<HloModule> new_module =
      ExtractComputationIntoNewModule(*fusion_computation);
  new_module->mutable_config().set_debug_options(debug_opts);

  CustomKernelFusionRewriter rewriter(&config.GetDeviceDescription());
  PriorityFusion fusion_pass(
      /*thread_pool=*/nullptr, config.GetDeviceDescription(),
      PriorityFusionOptions());
  TF_RETURN_IF_ERROR(rewriter.Run(new_module.get()).status());
  TF_RETURN_IF_ERROR(fusion_pass.Run(new_module.get()).status());

  // Select custom kernel fusion kernel.
  HloInstruction* custom_kernel_fusion =
      hlo_query::GetFirstInstructionWithOpcode(*new_module->entry_computation(),
                                               HloOpcode::kFusion);
  int64_t kernel_index = cutlass_config.kernel_index;
  TF_RETURN_IF_ERROR(
      UpdateFusionInstructionKernelIndex(custom_kernel_fusion, kernel_index));

  return new_module;
}

absl::StatusOr<std::unique_ptr<HloModule>> FusionExtractor(
    const HloFusionInstruction& fusion, const DebugOptions& debug_opts) {
  std::unique_ptr<HloModule> module = ExtractInstructionIntoNewModule(fusion);
  module->mutable_config().set_debug_options(debug_opts);
  return module;
}

absl::StatusOr<std::unique_ptr<HloModule>> CuDnnFusionExtractor(
    const HloFusionInstruction& fusion, const DebugOptions& debug_opts,
    const int plan_id) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      FusionExtractor(fusion, debug_opts));

  GpuBackendConfig gpu_config;
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind(std::string(kCuDnnFusionKind));
  // Provided a plan ID the autotuner just compiles one plan.
  backend_config.mutable_cudnn_fusion_config()->set_plan_id(plan_id);
  TF_RETURN_IF_ERROR(
      module->entry_computation()->root_instruction()->set_backend_config(
          gpu_config));
  return module;
}

AutotuneResult FromConfig(const BackendConfig& config) {
  AutotuneResult res;
  if (std::holds_alternative<GemmFusionAutotunerImpl::CuBlasConfig>(config)) {
    res.mutable_gemm()->set_algorithm(
        GemmFusionAutotunerImpl::BLAS_GEMM_DEFAULT);
  } else if (std::holds_alternative<
                 GemmFusionAutotunerImpl::CustomKernelFusionConfig>(config)) {
    res.mutable_custom_kernel_fusion()->set_kernel_index(
        std::get<GemmFusionAutotunerImpl::CustomKernelFusionConfig>(config)
            .kernel_index);
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

absl::Status DumpOriginalFusion(AutotunerCompileUtil& util,
                                const HloFusionInstruction& fusion,
                                int fusion_id) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      util.ExtractModule([&](const DebugOptions& debug_opts) {
                        return FusionExtractor(fusion, debug_opts);
                      }));
  module->set_name(std::string(fusion.name()));
  // Using the original module for its debug info and name in the first
  // parameter. It's better to include the name of both the original module
  // and the extracted module, to avoid name clashes.
  std::string rendered_graph_name =
      absl::StrCat("gemm_fusion_", fusion_id, ".", module->name(), ".dot");
  std::string rendered_graph = RenderGraph(rendered_graph_name, *module,
                                           RenderedGraphFormat::kDot, true);
  DumpToFileInDir(
      /*module=*/*fusion.GetModule(),
      /*file_prefix=*/"",
      /*file_suffix=*/rendered_graph_name,
      /*contents=*/rendered_graph);
  DumpToFileInDirOrStdout(
      /*module=*/*fusion.GetModule(),
      /*file_prefix=*/"",
      /*file_suffix=*/
      absl::StrCat("gemm_fusion_", fusion_id, ".", module->name(), ".txt"),
      /*contents=*/module->ToString());
  return absl::OkStatus();
}

absl::Status DumpAutotunedFusion(const AutotuneConfig& autotune_config,
                                 const se::SemanticVersion& toolkit_version,
                                 AutotunerCompileUtil& util,
                                 const AutotuneResult result,
                                 const HloFusionInstruction* fusion,
                                 int fusion_id) {
  TritonGemmConfig triton_gemm_config;
  if (result.has_triton()) {
    TF_ASSIGN_OR_RETURN(triton_gemm_config,
                        TritonGemmConfig::FromProto(result.triton()));
  }
  const se::DeviceDescription& device_desc =
      autotune_config.GetDeviceDescription();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      util.ExtractModule([&](const DebugOptions& debug_opts) {
        if (result.has_algorithm()) {
          return CuDnnFusionExtractor(*fusion, debug_opts,
                                      result.algorithm().algo_id());
        }
        if (result.has_triton()) {
          return TritonGemmAutotuneExtractor(
              triton_gemm_config, device_desc, fusion, debug_opts,
              /*allow_filtering_kernels_spilling_registers=*/true);
        }
        if (result.has_gemm()) {
          return CublasGemmAutotuneExtractor(autotune_config, device_desc,
                                             toolkit_version, fusion,
                                             debug_opts);
        }
        LOG(FATAL) << "Unknown result type: " << result.DebugString();
      }));
  module->set_name(std::string(fusion->name()));
  // Using the original module for its debug info and name in the first
  // parameter. It's better to include the name of both the original module
  // and the extracted module, to avoid name clashes.
  DumpToFileInDirOrStdout(
      /*module=*/*fusion->GetModule(),
      /*file_prefix=*/"",
      /*file_suffix=*/
      absl::StrCat("gemm_fusion_", fusion_id, ".", module->name(),
                   ".optimized.txt"),
      /*contents=*/module->ToString());
  return absl::OkStatus();
}

std::string ConfigToString(const BackendConfig& config) {
  if (std::holds_alternative<TritonGemmConfig>(config)) {
    return std::get<TritonGemmConfig>(config).ToString();
  }
  if (std::holds_alternative<GemmFusionAutotunerImpl::CuDnnConfig>(config)) {
    return absl::StrFormat(
        "cuDNN plan %d",
        std::get<GemmFusionAutotunerImpl::CuDnnConfig>(config).plan_id);
  }
  if (std::holds_alternative<GemmFusionAutotunerImpl::CuBlasConfig>(config)) {
    return "reference (cublas)";
  }
  LOG(FATAL) << "Unsupported config type: " << config.index();
}

std::string Serialize(const BackendConfig& config) {
  if (auto* triton_config = std::get_if<TritonGemmConfig>(&config)) {
    tsl::protobuf::TextFormat::Printer printer;
    printer.SetSingleLineMode(true);
    std::string result;
    printer.PrintToString(triton_config->ToProto(), &result);
    return result;
  }
  return ConfigToString(config);
}

absl::Status RewriteGemmFusionToCall(HloInstruction* fusion_instr) {
  // Falling back to cuBLAS: Converting the fusion to a Call, so that it
  // can be inlined back again.
  HloComputation* const computation = fusion_instr->parent();
  HloInstruction* const call =
      computation->AddInstruction(HloInstruction::CreateCall(
          fusion_instr->shape(), fusion_instr->operands(),
          fusion_instr->fused_instructions_computation()));
  return computation->ReplaceInstruction(fusion_instr, call);
}

absl::Status RewriteGemmFusionToCustomKernelFusion(
    HloInstruction* fusion_instr, se::DeviceDescription device_description,
    int64_t kernel_index) {
  // Rewrites gemm fusion to custom kernel fusion.
  // First convert the fusion to a call. Then inlines the call. Then
  // rewrites to custom kernel fusion.
  HloComputation* const computation = fusion_instr->parent();
  HloInstruction* const call =
      computation->AddInstruction(HloInstruction::CreateCall(
          fusion_instr->shape(), fusion_instr->operands(),
          fusion_instr->fused_instructions_computation()));
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(fusion_instr, call));
  HloPassPipeline pipeline("autotuner_custom_kernel_fusion_rewriter");
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<CustomKernelFusionRewriter>(&device_description,
                                               kernel_index);
  HloModule* hlo_module = call->GetModule();
  return pipeline.Run(hlo_module).status();
}

absl::Status HandleTritonGemm(HloInstruction* fusion_instr,
                              FusionBackendConfig& fusion_backend_config) {
  TF_ASSIGN_OR_RETURN(
      const TritonGemmConfig config,
      TritonGemmConfig::FromProto(fusion_backend_config.triton_gemm_config()));
  if (config.split_k > 1) {
    TF_RETURN_IF_ERROR(MakeDotSplitKBatch(fusion_instr, config));
  }
  return absl::OkStatus();
}

}  // anonymous namespace

absl::Status GemmFusionAutotunerRewriterVisitor::HandleFusion(
    HloInstruction* fusion_instr) {
  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      fusion_instr->backend_config<GpuBackendConfig>());
  FusionBackendConfig& fusion_backend_config =
      *gpu_config.mutable_fusion_backend_config();

  // Only autotune Triton, cuDNN, and custom kernel fusions.
  if (fusion_backend_config.kind() != kTritonGemmFusionKind &&
      fusion_backend_config.kind() != kCuDnnFusionKind &&
      fusion_backend_config.kind() != kCustomFusionKind) {
    return absl::OkStatus();
  }

  // Do not autotune if the backend config has already assigned tiling config.
  if (fusion_backend_config.has_triton_gemm_config()) {
    TF_RETURN_IF_ERROR(HandleTritonGemm(fusion_instr, fusion_backend_config));
    MarkAsChanged();
    return absl::OkStatus();
  }

  // Do not autotune if the backend config has valid config.
  if (fusion_backend_config.has_cudnn_fusion_config() ||
      fusion_backend_config.has_custom_fusion_config()) {
    return absl::OkStatus();
  }

  VLOG(4) << "Autotuning fusion instruction: " << fusion_instr->ToString();
  TF_ASSIGN_OR_RETURN(
      AutotuneResult autotune_result,
      AutotunerUtil::Autotune(
          fusion_instr, config_, [&]() -> absl::StatusOr<AutotuneResult> {
            absl::Status s;
            if (config_.IsDeviceless()) {
              s = absl::InternalError(absl::StrCat(
                  "Expect autotune result cache hit for deviceless "
                  "compilation (HLO: ",
                  fusion_instr->ToString(), ")"));
            } else {
              s = absl::InternalError("Expect autotune result cache hit.");
            }
            tsl::errors::InsertPayloads(
                s, {{std::string(kAutotuneCacheRequiredErrorPayloadKey), ""}});

            return s;
          }));
  VLOG(4) << "Autotuning result: " << autotune_result.ShortDebugString();

  if (autotune_result.has_triton()) {
    *fusion_backend_config.mutable_triton_gemm_config() =
        autotune_result.triton();
    TF_RETURN_IF_ERROR(fusion_instr->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(HandleTritonGemm(fusion_instr, fusion_backend_config));
    MarkAsChanged();
    return absl::OkStatus();
  }

  if (autotune_result.has_gemm()) {
    TF_RETURN_IF_ERROR(RewriteGemmFusionToCall(fusion_instr));
    MarkAsChanged();
    return absl::OkStatus();
  }

  if (autotune_result.has_custom_kernel_fusion()) {
    TF_RETURN_IF_ERROR(RewriteGemmFusionToCustomKernelFusion(
        fusion_instr, config_.GetDeviceDescription(),
        autotune_result.custom_kernel_fusion().kernel_index()));
    MarkAsChanged();
    return absl::OkStatus();
  }

  // Autotune result has a cuDNN fusion.
  CHECK(autotune_result.has_algorithm());
  fusion_backend_config.set_kind(std::string(kCuDnnFusionKind));
  fusion_backend_config.mutable_cudnn_fusion_config()->set_plan_id(
      autotune_result.algorithm().algo_id());
  TF_RETURN_IF_ERROR(fusion_instr->set_backend_config(gpu_config));
  MarkAsChanged();
  return absl::OkStatus();
}

bool GemmFusionAutotunerImpl::IsFusionKind(const HloInstruction& hlo,
                                           absl::string_view kind) {
  auto gpu_config = hlo.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return false;
  }
  return gpu_config->fusion_backend_config().kind() == kind;
}

// Methods required for sorting the configs.
bool GemmFusionAutotunerImpl::CuBlasConfig::operator<(
    const CuBlasConfig& other) const {
  return false;
}
bool GemmFusionAutotunerImpl::CuDnnConfig::operator<(
    const CuDnnConfig& other) const {
  return plan_id < other.plan_id;
}
bool GemmFusionAutotunerImpl::CustomKernelFusionConfig::operator<(
    const CustomKernelFusionConfig& other) const {
  return false;
}

bool GemmFusionAutotunerImpl::IsAutotuningEnabled() const {
  return debug_options_.xla_gpu_autotune_level() > 0 &&
         !debug_options_.xla_gpu_deterministic_ops();
}

static std::vector<BackendConfig> GenerateCustomKernelFusionConfigs(
    const HloFusionInstruction& fusion,
    se::DeviceDescription device_description) {
  const CustomKernelFusionPatternRegistry* patterns =
      CustomKernelFusionPatternRegistry::Default();
  HloComputation* computation = fusion.called_computation();
  // Get the first dot instruction in the fusion body.
  HloInstruction* dot_instruction =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  std::vector<CustomKernelFusionPattern::Match> match =
      patterns->Match(device_description, dot_instruction);

  // For Cutlass we expect only one match for a GEMM fusion.
  if (match.size() != 1) {
    return {};
  }

  CustomKernelFusionRegistry* registry = CustomKernelFusionRegistry::Default();
  auto* custom_kernel_fusion = registry->Lookup(match[0].config().name());

  // If custom fusion is not found it means that some of the build targets
  // might not be statically linked into the binary.
  if (custom_kernel_fusion == nullptr) {
    return {};
  }

  // There can be multiple kernels for a single fusion pattern, which are
  // selected by the kernel_index.
  // To get the number of kernels we can rewrite the fusion to custom kernel
  // fusion and count the number of loaded kernels.
  const HloComputation* fusion_computation = fusion.called_computation();
  std::unique_ptr<HloModule> new_module =
      ExtractComputationIntoNewModule(*fusion_computation);
  CustomKernelFusionRewriter rewriter(&device_description);
  absl::StatusOr<bool> changed = rewriter.Run(new_module.get());
  if (!changed.ok() || !*changed) {
    VLOG(2) << "Skip custom kernel config. Failed to rewrite custom kernel "
               "fusion: "
            << changed.status();
    return {};
  }

  HloInstruction* custom_kernel_fusion_instr =
      hlo_query::GetFirstInstructionWithOpcode(*new_module->entry_computation(),
                                               HloOpcode::kFusion);
  if (custom_kernel_fusion_instr == nullptr) {
    VLOG(2) << "Skip custom kernel config. Failed to find custom kernel "
               "fusion instruction in the rewritten module.";
    return {};
  }

  absl::StatusOr<std::vector<CustomKernel>> kernels =
      custom_kernel_fusion->LoadKernels(
          device_description,
          custom_kernel_fusion_instr->fused_instructions_computation());
  if (!kernels.ok()) {
    VLOG(2) << "Skip custom kernel config. Failed to load custom kernels: "
            << kernels.status();
    return {};
  }

  std::vector<BackendConfig> configs;
  configs.reserve(kernels.value().size());
  for (int i = 0; i < kernels.value().size(); ++i) {
    GemmFusionAutotunerImpl::CustomKernelFusionConfig config{
        /*kernel_index=*/i};
    configs.push_back(config);
  }

  return configs;
}

absl::StatusOr<std::vector<BackendConfig>>
GemmFusionAutotunerImpl::GenerateConfigs(const HloFusionInstruction& fusion) {
  const HloDotInstruction* dot =
      Cast<HloDotInstruction>(hlo_query::GetFirstInstructionWithOpcode(
          *fusion.called_computation(), HloOpcode::kDot));
  std::vector<BackendConfig> configs;

  if (!debug_options_.xla_gpu_experimental_disable_binary_libraries()) {
    // Add cuBLAS reference config, if available.
    TF_ASSIGN_OR_RETURN(int64_t rhs_contracting_index,
                        ContractingDimensionIndex(*dot, /*operand_number=*/1));
    if (algorithm_util::IsSupportedByCublasOrCublasLt(
            dot->precision_config().algorithm(), GetComputeCapability(), dot,
            rhs_contracting_index) &&
        !dot->sparse_operands() && IsAutotuningEnabled()) {
      configs.push_back(CuBlasConfig{});
    }

    // Add lib (e.g. cuDNN) plans, if available.
    if (AddLibConfigs(fusion, dot, configs)) return configs;
  }

  // Add CustomKernelFusion (Cutlass) configs, if available.
  // Go through all the instructions in the fusion body try to match them to
  // a custom kernel fusion pattern.
  if ((IsFusionKind(fusion, kCustomFusionKind) ||
       IsFusionKind(fusion, kTritonGemmFusionKind)) &&
      IsAutotuningEnabled() && !config_.IsDeviceless()) {
    std::vector<BackendConfig> custom_kernel_fusion_configs =
        GenerateCustomKernelFusionConfigs(fusion,
                                          config_.GetDeviceDescription());
    configs.insert(configs.end(), custom_kernel_fusion_configs.begin(),
                   custom_kernel_fusion_configs.end());
  }

  // Add triton configs.
  TF_ASSIGN_OR_RETURN(std::vector<TritonGemmConfig> triton_configs,
                      GenerateTritonConfigs(*dot));
  for (TritonGemmConfig& config : triton_configs) {
    configs.push_back(std::move(config));
  }
  return configs;
}

void ModifyPotentiallyFailingConfig(TritonGemmConfig& config, int minBitWidth,
                                    int kLdmatrixGranularity) {
  // TODO(b/337839570): Triton currently has a limitation where it crashes
  // on small block_k values depending on the bit-width of the inputs to the
  // dot. The logic below accounts for this limitation.
  // We don't do this for predicates now because as of
  // https://github.com/triton-lang/triton/commit/d9facf3a6edbc819c80d58b87e689bc0c2632756,
  // this leads to registers spilling (see b/388714585). Bug filed upstream:
  // https://github.com/triton-lang/triton/issues/5572. While it used to work
  // previously, it is quite limiting for predicates as the smallest acceptable
  // block_k value would be 256.
  if (minBitWidth > 1) {
    config.block_k =
        std::max(config.block_k, kLdmatrixGranularity / minBitWidth);
  }

  // Additionally, there are further issues happening on 8 bit types and
  // predicates that require additional restriction on block_m when num_warps
  // > 8 (see b/378660935). It's unclear if the issue extends beyond these
  // cases, so restrictions here are conservative to these.
  if (minBitWidth <= 8 && config.num_warps > 8) {
    config.block_m = std::max(config.block_m, 32);
  }
}

absl::StatusOr<std::vector<TritonGemmConfig>>
GemmFusionAutotunerImpl::GenerateTritonConfigs(const HloDotInstruction& dot) {
  // Default tiling when autotuning is disabled.
  constexpr TritonGemmConfig kDefaultConfig = {
      /*block_m=*/32, /*block_n=*/32,   /*block_k=*/32,
      /*split_k=*/1,  /*num_stages=*/1, /*num_warps=*/4,
      /*num_ctas=*/1};
  constexpr int kMinGemmElements = 2 * 32 * 32;
  bool small_dot = ShapeUtil::ElementsIn(dot.operand(0)->shape()) +
                       ShapeUtil::ElementsIn(dot.operand(1)->shape()) <=
                   kMinGemmElements;
  // TODO: b/393299275 - Remove this once the new emitter lands and we can
  // support slices in contracting dimension with splits.
  bool supports_contracting_split =
      HloBfsFindAll({&dot}, [&](const HloInstruction* node) {
        return node->opcode() == HloOpcode::kSlice;
      }).empty();
  bool autotune_contracting_split =
      supports_contracting_split &&
      debug_options_.xla_gpu_enable_split_k_autotuning();

  if (debug_options_.xla_gpu_experimental_enable_dynamic_dot_search_space()) {
    TritonDotFusionSearchSpace search_space(config_.GetDeviceDescription(),
                                            &dot);
    VLOG(1) << "Generating configs from search space: "
            << search_space.ToString();
    // We don't need to consider small_dot here. The new search space will
    // already generate a unique config for small problems.
    std::vector<TritonGemmConfig> configs = search_space.GenerateConfigs(
        /*force_contracting_split=*/autotune_contracting_split
            ? std::nullopt
            : std::make_optional(1));
    if (!debug_options_.xla_gpu_exhaustive_tiling_search()) {
      VLOG(1) << "Restricting configs to the default set.";
      configs = search_space.OptimizeConfigSet(
          configs, /*hints=*/GetDefaultTritonConfigs());
    }
    if (!IsAutotuningEnabled()) {
      // Keep the first config, which likely does not spill registers.
      configs.resize(1);
    }
    return configs;
  }

  // Retrieve the minimum bit-width participating in the dot. This is needed
  // to avoid autotuning configurations that are not supported by Triton. This
  // is used to restrict the values for tile_k.
  std::vector<const HloInstruction*> converts =
      HloBfsFindAll({&dot}, [&](const HloInstruction* node) {
        return node->opcode() == HloOpcode::kConvert;
      });
  PrimitiveType out = dot.shape().element_type();
  PrimitiveType in0 = dot.operand(0)->shape().element_type();
  PrimitiveType in1 = dot.operand(1)->shape().element_type();
  int minBitWidth =
      std::min({primitive_util::BitWidth(out), primitive_util::BitWidth(in0),
                primitive_util::BitWidth(in1)});
  for (auto convert : converts) {
    auto in_type = convert->operand(0)->shape().element_type();
    auto out_type = convert->shape().element_type();
    minBitWidth = std::min({minBitWidth, primitive_util::BitWidth(in_type),
                            primitive_util::BitWidth(out_type)});
  }

  std::vector<TritonGemmConfig> result_configs;
  TF_ASSIGN_OR_RETURN(TileSizeLimit limits, GetLimits(dot));

  // Generate the list of configurations (once).
  if (triton_configs_.empty()) {
    triton_configs_ = !IsAutotuningEnabled() ? std::vector(1, kDefaultConfig)
                      : debug_options_.xla_gpu_exhaustive_tiling_search()
                          ? GetExhaustiveTritonConfigs()
                          : GetDefaultTritonConfigs();
  }

  std::vector<TritonGemmConfig> triton_configs =
      small_dot ? std::vector(1, kDefaultConfig) : triton_configs_;

  // Split-K optimization enables more even utilization of a GPU in cases
  // where tiling just the non-contracting dimensions of a GEMM does not create
  // a sufficient number of thread block programs to occupy all available cores.
  // Around 5 full waves completely avoid the need for split-K.
  // n_tiles = split_k * (M * N) / (block_m * block_n)
  const int kCoreCount = config_.GetDeviceDescription().core_count();
  CHECK_GE(kCoreCount, 1);
  const int64_t kSufficientNumberOfTiles = kMaxWavesForSplitK * kCoreCount;
  const int64_t result_size = ShapeUtil::ElementsIn(dot.shape());
  const int64_t threads_per_warp =
      config_.GetDeviceDescription().threads_per_warp();

  // Triton configurations are adjusted and deduplicated.
  absl::flat_hash_set<TritonGemmConfig> added;
  for (TritonGemmConfig& config : triton_configs) {
    config.block_m = std::min(config.block_m, limits.block_m);
    config.block_n = std::min(config.block_n, limits.block_n);
    config.block_k = std::min(config.block_k, limits.block_k);
    int max_split_k = 1;
    if (autotune_contracting_split) {
      int64_t ratio = kSufficientNumberOfTiles * config.block_m *
                      config.block_n / result_size;
      max_split_k = 1 << std::max<int>(tsl::Log2Floor64(ratio), 0);
    }
    config.split_k = std::min(config.split_k, max_split_k);

    constexpr int kLdmatrixGranularity = 256;
    // Unfortunately, we need to apply corrections to configurations that are
    // potentially failing due to Triton limitations/bugs.
    ModifyPotentiallyFailingConfig(config, minBitWidth, kLdmatrixGranularity);

    // Sparse meta should have at least one element per thread.
    // Note: only 2:4 structured sparsity is currently supported.
    if (dot.sparse_operands()) {
      config.block_m = std::max(config.block_m, 64);
      config.num_warps = std::max(config.num_warps, 4);
      config.block_k = std::max(
          config.block_k,
          2 * std::max(kMinTileSize, kLdmatrixGranularity / minBitWidth));
      int meta_elements = config.block_m * config.block_k / 16;
      config.num_warps =
          std::min<int>(config.num_warps, meta_elements / threads_per_warp);
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
GemmFusionAutotunerImpl::CompileAll(AutotunerCompileUtil& compile_util,
                                    const BackendConfigs& task) {
  tsl::profiler::ScopedAnnotation annotation("XlaAutotunerCompilation");

  absl::flat_hash_map<const HloFusionInstruction*,
                      std::vector<ExecutableCandidate>>
      results;
  if (task.empty()) {
    return results;
  }

  const int log_every_n = GetLogEveryN();
  int64_t config_count = 0;
  for (const auto& [unused, configs] : task) {
    config_count += configs.size();
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

  auto compile = [&](const HloFusionInstruction* fusion,
                     const BackendConfig& config,
                     bool allow_filtering_kernels_spilling_registers)
      -> absl::StatusOr<std::unique_ptr<Executable>> {
    if (std::holds_alternative<TritonGemmConfig>(config)) {
      return compile_util.Compile([&](const DebugOptions& opts) {
        return TritonGemmAutotuneExtractor(
            std::get<TritonGemmConfig>(config), config_.GetDeviceDescription(),
            fusion, opts, allow_filtering_kernels_spilling_registers);
      });
    }

    if (std::holds_alternative<CuDnnConfig>(config)) {
      return compile_util
          .Compile([&](const DebugOptions& opts) {
            return CuDnnFusionExtractor(*fusion, opts,
                                        std::get<CuDnnConfig>(config).plan_id);
          })
          .value_or(nullptr);
    }

    if (std::holds_alternative<CuBlasConfig>(config)) {
      return compile_util.Compile([&](const DebugOptions& opts) {
        return CublasGemmAutotuneExtractor(config_,
                                           config_.GetDeviceDescription(),
                                           toolkit_version_, fusion, opts);
      });
    }

    if (std::holds_alternative<CustomKernelFusionConfig>(config)) {
      return compile_util.Compile([&](const DebugOptions& opts) {
        return CustomFusionKernelAutotuneExtractor(
            std::get<CustomKernelFusionConfig>(config), config_,
            toolkit_version_, fusion, opts);
      });
    }

    LOG(FATAL) << "Unsupported config type: " << config.index();
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

    absl::BlockingCounter counter(config_count);
    absl::Mutex results_mu;
    for (const auto& key_value : task) {
      const HloFusionInstruction* fusion = key_value.first;
      const std::vector<BackendConfig>& gemm_config_set = key_value.second;

      VLOG(10) << "Compiling fusion: " << fusion->name();
      VLOG(10) << "Dumping fusion computation: "
               << fusion->called_computation()->ToString();
      for (const BackendConfig& config : gemm_config_set) {
        thread_pool_->Schedule([&, fusion] {
          VLOG(10) << "Trying configuration forceable through: "
                      "--xla_gpu_override_gemm_autotuner='"
                   << Serialize(config) << "'";
          VLOG(10) << "WARNING: you are running in multithreaded-mode, the "
                      "last configuration printed out might not be the one "
                      "causing issues! Use "
                      "--xla_gpu_force_compilation_parallelism=1 to fix.";
          absl::StatusOr<std::unique_ptr<Executable>> executable =
              compile(fusion, config, gemm_config_set.size() > 1);
          TF_CHECK_OK(executable.status())
              << " - Failure occured when compiling fusion " << fusion->name()
              << " with config '" << ConfigToString(config)
              << "'\nFused HLO computation:\n"
              << fusion->fused_instructions_computation()->ToString();
          log(*executable != nullptr);
          if (*executable != nullptr) {
            absl::MutexLock lock(&results_mu);
            results[fusion].push_back({config, std::move(*executable)});
          }
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

    for (const auto& [fusion, gemm_config_set] : task) {
      VLOG(10) << "Compiling fusion: " << fusion->name();
      VLOG(10) << "Dumping fusion computation: "
               << fusion->called_computation()->ToString();
      for (const BackendConfig& config : gemm_config_set) {
        VLOG(10) << "Trying configuration forceable through: "
                    "--xla_gpu_override_gemm_autotuner='"
                 << Serialize(config) << "'";
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<Executable> executable,
            compile(fusion, config, gemm_config_set.size() > 1));
        log(executable != nullptr);
        if (executable != nullptr) {
          results[fusion].push_back({config, std::move(executable)});
        }
      }
    }
  }

  VLOG(1) << "Done compiling (successful: " << good_count.load() << ").";
  return results;
}

absl::Status GemmFusionAutotunerImpl::CompareBuffers(
    const HloFusionInstruction& fusion,
    const ScopedShapedBuffer& reference_buffer,
    const ScopedShapedBuffer& buffer, AutotuneResult& res) {
  const HloInstruction& root = *fusion.called_computation_root();
  BufferComparator comparator(root.shape(),
                              debug_options_.xla_gpu_autotune_gemm_rtol());
  TF_ASSIGN_OR_RETURN(se::Stream* const stream, config_.GetStream());

  TF_ASSIGN_OR_RETURN(
      bool outputs_match,
      comparator.CompareEqual(stream, /*current=*/buffer.root_buffer(),
                              /*expected=*/reference_buffer.root_buffer()));

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
  return absl::OkStatus();
}

absl::StatusOr<bool> GemmFusionAutotunerImpl::CheckRedZones(
    const RedzoneBuffers& rz_buffers, AutotuneResult& res) {
  TF_ASSIGN_OR_RETURN(se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
                      rz_buffers.RedzoneAllocator().CheckRedzones());
  if (rz_check_status.ok()) return true;
  LOG(ERROR) << "Red zone modified";
  res.mutable_failure()->set_kind(AutotuneResult::REDZONE_MODIFIED);
  res.mutable_failure()->set_msg(rz_check_status.RedzoneFailureMsg());
  CHECK(!config_.should_crash_on_check_failure());
  return false;
}

absl::StatusOr<AutotuneResult> GemmFusionAutotunerImpl::MeasurePerformance(
    AutotunerCompileUtil& compile_util, const HloFusionInstruction& fusion,
    const ExecutableCandidate& candidate,
    std::optional<ScopedShapedBuffer>& reference_buffer) {
  se::StreamExecutor* stream_exec = config_.GetExecutor();
  if (!stream_exec->SynchronizeAllActivity()) {
    return Internal("Failed to synchronize GPU for autotuning.");
  }
  TF_ASSIGN_OR_RETURN(se::Stream* const stream, config_.GetStream());

  VLOG(5) << "Trying : " << ConfigToString(candidate.config);
  AutotuneResult res = FromConfig(candidate.config);

  const HloComputation* fusion_computation = fusion.called_computation();
  TF_ASSIGN_OR_RETURN(auto rz_buffers,
                      RedzoneBuffers::FromInstruction(
                          *fusion_computation->FusionInstruction(), config_,
                          debug_options_, RedzoneBuffers::kAllInputs));

  TF_ASSIGN_OR_RETURN(
      ProfilingOutput profiling_output,
      compile_util.ProfileExecutable(candidate.executable.get(), stream,
                                     rz_buffers.input_buffers(),
                                     rz_buffers.input_shapes()));

  VLOG(5) << "Running the kernel took: " << profiling_output.duration;
  LOG_IF(WARNING, profiling_output.duration >= absl::Seconds(1))
      << "Slow kernel for " << fusion.called_computation()->ToString()
      << " took: " << profiling_output.duration << ". "
      << ConfigToString(candidate.config);

  *res.mutable_run_time() =
      tsl::proto_utils::ToDurationProto(profiling_output.duration);

  if (!config_.should_check_correctness()) {
    return res;
  }

  if (std::holds_alternative<CuBlasConfig>(candidate.config)) {
    reference_buffer = std::move(profiling_output.output);
    return res;
  }

  // Reference buffer is available when `config.should_check_correctness()`
  // is set and reference executable was compiled.
  if (reference_buffer.has_value()) {
    TF_ASSIGN_OR_RETURN(bool rz_ok, CheckRedZones(rz_buffers, res));
    if (!rz_ok) return res;

    TF_RETURN_IF_ERROR(CompareBuffers(fusion, *reference_buffer,
                                      profiling_output.output, res));
  }
  return res;
}

absl::StatusOr<std::vector<AutotuneResult>> GemmFusionAutotunerImpl::Profile(
    AutotunerCompileUtil& compile_util, const HloFusionInstruction& fusion,
    absl::Span<const ExecutableCandidate> candidates) {
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaAutotunerMeasurement:#hlo_op=%s#",
                           fusion.name());
  });
  VLOG(2) << "Profiling " << fusion.name() << ".";
  std::vector<AutotuneResult> results;
  std::optional<ScopedShapedBuffer> reference_buffer;
  for (int i = 0; i < candidates.size(); ++i) {
    absl::StatusOr<AutotuneResult> result = MeasurePerformance(
        compile_util, fusion, candidates[i], reference_buffer);
    // Treat register allocation error gracefully. If the compilation happens
    // with the driver during execution then the error could surface here.
    // It's enough to check this once here.
    if (stream_executor::IsPtxRegisterAllocationError(result.status())) {
      VLOG(5) << "Skipping candidate: " << ConfigToString(candidates[i].config)
              << ": " << result.status();
      continue;
    }

    if (stream_executor::IsMemoryAllocationError(result.status()) &&
        reference_buffer.has_value()) {
      LOG(WARNING)
          << "Autotuning candidate failed with out of memory error. Consider "
             "disabling correctness checking (i.e. --xla_gpu_autotune_level=3) "
             "to reduce autotuning memory usage.";
    }

    VLOG(2) << "Ran " << i + 1 << " configs out of " << candidates.size()
            << ".";
    TF_RETURN_IF_ERROR(result.status());
    results.push_back(std::move(*result));
  }
  VLOG(2) << "Done profiling " << fusion.name() << ".";
  return results;
}

std::vector<TritonGemmConfig>
GemmFusionAutotunerImpl::GetExhaustiveTritonConfigs() const {
  std::vector<TritonGemmConfig> configs;
  se::GpuComputeCapability gcc = GetComputeCapability();

  bool tune_ctas = !isRocm() && debug_options_.xla_gpu_enable_triton_hopper() &&
                   std::get<se::CudaComputeCapability>(gcc).IsAtLeastHopper();

  const int64_t threads_per_warp =
      config_.GetDeviceDescription().threads_per_warp();

  for (int num_stages : kNumStages) {
    for (int tile_m : kBlockSizes) {
      for (int tile_n : kBlockSizes) {
        for (int tile_k : kBlockSizes) {
          const int tile_lhs = tile_m * tile_k;
          const int tile_rhs = tile_k * tile_n;
          for (int num_warps : kNumWarps) {
            // Each thread should read at least one input element.
            if (num_warps * threads_per_warp > std::min(tile_lhs, tile_rhs)) {
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

static absl::Status DumpAutotuningLogs(const DebugOptions& debug_opts,
                                       const AutotuningLogs& autotuning_logs) {
  if (absl::string_view file_path = debug_opts.xla_gpu_dump_autotune_logs_to();
      !file_path.empty()) {
    std::string resolved_path;
    if (!tsl::io::ResolveTestPrefixes(file_path, resolved_path)) {
      return FailedPrecondition("File path can not be resolved: %s", file_path);
    }

    std::string textproto;
    tsl::protobuf::TextFormat::PrintToString(autotuning_logs, &textproto);

    TF_RETURN_IF_ERROR(
        tsl::WriteStringToFile(tsl::Env::Default(), resolved_path, textproto));
    LOG(INFO) << "Autotune logs serialized to file: " << resolved_path;
  }
  return absl::OkStatus();
}

absl::Status GemmFusionAutotunerImpl::Autotune(
    AutotunerCompileUtil& compile_util, const BackendConfigs& gemm_config_sets,
    AutoTuneCacheKeyCount fusion_count_map) {
  TF_ASSIGN_OR_RETURN(auto executable_sets,
                      CompileAll(compile_util, gemm_config_sets));

  // Sort the candidates to make their execution order well-defined for each
  // fusion.
  for (auto& [unused, candidates] : executable_sets) {
    absl::c_sort(candidates, [](const auto& a, const auto& b) {
      return a.config < b.config;
    });
  }

  AutotuningLogs autotuning_logs;
  int fusion_id = 0;
  for (const auto& [fusion, candidates] : executable_sets) {
    if (debug_options_.xla_gpu_dump_autotuned_gemm_fusions()) {
      TF_RETURN_IF_ERROR(DumpOriginalFusion(compile_util, *fusion, fusion_id));
    }

    TF_ASSIGN_OR_RETURN(std::vector<AutotuneResult> results,
                        Profile(compile_util, *fusion, candidates));

    // The reference config (if it exists) will be the first in the results,
    // due to how sorting the variants work.
    if (!debug_options_.xla_gpu_cublas_fallback() &&
        results.front().has_gemm()) {
      results.erase(results.begin());
    }

    const HloInstruction* root = fusion->called_computation_root();
    TF_ASSIGN_OR_RETURN(
        AutotuneResult best,
        PickBestResult(results, root->ToString(), root->GetModule()->config()));
    VLOG(2) << "Best time: "
            << tsl::proto_utils::FromDurationProto(best.run_time());

    if (debug_options_.xla_gpu_dump_autotuned_gemm_fusions()) {
      TF_RETURN_IF_ERROR(DumpAutotunedFusion(
          config_, toolkit_version_, compile_util, best, fusion, fusion_id++));
    }

    const AutotuneCacheKey key = AutotunerUtil::GetKey(fusion, config_);
    TF_ASSIGN_OR_RETURN(
        bool added, AutotunerUtil::AddResult(key, std::move(best), config_));
    if (!added) {
      // In the context of model server, concurrent autotuning is expected and
      // insertion of identical autotuning keys is accepted.
      LOG(WARNING) << "AutotunerUtil::AddResult already existed: "
                   << key.ToString();
    }

    if (!debug_options_.xla_gpu_dump_autotune_logs_to().empty()) {
      auto autotuning_log = autotuning_logs.add_logs();
      autotuning_log->set_fusion_name(std::string(fusion->name()));

      for (const auto& autotune_result : results) {
        auto log_result = autotuning_log->add_results();
        log_result->CopyFrom(autotune_result);
      }

      if (auto fusion_key_count = fusion_count_map.find(key);
          fusion_key_count != fusion_count_map.end()) {
        auto fusion_key = fusion_key_count->first;
        auto fusion_count = fusion_key_count->second;
        autotuning_log->set_fusion_count(fusion_count);
      }
    }
  }

  return DumpAutotuningLogs(debug_options_, autotuning_logs);
}

// Exchanges the results with the other hosts. The provided fingerprint must be
// sufficiently unique to avoid collisions when invoking the autotuner several
// times without invalidating the relevant key-value store. Collisions may
// result in the wrong results being fetched, leading to non-deterministic
// compilation. A good fingerprint uniquely identifies the input module, and
// the fusions that are being autotuned (up to 128-bit collisions :)).
static absl::Status ExchangeResults(KeyValueStoreInterface& key_value_store,
                                    const AutotuneCacheKeySet& keys_to_send,
                                    absl::string_view fingerprint,
                                    const int shard_index,
                                    const int shard_count) {
  AutotuneResults results;
  TF_RETURN_IF_ERROR(
      AutotunerUtil::SerializeAutotuneResults(&results, &keys_to_send));
  TF_ASSIGN_OR_RETURN(std::string results_str,
                      AutotuneResultsToString(results, true));
  constexpr absl::string_view kKeyPrefix = "gemm_fusion_autotuning_results";
  TF_RET_CHECK(!fingerprint.empty());
  const std::string local_key =
      absl::StrFormat("%s_%s_%d", kKeyPrefix, fingerprint, shard_index);

  absl::StatusOr<std::string> stored_result = key_value_store.TryGet(local_key);
  // Given a sufficiently unique fingerprint, if the result already exists, then
  // we may be recompiling a module that has already been autotuned within the
  // scope of the relevant key-value store. In that case, we don't need to do
  // anything.
  if (stored_result.status().code() == absl::StatusCode::kNotFound) {
    VLOG(2) << "Storing results for " << local_key;
    TF_RETURN_IF_ERROR(key_value_store.Set(local_key, results_str));
  } else if (!stored_result.ok()) {
    return stored_result.status();
  } else {
    // TODO(bchetioui): we should optimize this to avoid even computing the
    // results if they already exist.
    VLOG(2) << "Results already exist for " << local_key << ", skipping store.";
  }

  VLOG(2) << "Rank " << shard_index << ": published results at " << local_key;
  for (int i = 0; i < shard_count; ++i) {
    if (i == shard_index) {
      continue;
    }
    const std::string remote_key =
        absl::StrFormat("%s_%s_%d", kKeyPrefix, fingerprint, i);
    VLOG(2) << "Rank " << shard_index << ": waiting for results from rank " << i
            << " / " << shard_count << " at " << remote_key;
    TF_ASSIGN_OR_RETURN(
        std::string autotune_results_str,
        key_value_store.Get(
            remote_key,
            // TODO(b/361009609): reset to infinite duration once solved.
            // Using an infinite duration here leads to issues with MPI, see
            // https://github.com/google/jax/issues/22995.
            absl::Hours(24)));
    TF_RETURN_IF_ERROR(AutotunerUtil::LoadAutotuneResults(
        autotune_results_str, /*as_textproto=*/true, /*allow_override=*/true));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> GemmFusionAutotuner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER("GEMM fusion autotuner");

  const DebugOptions& debug_options = module->config().debug_options();
  GemmFusionAutotunerImpl autotuner(config_, toolkit_version_, debug_options,
                                    thread_pool_);
  GemmFusionCollector fusion_collector(&autotuner);
  TF_ASSIGN_OR_RETURN(
      GemmFusionCollectorResult fusions,
      fusion_collector.CollectGemmFusions(*module, execution_threads));
  AutotuneCacheKeySet keys_of_this_rank;

  const bool shard_autotuning = debug_options.xla_gpu_shard_autotuning() &&
                                key_value_store_.process_count > 1 &&
                                autotuner.IsAutotuningEnabled();
  if (shard_autotuning) {
    if (key_value_store_.key_value_store == nullptr) {
      return absl::FailedPreconditionError(
          "Sharded autotuning requested but key-value store is missing.");
    }
    fusions.keys_and_instructions = fusion_collector.Shard(
        fusions.keys_and_instructions, key_value_store_.process_index,
        key_value_store_.process_count);
    for (const auto& [key, _] : fusions.keys_and_instructions) {
      CHECK(keys_of_this_rank.insert(key).second);
    }
  }
  TF_ASSIGN_OR_RETURN(
      fusions.keys_and_instructions,
      fusion_collector.RemoveCached(fusions.keys_and_instructions));
  TF_ASSIGN_OR_RETURN(
      const BackendConfigs config_sets,
      fusion_collector.GenerateConfigs(fusions.keys_and_instructions));

  if (!autotuner.IsAutotuningEnabled()) {
    // Pick the first option for each gemm instead of autotuning.
    for (const auto& [fusion, tilings] : config_sets) {
      const AutotuneCacheKey key = AutotunerUtil::GetKey(fusion, config_);
      AutotuneResult res = FromConfig(tilings[0]);
      *res.mutable_run_time() =
          tsl::proto_utils::ToDurationProto(absl::ZeroDuration());
      TF_RETURN_IF_ERROR(AutotunerUtil::AddResult(key, res, config_).status());
    }
  } else if (!debug_options.xla_gpu_override_gemm_autotuner().empty()) {
    // TODO(gflegar): support overriding with non-Triton configs (cuBLAS, cuDNN)
    AutotuneResult::TritonGemmKey gemm_key;
    CHECK(tsl::protobuf::TextFormat::ParseFromString(
        debug_options.xla_gpu_override_gemm_autotuner(), &gemm_key));
    VLOG(1) << "Overriding GEMM autotuner with the following config: "
            << gemm_key.DebugString();
    for (const auto& [fusion, unused] : config_sets) {
      const AutotuneCacheKey key = AutotunerUtil::GetKey(fusion, config_);
      AutotuneResult res;
      *res.mutable_triton() = gemm_key;
      *res.mutable_run_time() =
          tsl::proto_utils::ToDurationProto(absl::ZeroDuration());
      TF_RETURN_IF_ERROR(AutotunerUtil::AddResult(key, res, config_).status());
    }
  } else if (!config_.IsDeviceless()) {
    TF_ASSIGN_OR_RETURN(AutotunerCompileUtil compile_util,
                        AutotunerCompileUtil::Create(config_, debug_options));
    std::string correctness_check_str = config_.should_check_correctness()
                                            ? "(with correctness check)"
                                            : "(without correctness check)";

    const int number_of_fusions_in_module = fusions.fusion_count_map.size();

    VLOG(1) << absl::StrFormat(
        "Rank %d / %d: autotuning %d / %d fusions for %s %s.",
        key_value_store_.process_index, key_value_store_.process_count,
        config_sets.size(), number_of_fusions_in_module, module->name(),
        correctness_check_str);
    TF_RETURN_IF_ERROR(autotuner.Autotune(compile_util, config_sets,
                                          std::move(fusions.fusion_count_map)));
    VLOG(1) << "Done autotuning.";

    // Construct a fingerprint corresponding to a hash of the module as well as
    // the fusions. It is important to fingerprint the module in addition to the
    // fusion to avoid collisions in the key-value store when several distinct
    // modules have the same fusions, and are compiled at different times by the
    // same PjRt client.

    // TODO(b/394763704): find a reliable way to perform sharded autotuning,
    // or eliminate the feature. See below for an explanation of some issues.
    //
    // Theoretically, we also want to include the hash of the module config
    // to ensure that a module compiled twice with different configs is
    // autotuned twice.
    //
    // This is important since the config could e.g. affect codegen, or the
    // space of possible parameters for autotuning. As a result, the autotuning
    // results could look very different for the same module.
    //
    // Why is it not done here? Well, proto serialization is non-deterministic
    // and may change across different builds. Which means that users who run
    // on several hosts with different CPUs may end up generating different
    // fingerprints for the same module config. They would then fail to
    // exchange results through the key value store, which would lead to
    // deadlocks. Therefore, we don't hash the module config here.
    //
    // The flip side is this: if we compile the same module twice in the same
    // client, but with a different module config each time, we may hit the
    // cache the second time and recover potentially inferior, or incomplete
    // autotuning results. This seems like a fairly contrived use case though,
    // and there seems to be no easy way to handle this without breaking through
    // a whole bunch of abstraction layers---so we do this for the time being
    // and will revisit this as we work on fixing the whole autotuning story.
    std::string fingerprint =
        absl::StrCat(module->GetFingerprint128(), "_", fusions.fingerprint);

    if (shard_autotuning && number_of_fusions_in_module > 0) {
      TF_RETURN_IF_ERROR(ExchangeResults(
          *key_value_store_.key_value_store, keys_of_this_rank, fingerprint,
          key_value_store_.process_index, key_value_store_.process_count));
    }
  }

  return GemmFusionAutotunerRewriterVisitor(config_).RunOnModule(
      module, execution_threads);
}

}  // namespace gpu
}  // namespace xla
