/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/triton_autotuner.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iterator>
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
#include "absl/strings/str_cat.h"
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
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/float_normalization.h"
#include "xla/service/gpu/autotuner_compile_util.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_comparator.h"
#include "xla/service/gpu/gemm_rewriter.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/instruction_fusion.h"
#include "xla/service/gpu/ir_emission_utils.h"
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
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/util/proto/proto_utils.h"

// Log levels used in this file:
// VLOG(1): Overview
// VLOG(2): Autotuning progress
// VLOG(3): Autotuning progress - more frequent
// VLOG(4): Print all fusions
// VLOG(5): Profiling information for every tiling

namespace xla {
namespace gpu {

using ProfilingOutput = AutotunerCompileUtil::ProfilingOutput;

namespace {

// Constructs an autotuning key for a gemm performed in Triton.
static AutotuneResult::TritonGemmKey GemmKey(int64_t block_m, int64_t block_n,
                                             int64_t block_k, int64_t split_k,
                                             int64_t num_stages,
                                             int64_t num_warps) {
  AutotuneResult::TritonGemmKey key;
  key.set_block_m(block_m);
  key.set_block_n(block_n);
  key.set_block_k(block_k);
  key.set_split_k(split_k);
  key.set_num_stages(num_stages);
  key.set_num_warps(num_warps);
  return key;
}

// Not a hard limit, just an assumption that should stay valid.
constexpr int kMaxTileSize = 512;

class TritonAutotunerVisitor : public DfsHloRewriteVisitor {
 public:
  explicit TritonAutotunerVisitor(const AutotuneConfig& config)
      : config_(config) {}

  Status HandleFusion(HloInstruction* hlo) override {
    TF_ASSIGN_OR_RETURN(auto backend_config,
                        hlo->backend_config<FusionBackendConfig>());
    if (backend_config.kind() != kTritonGemmFusionKind) {
      return OkStatus();
    }

    VLOG(4) << "Processing " << hlo->ToString();
    if (!backend_config.has_triton_gemm_config()) {
      TF_ASSIGN_OR_RETURN(
          AutotuneResult autotune_result,
          AutotunerUtil::Autotune(
              hlo, config_, [&]() -> StatusOr<AutotuneResult> {
                if (config_.IsDeviceless()) {
                  return InternalError(
                      "Expect autotune result cache hit for deviceless "
                      "compilation.");
                }
                return InternalError("Expect autotune result cache hit.");
              }));
      VLOG(4) << "Result: " << autotune_result.ShortDebugString();

      if (autotune_result.has_triton()) {
        *backend_config.mutable_triton_gemm_config() = autotune_result.triton();
        TF_RETURN_IF_ERROR(hlo->set_backend_config(backend_config));
      } else {
        // Falling back to cuBLAS: Converting the fusion to a Call, so that it
        // can be inlined back again.
        HloComputation* const computation = hlo->parent();
        HloInstruction* const call = computation->AddInstruction(
            HloInstruction::CreateCall(hlo->shape(), hlo->operands(),
                                       hlo->fused_instructions_computation()));
        TF_RETURN_IF_ERROR(computation->ReplaceInstruction(hlo, call));
        hlo = call;
      }
    }

    // This cannot be the "else" branch of the previous "if".
    if (backend_config.has_triton_gemm_config()) {
      const AutotuneResult::TritonGemmKey& tiling =
          backend_config.triton_gemm_config();
      if (tiling.split_k() > 1) {
        TF_RETURN_IF_ERROR(MakeDotSplitKBatch(hlo, tiling));
      }
    }

    MarkAsChanged();
    return OkStatus();
  }

 private:
  AutotuneConfig config_;
};

// This contains all alternative Triton GEMM configs related to one fusion.
struct GemmConfigSet {
  std::vector<AutotuneResult::TritonGemmKey> configs;
};

struct ExecutableCandidate {
  AutotuneResult::TritonGemmKey config;
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

  StatusOr<absl::flat_hash_map<const HloFusionInstruction*, GemmConfigSet>>
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

  Status HandleFusion(const HloInstruction* hlo) override {
    const HloFusionInstruction* fusion = Cast<HloFusionInstruction>(hlo);

    TF_ASSIGN_OR_RETURN(auto backend_config,
                        hlo->backend_config<FusionBackendConfig>());
    if (backend_config.kind() != kTritonGemmFusionKind ||
        backend_config.has_triton_gemm_config()) {
      return OkStatus();
    }

    AutotuneCacheKey key = AutotunerUtil::GetKey(hlo, config_);
    if (AutotunerUtil::IsInCache(key) || handled_fusions_.contains(key)) {
      return OkStatus();
    }

    CHECK(gemm_config_sets_.insert({fusion, GetGemmConfigSet(fusion)}).second);

    handled_fusions_.insert(key);
    return OkStatus();
  }

  Status DefaultAction(const HloInstruction* hlo) override {
    return OkStatus();
  }

 private:
  GemmConfigSet GetGemmConfigSet(const HloFusionInstruction* fusion) {
    const DebugOptions& debug_options =
        fusion->GetModule()->config().debug_options();
    se::StreamExecutor* stream_exec = config_.GetExecutor();
    return {GetPossibleMatmulAutotuneConfigs(
        *Cast<HloDotInstruction>(hlo_query::GetFirstInstructionWithOpcode(
            *fusion->called_computations().at(0), HloOpcode::kDot)),
        stream_exec->GetDeviceDescription().cuda_compute_capability(),
        debug_options, config_.ExhaustiveTilingSearch())};
  }

  AutotuneConfig config_;
  absl::flat_hash_map<const HloFusionInstruction*, GemmConfigSet>
      gemm_config_sets_;
  absl::flat_hash_set<AutotuneCacheKey> handled_fusions_;
};

// Search space for exhaustive matmul autotuning.
constexpr std::array<int, 6> BLOCK_SIZES = {16, 32, 64, 128, 256, 512};
constexpr std::array<int, 4> NUM_STAGES = {1, 2, 3, 4};
constexpr std::array<int, 4> NUM_WARPS = {2, 4, 8, 16};
constexpr std::array<int, 5> SPLIT_K = {1, 2, 4, 8, 16};

std::vector<AutotuneResult::TritonGemmKey> GetExhaustiveMatmulAutotuneConfigs(
    const se::CudaComputeCapability compute_capability, const int max_split_k) {
  std::vector<AutotuneResult::TritonGemmKey> configs;
  bool mma_layout_v2 =
      compute_capability.IsAtLeast(se::CudaComputeCapability::AMPERE);
  for (int num_warps : NUM_WARPS) {
    for (int num_stages : NUM_STAGES) {
      // Volta doesn't support num_stages > 2.
      if (!mma_layout_v2 && num_stages > 2) {
        continue;
      }
      for (int block_m : BLOCK_SIZES) {
        for (int block_n : BLOCK_SIZES) {
          // Exclude configs not supported by MMA layout v2.
          if (mma_layout_v2 && (block_m * block_n / 256) % num_warps != 0) {
            continue;
          }
          for (int block_k : BLOCK_SIZES) {
            for (int split_k : SPLIT_K) {
              if (split_k > max_split_k) {
                continue;
              }
              auto config = GemmKey(block_m, block_n, block_k, split_k,
                                    num_stages, num_warps);
              configs.push_back(std::move(config));
            }
          }
        }
      }
    }
  }
  return configs;
}

std::vector<AutotuneResult::TritonGemmKey> GetFixedMatmulAutotuneConfigs(
    const se::CudaComputeCapability compute_capability, const int max_split_k) {
  std::vector<AutotuneResult::TritonGemmKey> configs = {
      GemmKey(32, 32, 256, 1, 1, 4), GemmKey(64, 32, 32, 16, 1, 4),
      GemmKey(32, 64, 64, 4, 1, 4),  GemmKey(128, 128, 64, 4, 1, 4),
      GemmKey(16, 16, 256, 1, 1, 4), GemmKey(16, 128, 32, 16, 1, 4),
      GemmKey(16, 64, 128, 1, 1, 4), GemmKey(16, 128, 32, 8, 1, 4),
      GemmKey(16, 16, 512, 1, 1, 4), GemmKey(32, 16, 512, 1, 1, 4),
      GemmKey(64, 32, 64, 1, 2, 8)};
  if (compute_capability.IsAtLeast(se::CudaComputeCapability::AMPERE)) {
    absl::c_copy(
        std::vector<AutotuneResult::TritonGemmKey>{
            GemmKey(128, 256, 32, 1, 3, 8),  GemmKey(256, 128, 32, 1, 3, 8),
            GemmKey(256, 64, 32, 1, 4, 4),   GemmKey(64, 256, 32, 1, 4, 4),
            GemmKey(128, 64, 32, 1, 4, 4),   GemmKey(64, 128, 32, 1, 4, 4),
            GemmKey(256, 128, 128, 1, 3, 8), GemmKey(256, 64, 128, 1, 4, 4),
            GemmKey(64, 256, 128, 1, 4, 4),  GemmKey(128, 128, 128, 1, 4, 4),
            GemmKey(128, 64, 64, 1, 4, 4),   GemmKey(64, 128, 64, 1, 4, 4),
            GemmKey(128, 32, 64, 1, 4, 4),   GemmKey(64, 32, 64, 1, 4, 4),
            GemmKey(32, 128, 32, 1, 4, 4),   GemmKey(128, 128, 32, 1, 4, 4),
            GemmKey(16, 16, 256, 1, 3, 4),   GemmKey(128, 128, 64, 2, 1, 8),
            GemmKey(64, 64, 64, 1, 2, 4),    GemmKey(16, 64, 256, 8, 1, 4),
            GemmKey(256, 256, 128, 1, 3, 8)},
        std::back_inserter(configs));
  }
  if (compute_capability.IsAtLeast(se::CudaComputeCapability::HOPPER)) {
    configs.erase(
        std::remove_if(configs.begin(), configs.end(),
                       [](const AutotuneResult::TritonGemmKey& config) {
                         return (config.block_m() * config.block_n() / 256) %
                                    config.num_warps() !=
                                0;
                       }),
        configs.end());
  }
  configs.erase(
      std::remove_if(configs.begin(), configs.end(),
                     [&](const AutotuneResult::TritonGemmKey& config) {
                       return config.split_k() > max_split_k;
                     }),
      configs.end());
  return configs;
}

int GetLogEveryN() { return VLOG_IS_ON(3) ? 100 : 1000; }

StatusOr<std::unique_ptr<HloModule>> TritonGemmAutotuneExtractor(
    const AutotuneResult::TritonGemmKey& key,
    const se::DeviceDescription& gpu_device_info,
    const HloFusionInstruction* fusion, DebugOptions debug_opts) {
  std::unique_ptr<HloModule> new_module =
      AutotunerUtil::ExtractInstructionIntoNewModule(*fusion);
  // Reduce memory usage during compilation by disabling GPU runtime.
  debug_opts.set_xla_gpu_enable_xla_runtime_executable(false);
  new_module->mutable_config().set_debug_options(debug_opts);

  HloComputation* entry_computation = new_module->entry_computation();
  HloInstruction* cloned_dot_fusion = entry_computation->root_instruction();

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      cloned_dot_fusion->backend_config<FusionBackendConfig>());
  *backend_config.mutable_triton_gemm_config() = key;
  TF_RETURN_IF_ERROR(cloned_dot_fusion->set_backend_config(backend_config));

  if (key.split_k() > 1) {
    TF_RETURN_IF_ERROR(MakeDotSplitKBatch(cloned_dot_fusion, key));
    GpuFloatSupport bf16_support(BF16);
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
              root->shape(), ChooseFusionKind(*root->operand(0), *root), root));
      HloInstruction* init_value = root->mutable_operand(1);
      TF_CHECK_OK(
          entry_computation->ReplaceInstruction(root, fusion_instruction));
      fusion_instruction->FuseInstruction(init_value);
      TF_CHECK_OK(entry_computation->RemoveInstruction(init_value));
    }
  }
  return new_module;
}

StatusOr<std::unique_ptr<HloModule>> CublasGemmAutotuneExtractor(
    const AutotuneConfig& config, const HloFusionInstruction* fusion,
    const DebugOptions& debug_opts) {
  const HloComputation* fusion_computation =
      fusion->called_computations().at(0);
  std::unique_ptr<HloModule> new_module =
      AutotunerUtil::ExtractComputationIntoNewModule(*fusion_computation);
  new_module->mutable_config().set_debug_options(debug_opts);

  GemmRewriter rewriter(config.GetCudaComputeCapability());
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

StatusOr<absl::flat_hash_map<const HloFusionInstruction*, ExecutableSet>>
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
    const GemmConfigSet& gemm_config_set = key_value.second;
    config_count += gemm_config_set.configs.size();
  }
  // The cuBLAS configs:
  config_count += gemm_config_sets.size();

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
  auto compile =
      [&](const HloFusionInstruction* fusion,
          const AutotuneResult::TritonGemmKey& conf) -> StatusOr<bool> {
    CHECK(conf.block_m() <= kMaxTileSize);
    CHECK(conf.block_n() <= kMaxTileSize);
    CHECK(conf.block_k() <= kMaxTileSize);
    // TODO(b/296884861): Reenable GPU runtime, when it will have much smaller
    // memory overhead (regarding the size of the executables).
    // We can also remove the force_disable_gpu_runtime argument at that
    // point.
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                        util.Compile([&](const DebugOptions& opts) {
                          return TritonGemmAutotuneExtractor(
                              conf, gpu_device_info, fusion, opts);
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
      [&](const HloFusionInstruction* fusion) -> StatusOr<bool> {
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

      for (const AutotuneResult::TritonGemmKey& conf :
           gemm_config_set.configs) {
        thread_pool->Schedule([&, fusion] {
          StatusOr<bool> has_executable = compile(fusion, conf);
          TF_CHECK_OK(has_executable.status())
              << "Failure occured when compiling fusion " << fusion->name()
              << " with config '" << conf.ShortDebugString()
              << "'\nFused HLO computation:\n"
              << fusion->fused_instructions_computation()->ToString();
          log(has_executable.value());
          counter.DecrementCount();
        });
      }

      thread_pool->Schedule([&, fusion] {
        StatusOr<bool> has_executable = compile_reference_executable(fusion);
        TF_CHECK_OK(has_executable.status());
        log(has_executable.value());
        counter.DecrementCount();
      });
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

      for (const AutotuneResult::TritonGemmKey& gemm_config :
           gemm_config_set.configs) {
        TF_ASSIGN_OR_RETURN(bool has_executable, compile(fusion, gemm_config));
        log(has_executable);
      }

      TF_ASSIGN_OR_RETURN(bool has_executable,
                          compile_reference_executable(fusion));
      log(has_executable);
    }
  }

  VLOG(1) << "Done compiling (successful: " << good_count.load() << ").";

  return executable_sets;
}

// Runs matmul fusion contents without Triton - with cuBLAS, to measure time and
// generate a reference output.
StatusOr<ProfilingOutput> RunMatmulWithCublas(
    AutotunerCompileUtil& util, se::Stream* stream, Executable& executable,
    absl::Span<se::DeviceMemoryBase const> input_buffers,
    absl::Span<Shape const> input_shapes) {
  TF_ASSIGN_OR_RETURN(
      std::optional<ProfilingOutput> output,
      util.ProfileExecutable(&executable, stream, input_buffers, input_shapes));
  TF_RET_CHECK(output.has_value());
  return std::move(output.value());
}

StatusOr<AutotuneResult> Execute(const AutotuneConfig& config,
                                 AutotunerCompileUtil& util,
                                 const DebugOptions& debug_opts,
                                 const HloFusionInstruction* fusion,
                                 const ExecutableSet& executable_set) {
  const HloComputation* fusion_computation =
      fusion->called_computations().at(0);

  se::StreamExecutor* stream_exec = config.GetExecutor();
  if (!stream_exec->SynchronizeAllActivity()) {
    return InternalError("Failed to synchronize GPU for autotuning.");
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

  // Run with cuBLAS.
  std::optional<ScopedShapedBuffer> reference_buffer;
  absl::Duration cublas_duration;
  {
    TF_RET_CHECK(executable_set.reference != nullptr);
    TF_ASSIGN_OR_RETURN(
        ProfilingOutput output,
        RunMatmulWithCublas(util, stream, *executable_set.reference, inputs,
                            input_shapes));
    if (config.should_check_correctness()) {
      reference_buffer = std::move(output.output);
    }
    cublas_duration = output.duration;
  }

  const int log_every_n = GetLogEveryN();
  int64_t executable_count =
      static_cast<int64_t>(executable_set.candidates.size());
  int ran_so_far = 0;
  std::vector<AutotuneResult> results;
  VLOG(2) << "Running " << executable_count << " configs for " << fusion->name()
          << ".";
  for (const ExecutableCandidate& candidate : executable_set.candidates) {
    VLOG(5) << "Trying triton tiling: " << candidate.config.ShortDebugString();

    AutotuneResult res;
    *res.mutable_triton() = candidate.config;

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
                   << " took: " << profiling_output->duration
                   << ". config: " << candidate.config.ShortDebugString();
    }
    *res.mutable_run_time() =
        tsl::proto_utils::ToDurationProto(profiling_output->duration);

    if (config.should_check_correctness()) {
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
    results.push_back(res);
  }
  VLOG(2) << "Done running.";

  TF_ASSIGN_OR_RETURN(
      AutotuneResult best_triton,
      PickBestResult(results, root.ToString(), root.GetModule()->config()));

  if (debug_opts.xla_gpu_cublas_fallback() &&
      !debug_opts.xla_gpu_deterministic_ops()) {
    const absl::Duration best_triton_duration =
        tsl::proto_utils::FromDurationProto(best_triton.run_time());
    VLOG(2) << fusion->name() << ": time with cuBLAS: " << cublas_duration
            << ", best time with Triton: " << best_triton_duration;
    if (cublas_duration < best_triton_duration) {
      VLOG(2) << "Falling back to cuBLAS for " << fusion->name();

      AutotuneResult cublas;
      *cublas.mutable_run_time() =
          tsl::proto_utils::ToDurationProto(cublas_duration);
      // We will ignore this value anyway.
      cublas.mutable_gemm()->set_algorithm(CUBLAS_GEMM_DEFAULT);

      return cublas;
    }
  }

  return best_triton;
}

Status DumpAutotunedFusions(const AutotuneConfig& config,
                            AutotunerCompileUtil& util,
                            const AutotuneResult result,
                            const HloFusionInstruction* fusion, int fusion_id) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      util.ExtractModule([&](const DebugOptions& debug_opts) {
                        return TritonGemmAutotuneExtractor(
                            result.triton(),
                            config.GetExecutor()->GetDeviceDescription(),
                            fusion, debug_opts);
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
  return OkStatus();
}

Status Autotune(const AutotuneConfig& config, AutotunerCompileUtil& util,
                tsl::thread::ThreadPool* thread_pool,
                const DebugOptions& debug_opts,
                const absl::flat_hash_map<const HloFusionInstruction*,
                                          GemmConfigSet>& gemm_config_sets,
                int& fusion_id_for_dump) {
  absl::flat_hash_map<const HloFusionInstruction*, ExecutableSet>
      executable_sets;
  TF_ASSIGN_OR_RETURN(
      executable_sets,
      CompileMany(config, util, thread_pool, debug_opts, gemm_config_sets));

  for (const auto& key_value : executable_sets) {
    const HloFusionInstruction* fusion = key_value.first;
    const ExecutableSet& executable_set = key_value.second;

    TF_ASSIGN_OR_RETURN(AutotuneResult result, Execute(config, util, debug_opts,
                                                       fusion, executable_set));

    if (debug_opts.xla_gpu_dump_autotuned_triton_fusions()) {
      TF_RETURN_IF_ERROR(DumpAutotunedFusions(config, util, result, fusion,
                                              fusion_id_for_dump));
    }

    const AutotuneCacheKey key = AutotunerUtil::GetKey(fusion, config);
    if (!AutotunerUtil::AddResult(key, std::move(result))) {
      // In the context of model server, concurrent autotuning is expected and
      // insertion of identical autotuning keys is accepted.
      LOG(WARNING) << "AutotunerUtil::AddResult already existed: "
                   << key.ToString();
    }

    fusion_id_for_dump += 1;
  }

  return OkStatus();
}

}  // anonymous namespace

std::vector<AutotuneResult::TritonGemmKey> GetPossibleMatmulAutotuneConfigs(
    const HloDotInstruction& dot,
    const se::CudaComputeCapability compute_capability,
    const DebugOptions& debug_options, bool exhaustive_tiling_search) {
  // Avoid autotuning tiny fusions.
  constexpr int kMinGemmElements = 32 * 32;
  if (ShapeUtil::ElementsIn(dot.operand(0)->shape()) <= kMinGemmElements &&
      ShapeUtil::ElementsIn(dot.operand(1)->shape()) <= kMinGemmElements) {
    return {GemmKey(32, 32, 32, 1, 1, 4)};
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
             ? GetExhaustiveMatmulAutotuneConfigs(compute_capability,
                                                  max_split_k)
             : GetFixedMatmulAutotuneConfigs(compute_capability, max_split_k);
}

StatusOr<bool> TritonAutotuner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER("Triton autotuner");
  const DebugOptions& debug_options = module->config().debug_options();
  if (debug_options.xla_gpu_autotune_level() == 0) {
    return false;
  }

  if (!config_.IsDeviceless()) {
    TF_ASSIGN_OR_RETURN(std::optional<AutotunerCompileUtil> opt_compile_util,
                        AutotunerCompileUtil::Create(config_, debug_options));
    TF_RET_CHECK(opt_compile_util.has_value());
    AutotunerCompileUtil& compile_util = opt_compile_util.value();

    GemmConfigSetCollector gemm_config_set_collector(config_);
    absl::flat_hash_map<const HloFusionInstruction*, GemmConfigSet>
        gemm_config_sets;
    TF_ASSIGN_OR_RETURN(gemm_config_sets,
                        gemm_config_set_collector.CollectGemmConfigSets(
                            module, execution_threads));
    if (!gemm_config_sets.empty()) {
      std::string correctness_check_str = config_.should_check_correctness()
                                              ? "(with correctness check)"
                                              : "(without correctness check)";

      VLOG(1) << "Autotuning " << gemm_config_sets.size() << " fusions "
              << correctness_check_str << ".";
      int fusion_id_for_dump = 0;
      if (debug_options.xla_gpu_single_wave_autotuning()) {
        // Tune all fusions at once to save time.
        TF_RETURN_IF_ERROR(Autotune(config_, compile_util, thread_pool_,
                                    debug_options, gemm_config_sets,
                                    fusion_id_for_dump));
      } else {
        // Tune each fusion separately to avoid running out of memory.
        for (const auto& key_value : gemm_config_sets) {
          absl::flat_hash_map<const HloFusionInstruction*, GemmConfigSet>
              single_element_map({key_value});
          TF_RETURN_IF_ERROR(Autotune(config_, compile_util, thread_pool_,
                                      debug_options, single_element_map,
                                      fusion_id_for_dump));
        }
      }
      VLOG(1) << "Done autotuning.";
    }
  }

  return TritonAutotunerVisitor(config_).RunOnModule(module, execution_threads);
}

}  // namespace gpu
}  // namespace xla
