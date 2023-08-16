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

#include "tensorflow/compiler/xla/service/gpu/triton_autotuner.h"

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
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/autotuning.pb.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_clone_context.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/float_normalization.h"
#include "tensorflow/compiler/xla/service/gpu/autotuner_compile_util.h"
#include "tensorflow/compiler/xla/service/gpu/autotuner_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_float_support.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/tsl/platform/blocking_counter.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/threadpool.h"
#include "tensorflow/tsl/util/proto/proto_utils.h"

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

struct CompilationKey {
  template <typename H>
  friend H AbslHashValue(H h, const CompilationKey& k) {
    return H::combine(std::move(h), k.key.SerializeAsString());
  }

  bool operator==(const CompilationKey& k) const {
    return key.SerializeAsString() == k.key.SerializeAsString();
  }

  AutotuneResult::TritonGemmKey key;
};

class TritonAutotunerVisitor : public DfsHloRewriteVisitor {
 public:
  TritonAutotunerVisitor(
      const AutotuneConfig& config, tsl::thread::ThreadPool* thread_pool,
      std::optional<AutotunerCompileUtil> autotuner_compile_util)
      : config_(config),
        thread_pool_(thread_pool),
        autotuner_compile_util_(autotuner_compile_util) {}

  Status HandleFusion(HloInstruction* hlo) override {
    TF_ASSIGN_OR_RETURN(auto backend_config,
                        hlo->backend_config<FusionBackendConfig>());
    if (backend_config.kind() != kTritonGemmFusionKind) {
      return OkStatus();
    }

    VLOG(1) << "Processing " << hlo->ToString();
    if (!backend_config.has_triton_gemm_config()) {
      TF_ASSIGN_OR_RETURN(AutotuneResult autotune_result,
                          AutotunerUtil::Autotune(hlo, config_, [&] {
                            return AutotuneMatmulNoCache(hlo);
                          }));
      VLOG(1) << "Result: " << autotune_result.ShortDebugString();

      TF_RET_CHECK(autotune_result.has_triton());
      *backend_config.mutable_triton_gemm_config() = autotune_result.triton();
      TF_RETURN_IF_ERROR(hlo->set_backend_config(backend_config));
    }
    const AutotuneResult::TritonGemmKey& tiling =
        backend_config.triton_gemm_config();
    if (tiling.split_k() > 1) {
      TF_RETURN_IF_ERROR(MakeDotSplitKBatch(hlo, tiling));
    }

    MarkAsChanged();
    return OkStatus();
  }

 private:
  // Autotunes a matmul without using the autotuning cache.
  StatusOr<AutotuneResult> AutotuneMatmulNoCache(const HloInstruction* instr) {
    if (config_.IsDeviceless()) {
      return InternalError(
          "Expect autotune result cache hit for deviceless compilation.");
    }

    const HloComputation& fusion = *instr->called_computations()[0];
    se::StreamExecutor* stream_exec = config_.GetExecutor();
    if (!stream_exec->SynchronizeAllActivity()) {
      return InternalError("Failed to synchronize GPU for autotuning.");
    }
    se::DeviceMemoryAllocator* allocator = config_.GetAllocator();
    if (allocator == nullptr) {
      allocator = stream_exec->GetAllocator();
    }

    TF_ASSIGN_OR_RETURN(se::Stream* const stream,
                        allocator->GetStream(stream_exec->device_ordinal()));

    const DebugOptions& debug_opts = fusion.parent()->config().debug_options();

    TF_ASSIGN_OR_RETURN(
        se::RedzoneAllocator rz_allocator,
        AutotunerUtil::CreateRedzoneAllocator(config_, debug_opts));

    std::optional<ScopedShapedBuffer> reference_buffer;
    const HloInstruction& root = *fusion.root_instruction();
    BufferComparator comparator(root.shape(), fusion.parent()->config());

    const std::vector<AutotuneResult::TritonGemmKey> configurations =
        GetPossibleMatmulAutotuneConfigs(
            root, stream_exec->GetDeviceDescription().cuda_compute_capability(),
            config_.ExhaustiveTilingSearch());

    GpuDeviceInfo gpu_device_info = GetGpuDeviceInfo(config_.GetExecutor());

    absl::Mutex executables_mutex;
    absl::flat_hash_map<CompilationKey, std::unique_ptr<Executable>>
        executables;

    const int log_every_n = VLOG_IS_ON(1) ? 100 : 1000;

    auto compile =
        [&](const AutotuneResult::TritonGemmKey& conf) -> StatusOr<bool> {
      CHECK(conf.block_m() <= kMaxTileSize);
      CHECK(conf.block_n() <= kMaxTileSize);
      CHECK(conf.block_k() <= kMaxTileSize);
      TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                          autotuner_compile_util_->Compile([&] {
                            return TritonGemmAutotuneExtractor(
                                conf, gpu_device_info,
                                fusion.FusionInstruction());
                          }));
      const bool has_executable = executable != nullptr;
      absl::MutexLock lock(&executables_mutex);
      CHECK(executables.emplace(CompilationKey{conf}, std::move(executable))
                .second);
      return has_executable;
    };

    // Pre-compile all versions first.
    int executable_count = 0;
    if (thread_pool_ &&
        debug_opts.xla_gpu_force_compilation_parallelism() != 1) {
      const int config_count = configurations.size();
      LOG(INFO) << "Compiling " << config_count << " configs for "
                << instr->name() << " on " << thread_pool_->NumThreads()
                << " threads.";

      tsl::BlockingCounter counter(config_count);
      std::atomic<int> done_count = 0;
      std::atomic<int> good_count = 0;
      for (const AutotuneResult::TritonGemmKey& conf : configurations) {
        thread_pool_->Schedule([&] {
          StatusOr<bool> has_executable = compile(conf);
          TF_CHECK_OK(has_executable.status());

          const int done_so_far = done_count.fetch_add(1) + 1;
          const int good_so_far = has_executable.value()
                                      ? good_count.fetch_add(1) + 1
                                      : good_count.load();
          if (done_so_far % log_every_n == 0) {
            LOG(INFO) << "Compiled " << done_so_far << " of " << config_count
                      << " configs (successful: " << good_so_far << ")";
          }

          counter.DecrementCount();
        });
      }
      counter.Wait();
      executable_count = good_count.load();
    } else {
      LOG(WARNING) << "Compiling " << configurations.size() << " configs for "
                   << instr->name() << " on a single thread.";

      int good_so_far = 0;
      for (int i = 0; i < static_cast<int>(configurations.size()); ++i) {
        const AutotuneResult::TritonGemmKey& conf = configurations[i];
        TF_ASSIGN_OR_RETURN(bool has_executable, compile(conf));

        const int done_so_far = i + 1;
        if (has_executable) {
          good_so_far += 1;
        }

        if (done_so_far % log_every_n == 0) {
          LOG(INFO) << "Compiled " << done_so_far << " of "
                    << configurations.size()
                    << " configs (successful: " << good_so_far
                    << ", using a single thread).";
        }
      }
      executable_count = good_so_far;
    }
    LOG(INFO) << "Compiled all configs (successful: " << executable_count
              << ").";

    std::vector<se::DeviceMemoryBase> inputs;
    int64_t rng_state = 0;
    for (const HloInstruction* param : fusion.parameter_instructions()) {
      TF_ASSIGN_OR_RETURN(
          se::DeviceMemoryBase param_buffer,
          AutotunerUtil::CreateBuffer(rz_allocator, param->shape(), config_,
                                      rng_state));
      inputs.push_back(param_buffer);
    }

    if (config_.should_check_correctness()) {
      TF_ASSIGN_OR_RETURN(
          reference_buffer,
          RunMatmulWithCublas(fusion, stream, allocator, inputs));
    }

    std::vector<AutotuneResult> results;

    LOG(INFO) << "Running " << executable_count << " configs for "
              << instr->name() << ".";
    int ran_so_far = 0;
    for (const AutotuneResult::TritonGemmKey& conf : configurations) {
      VLOG(2) << "Trying triton tiling: " << conf.ShortDebugString();

      AutotuneResult res;
      *res.mutable_triton() = conf;

      auto it = executables.find(CompilationKey{conf});
      if (it == executables.end() || it->second == nullptr) {
        VLOG(2) << "Skipping this tiling.";
        continue;
      }
      TF_ASSIGN_OR_RETURN(std::optional<ProfilingOutput> profiling_output,
                          autotuner_compile_util_->ProfileExecutable(
                              it->second.get(), stream, inputs));
      if (++ran_so_far % log_every_n == 0) {
        LOG(INFO) << "Ran " << ran_so_far << " configs of " << executable_count
                  << ".";
      }

      if (!profiling_output) {
        VLOG(2) << "Skipping this tiling.";
        continue;
      }

      VLOG(2) << "Running the kernel took: " << profiling_output->duration;
      if (profiling_output->duration >= absl::Seconds(1)) {
        LOG(WARNING) << "Slow kernel for " << instr->name()
                     << " took: " << profiling_output->duration
                     << ". config: " << conf.ShortDebugString();
      }
      *res.mutable_run_time() =
          tsl::proto_utils::ToDurationProto(profiling_output->duration);

      if (config_.should_check_correctness()) {
        TF_ASSIGN_OR_RETURN(
            se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
            rz_allocator.CheckRedzones());
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
      results.push_back(res);
    }
    LOG(INFO) << "Ran all configs.";

    TF_ASSIGN_OR_RETURN(
        AutotuneResult best,
        PickBestResult(results, root.ToString(), root.GetModule()->config()));

    if (debug_opts.xla_gpu_dump_autotuned_triton_fusions()) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<HloModule> module,
          TritonGemmAutotuneExtractor(best.triton(),
                                      GetGpuDeviceInfo(config_.GetExecutor()),
                                      fusion.FusionInstruction()));
      module->set_name(std::string(fusion.FusionInstruction()->name()));
      // Using the original module for its debug info and name in the first
      // parameter. It's better to include the name of both the original module
      // and the extracted module, to avoid name clashes.
      DumpToFileInDirOrStdout(
          /*module=*/*instr->GetModule(),
          /*file_prefix=*/"",
          /*file_suffix=*/
          absl::StrCat("triton_fusion_", fusion_id_for_dump_++, ".",
                       module->name(), ".optimized.txt"),
          /*contents=*/module->ToString());
    }

    return best;
  }

  StatusOr<std::unique_ptr<HloModule>> TritonGemmAutotuneExtractor(
      const AutotuneResult::TritonGemmKey& key,
      const GpuDeviceInfo& gpu_device_info, const HloInstruction* fusion) {
    std::unique_ptr<HloModule> new_module =
        AutotunerUtil::ExtractInstructionIntoNewModule(*fusion);
    HloComputation* entry_computation = new_module->entry_computation();
    HloInstruction* cloned_dot_fusion = entry_computation->root_instruction();

    TF_ASSIGN_OR_RETURN(
        auto backend_config,
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
                root->shape(), ChooseFusionKind(*root->operand(0), *root),
                root));
        HloInstruction* init_value = root->mutable_operand(1);
        TF_CHECK_OK(
            entry_computation->ReplaceInstruction(root, fusion_instruction));
        fusion_instruction->FuseInstruction(init_value);
        TF_CHECK_OK(entry_computation->RemoveInstruction(init_value));
      }
    }
    return new_module;
  }

  // Runs matmul fusion contents without Triton - with cuBLAS, to generate
  // a reference output.
  StatusOr<ScopedShapedBuffer> RunMatmulWithCublas(
      const HloComputation& original_computation, se::Stream* stream,
      se::DeviceMemoryAllocator* allocator,
      absl::Span<se::DeviceMemoryBase const> input_buffers) {
    StatusOr<std::unique_ptr<Executable>> executable =
        autotuner_compile_util_->Compile([&] {
          return CublasGemmAutotuneExtractor(
              GetGpuDeviceInfo(config_.GetExecutor()), &original_computation);
        });
    TF_ASSIGN_OR_RETURN(std::optional<ProfilingOutput> output,
                        autotuner_compile_util_->ProfileExecutable(
                            executable->get(), stream, input_buffers));
    TF_RET_CHECK(output.has_value());
    return std::move(output->output);
  }

  StatusOr<std::unique_ptr<HloModule>> CublasGemmAutotuneExtractor(
      const GpuDeviceInfo& gpu_device_info, const HloComputation* fusion) {
    std::unique_ptr<HloModule> new_module =
        AutotunerUtil::ExtractComputationIntoNewModule(*fusion);
    GemmRewriter rewriter(config_.GetCudaComputeCapability());
    GpuInstructionFusion fusion_pass(/*may_duplicate=*/false,
                                     GetGpuDeviceInfo(config_.GetExecutor()));
    TF_RETURN_IF_ERROR(rewriter.Run(new_module.get()).status());
    TF_RETURN_IF_ERROR(fusion_pass.Run(new_module.get()).status());
    return new_module;
  }

  AutotuneConfig config_;
  tsl::thread::ThreadPool* thread_pool_;
  std::optional<AutotunerCompileUtil> autotuner_compile_util_;
  int fusion_id_for_dump_ = 0;
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

}  // anonymous namespace

std::vector<AutotuneResult::TritonGemmKey> GetPossibleMatmulAutotuneConfigs(
    const HloInstruction& instr,
    const se::CudaComputeCapability compute_capability,
    bool exhaustive_tiling_search) {
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
      std::max(1L, kSufficientNumberOfTiles * kMaxTileSize * kMaxTileSize /
                       ShapeUtil::ElementsIn(instr.shape()));
  return exhaustive_tiling_search
             ? GetExhaustiveMatmulAutotuneConfigs(compute_capability,
                                                  max_split_k)
             : GetFixedMatmulAutotuneConfigs(compute_capability, max_split_k);
}

StatusOr<bool> TritonAutotuner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER("Triton autotuner");
  if (module->config().debug_options().xla_gpu_autotune_level() == 0) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(
      std::optional<AutotunerCompileUtil> autotuner_compile_util,
      AutotunerCompileUtil::Create(config_, module->config().debug_options()));
  return TritonAutotunerVisitor{config_, thread_pool_, autotuner_compile_util}
      .RunOnModule(module, execution_threads);
}

}  // namespace gpu
}  // namespace xla
