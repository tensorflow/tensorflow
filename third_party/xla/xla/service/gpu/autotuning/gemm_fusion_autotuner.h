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
#ifndef XLA_SERVICE_GPU_AUTOTUNING_GEMM_FUSION_AUTOTUNER_H_
#define XLA_SERVICE_GPU_AUTOTUNING_GEMM_FUSION_AUTOTUNER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/autotuner_compile_util.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/xla.pb.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace gpu {

// Takes a gemm fusion and chooses between cuBLAS, cuDNN, and Triton backends.
// In the case of Triton, it also chooses the best tiling configuration.
//
// This pass uses three steps:
// 1. Generate all possible configs for each dot operation in the fusion.
// 2. Compile all the configs and profile them.
// 3. Rewrite HLO to use the best config.
//
// Note: this pass does not rewrite the fusion to use cuBLAS or cuDNN. This is
// done in a separate pass.
class GemmFusionAutotuner : public HloModulePass {
 public:
  explicit GemmFusionAutotuner(const AutotuneConfig& config,
                               const se::SemanticVersion& toolkit_version,
                               tsl::thread::ThreadPool* thread_pool,
                               const MultiProcessKeyValueStore& key_value_store)
      : config_(config),
        toolkit_version_(toolkit_version),
        thread_pool_(thread_pool),
        key_value_store_(key_value_store) {}

  absl::string_view name() const override { return "gemm-fusion-autotuner"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  AutotuneConfig config_;
  se::SemanticVersion toolkit_version_;
  tsl::thread::ThreadPool* thread_pool_;
  MultiProcessKeyValueStore key_value_store_;
};

class GemmFusionAutotunerImpl {
 public:
  GemmFusionAutotunerImpl(
      AutotuneConfig& config,
      const stream_executor::SemanticVersion& toolkit_version,
      DebugOptions debug_options, tsl::thread::ThreadPool* thread_pool)
      : config_(std::move(config)),
        toolkit_version_(toolkit_version),
        debug_options_(std::move(debug_options)),
        thread_pool_(thread_pool) {}

  struct CuBlasConfig {
    bool operator<(const CuBlasConfig& other) const;
  };
  struct CuDnnConfig {
    int64_t plan_id;
    bool operator<(const CuDnnConfig& other) const;
  };
  using BackendConfig =
      std::variant<CuBlasConfig, CuDnnConfig, TritonGemmConfig>;
  using BackendConfigs = std::vector<
      std::pair<const HloFusionInstruction*, std::vector<BackendConfig>>>;

  struct ExecutableCandidate {
    BackendConfig config;
    std::unique_ptr<Executable> executable;
  };

  // Generate all possible configs for a dot operation.
  absl::StatusOr<std::vector<BackendConfig>> GenerateConfigs(
      const HloFusionInstruction& fusion);
  absl::StatusOr<std::vector<TritonGemmConfig>> GenerateTritonConfigs(
      const HloDotInstruction& dot);

  // Compile all executables for all fusions.
  absl::StatusOr<absl::flat_hash_map<const HloFusionInstruction*,
                                     std::vector<ExecutableCandidate>>>
  CompileAll(AutotunerCompileUtil& compile_util, const BackendConfigs& task);

  // Profile all executables for a fusion.
  absl::StatusOr<std::vector<AutotuneResult>> Profile(
      AutotunerCompileUtil& compile_util, const HloFusionInstruction& fusion,
      absl::Span<const ExecutableCandidate> candidates);

  // Autotune and save the results to the autotuning cache.
  absl::Status Autotune(
      AutotunerCompileUtil& compile_util,
      const BackendConfigs& gemm_config_sets,
      absl::flat_hash_map<AutotuneCacheKey, uint64_t> fusion_count_map);

  // Helper methods.
  const AutotuneConfig& GetConfig() const { return config_; }
  bool IsAutotuningEnabled() const;
  static std::string ToString(const BackendConfig& config);

 private:
  se::CudaComputeCapability GetComputeCapability() const {
    return std::get<se::CudaComputeCapability>(
        config_.GetGpuComputeCapability());
  }

  std::vector<TritonGemmConfig> GetDefaultTritonConfigs() const;
  std::vector<TritonGemmConfig> GetExhaustiveTritonConfigs() const;

  AutotuneConfig config_;
  se::SemanticVersion toolkit_version_;
  DebugOptions debug_options_;
  tsl::thread::ThreadPool* thread_pool_;
  std::vector<TritonGemmConfig> triton_configs_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_AUTOTUNING_GEMM_FUSION_AUTOTUNER_H_
