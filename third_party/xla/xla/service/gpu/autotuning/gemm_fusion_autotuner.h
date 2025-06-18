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
#include <optional>
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
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/autotuner_compile_util.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/autotuning/redzone_buffers.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/shaped_buffer.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// Uses profile results to rewrite a gemm fusion to use the best backend.
class GemmFusionAutotunerRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmFusionAutotunerRewriterVisitor(const AutotuneConfig& config)
      : config_(config) {}

  absl::Status HandleFusion(HloInstruction* fusion_instr) override;

 private:
  AutotuneConfig config_;
};

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
  struct CustomKernelFusionConfig {
    int64_t kernel_index;
    bool operator<(const CustomKernelFusionConfig& other) const;
  };
  using BackendConfig =
      std::variant<CuBlasConfig, CuDnnConfig, CustomKernelFusionConfig,
                   TritonGemmConfig>;
  using BackendConfigs = std::vector<
      std::pair<const HloFusionInstruction*, std::vector<BackendConfig>>>;

  struct ExecutableCandidate {
    BackendConfig config;
    std::unique_ptr<Executable> executable;
    std::optional<AutotuneResult> result;
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

  static const int64_t BLAS_GEMM_DEFAULT;

 private:
  // Measures the performance of a single executable candidate.
  //
  // If required and the candidate is cuBLAS, this will save the output to the
  // reference buffer.
  //
  // If the candidate is not cuBLAS, this will check the redzones and compare
  // the outputs with the reference buffer.
  absl::StatusOr<AutotuneResult> MeasurePerformance(
      AutotunerCompileUtil& compile_util, const HloFusionInstruction& fusion,
      const ExecutableCandidate& candidate,
      std::optional<ScopedShapedBuffer>& reference_buffer);

  // Checks that the redzone buffers are correct, updates `res` otherwise.
  // Returns true if the redzones are correct, false otherwise.
  absl::StatusOr<bool> CheckRedZones(const RedzoneBuffers& rz_buffers,
                                     AutotuneResult& res);

  // Compares the outputs of the fusion with the reference buffer.
  // Updates `res` if the outputs do not match.
  absl::Status CompareBuffers(const HloFusionInstruction& fusion,
                              const ScopedShapedBuffer& reference_buffer,
                              const ScopedShapedBuffer& buffer,
                              AutotuneResult& res);

  se::GpuComputeCapability GetComputeCapability() const {
    return config_.GetGpuComputeCapability();
  }

  bool isRocm() const {
    return std::holds_alternative<se::RocmComputeCapability>(
        GetComputeCapability());
  }

  bool IsFusionKind(const HloInstruction& hlo, absl::string_view kind);

  bool AddLibConfigs(const HloFusionInstruction& fusion,
                     const HloDotInstruction* dot,
                     std::vector<BackendConfig>& configs);

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
