/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_AUTOTUNER_CONFIG_ASSIGNER_H_
#define XLA_BACKENDS_AUTOTUNER_CONFIG_ASSIGNER_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/config_runner.h"
#include "xla/backends/autotuner/hlo_extractor.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/tsl/concurrency/future.h"

namespace xla {

// ConfigAssigner is responsible for assigning a config to requested HLO
// instructions. The configs could be cached, default, first-supported, or
// tuned.
class ConfigAssigner {
 public:
  // TODO(b/519057668): Consolidate cache fallback options for better
  // readability.
  struct Options {
    bool use_default_config = false;
    bool select_first_config = false;
    bool expect_all_instructions_in_cache = false;
    bool dump_hlos = false;
    std::string dump_logs_to = "";

    // TODO(b/519057668): Remove below option and accept Autotuner instance in
    // ConfigAssigner.
    // Correctness check options
    bool check_buffers = true;
    float relative_tolerance = 1e-6;
    bool crash_on_check_failure = false;

    // Optimal config selection options
    int scratch_bytes_window_size_us = 2;

    std::string ToString() const;
  };

  using Config = CodegenOrchestrator::Config;

  static absl::StatusOr<std::unique_ptr<ConfigAssigner>> Create(
      Options options,
      std::unique_ptr<AutotunerCacheInterface> absl_nonnull
      optimal_config_cache,
      std::unique_ptr<CodegenOrchestrator> absl_nonnull orchestrator,
      std::unique_ptr<Profiler> absl_nullable profiler);

  // Online module-level entry point.
  absl::Status AssignConfigs(HloModule* module,
                             const InstructionFilterFn& should_assign_config);

  // Online sharded module-level entry point.
  absl::Status AssignConfigs(HloModule* module,
                             const InstructionFilterFn& should_assign_config,
                             MultiProcessKeyValueStore& sharding_kv_store);

  // Single instruction entry point.
  absl::Status AssignConfig(HloInstruction* instr);

  AutotunerCacheInterface::CacheStats GetCacheStats() const;

 private:
  ConfigAssigner(Options options,
                 std::unique_ptr<AutotunerCacheInterface> absl_nonnull cache,
                 std::unique_ptr<CodegenOrchestrator> absl_nonnull orchestrator,
                 std::unique_ptr<ConfigRunner> absl_nullable config_runner)
      : options_(options),
        optimal_config_cache_(std::move(cache)),
        orchestrator_(std::move(orchestrator)),
        config_runner_(std::move(config_runner)) {}

  using InstructionGroup = std::vector<HloInstruction*>;

  // Gets the best config for each given instruction and errors out if any of
  // them fails.
  absl::StatusOr<std::vector<Config>> GetConfigsForAll(
      const std::vector<InstructionGroup>& instruction_groups);

  // Returns a future that will contain the best config for the given HLO
  // instruction. The config could be one of the following depending on the
  // options:
  // 1. Check the cache.
  // 2. Check the default config.
  // 3. Check the first supported config.
  // 4. Tune the instruction.
  // Tuned config is updated in the cache if it is provided.
  tsl::Future<Config> GetConfig(const HloInstruction* instr);

  // Tunes and returns the best config. Thread-safe and returns a future that
  // will contain the best config.
  tsl::Future<Config> GetTunedConfig(const HloInstruction* instr);

  // Returns the cached config for the given HLO instruction, if any.
  // Otherwise, returns std::nullopt.
  std::optional<Config> LookUp(const HloInstruction* instr) const;
  // Inserts the given config into the cache for the given HLO instruction.
  absl::Status Insert(const HloInstruction* instr, const Config& config);

  // Dumps HLO before and after applying the config.
  absl::Status DumpHlo(const HloInstruction& instr, const Config& config);

  void LogConfigProfiles(
      const HloInstruction& instr,
      absl::Span<const ConfigRunner::ConfigProfile> profiles,
      absl::Span<const ConfigRunner::ConfigProfile> failed_configs);

  // Dumps the autotuning logs to the specified file path, vlogs if requested.
  absl::Status DumpTuningLogs();

  Options options_;
  std::unique_ptr<AutotunerCacheInterface> absl_nonnull optimal_config_cache_;
  std::unique_ptr<CodegenOrchestrator> absl_nonnull orchestrator_;
  std::unique_ptr<ConfigRunner> absl_nullable config_runner_;
  AutotuningLogs logs_;
  int dump_counter_ = 0;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_CONFIG_ASSIGNER_H_
