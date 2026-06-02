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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/autotuner/tuner.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {

using InstructionFilterFn = std::function<bool(const xla::HloInstruction&)>;

struct AutotuneConfig {
  bool check_buffers = true;
  float relative_tolerance = 1e-6;
  bool crash_on_check_failure = false;
  int scratch_bytes_window_size_us = 2;
  bool expect_all_instructions_in_cache = false;
  std::string dump_logs_to = "";
  bool exclude_cublas_config = false;
  bool select_first_config = false;
  bool use_default_config = false;
  bool dump_hlos = false;
  std::function<bool(const HloInstruction&)> allow_reg_spills_fn =
      [](const HloInstruction&) { return false; };

  std::string ToString() const;
};

class ConfigAssigner {
 public:
  struct Options {
    bool use_default_config = false;
    bool select_first_config = false;
    bool expect_all_instructions_in_cache = false;
    bool dump_hlos = false;
  };

  using Config = CodegenOrchestrator::Config;

  static absl::StatusOr<std::unique_ptr<ConfigAssigner>> Create(
      Options options, std::unique_ptr<AutotunerCacheInterface> cache,
      std::unique_ptr<CodegenOrchestrator> orchestrator,
      std::unique_ptr<Tuner> tuner = nullptr);

  static absl::StatusOr<std::unique_ptr<ConfigAssigner>> Create(
      std::vector<std::unique_ptr<CodegenBackend>> codegen_backends,
      std::unique_ptr<Profiler> profiler, AutotuneConfig autotune_config,
      std::unique_ptr<AutotunerCacheInterface> cache,
      tsl::thread::ThreadPool* thread_pool = nullptr);

  // Online module-level entry point.
  absl::Status AssignConfigs(HloModule* module,
                             const InstructionFilterFn& should_autotune);

  // Online sharded module-level entry point.
  absl::Status AssignConfigs(HloModule* module,
                             const InstructionFilterFn& should_autotune,
                             MultiProcessKeyValueStore& sharding_kv_store);

  // Single instruction entry point.
  absl::Status AssignConfig(HloInstruction* instr);

  AutotunerCacheInterface* cache() { return cache_.get(); }
  Tuner* tuner() { return tuner_.get(); }
  CodegenOrchestrator* orchestrator() { return orchestrator_.get(); }
  const Options& options() const { return options_; }

 private:
  ConfigAssigner(Options options,
                 std::unique_ptr<AutotunerCacheInterface> cache,
                 std::unique_ptr<CodegenOrchestrator> orchestrator,
                 std::unique_ptr<Tuner> tuner)
      : options_(options),
        cache_(std::move(cache)),
        orchestrator_(std::move(orchestrator)),
        tuner_(std::move(tuner)) {}

  // Implements the fallback strategy (cache -> default/first -> tuned).
  tsl::Future<Config> GetConfig(HloInstruction* instr);

  std::optional<Config> LookUp(const HloInstruction* instr);
  absl::Status Insert(const HloInstruction* instr, Config& config);

  // Dumps HLO before and after applying the config.
  absl::Status DumpHlo(HloInstruction* instr, const Config& config);

  Options options_;
  std::unique_ptr<AutotunerCacheInterface> cache_;
  std::unique_ptr<CodegenOrchestrator> orchestrator_;
  std::unique_ptr<Tuner> tuner_;
  int dump_counter_ = 0;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_CONFIG_ASSIGNER_H_
