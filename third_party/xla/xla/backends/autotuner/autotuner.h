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

#ifndef XLA_BACKENDS_AUTOTUNER_AUTOTUNER_H_
#define XLA_BACKENDS_AUTOTUNER_AUTOTUNER_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/config_runner.h"
#include "xla/backends/autotuner/hlo_extractor.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {

// Can tune configs for a given HLO module and returns the best config for
// each instruction.
class Autotuner {
 public:
  struct Options {
    int scratch_bytes_window_size_us = 2;
    ConfigRunner::CorrectnessCheckOptions correctness_check_options;
  };

  using Config = CodegenOrchestrator::Config;

  static absl::StatusOr<std::unique_ptr<Autotuner>> Create(
      absl_nonnull std::unique_ptr<CodegenOrchestrator> orchestrator,
      std::vector<absl_nonnull std::unique_ptr<Profiler>> profilers,
      Options options, tsl::thread::ThreadPool* thread_pool = nullptr);

  struct TuningResult {
    const HloInstruction* absl_nonnull instruction;
    Config config;
  };

  // This method extracts instructions, compiles them, profiles them,
  // and returns the selected best config for each instruction.
  absl::StatusOr<std::vector<TuningResult>> TuneConfigs(
      const HloModule& module, const InstructionFilterFn& should_autotune,
      bool tolerate_no_supported_configs = false) const;

 private:
  Autotuner(absl_nonnull std::unique_ptr<CodegenOrchestrator> orchestrator,
            std::vector<absl_nonnull std::unique_ptr<ConfigRunner>> runners,
            Options options, tsl::thread::ThreadPool* thread_pool);

  // Returns the best config for the given HLO instruction by profiling all
  // supported configs and selecting the best one.
  // If runner_index is not 0, the runner at the given index will be used
  // instead of the first runner.
  // The method is thread-safe.
  tsl::Future<Config> GetTunedConfig(const HloInstruction* absl_nonnull instr,
                                     int runner_index = 0) const;

  Options options_;

  absl_nonnull std::unique_ptr<CodegenOrchestrator> orchestrator_;
  std::vector<absl_nonnull std::unique_ptr<ConfigRunner>> runners_;
  tsl::thread::ThreadPool* absl_nullable thread_pool_;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_AUTOTUNER_H_
