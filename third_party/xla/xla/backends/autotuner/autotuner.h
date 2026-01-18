/* Copyright 2025 The OpenXLA Authors.

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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/executable.h"
#include "xla/service/shaped_buffer.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/fingerprint.h"

using InstructionFilterFn = std::function<bool(const xla::HloInstruction&)>;

namespace xla {

struct AutotuneConfig {
  // Whether to check the correctness of the output buffers and OOM reads on
  // Input Buffers.
  // Correctness check is only performed when a trustable reference output is
  // available.
  bool check_buffers = true;
  // Relative tolerance for correctness check.
  float relative_tolerance = 1e-6;
  // Whether to crash the process on check failure.
  bool crash_on_check_failure = false;
  // If true, in addition to the duration, the best algorithm will be chosen
  // based on the scratch bytes. This is only useful if backends use scratch
  // space for temporary tensors. The best config will be the one with the
  // smallest scratch space among top minimum duration configs in
  // scratch_bytes_window_size_us window.
  bool optimize_scratch_bytes = false;
  // Window size in microseconds to consider for scratch bytes optimization.
  int scratch_bytes_window_size_us = 4;
  // If true, the autotuner will return an error if the best config for a
  // certain instruction is not in the cache.
  bool expect_all_instructions_in_cache = false;
  // If not empty, detailed logs will be written to the specified file path
  // as a textproto representation of an `AutotuningLogs` proto message.
  std::string dump_logs_to = "";
  // TODO b/446618161 - Remove this when old triton emitter is
  // deprecated.
  // If true, autotuner will not select cublas configs for fusions. We still try
  // the configs as they can be used to check numerical issues with triton but
  // they are not considered for selection, unless there are no other options.
  bool exclude_cublas_config = false;
  // TODO b/446870267- Remove this option and use default configs rather than
  // the first config.
  // If true, the autotuner selects the first valid config instead of the best
  // performing one. This is to guarantee run-to-run determinism.
  bool select_first_config = false;
  // If true, use hardcoded default backend configs instead of autotuning.
  // Default configs depend on the backend order. Currently the first backend
  // that supports the instruction will be used (see b/446870267).
  // Note: If cache is provided, the cached config will be used instead of the
  // default config.
  bool use_default_config = false;
  // If true, dump the autotuned instructions to the modules's xla_dump_to or
  // to stdout if not set.
  bool dump_hlos = false;
  // Whether to allow or discard configs that ptxas warns will spill registers.
  bool allow_reg_spills = false;

  std::string ToString() const;
};

class Autotuner {
 public:
  static absl::StatusOr<std::unique_ptr<Autotuner>> Create(
      std::vector<std::unique_ptr<CodegenBackend>> codegen_backends,
      std::unique_ptr<Profiler> profiler, AutotuneConfig autotune_config,
      std::unique_ptr<AutotunerCacheInterface> cache,
      tsl::thread::ThreadPool* thread_pool = nullptr);

  // Autotune the given HLO instruction. If a cache is provided, the cached
  // config will be used if the instruction is in the cache. Otherwise, the
  // autotuner will try all supported configs from the registered codegen
  // backends for the given HLO instruction and apply the best one.
  absl::Status Autotune(HloInstruction* instr);

  // Autotune all instructions in the module for which the filter function
  // returns true. The instructions inside fusion computations will be
  // ignored.
  absl::Status Autotune(HloModule* module,
                        const InstructionFilterFn& should_autotune);

  // Same as above, but also takes a sharding KV store which helps to shard
  // the autotuning work across multiple processes.
  // This is used for distributed autotuning.
  absl::Status Autotune(HloModule* module,
                        const InstructionFilterFn& should_autotune,
                        MultiProcessKeyValueStore& sharding_kv_store);

 private:
  using InstructionsByFingerprint =
      absl::flat_hash_map<tsl::Fprint128, std::vector<HloInstruction*>,
                          tsl::Fprint128Hasher>;

  struct Config {
    CodegenBackend* codegen_backend;
    std::unique_ptr<BackendConfig> backend_config;

    std::string ToString() const;
  };
  struct ExecutableCandidate {
    Config config;
    std::unique_ptr<Executable> executable;
  };
  enum class FailureKind {
    kCompilationFailed,
    kExecutionFailed,
    kRedzoneCheckFailed,
    kWrongResults,
  };

  struct Failure {
    FailureKind kind;
    std::string message;

    std::string ToString() const;
    AutotuneResult::FailureResult ToProto() const;
  };
  // The result of profiling a single config for a given instruction. If the
  // profiling failed, as indicated by the failures of kind kCompilationFailed
  // or kExecutionFailed, the duration and scratch_bytes fields won't be set,
  // retaining the default values of 0.
  struct ConfigResult {
    Config config;
    std::optional<Failure> failure;
    absl::Duration duration = absl::ZeroDuration();
    int scratch_bytes = 0;

    std::string ToString(bool verbose = false) const;
    AutotuneResult ToProto() const;
  };

  Autotuner(std::vector<std::unique_ptr<CodegenBackend>> codegen_backends,
            std::unique_ptr<Profiler> profiler, AutotuneConfig autotune_config,
            std::unique_ptr<AutotunerCacheInterface> cache,
            tsl::thread::ThreadPool* thread_pool)
      : codegen_backends_(std::move(codegen_backends)),
        profiler_(std::move(profiler)),
        autotune_config_(autotune_config),
        cache_(std::move(cache)),
        thread_pool_(thread_pool) {}

  InstructionsByFingerprint GetAutotuningCandidates(
      const HloModule* module, const InstructionFilterFn& should_autotune);

  // Gets the default config for the given instruction.
  absl::StatusOr<Config> GetDefaultConfig(const HloInstruction& instr);

  // Gets the config for the given instruction. If instruction is in cache,
  // cached config is returned. If not in cache and use_default_config is
  // true, default config is returned. Otherwise, tunes all supported configs
  // to find the best config, inserts it into cache and returns it.
  absl::StatusOr<Config> GetConfig(HloInstruction* instr);
  // Gets the best config for the given instruction by compiling and profiling
  // all supported configs.
  absl::StatusOr<Config> TuneBestConfig(HloInstruction* instr);

  // TODO: b/407494653 - Directly use cache api when the configs are unified.
  // Translates from Autotuner::Config to AutotunerCacheInterface::Config and
  // the other way around.
  std::optional<Autotuner::Config> LookUp(const HloInstruction* instr);
  void Insert(const HloInstruction* instr, Autotuner::Config& config);

  absl::StatusOr<std::vector<Config>> GetSupportedConfigs(
      HloInstruction* instr);
  std::vector<absl::StatusOr<std::unique_ptr<Executable>>> CompileAll(
      HloInstruction* instr, std::vector<Config>& configs);
  absl::StatusOr<std::vector<ConfigResult>> ProfileAll(
      std::vector<ExecutableCandidate>& candidates);
  absl::StatusOr<ConfigResult> PickBestConfig(
      std::vector<ConfigResult>& results);

  std::optional<ScopedShapedBuffer> GetReferenceOutput(
      std::vector<ExecutableCandidate>& candidates,
      InputBuffers& input_buffers);

  std::optional<Failure> CheckBuffers(InputBuffers& input_buffers,
                                      ScopedShapedBuffer& output,
                                      ScopedShapedBuffer& reference);
  absl::Status IsValidExecutable(
      const absl::StatusOr<std::unique_ptr<Executable>>& executable) const;

  void LogConfigResults(const HloInstruction& instr,
                        const std::vector<ConfigResult>& results);
  absl::Status DumpLogsToFile();
  // Dumps HLO before and after applying the config.
  absl::Status DumpHlo(HloInstruction* instr, const Config& config);

  std::vector<std::unique_ptr<CodegenBackend>> codegen_backends_;
  std::unique_ptr<Profiler> profiler_;
  AutotuneConfig autotune_config_;
  std::unique_ptr<AutotunerCacheInterface> cache_;
  tsl::thread::ThreadPool* thread_pool_;
  AutotuningLogs logs_;
  int dump_counter_ = 0;
};
}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_AUTOTUNER_H_
