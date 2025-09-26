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

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/executable.h"
#include "xla/service/shaped_buffer.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/fingerprint.h"

using InstructionFilterFn = absl::FunctionRef<bool(const xla::HloInstruction&)>;

namespace xla {

struct AutotuneConfig {
  // Whether to skip configs that failed to compile.
  bool skip_failing_configs = true;
  // Whether to check the correctness of the output buffers and OOM reads on
  // Input Buffers.
  bool check_buffers = false;
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
};

class Autotuner {
 public:
  static absl::StatusOr<std::unique_ptr<Autotuner>> Create(
      std::vector<std::unique_ptr<CodegenBackend>> codegen_backends,
      std::unique_ptr<Profiler> profiler, AutotuneConfig autotune_config,
      std::unique_ptr<AutotunerCacheInterface> cache,
      tsl::thread::ThreadPool* thread_pool = nullptr);

  // Try all supported configs from the registered codegen backends for the
  // given HLO instruction and apply the best one.
  absl::Status Autotune(HloInstruction* instr);

  // Autotune all instructions in the module for which the filter function
  // returns true. The instructions inside fusion computations will be
  // ignored.
  absl::Status Autotune(HloModule* module,
                        const InstructionFilterFn& should_autotune);

 private:
  using InstructionsByFingerprint =
      absl::flat_hash_map<tsl::Fprint128, std::vector<HloInstruction*>,
                          tsl::Fprint128Hasher>;

  struct Config {
    CodegenBackend* codegen_backend;
    std::unique_ptr<BackendConfig> backend_config;
  };
  struct ExecutableCandidate {
    Config config;
    std::unique_ptr<Executable> executable;
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

  absl::StatusOr<Config> GetBestConfig(HloInstruction* instr);

  absl::StatusOr<std::vector<Config>> GetSupportedConfigs(
      HloInstruction* instr);
  std::vector<absl::StatusOr<std::unique_ptr<Executable>>> CompileAll(
      HloInstruction* instr, std::vector<Config>& configs);

  absl::StatusOr<Config> ProfileAndPickBest(
      HloInstruction* instr, std::vector<ExecutableCandidate>& candidates);

  absl::StatusOr<ScopedShapedBuffer> GetReferenceOutput(
      std::vector<ExecutableCandidate>& candidates,
      InputBuffers& input_buffers);

  absl::Status CheckBuffers(InputBuffers& input_buffers,
                            ScopedShapedBuffer& output,
                            ScopedShapedBuffer& reference);

  std::vector<std::unique_ptr<CodegenBackend>> codegen_backends_;
  std::unique_ptr<Profiler> profiler_;
  AutotuneConfig autotune_config_;
  std::unique_ptr<AutotunerCacheInterface> cache_;
  tsl::thread::ThreadPool* thread_pool_;
};
}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_AUTOTUNER_H_
