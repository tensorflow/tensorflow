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

#ifndef XLA_BACKENDS_AUTOTUNER_CONFIG_RUNNER_H_
#define XLA_BACKENDS_AUTOTUNER_CONFIG_RUNNER_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/executable.h"
#include "xla/service/shaped_buffer.h"

namespace xla {

// ConfigRunner is responsible for running and profiling a given set of
// executable candidates on device.
class ConfigRunner {
 public:
  struct CorrectnessCheckOptions {
    // Whether to check the correctness of the output buffers and OOM reads on
    // Input Buffers.
    bool enable_correctness_check = true;
    // Relative tolerance for correctness check.
    float relative_tolerance = 1e-6;
    // Whether to crash the process on check failure.
    bool crash_on_failure = false;
  };

  // TODO(b/519057668): Move these types to a shared header file.
  using Config = CodegenOrchestrator::Config;
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

  // TODO(b/519057668): Move this struct and ToProto to a header file.
  struct ConfigProfile {
    Config config;
    std::optional<Failure> failure;
    absl::Duration duration = absl::ZeroDuration();
    int scratch_bytes = 0;
    int cluster_index = -1;

    std::string ToString(bool verbose = false) const;
    AutotuneResult ToProto() const;
  };

  static absl::StatusOr<std::unique_ptr<ConfigRunner>> Create(
      std::unique_ptr<Profiler> profiler, CorrectnessCheckOptions options);

  absl::StatusOr<std::vector<ConfigProfile>> ProfileAll(
      std::vector<ExecutableCandidate> candidates,
      const HloInstruction* instr = nullptr);

 private:
  ConfigRunner(std::unique_ptr<Profiler> profiler,
               CorrectnessCheckOptions options)
      : profiler_(std::move(profiler)), options_(options) {}

  struct OutputCluster {
    ScopedShapedBuffer representative;
    int count = 0;
    bool has_trusted_member = false;
  };

  ConfigProfile ProfileCandidate(ExecutableCandidate candidate,
                                 InputBuffers& input_buffers,
                                 std::vector<OutputCluster>& clusters,
                                 bool is_trusted_config, bool allow_new_cluster)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(profiler_m_);

  int AssignToOutputCluster(std::vector<OutputCluster>& clusters,
                            ScopedShapedBuffer& output, bool is_trusted_config,
                            bool allow_new_cluster)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(profiler_m_);

  void DemoteNonWinningClusterConfigs(
      std::vector<ConfigProfile>& results,
      const std::vector<OutputCluster>& clusters);

 public:
  std::unique_ptr<Profiler> profiler_ ABSL_GUARDED_BY(profiler_m_);
  CorrectnessCheckOptions options_;
  absl::Mutex profiler_m_;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_CONFIG_RUNNER_H_
