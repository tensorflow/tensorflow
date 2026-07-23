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

#ifndef XLA_BACKENDS_AUTOTUNER_CODEGEN_ORCHESTRATOR_H_
#define XLA_BACKENDS_AUTOTUNER_CODEGEN_ORCHESTRATOR_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/executable.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {

// Orchestrates the config retrieval and compilation across all the registered
// codegen backends.
class CodegenOrchestrator {
 public:
  struct Options {
    // Whether to allow or discard configs that ptxas warns will spill
    // registers.
    std::function<bool(const HloInstruction&)> allow_reg_spills_fn =
        [](const HloInstruction&) { return false; };
    // TODO(b/519059655): Generalize and move to tuner.
    // If true, do not allow compilation of cublas or rocblas configs.
    bool exclude_cublas_config = false;
  };

  // TODO(b/444398084): Unify Cache::Config and CodegenOrchestrator::Config
  struct Config {
    CodegenBackend* codegen_backend;
    std::unique_ptr<BackendConfig> backend_config;

    std::string ToString() const;
  };

  struct MaybeExecutableCandidate {
    Config config;
    absl::StatusOr<std::unique_ptr<Executable>> executable;
  };

  static absl::StatusOr<std::unique_ptr<CodegenOrchestrator>> Create(
      std::vector<std::unique_ptr<CodegenBackend>> codegen_backends,
      Options options, tsl::thread::ThreadPool* thread_pool = nullptr);

  // Returns all supported configs across all registered backends.
  absl::StatusOr<std::vector<Config>> GetSupportedConfigs(
      const HloInstruction& instr) const;

  // Returns the default config from the first backend that supports it.
  absl::StatusOr<Config> GetDefaultConfig(const HloInstruction& instr) const;

  // Compiles all configs in parallel (if thread pool is present) and returns
  // their executable status.
  tsl::Future<std::vector<MaybeExecutableCandidate>> CompileAll(
      const HloInstruction& instr, std::vector<Config> configs) const;

  // Applies the configuration to the instruction using the appropriate backend.
  absl::Status ApplyConfig(HloInstruction& instr, const Config& config) const;

  // Compiles a single config.
  absl::StatusOr<std::unique_ptr<Executable>> Compile(
      const HloInstruction& instr, const Config& config) const;

  // TODO(b/444398084): Unify Cache::Config and CodegenOrchestrator::Config and
  // remove this api.
  const std::vector<std::unique_ptr<CodegenBackend>>& codegen_backends() const {
    return codegen_backends_;
  }

 private:
  CodegenOrchestrator(
      std::vector<std::unique_ptr<CodegenBackend>> codegen_backends,
      Options options, tsl::thread::ThreadPool* thread_pool)
      : codegen_backends_(std::move(codegen_backends)),
        options_(std::move(options)),
        thread_pool_(thread_pool) {}

  absl::Status IsValidExecutable(
      const absl::StatusOr<std::unique_ptr<Executable>>& executable,
      const HloInstruction& instr) const;

  std::vector<std::unique_ptr<CodegenBackend>> codegen_backends_;
  Options options_;
  tsl::thread::ThreadPool* thread_pool_;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_CODEGEN_ORCHESTRATOR_H_
