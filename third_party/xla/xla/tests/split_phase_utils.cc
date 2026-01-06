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

#include "xla/tests/split_phase_utils.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/evaluator/caching_hlo_evaluator.h"
#include "xla/hlo/evaluator/hlo_evaluator_interface.h"
#include "xla/pjrt/interpreter/interpreter_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo_runner_pjrt.h"

namespace xla {
namespace {
enum class SplitPhaseMode : uint8_t {
  // Split-phase compilation and execution is disabled. Both are performed in
  // the same runner.
  kDisabled,
  // Split-phase compilation is enabled. The runner will run in compile-only
  // mode and persist the executable to disk.
  kCompile,
  // Split-phase execution is enabled. The runner will run in execute-only mode
  // and load the executable from disk, falling back to performing compilation
  // if the executable is not found.
  kExecute
};

constexpr absl::string_view kModeEnvVar =
    "XLA_TEST_HLO_RUNNER_SPLIT_PHASE_MODE";
constexpr absl::string_view kDirEnvVar = "XLA_TEST_HLO_RUNNER_SPLIT_PHASE_DIR";

SplitPhaseMode GetSplitPhaseMode() {
  // Mode is presumed to stay the same for the lifetime of the program.
  static const SplitPhaseMode kMode = []() {
    const std::string name{kModeEnvVar};
    const char* val_buffer = std::getenv(name.c_str());
    if (val_buffer == nullptr) {
      return SplitPhaseMode::kDisabled;
    }

    const absl::string_view val(val_buffer);
    if (val == "disabled") {
      return SplitPhaseMode::kDisabled;
    }
    if (val == "compile") {
      return SplitPhaseMode::kCompile;
    }
    if (val == "execute") {
      return SplitPhaseMode::kExecute;
    }
    LOG(WARNING) << "Unknown value for " << kModeEnvVar << ": " << val
                 << ". GetSplitPhaseMode() will return kDisabled.";
    return SplitPhaseMode::kDisabled;
  }();
  return kMode;
}

std::optional<absl::string_view> GetSplitPhaseDir() {
  // Dir is presumed to stay the same for the lifetime of the program.
  static const absl::NoDestructor<std::optional<std::string>> kDir(
      []() -> std::optional<std::string> {
        const std::string name{kDirEnvVar};
        const char* val = std::getenv(name.c_str());
        if (val == nullptr) {
          return std::nullopt;
        }
        return std::string{val};
      }());
  if (!kDir->has_value()) {
    return std::nullopt;
  }
  return **kDir;
}
}  // namespace

std::unique_ptr<HloRunnerPjRt> MakeHloRunnerPjRtSplitPhaseAware(
    std::unique_ptr<PjRtClient> client) {
  const SplitPhaseMode mode = GetSplitPhaseMode();
  absl::string_view artifact_dir;
  if (mode != SplitPhaseMode::kDisabled) {
    std::optional<absl::string_view> split_phase_dir = GetSplitPhaseDir();
    if (!split_phase_dir.has_value()) {
      return nullptr;
    }
    artifact_dir = *std::move(split_phase_dir);
  }

  switch (mode) {
    case SplitPhaseMode::kDisabled:
      return std::make_unique<HloRunnerPjRt>(std::move(client));
    case SplitPhaseMode::kCompile:
      return std::make_unique<CompilePhaseHloRunnerPjRt>(
          std::move(client), std::move(artifact_dir));
    case SplitPhaseMode::kExecute:
      return std::make_unique<ExecutePhaseHloRunnerPjRt>(
          std::move(client), std::move(artifact_dir));
  }
  return nullptr;  // Should not reach here.
}

std::unique_ptr<InterpreterClient> MakeInterpreterClientSplitPhaseAware(
    absl::AnyInvocable<std::unique_ptr<HloEvaluatorInterface>() const>
        hlo_evaluator_factory) {
  const SplitPhaseMode mode = GetSplitPhaseMode();
  absl::string_view artifact_dir;
  if (mode != SplitPhaseMode::kDisabled) {
    std::optional<absl::string_view> split_phase_dir = GetSplitPhaseDir();
    if (!split_phase_dir.has_value()) {
      return nullptr;
    }
    artifact_dir = *std::move(split_phase_dir);
  }

  return std::make_unique<InterpreterClient>(
      [factory = std::move(hlo_evaluator_factory),
       artifact_dir = std::string{artifact_dir},
       mode]() -> std::unique_ptr<HloEvaluatorInterface> {
        switch (mode) {
          case SplitPhaseMode::kDisabled:
            return factory();
          case SplitPhaseMode::kCompile:
            return std::make_unique<CachingHloEvaluator>(
                factory(), std::move(artifact_dir),
                CachingHloEvaluator::kWrite);
          case SplitPhaseMode::kExecute:
            return std::make_unique<CachingHloEvaluator>(
                factory(), std::move(artifact_dir),
                CachingHloEvaluator::kReadAndEvaluateIfCacheMiss);
        }
        return nullptr;  // Should not reach here.
      });
}

// Execution errors are swallowed if and only if the split phase mode is set to
// kCompile.
bool HasPjRtSplitPhaseAwareSwallowExecutionErrors() {
  return GetSplitPhaseMode() == SplitPhaseMode::kCompile;
}

}  // namespace xla
