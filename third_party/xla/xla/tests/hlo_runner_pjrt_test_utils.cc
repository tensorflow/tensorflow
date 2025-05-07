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

#include "xla/tests/hlo_runner_pjrt_test_utils.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"

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

bool AbslParseFlag(absl::string_view text, SplitPhaseMode* mode,
                   std::string* error) {
  if (text == "disabled") {
    *mode = SplitPhaseMode::kDisabled;
    return true;
  }
  if (text == "compile") {
    *mode = SplitPhaseMode::kCompile;
    return true;
  }
  if (text == "execute") {
    *mode = SplitPhaseMode::kExecute;
    return true;
  }
  *error = "unknown value for SplitPhaseMode enumeration";
  return false;
}

std::string AbslUnparseFlag(SplitPhaseMode mode) {
  switch (mode) {
    case SplitPhaseMode::kDisabled:
      return "disabled";
    case SplitPhaseMode::kCompile:
      return "compile";
    case SplitPhaseMode::kExecute:
      return "execute";
  }
  return "should not reach here";
}

ABSL_FLAG(SplitPhaseMode, xla_pjrt_split_phase_mode, SplitPhaseMode::kDisabled,
          "If set to anything other than \"disabled\", split phase compilation "
          "mode is enabled. Specify \"compile\" to use compile-only mode, in "
          "which executables are compiled and then persisted to disk at the "
          "path specified by --xla_pjrt_split_phase_dir. Specify \"execute\" "
          "to use execute-only mode, in which executables are loaded from disk "
          "and then executed.");

ABSL_FLAG(
    std::optional<std::string>, xla_pjrt_split_phase_dir, std::nullopt,
    "The directory where intermediate results for split-phase compilation are "
    "persisted. Must be specified if --xla_pjrt_split_phase_mode is set.");

namespace xla {

std::unique_ptr<HloRunnerPjRt> MakeHloRunnerPjRtSplitPhaseAware(
    std::unique_ptr<PjRtClient> client,
    HloRunnerInterface::DeviceShapeRepresentationFn
        device_shape_representation_fn,
    HloRunnerInterface::DeviceShapeSizeFn device_shape_size_fn) {
  const SplitPhaseMode mode = absl::GetFlag(FLAGS_xla_pjrt_split_phase_mode);
  std::string artifact_dir;
  if (mode != SplitPhaseMode::kDisabled) {
    std::optional<std::string> split_phase_dir =
        absl::GetFlag(FLAGS_xla_pjrt_split_phase_dir);
    if (!split_phase_dir.has_value()) {
      return nullptr;
    }
    artifact_dir = *std::move(split_phase_dir);
  }

  switch (absl::GetFlag(FLAGS_xla_pjrt_split_phase_mode)) {
    case SplitPhaseMode::kDisabled:
      return std::make_unique<HloRunnerPjRt>(
          std::move(client), std::move(device_shape_representation_fn),
          std::move(device_shape_size_fn));
    case SplitPhaseMode::kCompile:
      return std::make_unique<CompilePhaseHloRunnerPjRt>(
          std::move(client), std::move(device_shape_representation_fn),
          std::move(device_shape_size_fn), std::move(artifact_dir));
    case SplitPhaseMode::kExecute:
      return std::make_unique<ExecutePhaseHloRunnerPjRt>(
          std::move(client), std::move(device_shape_representation_fn),
          std::move(device_shape_size_fn), std::move(artifact_dir));
  }
  return nullptr;  // Should not reach here.
}

}  // namespace xla
