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

#include "xla/service/gpu/early_exit_compilation_result.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/service/executable.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

absl::StatusOr<std::string> EarlyExitCompilationResult::SerializeAsString()
    const {
  return Unavailable(
      "SerializeAsString() is not supported by EarlyExitCompilationResult.");
}

absl::StatusOr<std::unique_ptr<Executable>>
EarlyExitCompilationResult::LoadExecutable(
    se::Platform::Id platform_id,
    const se::DeviceDescription& device_description,
    const DebugOptions& debug_options) && {
  return Unavailable(
      "LoadExecutable() is not supported by EarlyExitCompilationResult.");
}

absl::StatusOr<CompiledMemoryStats>
EarlyExitCompilationResult::GetCompiledMemoryStats() const {
  return absl::UnavailableError(
      "GetCompiledMemoryStats() is not supported by "
      "EarlyExitCompilationResult.");
}

}  // namespace gpu
}  // namespace xla
