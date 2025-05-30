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

#include "xla/backends/autotuner/autotuner.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/executor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

absl::Status Autotuner::Autotune(HloInstruction* instr) {
  if (codegen_backends_.empty()) {
    return absl::InvalidArgumentError("No codegen backends registered.");
  }
  VLOG(1) << "Autotuning HLO: " << instr->ToString();
  std::unique_ptr<BackendConfig> best_config;
  CodegenBackend* best_codegen_backend = nullptr;
  absl::Duration min_duration = absl::InfiniteDuration();
  for (auto& codegen_backend : codegen_backends_) {
    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<BackendConfig>> configs,
        codegen_backend->GetSupportedConfigs(*instr, stream_executor_));
    VLOG(1) << "Got " << configs.size()
            << " configs from codegen backend: " << codegen_backend->name();
    for (auto& config : configs) {
      VLOG(2) << "Trying to compile config: " << config->DebugString();
      auto executable_or_status = codegen_backend->Compile(*instr, *config);
      auto status = executable_or_status.status();
      // TODO b/407495547: Change it to tolerate only specific compilation
      // errors, as opposed to all errors.
      if (!status.ok() && autotune_config_.skip_failing_configs) {
        VLOG(1) << "Failed to compile: " << status;
        continue;
      }
      TF_ASSIGN_OR_RETURN(
          ProfileResult result,
          executor_->Profile(std::move(executable_or_status.value())));
      if (result.duration < min_duration) {
        best_config = std::move(config);
        best_codegen_backend = codegen_backend.get();
      }
      min_duration = std::min(min_duration, result.duration);
    }
  }
  if (!best_config || !best_codegen_backend) {
    return absl::InternalError("No config found!");
  }
  VLOG(1) << "Best config: " << best_config->DebugString()
          << " from codegen backend: " << best_codegen_backend->name()
          << " with duration: " << min_duration;
  return best_codegen_backend->ApplyConfig(*instr, *best_config);
}

}  // namespace xla
