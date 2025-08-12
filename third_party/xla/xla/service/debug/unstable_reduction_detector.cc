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

#include "xla/service/debug/unstable_reduction_detector.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/debug/unstable_reduction_finder.h"

namespace xla {

absl::StatusOr<bool> UnstableReductionDetector::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (module->config().debug_options().xla_detect_unstable_reductions() ==
      DebugOptions::UNSTABLE_REDUCTION_DETECTION_MODE_NONE) {
    return false;
  }
  std::vector<const HloInstruction*> unstable_reductions =
      FindUnstableReductionInstructions(module);

  if (unstable_reductions.empty()) {
    return false;
  }

  LOG(WARNING) << unstable_reductions.size()
               << " unstable reductions found in module '" << module->name()
               << "'";
  for (const HloInstruction* reduction : unstable_reductions) {
    LOG(WARNING) << "Unstable reduction: " << reduction->ToString();
  }
  if (module->config().debug_options().xla_detect_unstable_reductions() ==
      DebugOptions::UNSTABLE_REDUCTION_DETECTION_MODE_FAIL) {
    return absl::FailedPreconditionError(
        absl::StrFormat("%d unstable reductions found in module '%s'",
                        unstable_reductions.size(), module->name()));
  }
  return false;
}

}  // namespace xla
