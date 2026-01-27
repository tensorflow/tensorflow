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

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/debug/unstable_reduction_finder.h"
#include "xla/xla.pb.h"

namespace xla {

// Returns a string representation of the unstable reduction ops.
// The string representation has a deterministic order.
std::string UniqueReductionOpsAsString(
    const std::vector<const HloInstruction*>& unstable_reductions) {
  // We want to have a deterministic order of the unstable reduction ops, so we
  // use a set instead of a hash set.
  std::set<std::string> unstable_reduction_strings_set;
  std::vector<std::string> no_metadata_reduction_names;
  for (const HloInstruction* reduction : unstable_reductions) {
    const auto& metadata = reduction->metadata();
    const absl::string_view source_file = metadata.source_file();
    const int64_t source_line = metadata.source_line();
    const absl::string_view op_name = metadata.op_name();
    std::string unstable_reduction_string;
    if (metadata.op_name().empty()) {
      no_metadata_reduction_names.push_back(std::string(reduction->name()));
    } else {
      unstable_reduction_string =
          absl::StrCat(source_file, ":", source_line, ": ", op_name);
      unstable_reduction_strings_set.insert(unstable_reduction_string);
    }
  }

  std::string result;
  if (!unstable_reduction_strings_set.empty()) {
    result =
        absl::StrCat("List of unique reduction ops:\n",
                     absl::StrJoin(unstable_reduction_strings_set.begin(),
                                   unstable_reduction_strings_set.end(), "\n"));
  }
  if (!no_metadata_reduction_names.empty()) {
    absl::StrAppend(&result, "\n", no_metadata_reduction_names.size(),
                    " op names without metadata: ",
                    absl::StrJoin(no_metadata_reduction_names, ", "));
  }
  return result;
}

absl::StatusOr<bool> UnstableReductionDetector::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (module->config().debug_options().xla_detect_unstable_reductions() ==
      DebugOptions::DETECTION_MODE_NONE) {
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
      DebugOptions::DETECTION_MODE_FAIL) {
    std::string reduction_ops_string =
        UniqueReductionOpsAsString(unstable_reductions);
    return absl::FailedPreconditionError(absl::StrFormat(
        "%d unstable reductions found in module '%s'. %s",
        unstable_reductions.size(), module->name(), reduction_ops_string));
  }
  return false;
}

}  // namespace xla
