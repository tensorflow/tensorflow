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

#include "xla/backends/autotuner/config_selector.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "xla/backends/autotuner/config_runner.h"

namespace xla {

absl::StatusOr<ConfigRunner::ConfigProfile> PickBestConfig(
    std::vector<ConfigRunner::ConfigProfile>& results,
    int scratch_bytes_window_size_us) {
  absl::Duration min_duration = absl::InfiniteDuration();
  ConfigRunner::ConfigProfile* best_result = nullptr;
  std::vector<std::string> failures;
  for (ConfigRunner::ConfigProfile& result : results) {
    if (result.failure.has_value()) {
      failures.push_back(result.failure->ToString());
    } else if (result.duration < min_duration) {
      min_duration = result.duration;
      best_result = &result;
    }
  }

  if (best_result == nullptr) {
    std::string message = "All configs failed during profiling.";
    if (!failures.empty()) {
      absl::StrAppend(&message, "\nFailures (", failures.size(), "):\n",
                      absl::StrJoin(failures, "\n"));
    }
    return absl::NotFoundError(message);
  }

  const ConfigRunner::ConfigProfile* fastest_result = best_result;
  int64_t min_scratch_bytes = std::numeric_limits<int64_t>::max();
  absl::Duration duration_limit =
      min_duration + absl::Microseconds(scratch_bytes_window_size_us);
  absl::Duration min_duration_with_optimized_scratch_bytes =
      absl::InfiniteDuration();
  for (ConfigRunner::ConfigProfile& result : results) {
    if (!result.failure.has_value() && result.duration <= duration_limit) {
      bool current_result_is_better =
          result.scratch_bytes < min_scratch_bytes ||
          (result.scratch_bytes == min_scratch_bytes &&
           result.duration < min_duration_with_optimized_scratch_bytes);
      if (current_result_is_better) {
        min_scratch_bytes = result.scratch_bytes;
        min_duration_with_optimized_scratch_bytes = result.duration;
        best_result = &result;
      }
    }
  }
  if (best_result != fastest_result) {
    VLOG(2) << "Autotuner picked a slower config to save scratch memory. "
            << "Fastest config: " << fastest_result->ToString() << ". "
            << "Selected config: " << best_result->ToString() << ". "
            << "Tolerance: " << scratch_bytes_window_size_us << "us.";
  }

  return std::move(*best_result);
}

}  // namespace xla
