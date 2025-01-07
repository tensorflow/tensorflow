/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/python/aggregate_profile.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/python/xplane_to_profile_instructions.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla {

void AggregateProfiledInstructionsProto(
    absl::Span<const tensorflow::profiler::ProfiledInstructionsProto> profiles,
    int percentile,
    tensorflow::profiler::ProfiledInstructionsProto *result_profile) {
  if (percentile < 0 || percentile > 100) return;

  absl::flat_hash_map<std::string, HloLatencyInfo> hlo_latency_info;
  // Store costs information from each profile to the hash map.
  for (const auto &profile : profiles) {
    for (const auto &cost : profile.costs()) {
      hlo_latency_info[cost.name()].durations.emplace_back(cost.cost_us());
    }
  }
  for (const auto &iter : hlo_latency_info) {
    auto *cost = result_profile->add_costs();
    std::vector<double> durations = iter.second.durations;
    int index = 0;
    if (durations.size() > 1) {
      std::sort(durations.begin(), durations.end());
      index = percentile / 100.0 * (durations.size() - 1);
    }

    cost->set_cost_us(durations[index]);
    cost->set_name(iter.first);
  }
}

}  // namespace xla
