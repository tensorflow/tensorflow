/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/convert/profile_time_breakdown.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/math_utils.h"

namespace tensorflow {
namespace profiler {

void ProfileTimeBreakdown::SetCategoryTimePs(absl::string_view category,
                                             uint64_t time_ps) {
  time_ps_by_category_.insert_or_assign(category, time_ps);
}

uint64_t ProfileTimeBreakdown::PopCategoryTimePs(absl::string_view category) {
  uint64_t time_ps = 0;
  auto iter = time_ps_by_category_.find(category);
  if (iter != time_ps_by_category_.end()) {
    time_ps = iter->second;
    time_ps_by_category_.erase(iter);
  }
  return time_ps;
}

void ProfileTimeBreakdown::BreakdownSparseCoreV0Infeed() {
  // Infeed from SparseCoreV0 and outfeed to SparseCoreV0 are mostly identical
  // in compute since they do the same transformation. We can subtract out the
  // outfeed time from the infeed time to know how much time the TensorCore
  // actually spent waiting on SparseCoreV0.
  uint64_t bc_infeed_ps =
      PopCategoryTimePs(tsl::profiler::kHloSparseCoreV0Infeed);
  if (bc_infeed_ps == 0) return;
  uint64_t bc_outfeed_ps =
      CategoryTimePs(tsl::profiler::kHloSparseCoreV0Outfeed);

  uint64_t bc_infeed_transform_ps = std::min(bc_infeed_ps, bc_outfeed_ps);
  uint64_t bc_infeed_wait_ps = bc_infeed_ps - bc_infeed_transform_ps;

  SetCategoryTimePs(tsl::profiler::kHloSparseCoreV0InfeedWait,
                    bc_infeed_wait_ps);
  SetCategoryTimePs(tsl::profiler::kHloSparseCoreV0InfeedTransform,
                    bc_infeed_transform_ps);
}

std::string ProfileTimeBreakdown::DebugString() const {
  std::string str;
  for (const auto& [category, time_ps] : time_ps_by_category_) {
    absl::StrAppend(&str, category, ": ", tsl::profiler::PicoToUni(time_ps),
                    "\n");
  }
  absl::StrAppend(
      &str, "total_time: ", tsl::profiler::PicoToUni(total_time_ps_), "\n");
  absl::StrAppend(
      &str, "profile_time: ", tsl::profiler::PicoToUni(profile_time_ps_), "\n");
  return str;
}

}  // namespace profiler
}  // namespace tensorflow
