/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/ir/collective_device_list.h"

namespace xla {

CollectiveDeviceList::CollectiveDeviceList(
    absl::Span<const std::vector<int64_t>> replica_groups) {
  replica_groups_.reserve(replica_groups.size());
  for (auto g : replica_groups) {
    auto& group = replica_groups_.emplace_back();
    *group.mutable_replica_ids() = {g.begin(), g.end()};
  }
}

}  // namespace xla
