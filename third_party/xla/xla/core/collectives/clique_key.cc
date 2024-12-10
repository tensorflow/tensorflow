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

#include "xla/core/collectives/clique_key.h"

#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/global_device_id.h"

namespace xla {

CliqueKey::CliqueKey(std::vector<GlobalDeviceId> devices)
    : devices_(std::move(devices)) {}

absl::Span<const GlobalDeviceId> CliqueKey::devices() const { return devices_; }

std::optional<RankId> CliqueKey::rank(GlobalDeviceId id) const {
  if (auto it = absl::c_find(devices_, id); it != devices_.end()) {
    return RankId(it - devices_.begin());
  }
  return std::nullopt;
}
}  // namespace xla
