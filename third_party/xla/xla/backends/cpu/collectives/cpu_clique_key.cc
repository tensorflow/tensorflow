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

#include "xla/backends/cpu/collectives/cpu_clique_key.h"

#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/runtime/device_id.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {

bool CpuCliqueKey::IsSubsetOf(const CliqueKey& other) const {
  auto* other_cpu = absl::down_cast<const CpuCliqueKey*>(&other);
  if (other_cpu == nullptr) {
    return false;
  }

  return absl::c_all_of(devices(), [&](GlobalDeviceId id) {
    return absl::c_linear_search(other_cpu->devices(), id);
  });
}

std::string CpuCliqueKey::ToString() const {
  return absl::StrFormat("devices=[%s]", absl::StrJoin(devices(), ", "));
}

void CpuCliqueKey::HashValue(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices());
}

bool operator==(const CpuCliqueKey& a, const CpuCliqueKey& b) {
  return a.devices() == b.devices();
}

bool operator<(const CpuCliqueKey& a, const CpuCliqueKey& b) {
  return a.devices() < b.devices();
}

bool operator>(const CpuCliqueKey& a, const CpuCliqueKey& b) {
  return a.devices() > b.devices();
}

}  // namespace xla::cpu
