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

#include "xla/backends/gpu/collectives/gpu_clique_key.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {

std::string HumanReadableDeviceGroups(
    absl::Span<const std::vector<GlobalDeviceId>> device_groups,
    absl::string_view separator, size_t first, size_t last) {
  auto fmt = [&](std::string* out, absl::Span<const GlobalDeviceId> group) {
    absl::StrAppendFormat(out, "[%s]",
                          HumanReadableDevices(group, separator, 3, 1));
  };

  if (device_groups.size() > first + last) {
    return absl::StrCat(
        absl::StrJoin(device_groups.first(first), separator, fmt), "...",
        absl::StrJoin(device_groups.last(last), separator, fmt));
  }

  return absl::StrJoin(device_groups, separator, fmt);
}

GpuCliqueKey::GpuCliqueKey(std::vector<GlobalDeviceId> devices,
                           int64_t num_local_participants,
                           CommunicationId communication_id,
                           std::vector<IncarnationId> incarnations)
    : CliqueKey(std::move(devices)),
      num_local_participants_(num_local_participants),
      communication_id_(communication_id),
      incarnations_(std::move(incarnations)) {}

bool GpuCliqueKey::IsSubsetOf(const CliqueKey& other) const {
  auto* other_gpu = tsl::down_cast<const GpuCliqueKey*>(&other);
  if (other_gpu == nullptr) {
    return false;
  }

  return communication_id() == other_gpu->communication_id() &&
         absl::c_all_of(devices(),
                        [&](GlobalDeviceId id) {
                          return absl::c_linear_search(other_gpu->devices(),
                                                       id);
                        }) &&
         absl::c_all_of(incarnations_, [&](IncarnationId id) {
           return absl::c_linear_search(other_gpu->incarnations_, id);
         });
}

std::vector<GlobalDeviceId> GpuCliqueKey::GetRootDevices(int64_t nroots) const {
  int64_t nranks = devices().size();
  CHECK_LE(nroots, nranks) << "Can't select more root devices than available";

  // If nroots evenly divides nranks, then every root is assigned to
  // ranks_per_root devices. If nroots doesn't divide nranks, then we increase
  // the size of the first ranks_rem roots by 1.
  int64_t ranks_per_root = nranks / nroots;
  int64_t ranks_rem = nranks % nroots;

  std::vector<GlobalDeviceId> roots;
  roots.reserve(nroots);
  for (int64_t i = 0; i < nroots; ++i) {
    if (i < ranks_rem) {
      roots.push_back(devices()[i * (ranks_per_root + 1)]);
    } else {
      roots.push_back(devices()[ranks_rem * (ranks_per_root + 1) +
                                (i - ranks_rem) * ranks_per_root]);
    }
  }
  return roots;
}

std::string GpuCliqueKey::ToString() const {
  return absl::StrFormat(
      "devices=%d:[%s]; local_participants=%lld; communication_id=%v; "
      "incarnations=[%s]",
      devices().size(), HumanReadableDevices(devices()),
      num_local_participants_, communication_id_,
      absl::StrJoin(incarnations_, ", ",
                    [](std::string* out, IncarnationId id) {
                      absl::StrAppend(out, id.value());
                    }));
}

void GpuCliqueKey::HashValue(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices(), incarnations_);
}

bool operator==(const GpuCliqueKey& a, const GpuCliqueKey& b) {
  return a.devices() == b.devices() &&
         a.num_local_participants_ == b.num_local_participants_ &&
         a.incarnations_ == b.incarnations_;
}

bool operator!=(const GpuCliqueKey& a, const GpuCliqueKey& b) {
  return !(a == b);
}

// Constructs a tuple from the clique key for comparison purposes.
static auto CmpKey(const GpuCliqueKey& key) {
  return std::make_tuple(key.devices().size(), key.devices(),
                         key.num_local_participants(), key.incarnations());
}

bool operator<(const GpuCliqueKey& a, const GpuCliqueKey& b) {
  return CmpKey(a) < CmpKey(b);
}

bool operator>(const GpuCliqueKey& a, const GpuCliqueKey& b) {
  return CmpKey(a) > CmpKey(b);
}

}  // namespace xla::gpu
