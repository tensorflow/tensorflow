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

bool IsP2PStreamKind(AsyncStreamKind stream_kind) {
  switch (stream_kind) {
    case AsyncStreamKind::ASYNC_STREAM_KIND_P2P0:
    case AsyncStreamKind::ASYNC_STREAM_KIND_P2P1:
      return true;
    default:
      return false;
  }
}

CollectiveStreamId GetCollectiveStreamId(bool is_async,
                                         CollectiveStreamId stream_id,
                                         AsyncStreamKind stream_kind) {
  if (!is_async) {
    return CollectiveStreamId(0);
  }
  // TODO: Remove this fallback once AsyncStreamId is used everywhere.
  if (stream_id.value() == 0) {
    return CollectiveStreamId(static_cast<int64_t>(stream_kind) + 1);
  }
  return stream_id;
}

GpuCliqueKey::GpuCliqueKey(
    std::vector<GlobalDeviceId> devices, int64_t num_local_participants,
    bool is_p2p, std::vector<std::vector<GlobalDeviceId>> participant_groups,
    std::vector<IncarnationId> incarnations)
    : CliqueKey(std::move(devices)),
      num_local_participants_(num_local_participants),
      is_p2p_(is_p2p),
      participant_groups_(std::move(participant_groups)),
      incarnations_(std::move(incarnations)) {
  for (std::vector<GlobalDeviceId>& group : participant_groups_) {
    absl::c_sort(group);
  }
  // Compare the groups by their first element.
  auto compare_groups = [](const std::vector<GlobalDeviceId>& lhs,
                           const std::vector<GlobalDeviceId>& rhs) {
    CHECK(!lhs.empty());
    CHECK(!rhs.empty());
    return lhs[0] < rhs[0];
  };
  absl::c_sort(participant_groups_, compare_groups);
}

bool GpuCliqueKey::is_p2p() const { return is_p2p_; }

std::vector<std::vector<GlobalDeviceId>> GpuCliqueKey::ParticipantGroups()
    const {
  return participant_groups_;
};

bool GpuCliqueKey::IsSubsetOf(const CliqueKey& other) const {
  auto* other_gpu = tsl::down_cast<const GpuCliqueKey*>(&other);
  if (other_gpu == nullptr) {
    return false;
  }

  return is_p2p() == other_gpu->is_p2p() &&
         absl::c_all_of(devices(), [&](GlobalDeviceId id) {
           return absl::c_linear_search(other_gpu->devices(), id);
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
  std::string group_string = "";
  if (!participant_groups_.empty()) {
    std::vector<std::string> values;
    values.reserve(participant_groups_.size());
    for (const auto& group : participant_groups_) {
      values.push_back(absl::StrFormat("[%s]", HumanReadableDevices(group)));
    }
    group_string = absl::StrFormat("; groups=[%s]", absl::StrJoin(values, ","));
  }
  return absl::StrFormat(
      "devices=%d:[%s]; is_p2p=%v%s; local_participants=%lld; "
      "incarnations=[%s]",
      devices().size(), HumanReadableDevices(devices()), is_p2p_, group_string,
      num_local_participants_,
      absl::StrJoin(incarnations_, ", ",
                    [](std::string* out, IncarnationId id) {
                      absl::StrAppend(out, id.value());
                    }));
}

void GpuCliqueKey::HashValue(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices(), participant_groups_,
                           incarnations_);
}

bool operator==(const GpuCliqueKey& a, const GpuCliqueKey& b) {
  return a.devices() == b.devices() &&
         a.participant_groups_ == b.participant_groups_ &&
         a.num_local_participants_ == b.num_local_participants_ &&
         a.incarnations_ == b.incarnations_;
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
