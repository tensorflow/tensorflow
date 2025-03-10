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

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/service/global_device_id.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {

CollectiveStreamId GetCollectiveStreamId(bool is_async,
                                         AsyncStreamKind stream_kind) {
  // TODO(ezhulenev): This implementation does not look correct as stream IDs
  // are not really unique. Figure out if it's the case and fix either the code
  // or the documentation.
  int64_t stream_id = static_cast<int64_t>(stream_kind);
  return CollectiveStreamId(is_async ? stream_id + 1 : 0);
}

GpuCliqueKey::GpuCliqueKey(
    std::vector<GlobalDeviceId> devices, int64_t num_local_participants,
    CollectiveStreamId stream_id, AsyncStreamKind stream_kind,
    std::vector<std::vector<GlobalDeviceId>> participant_groups,
    GlobalDeviceId root_device, std::vector<uint64_t> incarnations)
    : CliqueKey(std::move(devices)),
      num_local_participants_(num_local_participants),
      stream_id_(stream_id),
      stream_kind_(stream_kind),
      participant_groups_(std::move(participant_groups)),
      root_device_(root_device),
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

CollectiveStreamId GpuCliqueKey::stream_id() const { return stream_id_; }

GlobalDeviceId GpuCliqueKey::root_device() const { return root_device_; }

bool GpuCliqueKey::IsSubsetOf(const CliqueKey& other) const {
  auto* other_gpu = tsl::down_cast<const GpuCliqueKey*>(&other);
  if (other_gpu == nullptr) {
    return false;
  }

  return stream_id_ == other_gpu->stream_id_ &&
         absl::c_all_of(devices(), [&](GlobalDeviceId id) {
           return absl::c_linear_search(other_gpu->devices(), id);
         });
}

std::vector<GpuCliqueKey> GpuCliqueKey::GetSubKeys(int64_t nroots) const {
  const auto& devs = devices();
  int64_t nranks = devs.size();
  CHECK(nroots <= nranks);
  int64_t rank_per_root = nranks / nroots;
  int64_t rank_rem = nranks % nroots;
  std::vector<GpuCliqueKey> subkeys;
  for (int64_t i = 0; i < nroots; ++i) {
    GpuCliqueKey subkey(*this);
    if (i < rank_rem) {
      subkey.root_device_ = devs[i * (rank_per_root + 1)];
    } else {
      subkey.root_device_ =
          devs[rank_rem * (rank_per_root + 1) + (i - rank_rem) * rank_per_root];
    }
    subkeys.push_back(subkey);
  }
  return subkeys;
}

std::string GpuCliqueKey::ToString() const {
  std::string group_string = "";
  if (!participant_groups_.empty()) {
    std::vector<std::string> values;
    values.reserve(participant_groups_.size());
    for (const auto& group : participant_groups_) {
      values.push_back("[" + GlobalDeviceIdsToString(group) + "]");
    }
    group_string = absl::StrFormat("; groups=[%s]", absl::StrJoin(values, ","));
  }
  return absl::StrFormat(
      "devices=[%s]; stream=%d%s; root_device=%lld; "
      "num_local_participants=%lld; incarnations=[%s]",
      GlobalDeviceIdsToString(devices()), stream_id_.value(), group_string,
      root_device_.value(), num_local_participants_,
      absl::StrJoin(incarnations_, ", "));
}

void GpuCliqueKey::HashValue(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices(), stream_id_,
                           participant_groups_, root_device_, incarnations_);
}

bool operator==(const GpuCliqueKey& a, const GpuCliqueKey& b) {
  return a.devices() == b.devices() && a.stream_id_ == b.stream_id_ &&
         a.participant_groups_ == b.participant_groups_ &&
         a.num_local_participants_ == b.num_local_participants_ &&
         a.root_device_ == b.root_device_ && a.incarnations_ == b.incarnations_;
}

// Constructs a tuple from the clique key for comparison purposes.
static auto CmpKey(const GpuCliqueKey& key) {
  return std::make_tuple(key.devices().size(), key.devices(), key.root_device(),
                         key.num_local_participants(), key.stream_id().value(),
                         key.incarnations());
}

bool operator<(const GpuCliqueKey& a, const GpuCliqueKey& b) {
  return CmpKey(a) < CmpKey(b);
}

bool operator>(const GpuCliqueKey& a, const GpuCliqueKey& b) {
  return CmpKey(a) > CmpKey(b);
}

}  // namespace xla::gpu
