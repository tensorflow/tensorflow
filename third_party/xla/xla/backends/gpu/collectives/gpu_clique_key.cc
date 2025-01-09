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
#include "tsl/platform/casts.h"
#include "tsl/platform/logging.h"

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
    std::vector<GlobalDeviceId> devices, CollectiveStreamId stream_id,
    AsyncStreamKind stream_kind,
    std::vector<std::vector<GlobalDeviceId>> participant_groups)
    : CliqueKey(std::move(devices)),
      stream_id_(stream_id),
      stream_kind_(stream_kind),
      participant_groups_(std::move(participant_groups)) {
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

bool GpuCliqueKey::IsSubsetOf(const CliqueKey& other) const {
  auto* other_nccl = tsl::down_cast<const GpuCliqueKey*>(&other);
  if (other_nccl == nullptr) return false;

  return stream_id_ == other_nccl->stream_id_ &&
         absl::c_all_of(devices(), [&](GlobalDeviceId id) {
           return absl::c_linear_search(other_nccl->devices(), id);
         });
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
  return absl::StrFormat("devices=[%s]; stream=%d%s",
                         GlobalDeviceIdsToString(devices()), stream_id_.value(),
                         group_string);
}

void GpuCliqueKey::HashValue(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices(), stream_id_,
                           participant_groups_);
}

bool operator==(const GpuCliqueKey& a, const GpuCliqueKey& b) {
  return a.devices() == b.devices() && a.stream_id_ == b.stream_id_ &&
         a.participant_groups_ == b.participant_groups_;
}

bool operator<(const GpuCliqueKey& a, const GpuCliqueKey& b) {
  if (a.devices().size() < b.devices().size()) return true;
  if (b.devices().size() < a.devices().size()) return false;

  if (a.devices() < b.devices()) return true;
  if (b.devices() < a.devices()) return false;

  return a.stream_id_.value() < b.stream_id_.value();
}

bool operator>(const GpuCliqueKey& a, const GpuCliqueKey& b) {
  if (a.devices().size() > b.devices().size()) return true;
  if (b.devices().size() > a.devices().size()) return false;

  if (a.devices() > b.devices()) return true;
  if (b.devices() > a.devices()) return false;

  // We still use `<` to order by stream id as we want to acquire sync cliques
  // before async ones.
  return a.stream_id_.value() < b.stream_id_.value();
}

}  // namespace xla::gpu
