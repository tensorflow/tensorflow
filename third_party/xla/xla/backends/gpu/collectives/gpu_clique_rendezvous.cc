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

#include "xla/backends/gpu/collectives/gpu_clique_rendezvous.h"

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/rendezvous.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

// Wrap GpuCliqueKey into a unique struct to guarantee we do not accidentally
// try to run multiple unrelated rendezvous for a same key.
struct GpuCliqueRendezvousKey {
  GpuCliqueKey clique_key;

  bool operator==(const GpuCliqueRendezvousKey& other) const {
    return clique_key == other.clique_key;
  }

  template <typename H>
  friend H AbslHashValue(H h, const GpuCliqueRendezvousKey& key) {
    return H::combine(std::move(h), key.clique_key);
  }
};

struct RankData {
  RankId rank;
  std::any data;
};

struct RankFormatter {
  void operator()(std::string* out, const RankData* param) const {
    absl::StrAppend(out, param->rank.value());
  }
};

}  // namespace

GpuCliqueRendezvous::GpuCliqueRendezvous(
    GpuCliqueKey clique_key, absl::btree_map<RankId, std::any> values)
    : clique_key_(std::move(clique_key)), values_(std::move(values)) {}

absl::StatusOr<std::shared_ptr<GpuCliqueRendezvous>> GpuCliqueRendezvous::Join(
    const GpuCliqueKey& clique_key, RankId rank, std::any data) {
  if (!clique_key.is_local()) {
    return InvalidArgument(
        "GpuClique rendezvous is not supported for non-local cliques");
  }

  VLOG(3) << absl::StrFormat("rank=[%d] Join gpu clique rendezvous: %s",
                             rank.value(), clique_key.ToString());

  std::string rendezvous_name =
      absl::StrFormat("GpuCliqueRendezvous for %s", clique_key.ToString());
  GpuCliqueRendezvousKey rendezvous_key = {clique_key};
  RankData rendezvous_value = {rank, std::move(data)};

  // A callback for rendezvous to construct the GpuCliqueRendezvous.
  auto callback = [&](absl::Span<const RankData*> values) {
    VLOG(3) << absl::StrFormat("[ranks=%s] Complete gpu clique rendezvous: %s",
                               absl::StrJoin(values, ",", RankFormatter{}),
                               clique_key.ToString());

    absl::btree_map<RankId, std::any> data;
    for (const auto* value : values) {
      data[value->rank] = std::move(value->data);
    }

    return GpuCliqueRendezvous(clique_key, std::move(data));
  };

  // We expect that all local participants will collectively allocate the
  // multicast memory.
  int64_t num_participants = clique_key.num_local_participants();
  return Rendezvous<GpuCliqueRendezvous>(rendezvous_name, rendezvous_key,
                                         rendezvous_value, num_participants,
                                         callback);
}

}  // namespace xla::gpu
