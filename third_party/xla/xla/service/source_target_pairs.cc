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

#include "xla/service/source_target_pairs.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/graphcycles/graphcycles.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

std::string SourceTargetPairs::ToString() const {
  auto formatter = [](std::string* out, const SourceTargetPair& pair) {
    absl::StrAppend(out, "{", pair.source, ",", pair.target, "}");
  };
  const std::string pairs_str = absl::StrJoin(pairs_, ",", formatter);
  return absl::StrCat("{", pairs_str, "}");
}

absl::StatusOr<SourceTargetPairs> SourceTargetPairs::FromString(
    absl::string_view str) {
  // reusing replica groups parsing.
  TF_ASSIGN_OR_RETURN(std::vector<ReplicaGroup> groups,
                      // absl::StatusOr<std::vector<ReplicaGroup>> groups =
                      ParseReplicaGroupsOnly(str));
  SourceTargetPairs res;
  for (const ReplicaGroup& group : groups) {
    if (group.replica_ids_size() != 2) {
      return Internal("Incorrect element size : %s", str);
    }
    res.emplace_back(group.replica_ids(0), group.replica_ids(1));
  }
  return res;
}

bool SourceTargetPairs::IsForwardCycle(const SourceTargetPairs& backedge,
                                       const SourceTargetPairs& others) {
  if (backedge.size() != 1) {
    return false;
  }
  const int64_t num_pairs = others.size() + 1;
  if (backedge[0].source != num_pairs - 1 || backedge[0].target != 0) {
    return false;
  }
  for (int64_t i = 0; i < num_pairs - 1; ++i) {
    const SourceTargetPair& pair = others[i];
    if (pair.source != i || pair.target != i + 1) {
      return false;
    }
  }
  return true;
}

bool SourceTargetPairs::IsBackwardCycle(const SourceTargetPairs& backedge,
                                        const SourceTargetPairs& others) {
  if (backedge.size() != 1) {
    return false;
  }
  const int64_t num_pairs = others.size() + 1;
  if (backedge[0].source != 0 || backedge[0].target != num_pairs - 1) {
    return false;
  }
  for (int64_t i = 0; i < num_pairs - 1; ++i) {
    const SourceTargetPair& pair = others[i];
    if (pair.source != i + 1 || pair.target != i) {
      return false;
    }
  }
  return true;
}

std::pair<SourceTargetPairs, SourceTargetPairs> SourceTargetPairs::SplitEdges(
    CycleType cycle_type) const {
  SourceTargetPairs back, fwd;
  size_t back_pair_index = cycle_type == CycleType::kBackward ? 0 : size() - 1;
  for (size_t i = 0; i < pairs_.size(); ++i) {
    if (i == back_pair_index) {
      back.push_back(pairs_[i]);
    } else {
      fwd.push_back(pairs_[i]);
    }
  }
  return {back, fwd};
}

// cannonical forward: {{0,1},{1,2},{2,3},{3,0}}
bool SourceTargetPairs::IsForwardCycle() const {
  size_t size = pairs_.size();
  if (size <= 1) return false;
  if (pairs_[size - 1].target != pairs_[0].source) {
    return false;
  }
  for (int64_t i = 0; i < size - 1; ++i) {
    int64_t expected_next = pairs_[i].source + 1;
    if (pairs_[i].target != expected_next ||
        pairs_[i + 1].source != expected_next) {
      return false;
    }
  }
  return true;
}

// cannonical backward: {{0,3},{1,0},{2,1},{3,2}}
bool SourceTargetPairs::IsBackwardCycle() const {
  size_t size = pairs_.size();
  if (size <= 1) return false;
  if (pairs_[0].target != pairs_[size - 1].source) {
    return false;
  }
  for (int64_t i = size - 1; i > 0; --i) {
    int64_t expected_next = pairs_[i].source - 1;
    if (pairs_[i].target != expected_next ||
        pairs_[i - 1].source != expected_next) {
      return false;
    }
  }
  return true;
}

// Assumptions: pairs are ordered and 0 based; there is only cycle type and all
// elements participating in it.
SourceTargetPairs::CycleType SourceTargetPairs::GetCycleType() const {
  if (this->size() > 1) {
    if (IsForwardCycle()) {
      return CycleType::kForward;
    }
    if (IsBackwardCycle()) {
      return CycleType::kBackward;
    }
  }
  return CycleType::kUnknown;
}

}  // namespace xla
