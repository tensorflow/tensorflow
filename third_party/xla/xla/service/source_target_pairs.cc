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
    res.push_back(group.replica_ids(0), group.replica_ids(1));
  }
  return res;
}

namespace {
int32_t GetNodeId(int64_t replica, GraphCycles& graph,
                  absl::flat_hash_map<int64_t, int32_t>& map) {
  if (!map.contains(replica)) {
    map.emplace(replica, graph.NewNode());
  }
  return map.at(replica);
}
}  // namespace

bool SourceTargetPairs::HasCycles() {
  GraphCycles graph;
  absl::flat_hash_map<int64_t, int32_t> replica_to_node_id;
  for (const SourceTargetPair& pair : pairs_) {
    const int source = GetNodeId(pair.source, graph, replica_to_node_id);
    const int target = GetNodeId(pair.target, graph, replica_to_node_id);
    if (!graph.InsertEdge(source, target)) {
      return true;
    }
  }
  return false;
}

// TODO: b/388623407 - remove assumptions that pairs are ordered and 0 based.
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

}  // namespace xla
