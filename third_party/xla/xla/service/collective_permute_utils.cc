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

#include "xla/service/collective_permute_utils.h"

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/graphcycles/graphcycles.h"

namespace xla {
namespace cp_utils {

using ::xla::HloCollectivePermuteInstruction;

std::string SourceTargetPairsString(const HloCollectivePermuteInstruction& cp) {
  auto formatter = absl::PairFormatter(
      [](std::string* out, int64_t value) { absl::StrAppend(out, "{", value); },
      ",",
      [](std::string* out, int64_t value) {
        absl::StrAppend(out, value, "}");
      });
  const std::string pairs_str =
      absl::StrJoin(cp.source_target_pairs(), ",", formatter);
  return absl::StrCat("{", pairs_str, "}");
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

bool HasCycles(const SourceTargetPairs& pairs) {
  GraphCycles graph;
  absl::flat_hash_map<int64_t, int32_t> replica_to_node_id;
  for (const SourceTargetPair& pair : pairs) {
    const int source = GetNodeId(pair.first, graph, replica_to_node_id);
    const int target = GetNodeId(pair.second, graph, replica_to_node_id);
    if (!graph.InsertEdge(source, target)) {
      return true;
    }
  }
  return false;
}

// TODO: b/388623407 - remove assumptions that pairs are ordered and 0 based.
bool IsForwardCycle(const SourceTargetPair& backedge,
                    const SourceTargetPairs& others) {
  const int64_t num_pairs = others.size() + 1;
  if (backedge.first != num_pairs - 1 || backedge.second != 0) {
    return false;
  }
  for (int64_t i = 0; i < num_pairs - 1; ++i) {
    const SourceTargetPair& pair = others[i];
    if (pair.first != i || pair.second != i + 1) {
      return false;
    }
  }
  return true;
}

bool IsBackwardCycle(const SourceTargetPair& backedge,
                     const SourceTargetPairs& others) {
  const int64_t num_pairs = others.size() + 1;
  if (backedge.first != 0 || backedge.second != num_pairs - 1) {
    return false;
  }
  for (int64_t i = 0; i < num_pairs - 1; ++i) {
    const SourceTargetPair& pair = others[i];
    if (pair.first != i + 1 || pair.second != i) {
      return false;
    }
  }
  return true;
}

}  // namespace cp_utils
}  // namespace xla
