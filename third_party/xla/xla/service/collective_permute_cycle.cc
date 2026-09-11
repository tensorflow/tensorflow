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

#include "xla/service/collective_permute_cycle.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "xla/service/source_target_pairs.h"

namespace xla {
namespace collective_permute_cycle {

bool IsForwardCycle(const SourceTargetPairs& backedge,
                    const SourceTargetPairs& others) {
  if (backedge.size() != 1) {
    return false;
  }
  const size_t num_pairs = others.size() + 1;
  if (backedge[0].source != num_pairs - 1 || backedge[0].target != 0) {
    return false;
  }
  for (size_t i = 0; i < num_pairs - 1; ++i) {
    const SourceTargetPair& pair = others[i];
    if (pair.source != i || pair.target != i + 1) {
      return false;
    }
  }
  return true;
}

bool IsBackwardCycle(const SourceTargetPairs& backedge,
                     const SourceTargetPairs& others) {
  if (backedge.size() != 1) {
    return false;
  }
  const size_t num_pairs = others.size() + 1;
  if (backedge[0].source != 0 || backedge[0].target != num_pairs - 1) {
    return false;
  }
  for (size_t i = 0; i < num_pairs - 1; ++i) {
    const SourceTargetPair& pair = others[i];
    if (pair.source != i + 1 || pair.target != i) {
      return false;
    }
  }
  return true;
}

std::pair<SourceTargetPairs, SourceTargetPairs> SplitEdges(
    const SourceTargetPairs& pairs, CycleType cycle_type) {
  SourceTargetPairs back, fwd;
  size_t back_pair_index =
      cycle_type == CycleType::kBackward ? 0 : pairs.size() - 1;
  for (size_t i = 0; i < pairs.size(); ++i) {
    if (i == back_pair_index) {
      back.push_back(pairs[i]);
    } else {
      fwd.push_back(pairs[i]);
    }
  }
  return {back, fwd};
}

// cannonical forward: {{0,1},{1,2},{2,3},{3,0}}
bool IsForwardCycle(const SourceTargetPairs& pairs) {
  size_t size = pairs.size();
  if (size <= 1) return false;
  if (pairs[size - 1].target != pairs[0].source) {
    return false;
  }
  for (size_t i = 0; i < size - 1; ++i) {
    int64_t expected_next = pairs[i].source + 1;
    if (pairs[i].target != expected_next ||
        pairs[i + 1].source != expected_next) {
      return false;
    }
  }
  return true;
}

// cannonical backward: {{0,3},{1,0},{2,1},{3,2}}
bool IsBackwardCycle(const SourceTargetPairs& pairs) {
  size_t size = pairs.size();
  if (size <= 1) return false;
  if (pairs[0].target != pairs[size - 1].source) {
    return false;
  }
  for (size_t i = size - 1; i > 0; --i) {
    int64_t expected_next = pairs[i].source - 1;
    if (pairs[i].target != expected_next ||
        pairs[i - 1].source != expected_next) {
      return false;
    }
  }
  return true;
}

// Assumptions: pairs are ordered and 0 based; there is only cycle type and all
// elements participating in it.
CycleType GetCycleType(const SourceTargetPairs& pairs) {
  if (pairs.size() > 1) {
    if (IsForwardCycle(pairs)) {
      return CycleType::kForward;
    }
    if (IsBackwardCycle(pairs)) {
      return CycleType::kBackward;
    }
  }
  return CycleType::kNone;
}

bool HasCycles(const SourceTargetPairs& pairs) {
  // Build source-target map for quick lookup.
  std::vector<int64_t> source_target_map(pairs.size(), -1);
  for (int64_t i = 0; i < pairs.size(); ++i) {
    int64_t source = pairs[i].source;
    int64_t target = pairs[i].target;
    while (source_target_map.size() <= source) source_target_map.push_back(-1);
    source_target_map[source] = target;
  }

  // Cache indices known to be acyclic.
  absl::flat_hash_set<int64_t> acyclic;

  // Search for cycles.
  int64_t n = source_target_map.size();
  for (int64_t i = 0; i < n; ++i) {
    absl::flat_hash_set<int64_t> path;
    int64_t current = i;
    while (current != -1 && !acyclic.contains(current)) {
      if (path.contains(current)) return true;
      path.insert(current);
      current = current < n ? source_target_map[current] : -1;
    }
    acyclic.insert(path.begin(), path.end());
  }

  // No cycles found.
  return false;
}

}  // namespace collective_permute_cycle
}  // namespace xla
