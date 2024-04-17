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

#include "xla/hlo/experimental/auto_sharding/auto_sharding_memory.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace spmd {

namespace {

using PrimIdx = int64_t;  // Indexes into the primitive list (ie, nodes & edges)
using LiveIdx = int64_t;  // Indexes into the liveness range (like a time point)
using GroupIdx = int64_t;  // Indexes into the list of groups

using PrimPair = std::pair<PrimIdx, PrimIdx>;
using LiveAndPrim = std::pair<LiveIdx, PrimIdx>;

struct Interval {
  LiveIdx lower = std::numeric_limits<LiveIdx>::max();
  LiveIdx upper = 0;

  bool IsValid() const { return lower <= upper; }
  int64_t length() const { return upper - lower + 1; }  // (closed interval)
};

}  // namespace

int64_t MemoryTermReducer::Reduce(
    int64_t num_lives, int64_t num_primitives,
    std::function<tsl::protobuf::RepeatedField<int64_t>(int64_t)>  // NOLINT
        live) {
  LOG(INFO) << "Memory Term Reducer beginning to reduce number of terms ...";

  // Clear internal state.
  reduced_live_.clear();
  reduced_groups_.clear();

  // For each primitive, determine the live interval it spans.
  int64_t num_terms = 0;
  std::vector<Interval> intervals(num_primitives);
  for (LiveIdx live_idx = 0; live_idx < num_lives; ++live_idx) {
    for (const PrimIdx prim_idx : live(live_idx)) {
      intervals[prim_idx].lower = std::min(intervals[prim_idx].lower, live_idx);
      intervals[prim_idx].upper = std::max(intervals[prim_idx].upper, live_idx);
      ++num_terms;
    }
  }

  // For each live index, track the primitives entering memory or being evicted.
  std::vector<absl::flat_hash_set<PrimIdx>> enter(num_lives), evict(num_lives);
  for (PrimIdx prim_idx = 0; prim_idx < num_primitives; ++prim_idx) {
    if (!intervals[prim_idx].IsValid()) continue;  // Not found in live matrix.
    enter[intervals[prim_idx].lower].insert(prim_idx);
    evict[intervals[prim_idx].upper].insert(prim_idx);
  }

  // A function to determine if one primitive would 'split' another.
  auto Splits = [&intervals](PrimIdx large_idx, PrimIdx small_idx) -> bool {
    return intervals[large_idx].lower < intervals[small_idx].lower &&
           intervals[large_idx].upper > intervals[small_idx].upper;
  };

  // A function to calculate the overlap between any pair.
  auto CalcOverlap = [&intervals, Splits](
                         int64_t prim0_idx,
                         int64_t prim1_idx) -> std::optional<Interval> {
    if (!intervals[prim0_idx].IsValid() || !intervals[prim1_idx].IsValid()) {
      return std::nullopt;  // Happens when prim is absent in matrix or vanishes
    }
    if (Splits(prim0_idx, prim1_idx) || Splits(prim1_idx, prim0_idx)) {
      return std::nullopt;  // Merging these would split one of the primitives.
    }
    const Interval overlap = {
        std::max(intervals[prim0_idx].lower, intervals[prim1_idx].lower),
        std::min(intervals[prim0_idx].upper, intervals[prim1_idx].upper)};
    return overlap;
  };

  // A function that merges a primitive (or members of a group) into a group.
  auto MergeIntoGroup = [num_primitives, this](
                            PrimIdx prim_idx,
                            absl::flat_hash_set<PrimIdx>& reduced_group) {
    if (prim_idx < num_primitives) {
      reduced_group.insert(prim_idx);
    } else {
      const auto& group = reduced_groups_[prim_idx - num_primitives];
      reduced_group.insert(group.begin(), group.end());
    }
  };

  // A function that calculates the # of terms a primitive (or group) uses.
  auto CalcNumTerms = [num_primitives, &intervals, this](
                          PrimIdx prim_idx,
                          std::optional<Interval> overlap = std::nullopt) {
    int64_t num_terms = intervals[prim_idx].length();
    if (overlap) num_terms -= overlap->length();
    if (prim_idx >= num_primitives && num_terms > 0) {
      num_terms += reduced_groups_[prim_idx - num_primitives].size();
    }
    return num_terms;
  };

  // A function to update a primitive after being merged into a group.
  auto UpdatePrimitive = [&intervals, &enter, &evict](
                             PrimIdx prim_idx,
                             const Interval& overlap) mutable {
    enter[intervals[prim_idx].lower].erase(prim_idx);
    evict[intervals[prim_idx].upper].erase(prim_idx);
    if (intervals[prim_idx].lower == overlap.lower) {
      intervals[prim_idx].lower = overlap.upper + 1;
    }
    if (intervals[prim_idx].upper == overlap.upper) {
      intervals[prim_idx].upper = overlap.lower - 1;
    }
    if (!intervals[prim_idx].IsValid()) return;  // It vanished.
    enter[intervals[prim_idx].lower].insert(prim_idx);
    evict[intervals[prim_idx].upper].insert(prim_idx);
  };

  // A function to sweep through live points & merge large overlaps.
  auto SweepAndMerge = [&num_lives, &intervals, &enter, &evict, &CalcOverlap,
                        &CalcNumTerms, &MergeIntoGroup, &UpdatePrimitive,
                        this]() -> bool {
    absl::btree_set<LiveAndPrim> actives;  // Active prims sorted by lower value
    absl::btree_multimap<int64_t, PrimPair> overlaps;
    for (LiveIdx live_idx = 0; live_idx < num_lives; ++live_idx) {
      for (const PrimIdx prim_idx : enter[live_idx]) {
        actives.insert({live_idx, prim_idx});
      }
      for (const PrimIdx prim_idx : evict[live_idx]) {
        actives.erase({intervals[prim_idx].lower, prim_idx});
        if (actives.empty()) continue;
        const LiveAndPrim& active = *actives.begin();
        overlaps.insert({active.first - live_idx, {prim_idx, active.second}});
      }
    }
    bool changed = false;
    for (const auto& [_, prim_pair] : overlaps) {
      PrimIdx prim0_idx = prim_pair.first, prim1_idx = prim_pair.second;
      std::optional<Interval> overlap = CalcOverlap(prim0_idx, prim1_idx);
      if (!overlap) continue;
      absl::flat_hash_set<PrimIdx> reduced_group;
      MergeIntoGroup(prim0_idx, reduced_group);
      MergeIntoGroup(prim1_idx, reduced_group);
      if (CalcNumTerms(prim0_idx) + CalcNumTerms(prim1_idx) <=
          CalcNumTerms(prim0_idx, overlap) + CalcNumTerms(prim1_idx, overlap) +
              overlap->length() + reduced_group.size()) {
        continue;  // Not reduced.
      }
      enter[overlap->lower].insert(intervals.size());
      evict[overlap->upper].insert(intervals.size());
      intervals.push_back({overlap->lower, overlap->upper});
      reduced_groups_.push_back(reduced_group);
      UpdatePrimitive(prim0_idx, *overlap);
      UpdatePrimitive(prim1_idx, *overlap);
      changed = true;
    }
    return changed;
  };

  while (SweepAndMerge()) {
    // Repeated until no additional reductions can be achieved.
  }

  // Remove any groups that have vanished.
  for (GroupIdx group_idx = reduced_groups_.size() - 1; group_idx >= 0;
       --group_idx) {
    if (intervals[num_primitives + group_idx].IsValid()) continue;
    intervals.erase(intervals.begin() + num_primitives + group_idx);
    reduced_groups_.erase(reduced_groups_.begin() + group_idx);
  }

  // Create the reduced live matrix.
  int64_t num_reduced_terms = 0;
  reduced_live_.resize(num_lives);
  for (PrimIdx prim_idx = 0; prim_idx < intervals.size(); ++prim_idx) {
    for (LiveIdx live_idx = intervals[prim_idx].lower;
         live_idx <= intervals[prim_idx].upper; ++live_idx) {
      reduced_live_[live_idx].push_back(prim_idx);
      ++num_reduced_terms;
    }
  }

  // Add in any additional terms that will be needed to define groups.
  for (const auto& group : reduced_groups_) num_reduced_terms += group.size();

  LOG(INFO) << "Memory Term Reducer reduced number of terms from " << num_terms
            << " to " << num_reduced_terms;
  return num_reduced_terms;
}

const std::vector<std::vector<int64_t>>& MemoryTermReducer::GetReducedLive()
    const {
  return reduced_live_;
}

const std::vector<absl::flat_hash_set<int64_t>>&
MemoryTermReducer::GetReducedGroups() const {
  return reduced_groups_;
}

}  // namespace spmd
}  // namespace xla
