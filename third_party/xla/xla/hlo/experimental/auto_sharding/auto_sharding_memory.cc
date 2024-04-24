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
using Interval = std::pair<LiveIdx, LiveIdx>;

bool IsValid(const Interval& interval) {
  return interval.first <= interval.second;
}

int64_t length(const Interval& interval) {
  return interval.second - interval.first + 1;  // (closed interval)
}

}  // namespace

std::pair<int64_t, int64_t> MemoryTermReducer::Reduce(
    int64_t num_lives, int64_t num_primitives,
    const std::function<
        tsl::protobuf::RepeatedField<int64_t>(int64_t)>&  // NOLINT
        live) {
  LOG(INFO) << "Memory Term Reducer beginning to reduce number of terms ...";

  // Clear internal state.
  reduced_live_.clear();
  reduced_intervals_.clear();
  reduced_groups_.clear();

  // For each primitive, determine the live interval it spans.
  int64_t num_terms = 0;
  reduced_intervals_.reserve(num_primitives);
  for (PrimIdx prim_idx = 0; prim_idx < num_primitives; ++prim_idx) {
    reduced_intervals_.push_back({std::numeric_limits<LiveIdx>::max(), 0});
  }
  for (LiveIdx live_idx = 0; live_idx < num_lives; ++live_idx) {
    for (const PrimIdx prim_idx : live(live_idx)) {
      reduced_intervals_[prim_idx].first =
          std::min(reduced_intervals_[prim_idx].first, live_idx);
      reduced_intervals_[prim_idx].second =
          std::max(reduced_intervals_[prim_idx].second, live_idx);
      ++num_terms;
    }
  }

  Reduce(num_lives, num_primitives);

  // Create the reduced live matrix.
  int64_t num_reduced_terms = 0;
  reduced_live_.resize(num_lives);
  for (PrimIdx prim_idx = 0; prim_idx < reduced_intervals_.size(); ++prim_idx) {
    for (LiveIdx live_idx = reduced_intervals_[prim_idx].first;
         live_idx <= reduced_intervals_[prim_idx].second; ++live_idx) {
      reduced_live_[live_idx].push_back(prim_idx);
      ++num_reduced_terms;
    }
  }

  // Add in any additional terms that will be needed to define groups.
  for (const auto& group : reduced_groups_) num_reduced_terms += group.size();

  LOG(INFO) << "Memory Term Reducer finished reducing the number of terms.";
  return {num_terms, num_reduced_terms};
}

std::pair<int64_t, int64_t> MemoryTermReducer::Reduce(
    int64_t num_lives, int64_t num_primitives,
    const std::function<std::pair<int64_t, int64_t>(int64_t)>& intervals) {
  LOG(INFO) << "Memory Term Reducer beginning to reduce number of terms ...";

  // Clear internal state.
  reduced_live_.clear();
  reduced_intervals_.clear();
  reduced_groups_.clear();

  // For each primitive, record the live interval it spans.
  int64_t num_terms = 0;
  reduced_intervals_.reserve(num_primitives);
  for (PrimIdx prim_idx = 0; prim_idx < num_primitives; ++prim_idx) {
    reduced_intervals_.push_back(intervals(prim_idx));
    num_terms += length(reduced_intervals_.back());
  }

  Reduce(num_lives, num_primitives);

  // Calculate the number of reduced terms.
  int64_t num_reduced_terms = 0;
  for (PrimIdx prim_idx = 0; prim_idx < reduced_intervals_.size(); ++prim_idx) {
    if (!IsValid(reduced_intervals_[prim_idx])) continue;
    num_reduced_terms += length(reduced_intervals_[prim_idx]);
  }

  // Add in any additional terms that will be needed to define groups.
  for (const auto& group : reduced_groups_) num_reduced_terms += group.size();

  LOG(INFO) << "Memory Term Reducer finished reducing the number of terms.";
  return {num_terms, num_reduced_terms};
}

void MemoryTermReducer::Reduce(int64_t num_lives, int64_t num_primitives) {
  // For each live index, track the primitives entering memory or being evicted.
  std::vector<absl::btree_set<PrimIdx>> enter(num_lives), evict(num_lives);
  for (PrimIdx prim_idx = 0; prim_idx < num_primitives; ++prim_idx) {
    if (!IsValid(reduced_intervals_[prim_idx])) {
      continue;  // Not found in live matrix.
    }
    enter[reduced_intervals_[prim_idx].first].insert(prim_idx);
    evict[reduced_intervals_[prim_idx].second].insert(prim_idx);
  }

  // A function to determine if one primitive would 'split' another.
  auto Splits = [this](PrimIdx large_idx, PrimIdx small_idx) -> bool {
    return reduced_intervals_[large_idx].first <
               reduced_intervals_[small_idx].first &&
           reduced_intervals_[large_idx].second >
               reduced_intervals_[small_idx].second;
  };

  // A function to calculate the overlap between any pair.
  auto CalcOverlap = [this, Splits](
                         int64_t prim0_idx,
                         int64_t prim1_idx) -> std::optional<Interval> {
    if (!IsValid(reduced_intervals_[prim0_idx]) ||
        !IsValid(reduced_intervals_[prim1_idx])) {
      return std::nullopt;  // Happens when prim is absent in matrix or vanishes
    }
    if (Splits(prim0_idx, prim1_idx) || Splits(prim1_idx, prim0_idx)) {
      return std::nullopt;  // Merging these would split one of the primitives.
    }
    const Interval overlap = {std::max(reduced_intervals_[prim0_idx].first,
                                       reduced_intervals_[prim1_idx].first),
                              std::min(reduced_intervals_[prim0_idx].second,
                                       reduced_intervals_[prim1_idx].second)};
    return overlap;
  };

  // A function that merges a primitive (or members of a group) into a group.
  auto MergeIntoGroup = [num_primitives, this](
                            PrimIdx prim_idx,
                            absl::btree_set<PrimIdx>& reduced_group) {
    if (prim_idx < num_primitives) {
      reduced_group.insert(prim_idx);
    } else {
      const auto& group = reduced_groups_[prim_idx - num_primitives];
      reduced_group.insert(group.begin(), group.end());
    }
  };

  // A function that calculates the # of terms a primitive (or group) uses.
  auto CalcNumTerms = [num_primitives, this](
                          PrimIdx prim_idx,
                          std::optional<Interval> overlap = std::nullopt) {
    int64_t num_terms = length(reduced_intervals_[prim_idx]);
    if (overlap) num_terms -= length(*overlap);
    if (prim_idx >= num_primitives && num_terms > 0) {
      num_terms += reduced_groups_[prim_idx - num_primitives].size();
    }
    return num_terms;
  };

  // A function to update a primitive after being merged into a group.
  auto UpdatePrimitive = [this, &enter, &evict](
                             PrimIdx prim_idx,
                             const Interval& overlap) mutable {
    enter[reduced_intervals_[prim_idx].first].erase(prim_idx);
    evict[reduced_intervals_[prim_idx].second].erase(prim_idx);
    if (reduced_intervals_[prim_idx].first == overlap.first) {
      reduced_intervals_[prim_idx].first = overlap.second + 1;
    }
    if (reduced_intervals_[prim_idx].second == overlap.second) {
      reduced_intervals_[prim_idx].second = overlap.first - 1;
    }
    if (!IsValid(reduced_intervals_[prim_idx])) return;  // It vanished.
    enter[reduced_intervals_[prim_idx].first].insert(prim_idx);
    evict[reduced_intervals_[prim_idx].second].insert(prim_idx);
  };

  // A function to sweep through live points & merge large overlaps.
  auto SweepAndMerge = [&num_lives, &enter, &evict, &CalcOverlap, &CalcNumTerms,
                        &MergeIntoGroup, &UpdatePrimitive, this]() -> bool {
    absl::btree_set<LiveAndPrim> actives;  // Active prims sorted by first value
    absl::btree_multimap<int64_t, PrimPair> overlaps;
    for (LiveIdx live_idx = 0; live_idx < num_lives; ++live_idx) {
      for (const PrimIdx prim_idx : enter[live_idx]) {
        actives.insert({live_idx, prim_idx});
      }
      for (const PrimIdx prim_idx : evict[live_idx]) {
        std::optional<LiveAndPrim> active;  // The active prim we overlap with.
        auto prim =
            actives.find({reduced_intervals_[prim_idx].first, prim_idx});
        if (auto next = prim; ++next != actives.end()) {
          active = *next;  // Choose a prim that started soon after we did.
        } else if (actives.begin()->second != prim_idx) {
          active = *actives.begin();  // Otherwise, choose the earliest prim.
        }
        actives.erase(prim);
        if (!active) continue;
        overlaps.insert({active->first - live_idx, {prim_idx, active->second}});
      }
    }
    bool changed = false;
    for (const auto& [_, prim_pair] : overlaps) {
      PrimIdx prim0_idx = prim_pair.first, prim1_idx = prim_pair.second;
      std::optional<Interval> overlap = CalcOverlap(prim0_idx, prim1_idx);
      if (!overlap) continue;
      absl::btree_set<PrimIdx> reduced_group;
      MergeIntoGroup(prim0_idx, reduced_group);
      MergeIntoGroup(prim1_idx, reduced_group);
      if (CalcNumTerms(prim0_idx) + CalcNumTerms(prim1_idx) <=
          CalcNumTerms(prim0_idx, overlap) + CalcNumTerms(prim1_idx, overlap) +
              length(*overlap) + reduced_group.size()) {
        continue;  // Not reduced.
      }
      enter[overlap->first].insert(reduced_intervals_.size());
      evict[overlap->second].insert(reduced_intervals_.size());
      reduced_intervals_.push_back({overlap->first, overlap->second});
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
    if (IsValid(reduced_intervals_[num_primitives + group_idx])) continue;
    reduced_intervals_.erase(reduced_intervals_.begin() + num_primitives +
                             group_idx);
    reduced_groups_.erase(reduced_groups_.begin() + group_idx);
  }
}

const std::vector<std::vector<int64_t>>& MemoryTermReducer::GetReducedLive()
    const {
  return reduced_live_;
}

const std::vector<std::pair<int64_t, int64_t>>&
MemoryTermReducer::GetReducedIntervals() const {
  return reduced_intervals_;
}

const std::vector<absl::btree_set<int64_t>>&
MemoryTermReducer::GetReducedGroups() const {
  return reduced_groups_;
}

}  // namespace spmd
}  // namespace xla
