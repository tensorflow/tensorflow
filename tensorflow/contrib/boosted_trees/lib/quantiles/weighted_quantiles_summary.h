// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#ifndef TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_QUANTILES_WEIGHTED_QUANTILES_SUMMARY_H_
#define TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_QUANTILES_WEIGHTED_QUANTILES_SUMMARY_H_

#include <cstring>
#include <vector>

#include "tensorflow/contrib/boosted_trees/lib/quantiles/weighted_quantiles_buffer.h"

namespace tensorflow {
namespace boosted_trees {
namespace quantiles {

// Summary holding a sorted block of entries with upper bound guarantees
// over the approximation error.
template <typename ValueType, typename WeightType,
          typename CompareFn = std::less<ValueType>>
class WeightedQuantilesSummary {
 public:
  using Buffer = WeightedQuantilesBuffer<ValueType, WeightType, CompareFn>;
  using BufferEntry = typename Buffer::BufferEntry;

  struct SummaryEntry {
    SummaryEntry(const ValueType& v, const WeightType& w, const WeightType& min,
                 const WeightType& max) {
      // Explicitly initialize all of memory (including padding from memory
      // alignment) to allow the struct to be msan-resistant "plain old data".
      //
      // POD = http://en.cppreference.com/w/cpp/concept/PODType
      memset(this, 0, sizeof(*this));

      value = v;
      weight = w;
      min_rank = min;
      max_rank = max;
    }

    SummaryEntry() {
      memset(this, 0, sizeof(*this));

      value = ValueType();
      weight = 0;
      min_rank = 0;
      max_rank = 0;
    }

    bool operator==(const SummaryEntry& other) const {
      return value == other.value && weight == other.weight &&
             min_rank == other.min_rank && max_rank == other.max_rank;
    }
    friend std::ostream& operator<<(std::ostream& strm,
                                    const SummaryEntry& entry) {
      return strm << "{" << entry.value << ", " << entry.weight << ", "
                  << entry.min_rank << ", " << entry.max_rank << "}";
    }

    // Max rank estimate for previous smaller value.
    WeightType PrevMaxRank() const { return max_rank - weight; }

    // Min rank estimate for next larger value.
    WeightType NextMinRank() const { return min_rank + weight; }

    ValueType value;
    WeightType weight;
    WeightType min_rank;
    WeightType max_rank;
  };

  // Re-construct summary from the specified buffer.
  void BuildFromBufferEntries(const std::vector<BufferEntry>& buffer_entries) {
    entries_.clear();
    entries_.reserve(buffer_entries.size());
    WeightType cumulative_weight = 0;
    for (const auto& entry : buffer_entries) {
      WeightType current_weight = entry.weight;
      entries_.emplace_back(entry.value, entry.weight, cumulative_weight,
                            cumulative_weight + current_weight);
      cumulative_weight += current_weight;
    }
  }

  // Re-construct summary from the specified summary entries.
  void BuildFromSummaryEntries(
      const std::vector<SummaryEntry>& summary_entries) {
    entries_.clear();
    entries_.reserve(summary_entries.size());
    entries_.insert(entries_.begin(), summary_entries.begin(),
                    summary_entries.end());
  }

  // Merges two summaries through an algorithm that's derived from MergeSort
  // for summary entries while guaranteeing that the max approximation error
  // of the final merged summary is no greater than the approximation errors
  // of each individual summary.
  // For example consider summaries where each entry is of the form
  // (element, weight, min rank, max rank):
  // summary entries 1: (1, 3, 0, 3), (4, 2, 3, 5)
  // summary entries 2: (3, 1, 0, 1), (4, 1, 1, 2)
  // merged: (1, 3, 0, 3), (3, 1, 3, 4), (4, 3, 4, 7).
  void Merge(const WeightedQuantilesSummary& other_summary) {
    // Make sure we have something to merge.
    const auto& other_entries = other_summary.entries_;
    if (other_entries.empty()) {
      return;
    }
    if (entries_.empty()) {
      BuildFromSummaryEntries(other_summary.entries_);
      return;
    }

    // Move current entries to make room for a new buffer.
    std::vector<SummaryEntry> base_entries(std::move(entries_));
    entries_.clear();
    entries_.reserve(base_entries.size() + other_entries.size());

    // Merge entries maintaining ranks. The idea is to stack values
    // in order which we can do in linear time as the two summaries are
    // already sorted. We keep track of the next lower rank from either
    // summary and update it as we pop elements from the summaries.
    // We handle the special case when the next two elements from either
    // summary are equal, in which case we just merge the two elements
    // and simultaneously update both ranks.
    auto it1 = base_entries.cbegin();
    auto it2 = other_entries.cbegin();
    WeightType next_min_rank1 = 0;
    WeightType next_min_rank2 = 0;
    while (it1 != base_entries.cend() && it2 != other_entries.cend()) {
      if (kCompFn(it1->value, it2->value)) {  // value1 < value2
        // Take value1 and use the last added value2 to compute
        // the min rank and the current value2 to compute the max rank.
        entries_.emplace_back(it1->value, it1->weight,
                              it1->min_rank + next_min_rank2,
                              it1->max_rank + it2->PrevMaxRank());
        // Update next min rank 1.
        next_min_rank1 = it1->NextMinRank();
        ++it1;
      } else if (kCompFn(it2->value, it1->value)) {  // value1 > value2
        // Take value2 and use the last added value1 to compute
        // the min rank and the current value1 to compute the max rank.
        entries_.emplace_back(it2->value, it2->weight,
                              it2->min_rank + next_min_rank1,
                              it2->max_rank + it1->PrevMaxRank());
        // Update next min rank 2.
        next_min_rank2 = it2->NextMinRank();
        ++it2;
      } else {  // value1 == value2
        // Straight additive merger of the two entries into one.
        entries_.emplace_back(it1->value, it1->weight + it2->weight,
                              it1->min_rank + it2->min_rank,
                              it1->max_rank + it2->max_rank);
        // Update next min ranks for both.
        next_min_rank1 = it1->NextMinRank();
        next_min_rank2 = it2->NextMinRank();
        ++it1;
        ++it2;
      }
    }

    // Fill in any residual.
    while (it1 != base_entries.cend()) {
      entries_.emplace_back(it1->value, it1->weight,
                            it1->min_rank + next_min_rank2,
                            it1->max_rank + other_entries.back().max_rank);
      ++it1;
    }
    while (it2 != other_entries.cend()) {
      entries_.emplace_back(it2->value, it2->weight,
                            it2->min_rank + next_min_rank1,
                            it2->max_rank + base_entries.back().max_rank);
      ++it2;
    }
  }

  // Compresses buffer into desired size. The size specification is
  // considered a hint as we always keep the first and last elements and
  // maintain strict approximation error bounds.
  // The approximation error delta is taken as the max of either the requested
  // min error or 1 / size_hint.
  // After compression, the approximation error is guaranteed to increase
  // by no more than that error delta.
  // This algorithm is linear in the original size of the summary and is
  // designed to be cache-friendly.
  void Compress(int64 size_hint, double min_eps = 0) {
    // No-op if we're already within the size requirement.
    size_hint = std::max(size_hint, int64{2});
    if (entries_.size() <= size_hint) {
      return;
    }

    // First compute the max error bound delta resulting from this compression.
    double eps_delta = TotalWeight() * std::max(1.0 / size_hint, min_eps);

    // Compress elements ensuring approximation bounds and elements diversity
    // are both maintained.
    int64 add_accumulator = 0, add_step = entries_.size();
    auto write_it = entries_.begin() + 1, last_it = write_it;
    for (auto read_it = entries_.begin(); read_it + 1 != entries_.end();) {
      auto next_it = read_it + 1;
      while (next_it != entries_.end() && add_accumulator < add_step &&
             next_it->PrevMaxRank() - read_it->NextMinRank() <= eps_delta) {
        add_accumulator += size_hint;
        ++next_it;
      }
      if (read_it == next_it - 1) {
        ++read_it;
      } else {
        read_it = next_it - 1;
      }
      (*write_it++) = (*read_it);
      last_it = read_it;
      add_accumulator -= add_step;
    }
    // Write last element and resize.
    if (last_it + 1 != entries_.end()) {
      (*write_it++) = entries_.back();
    }
    entries_.resize(write_it - entries_.begin());
  }

  // To construct the boundaries we first run a soft compress over a copy
  // of the summary and retrieve the values.
  // The resulting boundaries are guaranteed to both contain at least
  // num_boundaries unique elements and maintain approximation bounds.
  std::vector<ValueType> GenerateBoundaries(int64 num_boundaries) const {
    std::vector<ValueType> output;
    if (entries_.empty()) {
      return output;
    }

    // Generate soft compressed summary.
    WeightedQuantilesSummary<ValueType, WeightType, CompareFn>
        compressed_summary;
    compressed_summary.BuildFromSummaryEntries(entries_);
    // Set an epsilon for compression that's at most 1.0 / num_boundaries
    // more than epsilon of original our summary since the compression operation
    // adds ~1.0/num_boundaries to final approximation error.
    float compression_eps = ApproximationError() + (1.0 / num_boundaries);
    compressed_summary.Compress(num_boundaries, compression_eps);

    // Return boundaries.
    output.reserve(compressed_summary.entries_.size());
    for (const auto& entry : compressed_summary.entries_) {
      output.push_back(entry.value);
    }
    return output;
  }

  // To construct the desired n-quantiles we repetitively query n ranks from the
  // original summary. The following algorithm is an efficient cache-friendly
  // O(n) implementation of that idea which avoids the cost of the repetitive
  // full rank queries O(nlogn).
  std::vector<ValueType> GenerateQuantiles(int64 num_quantiles) const {
    std::vector<ValueType> output;
    if (entries_.empty()) {
      return output;
    }
    num_quantiles = std::max(num_quantiles, int64{2});
    output.reserve(num_quantiles + 1);

    // Make successive rank queries to get boundaries.
    // We always keep the first (min) and last (max) entries.
    for (size_t cur_idx = 0, rank = 0; rank <= num_quantiles; ++rank) {
      // This step boils down to finding the next element sub-range defined by
      // r = (rmax[i + 1] + rmin[i + 1]) / 2 where the desired rank d < r.
      WeightType d_2 = 2 * (rank * entries_.back().max_rank / num_quantiles);
      size_t next_idx = cur_idx + 1;
      while (next_idx < entries_.size() &&
             d_2 >= entries_[next_idx].min_rank + entries_[next_idx].max_rank) {
        ++next_idx;
      }
      cur_idx = next_idx - 1;

      // Determine insertion order.
      if (next_idx == entries_.size() ||
          d_2 < entries_[cur_idx].NextMinRank() +
                    entries_[next_idx].PrevMaxRank()) {
        output.push_back(entries_[cur_idx].value);
      } else {
        output.push_back(entries_[next_idx].value);
      }
    }
    return output;
  }

  // Calculates current approximation error which should always be <= eps.
  double ApproximationError() const {
    if (entries_.empty()) {
      return 0;
    }

    WeightType max_gap = 0;
    for (auto it = entries_.cbegin() + 1; it < entries_.end(); ++it) {
      max_gap = std::max(max_gap,
                         std::max(it->max_rank - it->min_rank - it->weight,
                                  it->PrevMaxRank() - (it - 1)->NextMinRank()));
    }
    return static_cast<double>(max_gap) / TotalWeight();
  }

  ValueType MinValue() const {
    return !entries_.empty() ? entries_.front().value
                             : std::numeric_limits<ValueType>::max();
  }
  ValueType MaxValue() const {
    return !entries_.empty() ? entries_.back().value
                             : std::numeric_limits<ValueType>::max();
  }
  WeightType TotalWeight() const {
    return !entries_.empty() ? entries_.back().max_rank : 0;
  }
  int64 Size() const { return entries_.size(); }
  void Clear() { entries_.clear(); }
  const std::vector<SummaryEntry>& GetEntryList() const { return entries_; }

 private:
  // Comparison function.
  static constexpr decltype(CompareFn()) kCompFn = CompareFn();

  // Summary entries.
  std::vector<SummaryEntry> entries_;
};

template <typename ValueType, typename WeightType, typename CompareFn>
constexpr decltype(CompareFn())
    WeightedQuantilesSummary<ValueType, WeightType, CompareFn>::kCompFn;

}  // namespace quantiles
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_QUANTILES_WEIGHTED_QUANTILES_SUMMARY_H_
