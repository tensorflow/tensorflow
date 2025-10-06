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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_BUFFER_INTERVAL_COMPARATOR_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_BUFFER_INTERVAL_COMPARATOR_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/buffer_value.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/service/memory_space_assignment/utils.h"

namespace xla {
namespace memory_space_assignment {

using MsaBufferIntervalCompare =
    GlobalDecreasingSizeBestFitHeap<HloValue>::BufferIntervalCompare;

// The MsaBufferInterval sorting interface that MemorySpaceAssignment expects.
class BufferIntervalComparator {
 public:
  virtual ~BufferIntervalComparator() = default;

  // A logging string explaining the sorting criteria. E.g., [ -size, offset ]
  // indicates we sort (desc) size, then (asc) offset.
  virtual std::string DescribeComparisonCriteria() const = 0;

  // A logging string containing the values used to sort buffer_interval.
  // E.g., we might return [ -1024, 100 ], if the criteria is [ -size,
  // offset ].
  virtual std::string CriteriaToString(
      const MsaBufferInterval& buffer_interval) = 0;

  // comparator.LessThan(lhs, rhs) will be used for MsaBufferIntervalCompare.
  virtual bool LessThan(const MsaBufferInterval& lhs,
                        const MsaBufferInterval& rhs) = 0;

  // Used to create a functor that can be passed to a method like std::sort.
  // E.g., absl::c_sort(v, comparator.GetComparisonFunctor());
  MsaBufferIntervalCompare GetComparisonFunctor() {
    return [this](const MsaBufferInterval& lhs, const MsaBufferInterval& rhs) {
      return LessThan(lhs, rhs);
    };
  }

  // Sorts the buffer intervals prioritizing results from the
  // most_significant_compare functor first.
  //
  // The most_significant_compare functor should return a negative number if
  // lhs should be sorted before rhs, a positive number if rhs should be sorted
  // before lhs, and 0 otherwise.
  virtual void Sort(absl::AnyInvocable<int(const MsaBufferInterval&,
                                           const MsaBufferInterval&)>
                        most_significant_compare,
                    std::vector<MsaBufferInterval>& buffer_intervals) {
    auto less_than = GetComparisonFunctor();
    absl::c_sort(buffer_intervals, [&less_than, &most_significant_compare](
                                       const MsaBufferInterval& a,
                                       const MsaBufferInterval& b) {
      int most_significant_compare_result = most_significant_compare(a, b);
      if (most_significant_compare_result != 0) {
        return most_significant_compare_result < 0;
      }
      return less_than(a, b);
    });
  }

 protected:
  BufferIntervalComparator() = default;
};

// A BufferIntervalComparator that utilizes MemoryBoundedness as its primary
// sorting criteria.
//
// This comparator caches HloValues -> latest use time.
class MemoryBoundednessBufferIntervalComparator
    : public BufferIntervalComparator {
 public:
  MemoryBoundednessBufferIntervalComparator(
      const CostAnalysis& cost_analysis,
      CostAnalysis::Cache* cost_analysis_cache);

  MemoryBoundednessBufferIntervalComparator(
      const CostAnalysis& cost_analysis,
      CostAnalysis::Cache* cost_analysis_cache,
      MsaSortOrderOverrides msa_sort_order_overrides);

  ~MemoryBoundednessBufferIntervalComparator() override = default;

  std::string DescribeComparisonCriteria() const override;
  std::string CriteriaToString(
      const MsaBufferInterval& buffer_interval) override;
  bool LessThan(const MsaBufferInterval& lhs,
                const MsaBufferInterval& rhs) override;

  // This is an optimized implementation, equivalent to the base class's Sort()
  // method.
  void Sort(absl::AnyInvocable<int(const MsaBufferInterval&,
                                   const MsaBufferInterval&)>
                most_significant_compare,
            std::vector<MsaBufferInterval>& buffer_intervals) override {
    const size_t num_buffer_intervals = buffer_intervals.size();
    std::vector<std::pair<ComparisonTuple, MsaBufferInterval*>>
        sorted_comparison_tuples_and_buffer_intervals;
    sorted_comparison_tuples_and_buffer_intervals.reserve(num_buffer_intervals);
    for (MsaBufferInterval& buffer_interval : buffer_intervals) {
      auto& comparison_tuple_and_buffer_interval =
          sorted_comparison_tuples_and_buffer_intervals.emplace_back();
      comparison_tuple_and_buffer_interval.first = GetTuple(buffer_interval);
      comparison_tuple_and_buffer_interval.second = &buffer_interval;
    }
    absl::c_sort(
        sorted_comparison_tuples_and_buffer_intervals,
        [&most_significant_compare](
            const std::pair<ComparisonTuple, const MsaBufferInterval*>& a,
            const std::pair<ComparisonTuple, const MsaBufferInterval*>& b) {
          int most_significant_compare_result =
              most_significant_compare(*a.second, *b.second);
          if (most_significant_compare_result != 0) {
            return most_significant_compare_result < 0;
          }
          return a.first < b.first;
        });
    std::vector<MsaBufferInterval> sorted_buffer_intervals;
    sorted_buffer_intervals.reserve(num_buffer_intervals);
    for (const auto& entry : sorted_comparison_tuples_and_buffer_intervals) {
      sorted_buffer_intervals.push_back(std::move(*entry.second));
    }
    buffer_intervals = std::move(sorted_buffer_intervals);
  }

 private:
  // See the value returned by DescribeComparisonCriteria() for the meaning of
  // each tuple element.
  using ComparisonTuple = std::tuple<int64_t, float, int64_t, int64_t, int64_t,
                                     int64_t, BufferValue::Id>;

  ComparisonTuple GetTuple(const MsaBufferInterval& buffer_interval);
  int64_t GetLatestUseTime(const MsaBufferInterval& buffer_interval);
  absl::flat_hash_map<const HloValue*, int64_t> buffer_to_latest_use_;
  const CostAnalysis& cost_analysis_;
  CostAnalysis::Cache* cost_analysis_cache_;

  // Config to override alternate memory assignment sorting order for filtered
  // buffers.
  MsaSortOrderOverrides msa_sort_order_overrides_;
};

// The default BufferIntervalComparator used for cross-program prefetching.
//
// This class caches HloValue -> {latest use, cumulative use size }.
class DefaultCrossProgramPrefetchBufferIntervalComparator
    : public BufferIntervalComparator {
 public:
  explicit DefaultCrossProgramPrefetchBufferIntervalComparator(
      const HloLiveRange& hlo_live_range,
      const MsaSortOrderOverrides& msa_sort_order_overrides);

  ~DefaultCrossProgramPrefetchBufferIntervalComparator() override = default;

  std::string DescribeComparisonCriteria() const override;
  std::string CriteriaToString(
      const MsaBufferInterval& buffer_interval) override;
  bool LessThan(const MsaBufferInterval& lhs,
                const MsaBufferInterval& rhs) override;

 private:
  // See the value returned by DescribeComparisonCriteria() for the meaning of
  // each tuple element.
  using ComparisonTuple =
      std::tuple<int64_t, int64_t, int64_t, int64_t, BufferValue::Id>;

  struct AdditionalSortData {
    int64_t latest_use = 0;
    int64_t cumulative_use_size = 0;
  };

  ComparisonTuple GetTuple(const MsaBufferInterval& buffer_interval);

  absl::flat_hash_map<const HloValue*, AdditionalSortData>
      additional_sort_data_;
  const HloLiveRange& hlo_live_range_;
  const MsaSortOrderOverrides& msa_sort_order_overrides_;
};

}  // namespace memory_space_assignment
}  // namespace xla

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_BUFFER_INTERVAL_COMPARATOR_H_
