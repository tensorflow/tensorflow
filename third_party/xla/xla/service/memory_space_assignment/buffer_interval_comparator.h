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

#include <cstdint>
#include <string>
#include <tuple>

#include "absl/container/flat_hash_map.h"
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
