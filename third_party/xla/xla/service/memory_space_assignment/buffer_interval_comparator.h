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
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
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

// Interface for providing the tuple used to sort MsaBufferIntervals.
//
// This abstraction separates the definition of a buffer's sorting criteria from
// the comparator mechanism itself. This is designed to enable an Agent via an
// API to seamlessly inject custom tensor sorting policies. An Agent
// implementing this interface only needs to return a scoring tuple, while
// ProviderBufferIntervalComparator automatically handles the `<` comparison
// between pairs of buffer intervals.
template <typename TupleType>
class BufferSorterProvider {
 public:
  virtual ~BufferSorterProvider() = default;
  // A logging string explaining the sorting criteria. E.g., [ -size, offset ]
  virtual std::string DescribeComparisonCriteria() const = 0;
  // Returns the full comparison tuple used to sort buffer intervals.
  virtual TupleType GetBufferSortingTuple(
      const MsaBufferInterval& buffer_interval) = 0;
};

// A BufferSorterProvider that utilizes memory boundedness as its primary
// sorting criterion.
//
// Specifically, the sorting tuple is defined as follows:
// 1. int64_t: Override priority (higher priority sorts earlier).
// 2. float: Inverse memory boundedness (-1.0 * memory_boundedness; higher
//    boundedness sorts earlier).
// 3. int64_t: Inverse buffer size (-1 * size; larger buffers sort earlier).
// 4. int64_t: Inverse buffer duration (start - end; longer duration sorts
//    earlier).
// 5. int64_t: Latest use time (buffers used later sort earlier).
// 6. int64_t: Buffer start time (buffers starting earlier sort earlier).
// 7. BufferValue::Id: Buffer ID (deterministic tie-breaker).
class MemoryBoundednessBufferSorterProvider
    : public BufferSorterProvider<
          std::tuple<int64_t, float, int64_t, int64_t, int64_t, int64_t,
                     BufferValue::Id>> {
 public:
  using ComparisonTuple = std::tuple<int64_t, float, int64_t, int64_t, int64_t,
                                     int64_t, BufferValue::Id>;

  MemoryBoundednessBufferSorterProvider(
      const CostAnalysis& cost_analysis,
      CostAnalysis::Cache* cost_analysis_cache,
      MsaSortOrderOverrides msa_sort_order_overrides = {});

  ~MemoryBoundednessBufferSorterProvider() override = default;

  std::string DescribeComparisonCriteria() const override;
  ComparisonTuple GetBufferSortingTuple(
      const MsaBufferInterval& buffer_interval) override;

 private:
  int64_t GetLatestUseTime(const MsaBufferInterval& buffer_interval);

  absl::flat_hash_map<const HloValue*, int64_t> buffer_to_latest_use_;
  const CostAnalysis& cost_analysis_;
  CostAnalysis::Cache* cost_analysis_cache_;
  MsaSortOrderOverrides msa_sort_order_overrides_;
};

// A BufferIntervalComparator that utilizes a BufferSorterProvider as its
// sorting criteria.
template <typename TupleType>
class ProviderBufferIntervalComparator : public BufferIntervalComparator {
 public:
  explicit ProviderBufferIntervalComparator(
      BufferSorterProvider<TupleType>& buffer_sorter_provider)
      : BufferIntervalComparator(),
        buffer_sorter_provider_(buffer_sorter_provider) {}

  ~ProviderBufferIntervalComparator() override = default;

  std::string DescribeComparisonCriteria() const override {
    return buffer_sorter_provider_.DescribeComparisonCriteria();
  }

  std::string CriteriaToString(
      const MsaBufferInterval& buffer_interval) override {
    return absl::StrCat(
        "[ ",
        absl::StrJoin(
            buffer_sorter_provider_.GetBufferSortingTuple(buffer_interval),
            ", "),
        " ]");
  }

  bool LessThan(const MsaBufferInterval& lhs,
                const MsaBufferInterval& rhs) override {
    return buffer_sorter_provider_.GetBufferSortingTuple(lhs) <
           buffer_sorter_provider_.GetBufferSortingTuple(rhs);
  }

 private:
  BufferSorterProvider<TupleType>& buffer_sorter_provider_;
};

template <typename ProviderType>
ProviderBufferIntervalComparator(ProviderType&)
    -> ProviderBufferIntervalComparator<typename ProviderType::ComparisonTuple>;

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
