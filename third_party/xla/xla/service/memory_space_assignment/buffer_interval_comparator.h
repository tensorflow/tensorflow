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

// A specialization for the cases were we can build keys for comparisons.
//
// Instead of supplying a binary LessThan function, the user only needs to
// supply a unary BuildComparisonKey function instead.
//
// This class caches HloValue -> ComparisonKey, so BuildComparisonKey is only
// called once per MsaBufferInterval.
//
// From our tests on real-world cases, we see ~100x calls to
// GetComparisonKeyCached compared to BuildComparisonKey, which justifies the
// caching.
template <typename ComparisonKeyT>
class BufferIntervalKeyedComparator : public BufferIntervalComparator {
 public:
  using ComparisonKey = ComparisonKeyT;
  bool LessThan(const MsaBufferInterval& lhs,
                const MsaBufferInterval& rhs) final {
    return GetComparisonKeyCached(lhs) < GetComparisonKeyCached(rhs);
  }
  const ComparisonKey& GetComparisonKeyCached(
      const MsaBufferInterval& buffer_interval) {
    auto [iter, newly_inserted] =
        buffer_to_comparison_key_.try_emplace(buffer_interval.buffer);
    if (newly_inserted) {
      iter->second = BuildComparisonKey(buffer_interval);
    } else {
      // DO_NOT_SUBMIT
      const auto& curr = BuildComparisonKey(buffer_interval);
      CHECK(iter->second == curr)
          << "\nprev="
          << absl::StrCat("[ ", absl::StrJoin(iter->second, ", "), " ]")
          << "\ncurr=" << absl::StrCat("[ ", absl::StrJoin(curr, ", "), " ]");
    }
    return iter->second;
  }

 private:
  virtual ComparisonKey BuildComparisonKey(
      const MsaBufferInterval& buffer_interval) = 0;

  absl::flat_hash_map<const HloValue*, ComparisonKey> buffer_to_comparison_key_;
};

// A BufferIntervalComparator that utilizes MemoryBoundedness as its primary
// sorting criteria.
//
// See the value returned by DescribeComparisonCriteria() for the meaning of
// each tuple element.
class MemoryBoundednessBufferIntervalComparator
    : public BufferIntervalKeyedComparator<
          std::tuple<int64_t, float, int64_t, int64_t, int64_t, int64_t,
                     BufferValue::Id> > {
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
 private:
  ComparisonKey BuildComparisonKey(
      const MsaBufferInterval& buffer_interval) override;
  int64_t GetLatestUseTime(const MsaBufferInterval& buffer_interval);
  const CostAnalysis& cost_analysis_;
  CostAnalysis::Cache* cost_analysis_cache_;

  // Config to override alternate memory assignment sorting order for filtered
  // buffers.
  MsaSortOrderOverrides msa_sort_order_overrides_;
};

// The default BufferIntervalComparator used for cross-program prefetching.
//
// See the value returned by DescribeComparisonCriteria() for the meaning of
// each tuple element.
class DefaultCrossProgramPrefetchBufferIntervalComparator
    : public BufferIntervalKeyedComparator<
          std::tuple<int64_t, int64_t, int64_t, int64_t, BufferValue::Id> > {
 public:
  explicit DefaultCrossProgramPrefetchBufferIntervalComparator(
      const HloLiveRange& hlo_live_range,
      const MsaSortOrderOverrides& msa_sort_order_overrides);

  ~DefaultCrossProgramPrefetchBufferIntervalComparator() override = default;

  std::string DescribeComparisonCriteria() const override;
  std::string CriteriaToString(
      const MsaBufferInterval& buffer_interval) override;
 private:
  ComparisonKey BuildComparisonKey(
      const MsaBufferInterval& buffer_interval) override;

  const HloLiveRange& hlo_live_range_;
  const MsaSortOrderOverrides& msa_sort_order_overrides_;
};

}  // namespace memory_space_assignment
}  // namespace xla

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_BUFFER_INTERVAL_COMPARATOR_H_
