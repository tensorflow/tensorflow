/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_PROFILE_TIME_BREAKDOWN_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_PROFILE_TIME_BREAKDOWN_H_

#include <cstdint>
#include <initializer_list>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/math_utils.h"

namespace tensorflow {
namespace profiler {

// Allows accumulating time spent in different HLO instruction categories to
// breakdown the total profile time and compute metrics of interest.
class ProfileTimeBreakdown {
 public:
  // Category should be the operator category disambiguated by xprof instead of
  // the original category from XLA.
  // For a correct time breakdown, we need to use the self time of operators,
  // instead of total time to avoid double counting. Note that for leaf ops,
  // self time and total time are the same.
  void IncrementCategoryTimePs(absl::string_view category,
                               uint64_t self_time_ps) {
    time_ps_by_category_[category] += self_time_ps;
    total_time_ps_ += self_time_ps;
  }

  // Profile time cannot be smaller than the total time in all categories.
  // If combining profiles across multiple cores, profile time should be the
  // profiling duration multiplied by the number of cores that were profiled.
  // go/autograppler_profile_time
  void SetProfileTimePs(uint64_t profile_time_ps) {
    DCHECK_LE(total_time_ps_, profile_time_ps);
    profile_time_ps_ = profile_time_ps;
  }

  // Breaks down "sparsecorev0 infeed" into two components:
  // 1) "sparsecorev0 infeed wait": Time spent waiting on the SparseCoreV0.
  // 2) "sparsecorev0 infeed transform": Time spent transforming activations in
  //    SparseCoreV0 layout into XLA layout.
  // Even though 2) is part of the overall embedding computation, it is time
  // spent doing work on the TensorCore.
  void BreakdownSparseCoreV0Infeed();

  // Duty cycle is the fraction of time an accelerator is being actively used.
  // go/accelerator-metrics-definitions#common-accelerator-metrics
  // go/ag-tpu-duty-cycle
  double DutyCycle() const { return TimeFraction(OnDutyTimePs()); }

  double IdleFraction() const { return TimeFraction(IdleTimePs()); }

  double InfeedFraction() const {
    return CategoryFraction(tsl::profiler::kHloInfeed);
  }

  double OutfeedFraction() const {
    return CategoryFraction(tsl::profiler::kHloOutfeed);
  }

  double SparseCoreV0InfeedFraction() const {
    return CategoriesFraction({tsl::profiler::kHloSparseCoreV0Infeed,
                               tsl::profiler::kHloSparseCoreV0InfeedWait,
                               tsl::profiler::kHloSparseCoreV0InfeedTransform});
  }

  double SparseCoreV0OutfeedFraction() const {
    return CategoryFraction(tsl::profiler::kHloSparseCoreV0Outfeed);
  }

  double AllReduceFraction() const {
    return CategoryFraction(tsl::profiler::kHloAllReduce);
  }

  double AllReduceFusionFraction() const {
    return CategoryFraction(tsl::profiler::kHloAllReduceFusion);
  }

  double SendRecvFraction() const {
    return CategoriesFraction(
        {tsl::profiler::kHloSend, tsl::profiler::kHloSendDone,
         tsl::profiler::kHloRecv, tsl::profiler::kHloRecvDone});
  }

  double HostSendRecvFraction() const {
    return CategoriesFraction(
        {tsl::profiler::kHloHostSend, tsl::profiler::kHloHostSendDone,
         tsl::profiler::kHloHostRecv, tsl::profiler::kHloHostRecvDone});
  }

  double CategoriesFraction(
      const std::initializer_list<absl::string_view>& categories) const {
    return TimeFraction(CategoriesTimePs(categories));
  }

  double CategoryFraction(absl::string_view category) const {
    return TimeFraction(CategoryTimePs(category));
  }

  uint64_t ProfileTimePs() const { return profile_time_ps_; }

  uint64_t TotalTimePs() const { return total_time_ps_; }

  uint64_t IdleTimePs() const { return profile_time_ps_ - total_time_ps_; }

  uint64_t OnDutyTimePs() const { return profile_time_ps_ - OffDutyTimePs(); }

  uint64_t OffDutyTimePs() const {
    return IdleTimePs() +
           CategoriesTimePs(
               {tsl::profiler::kHloInfeed, tsl::profiler::kHloOutfeed,
                tsl::profiler::kHloHostSend, tsl::profiler::kHloHostSendDone,
                tsl::profiler::kHloHostRecv, tsl::profiler::kHloHostRecvDone,
                tsl::profiler::kHloMegacoreFusion});
  }

  uint64_t InfeedTimePs() const {
    return CategoryTimePs(tsl::profiler::kHloInfeed);
  }

  uint64_t OutfeedTimePs() const {
    return CategoryTimePs(tsl::profiler::kHloOutfeed);
  }

  uint64_t SparseCoreV0InfeedWaitTimePs() const {
    return CategoryTimePs(tsl::profiler::kHloSparseCoreV0InfeedWait);
  }

  uint64_t SparseCoreV0InfeedTransformTimePs() const {
    return CategoryTimePs(tsl::profiler::kHloSparseCoreV0InfeedTransform);
  }

  uint64_t SparseCoreV0OutfeedTimePs() const {
    return CategoryTimePs(tsl::profiler::kHloSparseCoreV0Outfeed);
  }

  uint64_t AllReduceOrAllToAllTimePs() const {
    return CategoriesTimePs({tsl::profiler::kHloAllReduce,
                             tsl::profiler::kHloAllReduceFusion,
                             tsl::profiler::kHloAllToAll});
  }

  uint64_t SendTimePs() const {
    return CategoriesTimePs(
        {tsl::profiler::kHloSend, tsl::profiler::kHloSendDone});
  }

  uint64_t RecvTimePs() const {
    return CategoriesTimePs(
        {tsl::profiler::kHloRecv, tsl::profiler::kHloRecvDone});
  }

  uint64_t HostSendTimePs() const {
    return CategoriesTimePs(
        {tsl::profiler::kHloHostSend, tsl::profiler::kHloHostSendDone});
  }

  uint64_t HostRecvTimePs() const {
    return CategoriesTimePs(
        {tsl::profiler::kHloHostRecv, tsl::profiler::kHloHostRecvDone});
  }

  // Megacore fusion runs different operations on each core, e.g., a convolution
  // on one core and an all-reduce on the other core. In a trace, megacore
  // fusion is the parent operation, and its self time is the time that the core
  // executing the faster operation waits for the core executing the slower
  // operation to reach the synchronization point.
  uint64_t MegacoreFusionTimePs() const {
    return CategoryTimePs(tsl::profiler::kHloMegacoreFusion);
  }

  uint64_t HighFlopsComputeTimePs() const {
    return CategoriesTimePs({tsl::profiler::kHloConvolution,
                             tsl::profiler::kHloConvolutionBaseDilated,
                             tsl::profiler::kHloConvolutionWindowDilated,
                             tsl::profiler::kHloConvolutionFusion,
                             tsl::profiler::kHloOutputFusion});
  }

  // Calculated according to the "TC busy time" defined in go/tpu_kpis
  uint64_t TensorCoreBusyTimePs() const {
    return profile_time_ps_ - OffDutyTimePs() - SparseCoreV0InfeedWaitTimePs();
  }

  uint64_t CategoriesTimePs(
      const std::initializer_list<absl::string_view>& categories) const {
    uint64_t time_ps = 0;
    for (auto category : categories) {
      time_ps += CategoryTimePs(category);
    }
    return time_ps;
  }

  uint64_t CategoryTimePs(absl::string_view category) const {
    auto iter = time_ps_by_category_.find(category);
    return (iter == time_ps_by_category_.end()) ? 0 : iter->second;
  }

  template <typename Map>
  void ComputeCategoryFractions(Map& category_fractions) {
    for (const auto& [category, time_ps] : time_ps_by_category_) {
      category_fractions[category] = TimeFraction(time_ps);
    }
  }

  std::string DebugString() const;

 private:
  // Overwrites the time attributed to the given category.
  void SetCategoryTimePs(absl::string_view category, uint64_t time_ps);

  // Removes and returns the time attributed to the given category.
  uint64_t PopCategoryTimePs(absl::string_view category);

  double TimeFraction(uint64_t time_ps) const {
    return tsl::profiler::SafeDivide(time_ps, profile_time_ps_);
  }

  absl::flat_hash_map<std::string, uint64_t> time_ps_by_category_;
  uint64_t total_time_ps_ = 0;  // Sum of values in time_ps_by_category_.
  uint64_t profile_time_ps_ = 0;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_PROFILE_TIME_BREAKDOWN_H_
