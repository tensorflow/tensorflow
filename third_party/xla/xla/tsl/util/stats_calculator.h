/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_UTIL_STATS_CALCULATOR_H_
#define XLA_TSL_UTIL_STATS_CALCULATOR_H_

#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "xla/tsl/util/stat_summarizer_options.h"

namespace tsl {

template <typename ValueType, typename HighPrecisionValueType = double>
class Stat {
 public:
  void UpdateStat(ValueType v) {
    if (count_ == 0) {
      first_ = v;
    }

    newest_ = v;
    max_ = std::max(v, max_);
    min_ = std::min(v, min_);
    ++count_;
    sum_ += v;
    squared_sum_ += static_cast<HighPrecisionValueType>(v) * v;
  }

  void Reset() { new (this) Stat<ValueType, HighPrecisionValueType>(); }

  bool empty() const { return count_ == 0; }

  ValueType first() const { return first_; }

  ValueType newest() const { return newest_; }

  ValueType max() const { return max_; }

  ValueType min() const { return min_; }

  int64_t count() const { return count_; }

  ValueType sum() const { return sum_; }

  HighPrecisionValueType squared_sum() const { return squared_sum_; }

  bool all_same() const { return (count_ == 0 || min_ == max_); }

  HighPrecisionValueType avg() const {
    return empty() ? std::numeric_limits<ValueType>::quiet_NaN()
                   : static_cast<HighPrecisionValueType>(sum_) / count_;
  }

  // Returns sample variance.
  ValueType sample_variance() const {
    return all_same()
               ? 0
               : (squared_sum_ - std::pow(sum_, 2.0) / count_) / (count_ - 1);
  }

  // Returns population variance.
  ValueType variance() const {
    return all_same() ? 0 : (squared_sum_ / count_) - (avg() * avg());
  }

  // Returns population stddev.
  ValueType std_deviation() const {
    return all_same() ? 0 : std::sqrt(variance());
  }

  void OutputToStream(std::ostream* stream) const {
    if (empty()) {
      *stream << "count=0";
    } else if (all_same()) {
      *stream << "count=" << count_ << " curr=" << newest_;
      if (count_ > 1) *stream << "(all same)";
    } else {
      *stream << "count=" << count_ << " first=" << first_
              << " curr=" << newest_ << " min=" << min_ << " max=" << max_
              << " avg=" << avg() << " std=" << std_deviation();
    }
  }

  friend std::ostream& operator<<(std::ostream& stream,
                                  const Stat<ValueType>& stat) {
    stat.OutputToStream(&stream);
    return stream;
  }

 private:
  ValueType first_ = 0;
  ValueType newest_ = 0;
  ValueType max_ = std::numeric_limits<ValueType>::min();
  ValueType min_ = std::numeric_limits<ValueType>::max();
  int64_t count_ = 0;
  ValueType sum_ = 0;
  HighPrecisionValueType squared_sum_ = 0;
};

// A `StatWithPercentiles` inherited from `Stat`, also keeps track of the
// values added and can be used to compute the percentile values.
template <typename ValueType, typename HighPrecisionValueType = double>
class StatWithPercentiles : public Stat<ValueType, HighPrecisionValueType> {
 public:
  void UpdateStat(ValueType v) {
    Stat<ValueType, HighPrecisionValueType>::UpdateStat(v);
    values_.push_back(v);
  }

  // Returns the percentile value.
  ValueType percentile(int percentile) const {
    if (percentile < 0 || percentile > 100 || values_.empty()) {
      return std::numeric_limits<ValueType>::quiet_NaN();
    }
    std::vector<ValueType> values = values_;
    if (percentile == 100) {
      return values[values.size() - 1];
    } else {
      std::nth_element(values.begin(),
                       values.begin() + values.size() * percentile / 100,
                       values.end());
      return values[values.size() * percentile / 100];
    }
  }

  void OutputToStream(std::ostream* stream) const {
    Stat<ValueType, HighPrecisionValueType>::OutputToStream(stream);
    *stream << " p5=" << percentile(5) << " median=" << percentile(50)
            << " p95=" << percentile(95);
  }

 private:
  std::vector<ValueType> values_;
};

// A StatsCalculator assists in performance analysis of Graph executions.
//
// It summarizes time spent executing (on GPU/CPU), memory used etc for
// graph execution.
//
// For example usage see StatsSummarizer.
class StatsCalculator {
 public:
  enum SortingMetric {
    BY_NAME,
    BY_RUN_ORDER,
    BY_TIME,
    BY_MEMORY,
    BY_TYPE,
  };

  explicit StatsCalculator(const StatSummarizerOptions& options);

  // Returns a string detailing the accumulated runtime stats in a tab-separated
  // format which can be pasted into a spreadsheet for further analysis.
  std::string GetOutputString() const;

  std::string GetShortSummary() const;

  void ComputeStatsByType(
      std::map<std::string, int64_t>* node_type_map_count,
      std::map<std::string, int64_t>* node_type_map_time,
      std::map<std::string, int64_t>* node_type_map_memory,
      std::map<std::string, int64_t>* node_type_map_times_called,
      int64_t* accumulated_us) const;

  std::string GetStatsByNodeType() const;

  std::string GetStatsByMetric(const std::string& title,
                               SortingMetric sorting_metric,
                               int num_stats) const;

  // Returns number of runs.
  int num_runs() const { return static_cast<int>(run_total_us_.count()); }

  // Returns stats of total microseconds spent by all nodes in each run.
  const Stat<int64_t>& run_total_us() const { return run_total_us_; }

  void UpdateRunTotalUs(int64_t run_total_us) {
    run_total_us_.UpdateStat(run_total_us);
  }

  void UpdateMemoryUsed(int64_t memory) { memory_.UpdateStat(memory); }

  struct Detail {
    std::string name;
    std::string type;
    int64_t run_order;
    Stat<int64_t> elapsed_time;
    Stat<int64_t> mem_used;
    int64_t times_called;
  };

  const std::map<std::string, Detail>& GetDetails() const { return details_; }

  void AddNodeStats(const std::string& name, const std::string& type,
                    int64_t run_order, int64_t rel_end_us, int64_t mem_used);

 private:
  void OrderNodesByMetric(SortingMetric sorting_metric,
                          std::vector<const Detail*>* details) const;

  std::string HeaderString(const std::string& title) const;
  std::string ColumnString(const Detail& detail,
                           const int64_t cumulative_stat_on_node,
                           const Stat<int64_t>& stat) const;

  Stat<int64_t> run_total_us_;
  Stat<int64_t> memory_;

  std::map<std::string, Detail> details_;
  StatSummarizerOptions options_;
};

}  // namespace tsl

#endif  // XLA_TSL_UTIL_STATS_CALCULATOR_H_
