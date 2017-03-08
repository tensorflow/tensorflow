/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_UTIL_STAT_SUMMARIZER_H_
#define TENSORFLOW_UTIL_STAT_SUMMARIZER_H_

#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <string>

#include <stdlib.h>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class GraphDef;
class StepStats;
class NodeExecStats;

template <typename ValueType, typename HighPrecisionValueType = double>
class Stat {
 public:
  void UpdateStat(ValueType v) {
    newest_ = v;
    max_ = std::max(v, max_);
    min_ = std::min(v, min_);
    ++count_;
    sum_ += v;
    squared_sum_ += static_cast<HighPrecisionValueType>(v) * v;
  }

  void Reset() { new (this) Stat<ValueType, HighPrecisionValueType>(); }

  bool empty() const { return count_ == 0; }

  ValueType newest() const { return newest_; }

  ValueType max() const { return max_; }

  ValueType min() const { return min_; }

  int64 count() const { return count_; }

  ValueType sum() const { return sum_; }

  HighPrecisionValueType squared_sum() const { return squared_sum_; }

  bool all_same() const { return (count_ == 0 || min_ == max_); }

  HighPrecisionValueType avg() const {
    return empty() ? std::numeric_limits<ValueType>::quiet_NaN()
                   : static_cast<HighPrecisionValueType>(sum_) / count_;
  }

  ValueType std_deviation() const {
    return all_same() ? 0 : sqrt(squared_sum_ / count_ - avg() * avg());
  }

  void OutputToStream(std::ostream* stream) const {
    if (empty()) {
      *stream << "count=0";
    } else if (all_same()) {
      *stream << "curr=" << newest_ << " count=" << count_;
      if (count_ > 1) *stream << "(all same)";
    } else {
      *stream << "curr=" << newest_ << " count=" << count_ << " min=" << min_
              << " max=" << max_ << " avg=" << avg()
              << " std=" << std_deviation();
    }
  }

  friend std::ostream& operator<<(std::ostream& stream,
                                  const Stat<ValueType>& stat) {
    stat.OutputToStream(&stream);
    return stream;
  }

 private:
  ValueType newest_ = 0;
  ValueType max_ = std::numeric_limits<ValueType>::min();
  ValueType min_ = std::numeric_limits<ValueType>::max();
  int64 count_ = 0;
  ValueType sum_ = 0;
  HighPrecisionValueType squared_sum_ = 0;
};

// A class intended to make performance analysis easier by collecting StepStats
// and showing in an easily understandable format where CPU time is being spent.
// See tensorflow/examples/android/jni/tensorflow_jni.cc for an example usage.
class StatSummarizer {
 public:
  explicit StatSummarizer(const tensorflow::GraphDef& tensorflow_graph);

  // Adds another run's StepStats output to the aggregate counts.
  void ProcessStepStats(const StepStats& step_stats);

  // Prints all the accumulated runtime stats in a tab-separated format which
  // can be pasted into a spreadsheet for further analysis.
  void PrintStepStats() const;

  // Summarizes all nodes' stat in the order of node names defined in the graph.
  std::string GetStatsByOrderOfNodeDefinitions() const;

  // Summarizes all nodes' stat in the order of nodes getting executed.
  std::string GetStatsByRunOrder() const;

  // Summarizes all nodes' stat in the order of top durations.
  // Will stop printing if either cdf_cutoff_ratio or num_max_nodes_to_print
  // is hit.
  std::string GetStatsByTopDurations(
      double cdf_cutoff_ratio = 1.0,
      int num_max_nodes_to_print = std::numeric_limits<int>::max()) const;

  void Reset() {
    run_total_micros_.Reset();
    timing_details_.clear();
  }

  // Returns number of runs.
  int num_runs() const { return run_total_micros_.count(); }

  // Returns stats of total microseconds spent by all nodes in each run.
  const Stat<int64>& run_total_us() const { return run_total_micros_; }

 private:
  struct Detail {
    int64 first_start_micros;
    int64 first_rel_end_micros;
    int64 total_micros;
  };

  enum struct SortingMetric {
    BY_TOTAL_DURATION,
    BY_RUN_ORDER,
  };

  std::string GetStatsBySorting(SortingMetric sorting_metric,
                                double cdf_cutoff_ratio,
                                int num_max_nodes_to_print) const;

  std::string HeaderString() const;
  std::string ColumnString(const std::string& name, const Detail& detail,
                           int64 cumulative_time_us_on_node) const;
  std::string ShortSummary() const;

  int64 first_node_start_micros_;
  Stat<int64> run_total_micros_;
  std::vector<string> nodes_in_def_order_;
  std::map<std::string, Detail> timing_details_;
  std::map<string, string> node_types_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_STAT_SUMMARIZER_H_
