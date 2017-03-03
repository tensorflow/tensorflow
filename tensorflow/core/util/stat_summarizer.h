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

#include <stdlib.h>

#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <string>

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
  int64 count_ = 0;
  ValueType sum_ = 0;
  HighPrecisionValueType squared_sum_ = 0;
};

// Used to control the output of the statistics summarizer;
class StatSummarizerOptions {
 public:
  StatSummarizerOptions()
      : show_run_order(true),
        run_order_limit(0),
        show_time(true),
        time_limit(10),
        show_memory(true),
        memory_limit(10),
        show_type(true),
        show_summary(true) {}

  bool show_run_order;
  int run_order_limit;
  bool show_time;
  int time_limit;
  bool show_memory;
  int memory_limit;
  bool show_type;
  bool show_summary;
};

// A StatSummarizer assists in performance analysis of Graph executions.
//
// It summarizes time spent executing (on GPU/CPU), memory used etc. across
// multiple executions of a single Graph from the StepStats collected during
// graph execution.
//
// See tensorflow/tools/benchmark/benchmark_model.cc for an example usage.
class StatSummarizer {
 public:
  enum SortingMetric {
    BY_NAME,
    BY_RUN_ORDER,
    BY_TIME,
    BY_MEMORY,
    BY_TYPE,
  };

  explicit StatSummarizer(const StatSummarizerOptions& options);

  // Deprecated: Use StatSummarizer(const StatSummarizerOptions&) instead. The
  // GraphDef is not needed by the StatSummarizer.
  explicit StatSummarizer(const tensorflow::GraphDef& tensorflow_graph);

  // Adds another run's StepStats output to the aggregate counts.
  void ProcessStepStats(const StepStats& step_stats);

  // Returns a string detailing the accumulated runtime stats in a tab-separated
  // format which can be pasted into a spreadsheet for further analysis.
  std::string GetOutputString() const;

  std::string ShortSummary() const;

  // Prints the string returned by GetOutputString().
  void PrintStepStats() const;

  // Prints the output tensor sizes and types for each node.
  void PrintOutputs() const;

  std::string GetStatsByNodeType() const;

  std::string GetStatsByMetric(const string& title,
                               SortingMetric sorting_metric,
                               int num_stats) const;

  void Reset() {
    run_total_us_.Reset();
    memory_.Reset();
    details_.clear();
  }

  // Returns number of runs.
  int num_runs() const { return run_total_us_.count(); }

  // Returns stats of total microseconds spent by all nodes in each run.
  const Stat<int64>& run_total_us() const { return run_total_us_; }

 private:
  struct Detail {
    string name;
    string type;
    int64 run_order;
    Stat<int64> start_us;
    Stat<int64> rel_end_us;
    Stat<int64> mem_used;
    std::vector<TensorDescription> outputs;
  };

  void Validate(const Detail* detail, const NodeExecStats& ns) const;

  void OrderNodesByMetric(SortingMetric sorting_metric,
                          std::vector<const Detail*>* details) const;

  std::string HeaderString(const string& title) const;
  std::string ColumnString(const Detail& detail,
                           const int64 cumulative_stat_on_node,
                           const Stat<int64>& stat) const;

  Stat<int64> run_total_us_;
  Stat<int64> memory_;

  std::map<std::string, Detail> details_;
  StatSummarizerOptions options_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_STAT_SUMMARIZER_H_
