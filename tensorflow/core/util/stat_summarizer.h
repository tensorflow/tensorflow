/* Copyright 2016 Google Inc. All Rights Reserved.

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
#include <list>
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

static const int kNumValues = 1000;

template <typename T>
class Stat {
 public:
  Stat<T>() { Reset(); }

  void Reset() {
    values_.clear();
    min_value_ = std::numeric_limits<T>::max();
    max_value_ = std::numeric_limits<T>::min();
    avg_value_ = 0;
    current_value_ = 0;
    std_deviation_ = 0;
  }

  std::list<T> values_;
  T min_value_;
  T max_value_;
  double avg_value_;
  T current_value_;
  double std_deviation_;

  void UpdateStat(const T val) {
    current_value_ = val;
    values_.push_front(val);
    while (values_.size() > kNumValues) {
      values_.pop_back();
    }

    T total = 0;
    for (const T curr_val : values_) {
      min_value_ = std::min(min_value_, curr_val);
      max_value_ = std::max(max_value_, curr_val);
      total += curr_val;
    }
    avg_value_ = static_cast<double>(total) / values_.size();

    double sqr_total = 0.0;
    for (const T curr_val : values_) {
      const double delta = avg_value_ - curr_val;
      sqr_total += delta * delta;
    }
    std_deviation_ = std::sqrt(sqr_total / values_.size());
  }

  friend std::ostream& operator<<(std::ostream& stream, const Stat<T>& stat) {
    stream << "curr=" << stat.current_value_ << " min=" << stat.min_value_
           << " max=" << stat.max_value_
           << " avg=" << static_cast<int64>(stat.avg_value_)
           << " stddev=" << static_cast<int64>(stat.std_deviation_);
    return stream;
  }
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
  void PrintStepStats();

  void Reset() {
    num_runs_ = 0;

    run_total_us_.Reset();

    timing_total_us_ = 0;
    timing_totals_.clear();
  }

 private:
  void PrintHeaders();
  void PrintColumns(const char* name, const char* op, const double time_ms,
                    const double percentage);

  Stat<int64> run_total_us_;

  int num_runs_ = 0;
  int64 timing_total_us_ = 0;
  std::vector<string> nodes_;
  std::map<string, int64> timing_totals_;
  std::map<string, string> node_types_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_STAT_SUMMARIZER_H_
