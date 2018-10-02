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

#ifndef TENSORFLOW_CORE_UTIL_STAT_SUMMARIZER_H_
#define TENSORFLOW_CORE_UTIL_STAT_SUMMARIZER_H_

#include <stdlib.h>

#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/stat_summarizer_options.h"
#include "tensorflow/core/util/stats_calculator.h"

namespace tensorflow {

class GraphDef;
class StepStats;
class NodeExecStats;

// A StatSummarizer assists in performance analysis of Graph executions.
//
// It summarizes time spent executing (on GPU/CPU), memory used etc. across
// multiple executions of a single Graph from the StepStats collected during
// graph execution.
//
// See tensorflow/tools/benchmark/benchmark_model.cc for an example usage.
class StatSummarizer {
 public:
  explicit StatSummarizer(const StatSummarizerOptions& options);

  // Deprecated: Use StatSummarizer(const StatSummarizerOptions&) instead. The
  // GraphDef is not needed by the StatSummarizer.
  explicit StatSummarizer(const tensorflow::GraphDef& tensorflow_graph);

  ~StatSummarizer();

  // Adds another run's StepStats output to the aggregate counts.
  void ProcessStepStats(const StepStats& step_stats);

  // Returns a string detailing the accumulated runtime stats in a tab-separated
  // format which can be pasted into a spreadsheet for further analysis.
  std::string GetOutputString() const {
    return stats_calculator_->GetOutputString();
  }

  std::string ShortSummary() const {
    return stats_calculator_->GetShortSummary();
  }

  // Prints the string returned by GetOutputString().
  void PrintStepStats() const;

  // Prints the output tensor sizes and types for each node.
  void PrintOutputs() const;

  void ComputeStatsByType(
      std::map<std::string, int64_t>* node_type_map_count,
      std::map<std::string, int64_t>* node_type_map_time,
      std::map<std::string, int64_t>* node_type_map_memory,
      std::map<std::string, int64_t>* node_type_map_times_called,
      int64_t* accumulated_us) const {
    stats_calculator_->ComputeStatsByType(
        node_type_map_count, node_type_map_time, node_type_map_memory,
        node_type_map_times_called, accumulated_us);
  }

  std::string GetStatsByNodeType() const {
    return stats_calculator_->GetStatsByNodeType();
  }

  std::string GetStatsByMetric(const string& title,
                               StatsCalculator::SortingMetric sorting_metric,
                               int num_stats) const {
    return stats_calculator_->GetStatsByMetric(title, sorting_metric,
                                               num_stats);
  }

  int num_runs() const { return stats_calculator_->num_runs(); }

  // Returns stats of total microseconds spent by all nodes in each run.
  const Stat<int64_t>& run_total_us() const {
    return stats_calculator_->run_total_us();
  }

 private:
  void Validate(const std::vector<TensorDescription>* outputs,
                const NodeExecStats& ns) const;

  std::map<std::string, std::vector<TensorDescription> > outputs_;

  std::unique_ptr<StatsCalculator> stats_calculator_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_STAT_SUMMARIZER_H_
