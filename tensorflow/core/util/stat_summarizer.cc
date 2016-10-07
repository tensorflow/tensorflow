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

#include "tensorflow/core/util/stat_summarizer.h"

#include <iomanip>
#include <map>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

StatSummarizer::StatSummarizer(const tensorflow::GraphDef& tensorflow_graph) {
  LOG(INFO) << "StatSummarizer found " << tensorflow_graph.node_size()
            << " nodes";
  for (const auto& node : tensorflow_graph.node()) {
    nodes_in_def_order_.push_back(node.name());
    node_types_[node.name()] = node.op();
  }
}

void StatSummarizer::ProcessStepStats(const StepStats& step_stats) {
  int64 curr_total = 0;
  int64 mem_total = 0;
  if (timing_details_.empty() && !step_stats.dev_stats().empty() &&
      !step_stats.dev_stats(0).node_stats().empty()) {
    first_node_start_micros_ =
        step_stats.dev_stats(0).node_stats(0).all_start_micros();
  }

  for (const auto& ds : step_stats.dev_stats()) {
    for (const auto& ns : ds.node_stats()) {
      // timing stats
      const int64 curr_time = ns.all_end_rel_micros();
      curr_total += curr_time;
      auto result = timing_details_.emplace(ns.node_name(), Detail());
      Detail* detail = &(result.first->second);
      if (result.second) {
        detail->first_rel_end = curr_time;
        detail->first_start = ns.all_start_micros() - first_node_start_micros_;
        detail->total = curr_time;
        detail->outputs.resize(ns.output_size());
        for (const auto& output : ns.output()) {
          const int32 slot = output.slot();
          if ((slot < 0) || (slot >= ns.output_size())) {
            LOG(ERROR) << "Bad output slot '" << slot << "' for '"
                       << ns.node_name() << "'";
            continue;
          }
          detail->outputs[slot] = output.tensor_description();
        }
      } else {
        detail->total += curr_time;
        if (detail->outputs.size() != ns.output_size()) {
          LOG(WARNING) << "Number of outputs changed between runs for '"
                       << ns.node_name() << "' - was " << detail->outputs.size()
                       << ", now " << ns.output_size();
        } else {
          for (const auto& output : ns.output()) {
            const int32 slot = output.slot();
            if ((slot < 0) || (slot >= ns.output_size())) {
              LOG(ERROR) << "Bad output slot '" << slot << "' for '"
                         << ns.node_name() << "'";
              continue;
            }
            const auto& stored = detail->outputs[slot];
            const auto& current = output.tensor_description();
            bool do_shapes_match = true;
            if (stored.shape().dim_size() != current.shape().dim_size()) {
              do_shapes_match = false;
            } else {
              for (int i = 0; i < stored.shape().dim_size(); ++i) {
                if (stored.shape().dim(i).size() !=
                    current.shape().dim(i).size()) {
                  do_shapes_match = false;
                }
              }

              if ((stored.dtype() != current.dtype()) || !do_shapes_match) {
                LOG(WARNING) << "Output tensor changed between runs for '"
                             << ns.node_name();
              }
            }
          }
        }
      }

      // memory stats
      auto mem_result = memory_details_.emplace(ns.node_name(), Detail());
      bool first = mem_result.second;
      auto& val = mem_result.first->second;
      for (const auto& mem : ns.memory()) {
        const int64 mem_usage = mem.total_bytes();
        mem_total += mem_usage;

        if (first) {
          first = false;
          val.first_start = mem_total - mem_usage;
          val.first_rel_end = mem_usage;
          val.total = 0;
        } else if (mem_result.second) {
          val.first_rel_end += mem_usage;
        }
        val.total += mem_usage;
      }
    }
  }
  run_total_micros_.UpdateStat(curr_total);
  memory_.UpdateStat(mem_total);
}

std::string StatSummarizer::ShortSummary() const {
  std::stringstream stream;
  stream << run_total_micros_.count() << " runs, avg " << std::setprecision(4)
         << run_total_micros_.avg() / 1000.0 << " ms, " << node_types_.size()
         << " nodes defined " << timing_details_.size() << " nodes observed"
         << std::endl;
  stream << std::setprecision(9) << memory_.avg() / 1000.0 << " avg KB per run."
         << std::endl;
  return stream.str();
}

std::string StatSummarizer::HeaderString() const {
  std::stringstream stream;
  stream << std::setw(9) << "[start]" << std::setw(9) << "[first]"
         << std::setw(9) << "[avg]";
  stream << "\t" << std::setw(8) << "[%]"
         << " ";
  stream << "\t" << std::setw(8) << "[cdf%]"
         << " ";
  stream << "\t" << std::setw(10) << "[Op]";
  stream << "\t"
         << "[Name]";
  return stream.str();
}

std::string StatSummarizer::ColumnString(const std::string& name,
                                         const Detail& detail,
                                         int64 cumulative_stat_on_node,
                                         Stat<int64> stat) const {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(3) << std::setw(9)
         << detail.first_start / 1000.0;
  stream << std::fixed << std::setprecision(3) << std::setw(9)
         << detail.first_rel_end / 1000.0;

  double avg_time_ms = detail.total / 1000.0 / num_runs();
  stream << std::fixed << std::setprecision(3) << std::setw(9) << avg_time_ms;

  double percentage = detail.total * 100.0 / stat.sum();
  stream << "\t" << std::fixed << std::setprecision(3) << std::setw(7)
         << percentage << "%";

  double cdf_percentage = cumulative_stat_on_node * 100.0 / stat.sum();
  stream << "\t" << std::fixed << std::setprecision(3) << std::setw(7)
         << cdf_percentage << "%";

  stream << "\t" << std::setw(10);
  auto op_it = node_types_.find(name);
  if (op_it != node_types_.end()) {
    stream << op_it->second;
  } else {
    stream << " ";
  }

  stream << "\t" << name;

  return stream.str();
}

std::string StatSummarizer::GetStatsBySorting(SortingMetric sorting_metric,
                                              double cdf_cutoff_ratio,
                                              int num_max_nodes_to_print,
                                              bool use_memory) const {
  std::map<std::string, Detail> details;
  Stat<int64> stat;
  if (use_memory) {
    details = memory_details_;
    stat = memory_;
  } else {
    details = timing_details_;
    stat = run_total_micros_;
  }
  num_max_nodes_to_print =
      std::min<int>(num_max_nodes_to_print, details.size());

  std::stringstream stream;
  stream << ShortSummary() << std::endl;
  string out_opt = "duration";
  string unit = "ms";
  if (use_memory) {
    out_opt = "memory usage";
    unit = "KB";
  }
  if (sorting_metric == SortingMetric::BY_TOTAL) {
    stream << "============ Top by " << out_opt << " in " << unit
           << " =================" << std::endl;
  } else {
    CHECK(sorting_metric == SortingMetric::BY_RUN_ORDER);
    stream << "============ By run order (" << unit
           << ") =================" << std::endl;
  }
  stream << HeaderString() << std::endl;

  std::priority_queue<
      std::pair<int64, const std::pair<const std::string, Detail>*>>
      statistics;
  for (const auto& entry : details) {
    statistics.emplace(sorting_metric == SortingMetric::BY_TOTAL
                           ? entry.second.total
                           : -entry.second.first_start,
                       &entry);
  }

  const int64 cutoff_point = stat.sum() * cdf_cutoff_ratio;
  int64 accumulated_us = 0;

  for (int i = 0; !statistics.empty() && i < num_max_nodes_to_print &&
                  accumulated_us <= cutoff_point;
       ++i) {
    accumulated_us += statistics.top().second->second.total;
    stream << ColumnString(statistics.top().second->first,
                           statistics.top().second->second, accumulated_us,
                           stat)
           << std::endl;
    statistics.pop();
  }

  return stream.str();
}

std::string StatSummarizer::GetStatsByTopDurations(
    double cdf_cutoff, int num_max_nodes_to_print) const {
  return GetTimingStatsByTopDurations(cdf_cutoff, num_max_nodes_to_print);
}

std::string StatSummarizer::GetStatsByRunOrder() const {
  return GetTimingStatsByRunOrder();
}

std::string StatSummarizer::GetStatsByOrderOfNodeDefinitions() const {
  return GetTimingStatsByOrderOfNodeDefinitions();
}

std::string StatSummarizer::GetTimingStatsByTopDurations(
    double cdf_cutoff, int num_max_nodes_to_print) const {
  return GetStatsBySorting(SortingMetric::BY_TOTAL, cdf_cutoff,
                           num_max_nodes_to_print, false);
}

std::string StatSummarizer::GetTimingStatsByRunOrder() const {
  return GetStatsBySorting(SortingMetric::BY_RUN_ORDER, 1.0,
                           std::numeric_limits<int>::max(), false);
}

std::string StatSummarizer::GetTimingStatsByOrderOfNodeDefinitions() const {
  return GetStatsByOrderOfNodeDefinitions(false);
}

std::string StatSummarizer::GetMemoryStatsByUsage(
    double cdf_cutoff, int num_max_nodes_to_print) const {
  return GetStatsBySorting(SortingMetric::BY_TOTAL, cdf_cutoff,
                           num_max_nodes_to_print, true);
}

std::string StatSummarizer::GetMemoryStatsByRunOrder() const {
  return GetStatsBySorting(SortingMetric::BY_RUN_ORDER, 1.0,
                           std::numeric_limits<int>::max(), true);
}

std::string StatSummarizer::GetMemoryStatsByOrderOfNodeDefinitions() const {
  return GetStatsByOrderOfNodeDefinitions(true);
}

std::string StatSummarizer::GetStatsByOrderOfNodeDefinitions(
    bool use_memory) const {
  string type;
  if (use_memory) {
    type = "Memory Usage";
  } else {
    type = "Timings";
  }
  std::stringstream stream;
  stream << ShortSummary() << std::endl;
  stream << "============ " << type
         << " by order of graph definition =================" << std::endl;
  stream << HeaderString() << std::endl;

  int64 accumulated_us = 0;
  auto details = use_memory ? memory_details_ : timing_details_;
  Stat<int64> stat = use_memory ? memory_ : run_total_micros_;

  for (const auto& node_name_op : nodes_in_def_order_) {
    auto detail_it = details.find(node_name_op);
    if (detail_it == details.end()) {
      continue;
    }
    accumulated_us += detail_it->second.total;
    stream << ColumnString(detail_it->first, detail_it->second, accumulated_us,
                           stat)
           << std::endl;
    details.erase(detail_it);
  }

  if (!details.empty()) {
    stream << "============ "
           << "The rest have different names between NodeExecStats and GraphDef"
           << "============ " << std::endl;

    for (const auto& entry : details) {
      // Prints the remaining nodes whose names are different from the name in
      // graph definition.
      accumulated_us += entry.second.total;
      stream << ColumnString(entry.first, entry.second, accumulated_us, stat)
             << std::endl;
    }
  }

  return stream.str();
}

void StatSummarizer::PrintStepStats() const {
  LOG(INFO) << "Total time (us): " << run_total_micros_;
  LOG(INFO) << GetTimingStatsByRunOrder();
  LOG(INFO) << GetTimingStatsByTopDurations();
  LOG(INFO) << "Total Memory (bytes): " << memory_;
  LOG(INFO) << GetMemoryStatsByRunOrder();
  LOG(INFO) << GetMemoryStatsByUsage();
  LOG(INFO);
}

void StatSummarizer::PrintOutputs() const {
  std::priority_queue<
      std::pair<int64, const std::pair<const std::string, Detail>*>>
      timings;
  for (const auto& entry : timing_details_) {
    timings.emplace(-entry.second.first_start, &entry);
  }

  LOG(INFO) << "============ Node output tensor sizes in run order ========";
  while (!timings.empty()) {
    auto entry = timings.top();
    timings.pop();
    const Detail& detail = entry.second->second;
    std::stringstream stream;
    stream << entry.second->first << "\t" << detail.outputs.size();
    for (const auto& tensor : detail.outputs) {
      stream << "\t" << DataTypeString(tensor.dtype());
      stream << "\t" << tensor.shape().dim_size();
      for (const auto& d : tensor.shape().dim()) {
        stream << "\t" << d.size();
      }
    }
    LOG(INFO) << stream.str();
  }
}

}  // namespace tensorflow
